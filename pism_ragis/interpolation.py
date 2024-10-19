# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pism-ragis.
#
# PISM-RAGIS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: skip-file
"""
Module for data processing
"""

import sys
from typing import Hashable, Iterable, Optional, Union

import numpy as np
import petsc4py
import xarray as xr
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve

petsc4py.init(sys.argv)
from petsc4py import PETSc


def assemble_matrix(mask):
    """Assemble the matrix corresponding to the standard 5-point stencil
    approximation of the Laplace operator on the domain defined by
    mask == True, where mask is a 2D NumPy array.

    Uses zero Neumann BC at grid edges.

    The grid spacing is ignored, which is equivalent to assuming equal
    spacing in x and y directions.
    """
    # grid size
    nrow, ncol = mask.shape
    # create sparse matrix
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([nrow * ncol, nrow * ncol])
    A.setType("aij")  # sparse
    A.setPreallocationNNZ(5)

    # precompute values for setting
    # diagonal and non-diagonal entries
    diagonal = 4.0
    offdx = -1.0
    offdy = -1.0

    def R(i, j):
        "Map from the (row,column) pair to the linear row number."
        return i * ncol + j

    # loop over owned block of rows on this
    # processor and insert entry values
    row_start, row_end = A.getOwnershipRange()
    for row in range(row_start, row_end):
        i = row // ncol  # map row number to
        j = row - i * ncol  # grid coordinates

        if mask[i, j] == False:
            A[row, row] = diagonal
            continue

        D = diagonal

        # i
        if i == 0:  # top row
            D += offdy

        if i > 0:  # interior
            col = R(i - 1, j)
            A[row, col] = offdx

        if i < nrow - 1:  # interior
            col = R(i + 1, j)
            A[row, col] = offdx

        if i == nrow - 1:  # bottom row
            D += offdy

        # j
        if j == 0:  # left-most column
            D += offdx

        if j > 0:  # interior
            col = R(i, j - 1)
            A[row, col] = offdy

        if j < ncol - 1:  # interior
            col = R(i, j + 1)
            A[row, col] = offdy

        if j == ncol - 1:  # right-most column
            D += offdx

        A[row, row] = D

    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()

    return A


def assemble_rhs(rhs, X):
    """Assemble the right-hand side of the system approximating the
    Laplace equation.

    Modifies rhs in place; sets Dirichlet BC using X where X.mask ==
    False.
    """
    import numpy as np

    nrow, ncol = X.shape
    row_start, row_end = rhs.getOwnershipRange()

    # The right-hand side is zero everywhere except for Dirichlet nodes.
    rhs.set(0.0)

    # Create a flat index array for the range of rows owned by this process
    rows = np.arange(row_start, row_end, dtype=np.int32)

    # Map row numbers to grid coordinates
    i = rows // ncol
    j = rows % ncol

    # Find the indices where the mask is False
    mask_indices = np.where(X.mask[i, j] == False)[0]

    # Set the rhs values for the Dirichlet nodes
    rhs_values = 4.0 * X[i[mask_indices], j[mask_indices]]
    rhs.setValues(rows[mask_indices].astype(np.int32), rhs_values)

    rhs.assemble()


def create_iterative_solver():
    "Create the KSP solver"
    # create linear solver
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)

    # Use algebraic multigrid:
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.GAMG)
    ksp.setFromOptions()

    ksp.setInitialGuessNonzero(True)

    return ksp


def create_direct_solver():
    "Create the KSP solver"
    # create linear solver
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)

    pc = ksp.getPC()
    pc = ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()

    ksp.setInitialGuessNonzero(False)

    return ksp


def _fill_missing_petsc(field, matrix=None, method: str = "iterative"):
    """
    Fill missing values in a NumPy array 'field' using the matrix
    'matrix' approximating the Laplace operator.
    """

    if method == "iterative":
        ksp = create_iterative_solver()
    else:
        ksp = create_direct_solver()

    if matrix is None:
        A = assemble_matrix(field.mask)
    else:
        # PETSc.Sys.Print("Reusing the matrix...")
        A = matrix

    # obtain solution & RHS vectors
    x, b = A.getVecs()

    assemble_rhs(b, field)

    initial_guess = np.mean(field)

    # set the initial guess
    x.set(initial_guess)

    ksp.setOperators(A)

    # Solve Ax = b
    # PETSc.Sys.Print("Solving...")
    ksp.solve(b, x)
    # PETSc.Sys.Print("done.")

    # transfer solution to processor 0
    vec0, scatter = create_scatter(x)
    scatter_to_0(x, vec0, scatter)

    return vec0, A


def fill_missing_petsc(data, method: str = "iterative"):
    """
    Fill missing values in a NumPy array 'field' using the matrix
    'matrix' approximating the Laplace operator.
    """

    try:
        # Look if petsc4py has already been initialized
        PETSc = petsc4py.__getattribute__('PETSc')
    except AttributeError:
        # If not, initialize petsc4py with the PETSc that PISM was compiled against.
        import sys
        try:
            petsc4py.init(sys.argv, arch=PISM.version_info.PETSC_ARCH)
        except TypeError:
            # petsc4py on Debian 9 does not recognize the PETSC_ARCH of PETSc in the .deb package
            petsc4py.init(sys.argv)
        from petsc4py import PETSc

    arr, A = _fill_missing_petsc(data, method=method)
    if PETSc.COMM_WORLD.getRank() == 0:
        data_filled = arr[:].reshape(data.shape)
        return data_filled


def create_scatter(vector):
    "Create the scatter to processor 0."
    comm = vector.getComm()
    rank = comm.getRank()
    scatter, V0 = PETSc.Scatter.toZero(vector)
    scatter.scatter(vector, V0, False, PETSc.Scatter.Mode.FORWARD)
    comm.barrier()

    return V0, scatter


def scatter_to_0(vector, vector_0, scatter):
    "Scatter a distributed 'vector' to 'vector_0' on processor 0 using 'scatter'."
    comm = vector.getComm()
    scatter.scatter(vector, vector_0, False, PETSc.Scatter.Mode.FORWARD)
    comm.barrier()


def scatter_from_0(vector_0, vector, scatter):
    "Scatter 'vector_0' on processor 0 to a distributed 'vector' using 'scatter'."
    comm = vector.getComm()
    scatter.scatter(vector, vector_0, False, PETSc.Scatter.Mode.REVERSE)
    comm.barrier()


def create_laplacian_matrix(
    interior_points: np.ndarray, mask: np.ndarray, n: int, m: int
) -> csc_matrix:
    """
    Create the Laplacian matrix for the given interior points and mask.

    Parameters
    ----------
    interior_points : np.ndarray
        Array of interior points where the mask is False.
    mask : np.ndarray
        Boolean mask indicating the missing values.
    n : int
        Number of rows in the data array.
    m : int
        Number of columns in the data array.

    Returns
    -------
    csc_matrix
        The Laplacian matrix in CSC format.
    """
    row_indices = []
    col_indices = []
    data_values = []

    for k, (i, j) in enumerate(interior_points):
        row_indices.append(k)
        col_indices.append(k)
        data_values.append(-4)

        if i > 0:
            if mask[i - 1, j]:
                neighbor_index = np.where((interior_points == [i - 1, j]).all(axis=1))[
                    0
                ][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if i < n - 1:
            if mask[i + 1, j]:
                neighbor_index = np.where((interior_points == [i + 1, j]).all(axis=1))[
                    0
                ][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if j > 0:
            if mask[i, j - 1]:
                neighbor_index = np.where((interior_points == [i, j - 1]).all(axis=1))[
                    0
                ][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if j < m - 1:
            if mask[i, j + 1]:
                neighbor_index = np.where((interior_points == [i, j + 1]).all(axis=1))[
                    0
                ][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)

    # Create the sparse matrix using COO format
    L = coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(interior_points), len(interior_points)),
    ).tocsc()
    return L


def create_rhs_vector(
    data: np.ndarray, interior_points: np.ndarray, mask: np.ndarray, n: int, m: int
) -> np.ndarray:
    """
    Create the right-hand side vector for the linear system.

    Parameters
    ----------
    data : np.ndarray
        The data array with missing values.
    interior_points : np.ndarray
        Array of interior points where the mask is False.
    mask : np.ndarray
        Boolean mask indicating the missing values.
    n : int
        Number of rows in the data array.
    m : int
        Number of columns in the data array.

    Returns
    -------
    np.ndarray
        The right-hand side vector.
    """
    b = np.zeros(len(interior_points))

    for k, (i, j) in enumerate(interior_points):
        if i > 0 and ~mask[i - 1, j]:
            b[k] -= data[i - 1, j]
        if i < n - 1 and ~mask[i + 1, j]:
            b[k] -= data[i + 1, j]
        if j > 0 and ~mask[i, j - 1]:
            b[k] -= data[i, j - 1]
        if j < m - 1 and ~mask[i, j + 1]:
            b[k] -= data[i, j + 1]

    return b


def laplace(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values in the data array using the Laplacian method.

    Parameters
    ----------
    data : np.ndarray
        The data array with missing values.
    mask : np.ndarray
        Boolean mask indicating the missing values.

    Returns
    -------
    np.ndarray
        The data array with missing values filled.
    """

    data = data.copy()
    n, m = data.shape
    interior_points = np.argwhere(mask)

    # Create the Laplacian matrix
    L = create_laplacian_matrix(interior_points, mask, n, m)

    # Create the right-hand side vector
    b = create_rhs_vector(data, interior_points, mask, n, m)

    # Solve the linear system
    u = spsolve(L, b)
    # Fill in the missing values
    for k, (i, j) in enumerate(interior_points):
        data[i, j] = u[k]

    return data


@xr.register_dataarray_accessor("utils")
class InterpolationMethods:
    """
    Interpolationes methods for xarray DataArray.

    This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'interpolation' attribute.

    Parameters
    ----------

    xarray_obj : xr.DataArray
      The xarray DataArray to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the InterpolationMethods class.

        Parameters
        ----------

        xarray_obj : xr.DataArray
            The xarray DataArray to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def __repr__(self):
        """
        Interpolation methods.
        """
        return """
Interpolationes methods for xarray DataArray.

This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'interpolation' attribute.

Parameters
----------

xarray_obj : xr.DataArray
  The xarray DataArray to which to add the custom methods.
      """

    def fillna(
        self,
        dim: Optional[Union[str, Iterable[Hashable]]] = ["y", "x"],
        method: str = "laplace",
    ):
        """
        Fill missing values using Laplacian.
        """
        da = self._obj.load()
        data = da.to_numpy()
        mask = da.isnull()
        self._obj = xr.apply_ufunc(
            self._fillna,
            data,
            mask,
            input_core_dims=[dim, dim],
            output_core_dims=[dim],
            vectorize=True,
            dask="forbidden",
        )
        return self._obj

    def _fillna(self, data, mask):
        """
        Fill missing values.
        """

        result = laplace(data, mask)

        return result
