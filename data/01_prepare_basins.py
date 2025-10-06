# Copyright (C) 2024 Andy Aschwanden
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

# pylint: disable=consider-using-with,broad-exception-caught,unnecessary-lambda

"""
Prepare Greenland basins.
"""


from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union  # shapely>=1.8 or 2.x

from pism_ragis.download import download_archive

imbie = {
    "name": "GRE_Basins_IMBIE2_v1.3",
    "url": "http://imbie.org/wp-content/uploads/2016/09/",
    "suffix": "zip",
    "path": "imbie",
}
mouginot = {
    "name": "Greenland_Basins_PS_v1.4.2",
    "url": "http://www.cpom.ucl.ac.uk/imbie/basins/mouginot_2019/",
    "suffix": "tar.gz",
    "path": "mouginot",
}
crs = "EPSG:3413"


def union_with_intersecting(
    gdf_base: gpd.GeoDataFrame,
    gdf_poly: gpd.GeoDataFrame,
    fix_invalid: bool = False,
) -> gpd.GeoDataFrame:
    """
    Union base geometries with any intersecting polygons from another layer.

    For each row in ``gdf_base``, find all geometries in ``gdf_poly`` that
    spatially **intersect** it, compute the union of those intersecting
    geometries, and then union that result with the base row's geometry.
    If a base row has no intersecting geometries in ``gdf_poly``, its geometry
    is left unchanged.

    Parameters
    ----------
    gdf_base : geopandas.GeoDataFrame
        Base features whose geometries will be updated by union. Must have a
        valid CRS (``gdf_base.crs``).
    gdf_poly : geopandas.GeoDataFrame
        Candidate geometries to union with, matched by spatial intersection.
        If multiple geometries intersect a given base row, they are first
        dissolved via :func:`shapely.ops.unary_union`.
    fix_invalid : bool, optional
        If True, apply a small topological cleanup via ``buffer(0)`` to both
        layers before processing. This can help with self-intersections or
        slight ring defects. Default is False.

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of ``gdf_base`` where ``geometry`` is replaced by the union with
        any intersecting features from ``gdf_poly`` on a per-row basis.

    Notes
    -----
    - CRS handling: ``gdf_poly`` is reprojected to ``gdf_base.crs`` if needed.
    - Vectorized union is used when available (Shapely ≥ 2.0). A row-wise
      fallback is provided.
    - If you have very large inputs, consider spatially indexing or tiling
      beforehand for performance.

    Examples
    --------
    >>> out = union_with_intersecting(glaciers, shelves)
    >>> out.crs == glaciers.crs
    True
    """
    base = gdf_base.copy()

    # CRS alignment
    if base.crs != gdf_poly.crs:
        gdf_poly = gdf_poly.to_crs(base.crs)

    if fix_invalid:
        base = base.copy()
        base["geometry"] = base.geometry.buffer(0)
        gdf_poly = gdf_poly.copy()
        gdf_poly["geometry"] = gdf_poly.geometry.buffer(0)

    # Keep a copy of the right geometry in a non-active column so it survives sjoin
    polyX = gdf_poly.copy()
    polyX["geometry_right"] = polyX.geometry

    # Spatial join: left index is base index; also bring along geometry_right
    hits = gpd.sjoin(
        base[["geometry"]],
        polyX[["geometry", "geometry_right"]],
        predicate="intersects",
        how="left",
    )

    # Group intersecting right-hand geometries by base row (drop rows without matches)
    hits_nonnull = hits.dropna(subset=["index_right"])
    if hits_nonnull.empty:
        return base  # nothing to union

    grouped = hits_nonnull.groupby(level=0)["geometry_right"]
    inter_union = grouped.apply(lambda geoms: unary_union(list(geoms)))
    inter_union = gpd.GeoSeries(inter_union, crs=gdf_poly.crs)

    # Union only where we have something to union with
    mask = base.index.isin(inter_union.index)
    try:
        # Vectorized union (Shapely 2.x)
        base.loc[mask, "geometry"] = base.loc[mask, "geometry"].union(inter_union)
    except Exception:
        # Row-wise fallback
        base.loc[mask, "geometry"] = base.loc[mask].apply(lambda r: r.geometry.union(inter_union.loc[r.name]), axis=1)

    return base


def union_geometry_by_subregion(
    base: gpd.GeoDataFrame,
    repl: gpd.GeoDataFrame,
    key: str = "SUBREGION1",
) -> gpd.GeoDataFrame:
    """
    Union base geometries with matching subregion geometries from another layer.

    For each row in ``base``, look up a geometry in ``repl`` with the same
    categorical key (``key``). If present, union the two geometries and write
    the result back to that row; otherwise leave the row unchanged. When
    ``repl`` contains multiple rows for the same key, they are dissolved
    (unioned) first to create a single replacement geometry per key.

    Parameters
    ----------
    base : geopandas.GeoDataFrame
        Input features whose ``geometry`` column will be updated by union.
        Must have a valid CRS (``base.crs``).
    repl : geopandas.GeoDataFrame
        Replacement geometries keyed by the column ``key``. If multiple rows
        share the same key, they are dissolved prior to the union step.
        CRS is reprojected to match ``base.crs`` if needed.
    key : str, optional
        Name of the column used to match rows between ``base`` and ``repl``.
        Default is ``"SUBREGION1"``.

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of ``base`` in which rows with a matching key have
        ``geometry = geometry ∪ repl_geometry``. Rows without a match are
        unchanged.

    Notes
    -----
    - Uses vectorized union when available (Shapely ≥ 2.0); otherwise falls
      back to a row-wise operation.
    - If topology errors occur (e.g., invalid rings), consider pre-cleaning
      with ``gdf.geometry = gdf.buffer(0)`` or ``shapely.make_valid``.

    Examples
    --------
    >>> out = union_geometry_by_subregion(glaciers, shelves, key="SUBREGION1")
    >>> out.crs == glaciers.crs
    True
    """
    out = base.copy()

    # Align CRS
    if out.crs != repl.crs:
        repl = repl.to_crs(out.crs)

    # One replacement geometry per key (union duplicates)
    repl_one = repl[[key, "geometry"]].dissolve(by=key, as_index=False)

    # Map: key -> geometry
    geom_map = repl_one.set_index(key)["geometry"]

    # Rows that have a replacement
    mask = out[key].isin(geom_map.index)

    # Aligned series of replacement geoms
    repl_aligned = out.loc[mask, key].map(geom_map)

    # Pairwise union (vectorized when available; fallback to row-wise)
    try:
        out.loc[mask, "geometry"] = out.loc[mask, "geometry"].union(repl_aligned)
    except Exception:
        out.loc[mask, "geometry"] = out.loc[mask].apply(lambda r: r.geometry.union(geom_map[r[key]]), axis=1)

    return out


def prepare_basin(basin_dict: dict, col: str = "SUBREGION1"):
    """
    Prepare basin.

    Parameters
    ----------
    basin_dict : Dict
        Dictionary containing basin information such as name, url, suffix, and path.
    col : str, optional
        Column name to use, default is "SUBREGION1".
    """
    name = basin_dict["name"]
    print(f"Preparing {name}")
    url = basin_dict["url"] + basin_dict["name"] + "." + basin_dict["suffix"]

    archive = download_archive(url)

    path = Path(basin_dict["path"])
    path.mkdir(parents=True, exist_ok=True)
    archive.extractall(path=path)
    p = path / f"{name}.shp"

    basin_gp = gpd.read_file(p).to_crs(crs)
    basin_gp["geometry"] = basin_gp["geometry"].apply(lambda x: x.exterior)
    rings = basin_gp[basin_gp.geometry.geom_type == "LinearRing"]
    basin_gp.loc[rings.index, "geometry"] = rings.geometry.apply(lambda r: Polygon(r))
    g = basin_gp.geometry.make_valid()
    basin_gp.geometry = g
    shelves = gpd.read_file("basins/GRE_Basins_shelf_extensions.gpkg").to_crs(crs)
    gdf = union_geometry_by_subregion(basin_gp, shelves)

    m_no_periphery = gdf[gdf[col] != "ICE_CAP"]
    gis = gpd.GeoDataFrame(m_no_periphery.dissolve())

    gis["SUBREGION1"] = "GIS"
    m = pd.concat([gdf, gis])
    m.to_file(path / f"{name}_w_shelves.gpkg")
    print("Done.\n")


if __name__ == "__main__":

    print("Preparing datasets")
    for basin in [mouginot, imbie]:
        prepare_basin(basin)

    print("Preparing glaciers")
    p = "basins/Greenland_Basins_PS_v1.4.2.shp"
    s = "basins/GRE_Basins_shelf_extensions_merged.gpkg"
    basin_gp = gpd.read_file(p).to_crs(crs)
    shelves = gpd.read_file(s).to_crs(crs)
    gdf_out = union_with_intersecting(basin_gp, shelves)
    gdf_out.to_file("basins/Greenland_Basins_PS_v1.4.2_ext.gpkg")
