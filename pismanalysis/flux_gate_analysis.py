#!/usr/bin/env python
# Copyright (C) 2014-2019 Andy Aschwanden

from osgeo import ogr
from osgeo import osr
import os
from unidecode import unidecode
import itertools
import codecs
import operator
import numpy as np
import pylab as plt
import matplotlib as mpl
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.cm as cmx
import matplotlib.colors as mplcolors
from argparse import ArgumentParser
import pandas as pa
from palettable import colorbrewer
import statsmodels.api as sm
from netCDF4 import Dataset as NC
import re

import cf_units


def set_mode(mode, aspect_ratio=0.95):
    """
    Set the print mode, i.e. document and font size. Options are:
    - onecol: width=80mm, font size=8pt. Appropriate for 1-column figures
    - twocol: width=160mm, font size=8pt. Default.
              Appropriate for 2-column figures
    - medium: width=121mm, font size=7pt.
    - small_font: width=121mm, font size=6pt.
    - height: height=2.5in.
    - small: width=80mm, font size=6pt
    - presentation: width=85mm, font size=10pt. For presentations.
    """

    linestyle = "-"

    def set_onecol():
        """
        Define parameters for "publish" mode and return value for pad_inches
        """

        fontsize = 6
        lw = 0.5
        markersize = 2
        fig_width = 3.15  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.30

    def set_small():
        """
        Define parameters for "publish" mode and return value for pad_inches
        """

        fontsize = 6
        lw = 0.5
        markersize = 2
        fig_width = 3.15  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "lines.markeredgewidth": 0.2,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.20

    def set_72mm():
        """
        Define parameters for "72mm" mode and return value for pad_inches
        """

        fontsize = 6
        markersize = 3
        lw = 0.7
        fig_width = 2.8  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.35,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.20

    def set_50mm():
        """
        Define parameters for "72mm" mode and return value for pad_inches
        """

        fontsize = 5
        markersize = 2.5
        lw = 0.6
        fig_width = 2.0  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.3,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.10

    def set_medium():
        """
        Define parameters for "medium" mode and return value for pad_inches
        """

        fontsize = 8
        markersize = 3
        lw = 0.75
        fig_width = 3.15  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.10

    def set_small_font():
        """
        Define parameters for "small_font" mode and return value for pad_inches
        """

        fontsize = 6
        markersize = 2
        lw = 0.6
        fig_width = 3.15  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.10

    def set_large_font():
        """
        Define parameters for "large_font" mode and return value for pad_inches
        """

        fontsize = 10
        markersize = 9
        lw = 0.75
        fig_width = 6.2  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.20

    def set_presentation():
        """
        Define parameters for "presentation" mode and return value
        for pad_inches
        """

        fontsize = 8
        lw = 1.5
        markersize = 3
        fig_width = 6.64  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.75,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "legend.fontsize": fontsize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.2

    def set_twocol():
        """
        Define parameters for "twocol" mode and return value for pad_inches
        """

        fontsize = 7
        lw = 0.75
        markersize = 3
        fig_width = 6.3  # inch
        fig_height = aspect_ratio * fig_width  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.5,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "legend.fontsize": fontsize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.35

    def set_height():
        """
        Define parameters for "twocol" mode and return value for pad_inches
        """
        fontsize = 8
        lw = 1.1
        markersize = 1.5
        fig_height = 2.5  # inch
        fig_width = fig_height / aspect_ratio  # inch
        fig_size = [fig_width, fig_height]

        params = {
            "backend": "ps",
            "axes.linewidth": 0.65,
            "lines.linewidth": lw,
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "lines.linestyle": linestyle,
            "lines.markersize": markersize,
            "legend.fontsize": fontsize,
            "font.size": fontsize,
            "figure.figsize": fig_size,
        }

        plt.rcParams.update(params)

        return lw, 0.025

    if mode == "onecol":
        return set_onecol()
    elif mode == "small":
        return set_small()
    elif mode == "medium":
        return set_medium()
    elif mode == "72mm":
        return set_72mm()
    elif mode == "50mm":
        return set_50mm()
    elif mode == "small_font":
        return set_small_font()
    elif mode == "large_font":
        return set_large_font()
    elif mode == "presentation":
        return set_presentation()
    elif mode == "twocol":
        return set_twocol()
    elif mode == "height":
        return set_height()
    else:
        print(("%s mode not recognized, using onecol instead" % mode))
        return set_twocol()


def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """

    return zip(reversed(range(len(iterable))), reversed(iterable))


def get_rmsd(a, b, w=None):
    """
    Returns the (weighted) root mean square differences between a and b.

    Parameters
    ----------
    a, b : array_like
    w : weights

    Returns
    -------
    rmsd : scalar
    """

    if w is None:
        w = np.ones_like(a)
    # If observations or errors have missing values, don't count them
    c = (a.ravel() - b.ravel()) / w.ravel()
    if isinstance(c, np.ma.MaskedArray):
        # count all non-masked elements
        N = c.count()
        # reduce to non-masked elements
        return np.sqrt(np.linalg.norm(np.ma.compressed(c), 2) ** 2.0 / N), N
    else:
        N = c.shape[0]
        return np.sqrt(np.linalg.norm(c, 2) ** 2.0 / N), N


class FluxGate(object):

    """
    A class for FluxGates.

    Parameters
    ----------
    pos_id: int, pos of gate in array
    gate_name: string, name of flux gate
    gate_id: int, gate identification
    profile_axis: 1-d array, profile axis values
    profile_axis_units: string, udunits unit of axis
    profile_axis_name: string, descriptive name of axis

    """

    def __init__(
        self,
        pos_id,
        gate_name,
        gate_id,
        profile_axis,
        profile_axis_units,
        profile_axis_name,
        clon,
        clat,
        flightline,
        glaciertype,
        flowtype,
        *args,
        **kwargs,
    ):
        super(FluxGate, self).__init__(*args, **kwargs)
        self.pos_id = pos_id
        self.gate_name = gate_name
        self.gate_id = gate_id
        self.profile_axis = profile_axis
        self.profile_axis_units = profile_axis_units
        self.profile_axis_name = profile_axis_name
        self.clon = clon
        self.clat = clat
        self.flightline = flightline
        self.glaciertype = glaciertype
        self.flowtype = flowtype
        self.best_rmsd_exp_id = None
        self.best_rmsd = None
        self.best_corr_exp_id = None
        self.best_corr = None
        self.corr = None
        self.corr_units = None
        self.experiments = []
        self.exp_counter = 0
        self.has_observations = None
        self.has_fluxes = None
        self.has_stats = None
        self.linear_trend = None
        self.linear_bias = None
        self.linear_r2 = None
        self.linear_p = None
        self.N_corr = None
        self.N_rmsd = None
        self.observed_flux = None
        self.observed_flux_units = None
        self.observed_flux_error = None
        self.observed_mean = None
        self.observed_mean_units = None
        self.p_ols = None
        self.r2 = None
        self.r2_units = None
        self.rmsd = None
        self.rmsd_units = None
        self.S = None
        self.S_units = None
        self.sigma_obs = None
        self.sigma_obs_N = None
        self.sigma_obs_units = None
        self.varname = None
        self.varname_units = None

    def __repr__(self):
        return "FluxGate"

    def add_experiment(self, data):
        """
        Add an experiment to FluxGate

        """

        print(("      adding experiment to flux gate {0}".format(self.gate_name)))
        pos_id = self.pos_id
        fg_exp = FluxGateExperiment(data, pos_id)
        self.experiments.append(fg_exp)
        if self.varname is None:
            self.varname = data.varname
        if self.varname_units is None:
            self.varname_units = data.varname_units
        self.exp_counter += 1

    def add_observations(self, data):
        """
        Add observations to FluxGate

        """

        print(("      adding observations to flux gate {0}".format(self.gate_name)))
        pos_id = self.pos_id
        fg_obs = FluxGateObservations(data, pos_id)
        self.observations = fg_obs
        if self.has_observations is not None:
            print(
                (
                    "Flux gate {0} already has observations, overriding".format(
                        self.gate_name.encode("utf-8")
                    )
                )
            )
        self.has_observations = True

    def calculate_fluxes(self):
        """
        Calculate fluxes

        """

        if self.has_observations:
            self._calculate_observed_flux()
        self._calculate_experiment_fluxes()
        self.has_fluxes = True

    def calculate_stats(self):
        """
        Calculate statistics
        """

        if not self.has_fluxes:
            self.calculate_fluxes()
        corr = {}
        N_corr = {}
        N_rmsd = {}
        p_ols = {}
        r2 = {}
        rmsd = {}
        S = {}
        for exp in self.experiments:
            id = exp.id
            x = np.squeeze(self.profile_axis)
            obs_vals = np.squeeze(self.observations.values)
            sigma_obs = self.sigma_obs
            # mask values where obs is zero
            obs_vals = np.ma.masked_where(obs_vals == 0, obs_vals)
            if isinstance(obs_vals, np.ma.MaskedArray):
                obs_vals = obs_vals.filled(0)
            exp_vals = np.squeeze(self.experiments[id].values)
            # Calculate root mean square difference (RMSD), convert units
            my_rmsd, my_N_rmsd = get_rmsd(exp_vals, obs_vals)
            i_units = self.varname_units
            o_units = v_o_units
            i_units_cf = cf_units.Unit(i_units)
            o_units_cf = cf_units.Unit(o_units)
            rmsd[id] = i_units_cf.convert(my_rmsd, o_units_cf)
            N_rmsd[id] = my_N_rmsd
            obsS = pa.Series(data=obs_vals, index=x)
            expS = pa.Series(data=exp_vals, index=x)
            p_ols[id] = sm.OLS(expS, sm.add_constant(obsS), missing="drop").fit()
            r2[id] = p_ols[id].rsquared
            corr[id] = obsS.corr(expS)
        best_rmsd_exp_id = sorted(p_ols, key=lambda x: rmsd[x], reverse=False)[0]
        best_corr_exp_id = sorted(p_ols, key=lambda x: corr[x], reverse=True)[0]
        self.p_ols = p_ols
        self.best_rmsd_exp_id = best_rmsd_exp_id
        self.best_rmsd = rmsd[best_rmsd_exp_id]
        self.best_corr_exp_id = best_corr_exp_id
        self.best_corr = corr[best_corr_exp_id]
        self.rmsd = rmsd
        self.rmsd_units = v_o_units
        self.N_rmsd = N_rmsd
        self.S = S
        self.S_units = "1"
        self.r2 = r2
        self.r2_units = "1"
        self.corr = corr
        self.corr_units = "1"
        self.has_stats = True
        self.observed_mean = np.mean(self.observations.values)
        self.observed_mean_units = self.varname_units

    def _calculate_observed_flux(self):
        """
        Calculate observed flux

        Calculate observed flux using trapezoidal rule. If observations have
        asscociated errors, the error in observed flux is calculated as well.
        """

        x = self.profile_axis
        y = self.observations.values
        x_units = self.profile_axis_units
        y_units = self.varname_units
        int_val = self._line_integral(y, x)
        # Here we need to directly access udunits2 since we want to
        # multiply units
        if vol_to_mass:
            i_units_cf = (
                cf_units.Unit(x_units)
                * cf_units.Unit(y_units)
                * cf_units.Unit(ice_density_units)
            )
        else:
            i_units_cf = cf_units.Unit(x_units) * cf_units.Unit(y_units)
        o_units_cf = cf_units.Unit(v_flux_o_units)
        o_units_str = v_flux_o_units_str
        o_val = i_units_cf.convert(int_val, o_units_cf)
        observed_flux = o_val
        observed_flux_units = o_units_str
        if self.observations.has_error:
            y = self.observations.error
            int_val = self._line_integral(y, x)
            i_error = int_val
            o_error = i_units_cf.convert(i_error, o_units_cf)
            error_norm, N = get_rmsd(y, np.zeros_like(y, dtype="float32"))
            self.sigma_obs = error_norm
            self.sigma_obs_N = N
            self.sigma_obs_units = self.varname_units
            self.observed_flux_error = o_error

        self.observed_flux = observed_flux
        self.observed_flux_units = observed_flux_units

    def _calculate_experiment_fluxes(self):
        """
        Calculate experiment fluxes

        Calculated experiment fluxes using trapeziodal rule.

        """

        experiment_fluxes = {}
        experiment_fluxes_units = {}
        for exp in self.experiments:
            id = exp.id
            x = self.profile_axis
            y = exp.values
            x_units = self.profile_axis_units
            y_units = self.varname_units
            int_val = self._line_integral(y, x)
            # Here we need to directly access udunits2 since we want to
            # multiply units
            if vol_to_mass:
                i_units = (
                    cf_units.Unit(x_units)
                    * cf_units.Unit(y_units)
                    * cf_units.Unit(ice_density_units)
                )
            else:
                i_units = cf_units.Unit(x_units) * cf_units.Unit(y_units)
            o_units = cf_units.Unit(v_flux_o_units)
            o_units_str = v_flux_o_units_str
            o_val = i_units.convert(int_val, o_units)
            experiment_fluxes[id] = o_val
            experiment_fluxes_units[id] = o_units_str
            config = exp.config
            print(params_dict["grid_dx_meters"]["format"], label_params)
            if label_params:
                my_exp_str = ", ".join(
                    [
                        "=".join(
                            [
                                params_dict[key]["abbr"],
                                params_dict[key]["format"],
                            ]
                        )
                        for key in label_params
                    ]
                )
        self.experiment_fluxes = experiment_fluxes
        self.experiment_fluxes_units = experiment_fluxes_units

    def length(self):
        """
        Return length of the profile, rounded to the nearest meter.
        """

        return np.around(self.profile_axis.max())

    def return_gate_flux_str(self, islast):
        """
        Returns a LaTeX-table compatible string
        """

        if not self.has_stats:
            self.calculate_stats()
        gate_name = self.gate_name
        p_ols = self.p_ols
        observed_flux = self.observed_flux
        observed_flux_units = self.observed_flux_units
        experiment_fluxes = self.experiment_fluxes
        experiment_fluxes_units = self.experiment_fluxes_units
        observed_flux_error = self.observed_flux_error
        if observed_flux_error:
            obs_str = "$\pm$".join(
                ["{:2.1f}".format(observed_flux), "{:2.1f}".format(observed_flux_error)]
            )
        else:
            obs_str = "{:2.1f}".format(observed_flux)
        exp_str = " & ".join(
            [
                "".join(
                    [
                        "{:2.1f}".format(experiment_fluxes[key]),
                        "({:1.2f})".format(p_ols[key].rsquared),
                    ]
                )
                for key in experiment_fluxes
            ]
        )
        if islast:
            gate_str = " & ".join([gate_name, obs_str, "", exp_str])
        else:
            gate_str = " & ".join([gate_name, obs_str, "", " ".join([exp_str, r" \\"])])
        return gate_str

    def return_gate_flux_str_short(self):
        """
        Returns a LaTeX-table compatible string
        """

        if not self.has_stats:
            self.calculate_stats()
        gate_name = self.gate_name
        p_ols = self.p_ols
        observed_flux = self.observed_flux
        observed_flux_units = self.observed_flux_units
        experiment_fluxes = self.experiment_fluxes
        experiment_fluxes_units = self.experiment_fluxes_units
        observed_flux_error = self.observed_flux_error
        if observed_flux_error:
            obs_str = "$\pm$".join(
                ["{:2.1f}".format(observed_flux), "{:2.1f}".format(observed_flux_error)]
            )
        else:
            obs_str = "{:2.1f}".format(observed_flux)
        gate_str = " & ".join([gate_name, obs_str])
        return gate_str

    def _line_integral(self, y, x):
        """
        Return line integral using the composite trapezoidal rule

        Parameters
        ----------
        y: 1-d array_like
           Input array to integrate
        x: 1-d array_like
           Spacing between elements

        Returns
        -------
        trapz : float
                Definite integral as approximated by trapezoidal rule.
        """

        # Due to the variable length of profiles, we have masked arrays, with
        # masked values at the profile end. We can assume zero error here,
        # since it's not used for the computation

        if isinstance(y, np.ma.MaskedArray):
            y = y.filled(0)

        return float(np.squeeze(np.trapz(y, x)))

    def make_line_plot(self, **kwargs):
        """
        Make a plot.

        Make a line plot along a flux gate.
        """

        gate_name = self.gate_name
        experiments = self.experiments
        profile_axis = self.profile_axis
        profile_axis_name = self.profile_axis_name
        profile_axis_units = self.profile_axis_units
        i_units_cf = cf_units.Unit(profile_axis_units)
        o_units_cf = cf_units.Unit(profile_axis_out_units)
        profile_axis_out = i_units_cf.convert(profile_axis, o_units_cf)
        varname = self.varname
        v_units = self.varname_units
        has_observations = self.has_observations
        if not self.has_fluxes:
            self.calculate_fluxes()
        if has_observations:
            self.calculate_stats()

        labels = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if has_observations:
            obs = self.observations
            if legend == "long":
                if obs.has_error:
                    label = "obs: {:6.1f}$\pm${:4.1f}".format(
                        self.observed_flux, self.observed_flux_error
                    )
                else:
                    label = "obs: {:6.1f}".format(self.observed_flux)
            elif legend in ("short", "regress", "exp"):
                label = "observed"
            elif legend in ("attr"):
                label = "AERODEM"
            else:
                if obs.has_error:
                    label = "{:6.1f}$\pm${:4.1f}".format(
                        self.observed_flux, self.observed_flux_error
                    )
                else:
                    label = "{:6.1f}".format(self.observed_flux)
            has_error = obs.has_error
            i_vals = obs.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(v_o_units)
            obs_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            obs_max = np.max(obs_o_vals)
            if has_error:
                i_vals = obs.error
                obs_error_o_vals = i_units_cf.convert(i_vals, o_units_cf)
                ax.fill_between(
                    profile_axis_out,
                    obs_o_vals - obs_error_o_vals,
                    obs_o_vals + obs_error_o_vals,
                    color="0.85",
                )

            if simple_plot:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.35", label=label)
            else:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
                ax.plot(
                    profile_axis_out,
                    obs_o_vals,
                    dash_style,
                    color=obscolor,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )

        ne = len(experiments)
        lines_d = []
        lines_c = []
        # We need to carefully reverse list and properly order
        # handles and labels to have first experiment plotted on top
        for k, exp in enumerate(reversed(experiments)):
            i_vals = exp.values
            i_units_cf = cf_units.Unit(v_units)
            o_units_cf = cf_units.Unit(v_o_units)
            exp_o_vals = i_units_cf.convert(i_vals, o_units_cf)
            if normalize:
                exp_max = np.max(exp_o_vals)
                exp_o_vals *= obs_max / exp_max
            if "label_param_list" in list(kwargs.keys()):
                params = kwargs["label_param_list"]
                config = exp.config
                id = exp.id
                exp_str = ", ".join(
                    [
                        "=".join(
                            [
                                params_dict[key]["abbr"],
                                params_dict[key]["format"],
                            ]
                        )
                        for key in params
                    ]
                )
                if has_observations:
                    if legend == "long":
                        label = ", ".join(
                            [
                                ": ".join(
                                    [
                                        exp_str,
                                        "{:6.1f}".format(self.experiment_fluxes[id]),
                                    ]
                                ),
                                "=".join(["r", "{:1.2f}".format(self.corr[id])]),
                            ]
                        )
                        # label = ', '.join(['{:6.1f}'.format(self.experiment_fluxes[id]), '='.join(['r', '{:1.2f}'.format(self.corr[id])])])
                    elif legend == "attr":
                        [key for key in params]
                    elif legend == "exp":
                        # # Use this
                        # exp_str = ", ".join(
                        #     [
                        #         "=".join(
                        #             [
                        #                 params_dict[key]["abbr"],
                        #                 params_dict[key]["format"].format(config.get(key) * ice_density),
                        #             ]
                        #         )
                        #         for key in params
                        #     ]
                        # )
                        exp_str = ", ".join(
                            [
                                "=".join(
                                    [
                                        params_dict[key]["abbr"],
                                        params_dict[key]["format"].format(
                                            config.get(key)
                                        ),
                                    ]
                                )
                                for key in params
                            ]
                        )
                        label = exp_str
                    else:
                        label = "r={:1.2f}".format(self.corr[id])
                    if legend == "regress":
                        label = "{:4.0f}m, r={:1.2f}".format(
                            config["grid_dx_meters"], self.corr[id]
                        )
                    elif legend == "short":
                        label = "r={:1.2f}".format(self.corr[id])
                    else:
                        pass
                else:
                    label = config.get("grid_dx_meters")
                    # label = ", ".join([": ".join([exp_str, "{:6.1f}".format(self.experiment_fluxes[id])])])
                labels.append(label)
            my_color = my_colors[k]
            my_color_light = my_colors_light[k]
            if simple_plot:
                (line_c,) = ax.plot(
                    profile_axis_out, exp_o_vals, color=my_color, label=label
                )
                (line_d,) = ax.plot(profile_axis_out, exp_o_vals, color=my_color)
            else:
                (line_c,) = ax.plot(
                    profile_axis_out, exp_o_vals, "-", color=my_color, alpha=0.5
                )
                (line_d,) = ax.plot(
                    profile_axis_out,
                    exp_o_vals,
                    dash_style,
                    color=my_color,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    label=label,
                )
            labels.append(label)
            lines_d.append(line_d)
            lines_c.append(line_c)

        # ax.text(0.05, 0.85, 'r={:1.2f}'.format(self.corr[id]), transform=ax.transAxes)
        ax.set_xlim(0, np.max(profile_axis_out))
        xlabel = "{0} ({1})".format(profile_axis_name, profile_axis_out_units)
        ax.set_xlabel(xlabel)
        if varname in list(var_name_dict.keys()):
            v_name = var_name_dict[varname]
        else:
            v_name = varname
        ylabel = "{0} ({1})".format(v_name, v_o_units_str)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = handles[:0:-1]
        ordered_labels = labels[:0:-1]
        ordered_handles.insert(0, handles[0])
        ordered_labels.insert(0, labels[0])
        if legend != "none":
            if (legend == "short") or (legend == "regress") or (legend == "attr"):
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            else:
                lg = ax.legend(
                    ordered_handles,
                    ordered_labels,
                    loc="upper right",
                    title="{} ({})".format(flux_type, self.experiment_fluxes_units[0]),
                    shadow=True,
                    numpoints=numpoints,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=plt.gcf().transFigure,
                )
            fr = lg.get_frame()
            fr.set_lw(legend_frame_width)
        # Replot observations
        if has_observations:
            if simple_plot:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.35")
            else:
                ax.plot(profile_axis_out, obs_o_vals, "-", color="0.5")
                ax.plot(
                    profile_axis_out,
                    obs_o_vals,
                    dash_style,
                    color=obscolor,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                )
        if plot_title:
            plt.title(gate_name, loc="left")

        if normalize:
            gate_name = "_".join(
                [unidecode(gate.gate_name), varname, "normalized", "profile"]
            )
        else:
            gate_name = "_".join([unidecode(gate.gate_name), varname, "profile"])
        outname = ".".join([gate_name, "pdf"]).replace(" ", "_")
        print(("Saving {0}".format(outname)))
        fig.tight_layout()
        fig.savefig(outname)
        plt.close(fig)


class FluxGateExperiment(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateExperiment, self).__init__(*args, **kwargs)
        self.values = data.values[pos_id, Ellipsis]
        self.config = data.config
        self.id = data.id

    def __repr__(self):
        return "FluxGateExperiment"


class FluxGateObservations(object):
    def __init__(self, data, pos_id, *args, **kwargs):
        super(FluxGateObservations, self).__init__(*args, **kwargs)
        self.has_error = None
        self.values = data.values[pos_id, Ellipsis]
        if data.has_error:
            self.error = data.error[pos_id, Ellipsis]
            self.has_error = True

    def __repr__(self):
        return "FluxGateObservations"


class Dataset(object):

    """
    A base class for Experiments or Observations.

    Constructor opens netCDF file, attaches pointer to nc instance.

    """

    def __init__(self, filename, varname, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        print(("  opening NetCDF file %s ..." % filename))
        try:
            nc = NC(filename, "r")
        except:
            print(
                (
                    "ERROR:  file '%s' not found or not NetCDF format ... ending ..."
                    % filename
                )
            )
            import sys

            sys.exit(1)

        for name in nc.variables:
            v = nc.variables[name]
            if getattr(v, "standard_name", "") == varname:
                print(
                    ("variabe {0} found by its standard_name {1}".format(name, varname))
                )
                varname = name

        self.values = np.squeeze(nc.variables[varname][:])
        self.varname_units = nc.variables[varname].units
        self.varname = varname
        self.nc = nc

    def __repr__(self):
        return "Dataset"

    def __del__(self):
        # Close open file
        self.nc.close()


class ExperimentDataset(Dataset):

    """
    A derived class for experiments

    A derived class for handling PISM experiments.

    Experiments are identified by id. Config and run_stats
    attributes are attached to object as config dictionary.

    """

    def __init__(self, id, *args, **kwargs):
        super(ExperimentDataset, self).__init__(*args, **kwargs)

        print("Experiment {}".format(id))
        self.id = id
        self.config = dict()
        for v in ["pism_config", "run_stats", "config"]:
            if v in self.nc.variables:
                ncv = self.nc.variables[v]
                for attr in ncv.ncattrs():
                    self.config[attr] = getattr(ncv, attr)
            else:
                print("Variable {} not found".format(v))

    def __repr__(self):
        return "ExperimentDataset"


class ObservationsDataset(Dataset):

    """
    A derived class for observations.

    A derived class for handling observations.

    """

    def __init__(self, *args, **kwargs):
        super(ObservationsDataset, self).__init__(*args, **kwargs)
        self.has_error = None
        try:
            self.clon = self.nc.variables["clon"][:]
        except:
            self.clon = None
        try:
            self.clat = self.nc.variables["clat"][:]
        except:
            self.clat = None
        try:
            self.flightline = self.nc.variables["flightline"][:]
        except:
            self.flightline = None
        try:
            self.glaciertype = self.nc.variables["glaciertype"][:]
        except:
            self.glaciertype = None
        try:
            self.flowtype = self.nc.variables["flowtype"][:]
        except:
            self.flowtype = None
        varname = self.varname
        error_varname = "_".join([varname, "error"])
        if error_varname in list(self.nc.variables.keys()):
            self.error = self.nc.variables[error_varname][:]
            self.has_error = True

    def __repr__(self):
        return "ObservationsDataset"


def export_latex_table_flux(filename, flux_gates, params):
    """
    Create a latex table with fluxes through gates.

    Create a latex table with flux gates sorted by
    observed flux in decreasing order.

    Parameters
    ----------

    filename: string, name of the outputfile
    flux_gates: list of FluxGate objects
    params: dict, parameters to be listed
    """

    f = codecs.open(filename, "w", "utf-8")
    tab_str = " ".join(["{l r r c c}"])
    f.write(" ".join(["\\begin{tabular}", tab_str, "\n"]))
    f.write("\\toprule \n")
    f.write(
        "Glacier & flux & with & glacier type & flow type \\\ \n".format(
            v_flux_o_units_str_tex
        )
    )
    f.write(
        " &  ({}) &  ({}) \\\ \n".format(v_flux_o_units_str_tex, profile_axis_out_units)
    )
    f.write("\midrule \n")
    # We need to calculate fluxes first
    for gate in flux_gates:
        gate.calculate_fluxes()
    cum_flux = 0
    cum_flux_error = 0
    cum_length = 0
    # Sort: largest flux gate first
    for gate in sorted(flux_gates, key=lambda x: x.observed_flux, reverse=True):
        profile_axis = gate.profile_axis
        profile_axis_units = gate.profile_axis_units
        length = gate.length()
        i_units_cf = cf_units.Unit(profile_axis_units)
        o_units_cf = cf_units.Unit(profile_axis_out_units)
        profile_length = i_units_cf.convert(length, o_units_cf)
        glaciertype = glacier_types[gate.glaciertype]
        flowtype = flow_types[gate.flowtype]
        line_str = "".join(
            [
                gate.return_gate_flux_str_short(),
                "& {:2.1f} ".format(profile_length),
                "& {} ".format(glaciertype),
                "& {} ".format(flowtype),
                "\\\ \n",
            ]
        )
        cum_flux += gate.observed_flux
        cum_flux_error += gate.observed_flux_error**2
        cum_length += profile_length
        f.write(line_str)
    cum_flux_error = np.sqrt(cum_flux_error)
    f.write("\midrule \n")
    f.write(
        "Total & {:2.1f}$\pm${:2.1f} & {:2.1f} \\\ \n".format(
            cum_flux, cum_flux_error, cum_length
        )
    )
    f.write("\\bottomrule \n")
    f.write("\\end{tabular} \n")
    f.close


def export_latex_table_rmsd(filename, gate):
    """
    Create a latex table for a flux gate

    Create a latex table for a flux gate with a sorted list of
    experiments increasing in RMSD.

    Parameters
    ----------
    filename: string
    gate: FluxGate

    """

    i_units_cf = cf_units.Unit(gate.varname_units)
    o_units_cf = cf_units.Unit(v_o_units)
    error_norm = i_units_cf.convert(gate.sigma_obs, o_units_cf)
    f = codecs.open(filename, "w", "utf-8")
    f.write("\\begin{tabular} {l l cc }\n")
    f.write("\\toprule \n")
    f.write(
        "\multicolumn{{2}}{{l}}{{{}}} & flux  & $\chi_{{g}}$ \\\ \n".format(
            unidecode(gate.gate_name)
        )
    )
    f.write("&& ({})  & ({})  \\\ \n".format(v_flux_o_units_str_tex, v_o_units_str_tex))
    f.write("\midrule\n")
    observed_flux = gate.observed_flux
    observed_flux_error = gate.observed_flux_error
    f.write(
        "observed && {:3.1f}$\pm${:2.1f} & {:2.2f} \\\ \n".format(
            observed_flux, observed_flux_error, error_norm
        )
    )
    best_rmsd = gate.best_rmsd
    for k, val in enumerate(
        sorted(gate.p_ols, key=lambda x: gate.rmsd[x], reverse=False)
    ):
        config = gate.experiments[val].config
        my_flux = gate.experiment_fluxes[val]
        my_exp_str = ", ".join(
            [
                "=".join(
                    [
                        params_dict[key]["abbr"],
                        params_dict[key]["format"].format(config.get(key)),
                    ]
                )
                for key in label_params
            ]
        )
        if k == 0:
            f.write(
                "Experiment {} & {} & {:2.1f}  & \\textit{{{:1.2f}}} \\\ \n".format(
                    val, my_exp_str, my_flux, gate.rmsd[val]
                )
            )
        else:
            if gate.rmsd[val] < (error_norm + best_rmsd):
                f.write(
                    "Experiment {} & {} & {:2.1f}  & \\textit{{{:1.2f}}} \\\ \n".format(
                        val, my_exp_str, my_flux, gate.rmsd[val]
                    )
                )
            else:
                f.write(
                    "Experiment {} & {} & {:2.1f}  & {:1.2f} \\\ \n".format(
                        val, my_exp_str, my_flux, gate.rmsd[val]
                    )
                )
    f.write("\\bottomrule\n")
    f.write("\end{tabular}\n")
    f.close()


def export_latex_table_corr(filename, gate):
    """
    Create a latex table for a flux gate

    Create a latex table for a flux gate with a sorted list of
    experiments increasing in correlation coefficient.

    Parameters
    ----------
    filename: string
    gate: FluxGate

    """

    f = codecs.open(filename, "w", "utf-8")
    f.write("\\begin{tabular} {l l cc }\n")
    f.write("\\toprule \n")
    f.write(
        "\multicolumn{{2}}{{l}}{{{}}} &  $r$ & increase \\\ \n".format(
            unidecode(gate.gate_name)
        )
    )
    f.write("&& (-)  & (\%)  \\\ \n")
    f.write("\midrule\n")
    best_corr = gate.best_corr
    for k, val in enumerate(
        sorted(gate.p_ols, key=lambda x: gate.corr[x], reverse=False)
    ):
        config = gate.experiments[val].config
        my_flux = gate.experiment_fluxes[val]
        my_exp_str = ", ".join(
            [
                "=".join(
                    [
                        params_dict[key]["abbr"],
                        params_dict[key]["format"].format(config.get(key)),
                    ]
                )
                for key in label_params
            ]
        )
        corr = gate.corr[val]
        if k == 0:
            corr_0 = corr
            f.write(
                "Experiment {} & {}   & {:1.2f} \\\ \n".format(val, my_exp_str, corr)
            )
        else:
            pc_inc = (corr - corr_0) / corr_0 * 100
            f.write(
                "Experiment {} & {}  & {:1.2f} & +{:2.0f}\\\ \n".format(
                    val, my_exp_str, corr, pc_inc
                )
            )
    f.write("\\bottomrule\n")
    f.write("\end{tabular}\n")
    f.close()


def export_gate_table_rmsd(filename, exp):
    """
    Creates a latex table of flux gates sorted by rmsd.

    Parameters
    ----------
    filename: string
    exp: FluxGateExperiment

    """

    means = {}
    rmsds = {}
    rmsds_rels = {}
    for gate in flux_gates:
        id = gate.pos_id
        # Get uncertainty and convert units
        i_units_cf = cf_units.Unit(gate.observed_mean_units)
        o_units_cf = cf_units.Unit(v_o_units)
        mean = i_units_cf.convert(gate.observed_mean, o_units_cf)
        means[id] = mean
        # Get RMSD and convert units
        i_units_cf = cf_units.Unit(gate.rmsd_units)
        o_units_cf = cf_units.Unit(v_o_units)
        rmsd = i_units_cf.convert(gate.rmsd[exp.id], o_units_cf)
        rmsds[id] = rmsd
        rmsd_rel = rmsd / mean
        rmsds_rels[id] = rmsd_rel

    rmsds_rels_sorted = sorted(rmsds_rels, key=lambda x: rmsds_rels[x])

    f = codecs.open(filename, "w", "utf-8")
    tab_str = " ".join(["{l ccc }"])
    f.write(" ".join(["\\begin{tabular}", tab_str, "\n"]))
    f.write("\\toprule \n")
    f.write(
        "Glacier & $\\bar U_{\\textrm{{obs,g}}}$ & $\chi_{\\textrm{{g}}}$ & $\\tilde \chi_{\\textrm{{g}}}$ \\\ \n"
    )
    f.write(" & ({}) & ({}) & (-) \\\ \n".format(v_o_units_str_tex, v_o_units_str_tex))
    f.write("\midrule \n")

    for k in rmsds_rels_sorted:
        gate = flux_gates[k]
        line_str = " & ".join(
            [
                unidecode(gate.gate_name),
                "{:1.0f}".format(np.float(means[k])),
                "{:1.0f}".format(np.float(rmsds[k])),
                "{:1.2f} \\\ \n".format(np.float(rmsds_rels[k])),
            ]
        )

        f.write(line_str)
    f.write("\\bottomrule \n")
    f.write("\\end{tabular} \n")
    f.close


def export_csv_from_dict(filename, mdict, header=None, fmt=["%i", "%4.2f"]):
    """
    Creates a CSV file from a dictionary.

    Parameters
    ----------
    filename: string
    mdict: dictionary with id and data

    """

    ids = [x for x in mdict.keys()]
    values = [x for x in mdict.values()]
    data = np.vstack((ids, values))
    np.savetxt(
        filename, np.transpose(data), fmt=["%i", "%4.2f"], delimiter=",", header=header
    )


def write_experiment_table(outname):
    """
    Export a table with all experiments
    """

    f = codecs.open(outname, "w", "utf-8")
    r_str = "r" * ne
    tab_str = " ".join(["{l  c", r_str, "}"])
    f.write(" ".join(["\\begin{tabular}", tab_str, "\n"]))
    f.write("\\toprule \n")
    exp_str = " & ".join([params_dict[key]["abbr"] for key in label_params])
    line_str = " & ".join(["Experiment", exp_str])
    f.write(" ".join([line_str, r"\\", "\n"]))
    f.write("\midrule \n")
    for exp in flux_gates[0].experiments:
        config = exp.config
        id = exp.id
        param_str = " & ".join(
            [params_dict[key]["format"].format(config.get(key)) for key in label_params]
        )
        line_str = " ".join([" & ".join(["{:1.0f}".format(id), param_str]), "\\\ \n"])
        f.write(line_str)
    f.write("\\bottomrule \n")
    f.write("\\end{tabular} \n")
    f.close()


def make_correlation_figure(filename, exp):
    """
    Create a Pearson R correlation plot.

    Create a correlation plot for a given experiment, sorted by
    decreasing correlation

    Parameters
    ----------
    filename: string
    exp: FluxGateExperiment

    """
    corrs = {}
    for gate in flux_gates:
        id = gate.pos_id
        r = gate.corr[exp.id]
        if not np.isnan(r):
            corrs[id] = r
    sort_order = sorted(corrs, key=lambda x: corrs[x])
    corrs_sorted = [corrs[x] for x in sort_order]
    names_sorted = [flux_gates[x].gate_name for x in sort_order]
    gate_id_sorted = [flux_gates[x].gate_id for x in sort_order]
    corrs_dict = dict(zip(gate_id_sorted, corrs_sorted))
    lw, pad_inches = set_mode(print_mode, aspect_ratio=1.2)
    fig = plt.figure(figsize=[6.4, 12])
    ax = fig.add_subplot(111)
    height = 0.4
    y = np.arange(len(list(corrs.keys()))) + 1.25
    for k, corr in enumerate(corrs_sorted):
        if corr < 0.5:
            colorVal = "#d7191c"
        elif (corr >= pearson_r_threshold_low) and (corr < pearson_r_threshold_high):
            colorVal = "#ff7f00"
        else:
            colorVal = "#33a02c"
        ax.hlines(y[k], -1, corr, colors=colorVal, linestyle="dotted")
        ax.plot(corr, y[k], "o", markersize=5, color=colorVal)

    corr_median = np.nanmedian(list(corrs.values()))
    ax.vlines(corr_median, 0, y[-1], linestyle="dotted", color="0.5")
    print(("median correlation: {:1.2f}".format(corr_median)))
    plt.yticks(
        y,
        [
            "{} ({})".format(flux_gates[x].gate_name, flux_gates[x].gate_id)
            for x in sort_order
        ],
    )
    ax.set_xlabel("r (-)", labelpad=0.2)
    ax.set_xlim(-1, 1.1)
    ax.set_ylim(0, y[-1] + 1)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    # Only draw spine between the y-ticks
    ax.spines["left"].set_bounds(y[0], y[-1])
    ax.spines["bottom"].set_bounds(-1, 1)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    fig.tight_layout()
    print(("Saving {0}".format(filename)))
    fig.savefig(filename)
    plt.close("all")
    return corrs_dict


def make_regression():
    grid_dx_meters = [x.config["grid_dx_meters"] for x in gate.experiments]
    for gate in flux_gates:

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        d = {"grid_resolution": gridS, "RMSD": rmsdS}
        df = pa.DataFrame(d)
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # Calculate PISM trends and biases (intercepts)
        bias, trend = model.params
        # Calculate r-squared value
        r2 = model.rsquared
        # make x lims from 0 to 5000 m
        xmin, xmax = 0, 5000
        # Create figures
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot([grid_dx_meters[0], grid_dx_meters[-1]], bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]])*trend, color='0.2')
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color="0.2",
            markeredgewidth=markeredgewidth,
        )
        ax.set_xticks(grid_dx_meters)
        ax.set_xlabel("grid resolution (m)")
        ax.set_ylabel("$\chi$ ({})".format(v_o_units_str))
        ax.set_xlim(xmin, xmax)
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(30)
        plt.title(gate.gate_name)
        fig.tight_layout()
        gate_name = "_".join([unidecode(gate.gate_name), varname, "rmsd", "regress"])
        outname = ".".join([gate_name, "pdf"]).replace(" ", "_")
        print(("Saving {0}".format(outname)))
        fig.savefig(outname)
        plt.close("all")

    nocol = 4
    colormap = ["RdYlGn", "Diverging", nocol, 0]
    my_ok_colors = colorbrewer.get_map(*colormap).mpl_colors

    grid_dx_meters = [x.config["grid_dx_meters"] for x in gate.experiments]
    lw, pad_inches = set_mode(print_mode, aspect_ratio=1.25)

    # Create RMSD figure
    fig = plt.figure()
    # make x lims from 450 to 5000 m
    xmin, xmax = 0, 5000
    ax = fig.add_subplot(111)
    legend_handles = []
    for n, gate in enumerate(flux_gates):
        id = gate.pos_id

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        d = {"grid_resolution": gridS, "RMSD": rmsdS}
        df = pa.DataFrame(d)
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # trend and bias (intercept)
        bias, trend = model.params
        # r-squared value
        r2 = model.rsquared
        # p-value
        p = model.fpvalue
        f = model.fvalue

        gate.linear_trend = trend
        gate.linear_bias = bias
        gate.linear_r2 = r2
        gate.linear_p = p

        # select glaciers that don't have a significant (95%) trend
        # and denote them by a dashed line
        if p >= 0.05:
            ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
                linestyle="dashed",
                color="0.7",
                linewidth=0.5,
            )
        else:
            ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
                color="0.7",
                linewidth=0.5,
            )
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color="0.7",
            markeredgewidth=markeredgewidth,
            markeredgecolor="0.7",
            markersize=1.75,
        )

    for id in (1, 11, 19, 23):

        gate = flux_gates[id]
        # print(u"selecting glacier {}".format(gate.gate_name))

        # RMSD
        rmsd_data = list(gate.rmsd.values())
        rmsdS = pa.Series(data=rmsd_data, index=list(gate.rmsd.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
        d = {"grid_resolution": gridS, "RMSD": rmsdS}
        df = pa.DataFrame(d)
        model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
        # trend and bias (intercept)
        bias_selected, trend_selected = model.params
        p_selected = model.f_pvalue

        if id == 1:  # Jakobshavn
            colorVal = "#54278f"
        elif id == 11:  # Kong Oscar
            colorVal = "#006d2c"
        elif id == 19:  # Kangerdlugssuaq
            colorVal = "#08519c"
        elif id == 23:  # Koge Bugt S
            colorVal = "#a50f15"
        else:
            print("How did I get here?")

        if p_selected >= 0.05:
            (line_l,) = ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias_selected
                + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_selected,
                linestyle="dashed",
                color=colorVal,
                linewidth=0.5,
            )
        else:
            (line_l,) = ax.plot(
                [grid_dx_meters[0], grid_dx_meters[-1]],
                bias_selected
                + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_selected,
                color=colorVal,
                linewidth=0.5,
            )
        ax.plot(
            grid_dx_meters,
            rmsd_data,
            dash_style,
            color=colorVal,
            markeredgewidth=markeredgewidth * 0.8,
            markeredgecolor="0.2",
            markersize=1.75,
        )
        legend_handles.append(line_l)

    # all isbrae RMSD
    rmsd_data = list(rmsd_isbrae_cum_dict.values())
    rmsdS = pa.Series(data=rmsd_data, index=list(rmsd_isbrae_cum_dict.keys()))
    gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
    d = {"grid_resolution": gridS, "RMSD": rmsdS}
    df = pa.DataFrame(d)
    model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
    # Calculate PISM trends and biases (intercepts)
    bias_isbrae, trend_isbrae = model.params
    r2_isbrae = model.rsquared
    p_isbrae = model.f_pvalue
    if p_isbrae > 0.05:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_isbrae
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_isbrae,
            color="#8c510a",
            linewidth=1,
            linestyle="dashed",
        )
    else:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_isbrae
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_isbrae,
            color="#8c510a",
            linewidth=1,
        )
    ax.plot(
        grid_dx_meters,
        rmsd_data,
        dash_style,
        color="#8c510a",
        markeredgewidth=markeredgewidth,
    )
    legend_handles.append(line_l)

    print(("Isbrae regression r2 = {:2.2f}".format(r2_isbrae)))

    # all ice-stream RMSD
    rmsd_data = list(rmsd_ice_stream_cum_dict.values())
    rmsdS = pa.Series(data=rmsd_data, index=list(rmsd_ice_stream_cum_dict.keys()))
    gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
    d = {"grid_resolution": gridS, "RMSD": rmsdS}
    df = pa.DataFrame(d)
    model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
    # Calculate PISM trends and biases (intercepts)
    bias_ice_stream, trend_ice_stream = model.params
    r2_ice_stream = model.rsquared
    p_ice_stream = model.f_pvalue
    if p_ice_stream > 0.05:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_ice_stream
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_ice_stream,
            color="#01665e",
            linewidth=1,
            linestyle="dashed",
        )
    else:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_ice_stream
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_ice_stream,
            color="#01665e",
            linewidth=1,
        )

    ax.plot(
        grid_dx_meters,
        rmsd_data,
        dash_style,
        color="#01665e",
        markeredgewidth=markeredgewidth,
    )
    legend_handles.append(line_l)

    print(("Ice-stream regression r2 = {:2.2f}".format(r2_ice_stream)))

    # global RMSD
    rmsd_data = list(rmsd_cum_dict.values())
    rmsdS = pa.Series(data=rmsd_data, index=list(rmsd_cum_dict.keys()))
    gridS = pa.Series(data=grid_dx_meters, index=list(gate.rmsd.keys()))
    d = {"grid_resolution": gridS, "RMSD": rmsdS}
    df = pa.DataFrame(d)
    model = sm.OLS(rmsdS, sm.add_constant(gridS)).fit()
    # Calculate PISM trends and biases (intercepts)
    bias_global, trend_global = model.params
    r2_global = model.rsquared
    p_global = model.f_pvalue
    if p_global > 0.05:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_global
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_global,
            color="0.2",
            linewidth=1,
            linestyle="dashed",
        )
    else:
        (line_l,) = ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias_global
            + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend_global,
            color="0.2",
            linewidth=1,
        )
    ax.plot(
        grid_dx_meters,
        rmsd_data,
        dash_style,
        color="0.4",
        markeredgewidth=markeredgewidth,
    )
    legend_handles.append(line_l)

    print(("Global regression r2 = {:2.2f}".format(r2_global)))

    legend_labels = ["JIB", "good", "median", "poor", "isbr\u00E6", "ice-stream", "all"]
    ax.set_xticks(grid_dx_meters)
    ax.set_xlabel("grid resolution (m)")
    ax.set_ylabel("$\chi$ ({})".format(v_o_units_str))
    ax.set_xlim(xmin, xmax)

    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(40)

    fig.tight_layout()
    outname = ".".join(["rmsd_regression", "pdf"]).replace(" ", "_")
    print(("Saving {0}".format(outname)))
    fig.savefig(outname)
    plt.close("all")

    # Create R2 figures
    fig = plt.figure()
    # make x lims from 0 to 5000 m
    xmin, xmax = 0, 5000
    ax = fig.add_subplot(111)

    for n, gate in enumerate(flux_gates):
        # R2
        r2_data = list(gate.r2.values())
        r2S = pa.Series(data=r2_data, index=list(gate.r2.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.r2.keys()))
        d = {"grid_resolution": gridS, "R2": r2S}
        df = pa.DataFrame(d)
        model = sm.OLS(r2S, sm.add_constant(gridS)).fit()
        # Calculate PISM trends and biases (intercepts)
        bis, trend = model.params
        # Calculate r-squared value
        r2 = model.rsquared

        ax.plot(
            [grid_dx_meters[0], grid_dx_meters[-1]],
            bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
            color="#a6cee3",
            linewidth=0.35,
        )
        ax.plot(
            grid_dx_meters,
            r2_data,
            dash_style,
            color="#a6cee3",
            markeredgewidth=markeredgewidth,
            markeredgecolor="#1f78b4",
            markersize=1.75,
        )

    # global R2
    r2_data = list(r2_cum_dict.values())
    r2S = pa.Series(data=r2_data, index=list(r2_cum_dict.keys()))
    gridS = pa.Series(data=grid_dx_meters, index=list(gate.r2.keys()))
    d = {"grid_resolution": gridS, "R2": r2S}
    df = pa.DataFrame(d)
    model = sm.OLS(gridS, sm.add_constant(r2S)).fit()
    # Calculate PISM trends and biases (intercepts)
    bias, trend = model.params
    # Calculate r-squared value
    r2 = model.rsquared
    # plot trend line
    ax.plot(
        [grid_dx_meters[0], grid_dx_meters[-1]],
        bias + np.array([grid_dx_meters[0], grid_dx_meters[-1]]) * trend,
        color="0.2",
    )
    # plot errors
    ax.plot(
        grid_dx_meters,
        r2_data,
        dash_style,
        color="0.4",
        markeredgewidth=markeredgewidth,
    )
    # print statistics
    ax.text(
        0.05, 0.7, "r$^\mathregular{{2}}$={:1.2f}".format(r2), transform=ax.transAxes
    )
    ax.set_xticks(grid_dx_meters)
    ax.set_xlabel("grid resolution (m)")
    ax.set_ylabel("r$^\mathregular{{2}}$ (-)")
    ax.set_xlim(xmin, xmax)

    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(40)

    fig.tight_layout()
    outname = ".".join(["r2_regression", "pdf"]).replace(" ", "_")
    print(("Saving {0}".format(outname)))
    fig.savefig(outname)
    plt.close("all")

    # Create correlation figures
    fig = plt.figure()
    # make x lims from 0 to 5000 m
    xmin, xmax = 0, 5000

    jet = cm = plt.get_cmap("jet")
    cNorm = mplcolors.Normalize(vmin=0, vmax=15)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    ax = fig.add_subplot(111)
    for n, gate in enumerate(flux_gates):
        # correlation
        corr_data = list(gate.corr.values())
        corrS = pa.Series(data=corr_data, index=list(gate.corr.keys()))
        gridS = pa.Series(data=grid_dx_meters, index=list(gate.corr.keys()))
        d = {"grid_resolution": gridS, "correlation": corrS}
        df = pa.DataFrame(d)
        model = pa.ols(x=gridS, y=corrS)
        # Calculate PISM trends and biases (intercepts)
        trend, bias = model.beta
        # Calculate r-squared value
        r2 = model.r2

        colorVal = scalarMap.to_rgba(n)
        if corr_data[0] >= 0.85:
            ax.plot(
                grid_dx_meters,
                corr_data,
                dash_style,
                color=colorVal,
                markeredgewidth=markeredgewidth,
                markeredgecolor="k",
                markersize=2,
            )

    ax.set_xticks(grid_dx_meters)
    ax.set_xlabel("grid resolution (m)")
    ax.set_ylabel("correlation coefficient (-)")
    ax.set_xlim(500, 2000)
    ax.set_ylim(0.85, 1)

    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(40)

    fig.tight_layout()
    outname = ".".join(["pearson_r_regression", "pdf"]).replace(" ", "_")
    print(("Saving {0}".format(outname)))
    fig.savefig(outname)
    plt.close("all")


def write_shapefile(filename, flux_gates):
    """
    Writes metrics to a ESRI shape file.

    Paramters
    ----------
    filename: filename of ESRI shape file.
    flux_gates: list of FluxGates

    """

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(filename):
        os.remove(filename)
    ds = driver.CreateDataSource(filename)
    # Create spatialReference, EPSG 4326 (lonlat)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("Flux Gates", srs, ogr.wkbPoint)
    g_name = ogr.FieldDefn("name", ogr.OFTString)
    layer.CreateField(g_name)
    g_id = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(g_id)
    obs_flux = ogr.FieldDefn("obs", ogr.OFTReal)
    layer.CreateField(obs_flux)
    obs_flux_error = ogr.FieldDefn("obs_e", ogr.OFTReal)
    layer.CreateField(obs_flux_error)
    obs_mean = ogr.FieldDefn("obs_mean", ogr.OFTReal)
    layer.CreateField(obs_mean)
    gtype = ogr.FieldDefn("gtype", ogr.OFTInteger)
    layer.CreateField(gtype)
    ftype = ogr.FieldDefn("ftype", ogr.OFTInteger)
    layer.CreateField(ftype)
    flightline = ogr.FieldDefn("flightline", ogr.OFTInteger)
    layer.CreateField(flightline)
    for cnt in range(flux_gates[0].exp_counter):
        for var in ("skill", "r2", "r", "rmsd", "sigma", "exp"):
            my_exp = "_".join([var, str(int(cnt))])
            my_var = ogr.FieldDefn(my_exp, ogr.OFTReal)
            layer.CreateField(my_var)

    for svar in ("lin_trend", "lin_bias", "lin_r2", "lin_p"):
        ogrvar = ogr.FieldDefn(svar, ogr.OFTReal)
        layer.CreateField(ogrvar)

    featureIndex = 0
    layer_defn = layer.GetLayerDefn()
    for gate in flux_gates:
        # Create point
        geometry = ogr.Geometry(ogr.wkbPoint)
        geometry.SetPoint(0, float(gate.clon), float(gate.clat))
        # Create feature
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        feature.SetFID(featureIndex)
        i = feature.GetFieldIndex("id")
        feature.SetField(i, gate.gate_id)
        i = feature.GetFieldIndex("name")
        # This does not work though it should?
        # feature.SetField(i, gate.gate_name.encode('utf-8'))
        feature.SetField(i, unidecode(gate.gate_name))
        if gate.observed_flux is not None:
            i = feature.GetFieldIndex("obs")
            feature.SetField(i, gate.observed_flux)
        if gate.observed_flux_error is not None:
            i = feature.GetFieldIndex("obs_e")
            feature.SetField(i, gate.observed_flux_error)
        if gate.observed_mean is not None:
            i = feature.GetFieldIndex("obs_mean")
            feature.SetField(i, float(gate.observed_mean))
        # OGR doesn't like numpy.int8
        if gate.glaciertype is not (None or ""):
            i = feature.GetFieldIndex("gtype")
            feature.SetField(i, int(gate.glaciertype))
        if gate.flowtype is not (None or ""):
            i = feature.GetFieldIndex("ftype")
            feature.SetField(i, int(gate.flowtype))
        if gate.flightline is not (None or ""):
            i = feature.GetFieldIndex("flightline")
            feature.SetField(i, int(gate.flightline))
        if gate.linear_trend is not (None or ""):
            i = feature.GetFieldIndex("lin_trend")
            feature.SetField(i, gate.linear_trend)
        if gate.linear_bias is not (None or ""):
            i = feature.GetFieldIndex("lin_bias")
            feature.SetField(i, gate.linear_bias)
        if gate.linear_r2 is not (None or ""):
            i = feature.GetFieldIndex("lin_r2")
            feature.SetField(i, gate.linear_r2)
        if gate.linear_p is not (None or ""):
            i = feature.GetFieldIndex("lin_p")
            feature.SetField(i, gate.linear_p)
        for cnt in range(flux_gates[0].exp_counter):
            flux_exp = "_".join(["exp", str(int(cnt))])
            i = feature.GetFieldIndex(flux_exp)
            exp_flux = gate.experiment_fluxes[cnt]
            feature.SetField(i, exp_flux)
            if gate.p_ols is not (None or ""):
                r2_exp = "_".join(["r2", str(int(cnt))])
                i = feature.GetFieldIndex(r2_exp)
                feature.SetField(i, gate.p_ols[cnt].rsquared)
                corr_exp = "_".join(["r", str(int(cnt))])
                i = feature.GetFieldIndex(corr_exp)
                feature.SetField(i, gate.corr[cnt])
            rmsd_exp = "_".join(["rmsd", str(int(cnt))])
            i = feature.GetFieldIndex(rmsd_exp)
            if gate.rmsd is not (None or ""):
                i_units_cf = cf_units.Unit(gate.rmsd_units)
                o_units_cf = cf_units.Unit(v_o_units)
                chi = i_units_cf.convert(gate.rmsd[cnt], o_units_cf)
                feature.SetField(i, chi)
            sigma_exp = "_".join(["sigma", str(int(cnt))])
            i = feature.GetFieldIndex(sigma_exp)
            if gate.sigma_obs is not (None or ""):
                i_units_cf = cf_units.Unit(gate.varname_units)
                o_units_cf = cf_units.Unit(v_o_units)
                sigma_obs = i_units_cf.convert(gate.sigma_obs, o_units_cf)
                feature.SetField(i, sigma_obs)
        # Save feature
        layer.CreateFeature(feature)
        layer.SetFeature(feature)
        # Cleanup
        geometry = None
        feature = None

    ds = None


def export_rmsd_latex_table():
    """
    Used for "Complex Greenland Outlet Glacier Flow Captured
    """

    # RMSD LaTeX table
    outname = ".".join(["rmsd_cum_table", "tex"])
    print(("Saving {0}".format(outname)))
    with codecs.open(outname, "w", "utf-8") as f:
        f.write("\\begin{tabular} {l l ccccccccc }\n")
        f.write("\\toprule \n")
        f.write(
            "Exp.  & Parameters  & $\\tilde r_{\\textrm{ib}}$ & $\\tilde r_{\\textrm{is}}$  & $\\tilde r$ & $\chi_{\\textrm{ib}}$ & inc. & $\chi_{\\textrm{is}}$ & inc. & $\chi$ & inc.\\\ \n"
        )
        f.write(
            "  & & (-) & (-) & (-)  & ({}) & (\%) & ({}) & (\%) & ({}) & (\%)\\\ \n".format(
                v_o_units_str_tex, v_o_units_str_tex, v_o_units_str_tex
            )
        )
        f.write("\midrule\n")
        for k, exp in enumerate(rmsd_cum_dict_sorted):
            id = exp[0]
            rmsd_cum = exp[1]
            rmsd_isbrae_cum = rmsd_isbrae_cum_dict[id]
            rmsd_ice_stream_cum = rmsd_ice_stream_cum_dict[id]
            my_exp = flux_gates[0].experiments[id]
            config = my_exp.config
            corr = []
            corr_isbrae = []
            corr_ice_stream = []
            for gate in flux_gates:
                corr.append(gate.corr[id])
                if gate.flowtype == 0:
                    corr_isbrae.append(gate.corr[id])
                if gate.flowtype == 1:
                    corr_ice_stream.append(gate.corr[id])
            corr_median = np.nanmedian(corr)
            corr_isbrae_median = np.nanmedian(corr_isbrae)
            corr_ice_stream_median = np.nanmedian(corr_ice_stream)
            my_exp_str = ", ".join(
                [
                    "=".join(
                        [
                            params_dict[key]["abbr"],
                            params_dict[key]["format"].format(config.get(key)),
                        ]
                    )
                    for key in label_params
                ]
            )
            if k == 0:
                rmsd_cum_0 = rmsd_cum
                rmsd_isbrae_cum_0 = rmsd_isbrae_cum
                rmsd_ice_stream_cum_0 = rmsd_ice_stream_cum
            if k == 0:
                f.write(
                    " {:2.0f} & {} & {:2.2f} & {:2.2f} & {:2.2f} & {:1.0f} & & {:1.0f} & & {:1.0f} & \\\ \n".format(
                        id,
                        my_exp_str,
                        corr_isbrae_median,
                        corr_ice_stream_median,
                        corr_median,
                        rmsd_isbrae_cum,
                        rmsd_ice_stream_cum,
                        rmsd_cum,
                    )
                )
            else:
                pc_inc = (rmsd_cum - rmsd_cum_0) / rmsd_cum_0 * 100
                pc_isbrae_inc = (
                    (rmsd_isbrae_cum - rmsd_isbrae_cum_0) / rmsd_isbrae_cum_0 * 100
                )
                pc_ice_stream_inc = (
                    (rmsd_ice_stream_cum - rmsd_ice_stream_cum_0)
                    / rmsd_ice_stream_cum_0
                    * 100
                )
                f.write(
                    " {:2.0f} & {} & {:2.2f}  & {:2.2f} & {:2.2f} & {:1.0f} & +{:2.0f} & {:1.0f} & +{:2.0f}  & {:1.0f} & +{:2.0f} \\\ \n".format(
                        id,
                        my_exp_str,
                        corr_isbrae_median,
                        corr_ice_stream_median,
                        corr_median,
                        rmsd_isbrae_cum,
                        pc_isbrae_inc,
                        rmsd_ice_stream_cum,
                        pc_ice_stream_inc,
                        rmsd_cum,
                        pc_inc,
                    )
                )
        f.write("\\bottomrule\n")
        f.write("\end{tabular}\n")
        f.close()


# ##############################################################################
# MAIN
# ##############################################################################

if __name__ == "__main__":

    __spec__ = None

    # Set up the option parser
    parser = ArgumentParser()
    parser.description = (
        "Analyze flux gates. Used for 'Complex Greenland Outlet Glacier Flow Captured'."
    )
    parser.add_argument("FILE", nargs="*")
    parser.add_argument(
        "--aspect_ratio",
        dest="aspect_ratio",
        type=float,
        help='''Plot aspect ratio"''',
        default=0.8,
    )
    parser.add_argument(
        "--colormap",
        dest="colormap",
        nargs=1,
        help="""Name of matplotlib colormap""",
        default="tab20c",
    )
    parser.add_argument(
        "--label_params",
        dest="label_params",
        help='''comma-separated list of parameters that appear in the legend,
                      e.g. "sia_enhancement_factor"''',
        default="dataset",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Normalize experiments by muliplying with max(obs)/max(experiment)",
        default=False,
    )
    parser.add_argument(
        "--obs_file",
        dest="obs_file",
        help="""Profile file with observations. Default is None""",
        default=None,
    )
    parser.add_argument(
        "--export_table_file",
        dest="table_file",
        help="""If given, fluxes are exported to latex table. Default is None""",
        default=None,
    )
    parser.add_argument(
        "--no_figures",
        dest="make_figures",
        action="store_false",
        help="Do not make profile figures",
        default=True,
    )
    parser.add_argument(
        "--do_regress",
        dest="do_regress",
        action="store_true",
        help="Make grid resolution regression plots",
        default=False,
    )
    parser.add_argument(
        "--legend",
        dest="legend",
        choices=["default", "none", "long", "short", "regress", "exp", "attr"],
        help="Controls the legend, options are: \
                        'default' (default), 'none, 'short', long', 'regress'",
        default="default",
    )
    parser.add_argument(
        "--o_dir",
        dest="odir",
        help="output directory. Default: current directory",
        default="foo",
    )
    parser.add_argument(
        "--plot_title",
        dest="plot_title",
        action="store_true",
        help="Plots the flux gate name as title",
        default=False,
    )
    parser.add_argument(
        "--simple_plot",
        dest="simple_plot",
        action="store_true",
        help="Make simple line plot",
        default=False,
    )
    parser.add_argument(
        "--no_legend",
        dest="plot_legend",
        action="store_false",
        help="Don't plot a legend",
        default=True,
    )
    parser.add_argument(
        "-p",
        "--print_size",
        dest="print_mode",
        choices=[
            "onecol",
            "medium",
            "twocol",
            "height",
            "presentation",
            "small_font",
            "large_font",
            "50mm",
            "72mm",
        ],
        help="sets figure size and font size, available options are: \
                        'onecol','medium','twocol','presentation'",
        default="medium",
    )
    parser.add_argument(
        "-r",
        "--output_resolution",
        dest="out_res",
        help="""
                      Graphics resolution in dots per inch (DPI), default
                      = 300""",
        default=300,
    )
    parser.add_argument(
        "--y_lim", dest="y_lim", nargs=2, help="""Y lims""", default=[None, None]
    )
    parser.add_argument(
        "-v",
        "--variable",
        dest="varname",
        help="""Variable to plot, default = 'velsurf_mag'.""",
        default="velsurf_mag",
    )

    options = parser.parse_args()
    args = options.FILE

    np.seterr(all="warn")
    aspect_ratio = options.aspect_ratio
    tol = 1e-6
    normalize = options.normalize
    print_mode = options.print_mode
    obs_file = options.obs_file
    out_res = int(options.out_res)
    varname = options.varname
    table_file = options.table_file
    label_params = list(options.label_params.split(","))
    plot_title = options.plot_title
    legend = options.legend
    do_regress = options.do_regress
    make_figures = options.make_figures
    odir = options.odir
    simple_plot = options.simple_plot
    y_lim_min, y_lim_max = options.y_lim
    ice_density = 910.0
    ice_density_units = "910 kg m-3"
    vol_to_mass = False
    profile_axis_out_units = "km"
    pearson_r_threshold_high = 0.85
    pearson_r_threshold_low = 0.50

    if y_lim_min is not None:
        y_lim_min = np.float(y_lim_min)
    if y_lim_max is not None:
        y_lim_max = np.float(y_lim_max)

    if varname in ("velsurf_mag", "velbase_mag", "velsurf_normal", "v"):
        flux_type = "flux"
        v_o_units = "m yr-1"
        v_o_units_str = "m yr$^\mathregular{{-1}}$"
        v_o_units_str_tex = "m\,yr$^{-1}$"
        v_flux_o_units = "km2 yr-1"
        v_flux_o_units_str = "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$"
        v_flux_o_units_str_tex = "km$^2$\,yr$^{-1}$"
    elif varname in ("flux_mag", "flux_normal"):
        flux_type = "flux"
        v_o_units = "km2 yr-1"
        v_o_units_str = "km$^\mathregular{2}$ yr$^\mathregular{{-1}}$"
        v_o_units_str_tex = "km$^2$\,yr$^{-1}$"
        vol_to_mass = True
        v_flux_o_units = "Gt yr-1"
        v_flux_o_units_str = "Gt yr$^\mathregular{{-1}}$"
        v_flux_o_units_str_tex = "Gt\,yr$^{-1}$"
    elif varname in ("thk", "thickness", "land_ice_thickness"):
        flux_type = "area"
        v_o_units = "m"
        v_o_units_str = "m"
        v_o_units_str_tex = "m"
        vol_to_mass = False
        v_flux_o_units = "km2"
        v_flux_o_units_str = "km$^\mathregular{2}$"
        v_flux_o_units_str_tex = "km$^2$"
    elif varname in ("usurf", "surface", "surface_altitude"):
        flux_type = "area"
        v_o_units = "m"
        v_o_units_str = "m"
        v_o_units_str_tex = "m"
        vol_to_mass = False
        v_flux_o_units = "km2"
        v_flux_o_units_str = "km$^\mathregular{2}$"
        v_flux_o_units_str_tex = "km$^2$"
    else:
        print(("variable {} not supported".format(varname)))

    na = len(args)
    shade = 0.15
    colormap = options.colormap
    # FIXME: make option
    cstride = 2
    my_colors = plt.get_cmap(colormap).colors[::cstride]
    my_colors_light = []
    for rgb in my_colors:
        h, l, s = rgb_to_hls(*rgb[0:3])
        l *= 1 + shade
        l = min(0.9, l)
        l = max(0, l)
        my_colors_light.append(hls_to_rgb(h, l, s))

    alpha = 0.75
    dash_style = "o"
    numpoints = 1
    legend_frame_width = 0.25
    markeredgewidth = 0.2
    markeredgecolor = "k"
    obscolor = "0.4"

    params_dict = {
        "dataset": {"abbr": "Dataset", "format": "{}"},
        "dem": {"abbr": "DEM", "format": "{}"},
        "bed": {"abbr": "bed", "format": "{}"},
        "surface.pdd.factor_ice": {
            "abbr": "$f_{\mathregular{i}}$",
            "format": "{:1.0f}",
        },
        "surface.pdd.factor_snow": {
            "abbr": "$f_{\mathregular{s}}$",
            "format": "{:1.0f}",
        },
        "basal_resistance.pseudo_plastic.q": {"abbr": "$q$", "format": "{:1.2f}"},
        "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": {
            "abbr": "$\\delta$",
            "format": "{:1.4f}",
        },
        "stress_balance.sia.enhancement_factor": {
            "abbr": "$E_{\mathregular{SIA}}$",
            "format": "{:1.2f}",
        },
        "stress_balance.ssa.enhancement_factor": {
            "abbr": "$E_{\mathregular{SSA}}$",
            "format": "{:1.2f}",
        },
        "stress_balance.ssa.Glen_exponent": {
            "abbr": "$n_{\mathregular{SSA}}$",
            "format": "{:1.2f}",
        },
        "stress_balance.sia.Glen_exponent": {
            "abbr": "$n_{\mathregular{SIA}}$",
            "format": "{:1.2f}",
        },
        "grid_dx_meters": {"abbr": "ds", "format": "{:.0f}"},
        "flow_law.gpbld.water_frac_observed_limit": {
            "abbr": "$\omega$",
            "format": "{:1.2}",
        },
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": {
            "abbr": "$\phi_{\mathregular{min}}$",
            "format": "{:4.2f}",
        },
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": {
            "abbr": "$\phi_{\mathregular{max}}$",
            "format": "{:4.2f}",
        },
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": {
            "abbr": "$z_{\mathregular{min}}$",
            "format": "{:1.0f}",
        },
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": {
            "abbr": "$z_{\mathregular{max}}$",
            "format": "{:1.0f}",
        },
    }

    var_long = (
        "velsurf_mag",
        "velbase_mag",
        "velsurf_normal",
        "flux_mag",
        "flux_normal",
        "surface",
        "usurf",
        "surface_altitude" "thk",
        "thickness",
        "land_ice_thickness",
    )
    var_short = (
        "speed",
        "sliding speed",
        "speed",
        "flux",
        "flux",
        "altitude",
        "altitude",
        "altitude",
        "ice thickness",
        "ice thickness",
        "ice thickness",
    )
    var_name_dict = dict(list(zip(var_long, var_short)))

    flow_types = {0: "isbr{\\ae}", 1: "ice-stream", 2: "undefined"}
    glacier_types = {0: "ffmt", 1: "lvmt", 2: "ist", 3: "lt"}

    # Open first file
    filename = args[0]
    print(("  opening NetCDF file %s ..." % filename))
    try:
        nc0 = NC(filename, "r")
    except:
        print(
            (
                "ERROR:  file '%s' not found or not NetCDF format ... ending ..."
                % filename
            )
        )
        import sys

        sys.exit(1)

    # Get profiles from first file
    # All experiments have to contain the same profiles
    # Create flux gates
    profile_names = nc0.variables["profile_name"][:]
    flux_gates = []
    for pos_id, profile_name in enumerate(profile_names):
        profile_axis = nc0.variables["profile_axis"][pos_id]
        profile_axis_units = nc0.variables["profile_axis"].units
        profile_axis_name = nc0.variables["profile_axis"].long_name
        profile_id = int(nc0.variables["profile_id"][pos_id])
        try:
            clon = nc0.variables["clon"][pos_id]
        except:
            clon = 0.0
        try:
            clat = nc0.variables["clat"][pos_id]
        except:
            clat = 0.0
        try:
            flightline = nc0.variables["flightline"][pos_id]
        except:
            flightline = 0
        try:
            glaciertype = nc0.variables["glaciertype"][pos_id]
        except:
            glaciertype = ""
        try:
            flowtype = nc0.variables["flowtype"][pos_id]
        except:
            flowtype = ""
        flux_gate = FluxGate(
            pos_id,
            profile_name,
            profile_id,
            profile_axis,
            profile_axis_units,
            profile_axis_name,
            clon,
            clat,
            flightline,
            glaciertype,
            flowtype,
        )
        flux_gates.append(flux_gate)
    nc0.close()

    # If observations are provided, load observations
    if obs_file:
        obs = ObservationsDataset(obs_file, varname)
        for flux_gate in flux_gates:
            flux_gate.add_observations(obs)

    # Add experiments to flux gates
    for k, filename in enumerate(args):
        # id = re.search("id_(\b0*([1-9][0-9]*|0)\b)", filename).group(1)
        # pid = int(filename.split("id_")[1].split("_")[0])
        id = k
        experiment = ExperimentDataset(id, filename, varname)
        for flux_gate in flux_gates:
            flux_gate.add_experiment(experiment)

    # set the print mode
    lw, pad_inches = set_mode(print_mode, aspect_ratio=aspect_ratio)

    ne = len(flux_gates[0].experiments)
    ng = len(flux_gates)

    if table_file and obs_file:
        export_latex_table_flux(table_file, flux_gates, label_params)

    # make figure for each flux gate
    for gate in flux_gates:
        if make_figures:
            gate.make_line_plot(label_param_list=label_params)
        else:
            if not gate.has_fluxes:
                gate.calculate_fluxes()
            if gate.has_observations:
                gate.calculate_stats()

    if obs_file:
        # write rmsd and pearson r tables per gate
        for gate in flux_gates:
            gate_name = "_".join([unidecode(gate.gate_name), "rmsd", varname])
            outname = ".".join([gate_name, "tex"]).replace(" ", "_")
            # export_latex_table_rmsd(outname, gate)
            gate_name = "_".join([unidecode(gate.gate_name), "pearson_r", varname])
            outname = ".".join([gate_name, "csv"]).replace(" ", "_")
            ids = sorted(gate.p_ols, key=lambda x: gate.corr[x], reverse=True)
            corrs = [gate.corr[x] for x in ids]
            corrs_dict = dict(zip(ids, corrs))
            export_csv_from_dict(outname, corrs_dict, header="id,correlation")
            # outname = ".".join([gate_name, "tex"]).replace(" ", "_")
            # export_latex_table_corr(outname, gate)
        # write rmsd and person r figure per experiment
        for exp in flux_gates[0].experiments:
            exp_str = "_".join(["pearson_r_experiment", str(exp.id), varname])
            outname = ".".join([exp_str, "pdf"])
            corrs = make_correlation_figure(outname, exp)
            exp_str = "_".join(["coors_experiment", str(exp.id), varname])
            outname = ".".join([exp_str, "csv"])
            export_csv_from_dict(outname, corrs, header="id,correlation")
            exp_str = "_".join(["rmsd_experiment", str(exp.id), varname])
            outname = ".".join([exp_str, "tex"])
            export_gate_table_rmsd(outname, exp)

        experiments_df = []
        experiments_isbrae_df = []
        experiments_ice_stream_df = []
        experiments_undetermined_df = []
        glaciers_above_threshold = []
        lengths = [gate.length() for gate in flux_gates]
        for exp in range(ne):
            names = [gate.gate_name for gate in flux_gates]
            corrs = [gate.corr[exp] for gate in flux_gates]
            rmsds = [gate.rmsd[exp] for gate in flux_gates]
            Ns = [gate.N_rmsd[exp] for gate in flux_gates]
            gl_types = [int(gate.glaciertype) for gate in flux_gates]
            fl_types = [int(gate.flowtype) for gate in flux_gates]
            d = {
                "name": names,
                "correlation": corrs,
                "rmsd": rmsds,
                "N": Ns,
                "glacier_type": gl_types,
                "flow_type": fl_types,
                "length": lengths,
            }
            df = pa.DataFrame(d)
            # Select glaciers with correlation above threshold
            # df = df[df["correlation"] > pearson_r_threshold_high]
            exp_rmsd_cum = np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            experiments_df.append(df)
            # It would be nice to select for flow types later, with something like:
            #  [
            # np.sqrt(np.sum(df["rmsd"].values ** 2 * df["N"].values) * (1.0 / df["N"].values.sum()))
            # for df in experiments_df if df[df["flow_type"] == 0]
            # ]
            #
            experiments_isbrae_df.append(df[df["flow_type"] == 0])
            experiments_ice_stream_df.append(df[df["flow_type"] == 1])
            experiments_undetermined_df.append(df[df["flow_type"] == 2])

            no_glaciers_above_threshold = len(
                df[df["correlation"] > pearson_r_threshold_high]
            )

            print("Experiment {}".format(exp))
            print(
                "  Number of glaciers with r(all) > {}: {}".format(
                    pearson_r_threshold_high, no_glaciers_above_threshold
                )
            )
            glaciers_above_threshold.append(no_glaciers_above_threshold)
            print(
                "  RMS difference {:4.0f}".format(
                    np.sqrt(
                        np.sum(df["rmsd"].values ** 2 * df["N"].values)
                        * (1.0 / df["N"].values.sum())
                    )
                )
            )
            print("  median(pearson r(all): {:1.2f}".format(df["correlation"].median()))
            print(
                "  median(pearson r(isbrae): {:1.2f}".format(
                    df[df["flow_type"] == 0]["correlation"].median()
                )
            )
            print(
                "  median(pearson r(ice-stream): {:1.2f}".format(
                    df[df["flow_type"] == 1]["correlation"].median()
                )
            )
            print(
                "  median(pearson r(undetermined): {:1.2f}".format(
                    df[df["flow_type"] == 2]["correlation"].median()
                )
            )

        # Calculate cumulative values of rms differences of all glaciers together
        rmsd_cum_dict = {}
        rmsd_isbrae_cum_dict = {}
        rmsd_ice_stream_cum_dict = {}
        rmsd_undetermined_cum_dict = {}
        keys = range(ne)
        rmsd_cum = [
            np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            for df in experiments_df
        ]
        rmsd_cum_dict = dict(zip(keys, rmsd_cum))

        rmsd_isbrae_cum = [
            np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            for df in experiments_isbrae_df
        ]
        rmsd_isbrae_cum_dict = dict(zip(keys, rmsd_isbrae_cum))

        rmsd_ice_stream_cum = [
            np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            for df in experiments_isbrae_df
        ]
        rmsd_ice_stream_cum_dict = dict(zip(keys, rmsd_isbrae_cum))

        rmsd_undetermined_cum = [
            np.sqrt(
                np.sum(df["rmsd"].values ** 2 * df["N"].values)
                * (1.0 / df["N"].values.sum())
            )
            for df in experiments_isbrae_df
        ]
        rmsd_undetermined_cum_dict = dict(zip(keys, rmsd_isbrae_cum))
        rmsd_cum_dict_sorted = sorted(
            iter(rmsd_cum_dict.items()), key=operator.itemgetter(1)
        )

        outname = ".".join(["rmsd_sorted_{}".format(varname), "csv"])
        print(("  - saving {0}".format(outname)))
        export_csv_from_dict(outname, dict(rmsd_cum_dict_sorted), header="id,rmsd")

        corr = [df["correlation"].median() for df in experiments_df]
        corr_dict = dict(zip(keys, corr))
        corr_dict_sorted = sorted(
            iter(corr_dict.items()), key=operator.itemgetter(1), reverse=True
        )

        outname = ".".join(["pearson_r_sorted_{}".format(varname), "csv"])
        print(("  - saving {0}".format(outname)))
        export_csv_from_dict(outname, dict(corr_dict_sorted), header="id,correlation")

        glaciers_dict = dict(zip(keys, glaciers_above_threshold))
        glaciers_dict_sorted = sorted(
            iter(glaciers_dict.items()), key=operator.itemgetter(1), reverse=True
        )

        outname = ".".join(["glaciers_above_threshold_{}".format(varname), "csv"])
        print(("  - saving {0}".format(outname)))
        export_csv_from_dict(
            outname,
            dict(glaciers_dict_sorted),
            header="id,no_glaciers",
            fmt=["%i", "%i"],
        )

    gate = flux_gates[0]
    # make a global regression figure
    if do_regress:
        make_regression()

    # Write results to shape file
    outname = "statistics.shp"
    write_shapefile(outname, flux_gates)

    # Write a table with all experiments
    outname = ".".join(["experiment_table", "tex"])
    write_experiment_table(outname)
