#!/usr/bin/env python
#
# plotband.py
#
# Simple script to visualize phonon dispersion relations
#
# Copyright (c) 2014 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

import optparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# parser options


# font styles
mpl.rc("font", **{"family": "Times New Roman", "sans-serif": ["Helvetica"]})
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=16)
mpl.rc("axes", labelsize=16)
mpl.rc("lines", linewidth=1.5)
mpl.rc("legend", fontsize="small")
# line colors and styles
color = ["b", "g", "r", "m", "k", "c", "y", "r"]
lsty = ["-", "-", "-", "-", "--", "--", "--", "--"]


def get_kpath_and_kval(file_in):
    ftmp = open(file_in, "r")
    kpath = ftmp.readline().rstrip("\n").split()
    kval = ftmp.readline().rstrip("\n").split()
    ftmp.close()

    if kpath[0] == "#" and kval[0] == "#":
        kval_float = [float(val) for val in kval[1:]]
        kpath_list = []
        for i in range(len(kpath[1:])):
            if kpath[i + 1] == "G" or kpath[i + 1] == "GAMMA":
                kpath_list.append(r"$\Gamma$")
            else:
                kpath_list.append(r"$\mathrm{%s}$" % kpath[i + 1])

        return kpath_list, kval_float
    else:
        return [], []


def change_scale(array, str_scale):
    str_tmp = str_scale.lower()

    if str_tmp == "kayser":
        print("Band structure will be shown in units of cm^{-1}")
        return array

    elif str_tmp == "mev":
        print("Band structure will be shown in units of meV")
        kayser_to_mev = 0.0299792458 * 1.0e12 * 6.62606896e-34 / 1.602176565e-19 * 1000

        for i in range(len(array)):
            for j in range(len(array[i])):
                for k in range(1, len(array[i][j])):
                    array[i][j][k] *= kayser_to_mev

        return array

    elif str_tmp == "thz":
        print("Band structure will be shown in units of THz")
        kayser_to_thz = 0.0299792458

        for i in range(len(array)):
            for j in range(len(array[i])):
                for k in range(1, len(array[i][j])):
                    array[i][j][k] *= kayser_to_thz

        return array

    else:
        print("Unrecognizable option for --unit %s" % str_scale)
        print("Band structure will be shown in units of cm^{-1}")
        return array


def normalize_to_unity(array, array_axis):
    for i in range(len(array)):
        max_val = array[i][-1][0]

        factor_normalize = 1.0 / max_val

        for j in range(len(array[i])):
            array[i][j][0] *= factor_normalize

    max_val = array_axis[-1]
    factor_normalize = 1.0 / max_val

    for i in range(len(array_axis)):
        array_axis[i] *= factor_normalize

    return array, array_axis


def get_xy_minmax(array):
    xmin, xmax, ymin, ymax = [0, 0, 0, 0]

    for i in range(len(array)):
        xtmp = array[i][-1][0]
        xmax = max(xmax, xtmp)

    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(1, len(array[i][j])):
                ytmp = array[i][j][k]
                ymin = min(ymin, ytmp)
                ymax = max(ymax, ytmp)

    return xmin, xmax, ymin, ymax


def gridspec_setup(data_merged, xtickslabels, xticksvars):
    xmaxs = []
    xmins = []

    xticks_grids = []
    xticklabels_grids = []
    xticklabels_tmp = []
    xticks_tmp = []

    for i in range(len(xtickslabels)):
        if i == 0:
            xmins.append(xticksvars[0])
        else:
            if xticksvars[i] == xticksvars[i - 1]:
                xmaxs.append(xticksvars[i - 1])
                xmins.append(xticksvars[i])

                xticks_grids.append(xticks_tmp)
                xticklabels_grids.append(xticklabels_tmp)
                xticklabels_tmp = []
                xticks_tmp = []

        xticklabels_tmp.append(xtickslabels[i])
        xticks_tmp.append(xticksvars[i])

    xticks_grids.append(xticks_tmp)
    xticklabels_grids.append(xticklabels_tmp)
    xmaxs.append(xticksvars[-1])

    naxes = len(xticks_grids)
    nfiles = len(data_merged)

    data_all_axes = []

    for i in range(naxes):
        data_ax = []

        xmin_ax = xmins[i]
        xmax_ax = xmaxs[i]

        for j in range(nfiles):
            kval = np.array(data_merged[j][0:, 0])
            ix_xmin_arr = np.where(kval <= xmin_ax)
            ix_xmax_arr = np.where(kval >= xmax_ax)

            if len(ix_xmin_arr[0]) > 0:
                ix_xmin = int(ix_xmin_arr[0][-1])
            else:
                ix_xmin = 0

            if len(ix_xmax_arr[0]) > 0:
                ix_xmax = int(ix_xmax_arr[0][0])
            else:
                ix_xmax = -2

            data_ax.append(data_merged[j][ix_xmin : (ix_xmax + 1), :])

        data_all_axes.append(data_ax)

    return naxes, xticks_grids, xticklabels_grids, xmins, xmaxs, data_all_axes


def preprocess_data(files, unitname, normalize_xaxis, emin=None, emax=None):
    """_summary_"""

    xtickslabels, xticksvars = get_kpath_and_kval(files[0])

    data_merged = []

    for file in files:
        data_tmp = np.loadtxt(file, dtype=float)
        data_merged.append(data_tmp)

    data_merged = change_scale(data_merged, unitname)

    if normalize_xaxis:
        data_merged, xticksvars = normalize_to_unity(data_merged, xticksvars)

    xmin, xmax, ymin, ymax = get_xy_minmax(data_merged)

    if emin is None and emax is None:
        factor = 1.05
        ymin *= factor
        ymax *= factor
    else:
        if emin is not None:
            ymin = emin
        if emax is not None:
            ymax = emax
        if ymin > ymax:
            print("Warning: emin > emax")

    naxes, xticks_grids, xticklabels_grids, xmin, xmax, data_merged_grids = (
        gridspec_setup(data_merged, xtickslabels, xticksvars)
    )

    return (
        naxes,
        xticks_grids,
        xticklabels_grids,
        xmin,
        xmax,
        ymin,
        ymax,
        data_merged_grids,
    )


def run_plot(
    files,
    nax,
    xticks_ax,
    xticklabels_ax,
    xmin_ax,
    xmax_ax,
    ymin,
    ymax,
    data_merged_ax,
    unitname,
    print_key,
    show=True,
):
    # fig = plt.figure()
    width_ratios = []
    for xmin, xmax in zip(xmin_ax, xmax_ax):
        width_ratios.append(xmax - xmin)

    gs = GridSpec(nrows=1, ncols=nax, width_ratios=width_ratios)
    gs.update(wspace=0.1)

    for iax in range(nax):
        ax = plt.subplot(gs[iax])

        for i in range(len(data_merged_ax[iax])):
            if len(data_merged_ax[iax][i]) > 0:
                ax.plot(
                    data_merged_ax[iax][i][0:, 0],
                    data_merged_ax[iax][i][0:, 1],
                    linestyle=lsty[i],
                    color=color[i],
                    label=files[i],
                )

                for j in range(2, len(data_merged_ax[iax][i][0][0:])):
                    ax.plot(
                        data_merged_ax[iax][i][0:, 0],
                        data_merged_ax[iax][i][0:, j],
                        linestyle=lsty[i],
                        color=color[i],
                    )

        if iax == 0:
            if unitname.lower() == "mev":
                ax.set_ylabel("Frequency (meV)", labelpad=20)
            elif unitname.lower() == "thz":
                ax.set_ylabel("Frequency (THz)", labelpad=20)
            else:
                ax.set_ylabel("Frequency (cm${}^{-1}$)", labelpad=10)

        else:
            ax.set_yticklabels([])
            ax.set_yticks([])

        plt.axis([xmin_ax[iax], xmax_ax[iax], ymin, ymax])
        ax.set_xticks(xticks_ax[iax])
        ax.set_xticklabels(xticklabels_ax[iax])
        ax.xaxis.grid(True, linestyle="-")

        if print_key and iax == 0:
            ax.legend(loc="best", prop={"size": 10})

    if show:
        plt.show()


if __name__ == "__main__":
    """
    Simple script for visualizing phonon dispersion relations.
    Usage:
    $ python plot_band.py [options] file1.bands file2.bands ...

    For details of available options, please type
    $ python plot_band.py -h
    """

    usage = "usage: %prog [options] file1.bands file2.bands ... "
    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "--nokey",
        action="store_false",
        dest="print_key",
        default=True,
        help="don't print the key in the figure",
    )
    parser.add_option(
        "-u",
        "--unit",
        action="store",
        type="string",
        dest="unitname",
        default="kayser",
        help="print the band dispersion in units of UNIT. "
        "Available options are kayser, meV, and THz",
        metavar="UNIT",
    )
    parser.add_option(
        "--emin",
        action="store",
        type="float",
        dest="emin",
        help="minimum value of the energy axis",
    )
    parser.add_option(
        "--emax",
        action="store",
        type="float",
        dest="emax",
        help="maximum value of the energy axis",
    )
    parser.add_option(
        "--normalize",
        action="store_true",
        dest="normalize_xaxis",
        default=False,
        help="normalize the x axis to unity.",
    )

    options, args = parser.parse_args()
    files = args[0:]
    nfiles = len(files)

    if nfiles == 0:
        print("Usage: plotband.py [options] file1.bands file2.bands ...")
        print("For details of available options, please type\n$ python plotband.py -h")
        exit(1)
    else:
        print("Number of files = %d" % nfiles)

    nax, xticks_ax, xticklabels_ax, xmin_ax, xmax_ax, ymin, ymax, data_merged_ax = (
        preprocess_data(
            files, options.unitname, options.normalize_xaxis, options.emin, options.emax
        )
    )

    run_plot(
        files,
        nax,
        xticks_ax,
        xticklabels_ax,
        xmin_ax,
        xmax_ax,
        ymin,
        ymax,
        data_merged_ax,
        options.unitname,
        options.print_key,
    )
