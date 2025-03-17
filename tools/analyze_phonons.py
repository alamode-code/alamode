#!/usr/bin/env python
#
# analyze_phonons.py
#
# Simple interface to the command line script "analyze_phonons.cpp".
#
# Copyright (c) 2014, 2015, 2016 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
"""
This python script is a simple interface to
the C++ analyzer program "analyze_phonons.cpp".
To execute this script, the above c++ program has to be
compiled and made executable beforehand.
"""

import optparse
import os
import subprocess
import sys

_RETURN_CMD_DEFAULT = False


def set_average(options):
    if options.average_gamma:
        avg = "1"
    else:
        avg = "0"

    if options.isotope is None:
        isotope = "0"
        file_isotope = "none"
    else:
        isotope = "1"
        file_isotope = options.isotope
    return avg, isotope, file_isotope


def print_temperature_dep_lifetime(
    analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT
):
    """print_temperature_dep_lifetime

    Args:
        analyze_obj (str): "analyze_phonon".
        calc (str): calculation type.
        file_result (str): result of RTA.
        options (_type_): analyze_phonon options
        return_cmd (_type_, optional): execute commnd or not. Defaults to _RETURN_CMD_DEFAULT.

    Returns:
        list: command list if return_cmd=True.
    """
    avg, isotope, file_isotope = set_average(options)

    if options.kpoint is None or options.mode is None:
        sys.exit(
            "Please specify the temperature by --temp option,"
            "or specify both --kpoint and --mode when --calc=tau"
        )
    else:
        if len(options.kpoint.split(":")) != 1:
            sys.exit("Invalid usage of --kpoint for --calc=tau")
        if len(options.mode.split(":")) != 1:
            sys.exit("Invalid usage of --mode for --calc=tau")

        target_k = int(options.kpoint)
        target_s = int(options.mode)
        calc = "tau_temp"
        if return_cmd:
            command = [
                file_result,
                calc,
                avg,
                str(target_k),
                str(target_s),
                isotope,
                file_isotope,
            ]
            return command

        else:
            command = (
                analyze_obj
                + file_result
                + " "
                + calc
                + " "
                + avg
                + " "
                + str(target_k)
                + " "
                + str(target_s)
                + " "
                + isotope
                + " "
                + file_isotope
            )

            subprocess.call(command, shell=True)


def print_lifetime_at_given_temperature(
    analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT
):
    """print_lifetime_at_given_temperature

    Args:
        analyze_obj (str): "analyze_phonon".
        calc (str): calculation type.
        file_result (str): result of RTA.
        options (dict): options of analyze_phonon.
        return_cmd (_type_, optional): execute command or not. Defaults to _RETURN_CMD_DEFAULT.

    Returns:
        list: command list if return_cmd=True.
    """
    avg, isotope, file_isotope = set_average(options)

    if options.kpoint is None:
        beg_k = 1
        end_k = 0
    else:
        if len(options.kpoint.split(":")) == 1:
            beg_k = int(options.kpoint)
            end_k = beg_k
        elif len(options.kpoint.split(":")) == 2:
            arr = options.kpoint.split(":")
            beg_k, end_k = int(arr[0]), int(arr[1])
        else:
            sys.exit("Invalid usage of --kpoint for --calc=tau")

    if options.mode is None:
        beg_s = 1
        end_s = 0
    else:
        if len(options.mode.split(":")) == 1:
            beg_s = int(options.mode)
            end_s = beg_s
        elif len(options.mode.split(":")) == 2:
            arr = options.mode.split(":")
            beg_s, end_s = int(arr[0]), int(arr[1])
        else:
            sys.exit("Invalid usage of --mode for --calc=tau")
    if return_cmd:
        command = [
            file_result,
            calc,
            avg,
            str(beg_k),
            str(end_k),
            str(beg_s),
            str(end_s),
            options.temp,
            isotope,
            file_isotope,
        ]
        return command

    else:
        command = (
            analyze_obj
            + file_result
            + " "
            + calc
            + " "
            + avg
            + " "
            + str(beg_k)
            + " "
            + str(end_k)
            + " "
            + str(beg_s)
            + " "
            + str(end_s)
            + " "
            + options.temp
            + " "
            + isotope
            + " "
            + file_isotope
        )

        subprocess.call(command, shell=True)


def print_thermal_conductivity(
    analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT
):
    """def print_thermal_conductivity(analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT):
              + str(beg_s) + " " + str(end_s) + " " + options.temp \
              + " " + isotope + " " + file_isotope

    Args:
        analyze_obj (str): "analyze_phonon".
        calc (_type_): calculation type.
        file_result (_type_): result of RTA.
        options (_type_): options of analyze_phonon.
        return_cmd (_type_, optional): execute command or not. Defaults to _RETURN_CMD_DEFAULT.

    Returns:
        list: command list if return_cmd=True.
    """
    avg, isotope, file_isotope = set_average(options)

    if options.kpoint is not None:
        print("# Warning: --kpoint option is discarded")

    if options.mode is None:
        beg_s = 1
        end_s = 0
    else:
        if len(options.mode.split(":")) == 1:
            beg_s = int(options.mode)
            end_s = beg_s
        elif len(options.mode.split(":")) == 2:
            arr = options.mode.split(":")
            beg_s, end_s = int(arr[0]), int(arr[1])
        else:
            sys.exit("Invalid usage of --mode for --calc=kappa")

    if return_cmd:
        command = [
            file_result,
            calc,
            avg,
            str(beg_s),
            str(end_s),
            isotope,
            file_isotope,
        ]
        return command
    else:
        command = (
            analyze_obj
            + file_result
            + " "
            + calc
            + " "
            + avg
            + " "
            + str(beg_s)
            + " "
            + str(end_s)
            + " "
            + isotope
            + " "
            + file_isotope
        )

        subprocess.call(command, shell=True)


def print_thermal_conductivity_with_boundary(
    analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT
):
    """def print_thermal_conductivity_with_boundary(analyze_obj, calc, file_result, options, return_cmd=_RETURN_CMD_DEFAULT):


    Args:
        analyze_obj (str): "analyze_phonon".
        calc (str): calculation type.
        file_result (str): result of RTA.
        options (dict): options of analyze_phonon
        return_cmd (_type_, optional): _description_. Defaults to _RETURN_CMD_DEFAULT.

    Returns:
        list: command list if return_cmd=True.
    """
    avg, isotope, file_isotope = set_average(options)

    if options.kpoint is not None:
        print("# Warning: --kpoint option is discarded")

    if options.mode is None:
        beg_s = 1
        end_s = 0
    else:
        if len(options.mode.split(":")) == 1:
            beg_s = int(options.mode)
            end_s = beg_s
        elif len(options.mode.split(":")) == 2:
            arr = options.mode.split(":")
            beg_s, end_s = int(arr[0]), int(arr[1])
        else:
            sys.exit("Invalid usage of --mode for --calc=kappa_boundary")

    if options.size is None:
        boundary_size = 1000.0
    else:
        boundary_size = float(options.size)

    if return_cmd:
        command = [
            file_result,
            calc,
            avg,
            str(beg_s),
            str(end_s),
            isotope,
            file_isotope,
            str(boundary_size),
        ]
        return command
    else:
        command = (
            analyze_obj
            + file_result
            + " "
            + calc
            + " "
            + avg
            + " "
            + str(beg_s)
            + " "
            + str(end_s)
            + " "
            + isotope
            + " "
            + file_isotope
            + " "
            + str(boundary_size)
        )

        subprocess.call(command, shell=True)


def print_cumulative_thermal_conductivity(
    analyze_obj, cumulative_mode, file_result, options, return_cmd=_RETURN_CMD_DEFAULT
):
    """print_cumulative_thermal_conductivity

    Args:
        analyze_obj (str): "analyze_phonon"
        cumulative_mode (str): cumulative mode.
        file_result (str): result of RTA.
        options (dict): options of analyze_phonon.
        return_cmd (bool, optional): execute command or not. Defaults to _RETURN_CMD_DEFAULT.

    Returns:
        list: list of command if return_cmd=True.
    """

    calc = cumulative_mode
    avg, isotope, file_isotope = set_average(options)

    if options.temp is None:
        sys.exit("--temp is necessary when --calc=%s" % cumulative_mode)

    if options.kpoint is not None:
        print("# Warning: --kpoint option is discarded")

    if options.mode is None:
        beg_s = 1
        end_s = 0
    else:
        if len(options.mode.split(":")) == 1:
            beg_s = int(options.mode)
            end_s = beg_s
        elif len(options.mode.split(":")) == 2:
            arr = options.mode.split(":")
            beg_s, end_s = int(arr[0]), int(arr[1])
        else:
            sys.exit("Invalid usage of --mode for --calc=%s" % cumulative_mode)

    if options.length is None:
        max_len = 0.0
        d_len = 0.0
    else:
        if len(options.length.split(":")) == 2:
            arr = options.length.split(":")
            max_len, d_len = float(arr[0]), float(arr[1])
        else:
            sys.exit("Invalid usage of --length option")

    if cumulative_mode == "cumulative":
        if return_cmd:
            command = [
                file_result,
                calc,
                avg,
                str(beg_s),
                str(end_s),
                isotope,
                file_isotope,
                str(max_len),
                str(d_len),
                options.temp,
                options.nsample,
                options.gridtype,
            ]

        else:
            command = (
                analyze_obj
                + file_result
                + " "
                + calc
                + " "
                + avg
                + " "
                + str(beg_s)
                + " "
                + str(end_s)
                + " "
                + isotope
                + " "
                + file_isotope
                + " "
                + str(max_len)
                + " "
                + str(d_len)
                + " "
                + options.temp
                + " "
                + str(options.nsample)
                + " "
                + options.gridtype
            )

    else:
        size_flag = [0] * 3

        if options.direction is None:
            for i in range(3):
                size_flag[i] = 0
        else:
            if len(options.direction.split(":")) > 3:
                sys.exit("Invalid usage of --direction")

            arr = options.direction.split(":")
            for i in range(len(options.direction.split(":"))):
                size_flag[int(arr[i]) - 1] = 1

        if return_cmd:
            command = [
                file_result,
                calc,
                avg,
                str(beg_s),
                str(end_s),
                isotope,
                file_isotope,
                str(max_len),
                str(d_len),
                options.temp,
                str(size_flag[0]),
                str(size_flag[1]),
                str(size_flag[2]),
                options.nsample,
                options.gridtype,
            ]
        else:
            command = (
                analyze_obj
                + file_result
                + " "
                + calc
                + " "
                + avg
                + " "
                + str(beg_s)
                + " "
                + str(end_s)
                + " "
                + isotope
                + " "
                + file_isotope
                + " "
                + str(max_len)
                + " "
                + str(d_len)
                + " "
                + options.temp
                + " "
                + str(size_flag[0])
                + " "
                + str(size_flag[1])
                + " "
                + str(size_flag[2])
                + " "
                + str(options.nsample)
                + " "
                + options.gridtype
            )

    if return_cmd:
        return command
    else:
        subprocess.call(command, shell=True)


if __name__ == "__main__":

    def main():
        """make another main() to check whether global variables aren't used."""
        parser = optparse.OptionParser()
        parser.add_option("--temp", help="target temperature to analyze")
        parser.add_option("--mode", help="specify phonon mode index to print")
        parser.add_option("--kpoint", help="specify k-point index to print")

        parser.add_option(
            "--calc",
            metavar="tau|kappa|cumulative|cumulative2|kappa_boundary",
            help=(
                "specify what to print. Available options are "
                "tau (Lifetime, mean-free-path, etc.), "
                "kappa (Thermal conductivity), "
                "cumulative (Cumulative thermal conductivity), "
                "cumulative2 (Cumulative thermal conductivity with specific xyz-directions), "
                "and kappa_boundary (Thermal conductivity with boundary effect). "
                "When --calc=cumulative2, please specify the "
                "--direction option. When --calc=kappa_boundary, "
                "please specify the --size option."
            ),
        )

        parser.add_option(
            "--isotope",
            metavar="PREFIX.self_isotope",
            help="specify the file PREFIX.self_isotope to include the effect of "
            "phonon-isotope scatterings. When given, the phonon scattering rates will be"
            "updated as 1/tau_{new} = 1/tau_{phonon-phonon} + 1/tau_{phonon-isotope}."
            "The PREFIX.self_isotope can be generated using 'anphon' with ISOTOPE=2 option.",
        )

        parser.add_option(
            "--noavg",
            action="store_false",
            dest="average_gamma",
            default=True,
            help="do not average the damping functionat degenerate points",
        )

        parser.add_option(
            "--size",
            help="specify the grain boundary size in units of nm"
            "The default value is 1000 nm.",
        )

        parser.add_option(
            "--length",
            metavar="Lmax:dL",
            help="specify the maximum value of system size L and its"
            "step dL in units of nm."
            "The default value is --length=1000:10 .",
        )

        parser.add_option(
            "--nsample",
            metavar="100",
            default=1000,
            help="specify the number of sampling points used for calculating cumulative kappa.",
        )

        parser.add_option(
            "--gridtype",
            metavar="linear | log",
            default="log",
            help="specify whether the uniform grid of L (length) for cumulative kappa is "
            "generated in linear scale or logarithmic scale.",
        )

        group = optparse.OptionGroup(
            parser,
            "The following options are available/necessary when --calc=cumulative2",
        )

        group.add_option(
            "--direction",
            metavar="1|2|3",
            help="specify which direction (xyz) to consider the size effect. "
            "When --direction=1 (2, 3), phonon mean-free-paths (ell) along x (y, z) are "
            "compared with the system size L. Then, the cumulative thermal conductivity is "
            "calculated by considering phonon modes satisfying ell <= L.",
        )

        parser.add_option_group(group)

        options, args = parser.parse_args()
        if len(args) == 0:
            parser.print_help()
            sys.exit(10)

        file_result = args[0]

        dir_obj = os.path.dirname(__file__)
        analyze_obj = dir_obj + "/analyze_phonons "

        calc = options.calc

        # Compute phonon lifetimes, mean-free-path,
        # and mode-decomposed thermal conductivity
        if calc == "tau":
            # When --temp option is not specified, print temperature dependence
            # of phonon lifetimes at given mode and k point.
            if options.temp is None:
                print_temperature_dep_lifetime(analyze_obj, calc, file_result, options)

            # When --temp option is specified, print phonon lifetimes and
            # other quantities at the specified temperature.
            else:
                print_lifetime_at_given_temperature(
                    analyze_obj, calc, file_result, options
                )

        elif calc == "kappa":
            print_thermal_conductivity(analyze_obj, calc, file_result, options)

        # Compute cumulative thermal conductivity.
        elif calc == "cumulative" or calc == "cumulative2":
            print_cumulative_thermal_conductivity(
                analyze_obj, calc, file_result, options
            )

        elif calc == "kappa_boundary":
            print_thermal_conductivity_with_boundary(
                analyze_obj, calc, file_result, options
            )

        else:
            sys.exit("Invalid --calc option given")

    main()
