#!/usr/bin/env python
#
# analyzer.py
#
# Copyright (c) 2024  Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
"""
This python script is a post-processing tool for anphon.
"""

import optparse

from analyzer.calculator import Calculator


def get_optparse_options():
    parser = optparse.OptionParser()

    parser.add_option("--temp", help="target temperature to analyze", type=float)
    parser.add_option("--mode", help="specify phonon mode index to print", type=int)
    parser.add_option("--kpoint", help="specify k-point index to print", type=int)

    parser.add_option(
        "--calc",
        metavar="tau|kappa|cumulative|cumulative2|kappa_boundary",
        help=(
            "specify what to print. Available options are "
            "gamma (Phonon linewidth),"
            "tau (Lifetime, mean-free-path, etc.), "
            "kappa (Thermal conductivity), "
            "cumulative (Cumulative thermal conductivity), and "
            "cumulative2 (Cumulative thermal conductivity with specific xyz-directions). "
            "When --calc=cumulative2, please specify the "
            "--direction option."
        ),
    )

    parser.add_option(
        "--iso",
        metavar="PREFIX.self_isotope",
        help="specify the file PREFIX.self_isotope to include the effect of "
        "phonon-isotope scatterings. When given, the phonon scattering rates will be"
        "updated as 1/tau_{new} = 1/tau_{phonon-phonon} + 1/tau_{phonon-isotope}."
        "The PREFIX.self_isotope can be generated using 'anphon' with ISOTOPE=2 option.",
    )

    parser.add_option(
        "--3ph",
        metavar="PREFIX.result",
        dest="file_3ph",
        help="specify the file PREFIX.result to analyze",
    )

    parser.add_option(
        "--4ph",
        metavar="PREFIX.4ph.result",
        dest="file_4ph",
        help="specify the file PREFIX.4ph.result to include the effect of "
        "four-phonon scattering.",
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
        type=float,
        help="specify the grain boundary size in units of nm"
        "The default value is 1000 nm.",
    )

    parser.add_option(
        "--nsample",
        metavar="1000",
        default=1000,
        type=int,
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
        parser, "The following options are available/necessary when --calc=cumulative2"
    )

    group.add_option(
        "--direction",
        metavar="1|2|3",
        help="specify which direction (xyz) to consider the size effect. "
        "When --direction=1 (2, 3), phonon mean-free-paths (ell) along x (y, z) are "
        "compared with the system size L. Then, the cumulative thermal conductivity is "
        "calculated by considering phonon modes satisfying ell <= L. "
        "Multiple directions can be given as a colon-separated list, e.g. "
        "--direction=1:2.",
    )

    parser.add_option_group(group)
    options, args = parser.parse_args()

    return options


def main():
    options = get_optparse_options()

    calc = options.calc

    postproc = Calculator(
        options.file_3ph,
        file_result_4ph=options.file_4ph,
        file_isotope=options.iso,
        average_gamma=options.average_gamma,
    )

    if calc == "gamma":
        if options.temp is not None:
            postproc.print_linewidth(
                options.temp,
                four_phonon=(options.file_4ph is not None),
                isotope=(options.iso is not None),
            )
        else:
            # When --temp option is not specified, print temperature dependence
            # of phonon lifetimes at given mode and k point.
            if options.kpoint is None or options.mode is None:
                raise RuntimeError(
                    "Please specify the temperature by --temp option,"
                    "or specify both --kpoint and --mode when --calc=tau"
                )

            postproc.print_linewidth_mode(
                options.kpoint - 1,
                options.mode - 1,
                four_phonon=(options.file_4ph is not None),
                isotope=(options.iso is not None),
            )

    elif calc == "tau":
        if options.temp is not None:
            postproc.print_lifetime(
                options.temp,
                four_phonon=(options.file_4ph is not None),
                isotope=(options.iso is not None),
            )
        else:
            # When --temp option is not specified, print temperature dependence
            # of phonon lifetimes at given mode and k point.
            if options.kpoint is None or options.mode is None:
                raise RuntimeError(
                    "Please specify the temperature by --temp option,"
                    "or specify both --kpoint and --mode when --calc=tau"
                )

            postproc.print_lifetime_mode(
                options.kpoint - 1,
                options.mode - 1,
                four_phonon=(options.file_4ph is not None),
                isotope=(options.iso is not None),
            )

    elif calc == "kappa":
        postproc.print_thermal_conductivity(
            four_phonon=(options.file_4ph is not None),
            isotope=(options.iso is not None),
            len_boundary=options.size,
        )

    elif calc == "cumulative":
        if options.temp is None:
            raise RuntimeError("Please specify the temperature by --temp option")
        postproc.print_cumulative_kappa(
            options.temp,
            four_phonon=(options.file_4ph is not None),
            isotope=(options.iso is not None),
            nsamples=options.nsample,
            gridtype=options.gridtype,
        )

    elif calc == "cumulative2":
        if options.temp is None:
            raise RuntimeError("Please specify the temperature by --temp option")
        if options.direction is None:
            raise RuntimeError(
                "Please specify the --direction option when --calc=cumulative2"
            )

        entries = options.direction.split(":")
        if not all(t in ("1", "2", "3") for t in entries):
            raise RuntimeError(
                "Invalid --direction option. Please give 1, 2, or 3, "
                "or a colon-separated list of them (e.g. --direction=1:2)"
            )
        directions = sorted(set(int(t) - 1 for t in entries))

        postproc.print_cumulative_kappa(
            options.temp,
            four_phonon=(options.file_4ph is not None),
            isotope=(options.iso is not None),
            nsamples=options.nsample,
            gridtype=options.gridtype,
            directions=directions,
        )

    elif calc is None:
        print("Please specify the option --calc")

    else:
        raise RuntimeError("Invalid option --calc")


if __name__ == "__main__":
    main()
