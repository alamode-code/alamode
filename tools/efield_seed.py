#!/usr/bin/env python
"""efield_seed.py: seed the next run of an anphon EFIELD sweep.

    efield_seed.py PREFIX [--temp T]

Reads the relaxed structure of a finished SCPH/QHA relaxation from
PREFIX.atom_disp (Cartesian u0 [Bohr] per atom) and, when present,
PREFIX.umn_tensor (displacement gradient u_{mu nu}), at temperature T
(default: the last row), and prints the &displace (DISPMODE = 1) and &strain
blocks to paste into the input of the next field value.

A field sweep is NOT a restart: RESTART_SCPH / RESTART_QHA only reload the
converged results of the same EFIELD (a different field is refused). Each new
field is a new optimization, started from the previous relaxed structure.
"""

import argparse
import os

import numpy as np


def pick_row(filename, temp):
    data = np.atleast_2d(np.loadtxt(filename))
    if temp is None:
        return data[-1]
    rows = data[np.abs(data[:, 0] - temp) < 1.0e-6]
    if len(rows) == 0:
        raise SystemExit("T = %g K not found in %s" % (temp, filename))
    return rows[-1]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("prefix", help="PREFIX of the previous run")
    p.add_argument("--temp", type=float, help="temperature [K] (default: last row)")
    args = p.parse_args()

    u0 = pick_row(args.prefix + ".atom_disp", args.temp)
    print("# seeded from %s at T = %g K" % (args.prefix, u0[0]))
    print("&displace\n1")
    for u in u0[1:].reshape(-1, 3):
        print("  %20.12e %20.12e %20.12e" % tuple(u))
    print("/")

    print("&strain")
    if os.path.exists(args.prefix + ".umn_tensor"):
        umn = pick_row(args.prefix + ".umn_tensor", args.temp)[1:].reshape(3, 3)
    else:
        umn = np.zeros((3, 3))  # RELAX_STR = 1: the cell is not relaxed
    for row in umn:
        print("  %20.12e %20.12e %20.12e" % tuple(row))
    print("/")


if __name__ == "__main__":
    main()
