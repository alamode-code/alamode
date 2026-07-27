#!/usr/bin/env python3
#
# scph_to_qefc.py
#
# Simple script to add anharmonic correction to a QE force constant.
#
# Usage:
# $ scph_to_qefc.py original_QE.fc PREFIX.scph_fc2_correction target_temperature > new_QE.fc
#
# Copyright (c) 2019 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

"""
Generator of new QE fc file containing anharmonic correction
"""

import sys

import numpy as np


def parse_QEfc(file_QEfc):
    header = []
    nx = None
    ny = None
    nz = None
    fc2 = None
    nkd = None
    nat = None

    with open(file_QEfc, "r") as f:
        line = f.readline()
        header.append(line)
        nkd, nat, ibrav = [int(entry) for entry in line.strip().split()[:3]]

        nskip = 0
        if ibrav == 0:
            nskip = nat + nkd + 3
        else:
            nskip = nat + nkd

        for _ in range(nskip):
            header.append(f.readline())

        line = f.readline()
        header.append(line)
        epsil_flag = line.strip()

        if epsil_flag == "T":
            for _ in range(3 + 4 * nat):
                header.append(f.readline())

        nx, ny, nz = [int(i) for i in f.readline().split()]
        fc2 = np.zeros((3 * nat, 3 * nat, nx, ny, nz))

        for icrd in range(3):
            for jcrd in range(3):
                for iat in range(nat):
                    for jat in range(nat):
                        f.readline()

                        for m3 in range(nz):
                            for m2 in range(ny):
                                for m1 in range(nx):
                                    fc2[3 * iat + icrd][3 * jat + jcrd][m1][m2][m3] = (
                                        float(f.readline().split()[3])
                                    )

    return header, nat, nkd, nx, ny, nz, fc2


def get_structure_info_dfc2(file_dfc2):
    lavec = []
    nat = None
    nkd = None

    with open(file_dfc2, "r") as f:
        lavec.append([float(t) for t in f.readline().strip().split()])
        lavec.append([float(t) for t in f.readline().strip().split()])
        lavec.append([float(t) for t in f.readline().strip().split()])
        line = f.readline().strip().split()
        nat = int(line[0])
        nkd = int(line[1])

    return np.array(lavec), nat, nkd


def get_dfc2(file_dfc2, temp_in):
    detect_temp_tag = False
    dfc2 = []

    with open(file_dfc2, "r") as f:
        line = f.readline()
        temp = None

        while line:
            if "Temp" in line:
                temp = float(line.strip().split()[-1])
                if temp == temp_in:
                    detect_temp_tag = True
                    break

            line = f.readline()

        line = f.readline()
        while line:
            if "#" in line:
                break

            data = line.strip().split()
            if len(data) > 0:
                dfc2.append(data)

            line = f.readline()

    if not detect_temp_tag:
        print(
            "Delta FC2 data for the specified temperature (%.2f K) does not exist in %s"
            % (temp_in, file_dfc2)
        )
        exit(1)

    return np.array(dfc2)


def create_newfc2(nx, ny, nz, fc2_orig, dfc2_array):
    fc2_new = np.copy(fc2_orig)

    for entry in dfc2_array:
        m1 = int(entry[0])
        m2 = int(entry[1])
        m3 = int(entry[2])
        iat = int(entry[3])
        icrd = int(entry[4])
        jat = int(entry[5])
        jcrd = int(entry[6])
        dfc2 = float(entry[7])

        m1 = -m1
        m2 = -m2
        m3 = -m3

        if m1 < 0:
            m1 += nx
        if m2 < 0:
            m2 += ny
        if m3 < 0:
            m3 += nz

        if m1 >= nx:
            m1 -= nx
        if m2 >= ny:
            m2 -= ny
        if m3 >= nz:
            m3 -= nz

        fc2_new[3 * iat + icrd][3 * jat + jcrd][m1][m2][m3] += dfc2

    return fc2_new


def print_fc2(header_in, nx, ny, nz, nat, fc2_in):
    for line in header_in:
        print(line.rstrip())
    print("%4d %3d %3d" % (nx, ny, nz))

    for icrd in range(3):
        for jcrd in range(3):
            for iat in range(nat):
                for jat in range(nat):
                    print(" %3d %3d %3d %3d" % (icrd + 1, jcrd + 1, iat + 1, jat + 1))

                    for m3 in range(nz):
                        for m2 in range(ny):
                            for m1 in range(nx):
                                print(
                                    " %3d %3d %3d %19.11e"
                                    % (
                                        m1 + 1,
                                        m2 + 1,
                                        m3 + 1,
                                        fc2_in[3 * iat + icrd][3 * jat + jcrd][m1][m2][
                                            m3
                                        ],
                                    )
                                )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "\nUsage:\n > python convert_to_qefc2.py original_QEfc2 scph_fc2_correction temperature\n"
        )
        exit(1)

    file_QEfc2 = sys.argv[1]
    file_dfc2 = sys.argv[2]
    temp_in = float(sys.argv[3])

    header, nat, nkd, nx, ny, nz, fc2 = parse_QEfc(file_QEfc2)
    _, nat2, nkd2 = get_structure_info_dfc2(file_dfc2)

    if nat != nat2:
        print("ERROR: The number of atoms in the primitive cell is inconsistent.")
        exit(1)

    if nkd != nkd2:
        print("ERROR: The number of atomic elements is inconsistent.")
        exit(1)

    dfc2 = get_dfc2(file_dfc2, temp_in)
    fc2_new = create_newfc2(nx, ny, nz, fc2, dfc2)
    print_fc2(header, nx, ny, nz, nat, fc2_new)
