#
# QE.py
#
# Interface to Quantum ESPRESSO (http://www.quantum-espresso.org)
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
from __future__ import print_function

import copy
import math
import sys

import numpy as np


class QEParser(object):
    def __init__(self):
        self._prefix = None
        self._lattice_vector = None
        self._inverse_lattice_vector = None
        self._nat = 0
        self._x_fractional = None
        self._kd = None
        self._kdname = None
        self._counter = 1
        self._nzerofills = 0
        self._disp_conversion_factor = 1.0
        self._energy_conversion_factor = 1.0
        self._force_conversion_factor = 1.0
        self._initial_structure_loaded = False
        self._print_disp = True
        self._print_force = True
        self._print_energy = False
        self._print_born = False
        self._list_CONTROL = []
        self._list_SYSTEM = []
        self._list_ELECTRONS = []
        self._list_ATOMIC_SPECIES = []
        self._list_ATOMIC_POSITIONS = []
        self._list_K_POINTS = []
        self._list_CELL_PARAMETERS = []
        self._list_OCCUPATIONS = []
        self._celldm = [None] * 6
        self._BOHR_TO_ANGSTROM = 0.5291772108
        self._RYDBERG_TO_EV = 13.60569253

    def load_initial_structure(self, file_in):
        # Parse fortran namelists
        self._list_CONTROL = self._get_namelist(file_in, "&CONTROL")
        self._list_SYSTEM = self._get_namelist(file_in, "&SYSTEM")
        self._list_ELECTRONS = self._get_namelist(file_in, "&ELECTRONS")

        # Parse general options
        tags = [
            "ATOMIC_SPECIES",
            "ATOMIC_POSITIONS",
            "K_POINTS",
            "CELL_PARAMETERS",
            "OCCUPATIONS",
            "CONSTRAINTS",
            "ATOMIC_FORCES",
        ]

        self._list_ATOMIC_SPECIES = self._get_options("ATOMIC_SPECIES", tags, file_in)
        self._list_ATOMIC_POSITIONS = self._get_options(
            "ATOMIC_POSITIONS", tags, file_in
        )
        self._list_K_POINTS = self._get_options("K_POINTS", tags, file_in)
        self._list_CELL_PARAMETERS = self._get_options("CELL_PARAMETERS", tags, file_in)
        self._list_OCCUPATIONS = self._get_options("OCCUPATIONS", tags, file_in)

        # Set lattice vectors and fractional coordinates
        self._set_system_info()
        self._initial_structure_loaded = True

    def generate_structures(
        self, prefix, header_list, disp_list, updated_structure=None
    ):
        self._set_number_of_zerofill(len(disp_list))
        self._prefix = prefix
        self._counter = 1

        for header, disp in zip(header_list, disp_list):
            self._generate_input(header, disp)

    def parse(
        self,
        initial_pwin,
        pwout_files,
        pwout_file_offset,
        str_unit,
        output_flags,
        filter_emin=None,
        filter_emax=None,
    ):
        if not self._initial_structure_loaded:
            self.load_initial_structure(initial_pwin)

        self._set_unit_conversion_factor(str_unit)
        self._set_output_flags(output_flags)

        if self._print_disp or self._print_force:
            self._print_displacements_and_forces(
                pwout_files, pwout_file_offset, filter_emin, filter_emax
            )

        elif self._print_energy:
            self._print_energies(pwout_files, pwout_file_offset)

        elif self._print_born:
            self._print_borninfo(pwout_files)

    def get_displacements(self, pwout_files, unit="bohr"):
        if not self._initial_structure_loaded:
            raise RuntimeError(
                "Please call load_initial_structure before using this method"
            )

        x0 = np.round(self._x_fractional, 8)
        lavec_transpose = self._lattice_vector.transpose()
        vec_refold = np.vectorize(self._refold)

        disp_merged = []

        if unit == "bohr":
            unit_factor = 1.0 / self._BOHR_TO_ANGSTROM
        elif unit == "angstrom":
            unit_factor = 1.0
        else:
            raise RuntimeError(
                "Invalid unit type. Valid values are 'bohr' and 'angstrom'."
            )

        for search_target in pwout_files:
            x = self._get_coordinates_pwout(search_target)

            ndata, _, _ = np.shape(x)
            disp = np.zeros((ndata, self._nat, 3))

            for idata in range(ndata):
                disp[idata, :, :] = x[idata, :, :] - x0
                disp[idata, :, :] = np.dot(
                    vec_refold(disp[idata, :, :]), lavec_transpose
                )
                disp[idata, :, :] *= unit_factor

            disp_merged.extend(disp)

        return disp_merged

    def _generate_input(self, header, disp):
        filename = self._prefix + str(self._counter).zfill(self._nzerofills) + ".pw.in"

        with open(filename, "w") as f:
            for entry in self._list_CONTROL:
                f.write(entry)
            for entry in self._list_SYSTEM:
                f.write(entry)
            for entry in self._list_ELECTRONS:
                f.write(entry)
            for entry in self._list_ATOMIC_SPECIES:
                f.write(entry)
            f.write("ATOMIC_POSITIONS crystal\n")
            for i in range(self._nat):
                f.write(
                    "%s %20.15f %20.15f %20.15f\n"
                    % (
                        self._kdname[self._kd[i]],
                        self._x_fractional[i][0] + disp[i, 0],
                        self._x_fractional[i][1] + disp[i, 1],
                        self._x_fractional[i][2] + disp[i, 2],
                    )
                )
            for entry in self._list_K_POINTS:
                f.write(entry)
            for entry in self._list_CELL_PARAMETERS:
                f.write(entry)
            for entry in self._list_OCCUPATIONS:
                f.write(entry)
            f.write("\n")

        self._counter += 1

    def _print_displacements_and_forces(
        self, pwout_files, file_offset, filter_emin, filter_emax
    ):
        x0 = np.round(self._x_fractional, 8)
        lavec_transpose = self._lattice_vector.transpose() / self._BOHR_TO_ANGSTROM
        vec_refold = np.vectorize(self._refold)

        # Parse offset component
        if file_offset is None:
            disp_offset = np.zeros((1, self._nat, 3))
            force_offset = np.zeros((self._nat, 3))
            epot_offset = 0.0

        else:
            x_offset = self._get_coordinates_pwout(file_offset)
            if x_offset is None:
                raise RuntimeError(
                    "File %s does not contain position entry" % file_offset
                )

            ndata_offset, _, _ = np.shape(x_offset)
            if ndata_offset > 1:
                raise RuntimeError(
                    "File %s contains too many position entries" % file_offset
                )

            disp_offset = x_offset - x0
            force_offset = self._get_atomicforces_pwout(file_offset)
            if force_offset is None:
                raise RuntimeError("File %s does not contain force entry" % file_offset)
            try:
                force_offset = np.reshape(force_offset, (self._nat, 3))
            except:
                raise RuntimeError(
                    "File %s contains too many force entries" % file_offset
                )

            epot_offset = self._get_energies_pwout(file_offset)
            if epot_offset is None:
                raise RuntimeError(
                    "File %s does not contain energy entry" % file_offset
                )
            epot_offset = np.array(epot_offset, dtype=float)
            if len(epot_offset) > 1:
                raise RuntimeError(
                    "File %s contains too many energy entries" % file_offset
                )

        for search_target in pwout_files:
            x = self._get_coordinates_pwout(search_target)
            force = self._get_atomicforces_pwout(search_target)
            epot = self._get_energies_pwout(search_target)

            if x is None or force is None or epot is None:
                continue

            num_data_force = len(force) // (3 * self._nat)
            force = np.reshape(force, (num_data_force, self._nat, 3))
            num_data_disp, _, _ = np.shape(x)

            if (
                num_data_disp != num_data_force
                and self._print_disp
                and self._print_force
            ):
                print(
                    "Error: The number of entries of displacement and force is inconsistent."
                )
                print(
                    "Ndata disp : %d, Ndata force : %d"
                    % (num_data_disp, num_data_force)
                )
                exit(1)

            ndata_energy = len(epot)
            if ndata_energy != num_data_disp:
                raise RuntimeError(
                    "The numbers of displacement and energy entries are different."
                )

            epot = np.array(epot, dtype=float)
            epot -= epot_offset
            epot *= self._RYDBERG_TO_EV

            for idata in range(num_data_disp):
                if filter_emin is not None:
                    if filter_emin > epot[idata]:
                        continue

                if filter_emax is not None:
                    if filter_emax < epot[idata]:
                        continue

                if self._print_disp:
                    disp = x[idata, :, :] - x0 - disp_offset
                    disp = np.dot(vec_refold(disp[0, :, :]), lavec_transpose)
                    disp *= self._disp_conversion_factor

                if self._print_force:
                    f = force[idata, :, :] - force_offset
                    f *= self._force_conversion_factor

                print(
                    "# Filename: %s, Snapshot: %d, E_pot (eV): %s"
                    % (search_target, idata + 1, epot[idata])
                )

                if self._print_disp and self._print_force:
                    for i in range(self._nat):
                        print(
                            "%15.7F %15.7F %15.7F %20.8E %15.8E %15.8E"
                            % (
                                disp[i, 0],
                                disp[i, 1],
                                disp[i, 2],
                                f[i, 0],
                                f[i, 1],
                                f[i, 2],
                            )
                        )
                elif self._print_disp:
                    for i in range(self._nat):
                        print(
                            "%15.7F %15.7F %15.7F"
                            % (disp[i, 0], disp[i, 1], disp[i, 2])
                        )
                elif self._print_force:
                    for i in range(self._nat):
                        print("%15.8E %15.8E %15.8E" % (f[i, 0], f[i, 1], f[i, 2]))

    def _print_energies(self, pwout_files, file_offset):
        if file_offset is None:
            etot_offset = 0.0
        else:
            data = self._get_energies_pwout(file_offset)
            if data is None:
                raise RuntimeError(
                    "File %s does not contain energy entry" % file_offset
                )

            if len(data) > 1:
                raise RuntimeError(
                    "File %s contains too many energy entries" % file_offset
                )
            etot_offset = data[0]

        print("# Etot")
        for search_target in pwout_files:
            etot = self._get_energies_pwout(search_target)
            if etot is None:
                continue
            for idata in range(len(etot)):
                val = etot[idata] - etot_offset
                val *= self._energy_conversion_factor
                print("%19.11E" % val)

    def _print_borninfo(self, phout_files):
        for search_target in phout_files:
            dielec, borncharge = self._get_borninfo_phout(search_target)
            nat_prim, _, _ = np.shape(borncharge)

            for i in range(3):
                print(
                    "%16.8F %16.8F %16.8F" % (dielec[i, 0], dielec[i, 1], dielec[i, 2])
                )

            for j in range(nat_prim):
                for i in range(3):
                    print(
                        "%16.8F %16.8F %16.8F"
                        % (
                            borncharge[j, i, 0],
                            borncharge[j, i, 1],
                            borncharge[j, i, 2],
                        )
                    )

    def _set_system_info(self):
        list_mod = []

        for obj in self._list_SYSTEM:
            obj_split = obj.rstrip().split(",")
            for subobj in obj_split:
                if subobj:
                    index = subobj.find("=")
                    if index > 0:
                        subobj = subobj[:index] + " = " + subobj[index + 1 :]
                    list_mod.append(subobj)

        str_input = ""
        for entry in list_mod:
            str_input += entry + " "
        entrylist = str_input.split()

        for i in range(len(entrylist)):
            if "ibrav" in entrylist[i]:
                ibrav = int(entrylist[i + 2])

            if "nat" in entrylist[i]:
                self._nat = int(entrylist[i + 2])

            if "ntyp" in entrylist[i]:
                ntyp = int(entrylist[i + 2])

            if "celldm(1)" in entrylist[i]:
                # Do not assign the value if the comment character '!'
                # appears in front of the celldm(1) keyword
                has_comment = False
                for elem in self._list_SYSTEM:
                    if "celldm(1)" in elem:
                        has_comment = "!" == elem.strip().split()[0][0]

                if not has_comment:
                    self._celldm[0] = float(entrylist[i + 2])

            if "celldm(2)" in entrylist[i]:
                self._celldm[1] = float(entrylist[i + 2])

            if "celldm(3)" in entrylist[i]:
                self._celldm[2] = float(entrylist[i + 2])

            if "celldm(4)" in entrylist[i]:
                self._celldm[3] = float(entrylist[i + 2])

            if "celldm(5)" in entrylist[i]:
                self._celldm[4] = float(entrylist[i + 2])

            if "celldm(6)" in entrylist[i]:
                self._celldm[5] = float(entrylist[i + 2])

        self._set_lattice_vector(ibrav)
        self._set_fractional_coordinate()

    def _set_lattice_vector(self, ibrav):
        """.

        Computer lattice vector in units of Angstrom for given ibrav and celldm.
        Doc/INPUT_PW.txt was used as a reference.
        """

        lavec = np.zeros((3, 3))

        if ibrav == 0:
            if self._list_CELL_PARAMETERS is None:
                raise RuntimeError("CELL_PARAMETERS must be given when ibrav = 0.")

            mode = self._list_CELL_PARAMETERS[0].rstrip().split()

            if len(mode) == 1:
                raise RuntimeError(
                    "Error : Please specify either alat, bohr, or angstrom for CELL_PARAMETERS"
                )

            mode_str = mode[1].lower()

            for i in range(3):
                lavec[i][:] = [
                    float(entry)
                    for entry in self._list_CELL_PARAMETERS[i + 1].rstrip().split()
                ]
            lavec = np.array(lavec)

            if "alat" in mode_str:
                if not self._celldm[0]:
                    raise RuntimeError(
                        "celldm(1) must be given when 'alat' is used for CELL_PARAMETERS"
                    )

                for i in range(3):
                    for j in range(3):
                        lavec[i][j] *= self._celldm[0]

            elif "angstrom" in mode_str:
                # convert the lattice vectors in Bohr unit here to make them back to
                # the angstrom unit at the end of this method.
                for i in range(3):
                    for j in range(3):
                        lavec[i][j] /= self._BOHR_TO_ANGSTROM

            elif "bohr" not in mode_str:
                raise RuntimeError(
                    "Error : Invalid option for CELL_PARAMETERS: %s" % mode[1]
                )

        elif ibrav == 1:
            if not self._celldm[0]:
                raise RuntimeError("celldm(1) must be given when ibrav = 1.")

            else:
                a = self._celldm[0]
                lavec = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])

        elif ibrav == 2:
            if not self._celldm[0]:
                raise RuntimeError("celldm(1) must be given when ibrav = 2.")

            else:
                a = self._celldm[0] / 2.0
                lavec = np.array([[-a, 0.0, a], [0.0, a, a], [-a, a, 0.0]])

        elif ibrav == 3:
            if not self._celldm[0]:
                raise RuntimeError("celldm(1) must be given when ibrav = 3.")

            else:
                a = self._celldm[0] / 2.0
                lavec = np.array([[a, a, a], [-a, a, a], [-a, -a, a]])

        elif ibrav == 4:
            if not self._celldm[0] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1) and celldm(3) must be given when ibrav = 4."
                )

            else:
                a = self._celldm[0]
                c = self._celldm[0] * self._celldm[2]
                lavec = np.array(
                    [
                        [a, 0.0, 0.0],
                        [-0.5 * a, math.sqrt(3.0) / 2.0 * a, 0.0],
                        [0.0, 0.0, c],
                    ]
                )

        elif ibrav == 5 or ibrav == -5:
            if not self._celldm[0] or not self._celldm[3]:
                raise RuntimeError(
                    "celldm(1) and celldm(4) must be given when ibrav = 5, -5."
                )

            else:
                a = self._celldm[0]
                cosalpha = self._celldm[3]
                tx = a * math.sqrt((1.0 - cosalpha) / 2.0)
                ty = a * math.sqrt((1.0 - cosalpha) / 6.0)
                tz = a * math.sqrt((1.0 + 2.0 * cosalpha) / 3.0)

                if ibrav == 5:
                    lavec = np.array(
                        [[tx, -ty, tz], [0.0, 2.0 * ty, tz], [-tx, -ty, tz]]
                    )

                else:
                    a_prime = a / math.sqrt(3.0)
                    u = tz - 2.0 * math.sqrt(2.0) * ty
                    v = tz + math.sqrt(2.0) * ty

                    u *= a_prime
                    v *= a_prime

                    lavec = np.array([[u, v, v], [v, u, v], [v, v, u]])

        elif ibrav == 6:
            if not self._celldm[0] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1) and celldm(3) must be given when ibrav = 6."
                )

            else:
                a = self._celldm[0]
                c = self._celldm[0] * self._celldm[2]
                lavec = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]])

        elif ibrav == 7:
            if not self._celldm[0] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1) and celldm(3) must be given when ibrav = 7."
                )

            else:
                a = self._celldm[0]
                c = self._celldm[0] * self._celldm[2]
                lavec = np.array(
                    [
                        [a / 2.0, -a / 2.0, c / 2.0],
                        [a / 2.0, a / 2.0, c / 2.0],
                        [-a / 2.0, -a / 2.0, c / 2.0],
                    ]
                )

        elif ibrav == 8:
            if not self._celldm[0] or not self._celldm[1] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1), celldm(2), and celldm(3) must be given\
                when ibrav = 8."
                )

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]

                lavec = np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])

        elif ibrav == 9 or ibrav == -9:
            if not self._celldm[0] or not self._celldm[1] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1), celldm(2), and celldm(3) must be given\
                when ibrav = 9 or -9."
                )

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]

                if ibrav == 9:
                    lavec = np.array(
                        [
                            [a / 2.0, b / 2.0, 0.0],
                            [-a / 2.0, b / 2.0, 0.0],
                            [0.0, 0.0, c],
                        ]
                    )
                else:
                    lavec = np.array(
                        [
                            [a / 2.0, -b / 2.0, 0.0],
                            [a / 2.0, b / 2.0, 0.0],
                            [0.0, 0.0, c],
                        ]
                    )

        elif ibrav == 10:
            if not self._celldm[0] or not self._celldm[1] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1), celldm(2), and celldm(3) must be given\
                when ibrav = 10."
                )

            else:
                a = self._celldm[0] / 2.0
                b = self._celldm[0] * self._celldm[1] / 2.0
                c = self._celldm[0] * self._celldm[2] / 2.0
                lavec = np.array([[a, 0.0, c], [a, b, 0.0], [0.0, b, c]])

        elif ibrav == 11:
            if not self._celldm[0] or not self._celldm[1] or not self._celldm[2]:
                raise RuntimeError(
                    "celldm(1), celldm(2), and celldm(3) must be given\
                when ibrav = 11."
                )

            else:
                a = self._celldm[0] / 2.0
                b = self._celldm[0] * self._celldm[1] / 2.0
                c = self._celldm[0] * self._celldm[2] / 2.0
                lavec = np.array([[a, b, c], [-a, b, c], [-a, -b, c]])

        elif ibrav == 12:
            if (
                not self._celldm[0]
                or not self._celldm[1]
                or not self._celldm[2]
                or not self._celldm[3]
            ):
                raise RuntimeError(
                    "celldm(1), celldm(2), celldm(3), and celldm(4)\
                must be given when ibrav = 12."
                )

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]
                gamma = math.acos(self._celldm[3])
                lavec = np.array(
                    [
                        [a, 0.0, 0.0],
                        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
                        [0.0, 0.0, c],
                    ]
                )

        elif ibrav == -12:
            if (
                not self._celldm[0]
                or not self._celldm[1]
                or not self._celldm[2]
                or not self._celldm[4]
            ):
                raise RuntimeError(
                    "celldm(1), celldm(2), celldm(3), and celldm(5)\
                must be given when ibrav = -12."
                )

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]
                beta = math.acos(self._celldm[4])
                lavec = np.array(
                    [
                        [a, 0.0, 0.0],
                        [0.0, b, 0.0],
                        [c * math.cos(beta), 0.0, c * math.sin(beta)],
                    ]
                )

        elif ibrav == 13:
            if (
                not self._celldm[0]
                or not self._celldm[1]
                or not self._celldm[2]
                or not self._celldm[3]
            ):
                raise RuntimeError(
                    "celldm(1), celldm(2), celldm(3), and celldm(4)\
                must be given when ibrav = 13."
                )

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]
                gamma = math.acos(self._celldm[3])
                lavec = np.array(
                    [
                        [a / 2.0, 0.0, -c / 2.0],
                        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
                        [a / 2.0, 0.0, c / 2.0],
                    ]
                )

        elif ibrav == 14:
            if (
                not self._celldm[0]
                or not self._celldm[1]
                or not self._celldm[2]
                or not self._celldm[3]
                or not self._celldm[4]
                or not self._celldm[5]
            ):
                raise RuntimeError("All celldm must be given when ibrav = 14.")

            else:
                a = self._celldm[0]
                b = self._celldm[0] * self._celldm[1]
                c = self._celldm[0] * self._celldm[2]
                alpha = math.acos(self._celldm[3])
                beta = math.acos(self._celldm[4])
                gamma = math.acos(self._celldm[5])

                lavec = np.array(
                    [
                        [a, 0.0, 0.0],
                        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
                        [
                            c * math.cos(beta),
                            c
                            * (math.cos(alpha) - math.cos(beta) * math.cos(gamma))
                            / math.sin(gamma),
                            c
                            * math.sqrt(
                                1.0
                                + 2.0
                                * math.cos(alpha)
                                * math.cos(beta)
                                * math.cos(gamma)
                                - math.cos(alpha) ** 2
                                - math.cos(beta) ** 2
                                - math.cos(gamma) ** 2
                            )
                            / math.sin(gamma),
                        ],
                    ]
                )

        else:
            raise RuntimeError("Invalid ibrav = %s" % ibrav)

        # if celldm(1) is empty, calculate it from the lattice vector for later use.
        if not self._celldm[0]:
            self._celldm[0] = math.sqrt(np.dot(lavec[0][:], lavec[0][:]))

        # Transpose for later use
        lavec = lavec.transpose()

        # Convert to Angstrom unit
        for i in range(3):
            for j in range(3):
                lavec[i][j] *= self._BOHR_TO_ANGSTROM

        self._lattice_vector = lavec
        self._inverse_lattice_vector = np.linalg.inv(lavec)

    def _set_fractional_coordinate(self):
        list_tmp = self._list_ATOMIC_POSITIONS[0].rstrip().split()

        if len(list_tmp) == 1:
            raise RuntimeError(
                "Error : Please specify either alat, "
                " bohr, angstrom, or crystal for ATOMIC_POSITIONS"
            )

        mode_str = list_tmp[1].lower()
        if "crystal_sg" in mode_str:
            raise RuntimeError(
                "Error : Sorry. 'crystal_sg' is not supported in this script. "
                "Please use another option."
            )

        xtmp = np.zeros((self._nat, 3))
        kd = []

        for i in range(self._nat):
            list_tmp = self._list_ATOMIC_POSITIONS[i + 1].rstrip().split()
            kd.append(list_tmp[0])
            xtmp[i][:] = [float(j) for j in list_tmp[1:4]]

        # lattice_vector is in units of Angstrom, so the unit of aa_inv is (Angstrom)^-1
        aa_inv = copy.deepcopy(self._inverse_lattice_vector)

        if "alat" in mode_str:
            # atomic positions are in cartesian coordinates in units of the lattice parameter (celldim(1))
            a_angstrom = self._celldm[0] * self._BOHR_TO_ANGSTROM

            for i in range(3):
                for j in range(3):
                    aa_inv[i][j] *= a_angstrom

            for i in range(self._nat):
                xtmp[i][:] = np.dot(xtmp[i][:], aa_inv.transpose())

        elif "bohr" in mode_str:
            for i in range(3):
                for j in range(3):
                    aa_inv[i][j] *= self._BOHR_TO_ANGSTROM

            for i in range(self._nat):
                xtmp[i][:] = np.dot(xtmp[i][:], aa_inv.transpose())

        elif "angstrom" in mode_str:
            for i in range(self._nat):
                xtmp[i][:] = np.dot(xtmp[i][:], aa_inv.transpose())

        elif "crystal" not in mode_str:
            raise RuntimeError(
                "Error : Invalid option for ATOMIC_POSITIONS: %s" % mode_str
            )

        kdname = []
        for entry in kd:
            if entry not in kdname:
                kdname.append(entry)
        dict_kd = {}
        counter = 0
        for name in kdname:
            dict_kd[name] = counter
            counter += 1
        kd_int = []
        for entry in kd:
            kd_int.append(dict_kd[entry])

        self._kd = kd_int
        self._kdname = kdname
        self._x_fractional = xtmp

    def _set_number_of_zerofill(self, npattern):
        nzero = 1

        while True:
            npattern //= 10
            if npattern == 0:
                break
            nzero += 1

        self._nzerofills = nzero

    def _set_unit_conversion_factor(self, str_unit):
        if str_unit == "ev":
            self._disp_conversion_factor = self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = self._RYDBERG_TO_EV

        elif str_unit == "rydberg":
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 1.0

        elif str_unit == "hartree":
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 0.5

        else:
            raise RuntimeError("This cannot happen.")

        self._force_conversion_factor = (
            self._energy_conversion_factor / self._disp_conversion_factor
        )

    def _set_output_flags(self, output_flags):
        self._print_disp, self._print_force, self._print_energy, self._print_born = (
            output_flags
        )

    @property
    def nat(self):
        return self._nat

    @nat.setter
    def nat(self, nat):
        self._nat = nat

    @property
    def lattice_vector(self):
        return self._lattice_vector

    @lattice_vector.setter
    def lattice_vector(self, lattice_vector):
        self._lattice_vector = lattice_vector
        self._inverse_lattice_vector = np.linalg.inv(lattice_vector)

    @property
    def inverse_lattice_vector(self):
        return self._inverse_lattice_vector

    @property
    def kd(self):
        return self._kd

    @property
    def kd_in_str(self):
        return [self._kdname[i] for i in self._kd]

    @kd.setter
    def kd(self, kd):
        self._kd = kd

    @kd_in_str.setter
    def kd_in_str(self, kd_in_str):
        map_name2num = {}
        for i, name in enumerate(self._kdname):
            map_name2num[name] = i
        self._kd = [map_name2num[t] for t in kd_in_str]

    @property
    def atomic_kinds(self):
        return self._kd

    @property
    def x_fractional(self):
        return self._x_fractional

    @x_fractional.setter
    def x_fractional(self, x_fractional):
        self._x_fractional = x_fractional

    @property
    def list_system(self):
        return self._list_SYSTEM

    @list_system.setter
    def list_system(self, list_in):
        self._list_SYSTEM = list_in

    @property
    def list_cell_parameters(self):
        return self._list_CELL_PARAMETERS

    @list_cell_parameters.setter
    def list_cell_parameters(self, list_in):
        self._list_CELL_PARAMETERS = list_in

    @property
    def list_k_points(self):
        return self._list_K_POINTS

    @list_k_points.setter
    def list_k_points(self, list_in):
        self._list_K_POINTS = list_in

    @staticmethod
    def _get_namelist(file_in, namelist_tag):
        list_out = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                line_upper = line.upper()
                if namelist_tag in line_upper:
                    flag_add = True
                    list_out.append(line)
                elif line.strip() == "/":
                    flag_add = False
                elif flag_add:
                    list_out.append(line)

        if len(list_out) == 0:
            print("%s field not found" % namelist_tag)
            exit(1)

        list_out.append("/\n")
        return list_out

    @staticmethod
    def _get_options(option_tag, taglists, file_in):
        list_out = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if option_tag in line:
                    flag_add = True
                    list_out.append(line)
                elif len(line.split()) > 0 and line.split()[0] in taglists:
                    flag_add = False
                elif flag_add:
                    if line.strip():
                        list_out.append(line)

        return list_out

    @staticmethod
    def _refold(x):
        if x >= 0.5:
            return x - 1.0
        elif x < -0.5:
            return x + 1.0
        else:
            return x

    def _get_coordinates_pwout(self, pwout_file):
        """
        Return the fractional coordinates of atoms
        """
        search_flag = "site n.     atom                  positions (alat units)"
        x = np.zeros((self._nat, 3))
        num_data_disp_extra = 0
        basis = ""
        found_tag = False

        f = open(pwout_file, "r")
        line = f.readline()

        while line:
            if search_flag in line:
                found_tag = True
                for i in range(self._nat):
                    line = f.readline()
                    x[i][:] = [float(t) for t in line.rstrip().split()[6:9]]
                break
            line = f.readline()

        if not found_tag:
            # print("%s tag not found in %s" % (search_flag, pwout_file), file=sys.stderr)
            return None

        x = (
            self._celldm[0]
            * np.dot(x, self._inverse_lattice_vector.transpose())
            * self._BOHR_TO_ANGSTROM
        )

        # Search additional entries containing atomic position
        # (for parsing MD trajectory)
        search_flag2 = "ATOMIC_POSITIONS "
        x_additional = []

        while line:
            if search_flag2 in line:
                if not basis:
                    basis = line.rstrip().split()[1]

                num_data_disp_extra += 1
                for i in range(self._nat):
                    line = f.readline()
                    x_additional.extend([t for t in line.rstrip().split()[1:4]])
            line = f.readline()
        f.close()

        x_additional = np.array(x_additional, dtype=float)
        # The basis of the coordinate in x_additional can be different
        # from that of x. Therefore, perform basis conversion here.
        if num_data_disp_extra > 0:
            if "alat" in basis:
                conversion_mat = (
                    self._celldm[0]
                    * self._inverse_lattice_vector.transpose()
                    * self._BOHR_TO_ANGSTROM
                )
            elif "bohr" in basis:
                conversion_mat = (
                    self._inverse_lattice_vector.transpose * self._BOHR_TO_ANGSTROM
                )
            elif "angstrom" in basis:
                conversion_mat = self._inverse_lattice_vector.transpose()
            elif "crystal" in basis:
                conversion_mat = np.identity(3)
            else:
                raise RuntimeError("This cannot happen.")

            x_additional = np.reshape(x_additional, (num_data_disp_extra, self._nat, 3))
            for i in range(num_data_disp_extra):
                x_additional[i, :, :] = np.dot(x_additional[i, :, :], conversion_mat)

        if num_data_disp_extra <= 1:
            return np.reshape(x, (1, self._nat, 3))
        else:
            x_merged = np.zeros((num_data_disp_extra, self._nat, 3))
            x_merged[0, :, :] = x[:, :]
            x_merged[1:, :, :] = x_additional[:-1, :, :]
            return x_merged

    def _get_atomicforces_pwout(self, pwout_file):
        search_tag = "Forces acting on atoms (Ry/au):"
        search_tag_QE6 = "Forces acting on atoms (cartesian axes, Ry/au):"

        found_tag = False

        f = open(pwout_file, "r")
        line = f.readline()

        force = []

        while line:
            if search_tag in line or search_tag_QE6 in line:
                found_tag = True
                f.readline()
                for i in range(self._nat):
                    line = f.readline()
                    force.extend([t for t in line.rstrip().split()[6:9]])
            line = f.readline()
        f.close()

        if not found_tag:
            print("following search tags not found in %s" % pwout_file, file=sys.stderr)
            print(search_tag, file=sys.stderr)
            print(search_tag_QE6, file=sys.stderr)
            return None

        return np.array(force, dtype=float)

    @staticmethod
    def _get_energies_pwout(pwout_file):
        search_tag = "!    total energy"
        found_tag = False
        etot = []
        with open(pwout_file) as openfileobject:
            for line in openfileobject:
                if search_tag in line:
                    etot.extend([line.rstrip().split()[4]])
                    found_tag = True

        if not found_tag:
            print("%s tag not found in %s" % (search_tag, pwout_file), file=sys.stderr)
            return None

        return np.array(etot, dtype=float)

    @staticmethod
    def _get_borninfo_phout(phout_file):
        dielec = []
        borncharge = []

        search_tag1 = "Dielectric constant in cartesian axis"
        search_tags = ["Px", "Py", "Pz", "Ex", "Ey", "Ez"]
        f = open(phout_file, "r")
        line = f.readline()

        found_tag1 = False
        found_tag2 = False

        while line:
            if search_tag1 in line:
                found_tag1 = True
                f.readline()
                for i in range(3):
                    line = f.readline()
                    dielec.extend([float(t) for t in line.strip().split()[1:4]])

            if any(s in line for s in search_tags):
                found_tag2 = True
                borncharge.extend(float(t) for t in line.strip().split()[2:5])

            line = f.readline()
        f.close()

        if not found_tag1 or not found_tag2:
            print(
                "Dielectric constants or Born effective charges are not found"
                "in %s" % phout_file,
                file=sys.stderr,
            )
            return None

        nat2 = len(borncharge) // 9
        dielec = np.reshape(np.array(dielec[9:]), (3, 3))
        borncharge = np.reshape(np.array(borncharge), (nat2, 3, 3))
        return dielec, borncharge[: nat2 // 2, :, :]
