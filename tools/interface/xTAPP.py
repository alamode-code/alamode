#
# xTAPP.py
#
# Interface to xTAPP (http://xtapp.cp.is.s.u-tokyo.ac.jp)
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

import numpy as np


class XtappParser(object):
    def __init__(self):
        self._prefix = None
        self._lattice_vector = None
        self._inverse_lattice_vector = None
        self._kd = None
        self._header_part = None
        self._nat = 0
        self._x_fractional = None
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
        self._BOHR_TO_ANGSTROM = 0.5291772108
        self._RYDBERG_TO_EV = 13.60569253

    def load_initial_structure(self, file_in):
        lavec, nat, nkd, str_tappinput = self._read_tappinput(
            file_in, self._BOHR_TO_ANGSTROM
        )
        x, kd, str_atom = self._read_atomdata(file_in, nat, nkd)
        str_kpoint = self._read_kpdata(file_in)
        str_struct_opt, str_opt_constr = self._read_structure_optimize(file_in)

        str_header = ""
        for entry in str_tappinput:
            str_header += entry
        for entry in str_kpoint:
            str_header += entry
        for entry in str_struct_opt:
            str_header += entry
        for entry in str_opt_constr:
            str_header += entry
        for i in range(nkd + 1):
            str_header += str_atom[i]

        self._lattice_vector = lavec
        self._inverse_lattice_vector = np.linalg.inv(lavec)
        self._x_fractional = x
        self._nat = nat
        self._kd = kd
        self._header_part = str_header
        self._initial_structure_loaded = True

    def generate_structures(
        self, prefix, header_list, disp_list, updated_structure=None
    ):
        self._set_number_of_zerofill(len(disp_list))
        self._prefix = prefix

        for header, disp in zip(header_list, disp_list):
            self._generate_input(header, disp)

    def parse(
        self,
        initial_cg,
        stdout_files,
        stdout_file_offset,
        str_unit,
        output_flags,
        filter_emin=None,
        filter_emax=None,
    ):
        if not self._initial_structure_loaded:
            self.load_initial_structure(initial_cg)

        self._set_unit_conversion_factor(str_unit)
        self._set_output_flags(output_flags)

        if self._print_disp or self._print_force:
            self._print_displacements_and_forces(
                stdout_files, stdout_file_offset, filter_emin, filter_emax
            )
        elif self._print_energy:
            self._print_energies(stdout_files, stdout_file_offset)

    def _generate_input(self, header, disp):
        nsym = 1
        symop = []
        symop.append([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        denom_tran = 1
        has_inv = 0

        filename = self._prefix + str(self._counter).zfill(self._nzerofills) + ".cg"

        with open(filename, "w") as f:
            f.write("%s" % self._header_part)
            for i in range(self._nat):
                f.write(
                    "%i %20.15f %20.15f %20.15f\n"
                    % (
                        self._kd[i],
                        self._x_fractional[i][0] + disp[i, 0],
                        self._x_fractional[i][1] + disp[i, 1],
                        self._x_fractional[i][2] + disp[i, 2],
                    )
                )
            f.write("# symmetry data\n")
            f.write("&symmetry\n")
            f.write("  number_sym_op = %i\n" % nsym)
            f.write("  has_inversion = %i\n" % has_inv)
            f.write("  denom_trans = %i\n" % denom_tran)
            f.write("/\n")

            mat_tmp = np.zeros((3, 3), dtype=int)

            for elems in symop:
                for i in range(3):
                    for j in range(3):
                        mat_tmp[i][j] = elems[3 * i + j]

                mat_inv = np.matrix(mat_tmp).I

                for i in range(3):
                    for j in range(3):
                        f.write("%4i" % mat_inv[i, j])

                f.write("   ")
                for i in range(3):
                    f.write("%4i" % elems[9 + i])

                f.write("\n")

            f.write("\n")

        self._counter += 1

    def _set_number_of_zerofill(self, npattern):
        nzero = 1

        while True:
            npattern //= 10
            if npattern == 0:
                break
            nzero += 1

        self._nzerofills = nzero

    @staticmethod
    def _read_tappinput(file_in, Bohr_to_Ang):
        list_tappinput = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if "main" in line and "data" in line:
                    flag_add = True
                    list_tappinput.append(line)
                elif "#" in line:
                    flag_add = False
                elif flag_add:
                    list_tappinput.append(line)

        if len(list_tappinput) == 0:
            raise RuntimeError("main data entry not found")

        list_tappinput_new = []

        for obj in list_tappinput:
            obj_split = obj.rstrip().split(",")
            for subobj in obj_split:
                if subobj:
                    list_tappinput_new.append(subobj)

        str_input = ""
        for entry in list_tappinput_new:
            str_input += entry + " "
        entrylist = str_input.split()
        lavec_list = []

        a = 0.0
        nkd = 0
        nat = 0

        # get lattice_factor
        for i in range(len(entrylist)):
            if "lattice_factor" in entrylist[i]:
                a = float(entrylist[i + 2])

            if "lattice_list" in entrylist[i]:
                for j in range(9):
                    lavec_list.append(entrylist[i + j + 2])

            if "number_element" in entrylist[i]:
                nkd = int(entrylist[i + 2])

            if "number_atom" in entrylist[i]:
                nat = int(entrylist[i + 2])

        if a == 0.0:
            raise RuntimeError("Couldn't read lattice_factor")

        if nkd == 0:
            raise RuntimeError("Couldn't read number_element")

        if nat == 0:
            raise RuntimeError("Couldn't read number_atom")

        if len(lavec_list) != 9:
            raise RuntimeError("Couldn't read lattice_list")

        lavec = np.zeros((3, 3))
        a *= Bohr_to_Ang

        for i in range(3):
            for j in range(3):
                lavec[j][i] = a * float(lavec_list[3 * i + j])

        return lavec, nat, nkd, list_tappinput

    def _set_unit_conversion_factor(self, str_unit):
        if str_unit == "ev":
            self._disp_conversion_factor = self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = 2.0 * self._RYDBERG_TO_EV

        elif str_unit == "rydberg":
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 2.0

        elif str_unit == "hartree":
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 1.0

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

    @property
    def lattice_vector(self):
        return self._lattice_vector

    @property
    def inverse_lattice_vector(self):
        return self._inverse_lattice_vector

    @property
    def atomic_kinds(self):
        return self._kd

    @property
    def x_fractional(self):
        return self._x_fractional

    def _print_displacements_and_forces(
        self, stdout_files, file_offset, filter_emin, filter_emax
    ):
        x0 = np.round(self._x_fractional, 8)
        lavec_transpose = self._lattice_vector.transpose() / self._BOHR_TO_ANGSTROM
        vec_refold = np.vectorize(self._refold)

        if file_offset is None:
            disp_offset = np.zeros((self._nat, 3))
            force_offset = np.zeros((self._nat, 3))
            epot_offset = 0.0

        else:
            x0_offset = self._get_coordinates_xtapp(file_offset, self._nat)
            force_offset = self._get_atomicforces_xtapp(file_offset, self._nat)
            epot_offset = self._get_energies_xtapp(file_offset)
            try:
                x0_offset = np.reshape(x0_offset, (self._nat, 3))
            except ValueError as err:
                raise RuntimeError(
                    "File %s contains too many position entries" % file_offset
                ) from err
            disp_offset = x0_offset - x0

            try:
                force_offset = np.reshape(force_offset, (self._nat, 3))
            except ValueError as err:
                raise RuntimeError(
                    "File %s contains too many position entries" % file_offset
                ) from err

        for search_target in stdout_files:
            x = self._get_coordinates_xtapp(search_target, self._nat)
            force = self._get_atomicforces_xtapp(search_target, self._nat)
            epot = self._get_energies_xtapp(search_target)
            ndata_disp = len(x) // (3 * self._nat)
            ndata_force = len(force) // (3 * self._nat)

            if ndata_disp != ndata_force:
                print(
                    "Error: The number of entries of displacement and force is inconsistent."
                )
                print("Ndata disp : %d, Ndata force : %d" % (ndata_disp, ndata_force))
                exit(1)

            ndata_energy = len(epot)
            if ndata_energy != ndata_disp:
                raise RuntimeError(
                    "The numbers of displacement and energy entries are different."
                )

            ndata = ndata_disp
            x = np.reshape(x, (ndata, self._nat, 3))
            force = np.reshape(force, (ndata, self._nat, 3))
            epot -= epot_offset
            epot *= self._RYDBERG_TO_EV * 2.0

            for idata in range(ndata):
                if filter_emin is not None:
                    if filter_emin > epot[idata]:
                        continue

                if filter_emax is not None:
                    if filter_emax < epot[idata]:
                        continue

                if self._print_disp:
                    disp = x[idata, :, :] - x0 - disp_offset
                    disp = np.dot(vec_refold(disp), lavec_transpose)
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

    def _print_energies(self, stdout_files, file_offset):
        if file_offset is None:
            etot_offset = 0.0
        else:
            data = self._get_energies_xtapp(file_offset)
            if len(data) > 1:
                raise RuntimeError(
                    "File %s contains too many energy entries" % file_offset
                )

            etot_offset = data[0]

        print("# Etot")
        for search_target in stdout_files:
            etot = self._get_energies_xtapp(search_target)

            for idata in range(len(etot)):
                val = etot[idata] - etot_offset
                val *= self._energy_conversion_factor

                print("%19.11E" % val)

    @staticmethod
    def _read_atomdata(file_in, nat_in, nkd_in):
        list_atom = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if "atom" in line and "data" in line:
                    flag_add = True
                    list_atom.append(line)
                elif "#" in line.strip():
                    flag_add = False
                elif flag_add:
                    list_atom.append(line)

        if len(list_atom) == 0:
            raise RuntimeError("atom data entry not found")

        x_out = np.zeros((nat_in, 3), dtype=float)
        kd_out = np.zeros(nat_in, dtype=int)

        for i in range(nat_in):
            list_tmp = list_atom[i + nkd_in + 1].rstrip().split()
            kd_out[i] = int(list_tmp[0])
            for j in range(3):
                x_out[i][j] = float(list_tmp[j + 1])

        return x_out, kd_out, list_atom

    @staticmethod
    def _read_kpdata(file_in):
        list_kpoint = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if "k-points" in line.rstrip():
                    flag_add = True
                    list_kpoint.append(line)
                elif "#" in line.strip():
                    flag_add = False
                elif flag_add:
                    list_kpoint.append(line)

        if len(list_kpoint) == 0:
            raise RuntimeError("k-points data entry not found")

        return list_kpoint

    @staticmethod
    def _read_structure_optimize(file_in):
        list_opt = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if "struct_opt" in line.rstrip():
                    flag_add = True
                    list_opt.append(line)
                elif "#" in line.strip():
                    flag_add = False
                elif flag_add:
                    list_opt.append(line)

        if len(list_opt) == 0:
            raise RuntimeError("struct_opt entry not found")

        list_opt2 = []
        flag_add = False

        with open(file_in) as openfileobject:
            for line in openfileobject:
                if "str_opt_constr" in line.rstrip():
                    flag_add = True
                    list_opt2.append(line)
                elif "#" in line.strip():
                    flag_add = False
                elif flag_add:
                    list_opt2.append(line)

        if len(list_opt2) == 0:
            raise RuntimeError("str_opt_constr entry not found")

        return list_opt, list_opt2

    @staticmethod
    def _get_coordinates_xtapp(str_file, nat):
        found_tag = False
        f = open(str_file, "r")
        line = f.readline()

        x = []
        while line:
            if "atom_position" in line:
                found_tag = True
                for _ in range(nat):
                    line = f.readline()
                    x.extend([t for t in line.rstrip().split()[1:]])
                break
            line = f.readline()

        if not found_tag:
            raise RuntimeError("atom_position tag not found in %s" % str_file)

        f.close()

        return np.array(x, dtype=float)

    @staticmethod
    def _get_atomicforces_xtapp(str_file, nat):
        found_tag = False
        f = open(str_file, "r")
        line = f.readline()

        force = []
        while line:
            if "force" in line:
                found_tag = True
                for _ in range(nat):
                    line = f.readline()
                    force.extend([t for t in line.rstrip().split()])
                break
            line = f.readline()

        if not found_tag:
            raise RuntimeError("force tag not found in %s" % str_file)

        f.close()

        return np.array(force, dtype=float)

    @staticmethod
    def _get_energies_xtapp(str_file):
        search_tag = "total_energy"
        found_tag = False

        etot = []
        with open(str_file) as openfileobject:
            for line in openfileobject:
                if search_tag in line:
                    energy_str = line.rstrip().split()[2]
                    etot.extend([energy_str[:-1]])
                    found_tag = True

        if not found_tag:
            raise RuntimeError("%s tag not found in %s" % (search_tag, str_file))

        return np.array(etot, dtype=float)

    @staticmethod
    def _refold(x):
        if x >= 0.5:
            return x - 1.0
        elif x < -0.5:
            return x + 1.0
        else:
            return x
