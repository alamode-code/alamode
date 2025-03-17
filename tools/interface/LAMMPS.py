#
# LAMMPS.py
#
# Interface to LAMMPS (http://lammps.sandia.gov)
#
# Copyright (c) 2017-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#

import math

import numpy as np


class LammpsParser(object):
    def __init__(self):
        self._prefix = None
        self._lattice_vector = None
        self._inverse_lattice_vector = None
        self._kd = None
        self._charges = None
        self._common_settings = None
        self._nat = 0
        self._x_cartesian = None
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
        lammps_box_params = {}
        f = open(file_in, "r")
        f.readline()

        common_settings = []
        for line in f:
            if "Atoms" in line:
                break

            split_line = line.strip().split()
            if len(split_line) % 2 == 0:
                for i in range(len(split_line) // 2):
                    lammps_box_params[split_line[i + len(split_line) // 2]] = float(
                        split_line[i]
                    )
            common_settings.append(line.rstrip())

        atoms = []
        for line in f:
            if line.strip():
                atoms.append(line.rstrip().split())

        atoms = np.array(atoms)
        nat = len(atoms)
        ncols = len(atoms[0, :])

        if ncols == 5:
            kd = np.array(atoms[:, 1], dtype=int)
            x = np.array(atoms[:, 2:5], dtype=np.float64)
            charges = None
        elif ncols == 6:
            kd = np.array(atoms[:, 1], dtype=int)
            x = np.array(atoms[:, 3:6], dtype=np.float64)
            charges = np.array(atoms[:, 2], dtype=np.float64)

        self._common_settings = common_settings
        self._lattice_vector = self._compute_lattice_vector_from_boxparams(
            lammps_box_params
        )
        self._inverse_lattice_vector = np.linalg.inv(self._lattice_vector)
        self._nat = nat
        self._x_cartesian = x
        self._x_fractional = self._get_fractional_coordinate(
            x, self._inverse_lattice_vector
        )
        self._kd = kd
        self._charges = charges
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
        initial_lammps,
        dump_files,
        dump_file_offset,
        str_unit,
        output_flags,
        filter_emin=None,
        filter_emax=None,
    ):
        if not self._initial_structure_loaded:
            self.load_initial_structure(initial_lammps)

        self._set_unit_conversion_factor(str_unit)
        self._set_output_flags(output_flags)

        if self._print_disp and self._print_force:
            self._print_displacements_and_forces(dump_files, dump_file_offset)
        elif self._print_disp:
            self._print_displacements(dump_files, dump_file_offset)
        elif self._print_force:
            self._print_atomicforces(dump_files, dump_file_offset)

    def _generate_input(self, header, disp):
        filename = self._prefix + str(self._counter).zfill(self._nzerofills) + ".lammps"
        f = open(filename, "w")
        f.write("%s\n" % header)

        for line in self._common_settings:
            f.write("%s\n" % line)

        f.write("%s\n\n" % "Atoms")

        if self._charges is None:
            for i in range(self._nat):
                f.write("%5d %3d" % (i + 1, self._kd[i]))
                disp_tmp = np.dot(disp[i], self._lattice_vector.transpose())
                for j in range(3):
                    f.write("%20.15f" % (self._x_cartesian[i][j] + disp_tmp[j]))
                f.write("\n")
            f.write("\n")
        else:
            for i in range(self._nat):
                f.write("%5d %3d %11.6f" % (i + 1, self._kd[i], self._charges[i]))
                disp_tmp = np.dot(disp[i], self._lattice_vector.transpose())
                for j in range(3):
                    f.write("%20.15f" % (self._x_cartesian[i][j] + disp_tmp[j]))
                f.write("\n")
            f.write("\n")
        f.close()

        self._counter += 1

    def _print_displacements_and_forces(self, lammps_files, file_offset):
        if file_offset is None:
            disp_offset = np.zeros((self._nat, 3))
            force_offset = np.zeros((self._nat, 3))
        else:
            x0_offset, force_offset = self._get_coordinate_and_force_lammps(file_offset)
            try:
                x0_offset = np.reshape(x0_offset, (self._nat, 3))
                force_offset = np.reshape(force_offset, (self._nat, 3))
            except Exception as err:
                raise RuntimeError(
                    "File %s contains too many/few entries" % file_offset
                ) from err

            disp_offset = x0_offset - self._x_cartesian

        # Automatic detection of the input format

        is_dumped_file = False
        f = open(lammps_files[0], "r")
        for line in f:
            if "ITEM: TIMESTEP" in line:
                is_dumped_file = True
                break
        f.close()

        if is_dumped_file:
            # This version supports reading the data from MD trajectory
            for search_target in lammps_files:
                x, force = self._get_coordinate_and_force_lammps(search_target)
                ndata = len(x) // (3 * self._nat)
                x = np.reshape(x, (ndata, self._nat, 3))
                force = np.reshape(force, (ndata, self._nat, 3))

                for idata in range(ndata):
                    disp = x[idata, :, :] - self._x_cartesian - disp_offset
                    disp *= self._disp_conversion_factor
                    f = force[idata, :, :] - force_offset
                    f *= self._force_conversion_factor

                    print("# Filename: %s, Snapshot: %d" % (search_target, idata + 1))

                    for i in range(self._nat):
                        print(
                            "%20.14f %20.14f %20.14f %20.8E %15.8E %15.8E"
                            % (
                                disp[i, 0],
                                disp[i, 1],
                                disp[i, 2],
                                f[i, 0],
                                f[i, 1],
                                f[i, 2],
                            )
                        )
        else:
            raise RuntimeError(
                "Could not find ITEM: "
                "TIMESTEP keyword in the dump file %s" % lammps_files[0]
            )

    @staticmethod
    def _compute_lattice_vector_from_boxparams(box_params):
        xlo = box_params["xlo"]
        xhi = box_params["xhi"]
        ylo = box_params["ylo"]
        yhi = box_params["yhi"]
        zlo = box_params["zlo"]
        zhi = box_params["zhi"]
        if "xy" in box_params.keys():
            xy = box_params["xy"]
        if "xz" in box_params.keys():
            xz = box_params["xz"]
        if "yz" in box_params.keys():
            yz = box_params["yz"]

        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo
        a = lx
        b = math.sqrt(ly**2 + xy**2)
        c = math.sqrt(lz**2 + xz**2 + yz**2)
        cosalpha = (xy * xz + ly * yz) / (b * c)
        cosbeta = xz / c
        cosgamma = xy / b

        singamma = math.sqrt(1.0 - cosgamma**2)

        lavec = np.zeros((3, 3))

        lavec[0, 0] = a
        lavec[0, 1] = b * cosgamma
        lavec[1, 1] = b * singamma
        lavec[0, 2] = c * cosbeta
        lavec[1, 2] = c * (cosalpha - cosbeta * cosgamma) / singamma
        lavec[2, 2] = c * math.sqrt(
            1.0 - cosbeta**2 - ((cosalpha - cosbeta * cosgamma) / singamma) ** 2
        )

        return lavec

    @staticmethod
    def _get_fractional_coordinate(xc, aa_inv):
        if aa_inv is None:
            return None

        convmat = aa_inv.transpose()
        nat, _ = np.shape(xc)
        xf = np.zeros((nat, 3))

        for i in range(nat):
            xf[i] = np.dot(xc[i], convmat)

        return xf

    def _print_displacements(self, lammps_files, file_offset):
        if file_offset is None:
            disp_offset = np.zeros((self._nat, 3))
        else:
            x0_offset, _ = self._get_coordinate_and_force_lammps(file_offset)

            nentries = len(x0_offset)
            if nentries == 3 * self._nat:
                x0_offset = np.reshape(x0_offset, (self._nat, 3))
            else:
                raise RuntimeError(
                    "File %s contains too many/few entries" % file_offset
                )

            disp_offset = x0_offset - self._x_cartesian

        # Automatic detection of the input format

        is_dumped_file = False
        f = open(lammps_files[0], "r")
        for line in f:
            if "ITEM: TIMESTEP" in line:
                is_dumped_file = True
                break
        f.close()

        if is_dumped_file:
            # This version supports reading the data from MD trajectory
            for search_target in lammps_files:
                x, _ = self._get_coordinate_and_force_lammps(search_target)
                ndata = len(x) // (3 * self._nat)
                x = np.reshape(x, (ndata, self._nat, 3))

                for idata in range(ndata):
                    disp = x[idata, :, :] - self._x_cartesian - disp_offset
                    disp *= self._disp_conversion_factor

                    print("# Filename: %s, Snapshot: %d" % (search_target, idata + 1))

                    for i in range(self._nat):
                        print(
                            "%20.14f %20.14f %20.14f"
                            % (disp[i, 0], disp[i, 1], disp[i, 2])
                        )

        else:
            raise RuntimeError(
                "Could not find ITEM: "
                "TIMESTEP keyword in the dump file %s" % lammps_files[0]
            )

    def _print_atomicforces(self, lammps_files, file_offset):
        if file_offset is None:
            force_offset = np.zeros((self._nat, 3))
        else:
            _, force_offset = self._get_coordinate_and_force_lammps(file_offset)

            try:
                force_offset = np.reshape(force_offset, (self._nat, 3))
            except Exception as err:
                raise RuntimeError(
                    "File %s contains too many position entries" % file_offset
                ) from err

        for search_target in lammps_files:
            _, force = self._get_coordinate_and_force_lammps(search_target)
            ndata = len(force) // (3 * self._nat)
            force = np.reshape(force, (ndata, self._nat, 3))

            for idata in range(ndata):
                f = force[idata, :, :] - force_offset
                f *= self._force_conversion_factor

                print("# Filename: %s, Snapshot: %d" % (search_target, idata + 1))

                for i in range(self._nat):
                    print("%19.11E %19.11E %19.11E" % (f[i][0], f[i][1], f[i][2]))

    def _set_unit_conversion_factor(self, str_unit):
        if str_unit == "ev":
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 1.0

        elif str_unit == "rydberg":
            self._disp_conversion_factor = 1.0 / self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = 1.0 / self._RYDBERG_TO_EV

        elif str_unit == "hartree":
            self._disp_conversion_factor = 1.0 / self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = 0.5 / self._RYDBERG_TO_EV

        else:
            raise RuntimeError("This cannot happen")

        self._force_conversion_factor = (
            self._energy_conversion_factor / self._disp_conversion_factor
        )

    def _set_output_flags(self, output_flags):
        self._print_disp, self._print_force, self._print_energy, self._print_born = (
            output_flags
        )

    def _set_number_of_zerofill(self, npattern):
        nzero = 1

        while True:
            npattern //= 10
            if npattern == 0:
                break
            nzero += 1

        self._nzerofills = nzero

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

    @staticmethod
    def _get_coordinate_and_force_lammps(lammps_dump_file):
        add_flag = None
        ret = []

        with open(lammps_dump_file) as f:
            for line in f:
                if "ITEM:" in line:
                    if "ITEM: ATOMS id xu yu zu fx fy fz" in line:
                        add_flag = "id.xu"
                        continue
                    elif "ITEM: ATOMS element xu yu zu fx fy fz" in line:
                        add_flag = "element.xu"
                        id_ = 0
                        continue
                    else:
                        add_flag = None
                        continue

                if add_flag is not None:
                    if add_flag == "id.xu":
                        if line.strip():
                            entries = line.strip().split()
                            data_atom = [
                                int(entries[0]),
                                [float(t) for t in entries[1:4]],
                                [float(t) for t in entries[4:]],
                            ]
                            ret.append(data_atom)

                    elif add_flag == "element.xu":
                        if line.strip():
                            entries = line.strip().split()
                            id_ += 1
                            data_atom = [
                                int(id_),
                                [float(t) for t in entries[1:4]],
                                [float(t) for t in entries[4:]],
                            ]
                            ret.append(data_atom)

        # This sort is necessary since the order atoms of LAMMPS dump files
        # may change from the input structure file.

        ret_sorted = sorted(ret)
        ret_x = []
        ret_f = []
        for ret_atom in ret_sorted:
            ret_x.extend(ret_atom[1])
            ret_f.extend(ret_atom[2])

        return np.array(ret_x), np.array(ret_f)
