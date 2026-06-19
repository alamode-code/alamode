#
# VASP.py
#
# Interface to VASP (https://www.vasp.at)
#
# Copyright (c) 2014-2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
from __future__ import print_function

from importlib.util import find_spec

import numpy as np

try:
    # Try to import lxml
    from lxml import etree

    USING_LXML = True
except ModuleNotFoundError:
    USING_LXML = False
    try:
        # If lxml is not available, try to import ElementTree
        import xml.etree.ElementTree as etree
    except ModuleNotFoundError:
        # If failed, try cElementTree next
        import elementtree.ElementTree as etree

try:
    import py4vasp
except ModuleNotFoundError:
    pass


class VaspParser(object):
    def __init__(self):
        self._support_h5parse = find_spec("py4vasp") is not None
        self._prefix = None
        self._lattice_vector = None
        self._inverse_lattice_vector = None
        self._elements = None
        self._kd = None
        self._kd_dict = None
        self._nat_elem = None
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
        self._BOHR_TO_ANGSTROM = 0.529177210903
        self._RYDBERG_TO_EV = 13.60569253
        self._vca_mode = False
        self._vca_duplite_sites = None

    def set_vca_mode(self, vca_mode):
        self._vca_mode = vca_mode

    def load_initial_structure(self, file_in):
        with open(file_in, "r") as file_pos:
            file_pos.readline()
            a = float(file_pos.readline().rstrip())
            lavec = np.zeros((3, 3))
            for i in range(3):
                arr = file_pos.readline().rstrip().split()
                if len(arr) != 3:
                    raise RuntimeError("Could not read POSCAR properly")
                for j in range(3):
                    lavec[i, j] = a * float(arr[j])

            lavec = lavec.transpose()
            invlavec = np.linalg.inv(lavec)
            elements = file_pos.readline().rstrip().split()

            if elements[0].isdigit():
                nat_elem = [int(tmp) for tmp in elements]
                elements = []
            else:
                nat_elem = [int(tmp) for tmp in file_pos.readline().rstrip().split()]

            nat = np.sum(nat_elem)
            basis = file_pos.readline().strip()
            x = np.zeros((nat, 3))

            for i in range(nat):
                arr = file_pos.readline().rstrip().split()
                for j in range(3):
                    x[i][j] = float(arr[j])

            if basis[0] == "D" or basis[0] == "d":
                xf = x
            else:
                xf = np.dot(x, invlavec.transpose())

            if len(elements) == 0:
                kd_dict = {}
                for i in range(len(nat_elem)):
                    kd_dict["Element {}".format(i + 1)] = i
                kd = []
                for i in range(len(nat_elem)):
                    kd.extend([i] * nat_elem[i])
            else:
                kd_dict = {}
                icount_elem = 0
                for elem in elements:
                    if elem not in kd_dict:
                        kd_dict[elem] = icount_elem
                        icount_elem += 1

                kd = []
                for i, elem in enumerate(elements):
                    kd.extend([kd_dict[elem]] * nat_elem[i])

        if self._vca_mode:
            nat = np.sum(nat_elem)
            flag_scanned = np.zeros(nat, dtype=int)
            list_vca_duplite_sites = []
            for i in range(nat):
                if flag_scanned[i] == 0:
                    list_samesite_tmp = [i]
                    flag_scanned[i] = 1
                    for j in range(i + 1, nat):
                        if flag_scanned[j] == 0:
                            if np.allclose(xf[i], xf[j]):
                                list_samesite_tmp.append(j)
                                flag_scanned[j] = 1

                    list_vca_duplite_sites.append(list_samesite_tmp)

            self._vca_duplite_sites = list_vca_duplite_sites

        self._lattice_vector = lavec
        self._inverse_lattice_vector = invlavec
        self._elements = elements
        self._kd = np.array(kd, dtype=int)
        self._kd_dict = kd_dict
        self._nat_elem = nat_elem
        self._nat = np.sum(nat_elem)
        self._x_fractional = xf
        self._initial_structure_loaded = True

    def update_initial_structure(self, structure_dict):
        print(structure_dict)

    def generate_structures(
        self, prefix, header_list, disp_list, updated_structure=None
    ):
        self._set_number_of_zerofill(len(disp_list))
        self._prefix = prefix
        self._counter = 1

        if updated_structure:
            for header, disp in zip(header_list, disp_list):
                self._generate_input2(header, disp, updated_structure)
            self._generate_original_supercell(updated_structure)
        else:
            for header, disp in zip(header_list, disp_list):
                self._generate_input(header, disp)

    def parse(
        self,
        initial_poscar,
        target_files,
        offset_file,
        str_unit,
        output_flags,
        filter_emin=None,
        filter_emax=None,
    ):
        if not self._initial_structure_loaded:
            self.load_initial_structure(initial_poscar)

        self._set_unit_conversion_factor(str_unit)
        self._set_output_flags(output_flags)

        if self._print_disp or self._print_force:
            self._print_displacements_and_forces(
                target_files, offset_file, filter_emin, filter_emax
            )
        elif self._print_energy:
            self._print_energies(target_files, offset_file)

        elif self._print_born:
            self._print_borninfo(target_files)

    def get_displacements(self, target_files, unit="bohr"):
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

        for search_target in target_files:
            x, _ = self._get_coordinates_and_forces(search_target)

            ndata = len(x) // (3 * self._nat)
            x = np.reshape(x, (ndata, self._nat, 3))
            disp = np.zeros((ndata, self._nat, 3))

            for idata in range(ndata):
                disp[idata, :, :] = x[idata, :, :] - x0
                disp[idata, :, :] = np.dot(
                    vec_refold(disp[idata, :, :]), lavec_transpose
                )
                disp[idata, :, :] *= unit_factor

            disp_merged.extend(disp)

        return disp_merged

    def _generate_input(self, header, disp, save_supercell_structure=True):
        filename = self._prefix + str(self._counter).zfill(self._nzerofills) + ".POSCAR"

        with open(filename, "w") as f:
            f.write("%s\n" % header)
            f.write("%s\n" % "1.0")
            for i in range(3):
                f.write(
                    "%20.15f %20.15f %20.15f\n"
                    % (
                        self._lattice_vector[0][i],
                        self._lattice_vector[1][i],
                        self._lattice_vector[2][i],
                    )
                )

            for i in range(len(self._elements)):
                f.write("%s " % self._elements[i])
            if len(self._elements) > 0:
                f.write("\n")

            for i in range(len(self._nat_elem)):
                f.write("%d " % self._nat_elem[i])
            f.write("\n")

            f.write("Direct\n")

            for i in range(len(disp)):
                for j in range(3):
                    f.write("%20.15f" % (self._x_fractional[i][j] + disp[i][j]))
                f.write("\n")

        self._counter += 1

    def _generate_input2(self, header, disp, updated_structure):
        filename = self._prefix + str(self._counter).zfill(self._nzerofills) + ".POSCAR"
        lavec = updated_structure["lattice_vector"] * self._BOHR_TO_ANGSTROM
        kd = updated_structure["kd"]
        x_fractional = updated_structure["x_fractional"]

        with open(filename, "w") as f:
            f.write("%s\n" % header)
            f.write("%s\n" % "1.0")
            for i in range(3):
                f.write(
                    "%20.15f %20.15f %20.15f\n"
                    % (lavec[0][i], lavec[1][i], lavec[2][i])
                )

            for i in range(len(self._elements)):
                f.write("%s " % self._elements[i])
            if len(self._elements) > 0:
                f.write("\n")

            counts = []
            if len(kd) > 0:
                current_value = kd[0]
                current_count = 1

                for i in range(1, len(kd)):
                    if kd[i] == current_value:
                        current_count += 1
                    else:
                        counts.append(current_count)
                        current_value = kd[i]
                        current_count = 1

                counts.append(current_count)

            for count in counts:
                f.write("%d " % count)
            f.write("\n")

            f.write("Direct\n")

            for i in range(len(disp)):
                for j in range(3):
                    f.write("%20.15f" % (x_fractional[i][j] + disp[i][j]))
                f.write("\n")

        self._counter += 1

    def _generate_original_supercell(self, structure):
        filename = "SPOSCAR0"
        lavec = structure["lattice_vector"] * self._BOHR_TO_ANGSTROM
        kd = structure["kd"]
        x_fractional = structure["x_fractional"]

        with open(filename, "w") as f:
            f.write("Supercell without displacements\n")
            f.write("%s\n" % "1.0")
            for i in range(3):
                f.write(
                    "%20.15f %20.15f %20.15f\n"
                    % (lavec[0][i], lavec[1][i], lavec[2][i])
                )

            for elem in self._elements:
                f.write("%s " % elem)
            if len(self._elements) > 0:
                f.write("\n")

            counts = []
            if len(kd) > 0:
                current_value = kd[0]
                current_count = 1

                for i in range(1, len(kd)):
                    if kd[i] == current_value:
                        current_count += 1
                    else:
                        counts.append(current_count)
                        current_value = kd[i]
                        current_count = 1

                counts.append(current_count)

            for count in counts:
                f.write("%d " % count)
            f.write("\n")

            f.write("Direct\n")

            for i in range(len(x_fractional)):
                for j in range(3):
                    f.write("%20.15f" % x_fractional[i][j])
                f.write("\n")

    def _print_displacements_and_forces(
        self, target_files, file_offset, filter_emin, filter_emax
    ):
        x0 = np.round(self._x_fractional, 8)
        lavec_transpose = self._lattice_vector.transpose()
        vec_refold = np.vectorize(self._refold)

        if file_offset is None:
            disp_offset = np.zeros((self._nat, 3))
            force_offset = np.zeros((self._nat, 3))
            epot_offset = 0.0

        else:
            x0_offset, force_offset = self._get_coordinates_and_forces(file_offset)
            epot_offset, _ = self._get_energies(file_offset)
            epot_offset = np.array(epot_offset, dtype=float)
            try:
                x0_offset = np.reshape(x0_offset, (self._nat, 3))
            except ValueError as err:
                raise RuntimeError(
                    "File %s contains too many position entries" % file_offset
                ) from err

            try:
                force_offset = np.reshape(force_offset, (self._nat, 3))
            except ValueError as err:
                raise RuntimeError(
                    "File %s contains too many force entries" % file_offset
                ) from err

            disp_offset = x0_offset - x0
            # Apply refold to handle periodic boundary conditions properly (not necessary?)
            disp_offset = vec_refold(disp_offset)

            if len(epot_offset) > 1:
                raise RuntimeError(
                    "File %s contains too many energy entries" % file_offset
                )

        for search_target in target_files:
            x, force = self._get_coordinates_and_forces(search_target)
            epot, _ = self._get_energies(search_target)

            ndata = len(x) // (3 * self._nat)
            ndata2 = len(force) // (3 * self._nat)

            if (ndata != ndata2) and self._print_disp and self._print_force:
                raise RuntimeError(
                    "The numbers of displacement and force entries are different."
                )

            ndata_energy = len(epot)
            if ndata_energy != ndata:
                raise RuntimeError(
                    "The numbers of displacement and energy entries are different."
                )

            epot = np.array(epot, dtype=float)
            epot -= epot_offset

            if self._print_disp:
                x = np.reshape(x, (ndata, self._nat, 3))

            if self._print_force:
                force = np.reshape(force, (ndata, self._nat, 3))

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

                if self._vca_mode:
                    nat_vca = len(self._vca_duplite_sites)
                    disp_vca = np.zeros((nat_vca, 3))
                    f_vca = np.zeros((nat_vca, 3))
                    for i in range(nat_vca):
                        disp_vca[i, :] = disp[self._vca_duplite_sites[i][0], :]
                        f_vca[i, :] = np.sum(f[self._vca_duplite_sites[i], :], axis=0)

                    if self._print_disp and self._print_force:
                        for i in range(nat_vca):
                            print(
                                "{:15.7F} {:15.7F} {:15.7F} {:20.8E} {:15.8E} {:15.8E}".format(
                                    *disp_vca[i], *f_vca[i]
                                )
                            )

                    elif self._print_disp:
                        for i in range(nat_vca):
                            print("{:15.7F} {:15.7F} {:15.7F}".format(*disp_vca[i]))

                    elif self._print_force:
                        for i in range(nat_vca):
                            print("{:15.8E} {:15.8E} {:15.8E}".format(*f_vca[i]))

                else:
                    if self._print_disp and self._print_force:
                        for i in range(self._nat):
                            print(
                                "{:15.7F} {:15.7F} {:15.7F} {:20.8E} {:15.8E} {:15.8E}".format(
                                    *disp[i], *f[i]
                                )
                            )

                    elif self._print_disp:
                        for i in range(self._nat):
                            print("{:15.7F} {:15.7F} {:15.7F}".format(*disp[i]))

                    elif self._print_force:
                        for i in range(self._nat):
                            print("{:15.8E} {:15.8E} {:15.8E}".format(*f[i]))

    def _print_energies(self, xml_files, file_offset):
        print("# Etot, Ekin")

        etot_offset = 0.0
        ekin_offset = 0.0

        if file_offset:
            etot, ekin = self._get_energies(file_offset)
            if len(etot) > 1 or len(ekin) > 1:
                print("File %s contains too many energy entries" % file_offset)
                exit(1)
            if etot[0] != "N/A":
                etot_offset = float(etot[0])
            if ekin[0] != "N/A":
                ekin_offset = float(ekin[0])

        for search_target in xml_files:
            etot, ekin = self._get_energies(search_target)

            for i in range(len(etot)):
                if etot[i] != "N/A":
                    val_etot = float(etot[i]) - etot_offset
                    print(
                        "%15.8E" % (val_etot * self._energy_conversion_factor), end=" "
                    )
                else:
                    print("%s" % etot[i], end=" ")

                if ekin[i] != "N/A":
                    val_ekin = float(ekin[i]) - ekin_offset
                    print("%15.8E" % (val_ekin * self._energy_conversion_factor))
                else:
                    print("%s" % ekin[i])

    def _print_borninfo(self, target_files):
        for search_target in target_files:
            dielec, borncharge = self._get_borninfo(search_target)
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
            self._disp_conversion_factor = 1.0
            self._energy_conversion_factor = 1.0

        elif str_unit == "rydberg":
            self._disp_conversion_factor = 1.0 / self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = 1.0 / self._RYDBERG_TO_EV

        elif str_unit == "hartree":
            self._disp_conversion_factor = 1.0 / self._BOHR_TO_ANGSTROM
            self._energy_conversion_factor = 0.5 / self._RYDBERG_TO_EV

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

    @staticmethod
    def _refold(x):
        if x >= 0.5:
            return x - 1.0
        elif x < -0.5:
            return x + 1.0
        else:
            return x

    @staticmethod
    def parse_or_repair_xml_file(file_path):
        if USING_LXML:
            try:
                # Attempt to parse the XML file directly with lxml
                tree = etree.parse(file_path)
                return tree
            except etree.XMLSyntaxError:
                # If a parsing error occurs with lxml,
                # attempt to repair by re-reading the file with the 'recover' parser
                repair_parser = etree.XMLParser(recover=True)
                tree = etree.parse(file_path, parser=repair_parser)
                return tree
        else:
            # Fallback to use xml.etree.ElementTree if lxml is not available
            try:
                # Attempt to parse the XML file directly
                tree = etree.parse(file_path)
                return tree
            except etree.ParseError as e:
                print(f"Failed to parse XML file with ElementTree due to: {e}")
                print("Consider installing lxml for better XML parsing support.")
                # Handle the error or attempt a manual repair if necessary
                # Note: ElementTree doesn't have a built-in 'recover' mode like lxml,
                # so you may need to manually fix the XML or use a different strategy

    def _get_coordinates_and_forces(self, file_to_parse):
        hdf5_mode = file_to_parse.lower().split(".")[-1] in ["h5", "hdf5"]

        if hdf5_mode:
            # target file is HDF5 format

            if not self._support_h5parse:
                raise RuntimeError(
                    "failed to import py4vasp. Please install py4vasp by pip."
                )

            try:
                obj = py4vasp.Calculation.from_file(file_to_parse)
                forces = obj.force[:].read()
                x = np.ravel(forces["structure"]["positions"])
                f = np.ravel(forces["forces"])
                return x, f

            except ValueError as err:
                raise RuntimeError(
                    "Error in reading atomic positions and "
                    "forces from the HDF5 file: %s" % file_to_parse
                ) from err

        else:
            x = []
            f = []
            # assume that the target file is XML and use XML parser
            try:
                xml = self.parse_or_repair_xml_file(file_to_parse)
                root = xml.getroot()

                for elems in root.findall("calculation/structure/varray"):
                    str_coord = [elems2.text for elems2 in elems.findall("v")]
                    n = len(str_coord)

                    for i in range(n):
                        x.extend([t for t in str_coord[i].split()])

                for elems in root.findall("calculation/varray"):
                    if elems.get("name") == "forces":
                        str_force = [elems2.text for elems2 in elems.findall("v")]

                        for i in range(len(str_force)):
                            f.extend([t for t in str_force[i].split()])

                return np.array(x, dtype=float), np.array(f, dtype=float)

            except ValueError as err:
                raise RuntimeError(
                    "Error in reading atomic positions and "
                    "forces from the XML file: %s" % file_to_parse
                ) from err

    def _get_energies(self, file_to_parse):
        hdf5_mode = file_to_parse.lower().split(".")[-1] in ["h5", "hdf5"]

        if hdf5_mode:
            if not self._support_h5parse:
                raise RuntimeError(
                    "failed to import py4vasp. Please install py4vasp by pip."
                )

            try:
                obj = py4vasp.Calculation.from_file(file_to_parse)
                energy = obj.energy[:].read()

                etot_array = energy["free energy    TOTEN"]
                # There are no methods to parse kinetic energy implemented in py4vasp
                # TODO: fix here if py4vasp implements it.
                ekin_array = ["N/A"] * len(etot_array)

                return etot_array, ekin_array
            except ValueError as err:
                raise RuntimeError(
                    "Error in reading atomic positions and "
                    "forces from the HDF5 file: %s" % file_to_parse
                ) from err

        else:
            etot_array = []
            ekin_array = []

            try:
                xml = self.parse_or_repair_xml_file(file_to_parse)
                root = xml.getroot()

                for elems in root.findall("calculation/energy"):
                    etot = "N/A"
                    ekin = "N/A"

                    for elems2 in elems.findall("i"):
                        if elems2.get("name") == "e_fr_energy":
                            etot = elems2.text
                        if elems2.get("name") == "kinetic":
                            ekin = elems2.text

                    etot_array.append(etot)
                    ekin_array.append(ekin)

                return etot_array, ekin_array
            except ValueError as err:
                raise RuntimeError(
                    "Error in reading energies from the XML file: %s" % file_to_parse
                ) from err

    def _get_borninfo(self, file_to_parse):
        hdf5_mode = file_to_parse.lower().split(".")[-1] in ["h5", "hdf5"]

        if hdf5_mode:
            if not self._support_h5parse:
                raise RuntimeError(
                    "failed to import py4vasp. Please install py4vasp by pip."
                )

            try:
                # use raw method instead of Calculation because latter raises an error
                # when trying to parse electronic dielectric tensor alone.
                # TODO: clean up this part when py4vasp support sole parse of epsion(∞)
                raw = py4vasp.raw.File(file_to_parse)
                obj = py4vasp.raw.RawDielectricTensor(
                    raw._h5f["results/linear_response/electron_dielectric_tensor"],
                    ion=None,
                    independent_particle=None,
                    method=raw._h5f["results/linear_response/method_dielectric_tensor"][
                        ()
                    ],
                )
                dielec_tensor_elec = obj.electron[:]
            except ValueError as err:
                raise RuntimeError(
                    "Error in reading electronic dielectric tensor from the HDF5 file: %s"
                    % file_to_parse
                ) from err

            try:
                obj = py4vasp.Calculation.from_file(file_to_parse)
                borncharge = obj.born_effective_charge.read()["charge_tensors"]
            except ValueError as err:
                raise RuntimeError(
                    "Error in reading Born effective charges from the HDF5 file: %s"
                    % file_to_parse
                ) from err

            return dielec_tensor_elec, borncharge

        else:
            dielec = []
            borncharge = []

            try:
                xml = self.parse_or_repair_xml_file(file_to_parse)
                root = xml.getroot()

                for elems in root.findall("calculation/varray"):
                    if elems.get("name") in ["epsilon", "epsilon_scf"]:
                        str_tmp = [elems2.text for elems2 in elems.findall("v")]

                        for i in range(len(str_tmp)):
                            dielec.extend([float(t) for t in str_tmp[i].split()])

                for elems in root.findall("calculation/array"):
                    if elems.get("name") == "born_charges":
                        for elems2 in elems.findall("set"):
                            str_tmp = [elems3.text for elems3 in elems2.findall("v")]

                            for i in range(len(str_tmp)):
                                borncharge.extend(
                                    [float(t) for t in str_tmp[i].split()]
                                )

                nat = len(borncharge) // 9
                dielec = np.reshape(np.array(dielec), (3, 3))
                borncharge = np.reshape(np.array(borncharge), (nat, 3, 3))
                return dielec, borncharge
            except ValueError as err:
                raise RuntimeError(
                    "Error in reading Born charges from the XML file: %s"
                    % file_to_parse
                ) from err
