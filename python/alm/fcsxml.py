"""Python module for reading and writing alamode FCSXML."""

import itertools
import warnings
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
import spglib
from ase.units import Bohr

atom_names = (
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Uuq",
    "Uup",
    "Uuh",
    "Uus",
    "Uuo",
)


class Fcsxml(object):
    """Writer of harmonic and anharmonic interatomic force constants for alamode."""

    def __init__(self, lavec, xcoord, numbers, symprec=1.0e-3, unit='angstrom'):
        """Initialize the object with the input crystal structure.

        Parameters
        ----------
        lavec : array_like
            Basis vectors in units specified by 'unit'
            a, b, c are given as column vectors.
            shape=(3, 3), dtype='double'
        xcoord : array_like
            Fractional coordinates of atomic points.
            shape=(num_atoms, 3), dtype='double'
        numbers : array_like
            Atomic numbers.
            shape=(num_atoms,), dtype='intc'
        symprec: float
            The tolerance used for space group detection.
            Will be passed to spglib.

        """
        if unit == 'angstrom':
            self._lavec = np.array(lavec).transpose() / Bohr
        else:
            self._lavec = np.array(lavec).transpose()

        self._xf = np.array(xcoord)
        self._atomic_kinds = np.array(numbers)
        self._symprec = symprec
        self._generated_spacegroup = False
        self._xf_image = None
        self._xc_image = None
        self._map_p2s = None
        self._map_s2p = None
        self._nat_super = len(xcoord)
        self._nat_prim = None
        self._set_fcs = [False, False, False]
        self._fc2_info = None
        self._fc3_info = None
        self._fc4_info = None

        self._closest_mirror_images = None

        self._set_x_images()
        self._compute_closest_mirror_images()
        self._setup_symmetry()

    def set_force_constants(self, fc_values, fc_indices):
        """Set the force constants to write.

        Parameters
        ----------
        fc_values: array_like, dtype='float', shape=(num_fc)
            Array of force constant values to write in an alamode XML file.
            The unit of fc_values should be Rydberg/bohr**fc_order.

        fc_indices: array_like, dtype='int', shape=(num_fc, fc_order + 1)
            Array of flattened indices 3 * index_atom + index_xyz of the
            force constants.

        Note
        ----
        The expected input is all force constants in the supercell where
        the first column of fc_indices corresponds to the atoms in the
        primitive cell at the center. The `index_atom` of the atoms in the
        center primitive cell is determined from the output of spglib method,
        which can be obtained by the `get_atoms_in_primitive` method.

        The order of force constant is judged using the dimension of
        the input `fc_indices`.

        """
        if fc_indices.ndim != 2:
            msg = "elem_indices array has to be two dimensions."
            raise RuntimeError(msg)

        if len(fc_indices) != len(fc_values):
            msg = "length of fc_values and elem_indices must be the same."
            raise RuntimeError(msg)

        fc_order = len(fc_indices[0])
        (
            fc_values_trim,
            fc_indices_trim,
        ) = self._trim_and_sort_input_force_constants(fc_values, fc_indices)

        if fc_order == 2:
            self._fc2_info = [fc_values_trim, fc_indices_trim]

        elif fc_order == 3:
            self._fc3_info = [fc_values_trim, fc_indices_trim]

        elif fc_order == 4:
            self._fc4_info = [fc_values_trim, fc_indices_trim]

        else:
            msg = "The anharmonic terms beyond 4th-order cannot be saved."
            raise RuntimeError(msg)

    def write(self, filename="alamodefc.xml"):
        """Write the input force constants as a given filename.

        Parameters
        ----------
        filename : str
            The filename where the force constants are saved.
            default is "alamodefc.xml"

        """
        if self._fc2_info is None:
            warnings.warn(
                "The harmonic force constants are not set. "
                "Please use `set_force_constants` before calling write."
            )

        root = Element("Data")
        self._add_header_xml(root)
        self._add_structure_xml(root)
        self._add_symmetry_xml(root)
        self._add_force_constants_xml(root)
        self._pretty_print(root)
        tree = ElementTree(root)

        with open(filename, "wb") as file:
            tree.write(file, encoding="utf-8")

    def get_atoms_in_primitive(self):
        """Return the indices of atoms in the primitive cell.

        Returns
        -------
        atoms_in_primitive: array_like, dtype='int',
                            shape=(number_of_atoms_in_primitive)

        """
        if self._map_p2s is None:
            self._setup_symmetry()

        return self._map_p2s[:, 0]

    def _add_header_xml(self, root_in):

        elem = Element("ALM_version")
        elem.text = "None"
        root_in.append(elem)

    def _add_structure_xml(self, root_in):
        elem = Element("Structure")
        root_in.append(elem)
        subelem = SubElement(elem, "NumberOfAtoms")
        subelem.text = str(self._nat_super)
        atomic_kinds_unique = self._atomic_kinds[
            sorted(np.unique(self._atomic_kinds, return_index=True)[1])
        ]
        nkd = len(atomic_kinds_unique)
        subelem = SubElement(elem, "NumberOfElements")
        subelem.text = str(nkd)
        subelem = SubElement(elem, "AtomicElements")
        for i in range(nkd):
            subsubelem = SubElement(subelem, "element")
            subsubelem.set("number", str(i + 1))
            subsubelem.text = atom_names[atomic_kinds_unique[i]]
        subelem = SubElement(elem, "LatticeVector")
        for i in range(3):
            subsubelem = SubElement(subelem, "a%d" % (i + 1))
            str_avec = ""
            for j in range(3):
                str_avec += " " + "%.15e" % self._lavec[j][i]
            subsubelem.text = str_avec

        subelem = SubElement(elem, "Periodicity")
        subelem.text = "1 1 1"
        subelem = SubElement(elem, "Position")

        for i in range(self._nat_super):
            subsubelem = SubElement(subelem, "pos")
            subsubelem.set("index", str(i + 1))
            subsubelem.set("element", atom_names[self._atomic_kinds[i]])
            str_pos = ""
            for j in range(3):
                str_pos += " " + "%.15e" % self._xf[i][j]
            subsubelem.text = str_pos

    def _add_symmetry_xml(self, root_in):
        if self._map_p2s is None:
            self._setup_symmetry()

        elem = Element("Symmetry")
        root_in.append(elem)
        subelem = SubElement(elem, "NumberOfTranslations")
        ntrans = len(self._map_p2s[0, :])
        subelem.text = str(ntrans)
        subelem = SubElement(elem, "Translations")
        for i in range(ntrans):
            for j in range(self._nat_prim):
                subsubelem = SubElement(subelem, "map")
                subsubelem.set("tran", str(i + 1))
                subsubelem.set("atom", str(j + 1))
                subsubelem.text = str(self._map_p2s[j, i] + 1)

    def _add_force_constants_xml(self, root_in):

        elem = Element("ForceConstants")
        root_in.append(elem)
        if self._fc2_info:
            subelem = SubElement(elem, "HARMONIC")
            fcvals = self._fc2_info[0]
            clusters = self._fc2_info[1]
            for fcval, cluster in zip(fcvals, clusters):
                list_mir_imgs = self._closest_mirror_images[cluster[0] // 3][
                    cluster[1] // 3][0]
                num_mir_img = len(list_mir_imgs)
                fcval_scale = fcval / float(num_mir_img)
                for mir_img in list_mir_imgs:
                    subsubelem = SubElement(subelem, "FC2")
                    subsubelem.set(
                        "pair1",
                        str(self._map_s2p[cluster[0] // 3]["atom_num"][0] + 1)
                        + " "
                        + str(cluster[0] % 3 + 1),
                    )
                    subsubelem.set(
                        "pair2",
                        str(cluster[1] // 3 + 1)
                        + " "
                        + str(cluster[1] % 3 + 1)
                        + " "
                        + str(mir_img + 1),
                    )
                    subsubelem.text = "%.15e" % fcval_scale

        if self._fc3_info:
            self._add_anharmonic_fcs_to_treeelement(
                elem, self._fc3_info[1], self._fc3_info[0]
            )

        if self._fc4_info:
            self._add_anharmonic_fcs_to_treeelement(
                elem, self._fc4_info[1], self._fc4_info[0]
            )

    def _add_anharmonic_fcs_to_treeelement(self, elem_in, clusters, fcvals):
        fc_order = len(clusters[0])
        subelem = SubElement(elem_in, "ANHARM%d" % fc_order)
        subsubelem_key = "FC%d" % fc_order

        for fcval, cluster in zip(fcvals, clusters):
            if not self._is_sorted(cluster[1:]):
                continue
            atoms = cluster // 3
            xyz = cluster % 3 + 1
            mir_img_combs = self._get_mirror_image_combinations(atoms)
            num_mir_img = len(mir_img_combs)
            fcval_scale = fcval / float(num_mir_img)
            for mir_img in mir_img_combs:
                subsubelem = SubElement(subelem, subsubelem_key)
                subsubelem.set(
                    "pair1",
                    str(self._map_s2p[atoms[0]]["atom_num"][0] + 1) + " " + str(xyz[0]),
                )
                for i in range(1, fc_order):
                    subsubelem.set(
                        "pair%d" % (i + 1),
                        str(atoms[i] + 1)
                        + " "
                        + str(xyz[i])
                        + " "
                        + str(mir_img[i - 1] + 1),
                    )
                subsubelem.text = "%.15e" % fcval_scale

    def _get_mirror_image_combinations(self, atoms):
        unique_atoms = []
        indices_unique_counts = []
        iuniq = -1
        for atom in atoms[1:]:
            if atom not in unique_atoms:
                unique_atoms.append(atom)
                iuniq += 1
            indices_unique_counts.append(iuniq)

        mirror_image_combinations = []
        if len(unique_atoms) == 1:
            all_mir_img_comb = self._closest_mirror_images[atoms[0]][unique_atoms[0]][0]
            for icomb in range(len(all_mir_img_comb)):
                tmp_list = []
                for i in indices_unique_counts:
                    tmp_list.append(all_mir_img_comb[icomb])
                mirror_image_combinations.append(tmp_list)

        elif len(unique_atoms) == 2:
            all_mir_img_comb = list(
                itertools.product(
                    self._closest_mirror_images[atoms[0]][unique_atoms[0]][0],
                    self._closest_mirror_images[atoms[0]][unique_atoms[1]][0],
                )
            )
            for icomb in range(len(all_mir_img_comb)):
                tmp_list = []
                for i in indices_unique_counts:
                    tmp_list.append(all_mir_img_comb[icomb][i])
                mirror_image_combinations.append(tmp_list)

        elif len(unique_atoms) == 3:
            all_mir_img_comb = list(
                itertools.product(
                    self._closest_mirror_images[atoms[0]][unique_atoms[0]][0],
                    self._closest_mirror_images[atoms[0]][unique_atoms[1]][0],
                    self._closest_mirror_images[atoms[0]][unique_atoms[2]][0],
                )
            )
            for icomb in range(len(all_mir_img_comb)):
                tmp_list = []
                for i in indices_unique_counts:
                    tmp_list.append(all_mir_img_comb[icomb][i])
                mirror_image_combinations.append(tmp_list)

        return mirror_image_combinations

    def _pretty_print(self, current, parent=None, index=-1, depth=0):
        for i, node in enumerate(current):
            self._pretty_print(node, current, i, depth + 1)
        if parent is not None:
            if index == 0:
                parent.text = "\n" + ("  " * depth)
            else:
                parent[index - 1].tail = "\n" + ("  " * depth)
            if index == len(parent) - 1:
                current.tail = "\n" + ("  " * (depth - 1))

    def _set_x_images(self):

        self._xf_image = np.zeros((27, self._nat_super, 3), dtype="float")
        self._xf_image[0] = self._xf

        ishifts = [-1, 0, 1]
        icell = 0
        for ix, iy, iz in itertools.product(ishifts, ishifts, ishifts):
            if ix == 0 and iy == 0 and iz == 0:
                continue
            icell += 1
            self._xf_image[icell, :, 0] = self._xf[:, 0] + float(ix)
            self._xf_image[icell, :, 1] = self._xf[:, 1] + float(iy)
            self._xf_image[icell, :, 2] = self._xf[:, 2] + float(iz)

        self._xc_image = np.dot(self._xf_image, self._lavec.transpose())

    def _compute_closest_mirror_images(self):

        if self._xc_image is None:
            self._set_x_images()

        tolerance = 1.0e-3
        self._closest_mirror_images = []

        distances = np.zeros(27, dtype="float")
        for i in range(self._nat_super):
            mirror_image_tmp = []
            for j in range(self._nat_super):
                xdiff = self._xc_image[:, j, :] - self._xc_image[0, i, :]
                for icell in range(27):
                    distances[icell] = np.sqrt(np.dot(xdiff[icell, :], xdiff[icell, :]))

                index_sort = np.argsort(distances)
                minimum_dist = distances[index_sort[0]]

                mirror_image_tmp.append(
                    np.where(np.abs(distances[:] - minimum_dist) < tolerance)
                )

            self._closest_mirror_images.append(mirror_image_tmp)

    def _setup_symmetry(self):

        cell = (self._lavec.transpose(), self._xf, self._atomic_kinds)
        dataset = spglib.get_symmetry_dataset(cell, symprec=self._symprec)

        identity_matrix = np.identity(3)
        rotations = dataset["rotations"]
        translations = dataset["translations"]
        mapping_to_primitive = dataset["mapping_to_primitive"]
        nsym = len(dataset["rotations"])

        # Sort the list of symmetry operations in the lexicographical order
        # and create symnum_translation
        symmetry_ops = np.zeros((nsym, 12), dtype="float")
        symmetry_ops[:, :9] = np.reshape(rotations, (nsym, 9))
        symmetry_ops[:, 9:] = translations[:, :]
        for isym in range(nsym):
            for i in range(3):
                if symmetry_ops[isym, 9 + i] < 0.0:
                    symmetry_ops[isym, 9 + i] += 1.0

        index_sort = np.lexsort(np.rot90(symmetry_ops))
        symnum_translation = []

        for elem in index_sort:
            if np.array_equal(rotations[elem], identity_matrix):
                symnum_translation.append(elem)

        self._nat_prim = self._nat_super // len(symnum_translation)
        self._map_p2s = (np.ones((self._nat_prim,
                                  len(symnum_translation)), dtype="int") * -1)

        unique_set, atom_in_primitive_at_origin = np.unique(
            mapping_to_primitive, return_index=True
        )

        for index_p, index_s in enumerate(atom_in_primitive_at_origin):
            for itran, isym in enumerate(symnum_translation):
                xnew = self._xf[index_s, :] + translations[isym][:]

                for iat in np.where(mapping_to_primitive == unique_set[index_p])[0]:
                    xdiff = xnew - self._xf[iat]
                    xdiff -= np.array(np.rint(xdiff), dtype="float")

                    if np.dot(xdiff, xdiff) < 1.0e-5:
                        self._map_p2s[index_p][itran] = iat
                        break

        mytype = np.dtype(
            [
                ("atom_num", int, (1,)),
                ("tran_num", int, (1,)),
            ],
            align=True,
        )
        self._map_s2p = np.zeros(self._nat_super, dtype=mytype)
        for iat in range(self._nat_prim):
            for itran in range(len(symnum_translation)):
                jat = self._map_p2s[iat][itran]
                self._map_s2p[jat]["atom_num"] = iat
                self._map_s2p[jat]["tran_num"] = itran

    def _trim_and_sort_input_force_constants(self, fcvals_in, elems_in):
        atom_in_primitive_at_origin = self._map_p2s[:, 0]
        indices_to_search = []
        for atom in atom_in_primitive_at_origin:
            for i in range(3):
                indices_to_search.append(3 * atom + i)
        indices_to_search = np.array(indices_to_search)

        loc_to_use = np.where(np.in1d(elems_in[:, 0], indices_to_search))
        fcvals_trim = fcvals_in[loc_to_use]
        elems_trim = elems_in[loc_to_use]
        index_sort = np.lexsort(np.rot90(elems_trim))

        return fcvals_trim[index_sort], elems_trim[index_sort]

    @staticmethod
    def _is_sorted(a):
        return np.all(a[:-1] <= a[1:])
