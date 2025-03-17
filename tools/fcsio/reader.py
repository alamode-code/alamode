import itertools

import h5py
import numpy as np
import spglib
from ase.data import atomic_numbers
from lxml import etree


class ForceConstantParser:
    """
    A parser for reading force constants from files, supporting XML format.

    Attributes:
        filename (str): The path to the file containing the force constants.
        format (str): The format of the file, currently only 'xml' is supported.
        supercell (tuple): Contains the lattice vectors, fractional coordinates, and atomic numbers of the supercell.
        primitive_cell (tuple): Contains the lattice vectors, fractional coordinates, and atomic numbers of the primitive cell.
        root (etree.Element): The root element of the parsed XML file.
        map_p2s (numpy.ndarray): Mapping from primitive to supercell atom indices.
        map_s2p (numpy.ndarray): Mapping from supercell to primitive atom indices.
        transformation_matrix (numpy.ndarray): Matrix transforming between primitive and supercell lattice vectors.
        transformation_matrix_int (numpy.ndarray): Integer version of the transformation matrix.
        shift_vector (numpy.ndarray): Shift vector used in the transformation.

    Methods:
        __init__(self, filename, format='xml'): Initializes the parser with the file path and format.
        read(self, maxorder=4): Reads the force constants from the file up to a specified maximum order.
        _parse_structure(self): Parses the structural information from the file.
        _parse_xml(self): Parses the XML file and handles errors.
        _parse_structure_xml(self): Parses structural information specific to the XML format.
        _get_shift_vector_supercell(self): Calculates shift vectors for the supercell.
        _get_coordinate_after_transformation(self, xf_orig): Transforms coordinates based on the transformation matrix.
        _get_forceconstants_xml(self, maxorder=4): Parses force constants from an XML file.
    """

    def __init__(self, filename, format=None):
        """
        Initializes the ForceConstantParser with the specified file and format.

        Parameters:
            filename (str): The path to the file containing the force constants.
            format (str): The format of the file, defaults to None,
                          where the formation is inferred from the file suffix.
        """
        self.filename = filename
        if format is None:
            if filename.endswith(".xml") or filename.endswith(".XML"):
                self.format = "xml"
            elif filename.endswith(".h5") or filename.endswith(".hdf5"):
                self.format = "hdf5"
            else:
                raise ValueError("The format of the file is not supported.")
        self.supercell = None
        self.primitive_cell = None

        self.root = None
        self.map_p2s = None
        self.map_s2p = None
        self.transformation_matrix = None
        self.transformation_matrix_int = None
        self.shift_vector = None

        self._parse_structure()

    def read(self, maxorder=4):
        """
        Reads the force constants from the file up to a specified maximum order.

        Parameters:
            maxorder (int): The maximum order of force constants to read.

        Returns:
            dict: A dictionary containing the parsed force constants.
        """
        if self.format == "xml":
            return self._get_forceconstants_xml(maxorder=maxorder)
        elif self.format == "hdf5":
            return self._get_forceconstants_h5(maxorder=maxorder)

    def _parse_structure(self):
        """
        Parses the structural information from the file.
        """
        if self.format == "xml":
            self._parse_structure_xml()
        elif self.format == "hdf5":
            self._parse_structure_h5()

        lavec_super = self.supercell[0]
        lavec_prim = self.primitive_cell[0]
        self.transformation_matrix = np.dot(lavec_super, np.linalg.inv(lavec_prim))
        self.transformation_matrix_int = np.round(self.transformation_matrix).astype(
            int
        )

        if (
            np.abs(self.transformation_matrix - self.transformation_matrix_int).max()
            > 1.0e-3
        ):
            raise RuntimeError("The transformation matrix is not integer.")

        xf_tmp, self.shift_vectors = self._get_coordinate_after_transformation(
            self.supercell[1]
        )
        xf_prim_alm = xf_tmp[self.map_p2s[0, :], :]
        if np.abs(self.primitive_cell[1] - xf_prim_alm).max() > 1.0e-3:
            raise RuntimeError(
                "The primitive cell coordinates from spglib and FCS file does not match."
            )

    def _parse_xml(self):
        """
        Parses the XML file and handles errors.

        Returns:
            etree._ElementTree: The parsed XML tree.
        """
        try:
            # Attempt to parse the XML file directly with lxml
            tree = etree.parse(self.filename)
            return tree
        except etree.XMLSyntaxError:
            # If a parsing error occurs with lxml,
            # attempt to repair by re-reading the file with the 'recover' parser
            repair_parser = etree.XMLParser(recover=True)
            tree = etree.parse(self.filename, parser=repair_parser)
            return tree

    def _parse_structure_xml(self):
        """
        Parses structural information specific to the XML format.
        """
        if self.root is None:
            self.root = self._parse_xml().getroot()

        natom_super = int(self.root.find("Structure/NumberOfAtoms").text)
        ntrans = int(self.root.find("Symmetry/NumberOfTranslations").text)
        natom_prim = natom_super // ntrans

        vec1_super = [
            float(t) for t in self.root.find("Structure/LatticeVector/a1").text.split()
        ]
        vec2_super = [
            float(t) for t in self.root.find("Structure/LatticeVector/a2").text.split()
        ]
        vec3_super = [
            float(t) for t in self.root.find("Structure/LatticeVector/a3").text.split()
        ]
        lavec_super = np.array([vec1_super, vec2_super, vec3_super])

        xf_super = np.zeros((natom_super, 3), dtype=float)
        numbers_super = np.zeros(natom_super, dtype=int)
        for i, elems in enumerate(self.root.findall("Structure/Position/pos")):
            xf_super[i, :] = [float(t) for t in elems.text.split()]
            numbers_super[i] = atomic_numbers[elems.get("element")]

        self.supercell = (lavec_super, xf_super, numbers_super)

        self.primitive_cell = spglib.standardize_cell(
            self.supercell, to_primitive=True, no_idealize=True, symprec=1.0e-3
        )

        self.map_p2s = np.zeros((ntrans, natom_prim), dtype=int)
        self.map_s2p = np.zeros(natom_super, dtype=int)
        for _, elems in enumerate(self.root.findall("Symmetry/Translations/map")):
            itran = int(elems.get("tran")) - 1
            iatom = int(elems.get("atom")) - 1
            self.map_p2s[itran, iatom] = int(elems.text) - 1
            self.map_s2p[self.map_p2s[itran, iatom]] = iatom

    def _parse_structure_h5(self):
        """
        Parses structural information specific to the HDF5 format.
        """
        with h5py.File(self.filename, "r") as f:
            lavec_tmp = f["/SuperCell/lattice_vector"][:]
            xf_tmp = f["/SuperCell/fractional_coordinate"][:]
            kinds = f["/SuperCell/atomic_kinds"][:].astype(int)
            elems = f["/SuperCell/elements"][:].astype(str)
            numbers = np.array([atomic_numbers[elems[i]] for i in kinds])
            self.supercell = (lavec_tmp, xf_tmp, numbers)
            natom_super = int(f["/SuperCell/number_of_atoms"][()])
            lavec_tmp = f["/PrimitiveCell/lattice_vector"][:]
            xf_tmp = f["/PrimitiveCell/fractional_coordinate"][:]
            kinds = f["/PrimitiveCell/atomic_kinds"][:].astype(int)
            elems = f["/PrimitiveCell/elements"][:].astype(str)
            numbers = np.array([atomic_numbers[elems[i]] for i in kinds])
            cell_tmp = (lavec_tmp, xf_tmp, numbers)
            self.primitive_cell = cell_tmp
            # self.primitive_cell = spglib.standardize_cell(cell_tmp,
            #                                               to_primitive=True,
            #                                               no_idealize=True,
            #                                               symprec=1.0e-3)
            self.map_p2s = f["/SuperCell/mapping_table"][:].astype(int).T
            self.map_s2p = np.zeros(natom_super, dtype=int)
            for i, j in enumerate(self.map_p2s):
                self.map_s2p[j] = i

    def _get_shift_vector_supercell(self):
        """
        Calculates shift vectors for the supercell.

        Returns:
            numpy.ndarray: The calculated shift vectors.
        """

        shift_super = np.zeros((27, 3), dtype=int)

        icount = 1
        for ix, iy, iz in itertools.product([-1, 0, 1], repeat=3):
            if ix == 0 and iy == 0 and iz == 0:
                continue
            shift_super[icount, :] = [ix, iy, iz]
            icount += 1

        return shift_super

    def _get_coordinate_after_transformation(self, xf_orig):
        """
        Transforms coordinates based on the transformation matrix.

        Parameters:
            xf_orig (numpy.ndarray): The original fractional coordinates.

        Returns:
            tuple: Transformed coordinates and shift vectors.
        """
        positions_tmp = np.dot(xf_orig, self.transformation_matrix)
        rounded_array = np.round(positions_tmp)
        shift_vectors = np.where(
            np.abs(positions_tmp - rounded_array) < 1.0e-3,
            rounded_array,
            np.floor(positions_tmp),
        ).astype(int)
        positions_sub = positions_tmp - shift_vectors

        return positions_sub, shift_vectors

    def _get_forceconstants_xml(self, maxorder=4):
        """
        Parses force constants from an XML file.

        Parameters:
            maxorder (int): The maximum order of force constants to read.

        Returns:
            dict: A dictionary containing the parsed force constants.
        """
        if self.root is None:
            self.root = self._parse_xml().getroot()

        shift_super = self._get_shift_vector_supercell()
        shift_super = np.dot(shift_super, self.transformation_matrix_int)

        fcs_dic = {}

        for order in range(2, maxorder + 1):
            if order == 2:
                search_tag = "ForceConstants/HARMONIC/FC2"
            else:
                search_tag = "ForceConstants/ANHARM{:d}/FC{:d}".format(order, order)

            entries = self.root.findall(search_tag)
            n_entries = len(entries)

            if n_entries == 0:
                continue

            atom_indices = np.zeros((n_entries, order), dtype=int)
            atom_indices_mod = np.zeros((n_entries, order), dtype=int)
            index_shift_super = np.zeros((n_entries, order - 1), dtype=int)
            xyz_indices = np.zeros((n_entries, order), dtype=int)
            fcs_entries = np.zeros(n_entries, dtype=float)

            for i, elems in enumerate(entries):
                atom1, xyz1 = [int(t) - 1 for t in elems.get("pair1").split()]
                atom_indices[i, 0] = atom1
                xyz_indices[i, 0] = xyz1
                for j in range(1, order):
                    atom_n, xyz_n, icell_n = [
                        int(t) - 1 for t in elems.get("pair{:d}".format(j + 1)).split()
                    ]
                    atom_indices[i, j] = atom_n
                    xyz_indices[i, j] = xyz_n
                    index_shift_super[i, j - 1] = icell_n
                fcs_entries[i] = float(elems.text)

            shift_merged = (
                self.shift_vectors[atom_indices[:, 1:]]
                + shift_super[index_shift_super[:, 0:]]
            )
            atom_indices_mod[:, 0] = atom_indices[:, 0]
            atom_indices_mod[:, 1:] = self.map_s2p[atom_indices[:, 1:]]
            fcs_dic["fc{:d}".format(order)] = [
                atom_indices_mod,
                xyz_indices,
                shift_merged,
                fcs_entries,
            ]

        return fcs_dic

    def _get_forceconstants_h5(self, maxorder=4):
        """
        Parses force constants from an HDF5 file.

        Parameters:
            maxorder (int): The maximum order of force constants to read.

        Returns:
            dict: A dictionary containing the parsed force constants.
        """
        fcs_dic = {}

        xc_super = np.dot(self.supercell[1], self.supercell[0])
        lavec_prim = self.primitive_cell[0]
        inv_lavec = np.linalg.inv(lavec_prim)

        with h5py.File(self.filename, "r") as f:
            for order in range(2, maxorder + 1):
                search_tag = "/ForceConstants/Order{:d}".format(order)

                if search_tag in f:
                    atom_indices = f[search_tag + "/atom_indices"][:].astype(int)
                    atom_indices_super = f[search_tag + "/atom_indices_supercell"][
                        :
                    ].astype(int)
                    xyz_indices = f[search_tag + "/coord_indices"][:].astype(int)
                    shift_vectors_vel = f[search_tag + "/shift_vectors"][:].astype(
                        float
                    )
                    fcs_entries = f[search_tag + "/force_constant_values"][:].astype(
                        float
                    )

                    nrows = atom_indices.shape[0]
                    shift_vectors_vel = shift_vectors_vel.reshape((nrows, order - 1, 3))
                    shift_vectors = np.zeros((nrows, order - 1, 3), dtype=int)

                    for i in range(1, order):
                        xshift = (
                            xc_super[atom_indices_super[:, 0], :]
                            - xc_super[self.map_p2s[0, atom_indices[:, i]], :]
                        )
                        shift_vector_tmp = shift_vectors_vel[:, i - 1, :] + xshift
                        shift_vector_tmp = np.dot(shift_vector_tmp, inv_lavec)
                        shift_vector_int = np.round(shift_vector_tmp).astype(int)
                        shift_vectors[:, i - 1, :] = shift_vector_int

                    fcs_dic["fc{:d}".format(order)] = [
                        atom_indices,
                        xyz_indices,
                        shift_vectors,
                        fcs_entries,
                    ]

        return fcs_dic
