import numpy as np
import spglib
from ase.units import Bohr


class TaylorExpansionPotential:
    """
    A class to compute the Taylor expansion potential for a given atomic structure.

    Attributes:
    - maxorder (int): The maximum order of the Taylor expansion.
    - supercell0: The original supercell structure.
    - primitive_cell: The primitive cell derived from the supercell.
    - fcs_dic (dict): A dictionary containing the force constants.
    - atom_indices_taylor (dict): A dictionary mapping force constant keys to atom indices in the Taylor expansion.
    - coord_indices_taylor: Coordinates indices for the Taylor expansion.
    - fcs_values (dict): A dictionary of force constant values.
    - gamma_scaled_fcs_values (dict): A dictionary of gamma-scaled force constant values.
    - map_translation (numpy.ndarray): A mapping of translations for atom indices.
    - transformation_matrix (numpy.ndarray): A matrix transforming between primitive and supercell lattice vectors.
    - transformation_matrix_int (numpy.ndarray): Integer version of the transformation matrix.

    Methods:
    - __init__(self, supercell0, maxorder=4): Initializes the TaylorExpansionPotential object.
    - set_forceconstants(self, fcs_dic, primitive_cell_alm): Sets the force constants for the calculation.
    - compute(self, displacements): Computes the Taylor expansion potential and forces.
    - _build_structure(self): Builds the necessary structure information from the supercell.
    - _check_consistency_primitive_cell(primcell1, primcell2): Checks the consistency of the primitive cell.
    - _set_taylor_indices(self): Sets the indices for the Taylor expansion calculation.
    - gamma(flatten_indicies): Calculates the gamma values for given indices.
    """

    def __init__(self, supercell0, maxorder=4):
        """
        Initializes the TaylorExpansionPotential object with a given supercell and maximum order of expansion.

        Parameters:
        - supercell0: The initial supercell structure.
        - maxorder (int): The maximum order of the Taylor expansion to consider.
        """
        self.maxorder = maxorder
        self.supercell0 = supercell0
        self.primitive_cell = None

        self.fcs_dic = None
        self.atom_indices_taylor = None
        self.coord_indices_taylor = None
        self.fcs_values = None
        self.gamma_scaled_fcs_values = None
        self.map_translation = None
        self.transformation_matrix = None
        self.transformation_matrix_int = None

        self._build_structure()

    def set_forceconstants(self, fcs_dic, primitive_cell_alm):
        """
        Sets the force constants for the Taylor expansion calculation.

        Parameters:
        - fcs_dic (dict): A dictionary containing the force constants.
        - primitive_cell_alm: The primitive cell associated with the force constants.
        """
        self.fcs_dic = fcs_dic
        self._check_consistency_primitive_cell(self.primitive_cell, primitive_cell_alm)
        self._set_taylor_indices()

        self.fcs_values = {}
        self.gamma_scaled_fcs_values = {}
        self.coord_indices_taylor = {}
        self._force_segment_starts = {}
        self._force_group_atoms = {}
        self._force_group_coords = {}
        for fckey in self.fcs_dic.keys():
            order = int(fckey[2:])
            if order > self.maxorder:
                continue
            self.fcs_values[fckey] = self.fcs_dic[fckey][3]
            atom_indices_taylor = self.atom_indices_taylor[fckey]
            coord_indices_taylor = self.fcs_dic[fckey][1]
            flatten_indices = (
                self.supercell0.get_global_number_of_atoms() * atom_indices_taylor
                + coord_indices_taylor
            )
            gamma_values = self.gamma(flatten_indices)
            gamma_scaled_fcs = gamma_values * self.fcs_values[fckey]

            # Sort the entries by (first atom, first coordinate) so that the
            # force contributions sharing the same target component become
            # contiguous segments. The grouping is independent of the
            # translation because each translation permutes the atoms.
            sort_keys = 3 * atom_indices_taylor[:, 0] + coord_indices_taylor[:, 0]
            sort_order = np.argsort(sort_keys, kind="stable")
            keys_sorted = sort_keys[sort_order]
            segment_starts = np.flatnonzero(
                np.concatenate(([True], keys_sorted[1:] != keys_sorted[:-1]))
            )

            self.fcs_values[fckey] = self.fcs_values[fckey][sort_order]
            self.atom_indices_taylor[fckey] = atom_indices_taylor[sort_order]
            self.coord_indices_taylor[fckey] = coord_indices_taylor[sort_order]
            self.gamma_scaled_fcs_values[fckey] = gamma_scaled_fcs[sort_order]
            self._force_segment_starts[fckey] = segment_starts
            self._force_group_atoms[fckey] = self.atom_indices_taylor[fckey][
                segment_starts, 0
            ]
            self._force_group_coords[fckey] = self.coord_indices_taylor[fckey][
                segment_starts, 0
            ]

    def compute(self, displacements):
        """
        Computes the Taylor expansion potential and forces for given displacements.

        Parameters:
        - displacements (numpy.ndarray): The displacements of atoms in the supercell.

        Returns:
        - potential_energy (dict): A dictionary containing the potential energy calculated at each order of expansion.
        - atomic_forces (dict): A dictionary containing the forces on atoms calculated at each order of expansion.
        """
        if self.fcs_dic is None:
            raise RuntimeError("Force constants are not set.")

        if displacements.shape[1] != self.supercell0.get_global_number_of_atoms():
            raise RuntimeError("The number of displacements is inconsistent.")

        potential_energy = {}
        atomic_forces = {}

        nsnapshots = displacements.shape[0]
        natoms = self.supercell0.get_global_number_of_atoms()
        displacements_flat = displacements.reshape(nsnapshots, natoms * 3)

        for fckey in self.gamma_scaled_fcs_values.keys():
            order = int(fckey[2:])
            forces_flat = np.zeros((nsnapshots, natoms * 3), dtype=float)
            energy_taylor = np.zeros(nsnapshots, dtype=float)

            atom_indices_taylor = self.atom_indices_taylor[fckey]
            coord_indices_taylor = self.coord_indices_taylor[fckey]
            gamma_scaled_fcs = self.gamma_scaled_fcs_values[fckey]
            segment_starts = self._force_segment_starts[fckey]
            group_atoms = self._force_group_atoms[fckey]
            group_coords = self._force_group_coords[fckey]

            for itran in range(self.map_translation.shape[1]):
                atom_indices_mapped = self.map_translation[atom_indices_taylor, itran]
                flat_indices = 3 * atom_indices_mapped + coord_indices_taylor

                ff_tmp = displacements_flat[:, flat_indices[:, 1]] * gamma_scaled_fcs
                for j in range(2, order):
                    ff_tmp *= displacements_flat[:, flat_indices[:, j]]

                energy_taylor += np.einsum(
                    "ij,ij->i", ff_tmp, displacements_flat[:, flat_indices[:, 0]]
                )

                # The entries are pre-sorted by (first atom, first coordinate),
                # so the contributions to each force component form contiguous
                # segments and the scatter-add stays buffered.
                target_indices = (
                    3 * self.map_translation[group_atoms, itran] + group_coords
                )
                forces_flat[:, target_indices] += np.add.reduceat(
                    ff_tmp, segment_starts, axis=1
                )

            energy_taylor[:] *= 1.0 / float(order)
            potential_energy[fckey] = energy_taylor
            atomic_forces[fckey] = -forces_flat.reshape(nsnapshots, natoms, 3)

        energy_taylor = np.zeros(displacements.shape[0], dtype=float)
        for fckey in potential_energy.keys():
            energy_taylor += potential_energy[fckey]

        potential_energy["total"] = energy_taylor

        forces_taylor = np.zeros_like(displacements)
        for fckey in atomic_forces.keys():
            forces_taylor += atomic_forces[fckey]

        atomic_forces["total"] = forces_taylor

        return potential_energy, atomic_forces

    def _build_structure(self):
        """
        Builds the necessary structure information from the supercell,
        including the primitive cell and translation mappings.
        """
        cell = (
            np.array(self.supercell0.get_cell()),
            self.supercell0.get_scaled_positions(),
            self.supercell0.get_atomic_numbers(),
        )

        self.primitive_cell = spglib.standardize_cell(
            cell, to_primitive=True, no_idealize=True, symprec=1.0e-3
        )
        lavec_super = np.array(self.supercell0.get_cell())
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

        ntrans = (
            self.supercell0.get_global_number_of_atoms()
            // self.primitive_cell[1].shape[0]
        )
        self.map_translation = np.zeros(
            (self.supercell0.get_global_number_of_atoms(), ntrans), dtype=int
        )

        positions_tmp = np.dot(cell[1], self.transformation_matrix)
        atom_labels = np.zeros(len(positions_tmp), dtype=int)
        for i, position in enumerate(positions_tmp):
            position_diffs = self.primitive_cell[1] - position
            position_diffs_round = np.round(position_diffs)
            diffs_fraction = np.linalg.norm(
                position_diffs - position_diffs_round, axis=1
            )
            indices_mindiff = np.argmin(diffs_fraction)

            if diffs_fraction[indices_mindiff] > 1.0e-3:
                raise RuntimeError(
                    "Failed to find the corresponding primitive cell position."
                )
            atom_labels[i] = indices_mindiff

        transmat_inv = np.linalg.inv(self.transformation_matrix)

        for i, atom_index in enumerate(np.where(atom_labels == 0)[0]):
            xshift = np.round(positions_tmp[atom_index] - self.primitive_cell[1][0])
            xf_shifted = np.mod(cell[1] + np.dot(xshift, transmat_inv), 1.0)
            indices = np.zeros(len(xf_shifted), dtype=int)

            for j, coord in enumerate(xf_shifted):
                distances = np.linalg.norm(cell[1] - coord, axis=1)
                indices[j] = np.argmin(distances)

            self.map_translation[:, i] = indices

    @staticmethod
    def _check_consistency_primitive_cell(primcell1, primcell2):
        """
        Checks the consistency of the primitive cell with the one provided in the force constants file.

        Parameters:
        - primcell1: The primitive cell derived from the supercell.
        - primcell2: The primitive cell provided in the force constants file.
        """
        if np.abs(np.linalg.det(primcell1[0] - primcell2[0] * Bohr)) > 0.1:
            raise RuntimeError(
                "The primitive cell in the force constant file "
                "is different from the original structure."
            )

        if np.abs(primcell1[1] - primcell2[1]).max() > 1.0e-3:
            raise RuntimeError(
                "The atomic positions in the primitive cell in the force constant file "
                "are different from the original structure."
            )

    def _set_taylor_indices(self):
        """
        Sets the indices for the Taylor expansion calculation based on the force constants and the structure.
        """
        shift_fcs_unique = np.unique(self.fcs_dic["fc2"][2], axis=0).reshape(-1, 3)
        transmat_inv = np.linalg.inv(self.transformation_matrix)
        natom_prim = self.primitive_cell[2].shape[0]
        xf_super = self.supercell0.get_scaled_positions()

        # First generate the mapping table from (iat_prim, shifts) to (iat_super)
        map_to_super = np.zeros((natom_prim, len(shift_fcs_unique)), dtype=int)
        for atom in range(natom_prim):
            xf_prim_shifted = self.primitive_cell[1][atom] + shift_fcs_unique
            xf_super_shifted = np.mod(np.dot(xf_prim_shifted, transmat_inv), 1.0)
            distances = np.linalg.norm(
                xf_super[np.newaxis, :, :] - xf_super_shifted[:, np.newaxis, :], axis=2
            )
            map_to_super[atom] = np.argmin(distances, axis=1)

        # Encode each integer shift triplet into a single integer so that
        # entries can be located in shift_fcs_unique with a vectorized search.
        shift_min = shift_fcs_unique.min()
        shift_base = shift_fcs_unique.max() - shift_min + 1

        def encode_shifts(shifts):
            shifts_offset = shifts - shift_min
            return (
                shifts_offset[..., 0] * shift_base + shifts_offset[..., 1]
            ) * shift_base + shifts_offset[..., 2]

        codes_unique = encode_shifts(shift_fcs_unique)
        sort_order = np.argsort(codes_unique)
        codes_sorted = codes_unique[sort_order]
        map_to_super_sorted = map_to_super[:, sort_order]

        def lookup_shift_ids(shifts):
            if shifts.min() < shift_min or shifts.max() >= shift_min + shift_base:
                raise RuntimeError(
                    "A shift vector is outside the set found in fc2 entries."
                )
            codes = encode_shifts(shifts)
            shift_ids = np.searchsorted(codes_sorted, codes)
            if np.any(codes_sorted[np.minimum(shift_ids, len(codes_sorted) - 1)] != codes):
                raise RuntimeError(
                    "A shift vector is outside the set found in fc2 entries."
                )
            return shift_ids

        zero_shift_id = lookup_shift_ids(np.zeros((1, 3), dtype=int))[0]

        # Then, for each force constant entry, we map the indices (iat_prim, shifts) to (iat_super)
        self.atom_indices_taylor = {}
        for fckey in self.fcs_dic.keys():
            order = int(fckey[2:])
            if order > self.maxorder:
                continue
            atom_indices = self.fcs_dic[fckey][0]
            shift_fcs = np.asarray(self.fcs_dic[fckey][2]).reshape(-1, order - 1, 3)

            atom_indices_tmp = np.zeros((len(atom_indices), order), dtype=int)
            atom_indices_tmp[:, 0] = map_to_super_sorted[
                atom_indices[:, 0], zero_shift_id
            ]
            if order > 1:
                shift_ids = lookup_shift_ids(shift_fcs)
                atom_indices_tmp[:, 1:] = map_to_super_sorted[
                    atom_indices[:, 1:], shift_ids
                ]

            self.atom_indices_taylor[fckey] = atom_indices_tmp

    @staticmethod
    def gamma(flatten_indicies):
        """
        Calculates the gamma values for given indices, used in scaling the force constants.

        Parameters:
        - flatten_indicies (numpy.ndarray): A 2D array of indices for which to calculate gamma values.

        Returns:
        - A numpy array of gamma values for the given indices.
        """
        from math import factorial

        if len(flatten_indicies.shape) != 2:
            raise ValueError("Input array must be a 2D numpy array")

        n = flatten_indicies.shape[1]
        arr_tmp = np.copy(flatten_indicies)
        nsame = np.zeros_like(arr_tmp, dtype=int)

        ind_front = flatten_indicies[:, 0]
        nsame_to_front = np.ones(flatten_indicies.shape[0], dtype=int)

        arr_tmp.sort(axis=1)

        nuniq = np.ones(flatten_indicies.shape[0], dtype=int)
        iuniq = np.zeros(flatten_indicies.shape[0], dtype=int)

        nsame[:, 0] = 1

        for i in range(1, n):
            same_mask = arr_tmp[:, i] == arr_tmp[:, i - 1]
            nsame[np.arange(flatten_indicies.shape[0]), iuniq] += same_mask.astype(int)
            iuniq[~same_mask] += 1
            nsame[np.arange(flatten_indicies.shape[0]), iuniq] += (~same_mask).astype(
                int
            )
            nuniq += (~same_mask).astype(int)

            nsame_to_front += (flatten_indicies[:, i] == ind_front).astype(int)

        factorials = np.array([factorial(k) for k in range(n + 1)], dtype=int)
        denom = np.ones(flatten_indicies.shape[0], dtype=int)
        for i in range(n):
            denom *= factorials[nsame[:, i]]

        return nsame_to_front / denom
