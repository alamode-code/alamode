#!/usr/bin/env python3
"""
Apply SCPH (or QHA) force-constant corrections to an ALAMODE HDF5 file.

This script reads the bare second-order force constants from an ALAMODE HDF5
file, adds the temperature-dependent corrections stored in a ``*.scph_dfc2``
(or ``*.qha_dfc2``) file, and writes the renormalized force constants to a new
HDF5 file using the same format.

Cell conventions
----------------
The two inputs do *not* use the same cell to index the force constants:

* The HDF5 file stores FC2 in the **true primitive cell**.  ``atom_indices``
  run over the ``natmin`` true-primitive atoms, while ``atom_indices_supercell``
  and ``shift_vectors`` (Cartesian, Bohr) locate each atom inside the supercell.
* The dfc2 file (written by ANPHON) indexes atoms with the cell that ANPHON
  used as its primitive cell -- here the *conventional* cell stored in the
  HDF5 ``PrimitiveCell`` group.  When that cell contains more than one
  primitive translation (``number_of_primitive_translations > 1``) every
  physical correction is listed once *per copy* of the true-primitive atom.

The supercell is the common ground between the two: it is an integer tiling of
the conventional cell, and the dfc2 shift range (``KMESH_INTERPOLATE``) matches
it.  We therefore express each stored FC2 entry in conventional-cell
coordinates (using ``atom_indices_supercell`` + the supercell geometry to anchor
the central atom) and look up the single matching dfc2 correction.  Iterating
over the *stored* entries -- rather than over the dfc2 rows -- guarantees each
value is corrected exactly once and sidesteps the conventional-cell redundancy.

This mirrors the direction of the legacy ``dfc2.cpp`` reference (which only
handles the old XML format), adapted to the true-primitive vs. conventional
cell distinction of the HDF5 format.
"""

import argparse
import h5py
import numpy as np


def nint(x):
    """ALAMODE-compatible nearest integer rounding."""
    x_arr = np.asarray(x)
    return (x_arr + 0.5 - (x_arr < 0.0)).astype(np.int64)


class FC2Data:
    """Container for force constant data from an ALAMODE HDF5 file."""

    def __init__(self, fname_h5):
        """Load force constant data and cell information from the HDF5 file."""
        with h5py.File(fname_h5, "r") as h5file:
            self.values = h5file["ForceConstants/Order2/force_constant_values"][:]
            self.atom_indices = h5file["ForceConstants/Order2/atom_indices"][:]
            self.atom_indices_supercell = h5file[
                "ForceConstants/Order2/atom_indices_supercell"
            ][:]
            self.coord_indices = h5file["ForceConstants/Order2/coord_indices"][:]
            self.shift_vectors = h5file["ForceConstants/Order2/shift_vectors"][:]
            # PrimitiveCell here is the cell ANPHON treats as its primitive
            # cell (the conventional cell in the present workflow).
            self.lattice_vectors = h5file["PrimitiveCell/lattice_vector"][:]
            self.fractional_coords = h5file["PrimitiveCell/fractional_coordinate"][:]
            self.atomic_kinds = h5file["PrimitiveCell/atomic_kinds"][:]
            self.supercell_lattice_vectors = h5file["SuperCell/lattice_vector"][:]
            self.supercell_fractional_coords = h5file[
                "SuperCell/fractional_coordinate"
            ][:]
            self.supercell_atomic_kinds = h5file["SuperCell/atomic_kinds"][:]


class DFC2Correction:
    """Container for force constant corrections parsed from a dfc2 file."""

    def __init__(self, fname_dfc2, temperature):
        """
        Parse a dfc2 correction file for a specific temperature.

        Args:
            fname_dfc2: Path to the dfc2 correction file.
            temperature: Temperature in Kelvin.
        """
        self.temperature = temperature
        self._parse_file(fname_dfc2)

    def _parse_file(self, fname_dfc2):
        """Parse the dfc2 file format (header + corrections at the target T)."""
        with open(fname_dfc2, "r") as f:
            # Primitive-cell lattice vectors (rows = lattice vectors, Bohr).
            lattice = [[float(x) for x in f.readline().split()] for _ in range(3)]
            self.lattice = np.array(lattice)

            # Number of atoms / elements, then the element-name line.
            natoms, _ = [int(x) for x in f.readline().split()]
            _ = f.readline()  # element names

            positions = []
            atomic_kinds = []
            for _ in range(natoms):
                line = f.readline().split()
                positions.append([float(x) for x in line[0:3]])
                atomic_kinds.append(int(line[3]) - 1)
            self.positions = np.array(positions)
            self.atomic_kinds = np.array(atomic_kinds)

            self._read_corrections(f)

    def _read_corrections(self, f):
        """Read correction data for the target temperature."""
        shifts, atoms, coords, values = [], [], [], []
        current_temp_match = False

        for line in f:
            if line.startswith("#"):
                if "Temp" in line:
                    temp_in_file = float(line.split("=")[1].strip())
                    current_temp_match = abs(temp_in_file - self.temperature) < 0.01
            elif current_temp_match:
                parts = line.split()
                if len(parts) == 8:
                    shifts.append([int(x) for x in parts[0:3]])
                    atoms.append([int(parts[3]), int(parts[5])])
                    coords.append([int(parts[4]), int(parts[6])])
                    values.append(float(parts[7]))

        if len(values) == 0:
            raise ValueError(f"No corrections found at T={self.temperature} K")

        self.shifts = np.array(shifts, dtype=np.int64)
        self.atoms = np.array(atoms, dtype=np.int64)
        self.coords = np.array(coords, dtype=np.int64)
        self.values = np.array(values)

        print(f"Loaded {len(values)} corrections at T={self.temperature} K")


class FC2Updater:
    """Add dfc2 corrections to the bare FC2 stored in the HDF5 file."""

    # Geometric tolerance (Bohr) used when matching folded positions.
    CART_TOL = 1.0e-2
    # When --allow-cell-mismatch is used, warn if the matched atomic positions
    # of the two cells differ by more than this (Bohr).
    MISMATCH_WARN_CART = 0.2

    @staticmethod
    def _cell_mismatch_report(dfc2_lat, prim_lat):
        """Human-readable summary of how the two primitive cells differ."""
        dfc2_len = np.linalg.norm(dfc2_lat, axis=1)
        prim_len = np.linalg.norm(prim_lat, axis=1)
        vol_dfc2 = abs(np.linalg.det(dfc2_lat))
        vol_prim = abs(np.linalg.det(prim_lat))
        deform = dfc2_lat @ np.linalg.inv(prim_lat)  # dfc2 = deform @ HDF5
        lines = [
            "  dfc2 lattice |a_i| (Bohr): " + np.array2string(dfc2_len, precision=5),
            "  HDF5 lattice |a_i| (Bohr): " + np.array2string(prim_len, precision=5),
            "  length ratio dfc2/HDF5   : "
            + np.array2string(dfc2_len / prim_len, precision=5),
            f"  volume ratio dfc2/HDF5   : {vol_dfc2 / vol_prim:.5f}",
            "  lattice map (dfc2 = M @ HDF5):",
            "    " + np.array2string(deform, precision=5).replace("\n", "\n    "),
        ]
        return "\n".join(lines)

    @classmethod
    def _match_atom(cls, frac, prim_frac, prim_lat):
        """
        Find the primitive-cell atom matching ``frac`` modulo a lattice
        translation, comparing residuals in Cartesian (Bohr) space so the
        tolerance is independent of the (possibly skewed) lattice basis.

        Returns the atom index, its Cartesian residual (Bohr), and the integer
        translation T such that ``frac == prim_frac[atom] + T`` (mod rounding).
        """
        diff = frac - prim_frac
        residual = diff - np.rint(diff)
        cart_residual = residual @ prim_lat
        dists = np.linalg.norm(cart_residual, axis=1)
        atom = int(np.argmin(dists))
        trans = nint(frac - prim_frac[atom])
        return atom, float(dists[atom]), trans

    @classmethod
    def _fold_into_primcell(cls, cart, prim_frac, prim_lat, inv_prim):
        """
        Fold a Cartesian position into the (conventional) primitive cell.

        Returns the matching primitive-cell atom index and the integer lattice
        translation T such that ``cart == (prim_frac[atom] + T) @ prim_lattice``.
        """
        frac = cart @ inv_prim
        atom, dist, trans = cls._match_atom(frac, prim_frac, prim_lat)
        if dist > cls.CART_TOL:
            raise ValueError(
                "Could not fold a Cartesian position into the "
                "primitive cell; check that the supercell is an "
                "integer tiling of the PrimitiveCell."
            )
        return atom, trans

    @classmethod
    def update(
        cls, fc2_data, dfc2_correction, tol, verbose=False, allow_cell_mismatch=False
    ):
        """Return the renormalized FC2 values (bare + dfc2 corrections)."""
        prim_lat = (
            fc2_data.lattice_vectors
        )  # rows = conventional lattice vectors (Bohr)
        prim_frac = fc2_data.fractional_coords  # (n_conv, 3)
        prim_kind = fc2_data.atomic_kinds
        inv_prim = np.linalg.inv(prim_lat)  # cart @ inv_prim -> fractional

        # --- 1. The dfc2 cell should be the same cell as the HDF5 PrimitiveCell
        #        (ANPHON prints its own primitive cell as the dfc2 header).  If
        #        they differ only in size/shape -- e.g. corrections fitted at a
        #        different volume -- the atomic correspondence can still be
        #        established from fractional coordinates, but the transferred
        #        force constants are only an approximation.
        lattice_match = np.allclose(dfc2_correction.lattice, prim_lat, atol=1.0e-3)
        if not lattice_match:
            report = cls._cell_mismatch_report(dfc2_correction.lattice, prim_lat)
            if not allow_cell_mismatch:
                raise ValueError(
                    "The dfc2 lattice does not match the HDF5 PrimitiveCell lattice:\n"
                    + report
                    + "\nThe corrections were computed for a different cell. Re-run the "
                    "fit/SCPH on the matching structure, or pass --allow-cell-mismatch "
                    "to apply them anyway (atoms are matched by fractional coordinates "
                    "and the result is approximate)."
                )
            print("WARNING: applying corrections across mismatched primitive cells.")
            print(report)

        # Map each dfc2 atom onto a PrimitiveCell atom index by nearest position
        # modulo a lattice translation (an identity map when the cells agree).
        # Require a one-to-one correspondence so a deformed/mismatched cell that
        # cannot be aligned is rejected rather than silently mis-mapped.
        dfc2_to_prim = np.full(len(dfc2_correction.positions), -1, dtype=np.int64)
        used = {}
        max_frac_res = 0.0
        max_cart_res = 0.0
        for i, pos in enumerate(dfc2_correction.positions):
            residual = (pos - prim_frac) - np.rint(pos - prim_frac)
            fdist = np.linalg.norm(residual, axis=1)
            cdist = np.linalg.norm(residual @ prim_lat, axis=1)
            j = int(np.argmin(cdist))
            if lattice_match and cdist[j] > cls.CART_TOL:
                raise ValueError(
                    f"dfc2 atom {i} has no counterpart in the HDF5 PrimitiveCell."
                )
            if dfc2_correction.atomic_kinds[i] != prim_kind[j]:
                raise ValueError(
                    f"Atomic-kind mismatch for dfc2 atom {i} (nearest HDF5 atom {j})."
                )
            if j in used:
                raise ValueError(
                    f"dfc2 atoms {used[j]} and {i} both map to HDF5 atom {j}; "
                    f"cannot establish a one-to-one atom correspondence "
                    f"between the two cells."
                )
            used[j] = i
            dfc2_to_prim[i] = j
            max_frac_res = max(max_frac_res, float(fdist[j]))
            max_cart_res = max(max_cart_res, float(cdist[j]))

        if not lattice_match:
            print(
                f"  atoms matched 1:1 by fractional coordinates; max residual "
                f"{max_frac_res:.4f} (fractional), {max_cart_res:.4f} Bohr."
            )
            if max_cart_res > cls.MISMATCH_WARN_CART:
                print(
                    f"  CAUTION: internal coordinates differ by up to {max_cart_res:.3f} "
                    f"Bohr between the two cells; the correction transfer is approximate."
                )

        # --- 2. Build the correction lookup keyed in PrimitiveCell coordinates:
        #        (sx, sy, sz, atom0, coord0, atom1, coord1) -> value.
        #        Pre-compute each dfc2 row's key once for reuse in diagnostics.
        dfc2_keys = []
        for n in range(len(dfc2_correction.values)):
            sx, sy, sz = (int(s) for s in dfc2_correction.shifts[n])
            a0 = int(dfc2_to_prim[dfc2_correction.atoms[n, 0]])
            a1 = int(dfc2_to_prim[dfc2_correction.atoms[n, 1]])
            c0 = int(dfc2_correction.coords[n, 0])
            c1 = int(dfc2_correction.coords[n, 1])
            dfc2_keys.append((sx, sy, sz, a0, c0, a1, c1))

        lookup = {}
        n_dup_rows = 0
        for n, key in enumerate(dfc2_keys):
            val = dfc2_correction.values[n]
            if key in lookup:
                # Identical key must carry an identical value; otherwise the
                # output would silently depend on row order.
                if abs(lookup[key] - val) > tol:
                    raise ValueError(
                        f"The dfc2 file contains inconsistent duplicate entries for "
                        f"key {key}: {lookup[key]:.8e} vs {val:.8e}."
                    )
                n_dup_rows += 1
                continue
            lookup[key] = val

        # --- 3. Fold every supercell atom into the (conventional) primitive cell.
        sc_cart = (
            fc2_data.supercell_fractional_coords @ fc2_data.supercell_lattice_vectors
        )
        sc_to_prim = [
            cls._fold_into_primcell(sc_cart[a], prim_frac, prim_lat, inv_prim)
            for a in range(sc_cart.shape[0])
        ]

        # --- 4. Express each stored FC2 entry in PrimitiveCell coordinates and
        #        add the single matching dfc2 correction.
        fc2_updated = fc2_data.values.copy()
        consumed = set()
        consumed_count = {}  # key -> number of stored FC2 entries it was applied to
        n_applied = 0
        n_nonzero_applied = 0

        for i in range(len(fc2_updated)):
            sc0 = int(fc2_data.atom_indices_supercell[i, 0])
            atom0, trans0 = sc_to_prim[sc0]

            # Cartesian positions: atom0 at its supercell-home site, atom1 at the
            # image encoded by shift_vectors.
            x0 = sc_cart[sc0]
            x1 = x0 + fc2_data.shift_vectors[i]
            atom1, trans1 = cls._fold_into_primcell(x1, prim_frac, prim_lat, inv_prim)

            # Bring the central atom into the home cell (the dfc2 convention).
            shift = trans1 - trans0
            key = (
                int(shift[0]),
                int(shift[1]),
                int(shift[2]),
                atom0,
                int(fc2_data.coord_indices[i, 0]),
                atom1,
                int(fc2_data.coord_indices[i, 1]),
            )

            val = lookup.get(key)
            if val is not None:
                fc2_updated[i] += val
                consumed.add(key)
                consumed_count[key] = consumed_count.get(key, 0) + 1
                n_applied += 1
                if abs(val) > tol:
                    n_nonzero_applied += 1

        # --- 5. Diagnostics.
        #        A physical correction may be listed several times in the dfc2
        #        file (once per translational copy of an atom in the conventional
        #        cell).  An unconsumed nonzero row is an expected duplicate only
        #        if an *equivalent* correction was actually consumed.  Equivalence
        #        is tested with a translation-invariant signature -- the Cartesian
        #        bond vector together with the two coordinate directions uniquely
        #        identifies the force-constant component -- so a genuine miss can
        #        never be silently reclassified as a duplicate.
        def signature(n):
            a0 = int(dfc2_to_prim[dfc2_correction.atoms[n, 0]])
            a1 = int(dfc2_to_prim[dfc2_correction.atoms[n, 1]])
            vec = (prim_frac[a1] + dfc2_correction.shifts[n] - prim_frac[a0]) @ prim_lat
            return (
                int(dfc2_correction.coords[n, 0]),
                int(dfc2_correction.coords[n, 1]),
                tuple(np.round(vec, 3)),
            )

        consumed_sig = {
            signature(n) for n, key in enumerate(dfc2_keys) if key in consumed
        }

        n_redundant = 0
        unmatched = []
        for n, key in enumerate(dfc2_keys):
            val = dfc2_correction.values[n]
            if abs(val) <= tol or key in consumed:
                continue
            if signature(n) in consumed_sig:
                n_redundant += 1
            else:
                unmatched.append((key, float(val)))

        n_shared = sum(1 for c in consumed_count.values() if c > 1)

        print(
            f"Applied corrections to {n_applied} FC2 entries "
            f"({n_nonzero_applied} with |delta| > {tol:g})."
        )
        print(
            f"dfc2 nonzero rows: translational duplicates skipped {n_redundant}, "
            f"unmatched {len(unmatched)}."
        )
        if n_dup_rows:
            print(
                f"Note: collapsed {n_dup_rows} duplicate dfc2 row(s) with "
                f"identical keys (values consistent within tol)."
            )
        if n_shared:
            print(
                f"Note: {n_shared} correction(s) applied to multiple "
                f"symmetry-equivalent FC2 entries."
            )

        if unmatched:
            print(
                "\nWarning: the following nonzero corrections have no matching "
                "force constant in the original HDF5 file:"
            )
            shown = unmatched if verbose else unmatched[:10]
            for key, val in shown:
                sx, sy, sz, a0, c0, a1, c1 = key
                print(
                    f"  shift=({sx:d},{sy:d},{sz:d}) atom0={a0} coord0={c0} "
                    f"atom1={a1} coord1={c1} value={val:.8e}"
                )
            if not verbose and len(unmatched) > len(shown):
                print(
                    f"  ... and {len(unmatched) - len(shown)} more (use --verbose to list all)."
                )
            print(
                "\n If the values above are small enough (e.g. < 1e-8) this is\n"
                " usually harmless.  Otherwise, make sure the q-point grid set by\n"
                " KMESH_INTERPOLATE is not larger than the supercell of the original\n"
                " FC2, and that ANPHON detected the same space group as ALM."
            )

        return fc2_updated


class HDF5Writer:
    """Write updated force constants to an HDF5 file."""

    @staticmethod
    def copy_with_updated_fc2(fname_in, fname_out, fc2_updated, provenance=None):
        """
        Copy the HDF5 file, replacing only the Order2 force-constant values.

        All other datasets and attributes are preserved.  Optional provenance
        information is recorded as root attributes.
        """
        with h5py.File(fname_in, "r") as f_in, h5py.File(fname_out, "w") as f_out:

            def copy_item(name, obj):
                if name == "ForceConstants/Order2/force_constant_values":
                    dset = f_out.create_dataset(name, data=fc2_updated)
                    for attr_name, attr_value in obj.attrs.items():
                        dset.attrs[attr_name] = attr_value
                    return

                if isinstance(obj, h5py.Dataset):
                    data = obj[()] if obj.shape == () else obj[:]
                    f_out.create_dataset(
                        name, data=data, dtype=obj.dtype, compression=obj.compression
                    )
                    for attr_name, attr_value in obj.attrs.items():
                        f_out[name].attrs[attr_name] = attr_value

                elif isinstance(obj, h5py.Group):
                    if name not in f_out:
                        f_out.create_group(name)
                    for attr_name, attr_value in obj.attrs.items():
                        f_out[name].attrs[attr_name] = attr_value

            f_in.visititems(copy_item)

            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value

            if provenance:
                for attr_name, attr_value in provenance.items():
                    f_out.attrs[attr_name] = attr_value

        print(f"Saved updated force constants to: {fname_out}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Apply SCPH/QHA corrections to FC2 in an ALAMODE HDF5 file"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input HDF5 file with the bare force constants",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output HDF5 file for the renormalized force constants",
    )
    parser.add_argument(
        "--dfc2", required=True, help="dfc2 correction file (*.scph_dfc2 / *.qha_dfc2)"
    )
    parser.add_argument(
        "--temp", type=float, required=True, help="Temperature (K) for the corrections"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-10,
        help="Ignore |dfc2| below this threshold for diagnostics (default: 1e-10)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="List all unmatched corrections"
    )
    parser.add_argument(
        "--allow-cell-mismatch",
        action="store_true",
        help="Apply the corrections even if the dfc2 primitive cell "
        "differs in size/shape from the HDF5 PrimitiveCell "
        "(e.g. fitted at a different volume). Atoms are matched "
        "by fractional coordinates; the result is approximate "
        "and a diagnosis is printed.",
    )
    args = parser.parse_args()

    print(f"Loading force constants from: {args.input}")
    fc2_data = FC2Data(args.input)

    print(f"Loading corrections from: {args.dfc2}")
    dfc2_correction = DFC2Correction(args.dfc2, args.temp)

    print("Applying corrections...")
    fc2_updated = FC2Updater.update(
        fc2_data, dfc2_correction, args.tol, args.verbose, args.allow_cell_mismatch
    )

    print("Writing results...")
    provenance = {
        "dfc2_original_file": args.input,
        "dfc2_correction_file": args.dfc2,
        "dfc2_temperature": args.temp,
    }
    HDF5Writer.copy_with_updated_fc2(args.input, args.output, fc2_updated, provenance)

    print("Done!")


if __name__ == "__main__":
    main()
