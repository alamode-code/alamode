"""
Apply SCPH force constant corrections to ALAMODE HDF5 files.

This script reads force constants from an HDF5 file, applies temperature-
dependent corrections from a dfc2 file, and saves the updated force constants
to a new HDF5 file.
"""

import argparse
import h5py
import numpy as np


class FC2Data:
    """Container for force constant data from HDF5 file."""
    
    def __init__(self, fname_h5):
        """Load force constant data from HDF5 file."""
        with h5py.File(fname_h5, 'r') as h5file:
            self.values = h5file['ForceConstants/Order2/force_constant_values'][:]
            self.atom_indices = h5file['ForceConstants/Order2/atom_indices'][:]
            self.coord_indices = h5file['ForceConstants/Order2/coord_indices'][:]
            self.shift_vectors = h5file['ForceConstants/Order2/shift_vectors'][:]
            self.lattice_vectors = h5file['PrimitiveCell/lattice_vector'][:]
            self.fractional_coords = h5file['PrimitiveCell/fractional_coordinate'][:]
            self.atomic_kinds = h5file['PrimitiveCell/atomic_kinds'][:]
    
    def calculate_fractional_shifts(self):
        """Calculate fractional shift vectors for each force constant entry."""
        inv_lattice = np.linalg.inv(self.lattice_vectors.T)
        shifts_frac = []
        
        for shift, atoms in zip(self.shift_vectors, self.atom_indices):
            # Convert Cartesian shift to fractional coordinates
            shift_cart_to_frac = inv_lattice @ shift
            # Calculate relative position in fractional coordinates
            shift_frac = np.round(
                shift_cart_to_frac 
                - self.fractional_coords[atoms[1]] 
                + self.fractional_coords[atoms[0]]
            )
            shifts_frac.append(shift_frac)
        
        return np.array(shifts_frac)


class DFC2Correction:
    """Container for force constant corrections from dfc2 file."""
    
    def __init__(self, fname_dfc2, temperature):
        """
        Parse dfc2 correction file for a specific temperature.
        
        Args:
            fname_dfc2: Path to dfc2 correction file
            temperature: Temperature in Kelvin
        """
        self.temperature = temperature
        self._parse_file(fname_dfc2)
    
    def _parse_file(self, fname_dfc2):
        """Parse the dfc2 file format."""
        with open(fname_dfc2, 'r') as f:
            # Read primitive cell lattice vectors
            lattice = []
            for _ in range(3):
                lattice.append([float(x) for x in f.readline().split()])
            self.lattice = np.array(lattice)
            
            # Read number of atoms and elements
            natoms, _ = [int(x) for x in f.readline().split()]
            _ = f.readline()  # element names
            
            # Read atomic positions and indices
            positions = []
            atomic_kinds = []
            for _ in range(natoms):
                line = f.readline().split()
                positions.append([float(x) for x in line[0:3]])
                atomic_kinds.append(int(line[3]) - 1)
            
            self.positions = np.array(positions)
            self.atomic_kinds = np.array(atomic_kinds)
            
            # Read corrections for the specified temperature
            self._read_corrections(f)
    
    def _read_corrections(self, f):
        """Read correction data for the target temperature."""
        shifts = []
        atoms = []
        coords = []
        values = []
        
        current_temp_match = False
        
        for line in f:
            if line.startswith('#'):
                if 'Temp' in line:
                    temp_in_file = float(line.split('=')[1].strip())
                    current_temp_match = abs(temp_in_file - self.temperature) < 0.01
            elif current_temp_match:
                parts = line.split()
                if len(parts) == 8:
                    shifts.append([int(x) for x in parts[0:3]])
                    atoms.append([int(parts[3]), int(parts[5])])
                    coords.append([int(parts[4]), int(parts[6])])
                    values.append(float(parts[7]))
        
        if len(values) == 0:
            raise ValueError(
                f"No corrections found at T={self.temperature} K"
            )
        
        self.shifts = np.array(shifts)
        self.atoms = np.array(atoms)
        self.coords = np.array(coords)
        self.values = np.array(values)
        
        print(f"Loaded {len(values)} corrections at T={self.temperature} K")


class FC2Updater:
    """Update force constants with corrections using efficient lookup."""
    
    @staticmethod
    def create_composite_key(atoms, coords, shifts):
        """
        Create a structured array for efficient sorting and searching.
        
        Combines atom indices, coordinate indices, and shifts into a single
        sortable key
        """
        n = len(atoms)
        shifts_rounded = np.round(shifts).astype(np.int32)
        
        dtype = [
            ('atom0', np.int32), ('atom1', np.int32),
            ('coord0', np.int32), ('coord1', np.int32),
            ('shift0', np.int32), ('shift1', np.int32), ('shift2', np.int32)
        ]
        
        keys = np.zeros(n, dtype=dtype)
        keys['atom0'] = atoms[:, 0]
        keys['atom1'] = atoms[:, 1]
        keys['coord0'] = coords[:, 0]
        keys['coord1'] = coords[:, 1]
        keys['shift0'] = shifts_rounded[:, 0]
        keys['shift1'] = shifts_rounded[:, 1]
        keys['shift2'] = shifts_rounded[:, 2]
        
        return keys
    
    @staticmethod
    def update(fc2_data, shifts_frac, dfc2_correction):
        """
        Apply corrections to force constants using binary search.
        
        Time complexity: O(M log M + N log M) where M is the number of
        force constants and N is the number of corrections.
        """

        fc2_updated = fc2_data.values.copy()

        # update atom_indices and shifts_frac, 
        # which are necessary when the primitive cell defined in dfc2 file 
        # is different from that in HDF5 file
        convmat = dfc2_correction.lattice @ np.linalg.inv(fc2_data.lattice_vectors)
        convmat_int = np.rint(convmat).astype(int)

        print(convmat)
        print(convmat_int)

        dfc2_positions_converted = dfc2_correction.positions @ convmat
        map_dfc2_to_fc2 = {}
        for i, pos in enumerate(dfc2_positions_converted):
            diffs = fc2_data.fractional_coords - pos
            dists = np.linalg.norm(diffs - np.rint(diffs), axis=1)
            closest_atom = np.argmin(dists)
            if dists[closest_atom] > 1e-5:
                raise ValueError("Atomic positions in dfc2 file do not match those in HDF5 file.")
            assert(dfc2_correction.atomic_kinds[i] == fc2_data.atomic_kinds[closest_atom])
            shift_tmp = np.rint(pos - fc2_data.fractional_coords[closest_atom]).astype(int)
            map_dfc2_to_fc2[i] = [int(closest_atom), shift_tmp]
        
        print(map_dfc2_to_fc2)
        
        # Create sorted lookup table for force constants
        fc2_keys = FC2Updater.create_composite_key(
            fc2_data.atom_indices,
            fc2_data.coord_indices,
            shifts_frac
        )
        sort_idx = np.argsort(fc2_keys)
        fc2_keys_sorted = fc2_keys[sort_idx]
        
        # Apply each correction
        n_matched = 0
        n_total = len(dfc2_correction.values)
        
        for i in range(n_total):
            # Skip negligible corrections
            if abs(dfc2_correction.values[i]) < 1e-10:
                continue
            
            # Map dfc2 atom indices and shifts to fc2 data
            atom0_dfc2 = dfc2_correction.atoms[i, 0]
            atom1_dfc2 = dfc2_correction.atoms[i, 1]
            atom0_fc2, shift0 = map_dfc2_to_fc2[atom0_dfc2]
            atom1_fc2, shift1 = map_dfc2_to_fc2[atom1_dfc2]
            shift_fc2 = shift1 - shift0 + dfc2_correction.shifts[i] @ convmat_int
            # Create query key
            query_key = FC2Updater.create_composite_key(
                np.array([atom0_fc2, atom1_fc2]).reshape(1, 2),
                dfc2_correction.coords[i:i+1],
                shift_fc2.reshape(1, 3)
            )[0]
            
            # Binary search
            idx = np.searchsorted(fc2_keys_sorted, query_key)
            
            # Check if match found
            if idx < len(fc2_keys_sorted) and fc2_keys_sorted[idx] == query_key:
                original_idx = sort_idx[idx]
                fc2_updated[original_idx] += dfc2_correction.values[i]
                n_matched += 1
            else:
                print(
                    f"Warning: No match for correction {i}: "
                    f"atoms={dfc2_correction.atoms[i]}, "
                    f"coords={dfc2_correction.coords[i]}, "
                    f"shift={dfc2_correction.shifts[i]}"
                )
        
        print(f"Applied {n_matched}/{n_total} nonzero corrections")
        return fc2_updated


class HDF5Writer:
    """Write updated force constants to HDF5 file."""
    
    @staticmethod
    def copy_with_updated_fc2(fname_in, fname_out, fc2_updated):
        """
        Copy HDF5 file with updated force constant values.
        
        All data and attributes are preserved except the force constant
        values, which are replaced with the updated values.
        """
        with h5py.File(fname_in, 'r') as f_in, \
             h5py.File(fname_out, 'w') as f_out:
            
            def copy_item(name, obj):
                """Recursively copy datasets and groups."""
                # Replace force constants with updated values
                if name == 'ForceConstants/Order2/force_constant_values':
                    f_out.create_dataset(name, data=fc2_updated)
                    return
                
                if isinstance(obj, h5py.Dataset):
                    # Handle scalar vs array datasets
                    data = obj[()] if obj.shape == () else obj[:]
                    
                    f_out.create_dataset(
                        name,
                        data=data,
                        dtype=obj.dtype,
                        compression=obj.compression
                    )
                    
                    # Copy attributes
                    for attr_name, attr_value in obj.attrs.items():
                        f_out[name].attrs[attr_name] = attr_value
                
                elif isinstance(obj, h5py.Group):
                    if name not in f_out:
                        f_out.create_group(name)
                    
                    # Copy attributes
                    for attr_name, attr_value in obj.attrs.items():
                        f_out[name].attrs[attr_name] = attr_value
            
            # Copy all items
            f_in.visititems(copy_item)
            
            # Copy root attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
        
        print(f"Saved updated force constants to: {fname_out}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Apply SCPH corrections to force constants in HDF5 file"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Input HDF5 file with force constants"
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help="Output HDF5 file for updated force constants"
    )
    parser.add_argument(
        '--dfc2',
        required=True,
        help="File containing force constant corrections"
    )
    parser.add_argument(
        '--temp',
        type=float,
        required=True,
        help="Temperature (K) for corrections"
    )
    args = parser.parse_args()
    
    print(f"Loading force constants from: {args.input}")
    fc2_data = FC2Data(args.input)
    
    print(f"Calculating fractional shifts...")
    shifts_frac = fc2_data.calculate_fractional_shifts()
    
    print(f"Loading corrections from: {args.dfc2}")
    dfc2_correction = DFC2Correction(args.dfc2, args.temp)
    
    print(f"Applying corrections...")
    fc2_updated = FC2Updater.update(fc2_data, shifts_frac, dfc2_correction)
    
    print(f"Writing results...")
    HDF5Writer.copy_with_updated_fc2(args.input, args.output, fc2_updated)
    
    print("Done!")


if __name__ == "__main__":
    main()
