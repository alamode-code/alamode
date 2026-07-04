/*
 scph_result_io.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

namespace PHON_NS
{
// Plain-data descriptions of what goes into PREFIX.scph.h5 / PREFIX.qha.h5
// (schema "alamode:scph_state"). Assembled by ScphQhaCommon; no HDF5 types
// appear here.
struct ScphSettingsH5
{
    std::string mode;                    // "SCPH" or "QHA"
    unsigned int kmesh_interpolate[3] = {0, 0, 0};
    unsigned int kmesh_dense[3] = {0, 0, 0};
    std::vector<double> temperatures;    // K
    int nonanalytic = 0;                 // warn on mismatch at restart
    int selfenergy_offdiag = 1;          // exit on mismatch at restart
    int relax_str = 0;
};

// Primitive cell of the SCPH run; the virtual supercell (primitive cell
// tiled by KMESH_INTERPOLATE) is derived from it inside the writer.
struct ScphCellsH5
{
    Eigen::Matrix3d lavec_prim;          // columns = a1,a2,a3, bohr
    Eigen::MatrixXd xf_prim;             // [natmin, 3]
    std::vector<int> kinds;              // 0-based
    std::vector<std::string> elements;
    std::vector<double> masses_amu;
    int spin_polarized = 0;
    std::vector<std::vector<double>> magmom;
    int noncollinear = 0;
    int time_reversal_symmetry = 1;
    unsigned int ncell_grid[3] = {1, 1, 1};   // = KMESH_INTERPOLATE
};

// Renormalized FC2 on the virtual supercell, in the row layout of the
// alamode force-constant schema (/ForceConstants/Order2). base_values holds
// the coarse-mesh-folded harmonic FC2; values_per_temperature the total
// (harmonic + anharmonic correction) FC2 per temperature.
struct ScphFc2RowsH5
{
    Eigen::MatrixXi atom_indices;             // [nrows, 2]
    Eigen::MatrixXi atom_indices_super;       // [nrows, 2]
    Eigen::MatrixXi coord_indices;            // [nrows, 2]
    Eigen::MatrixXd shift_vectors;            // [nrows, 3], Cartesian bohr
    Eigen::ArrayXd base_values;               // [nrows], Ry/bohr^2
    std::vector<double> values_per_temperature; // [NT * nrows] row-major, Ry/bohr^2
    std::string variant;                      // "scph" or "qha"
};

// Writer/reader of the unified SCPH/QHA state file. All methods must be
// called from MPI rank 0 only. The file is written once, after the full
// temperature loop, as <filename>.part and published with an atomic rename,
// so a crash can never leave a partially written file under the final name.
class ScphResultIOH5
{
public:
    explicit ScphResultIOH5(std::string filename);

    ~ScphResultIOH5();

    // True when the file exists, carries the alamode:scph_state schema, and
    // is marked complete.
    [[nodiscard]] bool is_restartable() const;

    // Hard-exit on mesh/SELF_OFFDIAG mismatches, warn on NONANALYTIC —
    // mirroring the legacy text loader.
    void validate_settings(const ScphSettingsH5 &settings) const;

    // Load a /dymat dataset ("delta" or "delta_harm_renorm") for the
    // requested temperatures, which may be any subset of the file's grid
    // (a missing temperature is fatal). dymat_out is [NT][ns][ns][ncell].
    void load_dymat(const std::string &name, const std::vector<double> &temps_requested,
                    unsigned int ns, unsigned int ncell, std::complex<double> ****dymat_out) const;

    void load_v0(const std::vector<double> &temps_requested, std::vector<double> &v0_out) const;

    // Write the complete state atomically. delta_harm_renorm, v0, fc2, and
    // the convergence vectors may be null when the run does not produce
    // them (absent /convergence datasets mean "unknown", e.g. a legacy
    // import, and are accepted on read).
    void write_state(const ScphSettingsH5 &settings, const ScphCellsH5 &cells,
                     const std::complex<double> *const *const *const *delta_main,
                     const std::complex<double> *const *const *const *delta_harm_renorm,
                     const std::vector<double> *v0, const ScphFc2RowsH5 *fc2,
                     const std::vector<unsigned char> *converged_scph,
                     const std::vector<unsigned char> *converged_structure) const;

    // Refuse (or, with allow_unconverged, only warn about) temperatures
    // whose SCPH iteration or structural optimization did not converge.
    void check_convergence(const std::vector<double> &temps_requested,
                           bool allow_unconverged) const;

    [[nodiscard]] const std::string &get_filename() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace PHON_NS
