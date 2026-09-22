/*
 scph_result_io.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <complex>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace PHON_NS
{
// Plain-data descriptions of what goes into PREFIX.scph.h5 / PREFIX.qha.h5
// (schema "alamode:scph_state"). Assembled by ScphQhaCommon; no HDF5 types
// appear here.
struct ScphSettingsH5
{
    std::string mode;                                   // "SCPH" or "QHA"
    std::map<std::string, std::string> input_variables; // log header values
    unsigned int kmesh_interpolate[3] = {0, 0, 0};
    unsigned int kmesh_dense[3] = {0, 0, 0};
    std::vector<double> temperatures; // K
    int nonanalytic = 0;              // warn on mismatch at restart
    int selfenergy_offdiag = 1;       // exit on mismatch at restart
    int relax_str = 0;
};

// Primitive cell of the SCPH run; the virtual supercell (primitive cell
// tiled by KMESH_INTERPOLATE) is derived from it inside the writer.
struct ScphCellsH5
{
    Eigen::Matrix3d lavec_prim; // columns = a1,a2,a3, bohr
    Eigen::MatrixXd xf_prim;    // [natmin, 3]
    std::vector<int> kinds;     // 0-based
    std::vector<std::string> elements;
    std::vector<double> masses_amu;
    int spin_polarized = 0;
    std::vector<std::vector<double>> magmom;
    int noncollinear = 0;
    int time_reversal_symmetry = 1;
    unsigned int ncell_grid[3] = {1, 1, 1}; // = KMESH_INTERPOLATE
};

// Relaxed structure per temperature, written only when RELAX_STR != 0.
// It is the deformation that turns the *reference* primitive cell (the
// PrimitiveCell group of this file) into the equilibrium structure at that
// temperature: lavec -> (I + u_tensor) lavec and a Cartesian displacement
// u0 per primitive-cell atom. A follow-up KAPPA/BUBBLE run reads it to put
// itself on the relaxed crystal; the SCPH/QHA run itself never needs it,
// since it carries the deformation in reciprocal space.
//
// Two caveats for consumers. The structure is the one the text outputs
// .atom_disp / .umn_tensor report, which is one optimizer update past the
// structure whose /dymat rows were computed; the convergence test bounds
// that last step by COORD_CONV_TOL / CELL_CONV_TOL. And a temperature whose
// structural optimization failed still has a row here -- it holds the
// fallback structure, which may be the input distortion paired with
// reference harmonic data -- so /convergence/structure must be honored.
struct ScphStructureH5
{
    std::vector<double> u_tensor;       // [NT * 9] row-major, dimensionless
    std::vector<double> u0;             // [NT * 3 * natmin], Cartesian bohr
    std::vector<std::string> spg_label; // [NT], e.g. "P4mm (#99)"
};

// Fingerprint of the IFCs the producing run loaded (Fcs_phonon::fcs_nrows /
// fcs_checksum), written alongside a relaxed structure so that a consumer
// can verify it is deforming with the same force constants the relaxation
// used. Empty vectors mean the run recorded none.
//
// Only the writing side lives here so far: the comparison belongs with the
// run that adopts the structure, where it can be tested end to end. Entry
// counts should be compared exactly and the checksums with a relative
// tolerance, since summation order may differ between runs.
struct ScphProvenanceH5
{
    std::vector<std::size_t> fcs_nrows;
    std::vector<double> fcs_checksum;
};

// Renormalized FC2 on the virtual supercell, in the row layout of the
// alamode force-constant schema (/ForceConstants/Order2). base_values holds
// the coarse-mesh-folded harmonic FC2; values_per_temperature the total
// (harmonic + anharmonic correction) FC2 per temperature.
struct ScphFc2RowsH5
{
    Eigen::MatrixXi atom_indices;               // [nrows, 2]
    Eigen::MatrixXi atom_indices_super;         // [nrows, 2]
    Eigen::MatrixXi coord_indices;              // [nrows, 2]
    Eigen::MatrixXd shift_vectors;              // [nrows, 3], Cartesian bohr
    Eigen::ArrayXd base_values;                 // [nrows], Ry/bohr^2
    std::vector<double> values_per_temperature; // [NT * nrows] row-major, Ry/bohr^2
    std::string variant;                        // "scph" or "qha"
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
    void load_dymat(const std::string &name, const std::vector<double> &temps_requested, unsigned int ns,
                    unsigned int ncell, std::complex<double> ****dymat_out) const;

    void load_v0(const std::vector<double> &temps_requested, std::vector<double> &v0_out) const;

    // Relaxed structure at one temperature. Returns false when the file
    // carries no /structure group (RELAX_STR = 0, or a file written before
    // the group existed), leaving the outputs untouched. u_tensor_out is
    // row-major 3x3; u0_out is Cartesian bohr, 3 per primitive-cell atom.
    bool load_structure(double temp_requested, std::vector<double> &u_tensor_out, std::vector<double> &u0_out,
                        std::string &spg_label_out) const;

    // Write the complete state atomically. delta_harm_renorm, v0, fc2, and
    // the convergence vectors may be null when the run does not produce
    // them (absent /convergence datasets mean "unknown", e.g. a legacy
    // import, and are accepted on read).
    void write_state(const ScphSettingsH5 &settings, const ScphCellsH5 &cells,
                     const std::complex<double> *const *const *const *delta_main,
                     const std::complex<double> *const *const *const *delta_harm_renorm, const std::vector<double> *v0,
                     const ScphFc2RowsH5 *fc2, const std::vector<unsigned char> *converged_scph,
                     const std::vector<unsigned char> *converged_structure, const ScphStructureH5 *structure = nullptr,
                     const ScphProvenanceH5 *provenance = nullptr) const;

    // Refuse (or, with allow_unconverged, only warn about) temperatures
    // whose SCPH iteration or structural optimization did not converge.
    void check_convergence(const std::vector<double> &temps_requested, bool allow_unconverged) const;

    [[nodiscard]] const std::string &get_filename() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace PHON_NS
