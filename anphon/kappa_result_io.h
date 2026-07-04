/*
 kappa_result_io.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

namespace PHON_NS
{
// Plain-data description of the run-level metadata stored in
// PREFIX.kappa.h5 (schema "alamode:kappa_result"). Filled by Conductivity;
// no HDF5 types appear here so the physics code never depends on HighFive.
struct KappaFileMetaH5
{
    std::vector<double> temperatures;   // K
    int classical = 0;
    int ismear = -1;
    double smearing_width = 0.0;        // cm^-1
    std::string fcs_file;

    // Primitive cell (informational; validated on restart)
    Eigen::Matrix3d lattice_vector;     // columns = a1,a2,a3, bohr
    Eigen::MatrixXd x_fractional;       // [natmin, 3]
    std::vector<int> atomic_kinds;      // 0-based
    std::vector<std::string> elements;
    double volume = 0.0;

    // Which final-kappa datasets this run will produce (fixes the /kappa
    // layout at setup so no dataset is created after the file is published).
    bool with_kappa_3ph_only = false;
    bool with_kappa_coherent = false;
    bool with_kappa_coherent_block = false;
    bool with_kappa_spec = false;
    std::vector<double> energy_axis;    // cm^-1; used when with_kappa_spec
};

// Static description of one scattering channel ("3ph" or "4ph"): its k mesh
// and the per-mode quantities that do not change during the run.
struct KappaChannelMetaH5
{
    std::string tag;                        // "3ph" or "4ph"
    unsigned int nk_i[3] = {0, 0, 0};
    unsigned int nk_irred = 0;
    unsigned int ns = 0;
    Eigen::MatrixXd xk_irred;               // [nk_irred, 3] fractional q of representatives
    std::vector<double> weights;            // [nk_irred]
    std::vector<std::vector<int>> equiv_knum; // full-mesh k indices of each irreducible star
    Eigen::MatrixXd frequencies;            // [nk_irred, ns], cm^-1
    std::vector<double> velocities;         // [sum(multiplicity)*ns*3] flattened, m/s
};

// Crash-safe writer/reader of PREFIX.kappa.h5. All methods must be called
// from MPI rank 0 only; the class itself performs no MPI communication.
//
// Crash-consistency contract: every group/dataset a run will touch is
// created while the file is staged as PREFIX.kappa.h5.part and published
// with an atomic rename; after that, only in-place raw-data writes happen.
// Each gamma batch is committed in two flush+fsync steps (data rows first,
// then the gamma_computed flags), so a persisted flag of 1 guarantees the
// corresponding row is durable and a crash at any point leaves a readable
// file whose flags understate, never overstate, the finished work.
class KappaResultIOH5
{
public:
    explicit KappaResultIOH5(std::string filename);

    ~KappaResultIOH5();

    // Open PREFIX.kappa.h5, validating schema and metadata against the
    // current run (hard mismatches exit, soft ones warn), or (re)build it
    // when it is absent or its structure must change. Channels present in
    // an existing file are carried over with their computed gamma rows;
    // reset_channel discards the named channel's previous data instead.
    void open_or_create(const KappaFileMetaH5 &fmeta, const KappaChannelMetaH5 &channel,
                        bool reset_channel);

    // Validate an additional channel against an open file, creating it (via
    // rebuild-and-swap) when absent; reset_channel discards previous data.
    void ensure_channel(const KappaChannelMetaH5 &channel, bool reset_channel);

    // Copy every already-computed gamma row of the channel into
    // damping[row][itemp] (internal units) and return the row indices found.
    std::vector<int> load_computed_gamma(const std::string &tag, double **damping) const;

    // Durably commit nrow consecutive gamma rows starting at first_row
    // (values in internal units, converted to cm^-1 on disk), then their
    // completion flags.
    void store_gamma_batch(const std::string &tag, unsigned int first_row, unsigned int nrow,
                           const double *const *damping);

    // Bulk variant for scattered rows (legacy .result import).
    void store_gamma_rows(const std::string &tag, const std::vector<int> &rows,
                          const double *const *damping);

    // Fill the pre-created /kappa datasets (shapes fixed by KappaFileMetaH5)
    // and mark them valid. Null pointers skip the corresponding dataset.
    void store_kappa(const double *const *const *kappa_total,
                     const double *const *const *kappa_3ph_only,
                     const double *const *const *kappa_coherent,
                     const double *const *const *kappa_coherent_block,
                     const double *const *const *kappa_spec);

    [[nodiscard]] const std::string &get_filename() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace PHON_NS
