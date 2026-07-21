/*
 scph_qha_common.h

 Copyright (c) 2026

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi.h"
#include "mpi_common.h"
#include "ndarray.h"
#include "pointers.h"
#include "relaxation_types.h"
#include "scph_result_io.h"
#include "symmetry_core.h"
#include "system.h"

namespace PHON_NS
{
class DelVStrainData;

class ScphQhaCommon: protected Pointers
{
public:
    explicit ScphQhaCommon(class PHON *phon);

    ~ScphQhaCommon() override;

    // FILE_FORMAT = h5 (default) routes the restart state through the
    // unified PREFIX.scph.h5 / PREFIX.qha.h5 file; text keeps the legacy
    // .scph_dymat / .renorm_harm_dymat / .V0 trio. Set by the input parser.
    bool use_h5_io = true;

protected:
    // Shared state between Scph and Qha.
    std::unique_ptr<KpointMeshUniform> kmesh_coarse;
    std::unique_ptr<KpointMeshUniform> kmesh_dense;
    std::vector<int> kmap_coarse_to_dense;

    NDArray<std::complex<double>, 1> phi3_reciprocal;
    NDArray<std::complex<double>, 1> phi4_reciprocal;
    std::unique_ptr<PhaseFactorCache> phase_factor;

    NDArray<double, 2> omega2_harmonic;
    NDArray<std::complex<double>, 3> evec_harmonic;

    // Index of the Gamma point in kmesh_dense and the eigenvector-based assignment of the
    // three acoustic (translational) modes there, in the harmonic mode order. Set by
    // setup_eigvecs(); used instead of frequency-magnitude thresholds wherever acoustic
    // modes at Gamma must be singled out.
    int ik_gamma_dense = -1;
    std::vector<bool> is_acoustic_gamma_harm;
    NDArray<MinimumDistList, 3> mindist_list;
    NDArray<std::complex<double>, 4> mat_transform_sym;

    std::vector<Eigen::MatrixXcd> dymat_harm_short;
    std::vector<Eigen::MatrixXcd> dymat_harm_long;
    int compute_Cv_anharmonic = 0;
    unsigned int ialgo = 0;
    bool selfenergy_offdiagonal = true;

    // Per-temperature convergence of the SCPH fixed-point iteration and of
    // the structural optimization (rank 0 only; the main loops fill them).
    // Stored in the state file and enforced when the renormalized data are
    // consumed later. Empty vectors mean "unknown" (legacy import) and are
    // not written.
    std::vector<unsigned char> converged_scph_temp;
    std::vector<unsigned char> converged_str_temp;

    // Zeroth-order (static) potential energy V0(T) of the relaxed structure,
    // recorded per temperature by the structural-optimization drivers and
    // stored in the state file. Owned here (not by Relaxation) because it is
    // per-run result state of the SCPH/QHA drivers. The drivers size it on
    // every rank (exec entry) before any restart loader broadcasts into it.
    std::vector<double> V0;

    // Legacy-text restart IO for V0 (PREFIX.V0). The text file also serves
    // as the human-readable V0-vs-T output. Definitions in scph_io.cpp.
    void load_V0_from_file();

    void store_V0_to_file() const;

    void initialize_variables();

    void deallocate_variables();

    void setup_kmesh(unsigned int kmesh_dense_input[3], unsigned int kmesh_coarse_input[3], const char *mode_name,
                     const char *mapping_error_message);

    void setup_eigvecs();

    void setup_structural_data();

    void setup_pp_interaction(const bool prepare_v3);

    void zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                             const unsigned int NT) const;

    void load_scph_dymat_from_file(std::complex<double> ****dymat_out, std::string filename_dymat,
                                   const KpointMeshUniform *kmesh_dense_in, const KpointMeshUniform *kmesh_coarse_in,
                                   const unsigned int nonanalytic_in, const bool selfenergy_offdiagonal_in);

    void store_renormalized_dymat_to_file(const std::complex<double> *const *const *const *dymat_in,
                                          std::string filename_dymat, const KpointMeshUniform *kmesh_dense_in,
                                          const KpointMeshUniform *kmesh_coarse_in, const unsigned int nonanalytic_in,
                                          const bool selfenergy_offdiagonal_in);

    static void mpi_bcast_complex(std::complex<double> ****data, const unsigned int NT, const unsigned int nk,
                                  const unsigned int ns);

    void write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat, unsigned int NT,
                                         const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                         bool is_qha = false, int type = 0);

    ScphSettingsH5 build_scph_settings_h5(const std::string &mode_name, unsigned int NT, unsigned int nonanalytic_in,
                                          bool selfenergy_offdiagonal_in, int relax_str_in) const;

    ScphCellsH5 build_scph_cells_h5() const;

    ScphFc2RowsH5 build_fc2_rows_h5(const std::complex<double> *const *const *const *delta_dymat, unsigned int NT,
                                    const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                    const std::string &variant) const;

    // Rank-0 only: assemble the full state and publish it atomically.
    void write_scph_state_h5(const std::string &filename, const std::string &mode_name, unsigned int NT,
                             unsigned int nonanalytic_in, bool selfenergy_offdiagonal_in, int relax_str_in,
                             const std::string &variant, const std::complex<double> *const *const *const *delta_main,
                             const std::complex<double> *const *const *const *delta_harm_renorm,
                             const std::vector<double> *v0, const KpointMeshUniform *kmesh_coarse_in,
                             MinimumDistList ***mindist_list_in) const;

    // Collective: rank 0 reads PREFIX.<mode>.h5 and the data are broadcast.
    // Returns false (on every rank) when no usable h5 file exists so the
    // caller can fall back to the legacy text loaders.
    bool load_scph_state_h5(const std::string &filename, const std::string &mode_name, unsigned int NT,
                            unsigned int nonanalytic_in, bool selfenergy_offdiagonal_in, int relax_str_in,
                            std::complex<double> ****delta_main, std::complex<double> ****delta_harm_renorm,
                            std::vector<double> *v0);

    void postprocess(std::complex<double> ****delta_dymat, std::complex<double> ****delta_harmonic_dymat_renormalize,
                     std::complex<double> ****delta_dymat_scph_plus_bubble, const KpointMeshUniform *kmesh_coarse_in,
                     MinimumDistList ***mindist_list_in, bool is_qha = false, int bubble_in = 0);

    // One structural-optimization step's IFC update: recompute the strain-
    // and q0-renormalized IFC arrays (and the strain gradient of the PES)
    // for the structure currently held in ws.structure_state.
    void renormalize_ifcs_at_structure(StructuralOptWorkspace &ws);

    // Allocate the workspace buffers common to both structural-optimization
    // drivers, compute the reference V3/V4 elements and the strain
    // derivatives of the IFCs, create the optimizer, and detect the optical
    // modes at Gamma (the complement of the eigenvector-based acoustic
    // assignment). Collective: every rank must call it.
    void setup_structural_opt_buffers(StructuralOptWorkspace &ws);

    // Residual force/stress norms of one structural step, the per-step
    // du/residual report, and the history-table record.
    void compute_and_print_step_gradients(const StructuralOptWorkspace &ws, const std::complex<double> *v1_eff,
                                          const std::complex<double> *del_v0_del_umn_eff, double du0, double du_tensor,
                                          const std::string &spg_label, std::vector<StructOptStepRecord> &step_history,
                                          double &grad_norm, double &cell_grad_norm) const;

    // Final structure report of one temperature point.
    void print_final_structure(const RelaxationStructureState &state, RelaxationStrMode relax_mode, double temp,
                               bool last_temperature) const;

    // Rank-0 structural-optimization temperature loop shared by QHA now and
    // SCPH in a later phase. Driver-specific physics and acceptance policy
    // live behind IRelaxationModel hooks.
    void run_structural_optimization_loop(IRelaxationModel &model, StructuralOptLoopContext &ctx);

    // Print the initial atomic displacements (and, when the cell is relaxed,
    // the initial strain tensor) at the head of a temperature point.
    void print_initial_structure(const RelaxationStructureState &state, RelaxationStrMode relax_mode) const;

    void compute_V4_elements_mpi_over_kpoint(std::complex<double> ***v4_out,
                                             std::complex<double> ***evec_in, bool self_offdiag, bool relax,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const std::vector<int> &kmap_coarse_to_dense,
                                             const PhaseFactorCache *phase_storage_in,
                                             std::complex<double> *phi4_reciprocal_inout);

    void compute_V4_elements_mpi_over_band(std::complex<double> ***v4_out,
                                           std::complex<double> ***evec_in, bool self_offdiag,
                                           const KpointMeshUniform *kmesh_coarse_in,
                                           const KpointMeshUniform *kmesh_dense_in,
                                           const std::vector<int> &kmap_coarse_to_scph,
                                           const PhaseFactorCache *phase_storage_in,
                                           std::complex<double> *phi4_reciprocal_inout);

    void compute_V3_elements_mpi_over_kpoint(std::complex<double> ***v3_out,
                                             const std::complex<double> *const *const *evec_in, bool self_offdiag,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const PhaseFactorCache *phase_storage_in,
                                             std::complex<double> *phi3_reciprocal_inout);

    void calculate_del_v0_del_umn_renorm(std::complex<double> *del_v0_del_umn_renorm, double *C1_array,
                                         double **C2_array, double ***C3_array,
                                         std::array<std::array<double, 3>, 3> &eta_tensor,
                                         const std::array<std::array<double, 3>, 3> &u_tensor,
                                         const DelVStrainData &del_v_strain, const std::vector<double> &q0,
                                         double pvcell, const KpointMeshUniform *kmesh_dense_in);

    void compute_anharmonic_v1_array(std::complex<double> *v1_SCP, std::complex<double> *v1_renorm,
                                     std::complex<double> ***v3_renorm, std::complex<double> ***cmat_convert,
                                     double **omega2_anharm_T, double T_in, const KpointMeshUniform *kmesh_dense_in);

    void compute_anharmonic_del_v0_del_umn(std::complex<double> *del_v0_del_umn_SCP,
                                           std::complex<double> *del_v0_del_umn_renorm,
                                           const DelVStrainData &del_v_strain,
                                           const std::array<std::array<double, 3>, 3> &u_tensor,
                                           const std::vector<double> &q0, std::complex<double> ***cmat_convert,
                                           double **omega2_anharm_T, double T_in,
                                           const KpointMeshUniform *kmesh_dense_in);

    void get_derivative_central_diff(double delta_t, unsigned int nk, double **omega0, double **omega2,
                                     double **domega_dt);

    // C(k) = U_ref(k)^dagger * U_new(k); evec_new_at_k is mode-major
    // ([js][is] = component is of mode js, the storage layout of evec_harmonic
    // and exec_interpolation output), transposed internally.
    static void build_cmat_at_k(unsigned int ns, const Eigen::MatrixXcd &evec_ref_mat,
                                const std::complex<double> *const *evec_new_at_k, std::complex<double> **cmat_out);

    // Identify which modes of the CURRENT (renormalized) eigenbasis at Gamma are the three
    // translational (acoustic) modes, given the unitary C(k=Gamma) connecting the harmonic
    // basis to the current one: overlap(js) = sum_{is in acoustic_harm} |C[is][js]|^2, and the
    // three modes with the largest overlap are flagged. Robust against the reshuffling of the
    // sorted mode indices that occurs when a soft optical mode becomes nearly degenerate with
    // the acoustic modes during the SCPH iteration.
    std::vector<bool> classify_acoustic_modes_from_cmat(const std::complex<double> *const *cmat_at_gamma) const;

    void zerofill_elements_acoustic_at_gamma(std::complex<double> ***v_elems, int fc_order,
                                             unsigned int nk_dense_in, unsigned int nk_irred_coarse_in) const;

    bool use_band_parallel_v4() const;
};
} // namespace PHON_NS
