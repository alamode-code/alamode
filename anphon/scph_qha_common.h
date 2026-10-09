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
#include <cstdint>
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
#include "phonon.h"
#include "quartic_real_space.h"
#include "relaxation_types.h"
#include "scph_result_io.h"
#include "symmetry_core.h"
#include "system.h"
#include "v4_distributed.h"
#include "v4_service.h"

namespace PHON_NS
{
class DelVStrainData;
class DerivativeIFC;
class Ewald;

// Collaborators of the SCPH/QHA drivers (non-owning; owned by PHON, which outlives them).
// One struct because Scph and Qha forward the same list to their common base.
struct ScphQhaCollaborators
{
    const RunInfo &run;
    const Timer *timer;
    const Writes *writes;
    const System *system;
    const Symmetry *symmetry;
    const Kpoint *kpoint;
    const Fcs_phonon *fcs_phonon;
    const Ewald *ewald;
    const Dielec *dielec;
    const Dynamical *dynamical;
    const Integration *integration;
    Thermodynamics *thermodynamics;
    Dos *dos;
    AnharmonicCore *anharmonic_core;
    Selfenergy *selfenergy;
    Relaxation *relaxation;
};

class ScphQhaCommon
{
public:
    explicit ScphQhaCommon(const ScphQhaCollaborators &c);

    virtual ~ScphQhaCommon();

    // FILE_FORMAT = h5 (default) routes the restart state through the
    // unified PREFIX.scph.h5 / PREFIX.qha.h5 file; text keeps the legacy
    // .scph_dymat / .renorm_harm_dymat / .V0 trio. Set by the input parser.
    bool use_h5_io = true;

protected:
    // Collaborators, see ScphQhaCollaborators.
    const RunInfo &run;
    const Timer *timer;
    const Writes *writes;
    const System *system;
    const Symmetry *symmetry;
    const Kpoint *kpoint;
    const Fcs_phonon *fcs_phonon;
    const Ewald *ewald;
    const Dielec *dielec;
    const Dynamical *dynamical;
    const Integration *integration;
    Thermodynamics *thermodynamics;
    Dos *dos;
    AnharmonicCore *anharmonic_core;
    Selfenergy *selfenergy;
    Relaxation *relaxation;

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

    // EFIELD with Lambda: L[mn * ns + s] = sum_{k,i,b} E0_i Lambda_k,ib,mn Re e_s(kb) / sqrt(M_k)
    // [Ry/Bohr per normal coordinate and unit strain], with the eps8 mask of calculate_u0,
    // so that -E0 . sum_k (Lambda_k:u) u0_k = -sum_{mn,s} L_mn,s u_mn q0_s. Empty unless
    // EFIELD and Lambda are nonzero; set on every rank by setup_structural_opt_buffers.
    std::vector<double> efield_lambda_mode;
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

    // Temperature whose data each row actually holds (SCPH relaxation, rank 0):
    // T itself, the source temperature when a failed row got a copy of the last
    // converged one, NaN when it got the harmonic data. Written as
    // /convergence/data_temperature; empty means untracked and is not written.
    std::vector<double> data_temperature;

    // Converged relaxed structure per temperature (rank 0, RELAX_STR != 0).
    // Recorded next to converged_str_temp, so it holds whatever structure
    // the temperature loop actually accepted, including a fallback copied
    // from the last converged temperature. Written to the state file for a
    // follow-up KAPPA/BUBBLE run, and used by this run's BUBBLE > 0 step.
    ScphStructureH5 relaxed_structure;

    // Which temperatures actually recorded a structure. Not written to the
    // file: it guards against publishing zero rows when the temperature loop
    // skips an index, which the truncating iT of
    // run_structural_optimization_loop can do for a non-integral
    // (TMAX - TMIN) / DT.
    std::vector<unsigned char> relaxed_structure_recorded;

    // Record structure_state (and its space group) as the result of iT.
    // Sized on first use from converged_str_temp, the array it must stay
    // aligned with, so entries land in temperature order whichever way the
    // TMIN/TMAX sweep runs. Call it only after converged_str_temp is sized.
    void record_relaxed_structure(unsigned int iT, const RelaxationStructureState &structure_state);

    // Fill relaxed_structure for every temperature of this run from the
    // /structure group of a state file (restart of a relaxed run). Rank 0
    // reads; exits when the file carries no structure.
    void load_relaxed_structures_h5(const std::string &filename);

    // Static relaxed-structure energy V0(T), stored in the SCPH/QHA state file.
    // Size on every rank before restart broadcasts.
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

    // Project the dymat corrections delta[iT][is][js][icell] (real space on the
    // KMESH_INTERPOLATE supercell, as stored in the state file) in place onto the
    // space group of temperature iT: the run's operations (SymmListWithMap)
    // that are also operations spglib finds for the relaxed structure of iT
    // (RELAX_STR != 0), or all of them (RELAX_STR = 0); a temperature whose
    // operations do not form a group is skipped with a warning. Removes the ~1e-8 symmetry noise of the
    // eigenvector reconstruction before the state file / .scph_dfc2 are
    // written, so restart dymats and exported FC2 agree. Rank 0 computes;
    // collective (all ranks receive) when broadcast is true.
    void symmetrize_delta_dymat(std::complex<double> ****delta, unsigned int NT, bool broadcast = true) const;

    void setup_pp_interaction(const bool prepare_v3);

    void zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                             const unsigned int NT) const;

    // Rank 0: a correction from a file that took it against the analytic fc2 part only
    // (no delta_baseline marker) is rebased on the full harmonic matrix of this run,
    // delta += F^-1[D_analytic - D_harm] on the coarse mesh. Nothing to do for NONANALYTIC = 0.
    void convert_legacy_delta(std::complex<double> ****delta, unsigned int NT, const std::string &filename) const;

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

    // Print the wall-clock time since t_start for one stage of a structure
    // step (rank 0, VERBOSITY >= 2); t_start is a timer->elapsed() value.
    void print_stage_time(const std::string &label, double t_start) const;
    void print_stage_value(const std::string &label, double seconds) const;

    // Allocate the workspace buffers common to both structural-optimization
    // drivers, compute the reference V3/V4 elements and the strain
    // derivatives of the IFCs, create the optimizer, and detect the optical
    // modes at Gamma (the complement of the eigenvector-based acoustic
    // assignment). Collective: every rank must call it.
    void setup_structural_opt_buffers(StructuralOptWorkspace &ws);

    // Polarization of `state` (Born charges loaded) in the fixed-voltage model:
    // dt = A + B [e Bohr] with A = Omega_ref (e0 : u + 1/2 B2 : u : u) and
    // B = sum_k (Z*_k + Lambda_k : u) u0_k (B2: second-order tensor, Lambda: strain
    // derivative of the reduced Born charges),
    // pol = F (d_ref + dt) / (Omega_ref det F) and pol_ion = F B / (Omega_ref det F)
    // [uC/cm^2], F = I + u. Returns the field energy -E0 . (A + B) as used in V0 [Ry],
    // with the field force zE(u) of the strain of `state`.
    double efield_response(const StructuralOptWorkspace &ws, const RelaxationStructureState &state, double pol[3],
                           double pol_ion[3], double dt[3]) const;

    // The polarization (and the field energy) of `state`; full precision and dt
    // at VERBOSITY >= 2. No-op without Born charges.
    void print_efield_response(const StructuralOptWorkspace &ws, const RelaxationStructureState &state) const;

    // Residual force/stress norms of one structural step, the per-step
    // du/residual report, and the history-table record.
    void compute_and_print_step_gradients(const StructuralOptWorkspace &ws, const std::complex<double> *v1_eff,
                                          const std::complex<double> *del_v0_del_umn_eff, double du0, double du_tensor,
                                          const std::string &spg_label, std::vector<StructOptStepRecord> &step_history,
                                          double &grad_norm, double &cell_grad_norm) const;

    // Final structure report of one temperature point.
    void print_final_structure(const StructuralOptWorkspace &ws, RelaxationStrMode relax_mode, double temp,
                               bool last_temperature) const;

    // Rank-0 structural-optimization temperature loop shared by QHA now and
    // SCPH in a later phase. Driver-specific physics and acceptance policy
    // live behind IRelaxationModel hooks.
    void run_structural_optimization_loop(IRelaxationModel &model, StructuralOptLoopContext &ctx);

    // Print the initial atomic displacements (and, when the cell is relaxed,
    // the initial strain tensor) at the head of a temperature point.
    void print_initial_structure(const RelaxationStructureState &state, RelaxationStrMode relax_mode) const;

    // Row-distributed V4 (v4_service.h): built by build_v4_service, consumed by the
    // SCP solvers (fmat) and the q0 renormalization (q0_sweep); the other ranks serve
    // the contractions in v4_service->worker_loop() while rank 0 runs the loops.
    std::unique_ptr<V4Service> v4_service;

    // V4_REAL_SPACE (SCPH): 1 replaces the V4 tensor by contractions over the
    // folded real-space FC4 (rank 0; the other ranks build nothing and only wait
    // for the final opcode), 2 builds both and compares them (the tensor is used).
    int v4_real_space = 0;
    std::unique_ptr<quartic_rs::RealSpaceV4> v4_rs;
    mutable bool v4_rs_fmat_checked = false, v4_rs_q0_checked = false;
    // The SCP-matrix contraction with V4, plus the STRAIN_FC5 correction when present.
    void fmat_contract(const std::complex<double> *dvec, std::complex<double> ***fmat_all) const;
    void fmat_contract_v4(const std::complex<double> *dvec, std::complex<double> ***fmat_all) const;

    // STRAIN_FC5 (&relax, rank 0): the quartic couplings of the strained cell,
    // Phi4(u) = Phi4 + sum_mn u_mn dPhi4/du_mn, dPhi4/du_mn from the quintic IFCs
    // (DerivativeIFC::compute_dPhi4_dumn_groups). The 9 channels dPhi4/du_mn are folded
    // on one real-space slot set; dv4_fc5 contracts with the weights sum_mn u_mn
    // dPhi4/du_mn of the current strain (set in renormalize_ifcs_at_structure) and is
    // added to the V4 contractions: the SCP matrix (fmat_contract) and the q0 sweep.
    // fc5_q4mn[(c * nk + k) * ns^2 + a * ns + b] = sum_cd dV4/du_mn[(k, G)][a,b][c,d] q0[c] q0[d],
    // c the channel of mn
    // at every dense k: the q0 parts of the stress (calculate_del_v0_del_umn_renorm,
    // strain_vertex). Empty (null) without STRAIN_FC5.
    std::unique_ptr<quartic_rs::RealSpaceV4> dv4_fc5;
    std::vector<int> fc5_mn;                    // mn = 3 mu + nu of each channel (DIAG: 0, 4, 8)
    std::array<int, 9> fc5_channel_of_mn{};     // channel of mn, -1 when dropped
    std::vector<std::complex<double>> fc5_q4mn; // [(channel * nk + k) * ns^2 + ...]
    void build_fc5_correction(const DerivativeIFC &derivative_ifc);
    void update_fc5_at_structure(const std::vector<double> &q0, const std::array<std::array<double, 3>, 3> &u_tensor,
                                 StructuralOptWorkspace &ws);
    void q0_contract(const double *q0, const std::complex<double> *const *const *v3_with_umn,
                     std::complex<double> ***v3_renorm, std::complex<double> ***q4_q0) const;
    const double *const *v4_diag() const;

    // Choose the builder and the partition, allocate the local rows, build them
    // and gather the on-site diagonal. full_tensor: every element (SELF_OFFDIAG = 1
    // or structural relaxation); offdiag_fmat: SELF_OFFDIAG.
    void build_v4_service(bool full_tensor, bool offdiag_fmat);

    void zerofill_v4_acoustic_at_gamma(v4_distributed::V4RowBlock &v4_block) const;

    void compute_V4_elements_mpi_over_kpoint(v4_distributed::V4RowBlock &v4_block, std::complex<double> ***evec_in,
                                             bool self_offdiag, bool relax, const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const std::vector<int> &kmap_coarse_to_dense,
                                             const PhaseFactorCache *phase_storage_in,
                                             std::complex<double> *phi4_reciprocal_inout);

    void compute_V4_elements_mpi_over_band(v4_distributed::V4RowBlock &v4_block, std::complex<double> ***evec_in,
                                           bool self_offdiag, const KpointMeshUniform *kmesh_coarse_in,
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

    // Strain vertex M(is1, is2) = d Phi_k(is1, is2) / d u_{i1} (harmonic-mode
    // basis, dense k-point ik) at fixed occupations: del_v2 renormalized by the
    // strain (del2_v2) and the displacement q0 (del_v3). The SCP stress is
    // del_v0_del_umn_renorm + sum_k tr[M G^T] / (4 N).
    Eigen::MatrixXcd strain_vertex(const DelVStrainData &del_v_strain,
                                   const std::array<std::array<double, 3>, 3> &u_tensor, const std::vector<double> &q0,
                                   int i1, int ik, int nk) const;

    // With Fourier interpolation (KMESH_INTERPOLATE coarser than KMESH_SCPH), the q0- and
    // strain-renormalized harmonic matrix (and the FC5 part of the SCP matrix) of every
    // dense k is interpolated from the coarse mesh, and so are its derivatives. This
    // returns the dense occupation matrices G_k pulled back to the coarse points by the
    // adjoint of that Fourier interpolation (harmonic-mode basis of each coarse point c):
    //   sum_k tr[interp(M)_k G_k] = sum_c tr[M_c Geff_c]   for any coarse-mesh M,
    // which makes the explicit derivatives of the renormalized harmonic matrix in the force
    // and the stress exact for the interpolated model (SCP and iterative QHA; the
    // perturbative QHA does not interpolate). Not covered: the interpolated self-energy is
    // not self-adjoint (its quartic is coarse in k, dense in q), so the SCP force and
    // stress are still not exact derivatives of F_total (FE_scph_correction); the residual
    // was small in the BaTiO3 tests but is not bounded in general, e.g. near soft modes.
    // The adjoint reverses the Fourier step only, not the symmetrization at the irreducible
    // points and the star replication of the solve: it relies on vertices and occupations
    // that already carry the symmetry. Eigenvalue repairs in the SCP iteration break the
    // linearity the identity rests on.
    // Empty when the meshes coincide (the interpolation is the identity). The result is
    // cached and reused while the occupations (cmat_convert, omega2_anharm_T, T) are unchanged.
    const std::vector<Eigen::MatrixXcd> &coarse_occupation_matrices(std::complex<double> ***cmat_convert,
                                                                    double **omega2_anharm_T, double T_in,
                                                                    const KpointMeshUniform *kmesh_dense_in) const;
    mutable std::vector<Eigen::MatrixXcd> geff_cache;
    mutable std::uint64_t geff_cache_key = 0;

    void get_derivative_central_diff(double delta_t, unsigned int nk, double **omega0, double **omega2,
                                     double **domega_dt);

    // C(k) = U_ref(k)^dagger * U_new(k); evec_new_at_k is mode-major
    // ([js][is] = component is of mode js, the storage layout of evec_harmonic
    // and exec_interpolation output), transposed internally.
    static void build_cmat_at_k(unsigned int ns, const Eigen::MatrixXcd &evec_ref_mat,
                                const std::complex<double> *const *evec_new_at_k, std::complex<double> **cmat_out);

    // Identify Gamma acoustic modes by majority overlap with the harmonic
    // acoustic subspace: overlap(js) = sum_{is in acoustic_harm} |C[is][js]|^2.
    std::vector<bool> classify_acoustic_modes_from_cmat(const std::complex<double> *const *cmat_at_gamma) const;

    // Occupation-weighted SCP mode matrix at dense k:
    //   G(a,b) = sum_js C[a][js] f_js conj(C[b][js]),  C = cmat_at_k,
    // with f_js the displacement-correlation factor of SCP mode js at T_in (zero for the
    // Gamma acoustic modes, frequency floor eps8). Any Hermitian form
    // sum_js f_js (C^T M conj(C))(js,js) or sum_js f_js (C^+ M C)(js,js) then reduces to
    // sum_ab M(a,b) G(a,b) or sum_ab M(a,b) G(b,a): ns^2 per M instead of an ns^3 product.
    // Returns the number of non-acoustic modes that hit the frequency floor; the
    // optional is_acoustic_out receives the exclusion mask (empty away from Gamma).
    int scp_occupation_matrix(int ik, const std::complex<double> *const *cmat_at_k, const double *omega2_at_k,
                              double T_in, Eigen::MatrixXcd &G, std::vector<bool> *is_acoustic_out = nullptr) const;

    void zerofill_elements_acoustic_at_gamma(std::complex<double> ***v_elems, int fc_order, unsigned int nk_dense_in,
                                             unsigned int nk_irred_coarse_in) const;

    bool use_band_parallel_v4() const;
};
} // namespace PHON_NS
