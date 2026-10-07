/*
 relaxation.h

 Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <complex>
#include <memory>
#include "constants.h"
#include "fcs_phonon.h"
#include "kpoint.h"
#include "optimizers.h"
#include "phonon.h"
#include "relaxation_types.h"
#include "scph.h"
#include "strain_coupling_types.h"
#include "strain_reference_cell.h"
#include "symmetry_core.h"

namespace PHON_NS
{
// 1 e/Bohr^2 in C/m^2 (polarization and piezoelectric units; 1 C/m^2 = 0.017478 e/Bohr^2)
inline constexpr double e_bohr2_in_c_m2 = 1.6021766208e-19 / (Bohr_in_Angstrom * Bohr_in_Angstrom * 1.0e-20);

class DerivativeIFC;
class ElasticTensor;

class DelVStrainData
{
public:
    using MatrixXcdRowMajor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    MatrixXcdRowMajor del_v1;                           // [9][ns]
    MatrixXcdRowMajor del2_v1;                          // [81][ns]
    MatrixXcdRowMajor del3_v1;                          // [729][ns]
    std::vector<MatrixXcdRowMajor> del_v2;              // [9][nk][ns*ns]
    std::vector<MatrixXcdRowMajor> del2_v2;             // [81][nk][ns*ns]
    std::vector<std::vector<MatrixXcdRowMajor>> del_v3; // [9][nk][ns][ns*ns]

    DelVStrainData() = default;
    ~DelVStrainData() = default;

    void resize(const int nk, const int nmode)
    {
        nk_ = nk;
        nmode_ = nmode;
        const auto nmode2 = nmode * nmode;

        del_v1.resize(9, nmode);
        del2_v1.resize(81, nmode);
        del3_v1.resize(729, nmode);

        del_v2.resize(9);
        for (auto &mat: del_v2) {
            mat.resize(nk, nmode2);
        }

        del2_v2.resize(81);
        for (auto &mat: del2_v2) {
            mat.resize(nk, nmode2);
        }

        del_v3.resize(9);
        for (auto &per_strain: del_v3) {
            per_strain.resize(nk);
            for (auto &mat: per_strain) {
                mat.resize(nmode, nmode2);
            }
        }
    }

    int nk() const
    {
        return nk_;
    }
    int nmode() const
    {
        return nmode_;
    }

private:
    int nk_{0};
    int nmode_{0};
};

class Relaxation
{
public:
    Relaxation(const RunInfo &run, const System *system);

    ~Relaxation();

    int relax_str;

    // initial strain and displacement
    double init_u_tensor[3][3]{{0.0}};
    std::vector<double> init_u0;
    std::vector<InitialDisplacementMode> init_disp_modes;

    // Resolve init_disp_modes into init_u0 with the analytic Gamma-point dynamical matrix.
    // Needs Fcs_phonon::setup; must precede the symmetry analysis of the distorted cell.
    void set_init_u0_from_modes(const std::vector<FcsArrayWithCell> &fc2,
                                const std::vector<SymmetryOperationWithMapping> &symops_ref);

    // variables related to structural optimization
    int relax_algo;
    int max_str_iter;
    double coord_conv_tol;
    double mixbeta_coord;
    double alpha_steepest_decent;
    double cell_conv_tol;
    double mixbeta_cell;
    // When positive, also require the q0 force norm below this threshold
    // to prevent small-step convergence at a non-stationary point.
    double gradient_conv_tol;
    // When positive and relax_str == 2, also require the cell-gradient norm
    // (including pressure) below this threshold. Its units differ from the
    // coordinate-force threshold.
    double cell_gradient_conv_tol;
    // For relax_algo == 3 (GDIIS): if nonzero, apply the Farkas-Schlegel "controlled GDIIS"
    // step-acceptance criteria (step-length cap, coefficient/extrapolation cap, and
    // near-singularity rejection with error-vector rescaling). Enabled by default;
    // GDIIS_PLAIN = 1 in the &relax field switches back to the regular GDIIS.
    int gdiis_control;

    int set_init_str;
    // Reference space group; SET_INIT_STR = 3 re-seeds displacements when it is restored.
    int spacegroup_number_ref{0};
    double add_hess_diag;
    // BUBBLE_HESS = 1: the coordinate block of the optimizer Hessian is the
    // curvature of the SCP free energy (BUBBLE = 4) at the current structure,
    // with |eigenvalues| (saddle-free) instead of the SCP Gamma matrix + ADD_HESS_DIAG.
    int bubble_hess{0};
    double stat_pressure;
    // EFIELD [eV/Angstrom], Cartesian: static field coupling to the Born charges
    // through -E . sum_k Z*_k u0_k (fixed-E electric enthalpy).
    std::array<double, 3> efield;

    bool has_efield() const
    {
        return efield[0] != 0.0 || efield[1] != 0.0 || efield[2] != 0.0;
    }
    // POL_REF [C/m^2]: polarization of the reference structure. Output only: it
    // never enters an energy (fixed-voltage model). pol_ref_given: the tag was set.
    std::array<double, 3> pol_ref;
    bool pol_ref_given;
    bool has_pol_ref() const
    {
        return pol_ref[0] != 0.0 || pol_ref[1] != 0.0 || pol_ref[2] != 0.0;
    }
    // Clamped-ion proper piezoelectric tensor e0_ijk [e/Bohr^2], flat (i * 3 + j) * 3 + k,
    // from STRAINFILE /Piezoelectric; zero when absent, with STRAIN_IFC_DIR, or at a
    // fixed cell. Set on every rank by setup_relaxation; piezo0_source names it.
    std::array<double, 27> piezo0{};
    std::string piezo0_source{"none (e0 = 0)"};
    bool has_piezo0() const
    {
        return std::any_of(piezo0.begin(), piezo0.end(), [](const double x) { return x != 0.0; });
    }
    // Fixed-voltage cell gradient dH_E/du_mn = -Omega_ref E0_i e0_imn [Ry], flat m * 3 + n,
    // constant in u; set only when both EFIELD and e0 are nonzero (has_piezo_field_term).
    std::array<double, 9> efield_strain_gradient{};
    bool has_piezo_field_term() const
    {
        return has_efield() && has_piezo0();
    }
    // Phase 2 (STRAINFILE /Piezoelectric, optional and independent; zero when absent):
    // piezo2: the second-order clamped-ion tensor B_i,jk,lm = de0_ijk/du_lm [e/Bohr^2],
    //   flat i * 81 + (j * 3 + k) * 9 + l * 3 + m, symmetrized;
    // born_strain: Lambda_k,ib,mn = d(F^-1 Z*)_k,ib/du_mn [e] mapped onto the atoms of the
    //   primitive cell and ASR-corrected, flat k * 81 + (i * 3 + b) * 9 + m * 3 + n; empty
    //   when absent. Set on every rank by setup_relaxation; the sources name them.
    std::array<double, 243> piezo2{};
    std::string piezo2_source{"none (B = 0)"};
    std::vector<double> born_strain;
    std::string born_strain_source{"none (Lambda = 0)"};
    bool has_piezo2() const
    {
        return std::any_of(piezo2.begin(), piezo2.end(), [](const double x) { return x != 0.0; });
    }
    bool has_born_strain() const
    {
        return std::any_of(born_strain.begin(), born_strain.end(), [](const double x) { return x != 0.0; });
    }
    // -Omega_ref E0_i B_i,mn,pq [Ry], flat mn * 9 + pq: the strain curvature of the
    // field enthalpy; set only when both EFIELD and B are nonzero (has_piezo2_field_term).
    std::array<double, 81> efield_strain_curvature{};
    bool has_piezo2_field_term() const
    {
        return has_efield() && has_piezo2();
    }
    bool has_born_strain_field_term() const
    {
        return has_efield() && has_born_strain();
    }

    // STRAIN_COUPLING as given (-1: the deprecated RENORM_*/ELASTIC_CONST
    // tags set a combination it cannot express). The four switches below
    // are derived from it in the input parser and are the runtime state.
    int strain_coupling;
    int renorm_3to2nd;  // dV2/du: 1 cubic IFCs, 2 file (3 accepted as 2), 4 k-space file (undocumented)
    int renorm_2to1st;  // dV1/du: 0 zero, 1 harmonic IFCs, 2 file
    int renorm_34to1st; // d2V1/du2, d3V1/du3: 0 zero, 1 cubic and quartic IFCs
    // Source of the elastic constants entering V0(u): 1 computes the
    // clamped-ion C2 (and C3) analytically from the loaded IFCs, 2 reads
    // them from the file (default).
    int elastic_const;
    std::string strain_IFC_dir;
    std::string strain_file; // STRAINFILE: the HDF5 container replacing the text files

    // STRAIN_FC5 = 1: the strain derivative of the quartic IFCs from the quintic ones,
    // Phi4(u) = Phi4 + sum_mn u_mn dPhi4/du_mn (SCPH, RELAX_STR = 2, 4; rank 0).
    // STRAIN_FC5_CHANNELS = DIAG keeps only u_xx, u_yy, u_zz in that sum.
    int strain_fc5 = 0;
    bool strain_fc5_diag = false;

    // The source of the strain couplings and elastic constants (STRAIN_IFC_DIR
    // text files or the STRAINFILE container).
    strain_coupling::StrainSource strain_source() const
    {
        return strain_coupling::StrainSource{strain_IFC_dir, strain_file};
    }

    std::unique_ptr<Optimizer> optimizer;

    void create_optimizer(const size_t num_modes);

    // symprec is Symmetry::tolerance, cached for the spglib calls of this class.
    // ref_has_inversion: the reference cell has the inversion (missing-e0 warning).
    // symops_ref: the operations of the reference structure with the atom mapping
    // (Symmetry::SymmListWithMap_ref), used to symmetrize /Piezoelectric.
    void setup_relaxation(double symprec, bool ref_has_inversion,
                          const std::vector<SymmetryOperationWithMapping> &symops_ref);

    void compute_del_v_strain(const DerivativeIFC &derivative_ifc, const KpointMeshUniform *kmesh_coarse,
                              const KpointMeshUniform *kmesh_dense, DelVStrainData &del_v_strain,
                              double **omega2_harmonic, std::complex<double> ***evec_harmonic,
                              RelaxationStrMode relax_mode, MinimumDistList ***mindist_list,
                              const PhaseFactorCache *phase_cache_in) const;

    void setInitialDistortion(const double (*u_tensor_in)[3]);

    void set_init_structure_atT(RelaxationStructureState &structure_state, bool &converged_prev, int &str_diverged,
                                const int i_temp_loop, double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic) const;


    void set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array, const Fcs_phonon &fcs_phonon,
                               const Ewald &ewald) const;

    // STRAIN_COUPLING bit 1 clear: clamped-ion C2 and C3 computed from the loaded
    // harmonic and cubic IFCs (ElasticTensor), converted to Ry per primitive
    // cell. C1 (reference stress) is not contained in the IFC model and is
    // still taken from C1_array.in when present (zero otherwise).
    void set_elastic_constants_from_ifcs(double *C1_array, double **C2_array, double ***C3_array,
                                         const Fcs_phonon &fcs_phonon, const Ewald &ewald) const;

    // The stress tensor at the reference structure (C1): /Elastic/stress of
    // STRAINFILE when given (zero when the dataset is absent), else
    // C1_array.in of the working directory.
    void load_reference_stress(const ElasticTensor &elastic, double *C1_array) const;

    // Schema, required groups, and the reference cell of STRAINFILE against
    // the primitive cell of this run; runs on every rank.
    // Returns the match of its /ReferenceCell atoms onto the primitive cell.
    strain_parsers::AtomMatch validate_strain_file() const;

    // piezo0, piezo2 and born_strain (and the field terms efield_strain_gradient,
    // efield_strain_curvature) from STRAINFILE /Piezoelectric; every rank.
    // match maps the per-atom Lambda blocks of /ReferenceCell onto the primitive cell.
    void load_piezo(const strain_parsers::AtomMatch &match,
                    const std::vector<SymmetryOperationWithMapping> &symops_ref);

    // Adds the pV term and, under EFIELD, the piezoelectric enthalpy
    // -Omega_ref E0_i (e0_ikl u_kl + 1/2 B_i,kl,pq u_kl u_pq) (each term when its data is nonzero).
    void renormalize_v0_from_umn(double &v0_with_umn, double v0_ref, std::array<std::array<double, 3>, 3> &eta_tensor,
                                 double *C1_array, double **C2_array, double ***C3_array,
                                 const std::array<std::array<double, 3>, 3> &u_tensor, const double pvcell) const;

    void renormalize_v1_from_umn(std::complex<double> *, const std::complex<double> *const, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    void renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse, const std::vector<int> &kmap_coarse_to_dense,
                                 std::complex<double> **, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    void renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                 std::complex<double> ***, std::complex<double> ***, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    // Renormalization by the Gamma-point displacement q0. The quartic terms
    // enter through the contraction q4_q0[ik][a][b] = sum_{c,d} v4[ik][a,b][c,d]
    // q0[c] q0[d] computed by q0_contraction::contract_v4_with_q0 together with
    // v3_renorm in a single sweep over v4; q4_gamma is the Gamma block q4_q0[g].
    void renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense,
                                std::complex<double> *v1_renorm, std::complex<double> *v1_ref,
                                std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                const std::complex<double> *const *q4_gamma, const std::vector<double> &q0) const;

    void renormalize_v2_from_q0(std::complex<double> ***evec_harmonic, const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense, const std::vector<int> &kmap_coarse_to_dense,
                                std::complex<double> ****mat_transform_sym, std::complex<double> **delta_v2_renorm,
                                std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                const std::complex<double> *const *const *q4_q0, const std::vector<double> &q0) const;

    void renormalize_v0_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense, double &v0_renorm,
                                double v0_ref, std::complex<double> *v1_ref,
                                std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                const std::complex<double> *const *q4_gamma, const std::vector<double> &q0) const;

    void calculate_u0(const std::vector<double> &q0, std::vector<double> &u0, double **omega2_harmonic,
                      std::complex<double> ***evec_harmonic) const;

    // Returns true when the step exceeded the trust region and was scaled down.
    bool update_cell_coordinate(RelaxationStructureState &, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const *const *const,
                                const std::vector<int> &, double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic,
                                const Eigen::MatrixXd *coord_hessian = nullptr) const;

    void rescue_step_after_scp_failure(RelaxationStructureState &structure_state,
                                       const std::complex<double> *const v1_array_atT,
                                       const std::vector<int> &harm_optical_modes, double **omega2_harmonic,
                                       std::complex<double> ***evec_harmonic) const;

    int detect_spacegroup(const Eigen::Matrix3d &lavec, const std::vector<Eigen::Vector3d> &xf,
                          std::string &label) const;

    void distorted_cell_of(const RelaxationStructureState &state, Eigen::Matrix3d &lavec,
                           std::vector<Eigen::Vector3d> &xf) const;

    // spglib number of the structure of a state (SET_INIT_STR = 3). When
    // label is non-null it also receives the symbol, e.g. "P4mm (#99)".
    int spacegroup_of(const RelaxationStructureState &state, std::string *label = nullptr) const;

    std::string print_structure_and_symmetry(const RelaxationStructureState &structure_state,
                                             const std::complex<double> *del_v0_del_umn_atT) const;

    static void print_optimization_history(const std::vector<StructOptStepRecord> &step_history, const double temp,
                                           const bool with_cell, const bool show_scp_column,
                                           const unsigned int verbosity = 1);

    void check_str_divergence(int &diverged, const RelaxationStructureState &structure_state) const;


    void write_resfile_header(std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const;

    void write_resfile_atT(const RelaxationStructureState &structure_state, const double temperature,
                           std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const;

    void write_stepresfile_header_atT(std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                      std::ofstream &fout_step_u_tensor, const double temp) const;

    void write_stepresfile(const RelaxationStructureState &structure_state, const int i_str_loop,
                           std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                           std::ofstream &fout_step_u_tensor) const;

    static int get_xyz_string(const int, std::string &);

    // Green-Lagrange strain eta = sym(u) + 1/2 u u^T of the deformation gradient F = I + u
    // (u is the symmetric displacement-gradient tensor used as the cell variable).
    static void calculate_eta_tensor(std::array<std::array<double, 3>, 3> &,
                                     const std::array<std::array<double, 3>, 3> &);

private:
    void set_default_variables();

    void set_initial_q0(std::vector<double> &q0, std::complex<double> ***evec_harmonic) const;


    void set_initial_strain(std::array<std::array<double, 3>, 3> &u_tensor) const;

private:
    // Collaborators (non-owning; owned by PHON, which outlives this object).
    const RunInfo &run;
    const System *system;

    // Symmetry::tolerance, cached by setup_relaxation (Symmetry::setup_symmetry,
    // which broadcasts it, runs before).
    double symprec_{0.0};
};
} // namespace PHON_NS
