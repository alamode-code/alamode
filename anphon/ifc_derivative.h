#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "fcs_phonon.h"
#include "strain_coupling_types.h"

namespace PHON_NS
{

class KpointMeshUniform;
class PhaseFactorCache;
struct MinimumDistList;
class DelVStrainData;
class System;
class Symmetry;
class Dynamical;
class AnharmonicCore;

// Real-space strain derivatives of the IFCs for ALL strain-tensor components
// at once: one entry per group of heading IFC indices, carrying the values of
// every (mu_1 nu_1 ... mu_m nu_m) component, flattened in base 9 with digit
// mu_j*3+nu_j (most significant first). Produced by
// DerivativeIFC::compute_dV_dumn_all_real_space.
struct DeltaFcsStrainComponents
{
    std::vector<AtomCellSuper> pairs;
    std::vector<unsigned int> atoms_s;
    std::vector<Eigen::Vector3d> relvecs;
    std::vector<Eigen::Vector3d> relvecs_velocity;
    std::vector<double> values; // size 9^m
    // Bit p is set iff some FC entry of this group has tail Cartesian indices
    // forming the mu-combination p (base 3, most significant first). Components
    // whose mu-combination is untouched hold an exact 0.0 that no entry wrote.
    uint32_t touched_mu;
};

// Computes strain derivatives of the IFCs for the SCPH/QHA structural
// relaxation. All dependencies are explicit constructor arguments. The SCPH/QHA
// driver builds it on every rank where it is needed, i.e. after the input is
// parsed and PHON::setup_base() has run, and passes it to Relaxation.
class DerivativeIFC
{
public:
    using MatrixXcdRowMajor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    DerivativeIFC(const System &system_in, const Symmetry &symmetry_in, const Fcs_phonon &fcs_phonon_in, int my_rank_in,
                  int nprocs_in);
    ~DerivativeIFC() = default;

    // VERBOSITY of the run; >= 2 prints the stage timers on rank 0.
    void set_verbosity(const unsigned int verbosity)
    {
        verbosity_ = verbosity;
    }

    // Compute all 9^m real-space strain derivatives in one pass. Use
    // extract_strain_component for one component. fcs_aligned must be sorted
    // by its first n-m indices (sort_by_heading_indices(m)).
    static void compute_dV_dumn_all_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                               std::vector<DeltaFcsStrainComponents> &groups, std::size_t m,
                                               const Eigen::Matrix3d &convmat);

    // Materialize the delta IFCs of one flattened component (base-9 digits
    // mu_j*3+nu_j, most significant first). A group is emitted iff its
    // mu-combination was touched and, when emit_threshold >= 0, the value's
    // magnitude exceeds the threshold.
    static void extract_strain_component(const std::vector<DeltaFcsStrainComponents> &groups, std::size_t component,
                                         std::size_t m, double emit_threshold,
                                         std::vector<FcsArrayWithCell> &delta_fcs);

    // Materialize the delta IFCs of a linear combination of components,
    // sum_i weight_i * values[component_i] (e.g. a symmetrized off-diagonal
    // strain derivative). A group is emitted iff any term's mu-combination was
    // touched and the combined value passes emit_threshold.
    static void extract_strain_combination(const std::vector<DeltaFcsStrainComponents> &groups,
                                           const std::vector<std::pair<std::size_t, double>> &terms, std::size_t m,
                                           double emit_threshold, std::vector<FcsArrayWithCell> &delta_fcs);

    // Directional derivative of the IFCs along the strain tensors strain_dirs[j]
    // (one 3x3 tensor per derivative order); Gruneisen passes the identity to
    // get the isotropic-strain derivative in a single channel.
    static void compute_dV_dstrain_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                              std::vector<FcsArrayWithCell> &delta_fcs,
                                              const std::vector<Eigen::Matrix3d> &strain_dirs,
                                              const Eigen::Matrix3d &convmat, double emit_threshold);

    // Contraction of the last IFC leg with a constant per-atom displacement
    // field: the internal-strain (sublattice relaxation) leg of the
    // relaxed-ion strain derivative. S_field(3*kappa+lambda) is the lambda-th
    // Cartesian displacement (bohr) of primitive atom kappa; each entry
    // contributes fcs_val * S_field(tail index). fcs_aligned must be sorted
    // by the first (n-1) indices (sort_by_heading_indices(1)).
    static void compute_dV_dsublattice_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                  std::vector<FcsArrayWithCell> &delta_fcs,
                                                  const Eigen::VectorXd &sublattice_displacement,
                                                  double emit_threshold);

    // Homogeneous deformation plus sublattice displacements:
    //   d_lambda = sum_nu u(lambda, nu) R_nu + S(3*kappa + lambda).
    // R is the strain-kernel relative vector (valid by ASR); u is the
    // dimensionless displacement gradient. S is in Cartesian bohr, indexed
    // by user-cell atoms as in pairs[].index/3 after replicate_force_constant.
    // S may be empty for affine deformation or periodic with a non-primitive cell.
    struct DeformationField
    {
        // Dimensionless displacement-gradient tensor (the u_tensor of the
        // SCPH/QHA structural relaxation).
        Eigen::Matrix3d displacement_gradient = Eigen::Matrix3d::Zero();
        // Cartesian sublattice displacements in bohr, indexed by
        // 3*kappa+lambda over the atoms of the user-defined primitive cell
        // (the u0 of the relaxation). May be empty (purely affine).
        Eigen::VectorXd sublattice_displacement;
    };

    // Contraction of the last fields.size() legs of the IFCs with the given
    // displacement fields (one per contracted leg). Subsumes the strain and
    // sublattice kernels above. fcs_aligned must be sorted by the first
    // (n-m) indices (sort_by_heading_indices(m)).
    static void compute_dV_ddeform_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                              std::vector<FcsArrayWithCell> &delta_fcs,
                                              const std::vector<DeformationField> &fields,
                                              const Eigen::Matrix3d &convmat, double emit_threshold);

    // IFCs of the deformed structure [Taylor renormalization, fully in real
    // space]: for the order held in fcs_by_order[target_order_index],
    //   Phi^def(n) = Phi(n) + sum_{m>=1} (1/m!) Phi(n+m) contracted with the
    //   displacement field on its m tail legs,
    // truncated at the highest order available in fcs_by_order (the usual
    // force_constant_with_cell layout: index 0 = harmonic, 1 = cubic, ...).
    // The corrections are appended to the copied base list; entries with
    // identical index groups are summed by all downstream consumers.
    static void compute_deformed_ifcs(const std::vector<const std::vector<FcsArrayWithCell> *> &fcs_by_order,
                                      std::size_t target_order_index, const Eigen::Matrix3d &displacement_gradient,
                                      const Eigen::VectorXd &sublattice_displacement, const Eigen::Matrix3d &convmat,
                                      std::vector<FcsArrayWithCell> &fcs_deformed);

    // Convenience overload for any indexable container of per-order IFC lists
    // (e.g. Fcs_phonon::force_constant_with_cell).
    template <class FcsByOrder>
    static void compute_deformed_ifcs(const FcsByOrder &fcs_by_order, const std::size_t target_order_index,
                                      const Eigen::Matrix3d &displacement_gradient,
                                      const Eigen::VectorXd &sublattice_displacement, const Eigen::Matrix3d &convmat,
                                      std::vector<FcsArrayWithCell> &fcs_deformed)
    {
        std::vector<const std::vector<FcsArrayWithCell> *> ptrs;
        for (std::size_t i = 0; i < fcs_by_order.size(); ++i) {
            ptrs.push_back(&fcs_by_order[i]);
        }
        compute_deformed_ifcs(ptrs,
                              target_order_index,
                              displacement_gradient,
                              sublattice_displacement,
                              convmat,
                              fcs_deformed);
    }

    // FC3 of a relaxed structure, Phi3 + Phi4 : d with d = u . R + u0, sorted as
    // Fcs_phonon::setup sorts the cubic list (AnharmonicCore groups consecutive
    // equal index sets). u_tensor is the displacement gradient, row-major 3x3;
    // u0 the Cartesian shifts of the primitive-cell atoms (3 natmin). The Taylor
    // expansion is about the reference structure, so reference_lattice must be the
    // *undeformed* primitive lattice: (I + u) R would make the strain term
    // second-order wrong.
    template <class FcsByOrder>
    static void compute_deformed_cubic_ifcs(const FcsByOrder &fcs_by_order, const double *u_tensor,
                                            const std::vector<double> &u0, const Eigen::Matrix3d &reference_lattice,
                                            std::vector<FcsArrayWithCell> &fc3_deformed)
    {
        Eigen::Matrix3d displacement_gradient;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) displacement_gradient(i, j) = u_tensor[3 * i + j];
        }
        const Eigen::VectorXd sublattice =
            Eigen::Map<const Eigen::VectorXd>(u0.data(), static_cast<Eigen::Index>(u0.size()));
        compute_deformed_ifcs(fcs_by_order, 1, displacement_gradient, sublattice, reference_lattice, fc3_deformed);
        std::sort(fc3_deformed.begin(), fc3_deformed.end());
    }

    void compute_dV1_dumn(MatrixXcdRowMajor &dV1_dumn,
                          const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_d2V1_dumn2(MatrixXcdRowMajor &d2V1_dumn2,
                            const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_d3V1_dumn3(MatrixXcdRowMajor &d3V1_dumn3,
                            const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_dV2_dumn(std::vector<MatrixXcdRowMajor> &dV2_dumn,
                          const std::complex<double> *const *const *const evec_harmonic, unsigned int nk,
                          const double *const *xk_in) const;

    void compute_d2V2_dumn2(std::vector<MatrixXcdRowMajor> &d2V2_dumn2,
                            const std::complex<double> *const *const *const evec_harmonic, unsigned int nk,
                            const double *const *xk_in) const;

    void compute_dV3_dumn(std::vector<std::vector<MatrixXcdRowMajor>> &dV3_dumn,
                          const std::complex<double> *const *const *const evec_harmonic,
                          const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
                          const PhaseFactorCache *phase_cache_in) const;

    void set_del_v_fixed_cell(std::size_t nk, std::size_t ns, DelVStrainData &del_v_strain) const;

    void set_del_v_relax_cell(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                              std::size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic, int renorm_2to1st, int renorm_34to1st,
                              int renorm_3to2nd, const strain_coupling::StrainSource &strain_source,
                              MinimumDistList ***mindist_list, const PhaseFactorCache *phase_cache_in) const;

    void set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                        std::size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic, int renorm_2to1st, int renorm_34to1st,
                                        int renorm_3to2nd, const strain_coupling::StrainSource &strain_source,
                                        MinimumDistList ***mindist_list) const;

private:
    const System &system_;
    const Symmetry &symmetry_;
    const Fcs_phonon &fcs_phonon_;
    const int my_rank_;
    const int nprocs_;
    unsigned int verbosity_ = 0;

    void print_stage(const std::string &label, double t_start, bool newline_first = false) const;

    void read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                       const std::complex<double> *const *const *const evec_harmonic,
                                       std::vector<MatrixXcdRowMajor> &del_v2_del_umn, unsigned int nk) const;

    // Warn when the zero strain-force coupling (STRAIN_COUPLING bit 2 clear)
    // is used in a crystal whose symmetry allows a nonzero one.
    void warn_if_strain_force_coupling_allowed(MatrixXcdRowMajor &work,
                                               const std::complex<double> *const *const *const evec_harmonic) const;

    // Strain-force coupling (STRAIN_COUPLING bit 2): load the blocks from the
    // configured source, then turn them into del_v1. The two steps are kept
    // apart so that the text files and the HDF5 container feed the same code.
    void calculate_delv1_delumn_finite_difference(MatrixXcdRowMajor &del_v1_del_umn,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  const strain_coupling::StrainSource &strain_source) const;

    strain_coupling::StrainForceSet load_strain_force_set(const strain_coupling::StrainSource &strain_source) const;

    void process_strain_force_set(const strain_coupling::StrainForceSet &set, MatrixXcdRowMajor &del_v1_del_umn,
                                  const std::complex<double> *const *const *const evec_harmonic) const;

    // Strain-harmonic-IFC coupling (STRAIN_COUPLING bit 4), same split. The
    // loader also reads the force constants of every strained supercell.
    void calculate_delv2_delumn_finite_difference(double **omega2_harmonic,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
                                                  const KpointMeshUniform *kmesh_coarse,
                                                  const KpointMeshUniform *kmesh_dense,
                                                  const strain_coupling::StrainSource &strain_source,
                                                  MinimumDistList ***mindist_list) const;

    strain_coupling::StrainHarmonicSet
    load_strain_harmonic_set(const strain_coupling::StrainSource &strain_source,
                             std::vector<std::vector<FcsArrayWithCell>> &fc2_deformed) const;

    void process_strain_harmonic_set(const std::vector<strain_coupling::StrainHarmonicEntry> &entries,
                                     const std::vector<std::vector<FcsArrayWithCell>> &fc2_deformed,
                                     double **omega2_harmonic,
                                     const std::complex<double> *const *const *const evec_harmonic,
                                     std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
                                     const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                     MinimumDistList ***mindist_list) const;
};

} // namespace PHON_NS
