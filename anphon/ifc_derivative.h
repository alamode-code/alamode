#pragma once

#include <Eigen/Core>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "fcs_phonon.h"

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
// relaxation. All dependencies are explicit constructor arguments (no
// Pointers base): it is constructed by Relaxation after setup_base(), when
// every input already exists.
class DerivativeIFC
{
public:
    using MatrixXcdRowMajor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    DerivativeIFC(const System &system_in, const Symmetry &symmetry_in, const Fcs_phonon &fcs_phonon_in,
                  const Dynamical &dynamical_in, AnharmonicCore &anharmonic_core_in, int my_rank_in, int nprocs_in);
    ~DerivativeIFC() = default;

    // Single-pass computation of the m-th strain derivative of the IFCs in real
    // space for ALL 9^m strain-tensor components at once. One scan over
    // fcs_aligned replaces the 9^m per-component scans; use
    // extract_strain_component to materialize one component's delta IFCs.
    // fcs_aligned must be sorted by the first (n-m) indices
    // (sort_by_heading_indices(m)).
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
                              int renorm_3to2nd, const std::string &strain_ifc_dir, MinimumDistList ***mindist_list,
                              const PhaseFactorCache *phase_cache_in) const;

    void set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                        std::size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic, int renorm_2to1st, int renorm_34to1st,
                                        int renorm_3to2nd, const std::string &strain_ifc_dir,
                                        MinimumDistList ***mindist_list) const;

private:
    const System &system_;
    const Symmetry &symmetry_;
    const Fcs_phonon &fcs_phonon_;
    const Dynamical &dynamical_;
    AnharmonicCore &anharmonic_core_; // phi3(k) evaluation in the V3 kernel
    const int my_rank_;
    const int nprocs_;

    void read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                       const std::complex<double> *const *const *const evec_harmonic,
                                       std::vector<MatrixXcdRowMajor> &del_v2_del_umn, unsigned int nk) const;

    void calculate_delv1_delumn_finite_difference(MatrixXcdRowMajor &del_v1_del_umn,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  const std::string &strain_ifc_dir) const;

    void calculate_delv2_delumn_finite_difference(double **omega2_harmonic,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
                                                  const KpointMeshUniform *kmesh_coarse,
                                                  const KpointMeshUniform *kmesh_dense, int renorm_3to2nd,
                                                  const std::string &strain_ifc_dir,
                                                  MinimumDistList ***mindist_list) const;
};

} // namespace PHON_NS
