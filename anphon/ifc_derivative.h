#pragma once

#include <Eigen/Core>
#include <complex>
#include <string>
#include <utility>
#include <vector>
#include "fcs_phonon.h"
#include "pointers.h"

namespace PHON_NS
{

class KpointMeshUniform;
class PhaseFactorStorage;
class MinimumDistList;
class DelVStrainData;

class DerivativeIFC: protected Pointers
{
public:
    explicit DerivativeIFC(class PHON *phon);
    ~DerivativeIFC() = default;

    void compute_dV_dumn_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                    std::vector<FcsArrayWithCell> &delta_fcs,
                                    const std::vector<std::pair<int, int>> &strain_components,
                                    double emit_threshold) const;

    void compute_dV_dumn_real_space_m1(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                       std::vector<FcsArrayWithCell> &delta_fcs, int m1, int n1) const;

    void compute_dV_dumn_real_space_m2(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                       std::vector<FcsArrayWithCell> &delta_fcs,
                                       int m1, int n1,
                                       int m2, int n2) const;

    void compute_dV1_dumn(std::complex<double> **dV1_dumn,
                          const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_d2V1_dumn2(std::complex<double> **d2V1_dumn2,
                            const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_d3V1_dumn3(std::complex<double> **d3V1_dumn3,
                            const std::complex<double> *const *const *const evec_harmonic) const;

    void compute_dV2_dumn(std::complex<double> ***dV2_dumn,
                          const std::complex<double> *const *const *const evec_harmonic, unsigned int nk,
                          double **xk_in) const;

    void compute_d2V2_dumn2(std::complex<double> ***d2V2_dumn2,
                            const std::complex<double> *const *const *const evec_harmonic, unsigned int nk,
                            double **xk_in) const;

    void compute_dV3_dumn(std::complex<double> ****dV3_dumn, double **omega2_harmonic,
                          const std::complex<double> *const *const *const evec_harmonic,
                          const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
                          const PhaseFactorStorage *phase_storage_in) const;

    void set_del_v_fixed_cell(std::size_t nk, std::size_t ns, DelVStrainData &del_v_strain) const;

    void set_del_v_relax_cell(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                              std::size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic, int renorm_2to1st, int renorm_34to1st,
                              int renorm_3to2nd, const std::string &strain_ifc_dir, MinimumDistList ***mindist_list,
                              const PhaseFactorStorage *phase_storage_in) const;

    void set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                        std::size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic,
                                        int renorm_2to1st, int renorm_34to1st, int renorm_3to2nd,
                                        const std::string &strain_ifc_dir, MinimumDistList ***mindist_list) const;

private:
    void read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                       const std::complex<double> *const *const *const evec_harmonic,
                                       std::complex<double> ***del_v2_del_umn, unsigned int nk) const;

    void calculate_delv1_delumn_finite_difference(std::complex<double> **del_v1_del_umn,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  const std::string &strain_ifc_dir) const;

    void calculate_delv2_delumn_finite_difference(double **omega2_harmonic,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  std::complex<double> ***del_v2_del_umn,
                                                  const KpointMeshUniform *kmesh_coarse,
                                                  const KpointMeshUniform *kmesh_dense, int renorm_3to2nd,
                                                  const std::string &strain_ifc_dir,
                                                  MinimumDistList ***mindist_list) const;

    void make_supercell_mapping_by_symmetry_operations(int **symm_mapping_s) const;

    void make_inverse_translation_mapping(int **inv_translation_mapping) const;
};

} // namespace PHON_NS
