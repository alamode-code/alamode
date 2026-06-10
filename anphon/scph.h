/*
 scph.h

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Dense>
#include <complex>
#include "anharmonic_core.h"
#include "dynamical.h"
#include "gruneisen.h"
#include "kpoint.h"
#include "pointers.h"
#include "scph_qha_common.h"

namespace PHON_NS
{
class DistList
{
public:
    unsigned int cell_s;
    double dist;

    DistList();

    DistList(const unsigned int cell_s_, const double dist_) : cell_s(cell_s_), dist(dist_) {};

    bool operator<(const DistList &obj) const
    {
        return dist < obj.dist;
    }
};


class Scph: protected ScphQhaCommon
{
public:
    Scph(class PHON *phon);

    ~Scph();

    unsigned int kmesh_scph[3];
    unsigned int kmesh_interpolate[3];
    unsigned int bubble;

    bool restart_scph;
    bool warmstart_scph;
    bool lower_temp;
    double tolerance_scph;

    void exec_scph();

    void setup_scph();

    double mixalpha;
    unsigned int maxiter;
    bool print_self_consistent_fc2;
    double mix_anderson_ratio;
    unsigned int imix_scph;


    using ScphQhaCommon::calculate_del_v0_del_umn_renorm;
    using ScphQhaCommon::compute_anharmonic_del_v0_del_umn;
    using ScphQhaCommon::compute_anharmonic_v1_array;
    using ScphQhaCommon::compute_V3_elements_mpi_over_kpoint;
    using ScphQhaCommon::compute_V4_elements_mpi_over_band;
    using ScphQhaCommon::compute_V4_elements_mpi_over_kpoint;
    using ScphQhaCommon::ialgo;
    using ScphQhaCommon::load_scph_dymat_from_file;
    using ScphQhaCommon::postprocess;
    using ScphQhaCommon::selfenergy_offdiagonal;
    using ScphQhaCommon::store_renormalized_dymat_to_file;
    using ScphQhaCommon::write_anharmonic_correction_fc2;

    void compute_V3_elements_for_given_IFCs(std::complex<double> ***v3_out, double **omega2_harmonic_in,
                                            const int ngroup_v3_in, std::vector<double> *fcs_group_v3_in,
                                            std::vector<RelativeVector> *relvec_v3_in, double *invmass_v3_in,
                                            int **evec_index_v3_in, const std::complex<double> *const *const *evec_in,
                                            const bool self_offdiag, const KpointMeshUniform *kmesh_coarse_in,
                                            const KpointMeshUniform *kmesh_dense_in,
                                            const PhaseFactorStorage *phase_storage_in);


private:
    void set_default_variables();

    void exec_scph_main(std::complex<double> ****);

    void exec_scph_relax_cell_coordinate_main(std::complex<double> ****, std::complex<double> ****);

    void compute_anharmonic_frequency(std::complex<double> ***, double **, std::complex<double> ***, double, bool &,
                                      std::complex<double> ***, bool, std::complex<double> **,
                                      const unsigned int verbosity, const bool compact_progress = false);

    void compute_anharmonic_frequency_diis(std::complex<double> ***, double **, std::complex<double> ***,
                                           double, bool &, std::complex<double> ***, bool,
                                           std::complex<double> **, const unsigned int verbosity,
                                           const bool compact_progress = false);

    // Helper methods for compute_anharmonic_frequency
    void initialize_scph_iteration(const double temp, const bool flag_converged, double **omega2_prev,
                                   const unsigned int verbosity, Eigen::MatrixXd &omega_now, Eigen::MatrixXd &omega2_HA,
                                   std::vector<Eigen::MatrixXcd> &evec_initial,
                                   std::vector<Eigen::MatrixXcd> &evec_initial_adjoint,
                                   std::complex<double> ***cmat_convert) const;

    void setup_harmonic_dynamical_matrices(const Eigen::MatrixXd &omega2_HA,
                                           const std::vector<Eigen::MatrixXcd> &evec_initial,
                                           std::complex<double> **delta_v2_renorm, std::vector<Eigen::MatrixXcd> &Fmat0,
                                           std::complex<double> ***dymat_q_HA) const;

    void compute_qmat_and_dmat(const Eigen::MatrixXd &omega_now, const double temp,
                               std::complex<double> ***cmat_convert, std::vector<Eigen::MatrixXcd> &dmat_convert) const;

    void update_fmat_with_v4(const std::vector<Eigen::MatrixXcd> &Fmat0,
                             std::complex<double> *const *const *v4_array_all,
                             const std::vector<Eigen::MatrixXcd> &dmat_convert, const bool offdiag,
                             const unsigned int ik_irred, Eigen::MatrixXcd &Fmat) const;

    void diagonalize_and_symmetrize(const Eigen::MatrixXcd &Fmat, const std::vector<Eigen::MatrixXcd> &evec_initial,
                                    std::complex<double> ***v4_array_all, const unsigned int ik_irred,
                                    const unsigned int knum, const unsigned int knum_interpolate,
                                    const bool flag_converged, double **omega2_out, const unsigned int verbosity,
                                    int &icount, Eigen::VectorXd &eval_tmp, std::complex<double> ***dymat_q,
                                    bool *eval_repaired = nullptr) const;

    void interpolate_to_dense_mesh(std::complex<double> ***dymat_q,
                                   const std::complex<double> *const *const *dymat_q_HA,
                                   const std::vector<Eigen::MatrixXcd> &evec_initial, Eigen::MatrixXd &eval_interpolate,
                                   std::vector<Eigen::MatrixXcd> &evec_new, std::complex<double> ***cmat_convert,
                                   Eigen::MatrixXd &omega_now) const;

    bool check_convergence(const Eigen::MatrixXd &omega_now, const Eigen::MatrixXd &omega_old, const double conv_tol,
                           const unsigned int verbosity, const int iloop, double &diff) const;

    void update_frequency(const double temperature_in, const Eigen::MatrixXd &omega_in,
                          const std::vector<Eigen::MatrixXcd> &Fmat0, const std::vector<Eigen::MatrixXcd> &evec0,
                          std::complex<double> ***dymat0, std::complex<double> ***v4_array_all,
                          std::complex<double> ***cmat_convert, std::complex<double> ***dymat_out,
                          std::complex<double> ***evec_out, const bool offdiag, Eigen::MatrixXd &omega_out);


    static void get_permutation_matrix(const int ns, std::complex<double> **cmat_in,
                                       Eigen::MatrixXd &permutation_matrix);

    void find_degeneracy(std::vector<int> *degeneracy_out, unsigned int nk_in, double **eval_in) const;

    void compute_free_energy_bubble_SCPH(const unsigned int[3], std::complex<double> ****);

    void bubble_correction(std::complex<double> ****, std::complex<double> ****);

    std::vector<std::complex<double>> get_bubble_selfenergy(const KpointMeshUniform *kmesh_in, const unsigned int ns_in,
                                                            const double *const *eval_in,
                                                            const std::complex<double> *const *const *evec_in,
                                                            const unsigned int knum, const unsigned int snum,
                                                            const double temp_in,
                                                            const std::vector<std::complex<double>> &omegalist);
};

extern "C"
{
    void zgemm_(const char *transa, const char *transb, int *m, int *n, int *k, std::complex<double> *alpha,
                std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, std::complex<double> *beta,
                std::complex<double> *c, int *ldc);
}

} // namespace PHON_NS
