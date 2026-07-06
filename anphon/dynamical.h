/*
 dynamical.h

 Copyright (c) 2014-2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <complex>
#include <string>
#include <vector>
#include "blas_wrapper.h" // zgemm_ / zgemm_cpx (guarded for EIGEN_USE_BLAS); must follow <Eigen/...>
#include "fcs_phonon.h"
#include "kpoint.h"
#include "memory.h"
#include "pointers.h"

namespace PHON_NS
{
class DistWithCell
{
public:
    int cell;
    double dist;

    DistWithCell();

    DistWithCell(const int n, const double d) : cell(n), dist(d) {};
};

inline bool operator<(const DistWithCell a, const DistWithCell b)
{
    return a.dist < b.dist;
}

class DymatEigenValue
{
public:
    DymatEigenValue() :
        nk(0), ns(0), eval(nullptr), evec(nullptr), is_stored_eigvec(true), is_irreducible_only(false) {};

    DymatEigenValue(const bool stored_eigvec_, const bool store_irreducible_only_, const unsigned int nk_in,
                    const unsigned int ns_in) :
        nk(nk_in), ns(ns_in), is_stored_eigvec(stored_eigvec_), is_irreducible_only(store_irreducible_only_)
    {
        if (eval) deallocate(eval);
        if (evec) deallocate(evec);

        allocate(eval, nk_in, ns_in);
        if (is_stored_eigvec) {
            allocate(evec, nk_in, ns_in, ns_in);
        }
    };

    ~DymatEigenValue()
    {
        if (eval) deallocate(eval);
        if (evec) deallocate(evec);
    };

    void set_eigenvalues(const unsigned int n, double **eval_in);

    void set_eigenvectors(const unsigned int n, std::complex<double> ***evec_in);

    void set_eigenvals_and_eigenvecs(const unsigned int n, double **eval_in, std::complex<double> ***evec_in);

    double **get_eigenvalues() const;

    std::complex<double> ***get_eigenvectors() const;

private:
    unsigned int nk, ns;
    double **eval = nullptr;
    std::complex<double> ***evec = nullptr;
    bool is_stored_eigvec = true;
    bool is_irreducible_only = false;
};


class Dynamical: protected Pointers
{
public:
    Dynamical(class PHON *);

    ~Dynamical();

    unsigned int neval{};
    bool eigenvectors{};
    bool print_eigenvectors{};
    unsigned int nonanalytic{};
    bool participation_ratio{};
    unsigned int band_connection{};

    double na_sigma{};

    int **index_bconnect{};
    bool **is_imaginary{};

    DymatEigenValue *dymat_band, *dymat_general;

    void diagonalize_dynamical_all();

    void setup_dynamical();

    void eval_k(const double *, const double *, const std::vector<FcsArrayWithCell> &, double *,
                std::complex<double> **, const bool) const;

    void modify_eigenvectors() const;

    void eval_k_ewald(const double *, const double *,
                      //                      const std::vector<FcsClassExtent> &,
                      const std::vector<FcsArrayWithCell> &, double *, std::complex<double> **, const bool) const;

    double fold(const double) const;

    double freq(const double) const;

    void calc_participation_ratio_all(const unsigned int nk_in, const std::complex<double> *const *const *evec_in,
                                      double **ret, double ***ret_all) const;

    void calc_analytic_k(const double *, const std::vector<FcsClassExtent> &, std::complex<double> **) const;

    void calc_analytic_k(const double *, const std::vector<FcsArrayWithCell> &, std::complex<double> **) const;

    void calc_nonanalytic_k(const double *, const double *, std::complex<double> **) const;

    void calc_nonanalytic_k2(const double *, const double *, std::complex<double> **) const;

    void project_degenerate_eigenvectors(const Eigen::Matrix3d &lavec_p, const std::vector<FcsArrayWithCell> &fc2_in,
                                         double *xk_in, const std::vector<std::vector<double>> &project_directions,
                                         std::complex<double> **evec_out) const;

    std::vector<std::vector<double>> get_projection_directions() const;

    void set_projection_directions(const std::vector<std::vector<double>> projections_in);

    static void r2q(const double *xk_in, const unsigned int nx, const unsigned int ny, const unsigned int nz,
                    const unsigned int ns, MinimumDistList ***mindist_list_in, std::complex<double> ***dymat_r_in,
                    std::complex<double> **dymat_k_out);

    static void fourier_dymat_k_to_r(const unsigned int nk1, const unsigned int nk2, const unsigned int nk3,
                                     const unsigned int ns, const std::complex<double> *const *const *dymat_k,
                                     std::complex<double> ***dymat_r);

    void precompute_dymat_harm(const unsigned int nk_in, double **xk_in, double **kvec_in,
                               std::vector<Eigen::MatrixXcd> &dymat_short,
                               std::vector<Eigen::MatrixXcd> &dymat_long) const;


    void compute_renormalized_harmonic_frequency(
        double **omega2_out, std::complex<double> ***evec_harm_renormalized, std::complex<double> **delta_v2_renorm,
        const double *const *omega2_harmonic, const std::complex<double> *const *const *evec_harmonic,
        const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
        const std::vector<int> &kmap_interpolate_to_scph, std::complex<double> ****mat_transform_sym,
        MinimumDistList ***mindist_list, const unsigned int verbosity);

    void symmetrize_dynamical_matrix(const unsigned int ik, const KpointMeshUniform *kmesh_coarse,
                                     std::complex<double> ****mat_transform_sym, Eigen::MatrixXcd &dymat) const;

    void replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse,
                                         std::complex<double> ****mat_transform_sym,
                                         std::complex<double> ***dymat_inout) const;

    void replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse,
                                         std::complex<double> ****mat_transform_sym,
                                         std::vector<Eigen::MatrixXcd> &dymat_inout) const;

    void diagonalize_interpolated_matrix(std::complex<double> **, double *, std::complex<double> **, bool) const;

    double **get_xrs_image() const;

    void exec_interpolation(const unsigned int kmesh_orig[3], std::complex<double> ***dymat_r,
                            const unsigned int nk_dense, double **xk_dense, double **kvec_dense, double **eval_out,
                            std::complex<double> ***evec_out, const std::vector<Eigen::MatrixXcd> &dymat_short,
                            const std::vector<Eigen::MatrixXcd> &dymat_long, MinimumDistList ***mindist_list_in,
                            const bool use_precomputed_dymat = false, const bool return_sqrt = true) const;


    void calc_new_dymat_with_evec(std::complex<double> ***dymat_out, double **omega2_in,
                                  std::complex<double> ***evec_in, const KpointMeshUniform *kmesh_coarse,
                                  const std::vector<int> &kmap_interpolate_to_scph);


    void get_symmetry_gamma_dynamical(KpointMeshUniform *kmesh_in, const unsigned int natmin_in,
                                      const Eigen::MatrixXd &x_fractional_in,
                                      const std::vector<SymmetryOperationWithMapping> &symmlist,
                                      std::complex<double> ****&mat_transform_sym) const;

    void get_eigenvalues_dymat(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in,
                               const std::vector<FcsArrayWithCell> &fc2,
                               const std::vector<FcsArrayWithCell> &fc2_without_dipole_in, const bool require_evec,
                               double **eval_ret, std::complex<double> ***evec_ret);

private:
    void set_default_variables();

    void deallocate_variables();


    void prepare_mindist_list(std::vector<int> **) const;

    void calc_atomic_participation_ratio(const std::complex<double> *evec_in, double *ret) const;

    static double distance(double *, double *);

    void connect_band_by_eigen_similarity(const unsigned int nk_in, std::complex<double> ***evec,
                                          int **index_sorted) const;

    void detect_imaginary_branches(const KpointMeshUniform &kmesh_in, double **eval_in);


    std::vector<std::vector<double>> projection_directions;

    int transform_eigenvectors(double *xk_in, std::vector<double> perturb_direction, const double dk,
                               Eigen::MatrixXcd &evec_sub) const;


    void duplicate_xk_boundary(double *, std::vector<std::vector<double>> &);


    double **xshift_s;
    char UPLO{};
    std::complex<double> ***dymat{};
    std::vector<int> **mindist_list{};
};

extern "C"
{
    void zheev_(const char *jobz, const char *uplo, int *n, std::complex<double> *a, int *lda, double *w,
                std::complex<double> *work, int *lwork, double *rwork, int *info);
    // zgemm_ is declared in blas_wrapper.h (call it via zgemm_cpx).
}

// Replace per-branch data by its average over each degenerate subspace at
// one k point. eval_at_k holds the ns sorted eigenvalues; data is [ns][width]
// row-major (width = 1 for scalars, 3 for Cartesian vectors, ntemp for
// temperature rows) and is averaged column-wise within each group of
// consecutive eigenvalues closer than tol_omega (~0.01 cm^-1 by default).
inline void average_over_degenerate_modes(const int ns, const double *eval_at_k, const int width, double *data,
                                          const double tol_omega = 1.0e-7)
{
    std::vector<int> degeneracy_at_k;

    auto omega_prev = eval_at_k[0];
    auto ideg = 1;

    for (int j = 1; j < ns; ++j) {
        const auto omega_now = eval_at_k[j];
        if (std::abs(omega_now - omega_prev) < tol_omega) {
            ++ideg;
        } else {
            degeneracy_at_k.push_back(ideg);
            ideg = 1;
            omega_prev = omega_now;
        }
    }
    degeneracy_at_k.push_back(ideg);

    std::vector<double> data_sum(width);

    int is = 0;
    for (const auto ideg_now: degeneracy_at_k) {
        if (ideg_now > 1) {
            for (int l = 0; l < width; ++l)
                data_sum[l] = 0.0;
            for (int k = is; k < is + ideg_now; ++k) {
                for (int l = 0; l < width; ++l) {
                    data_sum[l] += data[k * width + l];
                }
            }
            for (int k = is; k < is + ideg_now; ++k) {
                for (int l = 0; l < width; ++l) {
                    data[k * width + l] = data_sum[l] / static_cast<double>(ideg_now);
                }
            }
        }
        is += ideg_now;
    }
}
} // namespace PHON_NS
