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
#include <memory>
#include <string>
#include <vector>
#include "blas_wrapper.h" // zgemm_ / zgemm_cpx (guarded for EIGEN_USE_BLAS); must follow <Eigen/...>
#include "fcs_phonon.h"
#include "kpoint.h"
#include "memory.h"
#include "ndarray.h"
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
    DymatEigenValue() : nk(0), ns(0), is_stored_eigvec(true), is_irreducible_only(false) {};

    DymatEigenValue(const bool stored_eigvec_, const bool store_irreducible_only_, const unsigned int nk_in,
                    const unsigned int ns_in) :
        nk(nk_in), ns(ns_in), is_stored_eigvec(stored_eigvec_), is_irreducible_only(store_irreducible_only_)
    {
        eval.resize(nk_in, ns_in);
        if (is_stored_eigvec) {
            evec.resize(nk_in, ns_in, ns_in);
        }
    };

    DymatEigenValue(const DymatEigenValue &) = delete;
    DymatEigenValue &operator=(const DymatEigenValue &) = delete;

    void set_eigenvalues(const unsigned int n, double **eval_in);

    void set_eigenvectors(const unsigned int n, std::complex<double> ***evec_in);

    void set_eigenvals_and_eigenvecs(const unsigned int n, double **eval_in, std::complex<double> ***evec_in);

    double **get_eigenvalues();

    const double *const *get_eigenvalues() const;

    std::complex<double> ***get_eigenvectors();

    const std::complex<double> *const *const *get_eigenvectors() const;

private:
    unsigned int nk, ns;
    NDArray<double, 2> eval;
    NDArray<std::complex<double>, 3> evec;
    bool is_stored_eigvec = true;
    bool is_irreducible_only = false;
};


class Dynamical: protected Pointers
{
public:
    Dynamical(class PHON *);

    ~Dynamical();

    unsigned int neval{};
    bool require_eigenvectors{};
    bool print_eigenvectors{};
    unsigned int nonanalytic{};
    bool participation_ratio{};
    unsigned int band_connection{};

    double na_sigma{};

    NDArray<int, 2> index_bconnect;
    NDArray<bool, 2> is_imaginary;

    std::unique_ptr<DymatEigenValue> dymat_band, dymat_general;

    void diagonalize_dynamical_all();

    void setup_dynamical();

    void eval_k(const double *, const double *, const std::vector<FcsArrayWithCell> &, double *,
                std::complex<double> **, const bool) const;

    void modify_eigenvectors() const;

    void eval_k_ewald(const double *, const double *, const std::vector<FcsArrayWithCell> &, double *,
                      std::complex<double> **, const bool) const;

    double freq(const double) const;

    std::vector<bool> detect_acoustic_modes_at_gamma(const std::complex<double> *const *evec_gamma,
                                                     double projection_threshold = 0.9,
                                                     bool verbose = true) const;

    void calc_participation_ratio_all(const unsigned int nk_in, const std::complex<double> *const *const *evec_in,
                                      double **ret, double ***ret_all) const;

    void calc_analytic_k(const double *, const std::vector<FcsClassExtent> &, std::complex<double> **) const;

    void calc_analytic_k(const double *, const std::vector<FcsArrayWithCell> &, std::complex<double> **) const;

    void calc_nonanalytic_k_parlinski(const double *, const double *, std::complex<double> **) const;

    void calc_nonanalytic_k_mixedspace(const double *, const double *, std::complex<double> **) const;

    void project_degenerate_eigenvectors(const Eigen::Matrix3d &lavec_p, const std::vector<FcsArrayWithCell> &fc2_in,
                                         double *xk_in, const std::vector<std::vector<double>> &project_directions,
                                         std::complex<double> **evec_out) const;

    std::vector<std::vector<double>> get_projection_directions() const;

    void set_projection_directions(const std::vector<std::vector<double>> projections_in);

    void precompute_dymat_harm(const unsigned int nk_in, double **xk_in, double **kvec_in,
                               std::vector<Eigen::MatrixXcd> &dymat_short,
                               std::vector<Eigen::MatrixXcd> &dymat_long) const;


    void compute_renormalized_harmonic_frequency(
        double **omega2_out, std::complex<double> ***evec_harm_renormalized, std::complex<double> **delta_v2_renorm,
        const double *const *omega2_harmonic, const std::complex<double> *const *const *evec_harmonic,
        const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
        const std::vector<int> &kmap_interpolate_to_scph, std::complex<double> ****mat_transform_sym,
        MinimumDistList ***mindist_list, const unsigned int verbosity);

    void diagonalize_interpolated_matrix(std::complex<double> **, double *, std::complex<double> **, bool) const;

    const double *const *get_xrs_image() const;

    void exec_interpolation(const unsigned int kmesh_orig[3], std::complex<double> ***dymat_r,
                            const unsigned int nk_dense, const double *const *xk_dense, const double *const *kvec_dense,
                            double **eval_out, std::complex<double> ***evec_out,
                            const std::vector<Eigen::MatrixXcd> &dymat_short,
                            const std::vector<Eigen::MatrixXcd> &dymat_long, MinimumDistList ***mindist_list_in,
                            const bool use_precomputed_dymat = false, const bool return_sqrt = true) const;


    void calc_new_dymat_with_evec(std::complex<double> ***dymat_out, double **omega2_in,
                                  std::complex<double> ***evec_in, const KpointMeshUniform *kmesh_coarse,
                                  const std::vector<int> &kmap_interpolate_to_scph);

    void get_eigenvalues_dymat(const unsigned int nk_in, const double *const *xk_in, const double *const *kvec_na_in,
                               const std::vector<FcsArrayWithCell> &fc2,
                               const std::vector<FcsArrayWithCell> &fc2_without_dipole_in, const bool require_evec,
                               double **eval_ret, std::complex<double> ***evec_ret);

private:
    void set_default_variables();

    void deallocate_variables();


    void prepare_mindist_list(std::vector<int> **) const;

    void calc_atomic_participation_ratio(const std::complex<double> *evec_in, double *ret) const;

    void connect_band_by_eigen_similarity(const unsigned int nk_in, std::complex<double> ***evec,
                                          int **index_sorted) const;

    void detect_imaginary_branches(const KpointMeshUniform &kmesh_in, double **eval_in);


    std::vector<std::vector<double>> projection_directions;

    int transform_eigenvectors(double *xk_in, std::vector<double> perturb_direction, const double dk,
                               Eigen::MatrixXcd &evec_sub) const;


    void duplicate_xk_boundary(const double *, std::vector<std::vector<double>> &);


    NDArray<double, 2> xshift_s;
    char UPLO{};
    NDArray<std::vector<int>, 2> mindist_list;
};

} // namespace PHON_NS
