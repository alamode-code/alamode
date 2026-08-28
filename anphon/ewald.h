/*
 ewald.h

 Copyright (c) 2015 Tatsuro Nishimoto

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>
#include "fcs_phonon.h"
#include "ndarray.h"
#include "pointers.h"
#include "system.h"

namespace PHON_NS
{
class Gvecs
{
public:
    Eigen::Vector3d vec;

    Gvecs();

    Gvecs(const double *arr)
    {
        for (unsigned int i = 0; i < 3; ++i) {
            vec[i] = arr[i];
        }
    };

    Gvecs(const Eigen::Vector3d _vec) : vec(_vec) {};
};

class DistInfo
{
public:
    int cell;
    double dist;

    DistInfo();

    DistInfo(const int n, const double d) : cell(n), dist(d) {};

    DistInfo(const DistInfo &obj) : cell(obj.cell), dist(obj.dist) {};

    DistInfo &operator=(const DistInfo &obj) = default;

    bool operator<(const DistInfo &obj) const
    {
        return dist < obj.dist;
    }
};

class Ewald: protected Pointers
{
public:
    Ewald(class PHON *);

    ~Ewald();

    bool is_longrange, print_fc2_ewald;
    std::string file_longrange;
    double prec_ewald;
    double rate_ab;

    NDArray<int, 2> multiplicity;
    double epsilon[3][3], epsilon_inv[3][3];
    double det_epsilon;
    NDArray<double, 3> Born_charge;

    std::vector<FcsArrayWithCell> fc2_without_dipole;

    void init();

    void add_longrange_matrix(const double *, const double *, std::complex<double> **);

    // Mass-free Fourier map of the dipole-dipole force constants at the
    // Cartesian wavevector xk_cart (1/bohr), with the macroscopic (G = 0)
    // term EXCLUDED:
    //   phi_out(3*kappa+a, 3*kappa'+b)
    //     = sum_R Phi^DD_ab(0 kappa; R kappa') e^{i q.(r(R kappa') - r(0 kappa))} - (G=0 term).
    // Dropping G = 0 separates the macroscopic electric field, so the q^2
    // coefficient of this object yields the fixed-E (clamped-ion) elastic
    // response of the dipole lattice (Born & Huang, Sec. 26-27). Used by
    // ElasticTensor for the long-range-corrected elastic constants.
    void calc_dipole_fcs_q(const Eigen::Vector3d &xk_cart, Eigen::MatrixXcd &phi_out);

private:
    std::vector<Gvecs> G_vectors_fcs;
    double lambda_fcs;
    double Gmax_fcs, Lmax_fcs;
    Eigen::Vector3i nl_fcs, ng_fcs;

    std::vector<Gvecs> G_vectors_dymat;
    double lambda_dymat;
    double Gmax_dymat, Lmax_dymat;
    Eigen::Vector3i nl_dymat, ng_dymat;
    bool force_permutation_sym;

    Eigen::Matrix3d epsilon_mat, invepsilon_mat;

    NDArray<std::vector<DistInfo>, 2> distall_ewald;

    void set_default_variables();

    void deallocate_variables();

    void prepare_Ewald(const Eigen::Matrix3d &dielectric);

    void prepare_G();

    void compute_ewald_fcs();

    void compute_ewald_fcs2();

    void get_pairs_of_minimum_distance(int, const int[3], const Eigen::MatrixXd &);

    void calc_longrange_fcs(int, int, int, int, int, double *);

    void calc_real_space_sum_ewald_fcs(int, int, double **);

    void calc_reciprocal_space_sum_ewald_fcs(int, int, double **);

    void calc_short_term_dynamical_matrix(int, int, double *, std::complex<double> **);

    void calc_long_term_dynamical_matrix(const int iat, const int jat, const Eigen::Vector3d &xk_in,
                                         const Eigen::Vector3d &kvec_in, std::complex<double> **mat_out);

    void calc_realspace_sum(const int iat, const int jat, const double xdist[3], const double lambda_in,
                            std::vector<std::vector<double>> &ret);

    void calc_anisotropic_hmat(double, const double *, Eigen::Matrix3d &hmat_out) const;

    void get_lambda_and_lgmax(const Eigen::Matrix3d &lavec, const Eigen::Matrix3d &rlavec,
                              const Eigen::Matrix3d &epsilon, const Eigen::Matrix3d &epsilon_inv, const double &p,
                              double &lambda_out, double &Lmax_out, double &Gmax_out, Eigen::Vector3i &lsize,
                              Eigen::Vector3i &gsize);
};
} // namespace PHON_NS
