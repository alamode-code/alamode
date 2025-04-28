/*
 ewald.h

 Copyright (c) 2015 Tatsuro Nishimoto

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include <string>
#include <vector>
#include <complex>
#include "system.h"
#include "fcs_phonon.h"

namespace PHON_NS {
class Gvecs {
public:
    double vec[3];

    Gvecs();

    Gvecs(const double *arr)
    {
        for (unsigned int i = 0; i < 3; ++i) {
            vec[i] = arr[i];
        }
    };
};

class DistInfo {
public:
    int cell;
    double dist;

    DistInfo();

    DistInfo(const int n,
             const double d) : cell(n), dist(d) {};

    DistInfo(const DistInfo &obj) : cell(obj.cell), dist(obj.dist) {};

    bool operator<(const DistInfo &obj) const
    {
        return dist < obj.dist;
    }
};

class Ewald : protected Pointers {
public:
    Ewald(class PHON *);

    ~Ewald();

    bool is_longrange, print_fc2_ewald;
    std::string file_longrange;
    double prec_ewald;
    double rate_ab;

    int **multiplicity;
    double epsilon[3][3], epsilon_inv[3][3];
    double det_epsilon;
    double ***Born_charge;

    std::vector<FcsArrayWithCell> fc2_without_dipole;

    void init();

    void add_longrange_matrix(const double *,
                              const double *,
                              std::complex<double> **);

private:

    std::vector<Gvecs> G_vector_sub;
    double lambda_sub;
    double Gmax_sub, Lmax_sub;
    int nl_sub[3], ng_sub[3], num_l_sub, num_g_sub;

    std::vector<Gvecs> G_vector;
    double lambda;
    double Gmax, Lmax;
    int nl[3], ng[3], num_l, num_g;
    bool force_permutation_sym;

    std::vector<DistInfo> **distall_ewald;

    void set_default_variables();

    void deallocate_variables();

    void prepare_Ewald(const Eigen::Matrix3d &dielectric);

    void prepare_G();

    void compute_ewald_fcs();

    void compute_ewald_fcs2();

    void get_pairs_of_minimum_distance(int,
                                       const int [3],
                                       const Eigen::MatrixXd &) const;

    void calc_longrange_fcs(int,
                            int,
                            int,
                            int,
                            int,
                            double *);

    void calc_short_term_ewald_fcs(int,
                                   int,
                                   double **);

    void calc_long_term_ewald_fcs(int,
                                  int,
                                  double **);

    void calc_short_term_dynamical_matrix(int,
                                          int,
                                          double *,
                                          std::complex<double> **);

    void calc_long_term_dynamical_matrix(const int iat,
                                         const int jat,
                                         const double *xk_in,
                                         const double *kvec_in,
                                         std::complex<double> **mat_out);

    void calc_realspace_sum(const int iat,
                            const int jat,
                            const double xdist[3],
                            const double lambda_in,
                            std::vector<std::vector<double>> &ret);

    void calc_anisotropic_hmat(double,
                               const double *,
                               Eigen::Matrix3d &hmat_out) const;
};
}
