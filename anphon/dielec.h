/*
 dielec.h

 Copyright (c) 2019 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>
#include "ndarray.h"
#include "pointers.h"

namespace PHON_NS
{
class Dielec: protected Pointers
{
public:
    Dielec(class PHON *);

    ~Dielec();

    void init();

    void run_dielec_calculation();

    const double *get_omega_grid(unsigned int &nomega) const;

    const double *const *const *get_dielectric_func() const;

    void compute_dielectric_function(const unsigned int nomega_in, const double *omega_grid_in, double *eval_in,
                                     std::complex<double> **evec_in, double ***dielec_out);

    int calc_dielectric_constant;
    unsigned int symmetrize_borncharge{};
    std::string file_born;

    std::vector<std::vector<double>> get_zstar_mode() const;

    // Mode effective charges from caller-supplied Gamma-point eigenvectors
    // (mass-weighted, [ns][3*natmin]); the eigenvectors are not modified.
    // The complex overload is invariant under eigenvector phase choices;
    // the real overload keeps the historical ZMODE convention (real part).
    void compute_mode_effective_charge(std::vector<std::vector<std::complex<double>>> &zstar_mode,
                                       const std::complex<double> *const *evec_in) const;

    void compute_mode_effective_charge(std::vector<std::vector<double>> &zstar_mode,
                                       const std::complex<double> *const *evec_in,
                                       const bool do_normalize) const;

    bool has_borncharge() const
    {
        return static_cast<bool>(borncharge);
    }

    const double *const *const *get_borncharge() const;

    Eigen::Matrix3d get_dielec_tensor() const;

private:
    void set_default_variables();

    void deallocate_variables();

    void setup_dielectric(const unsigned int verbosity = 1);

    void compute_mode_effective_charge(std::vector<std::vector<double>> &zstar_mode,
                                       const bool do_normalize = false) const;

    void load_born(const unsigned int flag_symmborn, const unsigned int verbosity = 1);

    NDArray<double, 1> omega_grid;
    NDArray<double, 3> dielec;
    unsigned int nomega;
    double emax, emin, delta_e;

    Eigen::Matrix3d dielec_tensor;
    NDArray<double, 3> borncharge;
};
} // namespace PHON_NS
