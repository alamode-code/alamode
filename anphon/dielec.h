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

    double *get_omega_grid(unsigned int &nomega) const;

    double ***get_dielectric_func() const;

    void compute_dielectric_function(const unsigned int nomega_in, double *omega_grid_in, double *eval_in,
                                     std::complex<double> **evec_in, double ***dielec_out);

    int calc_dielectric_constant;
    unsigned int symmetrize_borncharge{};
    std::string file_born;

    std::vector<std::vector<double>> get_zstar_mode() const;

    double ***get_borncharge() const;

    Eigen::Matrix3d get_dielec_tensor() const;

private:
    void set_default_variables();

    void deallocate_variables();

    void setup_dielectric(const unsigned int verbosity = 1);

    void compute_mode_effective_charge(std::vector<std::vector<double>> &zstar_mode,
                                       const bool do_normalize = false) const;

    void load_born(const unsigned int flag_symmborn, const unsigned int verbosity = 1);

    double *omega_grid;
    double ***dielec;
    unsigned int nomega;
    double emax, emin, delta_e;

    Eigen::Matrix3d dielec_tensor;
    double ***borncharge{};
};
} // namespace PHON_NS
