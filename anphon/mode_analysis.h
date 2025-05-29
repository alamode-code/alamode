/*
 mode_analysis.h

Copyright (c) 2018 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include <complex>
#include <vector>
#include <string>
#include "anharmonic_core.h"
#include "kpoint.h"

namespace PHON_NS
{

class KsListMode
{
public:
    double xk[3]{};
    int nmode;

    KsListMode();

    KsListMode(double xk_in[3],
               const int n)
    {
        for (int i = 0; i < 3; ++i) xk[i] = xk_in[i];
        nmode = n;
    }
};

class KpointListWithCoordinate
{
public:
    double xk[3];
    double x, y;
    int plane;
    int selection_type;

    KpointListWithCoordinate();

    KpointListWithCoordinate(const std::vector<double> &a,
                             const double x_in,
                             const double y_in,
                             const int plane_in,
                             const int selection_type_in)
    {
        for (int i = 0; i < 3; ++i) xk[i] = a[i];
        x = x_in;
        y = y_in;
        plane = plane_in;
        selection_type = selection_type_in;
    }
};

class ModeAnalysis: protected Pointers
{
public:
    ModeAnalysis(class PHON *);

    ~ModeAnalysis();

    void run_mode_analysis();

    void setup_mode_analysis();

    bool ks_analyze_mode;
    bool calc_imagpart;
    bool calc_realpart;
    bool calc_fstate_omega;
    bool calc_fstate_k;
    int print_V3;
    int print_V4;
    int calc_selfenergy;
    bool spectral_func;

    std::string ks_input;
    std::vector<unsigned int> kslist;

private:
    void set_default_variables();

    void deallocate_variables() const;

    std::vector<KsListMode> kslist_fstate_k;

    void calc_frequency_resolved_final_state(const unsigned int ntemp,
                                             const double *temperature,
                                             const double omega0,
                                             const unsigned int nomegas,
                                             const double *omega,
                                             const unsigned int ik_in,
                                             const unsigned int is_in,
                                             const KpointMeshUniform *kmesh_in,
                                             const double *const *eval_in,
                                             const std::complex<double> *const *const *evec_in,
                                             double ***ret) const;

    void calc_frequency_resolved_final_state_tetrahedron(const unsigned int ntemp,
                                                         double *temperature,
                                                         const double omega0,
                                                         const unsigned int nomegas,
                                                         const double *omega,
                                                         const unsigned int ik_in,
                                                         const unsigned int is_in,
                                                         const KpointMeshUniform *kmesh_in,
                                                         const double *const *eval_in,
                                                         const std::complex<double> *const *const *evec_in,
                                                         double ***ret) const;

    void print_momentum_resolved_final_state(const unsigned int,
                                             double *,
                                             const double);

    void print_frequency_resolved_final_state(const unsigned int,
                                              double *);

    void print_V3_elements() const;

    void print_V4_elements() const;

    void print_Phi3_elements() const;

    void print_Phi4_elements() const;

    void calc_V3norm2(const unsigned int,
                      const unsigned int,
                      const std::vector<KsListGroup> &,
                      std::vector<std::vector<double>> &) const;

    void calc_V4norm2(const unsigned int,
                      const unsigned int,
                      const std::vector<KsListGroup> &,
                      std::vector<std::vector<double>> &) const;

    void calc_Phi3(const unsigned int,
                   const unsigned int,
                   const std::vector<KsListGroup> &,
                   std::vector<std::vector<std::complex<double>>> &) const;

    void calc_Phi4(const unsigned int,
                   const unsigned int,
                   const std::vector<KsListGroup> &,
                   std::vector<std::vector<std::complex<double>>> &) const;

    void print_selfenergy(const unsigned int,
                          double *);

    void print_spectral_function(const unsigned int,
                                 const double *);

};
}
