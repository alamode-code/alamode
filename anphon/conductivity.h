/*
 conductivity.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <fstream>
#include <set>
#include <vector>
#include "anharmonic_core.h"
#include "dynamical.h"
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{
class Conductivity: protected Pointers
{

    friend class Iterativebte;

public:
    Conductivity(class PHON *);

    ~Conductivity();

    void setup_kappa();

    void calc_anharmonic_imagself();

    void compute_kappa();

    int calc_kappa_spec;
    unsigned int ntemp;
    double **damping3;
    double **damping4;
    double ***kappa;
    double ***kappa_3only;
    double ***kappa_spec;
    double ***kappa_coherent;
    double *temperature;
    int calc_coherent;
    int write_interpolation;

    int fph_rta;
    double len_boundary;

    void set_kmesh_coarse(const unsigned int nk_in[3]);

    KpointMeshUniform *get_kmesh_coarse() const;

    void set_conductivity_params(const std::string &file_result3_in, const std::string &file_result4_in,
                                 const bool restart_3ph_in, const bool restart_4ph_in);

    bool get_restart_conductivity(const int order) const;

    std::string get_filename_results(const int order) const;

    void set_interpolator(const std::string interpolator_in)
    {
        interpolator = interpolator_in;
    };

private:
    void set_default_variables();

    void deallocate_variables();

    double ***vel, ***vel_4ph;
    std::complex<double> ****velmat;
    unsigned int nk_3ph, ns;
    int nshift_restart, nshift_restart4;
    std::vector<int> vks_l, vks_done, vks_done4;
    std::set<int> vks_job, vks_job4;
    std::string file_coherent_elems;

    unsigned int nk_coarse[3] = {};
    KpointMeshUniform *kmesh_4ph = nullptr;
    DymatEigenValue *dymat_4ph = nullptr;
    PhaseFactorStorage *phase_storage_4ph = nullptr;

    std::fstream fs_result3, fs_result4;
    std::string file_result3, file_result4;
    bool restart_flag_3ph;
    bool restart_flag_4ph;

    std::string interpolator{};

    void setup_result_io(const int mode);

    void prepare_restart(const int mode);

    void load_restart_gamma_blocks(std::fstream &fs_result, const std::string &file_result,
                                   const unsigned int nk_irred, double **damping,
                                   std::vector<int> &vks_done_out, const char *label);

    void check_consistency_restart(std::fstream &fs_result, const std::string &file_result_in,
                                   const unsigned int nk_in[3], const unsigned int nk_irred_in, const Cell &primcell,
                                   const bool classical_in, const int ismear_in, const double epsilon_in,
                                   const double tmin_in, const double tmax_in, const double delta_t_in,
                                   const std::string &file_fcs_in);

    void write_header_result(std::fstream &fs_result, const std::string &file_result, const KpointMeshUniform *kmesh_in,
                             const Cell &primcell, const bool classical_in, const int ismear_in,
                             const double epsilon_in, const double tmin_in, const double tmax_in,
                             const double delta_t_in, const std::string &file_fcs_in);

    void calc_anharmonic_imagself3();

    void calc_anharmonic_imagself4();

    void setup_kappa_4ph();

    void lifetime_from_gamma(double **&, double **&);

    void write_result_gamma(unsigned int, unsigned int, double ***, double **, int);

    void average_self_energy_at_degenerate_point(const int n, const int m, const KpointMeshUniform *kmesh_in,
                                                 const double *const *eval_in, double **damping) const;

    void compute_frequency_resolved_kappa(const int ntemp, const int smearing_method, const KpointMeshUniform *kmesh_in,
                                          const double *const *eval_in, const double *const *const *const *kappa_mode,
                                          double ***kappa_spec_out) const;

    void compute_kappa_intraband(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                 const double *const *lifetime, double ***kappa_intra, double ***kappa_spec_out) const;

    void compute_kappa_coherent(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                const double *const *gamma_total, double ***kappa_coherent_out) const;

    void interpolate_data(const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
                          const double *const *val_coarse_in, double **val_dense_out) const;
};
} // namespace PHON_NS
