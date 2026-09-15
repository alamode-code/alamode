/*
 mode_analysis.h

Copyright (c) 2018 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>
#include "anharmonic_core.h"
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{

class ModeAnalysis: protected Pointers
{
public:
    ModeAnalysis(class PHON *);

    ~ModeAnalysis();

    void run_mode_analysis();

    void setup_mode_analysis();

    bool ks_analyze_mode;
    bool selfenergy_mode = false; // MODE = selfenergy: targets from &kpoint, mesh from KMESH
    std::string branches_spec = "all";
    bool interpolate = false;                   // INTERPOLATE: spectral function from the interpolated Sigma matrix
    unsigned int kmesh_coarse[3] = {0, 0, 0};   // KMESH_COARSE
    double omega_range[3] = {-1.0, -1.0, -1.0}; // OMEGA_RANGE min max step (cm^-1); negative: DOS grid

    // Interpolated spectral function A(q, omega) on the distinct target q (rank 0).
    std::vector<double> spectrum_omega;                                         // cm^-1
    std::vector<std::vector<double>> spectrum_xk;                               // [q][3]
    std::vector<double> spectrum_kaxis;                                         // [q], -1 without a path
    std::vector<std::vector<std::vector<double>>> spectrum_total;               // [T][q][omega]
    std::vector<std::vector<std::vector<std::vector<double>>>> spectrum_branch; // [T][q][branch][omega]

    void run_interpolated_spectrum(const unsigned int NT, const double *T_arr);
    bool calc_imagpart;
    bool calc_realpart;
    bool calc_fstate_omega;
    int print_V3;
    int print_V4;
    int calc_selfenergy;
    bool spectral_func;

    std::string ks_input;
    std::vector<unsigned int> kslist;

    // Off-mesh targets use shifted-grid kernels.
    struct OffMeshTarget
    {
        double xk[3];
        unsigned int snum;
        unsigned int id; // index in the input order (results[])
    };
    std::vector<OffMeshTarget> kslist_offmesh;
    std::vector<unsigned int> kslist_id; // input-order index of each kslist entry (rank 0)

    // Rank-0 results in input order for PREFIX.selfenergy.h5, in the text output's units.
    struct TargetResult
    {
        double xk[3]{};
        unsigned int branch{}; // 1-based
        bool on_mesh{};
        double kaxis{-1.0};            // path coordinate for KPMODE 1 targets, -1 otherwise
        double omega{};                // harmonic frequency, cm^-1
        std::vector<double> linewidth; // 2*Gamma (FWHM), cm^-1, per T
        std::vector<double> shift_tadpole, shift_bubble, shift_loop;         // cm^-1, per T
        std::vector<double> self_omega;                                      // cm^-1
        std::vector<std::vector<double>> self_real, self_imag;               // [T][omega], cm^-1
        std::vector<double> fstate_energy;                                   // cm^-1
        std::vector<std::vector<double>> fstate_absorption, fstate_emission; // [T][energy]
    };
    mutable std::vector<TargetResult> results; // filled by the (const) printers

    bool write_text() const;

    void write_results_hdf5(const unsigned int NT, const double *T_arr) const;

private:
    void set_default_variables();

    void deallocate_variables() const;


    void calc_frequency_resolved_final_state(const unsigned int ntemp, const double *temperature, const double omega0,
                                             const unsigned int nomegas, const double *omega, const unsigned int ik_in,
                                             const unsigned int is_in, const KpointMeshUniform *kmesh_in,
                                             const double *const *eval_in,
                                             const std::complex<double> *const *const *evec_in, double ***ret) const;

    void calc_frequency_resolved_final_state_tetrahedron(
        const unsigned int ntemp, double *temperature, const double omega0, const unsigned int nomegas,
        const double *omega, const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
        const double *const *eval_in, const std::complex<double> *const *const *evec_in, double ***ret) const;

    void print_frequency_resolved_final_state(const unsigned int, double *);

    void print_V3_elements() const;

    void print_V4_elements() const;

    void print_Phi3_elements() const;

    void print_Phi4_elements() const;

    void calc_V3norm2(const unsigned int, const unsigned int, const std::vector<KsListGroup> &,
                      std::vector<std::vector<double>> &) const;

    void calc_V4norm2(const unsigned int, const unsigned int, const std::vector<KsListGroup> &,
                      std::vector<std::vector<double>> &) const;

    void calc_Phi3(const unsigned int, const unsigned int, const std::vector<KsListGroup> &,
                   std::vector<std::vector<std::complex<double>>> &) const;

    void calc_Phi4(const unsigned int, const unsigned int, const std::vector<KsListGroup> &,
                   std::vector<std::vector<std::complex<double>>> &) const;

    void print_selfenergy(const unsigned int, double *);

    void print_selfenergy_offmesh(const unsigned int NT, const double *T_arr, const size_t number_offset);

    void print_spectral_function_offmesh(const unsigned int NT, const double *T_arr, const size_t number_offset,
                                         const unsigned int nomega, const double *omega_array,
                                         const double delta_omega);

    static void kramers_kronig_real(const unsigned int nomega, const double *omega_array, const double delta_omega,
                                    const double *imag, double *real);

    // Harmonic eigenpair at an arbitrary fractional k (mesh NAC-direction convention).
    void eigen_at(const double *xk, NDArray<double, 2> &eval, NDArray<std::complex<double>, 3> &evec) const;

    // Vertex listings for the off-mesh targets: every mesh partner (multiplicity 1),
    // rows keyed by fractional coordinates. kind: 0 |V3|^2, 1 Phi3, 2 |V4|^2, 3 Phi4.
    void print_vertex_offmesh(const int kind, const size_t number_offset) const;

    // Off-mesh FSTATE_W: distribute k over ranks and reduce to rank 0.
    void calc_frequency_resolved_final_state_offmesh(const unsigned int ntemp, const double *temperature,
                                                     const unsigned int nomegas, const double *omega, const double *xq,
                                                     const double omega_q, const std::complex<double> *evec_q,
                                                     const AnharmonicCore::ShiftedGrid &sg, double ***ret) const;

    void print_frequency_resolved_final_state_offmesh(const unsigned int NT, double *T_arr, const size_t number_offset,
                                                      const double *freq_array);

    void print_spectral_function(const unsigned int, const double *);
};
} // namespace PHON_NS
