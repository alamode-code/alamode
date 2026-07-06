/*
 phonon_dos.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <memory>
#include "ndarray.h"
#include <vector>
#include "dynamical.h"
#include "integration.h"
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{
class Dos: protected Pointers
{
public:
    Dos(class PHON *);

    ~Dos();

    void setup();

    // Allocate and initialize the uniform DOS k mesh (KPMODE = 2). Called
    // from Kpoint::kpoint_setups; Dos owns the mesh and deletes it in the
    // destructor.
    void create_kmesh_dos(const unsigned int nk_in[3], const std::vector<SymmetryOperation> &symmlist,
                          const Eigen::Matrix3d &rlavec_p, bool time_reversal_sym);

    void calc_dos_all();

    bool flag_dos;
    bool compute_dos;
    bool projected_dos, two_phonon_dos;
    bool longitudinal_projected_dos;
    bool auto_set_emin, auto_set_emax;
    int scattering_phase_space;

    int n_energy;
    double emin, emax, delta_e;
    std::vector<double> energy_dos;
    NDArray<double, 1> dos_phonon;
    NDArray<double, 2> pdos_phonon;
    NDArray<double, 1> longitude_dos;
    NDArray<double, 3> dos2_phonon;
    double total_sps3;
    NDArray<double, 3> sps3_mode;
    NDArray<double, 4> sps3_with_bose;

    std::unique_ptr<TetraNodes> tetra_nodes_dos;
    std::unique_ptr<KpointMeshUniform> kmesh_dos;
    std::unique_ptr<DymatEigenValue> dymat_dos;

    void calc_dos_from_given_frequency(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                       const unsigned int ntetra_in, const unsigned int *const *tetras_in,
                                       double *dos_out) const;

    void update_dos_energy_grid(const double emin_in, const double emax_in, const bool force_update = false);

private:
    void set_default_variables();

    void deallocate_variables();

    void calc_dos(const unsigned int nk, const unsigned int nk_irreducible, const unsigned int *map_k,
                  const double *const *eval, const unsigned int n, const std::vector<double> &energy,
                  const unsigned int neval, const int smearing_method, const unsigned int ntetra,
                  const unsigned int *const *tetras, double *ret) const;

    void calc_atom_projected_dos(const unsigned int nk, double *const *eval, const unsigned int n,
                                 const std::vector<double> &energy, double **ret, const unsigned int neval,
                                 const unsigned int natmin, const int smearing_method,
                                 std::complex<double> ***evec) const;

    void calc_two_phonon_dos(double *const *eval, const unsigned int n, const std::vector<double> &energy,
                             const int smearing_method, double ***ret) const;

    void calc_total_scattering_phase_space(double *const *eval_in, const int smearing_method, double ***ret_mode,
                                           double &ret) const;

    void calc_scattering_phase_space_with_Bose(const double *const *eval_in, const int smearing_method,
                                               double ****ret) const;

    void calc_scattering_phase_space_with_Bose_mode(const unsigned int nk, const unsigned int ns, const unsigned int N,
                                                    const double omega, const double *const *eval,
                                                    const double *temperature, const unsigned int *k_pair,
                                                    const int smearing_method, double **ret) const;

    void calc_longitudinal_projected_dos(const unsigned int nk, const double *const *xk_in,
                                         const Eigen::Matrix3d &rlavec_p, double *const *eval, const unsigned int n,
                                         const std::vector<double> &energy, double *ret, const unsigned int neval,
                                         const unsigned int natmin, const int smearing_method,
                                         std::complex<double> ***evec) const;
};
} // namespace PHON_NS
