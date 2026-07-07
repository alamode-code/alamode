/*
phonon_dos.cpp

Copyright (c) 2014-2021 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon_dos.h"
#include <Eigen/Geometry>
#include <iomanip>
#include <vector>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"

using namespace PHON_NS;

Dos::Dos(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Dos::~Dos()
{
    deallocate_variables();
}

void Dos::set_default_variables()
{
    flag_dos = false;
    compute_dos = true;
    projected_dos = false;
    two_phonon_dos = false;
    longitudinal_projected_dos = false;
    scattering_phase_space = 0;
    auto_set_emin = true;
    auto_set_emax = true;
    emin = 0.0;
    emax = 1000.0;
}

void Dos::deallocate_variables()
{
    if (dos_phonon) {
        dos_phonon.clear();
    }
    if (pdos_phonon) {
        pdos_phonon.clear();
    }
    if (longitude_dos) {
        longitude_dos.clear();
    }
    if (dos2_phonon) {
        dos2_phonon.clear();
    }
    if (sps3_mode) {
        sps3_mode.clear();
    }
    if (sps3_with_bose) {
        sps3_with_bose.clear();
    }

    tetra_nodes_dos.reset();
    kmesh_dos.reset();
    dymat_dos.reset();
}

void Dos::create_kmesh_dos(const unsigned int nk_in[3], const std::vector<SymmetryOperation> &symmlist,
                           const Eigen::Matrix3d &rlavec_p, const bool time_reversal_sym)
{
    kmesh_dos = std::make_unique<KpointMeshUniform>(nk_in);
    kmesh_dos->setup(symmlist, rlavec_p, time_reversal_sym, true);
}

void Dos::setup()
{
    // This function must not be called before dynamical->setup_dynamical()

    MPI_Bcast(&emin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&emax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&auto_set_emin, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&auto_set_emax, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&compute_dos, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&projected_dos, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&two_phonon_dos, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scattering_phase_space, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&longitudinal_projected_dos, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (kmesh_dos.get()) {
        flag_dos = true;
    } else {
        flag_dos = false;
    }

    if (flag_dos && delta_e < eps12) exit("Dos::setup()", "Too small delta_e");

    if (flag_dos) {

        update_dos_energy_grid(emin, emax, true);

        dymat_dos = std::make_unique<DymatEigenValue>(dynamical->require_eigenvectors, false, kmesh_dos->nk, dynamical->neval);

        if (integration->ismear == -1) {
            tetra_nodes_dos = std::make_unique<TetraNodes>(kmesh_dos->nk_i[0], kmesh_dos->nk_i[1], kmesh_dos->nk_i[2]);
            tetra_nodes_dos->setup();
        } else {
            tetra_nodes_dos = std::make_unique<TetraNodes>();
        }
    }
}

void Dos::update_dos_energy_grid(const double emin_in, const double emax_in, const bool force_update)
{
    if (auto_set_emin || force_update) {
        if (emin_in < 0.0) {
            emin = emin_in;
        } else {
            emin = 0.0;
        }
    }
    if (auto_set_emax || force_update) emax = emax_in;

    n_energy = static_cast<int>((emax_in - emin_in) / delta_e) + 1;
    energy_dos.clear();
    energy_dos.resize(n_energy);
    for (auto i = 0; i < n_energy; ++i) {
        energy_dos[i] = emin_in + delta_e * static_cast<double>(i);
    }
}


void Dos::calc_dos_all()
{
    const auto nk = kmesh_dos->nk;
    const auto neval = dynamical->neval;
    NDArray<double, 2> eval;

    eval.resize(neval, nk);

    for (unsigned int j = 0; j < nk; ++j) {
        for (unsigned int k = 0; k < neval; ++k) {
            eval[k][j] = in_kayser(dymat_dos->get_eigenvalues()[j][k]);
        }
    }

    auto emin_now = std::numeric_limits<double>::max();
    auto emax_now = std::numeric_limits<double>::min();

    for (size_t j = 0; j < kmesh_dos->nk_irred; ++j) {
        const auto jj = kmesh_dos->kpoint_irred_all[j][0].knum;
        for (size_t k = 0; k < neval; ++k) {
            emin_now = std::min(emin_now, eval[k][jj]);
            emax_now = std::max(emax_now, eval[k][jj]);
        }
    }

    emax_now += delta_e;
    update_dos_energy_grid(emin_now, emax_now);

    if (compute_dos) {
        dos_phonon.resize(n_energy);
        calc_dos(nk,
                 kmesh_dos->nk_irred,
                 kmesh_dos->kmap_to_irreducible.data(),
                 eval,
                 n_energy,
                 energy_dos,
                 neval,
                 integration->ismear,
                 tetra_nodes_dos->get_ntetra(),
                 tetra_nodes_dos->get_tetras(),
                 dos_phonon);
    }

    if (projected_dos) {
        pdos_phonon.resize(system->get_primcell().number_of_atoms, n_energy);
        calc_atom_projected_dos(nk,
                                eval,
                                n_energy,
                                energy_dos,
                                pdos_phonon,
                                neval,
                                system->get_primcell().number_of_atoms,
                                integration->ismear,
                                dymat_dos->get_eigenvectors());
    }

    if (longitudinal_projected_dos) {
        longitude_dos.resize(n_energy);
        calc_longitudinal_projected_dos(nk,
                                        kmesh_dos->xk,
                                        system->get_primcell().reciprocal_lattice_vector,
                                        eval,
                                        n_energy,
                                        energy_dos,
                                        longitude_dos,
                                        neval,
                                        system->get_primcell().number_of_atoms,
                                        integration->ismear,
                                        dymat_dos->get_eigenvectors());
    }

    eval.clear();

    if (two_phonon_dos) {
        dos2_phonon.resize(kmesh_dos->nk_irred, n_energy, 4);
        calc_two_phonon_dos(dymat_dos->get_eigenvalues(), n_energy, energy_dos, integration->ismear, dos2_phonon);
    }

    if (scattering_phase_space == 1) {
        sps3_mode.resize(kmesh_dos->nk_irred, dynamical->neval, 2);
        calc_total_scattering_phase_space(dymat_dos->get_eigenvalues(), integration->ismear, sps3_mode, total_sps3);
    } else if (scattering_phase_space == 2) {
        const auto Tmin = system->Tmin;
        const auto Tmax = system->Tmax;
        const auto dT = system->dT;
        const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

        sps3_with_bose.resize(kmesh_dos->nk_irred, dynamical->neval, NT, 2);
        calc_scattering_phase_space_with_Bose(dymat_dos->get_eigenvalues(), integration->ismear, sps3_with_bose);
    }
}

void Dos::calc_dos(const unsigned int nk, const unsigned int nk_irreducible, const unsigned int *map_k,
                   const double *const *eval, const unsigned int n, const std::vector<double> &energy,
                   const unsigned int neval, const int smearing_method, const unsigned int ntetra,
                   const unsigned int *const *tetras, double *ret) const
{
    NDArray<double, 1> weight;

#ifdef _OPENMP
#pragma omp parallel private(weight)
#endif
    {
        weight.resize(nk_irreducible);
#ifdef _OPENMP
#pragma omp for
#endif
        for (int i = 0; i < n; ++i) {

            ret[i] = 0.0;

            for (int k = 0; k < neval; ++k) {
                if (smearing_method == -1) {
                    integration
                        ->calc_weight_tetrahedron(nk_irreducible, map_k, eval[k], energy[i], ntetra, tetras, weight);
                } else {
                    integration
                        ->calc_weight_smearing(nk, nk_irreducible, map_k, eval[k], energy[i], smearing_method, weight);
                }

                for (int j = 0; j < nk_irreducible; ++j) {
                    ret[i] += weight[j];
                }
            }
        }
        weight.clear();
    }
}

void Dos::calc_atom_projected_dos(const unsigned int nk, double *const *eval, const unsigned int n,
                                  const std::vector<double> &energy, double **ret, const unsigned int neval,
                                  const unsigned int natmin, const int smearing_method,
                                  std::complex<double> ***evec) const
{
    // Calculate atom projected phonon-DOS

    int i;
    NDArray<unsigned int, 1> kmap_identity;
    NDArray<double, 1> weight;
    NDArray<double, 2> proj;

    if (mympi->my_rank == 0) std::cout << " PDOS = 1 : Calculating atom-projected phonon DOS ... ";

    kmap_identity.resize(nk);
    proj.resize(neval, nk);

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    for (unsigned int iat = 0; iat < natmin; ++iat) {

        for (unsigned int imode = 0; imode < neval; ++imode) {
            for (i = 0; i < nk; ++i) {

                proj[imode][i] = 0.0;

                for (unsigned int icrd = 0; icrd < 3; ++icrd) {
                    proj[imode][i] += std::norm(evec[i][imode][3 * iat + icrd]);
                }
            }
        }
#ifdef _OPENMP
#pragma omp parallel private(weight)
#endif
        {
            weight.resize(nk);
#ifdef _OPENMP
#pragma omp for
#endif
            for (i = 0; i < n; ++i) {
                ret[iat][i] = 0.0;

                for (unsigned int k = 0; k < neval; ++k) {
                    if (smearing_method == -1) {
                        integration->calc_weight_tetrahedron(nk,
                                                             kmap_identity,
                                                             eval[k],
                                                             energy[i],
                                                             tetra_nodes_dos->get_ntetra(),
                                                             tetra_nodes_dos->get_tetras(),
                                                             weight);
                    } else {
                        integration
                            ->calc_weight_smearing(nk, nk, kmap_identity, eval[k], energy[i], smearing_method, weight);
                    }

                    for (unsigned int j = 0; j < nk; ++j) {
                        ret[iat][i] += proj[k][j] * weight[j];
                    }
                }
            }

            weight.clear();
        }
    }
    proj.clear();
    kmap_identity.clear();

    if (mympi->my_rank == 0) std::cout << " done!\n";
}


void Dos::calc_longitudinal_projected_dos(const unsigned int nk, const double *const *xk_in,
                                          const Eigen::Matrix3d &rlavec_p, double *const *eval, const unsigned int n,
                                          const std::vector<double> &energy, double *ret, const unsigned int neval,
                                          const unsigned int natmin, const int smearing_method,
                                          std::complex<double> ***evec) const
{
    // Calculate atom projected phonon-DOS

    int i;
    NDArray<unsigned int, 1> kmap_identity;
    NDArray<double, 1> weight;
    NDArray<double, 2> proj;

    double xq_cart[3];

    Eigen::Vector3d qvec;
    Eigen::Vector3cd evec_kappa;

    if (mympi->my_rank == 0)
        std::cout << " LONGITUDE_DOS = 1 : Calculating longitudinal-mode projected phonon DOS ... ";

    kmap_identity.resize(nk);
    proj.resize(neval, nk);

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    double dot_prod_sum;
    double cross_prod_sum;

    for (unsigned int ik = 0; ik < nk; ++ik) {

        for (i = 0; i < 3; ++i) xq_cart[i] = xk_in[ik][i];

        rotvec(xq_cart, xq_cart, rlavec_p, 'T');

        const double norm = std::sqrt(xq_cart[0] * xq_cart[0] + xq_cart[1] * xq_cart[1] + xq_cart[2] * xq_cart[2]);

        if (norm > eps) {
            for (i = 0; i < 3; ++i) xq_cart[i] /= norm;
        }

        for (i = 0; i < 3; ++i) qvec(i) = xq_cart[i];

        for (unsigned int imode = 0; imode < neval; ++imode) {
            dot_prod_sum = 0.0;
            cross_prod_sum = 0.0;

            for (unsigned int iat = 0; iat < natmin; ++iat) {

                for (i = 0; i < 3; ++i) {
                    evec_kappa(i) = evec[ik][imode][3 * iat + i];
                }
                dot_prod_sum += std::norm(evec_kappa.dot(qvec));
                cross_prod_sum += evec_kappa.cross(qvec).squaredNorm();
            }

            double sum_weight = dot_prod_sum + cross_prod_sum;
            if (sum_weight > eps) {
                dot_prod_sum /= sum_weight;
                //                cross_prod_sum /= sum_weight;
            }
            proj[imode][ik] = dot_prod_sum;
        }
    }

#ifdef _OPENMP
#pragma omp parallel private(weight)
#endif
    {
        weight.resize(nk);
#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < n; ++i) {
            ret[i] = 0.0;

            for (unsigned int k = 0; k < neval; ++k) {
                if (smearing_method == -1) {
                    integration->calc_weight_tetrahedron(nk,
                                                         kmap_identity,
                                                         eval[k],
                                                         energy[i],
                                                         tetra_nodes_dos->get_ntetra(),
                                                         tetra_nodes_dos->get_tetras(),
                                                         weight);
                } else {
                    integration
                        ->calc_weight_smearing(nk, nk, kmap_identity, eval[k], energy[i], smearing_method, weight);
                }

                for (unsigned int j = 0; j < nk; ++j) {
                    ret[i] += proj[k][j] * weight[j];
                }
            }
        }

        weight.clear();
    }
    proj.clear();
    kmap_identity.clear();

    if (mympi->my_rank == 0) std::cout << " done!\n";
}


void Dos::calc_two_phonon_dos(double *const *eval_in, const unsigned int n, const std::vector<double> &energy,
                              const int smearing_method, double ***ret) const
{
    int i, j;
    int jk;
    int k;

    const auto nk = kmesh_dos->nk;
    const auto ns = dynamical->neval;
    const auto nk_reduced = kmesh_dos->nk_irred;

    const int ns2 = ns * ns;

    NDArray<unsigned int, 1> kmap_identity;

    NDArray<double, 2> e_tmp;
    NDArray<double, 2> weight;

    double xk_tmp[3];

    int loc;
    NDArray<int, 1> k_pair;

    if (mympi->my_rank == 0) {
        std::cout << " TDOS = 1 : Calculating two-phonon DOS for all irreducible k points.\n";
        std::cout << "            This may take a while ... ";
    }

    kmap_identity.resize(nk);
    e_tmp.resize(2, nk);
    weight.resize(n, nk);
    k_pair.resize(nk);

    const auto &xk = kmesh_dos->xk;

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    for (int ik = 0; ik < nk_reduced; ++ik) {

        auto knum = kmesh_dos->kpoint_irred_all[ik][0].knum;

        for (jk = 0; jk < nk; ++jk) {
            for (i = 0; i < 3; ++i) xk_tmp[i] = xk[knum][i] + xk[jk][i];
            k_pair[jk] = kmesh_dos->get_knum(xk_tmp);
        }

        for (i = 0; i < n; ++i) {
            for (j = 0; j < 2; ++j) {
                ret[ik][i][j] = 0.0;
            }
        }

        for (int ib = 0; ib < ns2; ++ib) {

            int is = ib / ns;
            int js = ib % ns;
#ifdef _OPENMP
#pragma omp parallel for private(loc)
#endif
            for (jk = 0; jk < nk; ++jk) {
                loc = k_pair[jk];
                e_tmp[0][jk] = in_kayser(eval_in[jk][is] + eval_in[loc][js]);
                e_tmp[1][jk] = in_kayser(eval_in[jk][is] - eval_in[loc][js]);
            }

            if (smearing_method == -1) {

                for (j = 0; j < 2; ++j) {
#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
                    for (i = 0; i < n; ++i) {
                        integration->calc_weight_tetrahedron(nk,
                                                             kmap_identity,
                                                             e_tmp[j],
                                                             energy[i],
                                                             tetra_nodes_dos->get_ntetra(),
                                                             tetra_nodes_dos->get_tetras(),
                                                             weight[i]);
                        for (k = 0; k < nk; ++k) {
                            ret[ik][i][j] += weight[i][k];
                        }
                    }
                }

            } else {

                for (j = 0; j < 2; ++j) {
#ifdef _OPENMP
#pragma omp parallel for private(k)
#endif
                    for (i = 0; i < n; ++i) {
                        integration->calc_weight_smearing(nk,
                                                          nk,
                                                          kmap_identity,
                                                          e_tmp[j],
                                                          energy[i],
                                                          smearing_method,
                                                          weight[i]);
                        for (k = 0; k < nk; ++k) {
                            ret[ik][i][j] += weight[i][k];
                        }
                    }
                }
            }
        }
    }

    e_tmp.clear();
    weight.clear();
    kmap_identity.clear();
    k_pair.clear();

    if (mympi->my_rank == 0) {
        std::cout << "done!\n";
    }
}

void Dos::calc_total_scattering_phase_space(double *const *eval_in, const int smearing_method, double ***ret_mode,
                                            double &ret) const
{
    int i, j;

    const auto nk = kmesh_dos->nk;
    const auto ns = dynamical->neval;
    const int ns2 = ns * ns;

    NDArray<unsigned int, 1> kmap_identity;

    if (mympi->my_rank == 0) {
        std::cout << " SPS = 1 : Calculating three-phonon scattering phase space ... ";
    }

    kmap_identity.resize(nk);

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    ret = 0.0;
    auto sps_sum1 = 0.0;
    auto sps_sum2 = 0.0;

    const auto &xk = kmesh_dos->xk;

    for (int ik = 0; ik < kmesh_dos->nk_irred; ++ik) {

        const auto knum = kmesh_dos->kpoint_irred_all[ik][0].knum;
        const auto multi = kmesh_dos->weight_k[ik];

        for (int is = 0; is < ns; ++is) {

            const auto omega0 = in_kayser(eval_in[knum][is]);

            auto sps_tmp1 = 0.0;
            auto sps_tmp2 = 0.0;
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                NDArray<double, 2> e_tmp;
                NDArray<double, 1> weight;
                int js, ks;
                int loc;
                double xk_tmp[3];

                weight.resize(nk);
                e_tmp.resize(2, nk);
#ifdef _OPENMP
#pragma omp for private(i, j), reduction(+ : sps_tmp1, sps_tmp2)
#endif
                for (int ib = 0; ib < ns2; ++ib) {

                    js = ib / ns;
                    ks = ib % ns;

                    for (int jk = 0; jk < nk; ++jk) {

                        for (i = 0; i < 3; ++i) xk_tmp[i] = xk[knum][i] + xk[jk][i];
                        loc = kmesh_dos->get_knum(xk_tmp);

                        e_tmp[0][jk] = in_kayser(eval_in[jk][js] + eval_in[loc][ks]);
                        e_tmp[1][jk] = in_kayser(eval_in[jk][js] - eval_in[loc][ks]);
                    }

                    if (smearing_method == -1) {
                        integration->calc_weight_tetrahedron(nk,
                                                             kmap_identity,
                                                             e_tmp[0],
                                                             omega0,
                                                             tetra_nodes_dos->get_ntetra(),
                                                             tetra_nodes_dos->get_tetras(),
                                                             weight);

                        for (j = 0; j < nk; ++j) sps_tmp1 += weight[j];

                        integration->calc_weight_tetrahedron(nk,
                                                             kmap_identity,
                                                             e_tmp[1],
                                                             omega0,
                                                             tetra_nodes_dos->get_ntetra(),
                                                             tetra_nodes_dos->get_tetras(),
                                                             weight);
                        for (j = 0; j < nk; ++j) sps_tmp2 += weight[j];

                    } else {

                        integration
                            ->calc_weight_smearing(nk, nk, kmap_identity, e_tmp[0], omega0, smearing_method, weight);
                        for (j = 0; j < nk; ++j) sps_tmp1 += weight[j];
                        integration
                            ->calc_weight_smearing(nk, nk, kmap_identity, e_tmp[1], omega0, smearing_method, weight);
                        for (j = 0; j < nk; ++j) sps_tmp2 += weight[j];
                    }
                }
                e_tmp.clear();
                weight.clear();
            }
            sps_sum1 += multi * sps_tmp1;
            sps_sum2 += multi * sps_tmp2;

            ret_mode[ik][is][0] = sps_tmp1;
            ret_mode[ik][is][1] = sps_tmp2;
        }
    }

    kmap_identity.clear();

    ret = (sps_sum1 + 2.0 * sps_sum2) / (3.0 * static_cast<double>(std::pow(ns, 3.0)));

    if (mympi->my_rank == 0) {
        std::cout << "done!\n";
    }
}

void Dos::calc_dos_from_given_frequency(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                        const unsigned int ntetra_in, const unsigned int *const *tetras_in,
                                        double *dos_out) const
{
    const auto nk = kmesh_in->nk;
    const auto neval = dynamical->neval;
    NDArray<double, 2> eval;

    eval.resize(neval, nk);
    for (unsigned int j = 0; j < nk; ++j) {
        for (unsigned int k = 0; k < neval; ++k) {
            eval[k][j] = in_kayser(eval_in[j][k]);
        }
    }


    calc_dos(nk,
             kmesh_in->nk_irred,
             kmesh_in->kmap_to_irreducible.data(),
             eval,
             n_energy,
             energy_dos,
             neval,
             integration->ismear,
             ntetra_in,
             tetras_in,
             dos_out);

    eval.clear();
}

void Dos::calc_scattering_phase_space_with_Bose(const double *const *eval_in, const int smearing_method,
                                                double ****ret) const
{
    unsigned int i, j;
    unsigned int knum;
    double xk_tmp[3];
    NDArray<double, 2> ret_mode;
    double omega0;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    NDArray<double, 1> temperature;
    int ik, iT;
    const auto nk_irred = kmesh_dos->nk_irred;
    const auto nk = kmesh_dos->nk;
    const auto ns = dynamical->neval;
    unsigned int imode;
    NDArray<unsigned int, 1> k2_arr;
    NDArray<double, 2> recv_buf;
    const auto omega_max = emax;
    const auto omega_min = emin;

    std::vector<int> ks_g, ks_l;

    if (mympi->my_rank == 0) {
        std::cout << " SPS = 2 : Calculating three-phonon scattering phase space\n";
        std::cout << "           with the Bose distribution function ...";
    }

    const auto N = static_cast<int>((Tmax - Tmin) / dT) + 1;
    temperature.resize(N);
    for (i = 0; i < N; ++i) temperature[i] = Tmin + static_cast<double>(i) * dT;

    k2_arr.resize(nk);

    for (i = 0; i < nk_irred; ++i) {
        for (j = 0; j < ns; ++j) {
            for (unsigned int k = 0; k < N; ++k) {
                ret[i][j][k][0] = 0.0;
                ret[i][j][k][1] = 0.0;
            }
        }
    }

    ret_mode.resize(N, 2);

    const auto nks_total = nk_irred * ns;
    const auto nks_each_thread = nks_total / mympi->nprocs;
    const auto nrem = nks_total - nks_each_thread * mympi->nprocs;

    if (nrem > 0) {
        recv_buf.resize((nks_each_thread + 1) * mympi->nprocs, 2 * N);
    } else {
        recv_buf.resize(nks_total, 2 * N);
    }

    ks_g.clear();
    for (ik = 0; ik < nk_irred; ++ik) {

        knum = kmesh_dos->kpoint_irred_all[ik][0].knum;

        for (imode = 0; imode < ns; ++imode) {

            omega0 = in_kayser(eval_in[knum][imode]);
            if (omega0 < omega_min || omega0 > omega_max) continue;

            ks_g.push_back(ik * ns + imode);
        }
    }

    ks_l.clear();
    unsigned int count = 0;
    for (auto it = ks_g.begin(); it != ks_g.end(); ++it) {
        if (count % mympi->nprocs == mympi->my_rank) {
            ks_l.push_back(*it);
        }
        ++count;
    }

    unsigned int nks_tmp;
    if (ks_g.size() % mympi->nprocs > 0) {
        nks_tmp = ks_g.size() / mympi->nprocs + 1;
    } else {
        nks_tmp = ks_g.size() / mympi->nprocs;
    }
    if (ks_l.size() < nks_tmp) ks_l.push_back(-1);

    for (i = 0; i < nks_tmp; ++i) {

        int iks = ks_l[i];

        if (iks == -1) {

            for (iT = 0; iT < N; ++iT) {
                ret_mode[iT][0] = 0.0;
                ret_mode[iT][1] = 0.0;
            }

        } else {

            knum = kmesh_dos->kpoint_irred_all[iks / ns][0].knum;
            const auto snum = iks % ns;

            for (unsigned int k1 = 0; k1 < nk; ++k1) {
                for (j = 0; j < 3; ++j) xk_tmp[j] = kmesh_dos->xk[knum][j] - kmesh_dos->xk[k1][j];
                unsigned int k2 = kmesh_dos->get_knum(xk_tmp);
                k2_arr[k1] = k2;
            }

            omega0 = eval_in[knum][snum];
            calc_scattering_phase_space_with_Bose_mode(nk,
                                                       ns,
                                                       N,
                                                       omega0,
                                                       eval_in,
                                                       temperature,
                                                       k2_arr,
                                                       smearing_method,
                                                       ret_mode);
        }
        MPI_Gather(&ret_mode[0][0],
                   2 * N,
                   MPI_DOUBLE,
                   &recv_buf[mympi->nprocs * i][0],
                   2 * N,
                   MPI_DOUBLE,
                   0,
                   MPI_COMM_WORLD);
    }

    count = 0;
    for (ik = 0; ik < nk_irred; ++ik) {

        knum = kmesh_dos->kpoint_irred_all[ik][0].knum;

        for (imode = 0; imode < ns; ++imode) {

            omega0 = in_kayser(eval_in[knum][imode]);
            if (omega0 < omega_min || omega0 > omega_max) continue;

            for (iT = 0; iT < N; ++iT) {
                ret[ik][imode][iT][0] = recv_buf[count][2 * iT];
                ret[ik][imode][iT][1] = recv_buf[count][2 * iT + 1];
            }
            ++count;
        }
    }

    ret_mode.clear();
    k2_arr.clear();
    recv_buf.clear();
    temperature.clear();

    if (mympi->my_rank == 0) {
        std::cout << " done!\n";
    }
}

void Dos::calc_scattering_phase_space_with_Bose_mode(const unsigned int nk, const unsigned int ns, const unsigned int N,
                                                     const double omega, const double *const *eval,
                                                     const double *temperature, const unsigned int *k_pair,
                                                     const int smearing_method, double **ret) const
{
    int ib;
    unsigned int i, is, js, k1, k2;
    const auto ns2 = ns * ns;
    double omega1, omega2;
    double temp;
    double ret1, ret2;
    double n1, n2, f1, f2;

    NDArray<double, 2> energy_tmp;
    NDArray<double, 2> weight;
    NDArray<double, 3> delta_arr;

    NDArray<unsigned int, 1> kmap_identity;

    delta_arr.resize(nk, ns2, 2);

    kmap_identity.resize(nk);
    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

    const auto omega0 = in_kayser(omega);

#ifdef _OPENMP
#pragma omp parallel private(i, is, js, k1, k2, omega1, omega2, energy_tmp, weight)
#endif
    {
        energy_tmp.resize(2, nk);
        weight.resize(2, nk);
#ifdef _OPENMP
#pragma omp for
#endif
        for (ib = 0; ib < ns2; ++ib) {
            is = ib / ns;
            js = ib % ns;

            for (k1 = 0; k1 < nk; ++k1) {

                k2 = k_pair[k1];

                omega1 = eval[k1][is];
                omega2 = eval[k2][js];

                energy_tmp[0][k1] = in_kayser(omega1 + omega2);
                energy_tmp[1][k1] = in_kayser(omega1 - omega2);
            }

            if (smearing_method == -1) {
                for (i = 0; i < 2; ++i) {
                    integration->calc_weight_tetrahedron(nk,
                                                         kmap_identity,
                                                         energy_tmp[i],
                                                         omega0,
                                                         tetra_nodes_dos->get_ntetra(),
                                                         tetra_nodes_dos->get_tetras(),
                                                         weight[i]);
                }
            } else {
                for (i = 0; i < 2; ++i) {
                    integration->calc_weight_smearing(nk,
                                                      nk,
                                                      kmap_identity,
                                                      energy_tmp[i],
                                                      omega0,
                                                      smearing_method,
                                                      weight[i]);
                }
            }

            for (k1 = 0; k1 < nk; ++k1) {
                delta_arr[k1][ib][0] = weight[0][k1];
                delta_arr[k1][ib][1] = weight[1][k1];
            }
        }

        energy_tmp.clear();
        weight.clear();
    }

    for (unsigned int iT = 0; iT < N; ++iT) {
        temp = temperature[iT];
        ret1 = 0.0;
        ret2 = 0.0;
#ifdef _OPENMP
#pragma omp parallel for private(k1, k2, is, js, omega1, omega2, n1, n2, f1, f2), reduction(+ : ret1, ret2)
#endif
        for (ib = 0; ib < ns2; ++ib) {

            is = ib / ns;
            js = ib % ns;

            for (k1 = 0; k1 < nk; ++k1) {

                k2 = k_pair[k1];

                omega1 = eval[k1][is];
                omega2 = eval[k2][js];

                if (omega1 < eps12 || omega2 < eps12) continue;

                if (thermodynamics->classical) {
                    f1 = thermodynamics->fC(omega1, temp);
                    f2 = thermodynamics->fC(omega2, temp);
                    n1 = f1 + f2;
                    n2 = f1 - f2;
                } else {
                    f1 = thermodynamics->fB(omega1, temp);
                    f2 = thermodynamics->fB(omega2, temp);
                    n1 = f1 + f2 + 1.0;
                    n2 = f1 - f2;
                }

                ret1 += delta_arr[k1][ib][0] * n1;
                ret2 += -delta_arr[k1][ib][1] * n2;
            }
        }
        ret[iT][0] = ret1;
        ret[iT][1] = ret2;
    }

    delta_arr.clear();
    kmap_identity.clear();
}
