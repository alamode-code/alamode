#include "iterativebte.h"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "isotope.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "phonon_velocity.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

//mpic++ -o3 -std=c++11 -I../include -I/Users/wenhao/mylib/include -I/Users/wenhao/mylib/spg/include -I/Users/wenhao/mylib/fftw/3.3.9/include -c iterativebte.cpp
//
// DONE: test tetrahedron method
// DONE: test with isotope
// TODO: test with more grid
// TODO: calculation from a restart
// TODO: write .result file

using namespace PHON_NS;

Iterativebte::Iterativebte(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Iterativebte::~Iterativebte()
{
    deallocate_variables();
}

void Iterativebte::set_default_variables()
{
    // public
    do_iterative = true;
    Temperature = nullptr;
    ntemp = 0;
    min_cycle = 5;
    max_cycle = 20;
    mixing_factor = 0.9;
    convergence_criteria = 0.02;
    kappa = nullptr;
    use_triplet_symmetry = true;
    sym_permutation = false;

    // private
    vel = nullptr;
    dFold = nullptr;
    dFnew = nullptr;
    L_absorb = nullptr;
    L_emitt = nullptr;
    damping4 = nullptr;
}

void Iterativebte::deallocate_variables()
{
    // Temperature is a non-owning alias of conductivity->temperature.
    if (kappa) {
        deallocate(kappa);
    }
    if (vel) {
        deallocate(vel);
    }
    if (dFold) {
        deallocate(dFold);
    }
    if (dFnew) {
        deallocate(dFnew);
    }
    if (L_absorb) {
        deallocate(L_absorb);
    }
    if (L_emitt) {
        deallocate(L_emitt);
    }
    if (damping4) {
        deallocate(damping4);
    }
}

void Iterativebte::setup_iterative()
{
    nk_3ph = dos->kmesh_dos->nk;
    ns = dynamical->neval;
    ns2 = ns * ns;

    MPI_Bcast(&max_cycle, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_cycle, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mixing_factor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&convergence_criteria, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // The 4ph channel follows conductivity->fph_rta, decided by the
    // INCLUDE_4PH tag at parse time (same as SOLVER = RTA).

    // Temperature grid in K, owned by Conductivity (non-owning alias here).
    conductivity->init_temperature_grid();
    ntemp = conductivity->ntemp;
    Temperature = conductivity->temperature;

    // Full-grid velocities in atomic units on every rank (calc_kappa and
    // the boundary rate convert units at the point of use).
    phonon_velocity->gather_group_velocities_mesh(*dos->kmesh_dos, system->get_primcell().lattice_vector, vel, 1.0,
                                                  true);

    allocate(kappa, ntemp, 3, 3);

    // distribute q point among the processors
    auto nk_ir = dos->kmesh_dos->nk_irred;

    nk_l.clear();
    for (auto i = 0; i < nk_ir; ++i) {
        if (i % mympi->nprocs == mympi->my_rank) nk_l.push_back(i);
    }

    nklocal = nk_l.size();

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " Iterative solution" << '\n';
        std::cout << " ==================" << '\n';
        std::cout << " MIN_CYCLE = " << min_cycle << ", MAX_CYCLE = " << max_cycle << '\n';
        std::cout << " ITER_THRESHOLD = " << std::setw(10) << std::right << std::setprecision(4) << convergence_criteria
                  << '\n';
        std::cout << '\n';
        std::cout << " Distribute q point ... ";
        std::cout << " Number of q point pre process: " << std::setw(5) << nklocal << '\n';
        std::cout << '\n';
    }

    get_triplets();

    write_result();
}

void Iterativebte::get_triplets()
{
    localnk_triplets_emitt.clear();  // pairs k3 = k1 - k2 ( -k1 + k2 + k3 = G )
    localnk_triplets_absorb.clear(); // pairs k3 = k1 + k2 (  k1 + k2 + k3 = G )

    int counter = 0;
    int counter2 = 0;

    for (unsigned int i = 0; i < nklocal; ++i) {

        auto ik = nk_l[i];
        std::vector<KsListGroup> triplet;
        std::vector<KsListGroup> triplet2;

        // k3 = k1 - k2
        dos->kmesh_dos->get_unique_triplet_k(ik, symmetry->SymmList, use_triplet_symmetry, sym_permutation, triplet);

        // k3 = - (k1 + k2)
        dos->kmesh_dos
            ->get_unique_triplet_k(ik, symmetry->SymmList, use_triplet_symmetry, sym_permutation, triplet2, 1);

        counter += triplet.size();
        counter2 += triplet2.size();

        localnk_triplets_emitt.push_back(triplet);
        localnk_triplets_absorb.push_back(triplet2);
    }

    kplength_emitt = counter; // remember number of unique pairs
    kplength_absorb = counter2;

    // Flattened triplet index shared by setup_L_*, calc_Q_from_L and the
    // iteration loop: row of triplet j of local k point ik in L is
    // offset_*[ik] + j.
    pairs_emitt.clear();
    pairs_absorb.clear();
    offset_emitt.assign(nklocal, 0);
    offset_absorb.assign(nklocal, 0);

    for (int ik = 0; ik < nklocal; ++ik) {
        offset_emitt[ik] = static_cast<int>(pairs_emitt.size());
        for (size_t j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {
            pairs_emitt.push_back({ik, static_cast<int>(j)});
        }
    }
    for (int ik = 0; ik < nklocal; ++ik) {
        offset_absorb[ik] = static_cast<int>(pairs_absorb.size());
        for (size_t j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {
            pairs_absorb.push_back({ik, static_cast<int>(j)});
        }
    }
    if (pairs_emitt.size() != static_cast<size_t>(kplength_emitt)) {
        exit("get_triplets", "Emitt: pair length not equal!");
    }
    if (pairs_absorb.size() != static_cast<size_t>(kplength_absorb)) {
        exit("get_triplets", "absorb: pair length not equal!");
    }
}

void Iterativebte::do_iterativebte()
{
    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " Calculate once for the transition probability L(absorb) and L(emitt)" << '\n';
        std::cout << " Size of L (MB) (approx.) = "
                  << memsize_in_MB(sizeof(double), kplength_absorb + kplength_emitt, ns, ns2) << " ... "
                  << std::flush;
    }

    if (integration->ismear >= 0) {
        setup_L_smear();
    } else if (integration->ismear == -1) {
        setup_L_tetra();
    }

    if (mympi->my_rank == 0) {
        std::cout << "     DONE !" << '\n';
    }

    iterative_solver();

    write_kappa_iterative();
}


void Iterativebte::setup_L_smear()
{
    // we calculate V for all pairs L+(local_nk*eachpair,ns,ns2) and L-

    allocate(L_absorb, kplength_absorb, ns, ns2);
    allocate(L_emitt, kplength_emitt, ns, ns2);

    const auto epsilon = integration->epsilon;

    const auto omega_tmp = dos->dymat_dos->get_eigenvalues();
    const auto evec_tmp = dos->dymat_dos->get_eigenvectors();

    // The loops run over the flattened triplet index (pairs_emitt/absorb,
    // built in get_triplets) and parallelize over triplets with the
    // thread-safe serial V3 (per-thread reciprocal-FC3 workspace).

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::complex<double>> phi3_work(anharmonic_core->get_ngroup_fcs(3));
        int kindex_work[2] = {-1, -1};
        unsigned int arr_loc[3];
        std::array<double, 2> epsilon2;

        // emitt k1 -> k2 + k3
        // V(-q1, q2, q3) delta(w1 - w2 - w3)
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int idx = 0; idx < static_cast<int>(pairs_emitt.size()); ++idx) {

            const auto ik = pairs_emitt[idx][0];
            const auto j = pairs_emitt[idx][1];
            const int kk1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ik]][0].knum;
            const auto &pair = localnk_triplets_emitt[ik][j];
            const int kk2 = pair.group[0].ks[0];
            const int kk3 = pair.group[0].ks[1];

            for (int is1 = 0; is1 < ns; ++is1) {
                arr_loc[0] = dos->kmesh_dos->kindex_minus_xk[kk1] * ns + is1;
                const auto w1 = omega_tmp[kk1][is1];

                for (int ib = 0; ib < ns2; ++ib) {
                    const int is2 = ib / ns;
                    const int is3 = ib % ns;

                    arr_loc[1] = kk2 * ns + is2;
                    arr_loc[2] = kk3 * ns + is3;
                    const auto w2 = omega_tmp[kk2][is2];
                    const auto w3 = omega_tmp[kk3][is3];

                    double delta_loc = 0.0;
                    if (integration->ismear == 0) {
                        delta_loc = delta_lorentz(w1 - w2 - w3, epsilon);
                    } else if (integration->ismear == 1) {
                        delta_loc = delta_gauss(w1 - w2 - w3, epsilon);
                    } else if (integration->ismear == 2) {
                        integration->adaptive_sigma->get_sigma(kk2, is2, kk3, is3, epsilon2);
                        delta_loc = delta_gauss(w1 - w2 - w3, epsilon2[0]);
                    }

                    const auto v3_tmp2 = std::norm(anharmonic_core->V3(arr_loc, dos->kmesh_dos->xk, omega_tmp,
                                                                       evec_tmp, phi3_work.data(), kindex_work));

                    L_emitt[idx][is1][ib] = (pi / 4.0) * v3_tmp2 * delta_loc / static_cast<double>(nk_3ph);
                }
            }
        }

        // absorption k1 + k2 -> -k3
        // V(q1, q2, q3) since k3 = - (k1 + k2)
#ifdef _OPENMP
#pragma omp for
#endif
        for (int idx = 0; idx < static_cast<int>(pairs_absorb.size()); ++idx) {

            const auto ik = pairs_absorb[idx][0];
            const auto j = pairs_absorb[idx][1];
            const int kk1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ik]][0].knum;
            const auto &pair = localnk_triplets_absorb[ik][j];
            const int kk2 = pair.group[0].ks[0];
            const int kk3 = pair.group[0].ks[1];

            for (int is1 = 0; is1 < ns; ++is1) {
                arr_loc[0] = kk1 * ns + is1;
                const auto w1 = omega_tmp[kk1][is1];

                for (int ib = 0; ib < ns2; ++ib) {
                    const int is2 = ib / ns;
                    const int is3 = ib % ns;

                    arr_loc[1] = kk2 * ns + is2;
                    arr_loc[2] = kk3 * ns + is3;
                    const auto w2 = omega_tmp[kk2][is2];
                    const auto w3 = omega_tmp[kk3][is3];

                    double delta_loc = 0.0;
                    if (integration->ismear == 0) {
                        delta_loc = delta_lorentz(w1 + w2 - w3, epsilon);
                    } else if (integration->ismear == 1) {
                        delta_loc = delta_gauss(w1 + w2 - w3, epsilon);
                    } else if (integration->ismear == 2) {
                        integration->adaptive_sigma->get_sigma(kk2, is2, kk3, is3, epsilon2);
                        // epsilon2[1] is built from (v2 + v3), the gradient of the
                        // argument of delta(w1 + w2 - w3) over the mesh cell —
                        // the same channel convention as the SERTA path.
                        delta_loc = delta_gauss(w1 + w2 - w3, epsilon2[1]);
                    }

                    const auto v3_tmp2 = std::norm(anharmonic_core->V3(arr_loc, dos->kmesh_dos->xk, omega_tmp,
                                                                       evec_tmp, phi3_work.data(), kindex_work));

                    L_absorb[idx][is1][ib] = (pi / 4.0) * v3_tmp2 * delta_loc / static_cast<double>(nk_3ph);
                }
            }
        }
    }
}

void Iterativebte::setup_L_tetra()
{
    // we calculate V for all pairs L+(local_nk*eachpair,ns,ns2) and L-

    allocate(L_absorb, kplength_absorb, ns, ns2);
    allocate(L_emitt, kplength_emitt, ns, ns2);

    unsigned int *kmap_identity;
    allocate(kmap_identity, nk_3ph);
    for (auto i = 0; i < nk_3ph; ++i)
        kmap_identity[i] = i;

    const auto omega_tmp = dos->dymat_dos->get_eigenvalues();
    const auto evec_tmp = dos->dymat_dos->get_eigenvectors();

    // Pass 1: tetrahedron weights into L (per-thread energy/weight buffers).
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double *energy_tmp;
        double *weight_tetra;
        allocate(energy_tmp, nk_3ph);
        allocate(weight_tetra, nk_3ph);
        double xk_tmp[3];

#ifdef _OPENMP
#pragma omp for
#endif
        for (int iks = 0; iks < nklocal * ns * ns2; ++iks) {

            const int ik = iks / (ns * ns2);
            const int is1 = (iks / ns2) % ns;
            const int ib = iks % ns2;
            const int is2 = ib / ns;
            const int is3 = ib % ns;

            const int kk1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ik]][0].knum;
            const auto w1 = omega_tmp[kk1][is1];

            // emission: delta(w1 - w2 - w3) with k3 = k1 - k2
            for (int k2 = 0; k2 < nk_3ph; k2++) {
                for (auto i = 0; i < 3; ++i) {
                    xk_tmp[i] = dos->kmesh_dos->xk[kk1][i] - dos->kmesh_dos->xk[k2][i];
                }
                const auto k3 = dos->kmesh_dos->get_knum(xk_tmp);
                energy_tmp[k2] = omega_tmp[k2][is2] + omega_tmp[k3][is3];
            }
            integration->calc_weight_tetrahedron(nk_3ph,
                                                 kmap_identity,
                                                 energy_tmp,
                                                 w1,
                                                 dos->tetra_nodes_dos->get_ntetra(),
                                                 dos->tetra_nodes_dos->get_tetras(),
                                                 weight_tetra);

            for (size_t j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {
                const auto &pair = localnk_triplets_emitt[ik][j];
                L_emitt[offset_emitt[ik] + j][is1][ib] = (pi / 4.0) * weight_tetra[pair.group[0].ks[0]];
            }

            // absorption: delta(w1 + w2 - w3) with k3 = -(k1 + k2)
            for (int k2 = 0; k2 < nk_3ph; k2++) {
                for (auto i = 0; i < 3; ++i) {
                    xk_tmp[i] = dos->kmesh_dos->xk[kk1][i] + dos->kmesh_dos->xk[k2][i];
                }
                const auto k3 = dos->kmesh_dos->get_knum(xk_tmp);
                energy_tmp[k2] = -omega_tmp[k2][is2] + omega_tmp[k3][is3];
            }
            integration->calc_weight_tetrahedron(nk_3ph,
                                                 kmap_identity,
                                                 energy_tmp,
                                                 w1,
                                                 dos->tetra_nodes_dos->get_ntetra(),
                                                 dos->tetra_nodes_dos->get_tetras(),
                                                 weight_tetra);

            for (size_t j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {
                const auto &pair = localnk_triplets_absorb[ik][j];
                L_absorb[offset_absorb[ik] + j][is1][ib] = (pi / 4.0) * weight_tetra[pair.group[0].ks[0]];
            }
        }

        deallocate(energy_tmp);
        deallocate(weight_tetra);
    }

    // Pass 2: multiply |V3|^2, parallel over triplets so the per-triplet
    // reciprocal-FC3 cache in the per-thread workspace is reused across the
    // ns^3 band combinations (the previous single-pass loop recomputed it
    // ns^2 times per triplet). V3 is skipped where the weight vanished.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<std::complex<double>> phi3_work(anharmonic_core->get_ngroup_fcs(3));
        int kindex_work[2] = {-1, -1};
        unsigned int arr_loc[3];

#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int idx = 0; idx < static_cast<int>(pairs_emitt.size()); ++idx) {

            const auto ik = pairs_emitt[idx][0];
            const auto j = pairs_emitt[idx][1];
            const int kk1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ik]][0].knum;
            const auto &pair = localnk_triplets_emitt[ik][j];
            const int kk2 = pair.group[0].ks[0];
            const int kk3 = pair.group[0].ks[1];

            // emitt k1 -> k2 + k3 : V(-q1, q2, q3)
            for (int is1 = 0; is1 < ns; ++is1) {
                arr_loc[0] = dos->kmesh_dos->kindex_minus_xk[kk1] * ns + is1;
                for (int ib = 0; ib < ns2; ++ib) {
                    if (L_emitt[idx][is1][ib] == 0.0) continue;
                    arr_loc[1] = kk2 * ns + ib / ns;
                    arr_loc[2] = kk3 * ns + ib % ns;
                    L_emitt[idx][is1][ib] *= std::norm(anharmonic_core->V3(arr_loc, dos->kmesh_dos->xk, omega_tmp,
                                                                           evec_tmp, phi3_work.data(), kindex_work));
                }
            }
        }

#ifdef _OPENMP
#pragma omp for
#endif
        for (int idx = 0; idx < static_cast<int>(pairs_absorb.size()); ++idx) {

            const auto ik = pairs_absorb[idx][0];
            const auto j = pairs_absorb[idx][1];
            const int kk1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ik]][0].knum;
            const auto &pair = localnk_triplets_absorb[ik][j];
            const int kk2 = pair.group[0].ks[0];
            const int kk3 = pair.group[0].ks[1];

            // absorption k1 + k2 -> -k3 : V(q1, q2, q3)
            for (int is1 = 0; is1 < ns; ++is1) {
                arr_loc[0] = kk1 * ns + is1;
                for (int ib = 0; ib < ns2; ++ib) {
                    if (L_absorb[idx][is1][ib] == 0.0) continue;
                    arr_loc[1] = kk2 * ns + ib / ns;
                    arr_loc[2] = kk3 * ns + ib % ns;
                    L_absorb[idx][is1][ib] *= std::norm(anharmonic_core->V3(arr_loc, dos->kmesh_dos->xk, omega_tmp,
                                                                            evec_tmp, phi3_work.data(), kindex_work));
                }
            }
        }
    }

    deallocate(kmap_identity);
}


void Iterativebte::calc_Q_from_L(double **&n, double **&q1)
{
    int s1, s2, s3;
    double n1, n2, n3;

    double **Qemit;
    double **Qabsorb;
    allocate(Qemit, nklocal, ns);
    allocate(Qabsorb, nklocal, ns);

    for (auto ik = 0; ik < nklocal; ++ik) {
        for (s1 = 0; s1 < ns; ++s1) {
            Qemit[ik][s1] = 0.0;
            Qabsorb[ik][s1] = 0.0;
        }
    }

    // emit
    for (auto ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto j = 0; j < localnk_triplets_emitt[ik].size(); ++j) {

            auto pair = localnk_triplets_emitt[ik][j];
            auto multi = static_cast<double>(pair.group.size());
            auto k2 = pair.group[0].ks[0];
            auto k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                n1 = n[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    n2 = n[k2][s2];
                    n3 = n[k3][s3];
                    Qemit[ik][s1] += 0.5 * (n1 * (n2 + 1.0) * (n3 + 1.0)) * L_emitt[offset_emitt[ik] + j][s1][ib] *
                                     multi;
                }
            }
        }
    }

    // absorb k1 + k2 -> -k3
    for (auto ik = 0; ik < nklocal; ++ik) {

        auto tmpk = nk_l[ik];
        const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

        for (auto j = 0; j < localnk_triplets_absorb[ik].size(); ++j) {

            auto pair = localnk_triplets_absorb[ik][j];
            auto multi = static_cast<double>(pair.group.size());
            auto k2 = pair.group[0].ks[0];
            auto k3 = pair.group[0].ks[1];

            for (s1 = 0; s1 < ns; ++s1) {
                n1 = n[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    n2 = n[k2][s2];
                    n3 = n[k3][s3];
                    Qabsorb[ik][s1] += (n1 * n2 * (n3 + 1.0)) * L_absorb[offset_absorb[ik] + j][s1][ib] * multi;
                }
            }
        }
    }

    for (auto ik = 0; ik < nklocal; ++ik) {
        for (s1 = 0; s1 < ns; ++s1) {
            q1[ik][s1] = Qemit[ik][s1] + Qabsorb[ik][s1];
        }
    }
    deallocate(Qemit);
    deallocate(Qabsorb);
}

void Iterativebte::calc_damping4()
{
    // 4ph linewidths from the SERTA machinery in Conductivity, interpolated
    // onto the dense 3ph mesh and scattered to the local k points here.
    double **damping4_dense = nullptr;
    allocate(damping4_dense, dos->kmesh_dos->nk_irred * ns, ntemp);

    conductivity->compute_damping4_interpolated(dos->kmesh_dos, damping4_dense);

    allocate(damping4, ntemp, nklocal, ns);
    for (auto ik = 0; ik < nklocal; ++ik) {
        auto tmpk = nk_l[ik];
        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            for (auto is = 0; is < ns; ++is) {
                damping4[itemp][ik][is] = damping4_dense[tmpk * ns + is][itemp];
            }
        }
    }

    deallocate(damping4_dense);
}

void Iterativebte::iterative_solver()
{
    // f_new = f_new * mixing_factor + f_old * (1 - mixing_factor)
    std::vector<double> convergence_history; // store | f_n - f_{n-1} | L2 norm
    convergence_history.clear();

    double **Q;
    double **kappa_new;
    double **kappa_old;

    allocate(kappa_new, 3, 3);
    allocate(kappa_old, 3, 3);
    allocate(Q, nklocal, ns);

    allocate(dFold, nk_3ph, ns, 3);

    double **fb;
    double **dndt;
    allocate(dndt, nklocal, ns);
    allocate(fb, nk_3ph, ns);

    double **isotope_damping_loc;
    if (isotope->include_isotope) {
        double **isotope_damping;
        allocate(isotope_damping, dos->kmesh_dos->nk_irred, ns);

        if (mympi->my_rank == 0) {
            for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ik++) {
                for (auto is = 0; is < ns; is++) {
                    isotope_damping[ik][is] = isotope->gamma_isotope[ik][is];
                }
            }
        }

        MPI_Bcast(&isotope_damping[0][0], dos->kmesh_dos->nk_irred * ns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        allocate(isotope_damping_loc, nklocal, ns); // this is for reducing some memory usage
        for (auto ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            for (auto is = 0; is < ns; is++) {
                isotope_damping_loc[ik][is] = isotope_damping[tmpk][is];
            }
        }

        deallocate(isotope_damping);
    }

    double **boundary_damping_loc;
    if (conductivity->len_boundary > eps) {

        allocate(boundary_damping_loc, nklocal, ns);

        for (auto ik = 0; ik < nklocal; ++ik) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum; // k index in full grid
            for (auto is = 0; is < ns; is++) {
                auto vel_norm = 0.0;
                for (auto j = 0; j < 3; ++j) {
                    vel_norm += vel[k1][is][j] * vel[k1][is][j];
                }
                // vel here is in atomic units (unlike the SERTA path, which
                // converts to m/s first): |v_au| * Bohr[m] / L[m] equals the
                // SERTA expression |v_SI| / L * time_ry, i.e. the boundary
                // rate in the same internal units as gamma.
                vel_norm = std::sqrt(vel_norm);
                boundary_damping_loc[ik][is] =
                    vel_norm * Bohr_in_Angstrom * 1.0e-10 / conductivity->len_boundary;
            }
        }
    }

    if (conductivity->fph_rta > 0) {
        calc_damping4();
    }

    if (mympi->my_rank == 0) {
        std::cout << '\n' << " Iteration starts ..." << '\n' << '\n';
    }


    // we solve iteratively for each temperature
    int ik, is, ix, iy;

    const int nsym = symmetry->SymmList.size();

    // ------------------------------------------------------------------
    // Symmetry expansion table for the irreducible-wedge iteration.
    // Every full-grid point p equals (time reversal x) R applied to its
    // irreducible representative, and the deviation function transforms
    // as a Cartesian vector [f_{Rk} = R f_k, f_{-k} = -f_k], so
    //   dF(p) = expand_mat[p] . dF(rep(p)),
    // with the time-reversal sign folded into the matrix. The update is
    // therefore evaluated on the wedge only and the full-grid dF is
    // reconstructed after each cycle. knum_sym applies (S^{-1})^T to the
    // fractional k, and k_cart = M k_frac with M columns the reciprocal
    // lattice vectors, hence R_cart = M (S^{-1})^T M^{-1}.
    // ------------------------------------------------------------------
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    std::vector<Eigen::Matrix3d> expand_mat(nk_3ph);
    {
        Eigen::Matrix3d mat_k2cart;
        const auto &rlavec = system->get_primcell().reciprocal_lattice_vector;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                mat_k2cart(j, i) = rlavec(i, j); // columns = reciprocal vectors
            }
        }
        const Eigen::Matrix3d mat_cart2k = mat_k2cart.inverse();

        for (unsigned int tmpk = 0; tmpk < nk_irred; ++tmpk) {
            const auto kref = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
            for (const auto &kp: dos->kmesh_dos->kpoint_irred_all[tmpk]) {
                const auto p = kp.knum;
                auto found = false;
                for (auto isym = 0; isym < nsym && !found; ++isym) {
                    const auto krot = dos->kmesh_dos->knum_sym(kref, symmetry->SymmList[isym].rotation);
                    const auto direct = (p == krot);
                    const auto time_reversed =
                        symmetry->time_reversal_sym && p == dos->kmesh_dos->kindex_minus_xk[krot];
                    if (!direct && !time_reversed) continue;
                    const Eigen::Matrix3d srot_inv_t =
                        symmetry->SymmList[isym].rotation.cast<double>().inverse().transpose();
                    const Eigen::Matrix3d rot_cart = mat_k2cart * srot_inv_t * mat_cart2k;
                    expand_mat[p] = direct ? rot_cart : Eigen::Matrix3d(-rot_cart);
                    found = true;
                }
                if (!found) {
                    exit("iterative_solver", "cannot find the symmetry operation generating an equivalent k");
                }
            }
        }
    }

    // Wedge buffers of the deviation function (local rows + allreduced sum)
    double *dF_ir_loc = nullptr;
    double *dF_ir_glob = nullptr;
    allocate(dF_ir_loc, nk_irred * ns * 3);
    allocate(dF_ir_glob, nk_irred * ns * 3);

    // start iteration

    double norm;
    double local_difference;

    for (auto itemp = 0; itemp < ntemp; ++itemp) {

        double beta = 1.0 / (thermodynamics->T_to_Ryd * Temperature[itemp]);

        calc_boson(itemp, fb, dndt);

        if (mympi->my_rank == 0) {
            std::cout << " Temperature step ..." << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                      << Temperature[itemp] << " K" << "    -----------------------------\n";
            std::cout << "      Kappa [W/mK]        xx          xy          xz"
                      << "          yx          yy          yz" << "          zx          zy          zz    |df' - df|\n";
        }

        calc_Q_from_L(fb, Q);

        for (ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum; // k index in full grid
            average_over_degenerate_modes(ns, dos->dymat_dos->get_eigenvalues()[k1], 1, Q[ik]);
        }

        for (ik = 0; ik < nk_3ph; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (ix = 0; ix < 3; ++ix) {
                    dFold[ik][is][ix] = 0.0;
                }
            }
        }

        for (auto itr = 0; itr < max_cycle; ++itr) {

            local_difference = 0.0;

            if (mympi->my_rank == 0) {
                std::cout << "   -> iter " << std::setw(3) << itr << ": " << std::flush;
            }

            // zero the local wedge rows for the MPI_Allreduce
            for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii) dF_ir_loc[ii] = 0.0;

            // The scattering partners in L are stored for the representative
            // points; equivalent points carry no new information because
            // dF(Rk) = R dF(k), so only the wedge is updated here.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : local_difference)
#endif
            for (int ikl = 0; ikl < nklocal; ++ikl) {

                const auto tmpk = nk_l[ikl];
                const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
                const auto num_equivalent =
                    static_cast<double>(dos->kmesh_dos->kpoint_irred_all[tmpk].size());

                double **Wks_loc;
                allocate(Wks_loc, ns, 3);

                for (int s1 = 0; s1 < ns; ++s1) {

                    for (int ix = 0; ix < 3; ++ix) {
                        Wks_loc[s1][ix] = 0.0;
                    }

                    // emitt k1 -> k2 + k3
                    for (size_t j = 0; j < localnk_triplets_emitt[ikl].size(); ++j) {

                        const auto &pair = localnk_triplets_emitt[ikl][j];
                        const int kp_index = offset_emitt[ikl] + static_cast<int>(j);

                        for (size_t ig = 0; ig < pair.group.size(); ig++) {

                            const int k2 = pair.group[ig].ks[0];
                            const int k3 = pair.group[ig].ks[1];

                            for (int ib = 0; ib < ns2; ++ib) {
                                const int s2 = ib / ns;
                                const int s3 = ib % ns;

                                const double nn1 = fb[k1][s1];
                                const double nn2 = fb[k2][s2];
                                const double nn3 = fb[k3][s3];
                                for (int ix = 0; ix < 3; ++ix) {
                                    Wks_loc[s1][ix] -= 0.5 * (dFold[k2][s2][ix] + dFold[k3][s3][ix]) * nn1 *
                                                       (nn2 + 1.0) * (nn3 + 1.0) * L_emitt[kp_index][s1][ib];
                                }
                            }
                        }
                    }

                    // absorb k1 + k2 -> -k3
                    for (size_t j = 0; j < localnk_triplets_absorb[ikl].size(); ++j) {

                        const auto &pair = localnk_triplets_absorb[ikl][j];
                        const int kp_index = offset_absorb[ikl] + static_cast<int>(j);

                        for (size_t ig = 0; ig < pair.group.size(); ig++) {

                            const int k2 = pair.group[ig].ks[0];
                            const int k3 = pair.group[ig].ks[1];
                            const int k3_minus = dos->kmesh_dos->kindex_minus_xk[k3];

                            for (int ib = 0; ib < ns2; ++ib) {
                                const int s2 = ib / ns;
                                const int s3 = ib % ns;

                                const double nn1 = fb[k1][s1];
                                const double nn2 = fb[k2][s2];
                                const double nn3 = fb[k3][s3];
                                for (int ix = 0; ix < 3; ++ix) {
                                    Wks_loc[s1][ix] += (dFold[k2][s2][ix] - dFold[k3_minus][s3][ix]) * nn1 * nn2 *
                                                       (nn3 + 1.0) * L_absorb[kp_index][s1][ib];
                                }
                            }
                        }
                    }

                } // s1

                // Wks_loc is a contiguous [ns][3] block from allocate().
                average_over_degenerate_modes(ns, dos->dymat_dos->get_eigenvalues()[k1], 3, Wks_loc[0]);

                for (int s1 = 0; s1 < ns; ++s1) {

                    double Q_final = Q[ikl][s1];
                    if (isotope->include_isotope) {
                        Q_final += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * isotope_damping_loc[ikl][s1];
                    }

                    if (conductivity->len_boundary > eps) {
                        Q_final += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * boundary_damping_loc[ikl][s1];
                    }

                    if (conductivity->fph_rta > 0) {
                        Q_final += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * damping4[itemp][ikl][s1];
                    }

                    for (int ix = 0; ix < 3; ix++) {
                        double fnew;
                        if (Q_final < 1.0e-50 || dos->dymat_dos->get_eigenvalues()[k1][s1] < eps8) {
                            fnew = 0.0;
                        } else {
                            fnew = (-vel[k1][s1][ix] * dndt[ikl][s1] / beta - Wks_loc[s1][ix]) / Q_final;
                        }
                        if (itr > 0) {
                            fnew = fnew * mixing_factor + dFold[k1][s1][ix] * (1.0 - mixing_factor);
                            // weight by the star multiplicity so the residual
                            // norm matches the previous full-mesh definition
                            local_difference += num_equivalent * pow2(fnew - dFold[k1][s1][ix]);
                        }
                        dF_ir_loc[(tmpk * ns + s1) * 3 + ix] = fnew;
                    }
                }

                deallocate(Wks_loc);
            } // ikl

            // check convergence, if converged, stop, if not, update dF and print kappa
            norm = 0.0;
            MPI_Allreduce(&local_difference, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            convergence_history.push_back(norm);

            auto converged1 = false;
            auto converged2 = false;
            if (itr >= min_cycle) converged1 = check_convergence(convergence_history);

            if (converged1) {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa_new[ix][iy] = kappa_old[ix][iy];
                    }
                }
            } else {
                MPI_Allreduce(&dF_ir_loc[0], &dF_ir_glob[0], nk_irred * ns * 3, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);

                // Reconstruct the full-grid deviation function from the wedge.
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int p = 0; p < nk_3ph; ++p) {
                    const auto irr = dos->kmesh_dos->kmap_to_irreducible[p];
                    const auto &m = expand_mat[p];
                    for (int s = 0; s < ns; ++s) {
                        const double fx = dF_ir_glob[(irr * ns + s) * 3 + 0];
                        const double fy = dF_ir_glob[(irr * ns + s) * 3 + 1];
                        const double fz = dF_ir_glob[(irr * ns + s) * 3 + 2];
                        dFold[p][s][0] = m(0, 0) * fx + m(0, 1) * fy + m(0, 2) * fz;
                        dFold[p][s][1] = m(1, 0) * fx + m(1, 1) * fy + m(1, 2) * fz;
                        dFold[p][s][2] = m(2, 0) * fx + m(2, 1) * fy + m(2, 2) * fz;
                    }
                }

                calc_kappa(itemp, dFold, kappa_new);
                //print kappa
                if (mympi->my_rank == 0) {
                    for (ix = 0; ix < 3; ++ix) {
                        for (iy = 0; iy < 3; ++iy) {
                            std::cout << std::setw(12) << std::scientific << std::setprecision(2) << kappa_new[ix][iy];
                        }
                    }
                    norm = std::pow(norm, 0.5);
                    std::cout << std::setw(14) << std::scientific << std::setprecision(2) << norm << '\n' << std::flush;
                }
                if (itr >= min_cycle) converged2 = check_convergence(kappa_old, kappa_new);
            }

            if (converged1 || converged2) {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_old[ix][iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   -> Converged is achieved                 "
                              << "                                            "
                              << "                                    " << std::setw(14) << std::scientific
                              << std::setprecision(2) << norm << '\n' << std::flush;
                }
                break;

            } else {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa_old[ix][iy] = kappa_new[ix][iy];
                    }
                }
            }

            if (itr == (max_cycle - 1)) {
                // update kappa even if not converged
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_new[ix][iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   -> iter     Warning !! max cycle reached but kappa not converged \n" << std::flush;
                }
            }

        } // iter
        write_Q_dF(itemp, Q, dFold);

    } // itemp

    deallocate(Q);
    deallocate(dndt);

    deallocate(kappa_new);
    deallocate(kappa_old);
    deallocate(fb);
    deallocate(dF_ir_loc);
    deallocate(dF_ir_glob);
    if (isotope->include_isotope) {
        deallocate(isotope_damping_loc);
        //deallocate(isotope_damping);
    }
    if (mympi->my_rank == 0) {
        fs_result.close();
    }
}

void Iterativebte::calc_boson(int itemp, double **&b_out, double **&dndt_out)
{
    auto etemp = Temperature[itemp];
    double omega;
    for (auto ik = 0; ik < nk_3ph; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            omega = dos->dymat_dos->get_eigenvalues()[ik][is];
            b_out[ik][is] = thermodynamics->fB(omega, etemp);
        }
    }

    const double t_to_ryd = thermodynamics->T_to_Ryd;

    for (auto ik = 0; ik < nklocal; ++ik) {
        auto ikr = nk_l[ik];
        auto k1 = dos->kmesh_dos->kpoint_irred_all[ikr][0].knum;
        for (auto is = 0; is < ns; ++is) {
            omega = dos->dymat_dos->get_eigenvalues()[k1][is];
            auto x = omega / (t_to_ryd * etemp);
            dndt_out[ik][is] = pow2(1.0 / (2.0 * sinh(0.5 * x))) * x / etemp;
        }
    }
}


void Iterativebte::calc_kappa(int itemp, double ***&df, double **&kappa_out)
{
    auto etemp = Temperature[itemp];
    double omega;
    double beta = 1.0 / (thermodynamics->T_to_Ryd * etemp);
    double **tmpkappa;
    allocate(tmpkappa, 3, 3);

    for (auto ix = 0; ix < 3; ++ix) {
        for (auto iy = 0; iy < 3; ++iy) {
            tmpkappa[ix][iy] = 0.0;
        }
    }

    const double conversionfactor =
        Ryd / (time_ry * Bohr_in_Angstrom * 1.0e-10 * nk_3ph * system->get_primcell().volume);

    for (auto k1 = 0; k1 < nk_3ph; ++k1) {
        for (auto s1 = 0; s1 < ns; ++s1) {

            omega = dos->dymat_dos->get_eigenvalues()[k1][s1];
            double n1 = thermodynamics->fB(omega, etemp);
            double factor = beta * omega * n1 * (n1 + 1.0);

            for (auto ix = 0; ix < 3; ++ix) {
                for (auto iy = 0; iy < 3; ++iy) {
                    tmpkappa[ix][iy] += -factor * vel[k1][s1][ix] * df[k1][s1][iy];
                    // df in unit bohr/K
                }
            }
        }
    }

    for (auto ix = 0; ix < 3; ++ix) {
        for (auto iy = 0; iy < 3; ++iy) {
            kappa_out[ix][iy] = tmpkappa[ix][iy] * conversionfactor;
        }
    }

    deallocate(tmpkappa);
}


bool Iterativebte::check_convergence(double **&k_old, double **&k_new)
{
    // check diagonal components only, since they are the most important
    double max_diff = -100;
    double diff;
    for (auto ix = 0; ix < 3; ++ix) {
        diff = std::abs(k_new[ix][ix] - k_old[ix][ix]) / std::abs(k_old[ix][ix]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff < convergence_criteria;
}

bool Iterativebte::check_convergence(const std::vector<double> &history)
{
    auto size = history.size();
    double last = history[size - 1];
    double lastlast = history[size - 2];
    if (last > lastlast) {
        return true;
    } else
        return false;
}


void Iterativebte::write_kappa_iterative()
{
    // TODO: combine this function into write_phonons.cpp
    if (mympi->my_rank == 0) {

        auto file_kappa = input->job_title + ".kl_iter";

        std::ofstream ofs_kl;

        ofs_kl.open(file_kappa.c_str(), std::ios::out);
        if (!ofs_kl) exit("write_kappa_iterative", "Could not open file_kappa");

        ofs_kl << "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]" << '\n';
        ofs_kl << "# Iterative result." << '\n';

        if (isotope->include_isotope) ofs_kl << "# Isotope effects are included." << '\n';
        if (conductivity->fph_rta > 0) ofs_kl << "# 4ph is included non-iteratively." << '\n';
        if (conductivity->len_boundary > eps) {
            ofs_kl << "# Size of boundary " << std::scientific << std::setprecision(2)
                   << conductivity->len_boundary * 1e9 << " [nm]" << '\n';
        }

        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2) << Temperature[itemp];
            for (auto ix = 0; ix < 3; ++ix) {
                for (auto iy = 0; iy < 3; ++iy) {
                    ofs_kl << std::setw(15) << std::scientific << std::setprecision(4) << kappa[itemp][ix][iy];
                }
            }
            ofs_kl << '\n';
        }
        ofs_kl.close();
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------" << '\n' << '\n';
        std::cout << " Lattice thermal conductivity is stored in the file " << file_kappa << '\n';
    }
}

void Iterativebte::write_result()
{
    // write Q and W for all phonon, only phonon in irreducible BZ is written
    int i;
    double Ry_to_kayser = Hz_to_kayser / time_ry;

    if (mympi->my_rank == 0) {
        std::cout << " Prepare result file ..." << '\n';

        fs_result.open(conductivity->get_filename_results(3).c_str(), std::ios::out);

        if (!fs_result) {
            exit("setup_result_io", "Could not open file_result3");
        }

        fs_result << "## General information" << '\n';
        fs_result << "#SYSTEM" << '\n';
        fs_result << system->get_primcell().number_of_atoms << " " << system->get_primcell().number_of_elems
                  << '\n';
        fs_result << system->get_primcell().volume << '\n';
        fs_result << "#END SYSTEM" << '\n';

        fs_result << "#KPOINT" << '\n';
        fs_result << dos->kmesh_dos->nk_i[0] << " " << dos->kmesh_dos->nk_i[1] << " " << dos->kmesh_dos->nk_i[2]
                  << '\n';
        fs_result << dos->kmesh_dos->nk_irred << '\n';

        for (int i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
            fs_result << std::setw(6) << i + 1 << ":";
            for (int j = 0; j < 3; ++j) {
                fs_result << std::setw(15) << std::scientific << dos->kmesh_dos->kpoint_irred_all[i][0].kval[j];
            }
            fs_result << std::setw(12) << std::fixed << dos->kmesh_dos->weight_k[i] << '\n';
        }
        fs_result.unsetf(std::ios::fixed);

        fs_result << "#END KPOINT" << '\n';

        fs_result << "#CLASSICAL" << '\n';
        fs_result << thermodynamics->classical << '\n';
        fs_result << "#END CLASSICAL" << '\n';

        fs_result << "#FCSXML" << '\n';
        fs_result << fcs_phonon->file_fcs << '\n';
        fs_result << "#END  FCSXML" << '\n';

        fs_result << "#SMEARING" << '\n';
        fs_result << integration->ismear << '\n';
        fs_result << integration->epsilon * Ry_to_kayser << '\n';
        fs_result << "#END SMEARING" << '\n';

        fs_result << "#TEMPERATURE" << '\n';
        fs_result << system->Tmin << " " << system->Tmax << " " << system->dT << '\n';
        fs_result << "#END TEMPERATURE" << '\n';

        fs_result << "##END General information" << '\n';

        fs_result << "##Phonon Frequency" << '\n';
        fs_result << "#K-point (irreducible), Branch, Omega (cm^-1), Group velocity (m/s)" << '\n';

        double factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;
        for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
            const int ik = dos->kmesh_dos->kpoint_irred_all[i][0].knum;
            for (auto is = 0; is < dynamical->neval; ++is) {
                fs_result << std::setw(6) << i + 1 << std::setw(6) << is + 1;
                fs_result << std::setw(15) << writes->in_kayser(dos->dymat_dos->get_eigenvalues()[ik][is]);
                fs_result << std::setw(15) << vel[ik][is][0] * factor << std::setw(15) << vel[ik][is][1] * factor
                          << std::setw(15) << vel[ik][is][2] * factor << '\n';
            }
        }

        fs_result << "##END Phonon Frequency" << '\n' << '\n';
        fs_result << "##Q and W at each temperature" << '\n';
    }
}

void Iterativebte::write_Q_dF(int itemp, double **&q, double ***&df)
{
    auto etemp = Temperature[itemp];

    auto nk_ir = dos->kmesh_dos->nk_irred;
    double **Q_tmp;
    double **Q_all;
    allocate(Q_all, nk_ir, ns);
    allocate(Q_tmp, nk_ir, ns);
    for (auto ik = 0; ik < nk_ir; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            Q_all[ik][is] = 0.0;
            Q_tmp[ik][is] = 0.0;
        }
    }
    for (auto ik = 0; ik < nklocal; ++ik) {
        auto tmpk = nk_l[ik];
        for (auto is = 0; is < ns; ++is) {
            Q_tmp[tmpk][is] = q[ik][is];
        }
    }
    MPI_Allreduce(&Q_tmp[0][0], &Q_all[0][0], nk_ir * ns, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    deallocate(Q_tmp);

    // now we have Q
    if (mympi->my_rank == 0) {
        fs_result << std::setw(10) << etemp << '\n';

        for (auto ik = 0; ik < nk_ir; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                auto k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
                fs_result << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << '\n';
                fs_result << std::setw(15) << std::scientific << std::setprecision(5) << Q_all[ik][is] << std::setw(15)
                          << std::scientific << std::setprecision(5) << df[k1][is][0] << std::setw(15)
                          << std::scientific << std::setprecision(5) << df[k1][is][1] << std::setw(15)
                          << std::scientific << std::setprecision(5) << df[k1][is][2] << '\n';
            }
        }
        fs_result << '\n';
    }
    deallocate(Q_all);
}
