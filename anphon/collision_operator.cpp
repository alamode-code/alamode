/*
 collision_operator.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "collision_operator.h"
#include <complex>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "isotope.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "symmetry_core.h"
#include "system.h"

using namespace PHON_NS;

CollisionOperator::CollisionOperator(PHON *phon) : Pointers(phon)
{
    kplength_emitt = 0;
    kplength_absorb = 0;
    nk_3ph = 0;
    nklocal = 0;
    ns = 0;
    ns2 = 0;
    use_triplet_symmetry = true;
    sym_permutation = false;
    with_isotope = false;
    L_absorb = nullptr;
    L_emitt = nullptr;
}

CollisionOperator::~CollisionOperator()
{
    if (L_absorb) {
        deallocate(L_absorb);
    }
    if (L_emitt) {
        deallocate(L_emitt);
    }
}

void CollisionOperator::setup()
{
    nk_3ph = dos->kmesh_dos->nk;
    ns = dynamical->neval;
    ns2 = ns * ns;

    // distribute the irreducible q points among the processors
    const auto nk_ir = dos->kmesh_dos->nk_irred;

    nk_l.clear();
    for (auto i = 0; i < nk_ir; ++i) {
        if (i % mympi->nprocs == mympi->my_rank) nk_l.push_back(i);
    }

    nklocal = static_cast<int>(nk_l.size());

    get_triplets();

    build_expansion_table();
}

void CollisionOperator::get_triplets()
{
    localnk_triplets_emitt.clear();  // pairs k3 = k1 - k2 ( -k1 + k2 + k3 = G )
    localnk_triplets_absorb.clear(); // pairs k3 = k1 + k2 (  k1 + k2 + k3 = G )

    int counter = 0;
    int counter2 = 0;

    for (int i = 0; i < nklocal; ++i) {

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

    // Flattened triplet index shared by build_L, calc_Q_from_L and
    // calc_W_at: row of triplet j of local k point ik in L is
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

void CollisionOperator::build_expansion_table()
{
    // ------------------------------------------------------------------
    // Symmetry expansion table for the irreducible-wedge iteration.
    // Every full-grid point p equals (time reversal x) R applied to its
    // irreducible representative, and the deviation function transforms
    // as a Cartesian vector [f_{Rk} = R f_k, f_{-k} = -f_k], so
    //   dF(p) = expand_mat[p] . dF(rep(p)),
    // with the time-reversal sign folded into the matrix. knum_sym
    // applies (S^{-1})^T to the fractional k, and k_cart = M k_frac with
    // M columns the reciprocal lattice vectors, hence
    // R_cart = M (S^{-1})^T M^{-1}.
    // ------------------------------------------------------------------
    const int nsym = symmetry->SymmList.size();
    const auto nk_irred = dos->kmesh_dos->nk_irred;

    expand_mat.resize(nk_3ph);

    Eigen::Matrix3d mat_k2cart;
    const auto &rlavec = system->get_primcell().reciprocal_lattice_vector;
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            mat_k2cart(j, i) = rlavec(i, j); // columns = reciprocal vectors
        }
    }
    const Eigen::Matrix3d mat_cart2k = mat_k2cart.inverse();

    littlegroup_proj.resize(nk_irred);

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
                exit("build_expansion_table", "cannot find the symmetry operation generating an equivalent k");
            }
        }

        // Little group of the representative: all operations (including
        // time-reversal-composed ones) that map it onto itself; averaging
        // their Cartesian representations projects onto invariant vectors.
        Eigen::Matrix3d proj = Eigen::Matrix3d::Zero();
        auto nops = 0;
        for (auto isym = 0; isym < nsym; ++isym) {
            const auto krot = dos->kmesh_dos->knum_sym(kref, symmetry->SymmList[isym].rotation);
            const auto direct = (kref == static_cast<int>(krot));
            const auto time_reversed =
                symmetry->time_reversal_sym && kref == static_cast<int>(dos->kmesh_dos->kindex_minus_xk[krot]);
            if (!direct && !time_reversed) continue;
            const Eigen::Matrix3d srot_inv_t =
                symmetry->SymmList[isym].rotation.cast<double>().inverse().transpose();
            const Eigen::Matrix3d rot_cart = mat_k2cart * srot_inv_t * mat_cart2k;
            if (direct) {
                proj += rot_cart;
                ++nops;
            }
            if (time_reversed) {
                proj -= rot_cart;
                ++nops;
            }
        }
        littlegroup_proj[tmpk] = proj / static_cast<double>(nops);
    }
}

void CollisionOperator::project_onto_littlegroup(double *dF_ir) const
{
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    for (unsigned int ik = 0; ik < nk_irred; ++ik) {
        const auto &m = littlegroup_proj[ik];
        for (int s = 0; s < ns; ++s) {
            double *v = dF_ir + (static_cast<size_t>(ik) * ns + s) * 3;
            const double vx = v[0], vy = v[1], vz = v[2];
            v[0] = m(0, 0) * vx + m(0, 1) * vy + m(0, 2) * vz;
            v[1] = m(1, 0) * vx + m(1, 1) * vy + m(1, 2) * vz;
            v[2] = m(2, 0) * vx + m(2, 1) * vy + m(2, 2) * vz;
        }
    }
}

void CollisionOperator::build_L()
{
    if (integration->ismear >= 0) {
        setup_L_smear();
    } else if (integration->ismear == -1) {
        setup_L_tetra();
    }
    if (with_isotope) {
        build_L_isotope();
    }
}

void CollisionOperator::build_L_isotope()
{
    // Elastic isotope-disorder kernel between mode (k1,s1) at a local wedge
    // point and every mode (k2,s2) of the full mesh (Tamura formula),
    // discretized exactly like Isotope::calc_isotope_selfenergy[_tetra] so
    // that the row sums reproduce the SERTA isotope linewidths.
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto eval_in = dos->dymat_dos->get_eigenvalues();
    const auto evec_in = dos->dymat_dos->get_eigenvectors();
    const auto &g2 = isotope->isotope_factor;
    const auto &kind = system->get_primcell().kind;

    L_iso.assign(static_cast<size_t>(nklocal) * ns, {});

    // Degeneracy-averaged frequencies (tetrahedron path only), built once.
    const auto tol_degenerate = 1.0e-7 * time_ry / Hz_to_kayser;
    double **eval_tetra = nullptr;
    if (integration->ismear == -1) {
        allocate(eval_tetra, ns, nk_3ph);
        for (int ik = 0; ik < nk_3ph; ++ik) {
            auto begin = 0;
            auto omega_ref = eval_in[ik][0];
            auto omega_sum = eval_in[ik][0];
            for (int is = 1; is < ns; ++is) {
                const auto omega_now = eval_in[ik][is];
                if (std::abs(omega_now - omega_ref) < tol_degenerate) {
                    omega_sum += omega_now;
                } else {
                    const auto omega_avg = omega_sum / static_cast<double>(is - begin);
                    for (auto js = begin; js < is; ++js) eval_tetra[js][ik] = omega_avg;
                    begin = is;
                    omega_ref = omega_now;
                    omega_sum = omega_now;
                }
            }
            const auto omega_avg = omega_sum / static_cast<double>(ns - begin);
            for (auto js = begin; js < ns; ++js) eval_tetra[js][ik] = omega_avg;
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        double *energy_tmp = nullptr;
        double **weight_tetra = nullptr;
        unsigned int *kmap_identity = nullptr;
        if (integration->ismear == -1) {
            allocate(energy_tmp, nk_3ph);
            allocate(weight_tetra, ns, nk_3ph);
            allocate(kmap_identity, nk_3ph);
            for (int ik = 0; ik < nk_3ph; ++ik) kmap_identity[ik] = ik;
        }

#ifdef _OPENMP
#pragma omp for
#endif
        for (int irow = 0; irow < nklocal * ns; ++irow) {
            const int ikl = irow / ns;
            const int s1 = irow % ns;
            const int k1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ikl]][0].knum;
            const auto omega1 = eval_in[k1][s1];
            if (omega1 < eps8) continue;

            auto &row = L_iso[irow];

            const auto prod_overlap = [&](const int k2, const int s2) {
                auto prod = 0.0;
                for (auto iat = 0; iat < natmin; ++iat) {
                    auto dprod = std::complex<double>(0.0, 0.0);
                    for (auto icrd = 0; icrd < 3; ++icrd) {
                        dprod += std::conj(evec_in[k2][s2][3 * iat + icrd]) * evec_in[k1][s1][3 * iat + icrd];
                    }
                    prod += g2[kind[iat]] * std::norm(dprod);
                }
                return prod;
            };

            if (integration->ismear >= 0) {
                const auto epsilon = integration->epsilon;
                const auto prefactor = pi * omega1 * 0.25 / static_cast<double>(nk_3ph);
                for (int k2 = 0; k2 < nk_3ph; ++k2) {
                    for (int s2 = 0; s2 < ns; ++s2) {
                        const auto omega2 = eval_in[k2][s2];
                        double delta_loc = 0.0;
                        double peak;
                        if (integration->ismear == 0) {
                            delta_loc = delta_lorentz(omega1 - omega2, epsilon);
                            peak = delta_lorentz(0.0, epsilon);
                        } else if (integration->ismear == 1) {
                            delta_loc = delta_gauss(omega1 - omega2, epsilon);
                            peak = delta_gauss(0.0, epsilon);
                        } else {
                            double eps_loc;
                            integration->adaptive_sigma->get_sigma(k2, s2, eps_loc);
                            delta_loc = delta_gauss(omega1 - omega2, eps_loc);
                            peak = delta_gauss(0.0, eps_loc);
                        }
                        // Sparsity cutoff far off shell (<= 1e-12 of the
                        // on-shell weight); negligible for the row sums.
                        if (delta_loc <= 1.0e-12 * peak) continue;
                        const auto val = prefactor * omega2 * delta_loc * prod_overlap(k2, s2);
                        if (val == 0.0) continue;
                        row.push_back({k2, s2, val});
                    }
                }
            } else {
                // Tetrahedron: weights per s2 over the k2 grid, then
                // block-averaged within degenerate manifolds (mirrors
                // Isotope::calc_isotope_selfenergy_tetra).
                for (int s2 = 0; s2 < ns; ++s2) {
                    for (int ik = 0; ik < nk_3ph; ++ik) energy_tmp[ik] = eval_tetra[s2][ik];
                    integration->calc_weight_tetrahedron(nk_3ph,
                                                         kmap_identity,
                                                         energy_tmp,
                                                         omega1,
                                                         dos->tetra_nodes_dos->get_ntetra(),
                                                         dos->tetra_nodes_dos->get_tetras(),
                                                         weight_tetra[s2]);
                }
                for (int k2 = 0; k2 < nk_3ph; ++k2) {
                    auto begin = 0;
                    auto omega_ref = eval_tetra[0][k2];
                    for (int s2 = 1; s2 <= ns; ++s2) {
                        if (s2 < ns && std::abs(eval_tetra[s2][k2] - omega_ref) < tol_degenerate) continue;
                        if (s2 - begin > 1) {
                            auto wsum = 0.0;
                            for (auto js = begin; js < s2; ++js) wsum += weight_tetra[js][k2];
                            wsum /= static_cast<double>(s2 - begin);
                            for (auto js = begin; js < s2; ++js) weight_tetra[js][k2] = wsum;
                        }
                        if (s2 < ns) {
                            begin = s2;
                            omega_ref = eval_tetra[s2][k2];
                        }
                    }
                }
                const auto prefactor = pi * omega1 * 0.25;
                for (int k2 = 0; k2 < nk_3ph; ++k2) {
                    for (int s2 = 0; s2 < ns; ++s2) {
                        if (weight_tetra[s2][k2] == 0.0) continue;
                        const auto val =
                            prefactor * weight_tetra[s2][k2] * eval_tetra[s2][k2] * prod_overlap(k2, s2);
                        if (val == 0.0) continue;
                        row.push_back({k2, s2, val});
                    }
                }
            }
        }

        if (integration->ismear == -1) {
            deallocate(energy_tmp);
            deallocate(weight_tetra);
            deallocate(kmap_identity);
        }
    }

    if (eval_tetra) deallocate(eval_tetra);
}

void CollisionOperator::add_isotope_diagonal(const double *const *sqrt_occ, double **q_inout) const
{
    for (int ikl = 0; ikl < nklocal; ++ikl) {
        const int k1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ikl]][0].knum;
        for (int s = 0; s < ns; ++s) {
            auto sum = 0.0;
            for (const auto &entry: L_iso[static_cast<size_t>(ikl) * ns + s]) {
                sum += sqrt_occ[entry.knum][entry.snum] * entry.val;
            }
            q_inout[ikl][s] += 2.0 * sqrt_occ[k1][s] * sum;
        }
    }
}

void CollisionOperator::setup_L_smear()
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

void CollisionOperator::setup_L_tetra()
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

void CollisionOperator::calc_Q_from_L(const double *const *sqrt_occ, double **q1) const
{
    int s1, s2, s3;
    double g1, g2, g3;

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
                g1 = sqrt_occ[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    g2 = sqrt_occ[k2][s2];
                    g3 = sqrt_occ[k3][s3];
                    Qemit[ik][s1] += 0.5 * g1 * g2 * g3 * L_emitt[offset_emitt[ik] + j][s1][ib] * multi;
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
                g1 = sqrt_occ[k1][s1];

                for (int ib = 0; ib < ns2; ++ib) {
                    s2 = ib / ns;
                    s3 = ib % ns;
                    g2 = sqrt_occ[k2][s2];
                    g3 = sqrt_occ[k3][s3];
                    Qabsorb[ik][s1] += g1 * g2 * g3 * L_absorb[offset_absorb[ik] + j][s1][ib] * multi;
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

void CollisionOperator::calc_W_at(const int ikl, const double *const *sqrt_occ, const double *const *const *dF,
                                  double **Wks_out) const
{
    // The scattering partners in L are stored for the representative
    // points; equivalent points carry no new information because
    // dF(Rk) = R dF(k), so W is only needed at the wedge points.
    const int k1 = dos->kmesh_dos->kpoint_irred_all[nk_l[ikl]][0].knum;

    for (int s1 = 0; s1 < ns; ++s1) {

        for (int ix = 0; ix < 3; ++ix) {
            Wks_out[s1][ix] = 0.0;
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

                    const auto ggg = sqrt_occ[k1][s1] * sqrt_occ[k2][s2] * sqrt_occ[k3][s3];
                    for (int ix = 0; ix < 3; ++ix) {
                        Wks_out[s1][ix] -= 0.5 * (dF[k2][s2][ix] + dF[k3][s3][ix]) * ggg *
                                           L_emitt[kp_index][s1][ib];
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

                    const auto ggg = sqrt_occ[k1][s1] * sqrt_occ[k2][s2] * sqrt_occ[k3][s3];
                    for (int ix = 0; ix < 3; ++ix) {
                        Wks_out[s1][ix] += (dF[k2][s2][ix] - dF[k3_minus][s3][ix]) * ggg *
                                           L_absorb[kp_index][s1][ib];
                    }
                }
            }
        }

    } // s1

    if (with_isotope) {
        // Elastic in-scattering: -2 g1 g2 w(row,partner) dF(partner);
        // together with the row-sum diagonal this annihilates constant
        // fields exactly.
        for (int s1 = 0; s1 < ns; ++s1) {
            const auto g1 = 2.0 * sqrt_occ[k1][s1];
            for (const auto &entry: L_iso[static_cast<size_t>(ikl) * ns + s1]) {
                for (int ix = 0; ix < 3; ++ix) {
                    Wks_out[s1][ix] -= g1 * sqrt_occ[entry.knum][entry.snum] * entry.val *
                                       dF[entry.knum][entry.snum][ix];
                }
            }
        }
    }
}

void CollisionOperator::assemble_dense_rows(const double *const *sqrt_occ, const double *const *Qfin_loc,
                                            double *slab) const
{
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const size_t ncol = static_cast<size_t>(nk_irred) * ns * 3;

    std::vector<double> sqrt_mult(nk_irred);
    for (unsigned int ik = 0; ik < nk_irred; ++ik) {
        sqrt_mult[ik] = std::sqrt(static_cast<double>(dos->kmesh_dos->kpoint_irred_all[ik].size()));
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int ikl = 0; ikl < nklocal; ++ikl) {

        const auto tmpk = nk_l[ikl];
        const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
        const auto sqrt_mrow = sqrt_mult[tmpk];

        for (int s1 = 0; s1 < ns; ++s1) {

            const size_t base_row = (static_cast<size_t>(ikl) * ns + s1) * 3;

            // Fold a coupling coefficient to the full-grid mode (kt, st)
            // onto the wedge: dF(kt) = expand_mat[kt] . dF_ir(rep(kt)).
            const auto scatter = [&](const double coeff, const int kt, const int st) {
                const auto ikc = dos->kmesh_dos->kmap_to_irreducible[kt];
                const auto &B = expand_mat[kt];
                const auto factor = coeff * sqrt_mrow / sqrt_mult[ikc];
                const size_t colbase = (static_cast<size_t>(ikc) * ns + st) * 3;
                for (auto x = 0; x < 3; ++x) {
                    for (auto y = 0; y < 3; ++y) {
                        slab[(base_row + x) * ncol + colbase + y] += factor * B(x, y);
                    }
                }
            };

            // Diagonal (row and column blocks coincide, so the metric
            // factors cancel).
            for (auto x = 0; x < 3; ++x) {
                slab[(base_row + x) * ncol + (static_cast<size_t>(tmpk) * ns + s1) * 3 + x] += Qfin_loc[ikl][s1];
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
                        const auto ggg = sqrt_occ[k1][s1] * sqrt_occ[k2][s2] * sqrt_occ[k3][s3];
                        const auto coeff = -0.5 * ggg * L_emitt[kp_index][s1][ib];
                        if (coeff == 0.0) continue;
                        scatter(coeff, k2, s2);
                        scatter(coeff, k3, s3);
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
                        const auto ggg = sqrt_occ[k1][s1] * sqrt_occ[k2][s2] * sqrt_occ[k3][s3];
                        const auto coeff = ggg * L_absorb[kp_index][s1][ib];
                        if (coeff == 0.0) continue;
                        scatter(coeff, k2, s2);
                        scatter(-coeff, k3_minus, s3);
                    }
                }
            }

            if (with_isotope) {
                const auto g1 = 2.0 * sqrt_occ[k1][s1];
                for (const auto &entry: L_iso[static_cast<size_t>(ikl) * ns + s1]) {
                    scatter(-g1 * sqrt_occ[entry.knum][entry.snum] * entry.val, entry.knum, entry.snum);
                }
            }
        }
    }
}

void CollisionOperator::reconstruct_full_from_wedge(const double *dF_ir, double ***dF_full) const
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int p = 0; p < nk_3ph; ++p) {
        const auto irr = dos->kmesh_dos->kmap_to_irreducible[p];
        const auto &m = expand_mat[p];
        for (int s = 0; s < ns; ++s) {
            const double fx = dF_ir[(irr * ns + s) * 3 + 0];
            const double fy = dF_ir[(irr * ns + s) * 3 + 1];
            const double fz = dF_ir[(irr * ns + s) * 3 + 2];
            dF_full[p][s][0] = m(0, 0) * fx + m(0, 1) * fy + m(0, 2) * fz;
            dF_full[p][s][1] = m(1, 0) * fx + m(1, 1) * fy + m(1, 2) * fz;
            dF_full[p][s][2] = m(2, 0) * fx + m(2, 1) * fy + m(2, 2) * fz;
        }
    }
}
