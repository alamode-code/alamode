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
    Temperature = nullptr;
    ntemp = 0;
    min_cycle = 5;
    max_cycle = 20;
    mixing_factor = 0.9;
    convergence_criteria = 0.02;
    kappa = nullptr;

    // private
    vel = nullptr;
    dFold = nullptr;
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

    // Build the collision operator: wedge distribution over the MPI ranks,
    // triplet lists of the local k points and the symmetry table.
    collision_op = std::make_unique<CollisionOperator>(phon);
    collision_op->setup();

    nk_l = collision_op->get_local_irred_ks();
    nklocal = collision_op->get_nklocal();

    // Per-temperature state in PREFIX.kappa.h5 (/iterativebte): open or
    // create the group and restore previously computed temperatures, which
    // the solver then skips (each temperature is independent, so skipping
    // reproduces an uninterrupted run exactly). RESTART = 0 discards them.
    t_computed.assign(ntemp, 0);
    t_converged.assign(ntemp, 0);
    if (conductivity->get_use_h5_io()) {
        const auto reset = !conductivity->get_restart_conductivity(3);
        ibte_io = conductivity->setup_ibte_io(dos->kmesh_dos->nk_i, dos->kmesh_dos->nk_irred, ns, reset);
        if (ibte_io) {
            const auto flags = ibte_io->load_ibte_computed();
            unsigned int n_done = 0, n_unconv = 0;
            for (size_t i = 0; i < flags.size() && i < t_computed.size(); ++i) {
                t_computed[i] = flags[i];
                if (flags[i]) ++n_done;
            }
            for (unsigned int i = 0; i < ntemp; ++i) {
                if (!t_computed[i]) continue;
                unsigned char conv = 0;
                ibte_io->load_ibte_kappa(i, &kappa[i][0][0], conv);
                t_converged[i] = conv;
                if (!conv) ++n_unconv;
            }
            if (n_done > 0) {
                std::cout << '\n' << " RESTART: " << n_done << " of " << ntemp
                          << " temperature points were restored from the kappa.h5 file.\n";
                if (n_unconv > 0) {
                    std::cout << "          " << n_unconv
                              << " of them are NOT converged; their iteration will be continued\n"
                              << "          from the stored deviation function.\n";
                } else {
                    std::cout << "          All of them are converged and will be skipped.\n";
                }
            }
        }
    }
    MPI_Bcast(t_computed.data(), static_cast<int>(ntemp), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(t_converged.data(), static_cast<int>(ntemp), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

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

    write_result();
}

void Iterativebte::do_iterativebte()
{
    auto all_done = ntemp > 0;
    for (unsigned int i = 0; i < ntemp; ++i) {
        if (!t_computed[i] || !t_converged[i]) all_done = false;
    }
    if (all_done) {
        // Every temperature was restored from the kappa.h5 file; the
        // expensive L matrices are not needed at all.
        if (mympi->my_rank == 0) {
            std::cout << '\n' << " All temperature points were restored from the kappa.h5 file;\n"
                      << " skipping the calculation of the transition probabilities.\n";
        }
        write_kappa_iterative();
        return;
    }

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " Calculate once for the transition probability L(absorb) and L(emitt)" << '\n';
        std::cout << " Size of L (MB) (approx.) = "
                  << memsize_in_MB(sizeof(double), collision_op->get_kplength_total(), ns, ns2) << " ... "
                  << std::flush;
    }

    collision_op->build_L();

    if (mympi->my_rank == 0) {
        std::cout << "     DONE !" << '\n';
    }

    iterative_solver();

    write_kappa_iterative();
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

    const auto nk_irred = dos->kmesh_dos->nk_irred;

    // Wedge buffers of the deviation function (local rows + allreduced sum),
    // plus a snapshot of the lowest-residual iterate for runs that end
    // without convergence.
    double *dF_ir_loc = nullptr;
    double *dF_ir_glob = nullptr;
    double *dF_ir_best = nullptr;
    allocate(dF_ir_loc, nk_irred * ns * 3);
    allocate(dF_ir_glob, nk_irred * ns * 3);
    allocate(dF_ir_best, nk_irred * ns * 3);

    // start iteration

    double local_reduce[2], global_reduce[2];

    // Per-temperature record for the final summary.
    std::vector<int> iterations_used(ntemp, -1);
    std::vector<double> final_residual(ntemp, 0.0);

    for (auto itemp = 0; itemp < ntemp; ++itemp) {

        if (t_computed[itemp] && t_converged[itemp]) {
            if (mympi->my_rank == 0) {
                std::cout << " Temperature step ..." << std::setw(10) << std::right << std::fixed
                          << std::setprecision(2) << Temperature[itemp]
                          << " K    restored from the kappa.h5 file; skipped.\n";
            }
            continue;
        }

        // Warm start: a computed-but-unconverged temperature continues its
        // iteration from the stored deviation function instead of restarting
        // from zero.
        const auto warm_start = t_computed[itemp] && !t_converged[itemp] &&
                                conductivity->get_use_h5_io();

        auto converged_this_temp = false;

        double beta = 1.0 / (thermodynamics->T_to_Ryd * Temperature[itemp]);

        calc_boson(itemp, fb, dndt);

        if (mympi->my_rank == 0) {
            std::cout << " Temperature step ..." << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                      << Temperature[itemp] << " K" << "    -----------------------------\n";
            if (warm_start) {
                std::cout << "      (unconverged result found; continuing from the stored deviation function)\n";
            }
            std::cout << "      Kappa [W/mK]        xx          xy          xz"
                      << "          yx          yy          yz"
                      << "          zx          zy          zz  |df'-df|/|df|\n";
        }

        collision_op->calc_Q_from_L(fb, Q);

        for (ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum; // k index in full grid
            average_over_degenerate_modes(ns, dos->dymat_dos->get_eigenvalues()[k1], 1, Q[ik]);
        }

        if (warm_start) {
            if (ibte_io) {
                ibte_io->load_ibte_dF(itemp, dF_ir_glob);
            }
            MPI_Bcast(dF_ir_glob, static_cast<int>(nk_irred * ns * 3), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            collision_op->reconstruct_full_from_wedge(dF_ir_glob, dFold);
        } else {
            for (ik = 0; ik < nk_3ph; ++ik) {
                for (is = 0; is < ns; ++is) {
                    for (ix = 0; ix < 3; ++ix) {
                        dFold[ik][is][ix] = 0.0;
                    }
                }
            }
        }

        // Relative-residual history of this temperature and the best
        // (lowest-residual) iterate seen so far.
        std::vector<double> res_history;
        double kappa_best[9];
        double res_best = -1.0;
        int itr_best = -1;

        for (auto itr = 0; itr < max_cycle; ++itr) {

            double local_difference = 0.0;
            double local_norm2 = 0.0;

            if (mympi->my_rank == 0) {
                std::cout << "   -> iter " << std::setw(3) << itr << ": " << std::flush;
            }

            // zero the local wedge rows for the MPI_Allreduce
            for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii) dF_ir_loc[ii] = 0.0;

            // The in-scattering action W is evaluated on the wedge only:
            // equivalent points carry no new information because
            // dF(Rk) = R dF(k).
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : local_difference, local_norm2)
#endif
            for (int ikl = 0; ikl < nklocal; ++ikl) {

                const auto tmpk = nk_l[ikl];
                const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
                const auto num_equivalent =
                    static_cast<double>(dos->kmesh_dos->kpoint_irred_all[tmpk].size());

                double **Wks_loc;
                allocate(Wks_loc, ns, 3);

                collision_op->calc_W_at(ikl, fb, dFold, Wks_loc);

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
                        local_norm2 += num_equivalent * pow2(fnew);
                        dF_ir_loc[(tmpk * ns + s1) * 3 + ix] = fnew;
                    }
                }

                deallocate(Wks_loc);
            } // ikl

            local_reduce[0] = local_difference;
            local_reduce[1] = local_norm2;
            MPI_Allreduce(local_reduce, global_reduce, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Relative change of the deviation function; the iterate is
            // always evaluated (the old code skipped the update when the
            // residual grew and silently reported that as convergence).
            const auto rel = (itr > 0 && global_reduce[1] > 0.0)
                                 ? std::sqrt(global_reduce[0] / global_reduce[1])
                                 : 0.0;
            if (itr > 0) res_history.push_back(rel);

            MPI_Allreduce(&dF_ir_loc[0], &dF_ir_glob[0], nk_irred * ns * 3, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);

            // Reconstruct the full-grid deviation function from the wedge.
            collision_op->reconstruct_full_from_wedge(dF_ir_glob, dFold);

            calc_kappa(itemp, dFold, kappa_new);
            //print kappa
            if (mympi->my_rank == 0) {
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        std::cout << std::setw(12) << std::scientific << std::setprecision(2) << kappa_new[ix][iy];
                    }
                }
                std::cout << std::setw(14) << std::scientific << std::setprecision(2) << rel << '\n' << std::flush;
            }

            // Keep the lowest-residual iterate in case the run ends without
            // convergence (for a monotonically improving iteration this is
            // simply the last one).
            if (itr > 0 && (res_best < 0.0 || rel < res_best)) {
                res_best = rel;
                itr_best = itr;
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa_best[3 * ix + iy] = kappa_new[ix][iy];
                    }
                }
                for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii) dF_ir_best[ii] = dF_ir_glob[ii];
            }

            auto converged = false;
            if (itr >= min_cycle) converged = check_convergence(kappa_old, kappa_new);

            if (converged) {
                converged_this_temp = true;
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_old[ix][iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   -> Converged is achieved                 "
                              << "                                            "
                              << "                                    " << std::setw(14) << std::scientific
                              << std::setprecision(2) << rel << '\n' << std::flush;
                }
                iterations_used[itemp] = itr;
                final_residual[itemp] = rel;
                break;
            }

            for (ix = 0; ix < 3; ++ix) {
                for (iy = 0; iy < 3; ++iy) {
                    kappa_old[ix][iy] = kappa_new[ix][iy];
                }
            }

            // Divergence: the relative residual grew in two consecutive
            // iterations. Stop, keep the best iterate, and mark the
            // temperature as NOT converged instead of pretending otherwise.
            const auto nres = res_history.size();
            const auto diverged = itr >= min_cycle && nres >= 3 &&
                                  res_history[nres - 1] > res_history[nres - 2] &&
                                  res_history[nres - 2] > res_history[nres - 3];

            if (diverged || itr == (max_cycle - 1)) {
                for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii) dF_ir_glob[ii] = dF_ir_best[ii];
                collision_op->reconstruct_full_from_wedge(dF_ir_glob, dFold);
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_best[3 * ix + iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    if (diverged) {
                        std::cout << "   -> WARNING: the iteration is diverging. Keeping the lowest-residual\n"
                                  << "               iterate (iter " << itr_best << ", |df'-df|/|df| = "
                                  << std::scientific << std::setprecision(2) << res_best << ") and marking this\n"
                                  << "               temperature as NOT converged."
                                  << " Consider reducing IBTE_MIXING.\n" << std::flush;
                    } else {
                        std::cout << "   -> WARNING: max cycle reached but NOT converged. Keeping the\n"
                                  << "               lowest-residual iterate (iter " << itr_best
                                  << ", |df'-df|/|df| = " << std::scientific << std::setprecision(2) << res_best
                                  << ").\n" << std::flush;
                    }
                }
                iterations_used[itemp] = itr;
                final_residual[itemp] = res_best;
                break;
            }

        } // iter
        t_converged[itemp] = converged_this_temp ? 1 : 0;
        write_Q_dF(itemp, Q, dFold, converged_this_temp);

    } // itemp

    // Per-temperature summary (computed this run only).
    if (mympi->my_rank == 0) {
        std::cout << '\n' << " Iterative BTE summary\n";
        std::cout << "     T [K]   iterations   |df'-df|/|df|   converged\n";
        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            if (iterations_used[itemp] < 0) continue; // restored and skipped
            std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(2) << Temperature[itemp]
                      << std::setw(11) << iterations_used[itemp] << "   " << std::setw(13) << std::scientific
                      << std::setprecision(2) << final_residual[itemp] << "   " << std::setw(9)
                      << (t_converged[itemp] ? "yes" : "NO") << '\n';
        }
        std::cout << std::flush;
    }

    deallocate(Q);
    deallocate(dndt);

    deallocate(kappa_new);
    deallocate(kappa_old);
    deallocate(fb);
    deallocate(dF_ir_loc);
    deallocate(dF_ir_glob);
    deallocate(dF_ir_best);
    if (isotope->include_isotope) {
        deallocate(isotope_damping_loc);
        //deallocate(isotope_damping);
    }
    if (mympi->my_rank == 0 && !conductivity->get_use_h5_io()) {
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



void Iterativebte::write_kappa_iterative()
{
    writes->writeKappaIterative(ntemp, Temperature, kappa, t_converged);
}

void Iterativebte::write_result()
{
    // Legacy text Q/dF file (FILE_FORMAT = text only): the h5 path stores
    // the same data per temperature in the /iterativebte group of
    // PREFIX.kappa.h5 instead of reusing the legacy .result filename with
    // an incompatible format.
    if (conductivity->get_use_h5_io()) return;

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

void Iterativebte::write_Q_dF(int itemp, double **&q, double ***&df, const bool converged)
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
        if (ibte_io) {
            // Durable per-temperature commit into /iterativebte: Q_all is a
            // contiguous [nk_ir][ns] block; dF is flattened at the wedge
            // representatives.
            std::vector<double> df_flat(static_cast<size_t>(nk_ir) * ns * 3);
            for (auto ik = 0; ik < nk_ir; ++ik) {
                const auto k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
                for (auto is = 0; is < ns; ++is) {
                    for (auto j = 0; j < 3; ++j) {
                        df_flat[(static_cast<size_t>(ik) * ns + is) * 3 + j] = df[k1][is][j];
                    }
                }
            }
            ibte_io->store_ibte_temperature(itemp, &Q_all[0][0], df_flat.data(), &kappa[itemp][0][0],
                                            converged ? 1 : 0);
        } else if (!conductivity->get_use_h5_io()) {
            fs_result << std::setw(10) << etemp << '\n';

            for (auto ik = 0; ik < nk_ir; ++ik) {
                for (auto is = 0; is < ns; ++is) {
                    auto k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
                    fs_result << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << '\n';
                    fs_result << std::setw(15) << std::scientific << std::setprecision(5) << Q_all[ik][is]
                              << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][0]
                              << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][1]
                              << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][2] << '\n';
                }
            }
            fs_result << '\n';
        }
    }
    deallocate(Q_all);
}
