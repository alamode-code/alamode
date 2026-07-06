#include "iterativebte.h"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "constants.h"
#include "dense_symmetric_eigen.h"
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
    use_variational = false;
    use_direct = false;
    isotope_inscattering = true;
    cg_symmetry_checked = false;
    dbte_assembly_checked = false;
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
    MPI_Bcast(&use_variational, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_direct, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&isotope_inscattering, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    // The 4ph channel follows conductivity->fph_rta, decided by the
    // INCLUDE_4PH tag at parse time (same as SOLVER = RTA).

    // Temperature grid in K, owned by Conductivity (non-owning alias here).
    conductivity->init_temperature_grid();
    ntemp = conductivity->ntemp;
    Temperature = conductivity->temperature;

    // Full-grid velocities in atomic units on every rank (calc_kappa and
    // the boundary rate convert units at the point of use).
    phonon_velocity->gather_group_velocities_mesh(*dos->kmesh_dos,
                                                  system->get_primcell().lattice_vector,
                                                  vel,
                                                  1.0,
                                                  true);

    allocate(kappa, ntemp, 3, 3);

    // Build the collision operator: wedge distribution over the MPI ranks,
    // triplet lists of the local k points and the symmetry table.
    collision_op = std::make_unique<CollisionOperator>(*dos->kmesh_dos,
                                                       *dos->tetra_nodes_dos,
                                                       *dos->dymat_dos,
                                                       *system,
                                                       *symmetry,
                                                       *integration,
                                                       *anharmonic_core,
                                                       dynamical->neval,
                                                       mympi->my_rank,
                                                       mympi->nprocs);
    collision_op->set_isotope_channel(isotope->include_isotope && isotope_inscattering, isotope->isotope_factor.data());
    collision_op->setup();

    if (collision_op->has_isotope_channel() && mympi->my_rank == 0) {
        std::cout << '\n'
                  << " ISOTOPE_INSCATTERING = 1: the elastic isotope-disorder channel enters\n"
                  << " the collision operator with its in-scattering term (its diagonal is the\n"
                  << " operator row sum; set ISOTOPE_INSCATTERING = 0 for the RTA-level diagonal).\n";
    }

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
                std::cout << '\n'
                          << " RESTART: " << n_done << " of " << ntemp
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
        if (use_direct) {
            std::cout << " Direct solution (dense eigendecomposition of the collision kernel)" << '\n';
            std::cout << " ===================================================================" << '\n';
            std::cout << " Diagnostic solver: full spectrum, PSD check, near-null analysis." << '\n';
        } else if (use_variational) {
            std::cout << " Variational solution (preconditioned conjugate gradients)" << '\n';
            std::cout << " ==========================================================" << '\n';
            std::cout << " MAX_CYCLE = " << max_cycle << '\n';
            std::cout << " ITER_THRESHOLD (relative residual; the kappa error is quadratic in it) = " << std::setw(10)
                      << std::right << std::setprecision(4) << convergence_criteria << '\n';
        } else {
            std::cout << " Iterative solution" << '\n';
            std::cout << " ==================" << '\n';
            std::cout << " MIN_CYCLE = " << min_cycle << ", MAX_CYCLE = " << max_cycle << '\n';
            std::cout << " ITER_THRESHOLD = " << std::setw(10) << std::right << std::setprecision(4)
                      << convergence_criteria << '\n';
        }
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
            std::cout << '\n'
                      << " All temperature points were restored from the kappa.h5 file;\n"
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
    double **Qfin;
    double **kappa_new;
    double **kappa_old;

    allocate(kappa_new, 3, 3);
    allocate(kappa_old, 3, 3);
    allocate(Q, nklocal, ns);
    allocate(Qfin, nklocal, ns);

    allocate(dFold, nk_3ph, ns, 3);

    double **fb;
    double **gb; // sqrt(n(n+1)): the detailed-balance-symmetric kernel table
    double **dndt;
    allocate(dndt, nklocal, ns);
    allocate(fb, nk_3ph, ns);
    allocate(gb, nk_3ph, ns);

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
                boundary_damping_loc[ik][is] = vel_norm * Bohr_in_Angstrom * 1.0e-10 / conductivity->len_boundary;
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
        const auto warm_start = t_computed[itemp] && !t_converged[itemp] && conductivity->get_use_h5_io();

        auto converged_this_temp = false;

        double beta = 1.0 / (thermodynamics->T_to_Ryd * Temperature[itemp]);

        calc_boson(itemp, fb, dndt);

        for (ik = 0; ik < nk_3ph; ++ik) {
            for (is = 0; is < ns; ++is) {
                gb[ik][is] = std::sqrt(fb[ik][is] * (fb[ik][is] + 1.0));
            }
        }

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

        collision_op->calc_Q_from_L(gb, Q);

        for (ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum; // k index in full grid
            average_over_degenerate_modes(ns, dos->dymat_dos->get_eigenvalues()[k1], 1, Q[ik]);
        }

        // Total diagonal: the 3ph out-scattering part plus the add-on
        // channels, shared by both solvers. The isotope diagonal comes from
        // the operator row sums when its in-scattering is active (so the
        // channel annihilates constant fields exactly), and from the SERTA
        // linewidths otherwise; boundary is exactly diagonal, and 4ph is
        // treated at the RTA level.
        for (ik = 0; ik < nklocal; ik++) {
            auto tmpk = nk_l[ik];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
            for (int s1 = 0; s1 < ns; ++s1) {
                auto qf = Q[ik][s1];
                if (isotope->include_isotope && !collision_op->has_isotope_channel()) {
                    qf += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * isotope_damping_loc[ik][s1];
                }
                if (conductivity->len_boundary > eps) {
                    qf += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * boundary_damping_loc[ik][s1];
                }
                if (conductivity->fph_rta > 0) {
                    qf += fb[k1][s1] * (fb[k1][s1] + 1.0) * 2.0 * damping4[itemp][ik][s1];
                }
                Qfin[ik][s1] = qf;
            }
        }
        if (collision_op->has_isotope_channel()) {
            collision_op->add_isotope_diagonal(gb, Qfin);
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

        if (use_direct) {
            // Dense eigendecomposition of the collision kernel.
            int iters = 0;
            double fres = 0.0;
            converged_this_temp = solve_direct_at_temperature(itemp, beta, gb, Qfin, iters, fres);
            iterations_used[itemp] = iters;
            final_residual[itemp] = fres;
            t_converged[itemp] = converged_this_temp ? 1 : 0;
            write_Q_dF(itemp, Q, dFold, converged_this_temp);
            continue;
        }

        if (use_variational) {
            // Conjugate-gradient solution of the same linear system.
            int iters = 0;
            double fres = 0.0;
            converged_this_temp =
                solve_variational_cg(itemp, beta, gb, Qfin, warm_start ? dF_ir_glob : nullptr, iters, fres);
            iterations_used[itemp] = iters;
            final_residual[itemp] = fres;
            t_converged[itemp] = converged_this_temp ? 1 : 0;
            write_Q_dF(itemp, Q, dFold, converged_this_temp);
            continue;
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
            for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii)
                dF_ir_loc[ii] = 0.0;

                // The in-scattering action W is evaluated on the wedge only:
                // equivalent points carry no new information because
                // dF(Rk) = R dF(k).
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : local_difference, local_norm2)
#endif
            for (int ikl = 0; ikl < nklocal; ++ikl) {

                const auto tmpk = nk_l[ikl];
                const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
                const auto num_equivalent = static_cast<double>(dos->kmesh_dos->kpoint_irred_all[tmpk].size());

                double **Wks_loc;
                allocate(Wks_loc, ns, 3);

                collision_op->calc_W_at(ikl, gb, dFold, Wks_loc);

                // Wks_loc is a contiguous [ns][3] block from allocate().
                average_over_degenerate_modes(ns, dos->dymat_dos->get_eigenvalues()[k1], 3, Wks_loc[0]);

                for (int s1 = 0; s1 < ns; ++s1) {

                    const double Q_final = Qfin[ikl][s1];

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
            const auto rel = (itr > 0 && global_reduce[1] > 0.0) ? std::sqrt(global_reduce[0] / global_reduce[1]) : 0.0;
            if (itr > 0) res_history.push_back(rel);

            MPI_Allreduce(&dF_ir_loc[0], &dF_ir_glob[0], nk_irred * ns * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
                for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii)
                    dF_ir_best[ii] = dF_ir_glob[ii];
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
                              << std::setprecision(2) << rel << '\n'
                              << std::flush;
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
            const auto diverged = itr >= min_cycle && nres >= 3 && res_history[nres - 1] > res_history[nres - 2] &&
                                  res_history[nres - 2] > res_history[nres - 3];

            if (diverged || itr == (max_cycle - 1)) {
                for (unsigned int ii = 0; ii < nk_irred * ns * 3; ++ii)
                    dF_ir_glob[ii] = dF_ir_best[ii];
                collision_op->reconstruct_full_from_wedge(dF_ir_glob, dFold);
                for (ix = 0; ix < 3; ++ix) {
                    for (iy = 0; iy < 3; ++iy) {
                        kappa[itemp][ix][iy] = kappa_best[3 * ix + iy];
                    }
                }
                if (mympi->my_rank == 0) {
                    if (diverged) {
                        std::cout << "   -> WARNING: the iteration is diverging. Keeping the lowest-residual\n"
                                  << "               iterate (iter " << itr_best
                                  << ", |df'-df|/|df| = " << std::scientific << std::setprecision(2) << res_best
                                  << ") and marking this\n"
                                  << "               temperature as NOT converged."
                                  << " Consider reducing IBTE_MIXING.\n"
                                  << std::flush;
                    } else {
                        std::cout << "   -> WARNING: max cycle reached but NOT converged. Keeping the\n"
                                  << "               lowest-residual iterate (iter " << itr_best
                                  << ", |df'-df|/|df| = " << std::scientific << std::setprecision(2) << res_best
                                  << ").\n"
                                  << std::flush;
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
    deallocate(Qfin);
    deallocate(dndt);

    deallocate(kappa_new);
    deallocate(kappa_old);
    deallocate(fb);
    deallocate(gb);
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

void Iterativebte::build_wedge_system(const int itemp, const double beta, double **Qfin_loc, std::vector<double> &qdiag,
                                      std::vector<double> &wrow, std::vector<unsigned char> &mask,
                                      std::vector<double> &b) const
{
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const size_t nrows = static_cast<size_t>(nk_irred) * ns;
    const auto etemp = Temperature[itemp];
    const auto eval = dos->dymat_dos->get_eigenvalues();
    const auto t_to_ryd = thermodynamics->T_to_Ryd;

    // Full-wedge diagonal, replicated on every rank.
    std::vector<double> qdiag_loc(nrows, 0.0);
    qdiag.assign(nrows, 0.0);
    for (int ikl = 0; ikl < nklocal; ++ikl) {
        for (int s = 0; s < ns; ++s) {
            qdiag_loc[static_cast<size_t>(nk_l[ikl]) * ns + s] = Qfin_loc[ikl][s];
        }
    }
    MPI_Allreduce(qdiag_loc.data(), qdiag.data(), static_cast<int>(nrows), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Star multiplicities (the metric), mask of the excluded modes, and the
    // right-hand side.
    wrow.assign(nrows, 0.0);
    mask.assign(nrows, 0);
    b.assign(nrows * 3, 0.0);

    for (unsigned int ik = 0; ik < nk_irred; ++ik) {
        const int k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
        const auto mult = static_cast<double>(dos->kmesh_dos->kpoint_irred_all[ik].size());
        for (int s = 0; s < ns; ++s) {
            const auto row = static_cast<size_t>(ik) * ns + s;
            const auto omega = eval[k1][s];
            if (qdiag[row] < 1.0e-50 || omega < eps8) {
                mask[row] = 1;
                continue;
            }
            wrow[row] = mult;
            const auto xred = omega / (t_to_ryd * etemp);
            const auto dndt_val = pow2(1.0 / (2.0 * sinh(0.5 * xred))) * xred / etemp;
            for (auto j = 0; j < 3; ++j) {
                b[row * 3 + j] = -vel[k1][s][j] * dndt_val / beta;
            }
        }
    }

    project_wedge_vector(b, mask);
}

void Iterativebte::project_wedge_vector(std::vector<double> &v, const std::vector<unsigned char> &mask) const
{
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const size_t nrows = static_cast<size_t>(nk_irred) * ns;
    const auto eval = dos->dymat_dos->get_eigenvalues();

    for (unsigned int ik = 0; ik < nk_irred; ++ik) {
        const int k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
        average_over_degenerate_modes(ns, eval[k1], 3, &v[static_cast<size_t>(ik) * ns * 3]);
    }
    for (size_t row = 0; row < nrows; ++row) {
        if (!mask[row]) continue;
        for (auto j = 0; j < 3; ++j)
            v[row * 3 + j] = 0.0;
    }
}

bool Iterativebte::solve_direct_at_temperature(const int itemp, const double beta, double **sqrt_occ, double **Qfin_loc,
                                               int &iterations_out, double &residual_out)
{
    // SOLVER = DBTE: assemble the multiplicity-symmetrized dense operator
    // from the stored L entries (one-application cost), transform it to the
    // Omega normalization Omega = D^{-1/2} A D^{-1/2} with D = diag(n(n+1))
    // - whose diagonal is 1/tau, so the eigenvalues are scattering rates -
    // take its full eigendecomposition, and report the spectrum diagnostics
    // that the matrix-free solvers cannot access: discretization asymmetry,
    // positive-semidefiniteness, near-null modes with their overlap onto
    // the momentum-drift directions, and the sensitivity of kappa to a
    // low-eigenvalue cutoff. Intended for small meshes.
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const size_t nrows = static_cast<size_t>(nk_irred) * ns;
    const size_t nrows3 = nrows * 3;
    const auto eval = dos->dymat_dos->get_eigenvalues();

    std::vector<double> qdiag(nrows), wrow(nrows);
    std::vector<unsigned char> mask(nrows);
    std::vector<double> b(nrows3);
    build_wedge_system(itemp, beta, Qfin_loc, qdiag, wrow, mask, b);

    // Active (unmasked) scalar rows and the compressed dimension.
    std::vector<int> act_rows;
    for (size_t row = 0; row < nrows; ++row) {
        if (!mask[row]) act_rows.push_back(static_cast<int>(row));
    }
    const int nact3 = 3 * static_cast<int>(act_rows.size());

    constexpr int dbte_max_dim = 25000;
    if (nact3 > dbte_max_dim) {
        exit("solve_direct_at_temperature",
             "The dense collision kernel exceeds the current DBTE size limit\n"
             " (3 * nk_irred * nbranches too large for the rank-0 LAPACK backend).\n"
             " Use a coarser k mesh, or SOLVER = IBTE/VBTE; memory-distributed\n"
             " (ELPA/ScaLAPACK) and GPU (MAGMA/cuSOLVER) backends are planned.");
    }

    // Row-distributed assembly, gathered on rank 0.
    const size_t nrows_loc = static_cast<size_t>(nklocal) * ns * 3;
    std::vector<double> slab(nrows_loc * nrows3, 0.0);
    collision_op->assemble_dense_rows(sqrt_occ, Qfin_loc, slab.data());

    // One-time cross-validation of the assembled rows against the
    // independently coded matrix-free application (calc_W_at + diagonal):
    // both must produce identical results for any wedge vector.
    if (!dbte_assembly_checked) {
        dbte_assembly_checked = true;
        std::vector<double> xtest(nrows3, 0.0);
        for (size_t row = 0; row < nrows; ++row) {
            if (mask[row]) continue;
            for (auto j = 0; j < 3; ++j)
                xtest[row * 3 + j] = b[row * 3 + j] / qdiag[row];
        }
        project_wedge_vector(xtest, mask);
        collision_op->reconstruct_full_from_wedge(xtest.data(), dFold);

        double maxdiff_loc = 0.0, scale_loc = 0.0;
        for (int ikl = 0; ikl < nklocal; ++ikl) {
            const auto tmpk = nk_l[ikl];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;
            const auto sqrt_mrow = std::sqrt(static_cast<double>(dos->kmesh_dos->kpoint_irred_all[tmpk].size()));

            double **Wks_loc;
            allocate(Wks_loc, ns, 3);
            collision_op->calc_W_at(ikl, sqrt_occ, dFold, Wks_loc);

            for (int s1 = 0; s1 < ns; ++s1) {
                const auto grow = static_cast<size_t>(tmpk) * ns + s1;
                for (auto x = 0; x < 3; ++x) {
                    // Assembled row applied to the metric-scaled vector...
                    double y_mat = 0.0;
                    const double *rowp = slab.data() + ((static_cast<size_t>(ikl) * ns + s1) * 3 + x) * nrows3;
                    for (size_t c = 0; c < nrows; ++c) {
                        const auto sq = std::sqrt(wrow[c] > 0.0 ? wrow[c] : 1.0);
                        for (auto y = 0; y < 3; ++y) {
                            y_mat += rowp[c * 3 + y] * sq * xtest[c * 3 + y];
                        }
                    }
                    // ...equals the metric-scaled matrix-free application.
                    const auto y_free =
                        sqrt_mrow * (qdiag[grow] * xtest[grow * 3 + x] + (mask[grow] ? 0.0 : Wks_loc[s1][x]));
                    maxdiff_loc = std::max(maxdiff_loc, std::abs(y_mat - y_free));
                    scale_loc = std::max(scale_loc, std::abs(y_free));
                }
            }
            deallocate(Wks_loc);
        }
        double maxdiff = 0.0, scale = 0.0;
        MPI_Allreduce(&maxdiff_loc, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&scale_loc, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (mympi->my_rank == 0) {
            std::cout << "      (assembly cross-check vs the matrix-free operator: max rel diff = " << std::scientific
                      << std::setprecision(1) << (scale > 0.0 ? maxdiff / scale : 0.0) << ")\n";
        }
    }

    std::vector<double> S; // full matrix, row-major, rank 0 only
    if (mympi->my_rank == 0) {
        S.assign(nrows3 * nrows3, 0.0);
        std::vector<double> buf;
        for (auto r = 0; r < mympi->nprocs; ++r) {
            // Wedge rows of rank r follow the round-robin distribution.
            std::vector<int> irows;
            for (unsigned int i = 0; i < nk_irred; ++i) {
                if (static_cast<int>(i) % mympi->nprocs == r) irows.push_back(static_cast<int>(i));
            }
            const double *src = nullptr;
            if (r == 0) {
                src = slab.data();
            } else {
                buf.resize(irows.size() * ns * 3 * nrows3);
                MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                src = buf.data();
            }
            for (size_t il = 0; il < irows.size(); ++il) {
                const size_t gbase = static_cast<size_t>(irows[il]) * ns * 3;
                for (int rs = 0; rs < ns * 3; ++rs) {
                    std::copy(src + (il * ns * 3 + rs) * nrows3,
                              src + (il * ns * 3 + rs + 1) * nrows3,
                              S.begin() + (gbase + rs) * nrows3);
                }
            }
        }
    } else {
        MPI_Send(slab.data(), static_cast<int>(slab.size()), MPI_DOUBLE, 0, mympi->my_rank, MPI_COMM_WORLD);
    }
    slab.clear();
    slab.shrink_to_fit();

    double kappa9[9] = {};
    double residual = 0.0;
    std::vector<double> dF_wedge(nrows3, 0.0);

    if (mympi->my_rank == 0) {

        // Degeneracy-reduced basis: one orthonormal (symmetric) combination
        // per degenerate block, u_I = d^{-1/2} sum_{s in I} e_s. This
        // performs the degeneracy averaging of the iterative solvers
        // exactly, without the 3(d-1) artificial null modes a projector
        // P S P would inject into the spectrum. Masked rows never share a
        // block with active ones (degenerate partners have equal omega).
        const auto tol_omega = 1.0e-7;
        std::vector<std::pair<int, int>> blocks; // [first,last) active scalar rows within one ik
        for (unsigned int ik = 0; ik < nk_irred; ++ik) {
            const int k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
            int begin = 0;
            for (int s = 1; s <= ns; ++s) {
                if (s < ns && std::abs(eval[k1][s] - eval[k1][begin]) < tol_omega) continue;
                if (!mask[static_cast<size_t>(ik) * ns + begin]) {
                    blocks.push_back({static_cast<int>(ik * ns) + begin, static_cast<int>(ik * ns) + s});
                }
                if (s < ns) begin = s;
            }
        }
        const int nred3 = 3 * static_cast<int>(blocks.size());

        std::vector<double> A(static_cast<size_t>(nred3) * nred3, 0.0);
        std::vector<double> inv_sqrt_d(blocks.size());
        std::vector<double> gblk(blocks.size()); // sqrt(n(n+1)) per block (equal within it)
        for (size_t I = 0; I < blocks.size(); ++I) {
            inv_sqrt_d[I] = 1.0 / std::sqrt(static_cast<double>(blocks[I].second - blocks[I].first));
            const auto ikI = static_cast<unsigned int>(blocks[I].first / ns);
            const int k1I = dos->kmesh_dos->kpoint_irred_all[ikI][0].knum;
            gblk[I] = sqrt_occ[k1I][blocks[I].first % ns];
        }
        for (size_t I = 0; I < blocks.size(); ++I) {
            for (size_t J = 0; J < blocks.size(); ++J) {
                const auto fac = inv_sqrt_d[I] * inv_sqrt_d[J];
                for (auto x = 0; x < 3; ++x) {
                    for (auto y = 0; y < 3; ++y) {
                        double sum = 0.0;
                        for (auto i = blocks[I].first; i < blocks[I].second; ++i) {
                            for (auto j = blocks[J].first; j < blocks[J].second; ++j) {
                                sum += S[(static_cast<size_t>(i) * 3 + x) * nrows3 + static_cast<size_t>(j) * 3 + y];
                            }
                        }
                        A[(static_cast<size_t>(J) * 3 + y) * nred3 + static_cast<size_t>(I) * 3 + x] = fac * sum;
                    }
                }
            }
        }
        S.clear();
        S.shrink_to_fit();

        // Omega normalization: divide out the occupation metric so the
        // diagonal becomes 1/tau and the eigenvalues are scattering rates.
        for (size_t J = 0; J < blocks.size(); ++J) {
            for (size_t I = 0; I < blocks.size(); ++I) {
                const auto fac = 1.0 / (gblk[I] * gblk[J]);
                for (auto x = 0; x < 3; ++x) {
                    for (auto y = 0; y < 3; ++y) {
                        A[(J * 3 + y) * nred3 + I * 3 + x] *= fac;
                    }
                }
            }
        }

        // Right-hand side in the Omega-normalized metric, block-reduced (b
        // is block constant, so the reduced component is sqrt(d) b, and the
        // congruence divides by g).
        std::vector<double> btil3(nred3);
        for (size_t I = 0; I < blocks.size(); ++I) {
            const auto srow = blocks[I].first; // representative scalar row
            const auto fac = std::sqrt(wrow[srow]) / (inv_sqrt_d[I] * gblk[I]);
            for (auto x = 0; x < 3; ++x) {
                btil3[I * 3 + x] = fac * b[static_cast<size_t>(srow) * 3 + x];
            }
        }

        // Restrict to the little-group-invariant subspace. The collision
        // operator is only defined on fields with dF(Rk) = R dF(k); the
        // complementary vector components at high-symmetry k points carry an
        // arbitrary (choice-of-operation dependent, non-symmetric) extension
        // that must not enter the spectrum. Per block, an orthonormal basis
        // of range(P_littlegroup) is taken from the projector eigenvectors.
        std::vector<Eigen::Matrix3d> Ebasis(blocks.size());
        std::vector<int> mdim(blocks.size()), offs(blocks.size());
        int ndim = 0;
        for (size_t I = 0; I < blocks.size(); ++I) {
            const auto ikI = static_cast<unsigned int>(blocks[I].first / ns);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(
                collision_op->get_littlegroup_projector(static_cast<int>(ikI)));
            Ebasis[I] = es.eigenvectors(); // eigenvalues ascending
            auto m = 0;
            for (auto a = 0; a < 3; ++a) {
                if (es.eigenvalues()(a) > 0.5) ++m;
            }
            mdim[I] = m;
            offs[I] = ndim;
            ndim += m;
        }
        const auto ecol = [&](const size_t I, const int a) { // a-th invariant basis vector
            return Ebasis[I].col(3 - mdim[I] + a);
        };

        std::vector<double> btil;
        {
            std::vector<double> Ainv(static_cast<size_t>(ndim) * ndim, 0.0);
            std::vector<double> btil_inv(ndim, 0.0);
            for (size_t I = 0; I < blocks.size(); ++I) {
                for (auto a = 0; a < mdim[I]; ++a) {
                    const auto eI = ecol(I, a);
                    double sum_b = 0.0;
                    for (auto x = 0; x < 3; ++x)
                        sum_b += eI(x) * btil3[I * 3 + x];
                    btil_inv[offs[I] + a] = sum_b;
                    for (size_t J = 0; J < blocks.size(); ++J) {
                        for (auto bb = 0; bb < mdim[J]; ++bb) {
                            const auto eJ = ecol(J, bb);
                            double sum = 0.0;
                            for (auto x = 0; x < 3; ++x) {
                                for (auto y = 0; y < 3; ++y) {
                                    sum +=
                                        eI(x) *
                                        A[(static_cast<size_t>(J) * 3 + y) * nred3 + static_cast<size_t>(I) * 3 + x] *
                                        eJ(y);
                                }
                            }
                            Ainv[static_cast<size_t>(offs[J] + bb) * ndim + offs[I] + a] = sum;
                        }
                    }
                }
            }
            A = std::move(Ainv);
            btil = std::move(btil_inv);
        }

        double asym2 = 0.0, norm2 = 0.0;
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < i; ++j) {
                const auto aij = A[static_cast<size_t>(j) * ndim + i];
                const auto aji = A[static_cast<size_t>(i) * ndim + j];
                asym2 += pow2(aij - aji);
                norm2 += pow2(aij) + pow2(aji);
            }
            norm2 += pow2(A[static_cast<size_t>(i) * ndim + i]);
        }
        const auto asym = norm2 > 0.0 ? std::sqrt(2.0 * asym2 / norm2) : 0.0;
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < i; ++j) {
                const auto sym = 0.5 * (A[static_cast<size_t>(j) * ndim + i] + A[static_cast<size_t>(i) * ndim + j]);
                A[static_cast<size_t>(j) * ndim + i] = sym;
                A[static_cast<size_t>(i) * ndim + j] = sym;
            }
        }

        std::cout << "      dense collision kernel: dimension " << ndim << " (" << nrows3 - nact3 << " excluded rows, "
                  << nact3 - nred3 << " folded into degenerate blocks, " << nred3 - ndim
                  << " non-invariant components removed), memory " << std::fixed << std::setprecision(1)
                  << static_cast<double>(ndim) * ndim * 8.0 / 1048576.0 << " MB\n";
        std::cout << "      asymmetry |S - S^T|_F / |S|_F = " << std::scientific << std::setprecision(1) << asym
                  << " (on the invariant subspace; symmetrized for the eigendecomposition)\n";

        std::vector<double> evals;
        solve_dense_symmetric(ndim, A, evals);

        const auto lam_max = evals.empty() ? 0.0 : evals.back();
        const auto lam_tol = lam_max * 1.0e-12;
        const auto to_kayser = Hz_to_kayser / time_ry; // rates: internal -> cm^-1
        int n_negative = 0, n_nearnull = 0;
        for (const auto lam: evals) {
            if (lam < -lam_tol) ++n_negative;
            if (std::abs(lam) < 1.0e-8 * lam_max) ++n_nearnull;
        }
        std::cout << "      eigenvalues (Omega normalization; scattering rates 1/tau in cm^-1):\n";
        std::cout << "        min = " << std::scientific << std::setprecision(2) << evals.front() * to_kayser
                  << ", max = " << lam_max * to_kayser << "; negative (< -1e-12 max): " << n_negative
                  << "; near-null (|lambda| < 1e-8 max): " << n_nearnull << '\n';
        if (n_negative > 0) {
            std::cout << "      WARNING: the physical collision kernel is positive semidefinite;\n"
                      << "               negative eigenvalues indicate a too-coarse mesh or an\n"
                      << "               inadequate smearing width.\n";
        }

        // Overlap of the softest modes with the momentum-drift space
        // (candidate fields dF(k) = k_cart, branch independent, up to
        // occupation weighting), orthonormalized in the compressed metric.
        {
            Eigen::Matrix3d mat_k2cart;
            const auto &rlavec = system->get_primcell().reciprocal_lattice_vector;
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j)
                    mat_k2cart(j, i) = rlavec(i, j);
            }

            std::vector<std::vector<double>> drift(3, std::vector<double>(ndim, 0.0));
            for (size_t I = 0; I < blocks.size(); ++I) {
                const auto ik = static_cast<unsigned int>(blocks[I].first / ns);
                const int k1 = dos->kmesh_dos->kpoint_irred_all[ik][0].knum;
                Eigen::Vector3d kf;
                for (auto j = 0; j < 3; ++j)
                    kf(j) = dos->kmesh_dos->xk[k1][j];
                const Eigen::Vector3d kc = mat_k2cart * kf;
                // In the Omega variables the drift candidate carries the
                // occupation weight (y = G x), projected onto the invariant
                // basis of the block.
                const auto fac = std::sqrt(wrow[blocks[I].first]) * gblk[I] / inv_sqrt_d[I];
                for (auto a = 0; a < mdim[I]; ++a) {
                    const auto eI = ecol(I, a);
                    for (auto c = 0; c < 3; ++c) {
                        drift[c][offs[I] + a] = fac * eI(c) * kc(c);
                    }
                }
            }
            std::vector<int> kept;
            for (auto a = 0; a < 3; ++a) {
                for (const auto kb: kept) {
                    double pr = 0.0;
                    for (int i = 0; i < ndim; ++i)
                        pr += drift[a][i] * drift[kb][i];
                    for (int i = 0; i < ndim; ++i)
                        drift[a][i] -= pr * drift[kb][i];
                }
                double nrm = 0.0;
                for (int i = 0; i < ndim; ++i)
                    nrm += pow2(drift[a][i]);
                if (nrm > 1.0e-24) {
                    nrm = 1.0 / std::sqrt(nrm);
                    for (int i = 0; i < ndim; ++i)
                        drift[a][i] *= nrm;
                    kept.push_back(a);
                }
            }
            for (auto m = 0; m < std::min(3, ndim); ++m) {
                double ov2 = 0.0;
                for (const auto kb: kept) {
                    double pr = 0.0;
                    for (int i = 0; i < ndim; ++i)
                        pr += A[static_cast<size_t>(m) * ndim + i] * drift[kb][i];
                    ov2 += pow2(pr);
                }
                std::cout << "      softest mode " << m << ": 1/tau = " << std::scientific << std::setprecision(2)
                          << evals[m] * to_kayser << " cm^-1, overlap with the momentum-drift space = " << std::fixed
                          << std::setprecision(3) << std::sqrt(ov2) << '\n';
            }
        }

        // kappa via the spectral pseudo-inverse; the cutoff table exposes
        // near-singular directions.
        std::vector<double> coef(ndim, 0.0);
        const auto lam_null = std::max(0.0, lam_max * 1.0e-12);
        int n_null = 0;
        for (int i = 0; i < ndim; ++i) {
            if (evals[i] <= lam_null) {
                ++n_null;
                continue;
            }
            double pr = 0.0;
            for (int j = 0; j < ndim; ++j)
                pr += A[static_cast<size_t>(i) * ndim + j] * btil[j];
            coef[i] = pr / evals[i];
        }
        if (n_null > 0) {
            std::cout << "      " << n_null << " numerically null mode(s) (lambda <= 1e-12 max) excluded\n";
        }

        const auto kappa_of_drop = [&](const int mdrop, double *k9, std::vector<double> *dF_keep) {
            std::vector<double> xt(ndim, 0.0);
            int skipped = 0;
            for (int i = 0; i < ndim; ++i) {
                if (coef[i] == 0.0) continue;
                if (skipped < mdrop) {
                    ++skipped;
                    continue;
                }
                const auto c = coef[i];
                for (int j = 0; j < ndim; ++j)
                    xt[j] += c * A[static_cast<size_t>(i) * ndim + j];
            }
            // Expand: invariant basis -> Cartesian per block, undo the Omega
            // congruence (divide by g), then block-constant over degenerate
            // partners and the multiplicity metric.
            std::vector<double> dF(nrows3, 0.0);
            for (size_t I = 0; I < blocks.size(); ++I) {
                const auto fac = inv_sqrt_d[I] / (std::sqrt(wrow[blocks[I].first]) * gblk[I]);
                double v3[3] = {0.0, 0.0, 0.0};
                for (auto a = 0; a < mdim[I]; ++a) {
                    const auto eI = ecol(I, a);
                    for (auto x = 0; x < 3; ++x)
                        v3[x] += eI(x) * xt[offs[I] + a];
                }
                for (auto srow = blocks[I].first; srow < blocks[I].second; ++srow) {
                    for (auto x = 0; x < 3; ++x) {
                        dF[static_cast<size_t>(srow) * 3 + x] = fac * v3[x];
                    }
                }
            }
            collision_op->reconstruct_full_from_wedge(dF.data(), dFold);
            double **ktmp;
            allocate(ktmp, 3, 3);
            calc_kappa(itemp, dFold, ktmp);
            for (auto a = 0; a < 3; ++a) {
                for (auto c2 = 0; c2 < 3; ++c2)
                    k9[3 * a + c2] = ktmp[a][c2];
            }
            deallocate(ktmp);
            if (dF_keep) *dF_keep = dF;
        };

        std::cout << "      kappa_xx/yy/zz vs dropping the m softest (non-null) modes:\n";
        double k9tmp[9];
        for (const auto mdrop: {0, 1, 2, 4, 8, 16}) {
            if (mdrop >= ndim - n_null) break;
            if (mdrop == 0) {
                kappa_of_drop(0, kappa9, &dF_wedge);
                std::copy(kappa9, kappa9 + 9, k9tmp);
            } else {
                kappa_of_drop(mdrop, k9tmp, nullptr);
            }
            std::cout << "        m = " << std::setw(3) << mdrop << ":" << std::scientific << std::setprecision(4)
                      << std::setw(13) << k9tmp[0] << std::setw(13) << k9tmp[4] << std::setw(13) << k9tmp[8]
                      << "  [W/mK]\n";
        }

        // Fraction of the right-hand side living in the excluded null space
        // (the direct analogue of an unresolvable residual).
        {
            double r2 = 0.0, b2 = 0.0;
            for (int i = 0; i < ndim; ++i) {
                double pr = 0.0;
                for (int j = 0; j < ndim; ++j)
                    pr += A[static_cast<size_t>(i) * ndim + j] * btil[j];
                b2 += pow2(pr);
                if (evals[i] <= lam_null) r2 += pow2(pr);
            }
            residual = b2 > 0.0 ? std::sqrt(r2 / b2) : 0.0;
            std::cout << "      |b| fraction in the excluded null space = " << std::scientific << std::setprecision(2)
                      << residual << '\n'
                      << std::flush;
        }
    }

    MPI_Bcast(kappa9, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&residual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(dF_wedge.data(), static_cast<int>(nrows3), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (auto a = 0; a < 3; ++a) {
        for (auto c2 = 0; c2 < 3; ++c2)
            kappa[itemp][a][c2] = kappa9[3 * a + c2];
    }
    collision_op->reconstruct_full_from_wedge(dF_wedge.data(), dFold);

    iterations_out = 1;
    residual_out = residual;
    return true;
}

bool Iterativebte::solve_variational_cg(const int itemp, const double beta, double **sqrt_occ, double **Qfin_loc,
                                        const double *x0_wedge, int &iterations_out, double &residual_out)
{
    // Solve (Q_diag + W) dF = b with preconditioned conjugate gradients.
    // In the dF variables the off-diagonal couplings are the equilibrium
    // rates n1 n2 (n3+1) |V3|^2 delta(...), which detailed balance makes
    // symmetric under exchange of the coupled modes on the energy shell, so
    // the operator is self-adjoint under the plain star-multiplicity-
    // weighted inner product <u,v> = sum_{wedge k,s} mult_k u.v used by all
    // dot products below; the preconditioner is the diagonal. (Smearing
    // spreads slightly off shell, which the self-adjointness check below
    // quantifies.)
    // Because kappa is the value of the variational functional, its error is
    // quadratic in the residual, so ITER_THRESHOLD acts on the relative
    // residual here.
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const size_t nrows = static_cast<size_t>(nk_irred) * ns;
    const size_t nrows3 = nrows * 3;
    const auto etemp = Temperature[itemp];
    const auto eval = dos->dymat_dos->get_eigenvalues();
    const auto t_to_ryd = thermodynamics->T_to_Ryd;

    std::vector<double> qdiag(nrows), wrow(nrows);
    std::vector<unsigned char> mask(nrows);
    std::vector<double> b(nrows3);
    build_wedge_system(itemp, beta, Qfin_loc, qdiag, wrow, mask, b);

    // Degeneracy projection + masking of a wedge vector (idempotent).
    auto project = [&](std::vector<double> &v) { project_wedge_vector(v, mask); };

    // Weighted dot product; every rank holds the full replicated vectors,
    // so this involves no communication.
    auto wdot = [&](const std::vector<double> &u, const std::vector<double> &v) {
        double sum = 0.0;
        for (size_t row = 0; row < nrows; ++row) {
            if (mask[row]) continue;
            sum += wrow[row] *
                   (u[row * 3] * v[row * 3] + u[row * 3 + 1] * v[row * 3 + 1] + u[row * 3 + 2] * v[row * 3 + 2]);
        }
        return sum;
    };

    // y = (Q_diag + W) x on the wedge; dFold serves as the full-grid scratch.
    std::vector<double> yloc(nrows3);
    auto apply_operator = [&](const std::vector<double> &x, std::vector<double> &y) {
        collision_op->reconstruct_full_from_wedge(x.data(), dFold);
        std::fill(yloc.begin(), yloc.end(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int ikl = 0; ikl < nklocal; ++ikl) {
            const auto tmpk = nk_l[ikl];
            const int k1 = dos->kmesh_dos->kpoint_irred_all[tmpk][0].knum;

            double **Wks_loc;
            allocate(Wks_loc, ns, 3);
            collision_op->calc_W_at(ikl, sqrt_occ, dFold, Wks_loc);
            average_over_degenerate_modes(ns, eval[k1], 3, Wks_loc[0]);

            for (int s = 0; s < ns; ++s) {
                const auto row = static_cast<size_t>(tmpk) * ns + s;
                if (mask[row]) continue;
                for (auto j = 0; j < 3; ++j) {
                    yloc[row * 3 + j] = qdiag[row] * x[row * 3 + j] + Wks_loc[s][j];
                }
            }
            deallocate(Wks_loc);
        }
        MPI_Allreduce(yloc.data(), y.data(), static_cast<int>(nrows3), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    };

    auto precondition = [&](const std::vector<double> &rin, std::vector<double> &zout) {
        for (size_t row = 0; row < nrows; ++row) {
            if (mask[row]) {
                for (auto j = 0; j < 3; ++j)
                    zout[row * 3 + j] = 0.0;
            } else {
                for (auto j = 0; j < 3; ++j)
                    zout[row * 3 + j] = rin[row * 3 + j] / qdiag[row];
            }
        }
    };

    // One-time numerical self-adjointness check of the discretized operator
    // on the subspace CG actually explores: the first two (preconditioned)
    // Krylov vectors of the physical right-hand side. Fixed-width smearing
    // is symmetric analytically up to off-shell occupation factors;
    // tetrahedron and adaptive weights are direction dependent and break
    // the symmetry more strongly away from the energy shell.
    if (!cg_symmetry_checked) {
        cg_symmetry_checked = true;
        std::vector<double> u(nrows3), v(nrows3), Au(nrows3), Av(nrows3);
        u = b;
        collision_op->project_onto_littlegroup(u.data());
        project(u);
        precondition(u, v); // v <- M^-1 u (temporarily)
        u = v;              // u = M^-1 b
        apply_operator(u, Au);
        precondition(Au, v); // v = M^-1 A M^-1 b, the second Krylov vector
        apply_operator(v, Av);
        const auto s1 = wdot(u, Av);
        const auto s2 = wdot(Au, v);
        const auto scale = std::max(std::abs(s1), std::abs(s2));
        const auto asym = scale > 0.0 ? std::abs(s1 - s2) / scale : 0.0;
        if (mympi->my_rank == 0) {
            std::cout << "      (operator self-adjointness on the Krylov subspace: |<u,Av>-<Au,v>|/|<u,Av>| = "
                      << std::scientific << std::setprecision(1) << asym << ")\n";
            if (asym > 1.0e-3) {
                std::cout << "      Note: the discretized collision operator is not exactly symmetric\n"
                          << "      (tetrahedron/adaptive smearing weights); judge the run by the final\n"
                          << "      relative residual, which is recomputed explicitly at the end.\n";
            }
        }
    }

    // PCG with the diagonal preconditioner.
    std::vector<double> x(nrows3, 0.0), r(nrows3), z(nrows3), p(nrows3), Ap(nrows3);

    const auto bnorm = std::sqrt(wdot(b, b));
    iterations_out = 0;
    residual_out = 0.0;

    if (bnorm <= 0.0) {
        // No driving force (all modes masked): dF = 0 is the exact solution.
        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                kappa[itemp][i][j] = 0.0;
        std::fill(x.begin(), x.end(), 0.0);
        collision_op->reconstruct_full_from_wedge(x.data(), dFold);
        return true;
    }

    if (x0_wedge) {
        for (size_t i = 0; i < nrows3; ++i)
            x[i] = x0_wedge[i];
        project(x);
        apply_operator(x, Ap);
        for (size_t i = 0; i < nrows3; ++i)
            r[i] = b[i] - Ap[i];
    } else {
        r = b;
    }

    precondition(r, z);
    p = z;
    auto rz = wdot(r, z);

    auto converged = false;
    auto rel = std::sqrt(wdot(r, r)) / bnorm;
    auto res_best = rel;
    std::vector<double> x_best = x;
    int itr_best = 0;
    int n_growth = 0;

    for (auto itr = 0; itr < max_cycle; ++itr) {

        if (mympi->my_rank == 0) {
            std::cout << "   -> iter " << std::setw(3) << itr << ": " << std::flush;
        }

        apply_operator(p, Ap);
        const auto pAp = wdot(p, Ap);
        if (pAp <= 0.0) {
            if (mympi->my_rank == 0) {
                std::cout << '\n'
                          << "   -> WARNING: <p, Ap> <= 0 encountered; the discretized operator is not\n"
                          << "               positive definite on this vector. Stopping with the best iterate.\n"
                          << std::flush;
            }
            break;
        }
        const auto alpha = rz / pAp;
        for (size_t i = 0; i < nrows3; ++i)
            x[i] += alpha * p[i];
        for (size_t i = 0; i < nrows3; ++i)
            r[i] -= alpha * Ap[i];

        const auto rel_new = std::sqrt(wdot(r, r)) / bnorm;
        if (rel_new > rel) {
            ++n_growth;
        } else {
            n_growth = 0;
        }
        rel = rel_new;

        // Progress display: kappa of the current iterate.
        collision_op->reconstruct_full_from_wedge(x.data(), dFold);
        double **kappa_now;
        allocate(kappa_now, 3, 3);
        calc_kappa(itemp, dFold, kappa_now);
        if (mympi->my_rank == 0) {
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    std::cout << std::setw(12) << std::scientific << std::setprecision(2) << kappa_now[i][j];
                }
            }
            std::cout << std::setw(14) << std::scientific << std::setprecision(2) << rel << '\n' << std::flush;
        }

        if (rel < res_best) {
            res_best = rel;
            itr_best = itr + 1;
            x_best = x;
        }

        iterations_out = itr + 1;

        if (rel < convergence_criteria) {
            converged = true;
            for (auto i = 0; i < 3; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    kappa[itemp][i][j] = kappa_now[i][j];
                }
            }
            deallocate(kappa_now);
            if (mympi->my_rank == 0) {
                std::cout << "   -> Converged is achieved                 "
                          << "                                            "
                          << "                                    " << std::setw(14) << std::scientific
                          << std::setprecision(2) << rel << '\n'
                          << std::flush;
            }
            break;
        }
        deallocate(kappa_now);

        if (n_growth >= 3) {
            if (mympi->my_rank == 0) {
                std::cout << "   -> WARNING: the residual keeps growing (non-symmetric discretization?).\n"
                          << "               Stopping with the best iterate.\n"
                          << std::flush;
            }
            break;
        }

        precondition(r, z);
        const auto rz_new = wdot(r, z);
        const auto beta_cg = rz_new / rz;
        rz = rz_new;
        for (size_t i = 0; i < nrows3; ++i)
            p[i] = z[i] + beta_cg * p[i];
    }

    if (!converged) {
        // Keep the lowest-residual iterate and report honestly.
        x = x_best;
        collision_op->reconstruct_full_from_wedge(x.data(), dFold);
        double **kappa_now;
        allocate(kappa_now, 3, 3);
        calc_kappa(itemp, dFold, kappa_now);
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                kappa[itemp][i][j] = kappa_now[i][j];
            }
        }
        deallocate(kappa_now);
        if (mympi->my_rank == 0) {
            std::cout << "   -> WARNING: NOT converged within MAX_CYCLE = " << max_cycle
                      << " CG iterations. Keeping the lowest-residual iterate (iter " << itr_best
                      << ", |r|/|b| = " << std::scientific << std::setprecision(2) << res_best << ").\n"
                      << std::flush;
        }
    }

    // Explicit final residual of the kept iterate: guards against drift of
    // the recursive CG residual when the discretized operator is not
    // exactly symmetric (tetrahedron/adaptive weights).
    apply_operator(x, Ap);
    for (size_t i = 0; i < nrows3; ++i)
        r[i] = b[i] - Ap[i];
    const auto rel_true = std::sqrt(wdot(r, r)) / bnorm;
    residual_out = rel_true;
    if (converged && rel_true > convergence_criteria) {
        converged = false;
        if (mympi->my_rank == 0) {
            std::cout << "   -> WARNING: the recursive CG residual converged but the explicitly\n"
                      << "               recomputed one is " << std::scientific << std::setprecision(2) << rel_true
                      << "; marking this temperature as NOT converged.\n"
                      << std::flush;
        }
    }
    if (mympi->my_rank == 0) {
        std::cout << "      final |r|/|b| (explicit) = " << std::scientific << std::setprecision(2) << rel_true << '\n'
                  << std::flush;
    }

    // dFold holds the reconstruction of the kept iterate, so write_Q_dF
    // stores data consistent with kappa[itemp].
    collision_op->reconstruct_full_from_wedge(x.data(), dFold);
    return converged;
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
        fs_result << system->get_primcell().number_of_atoms << " " << system->get_primcell().number_of_elems << '\n';
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
            ibte_io->store_ibte_temperature(itemp,
                                            &Q_all[0][0],
                                            df_flat.data(),
                                            &kappa[itemp][0][0],
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
