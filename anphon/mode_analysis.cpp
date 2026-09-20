/*
 mode_analysis.cpp

Copyright (c) 2018 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mode_analysis.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "anharmonic_core.h"
#include "constants.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "hdf5_parser.h"
#include "integration.h"
#include "interpolation.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "selfenergy.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

namespace
{
// [lo, hi) of the branches degenerate with s (same tolerance as the weight averaging).
std::pair<int, int> degenerate_block(const double *w, const int ns, const int s)
{
    const auto tol = 1.0e-7 * time_ry / Hz_to_kayser;
    int lo = s, hi = s + 1;
    while (lo > 0 && std::abs(w[lo - 1] - w[s]) < tol) --lo;
    while (hi < ns && std::abs(w[hi] - w[s]) < tol) ++hi;
    return {lo, hi};
}

} // namespace

ModeAnalysis::ModeAnalysis(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

ModeAnalysis::~ModeAnalysis()
{
    deallocate_variables();
}

void ModeAnalysis::set_default_variables()
{
    ks_analyze_mode = false;
    calc_imagpart = true;
    calc_realpart = false;
    calc_fstate_omega = false;
    print_V3 = 0;
    print_V4 = 0;
    calc_selfenergy = 0;
    spectral_func = false;
}

void ModeAnalysis::deallocate_variables() const
{}

void ModeAnalysis::setup_mode_analysis()
{
    // Judge if ks_analyze_mode should be turned on or not.

    unsigned int i;

    MPI_Bcast(&selfenergy_mode, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&interpolate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(kmesh_coarse, 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(omega_range, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kpoint->target_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (selfenergy_mode && kpoint->target_mode == 1) {
        // Path targets: build the band-structure k points (collective).
        kpoint->setup_kpoint_band(kpoint->kpInp_targets, system->get_primcell().reciprocal_lattice_vector);
    }

    if (run.my_rank == 0) {
        const auto ns = dynamical->neval;
        // Off-mesh targets require shifted-grid support for every requested quantity.
        auto add_target = [&](const double *ktmp, const unsigned int snum_tmp, const double kaxis) {
            if (snum_tmp <= 0 || snum_tmp > ns) exit("setup_mode_analysis", "Mode index out of range.");
            const auto knum_tmp = dos->kmesh_dos->get_knum(ktmp);
            TargetResult r;
            for (auto j = 0; j < 3; ++j) r.xk[j] = ktmp[j];
            r.branch = snum_tmp;
            r.on_mesh = knum_tmp != -1;
            r.kaxis = kaxis;
            const auto id = static_cast<unsigned int>(results.size());
            results.push_back(r);
            if (knum_tmp == -1) {
                const bool supported =
                    (calc_selfenergy || spectral_func || print_V3 || print_V4 || calc_fstate_omega || interpolate) &&
                    integration->ismear != 2 && anharmonic_core->quartic_mode != 2;
                if (!supported) {
                    exit(
                        "setup_mode_analysis",
                        "A target k point is not on the k-point grid. Points off the grid are supported only for"
                        " SELF_ENERGY/LINEWIDTH (with REALPART/SHIFT), SELF_W, FSTATE_W, PRINTV3/4 and INTERPOLATE with"
                        " ISMEAR = -1, 0, 1 (no QUARTIC = 2).");
                }
                kslist_offmesh.push_back({{ktmp[0], ktmp[1], ktmp[2]}, snum_tmp - 1, id});
            } else {
                kslist.push_back(knum_tmp * ns + snum_tmp - 1);
                kslist_id.push_back(id);
            }
        };

        kslist.clear();
        kslist_offmesh.clear();
        kslist_id.clear();
        results.clear();

        if (selfenergy_mode) {
            ks_analyze_mode = true;
            if (run.verbosity > 0) {
                std::cout << "\n MODE = selfenergy: analysis of the phonon modes given in the &kpoint field.\n\n";
            }
            std::vector<unsigned int> branches; // 1-based
            {
                std::string spec = branches_spec;
                std::transform(spec.begin(), spec.end(), spec.begin(), ::tolower);
                if (spec == "all") {
                    for (unsigned int s = 1; s <= ns; ++s) branches.push_back(s);
                } else {
                    std::vector<std::string> items;
                    boost::split(items, spec, boost::is_any_of(","), boost::token_compress_on);
                    for (auto &item: items) {
                        boost::trim(item);
                        if (item.empty()) continue;
                        const auto dash = item.find('-');
                        const auto lo = boost::lexical_cast<unsigned int>(item.substr(0, dash));
                        const auto hi =
                            dash == std::string::npos ? lo : boost::lexical_cast<unsigned int>(item.substr(dash + 1));
                        if (lo < 1 || hi > ns || lo > hi) exit("setup_mode_analysis", "BRANCHES out of range.");
                        for (auto b = lo; b <= hi; ++b) branches.push_back(b);
                    }
                }
            }
            if (kpoint->target_mode == 0) {
                for (const auto &kp: kpoint->kpInp_targets) {
                    double ktmp[3];
                    for (auto j = 0; j < 3; ++j) ktmp[j] = boost::lexical_cast<double>(kp.kpelem[j]);
                    for (const auto b: branches) add_target(ktmp, b, -1.0);
                }
            } else if (kpoint->target_mode == 1) {
                for (unsigned int ik = 0; ik < kpoint->kpoint_bs->nk; ++ik) {
                    for (const auto b: branches) add_target(kpoint->kpoint_bs->xk[ik], b, kpoint->kpoint_bs->kaxis[ik]);
                }
            } else {
                exit("setup_mode_analysis", "MODE = selfenergy needs KPMODE = 0 or 1 in the &kpoint field.");
            }
            if (run.verbosity > 0) {
                std::cout << " The number of targets = " << kslist.size() + kslist_offmesh.size() << " ("
                          << kslist_offmesh.size() << " off the k-point grid)\n";
            }
        } else if (!ks_input.empty()) {
            ks_analyze_mode = true;

            if (run.verbosity > 0) {
                std::cout << "\n KS_INPUT-tag is given : Analysis on the specified phonon modes\n";
                std::cout << " will be performed instead of thermal conductivity calculation.\n\n";
            }
            warn("setup_mode_analysis",
                 "KS_INPUT is deprecated: use MODE = selfenergy with the targets in &kpoint and the\n"
                 " integration mesh (KMESH) and the quantity switches in &selfenergy.");

            std::ifstream ifs_ks;
            ifs_ks.open(ks_input.c_str(), std::ios::in);
            if (!ifs_ks) exit("setup_mode_analysis", "Cannot open file KS_INPUT");

            unsigned int nlist;
            double ktmp[3];
            unsigned int snum_tmp;

            ifs_ks >> nlist;

            if (nlist <= 0) exit("setup_mode_analysis", "First line in KS_INPUT files should be a positive integer.");

            {
                for (i = 0; i < nlist; ++i) {
                    ifs_ks >> ktmp[0] >> ktmp[1] >> ktmp[2] >> snum_tmp;
                    add_target(ktmp, snum_tmp, -1.0);
                }
                if (run.verbosity > 0) {
                    std::cout << " The number of entries = " << kslist.size() + kslist_offmesh.size() << '\n';
                }
            }

            ifs_ks.close();
        } else {
            ks_analyze_mode = false;
        }
    }

    MPI_Bcast(&ks_analyze_mode, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_realpart, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_fstate_omega, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_V3, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_V4, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_selfenergy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&spectral_func, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    unsigned int nlist;

    {
        NDArray<unsigned int, 1> kslist_arr;
        nlist = kslist.size();

        // Broadcast kslist

        MPI_Bcast(&nlist, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        kslist_arr.resize(nlist);

        if (run.my_rank == 0) {
            for (i = 0; i < nlist; ++i) kslist_arr[i] = kslist[i];
        }
        MPI_Bcast(&kslist_arr[0], nlist, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (run.my_rank > 0) {
            kslist.clear();
            for (i = 0; i < nlist; ++i) kslist.push_back(kslist_arr[i]);
        }
        kslist_arr.clear();

        unsigned int noff = kslist_offmesh.size();
        MPI_Bcast(&noff, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (run.my_rank > 0) kslist_offmesh.resize(noff);
        for (i = 0; i < noff; ++i) {
            MPI_Bcast(kslist_offmesh[i].xk, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&kslist_offmesh[i].snum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
    }

    if (ks_analyze_mode) {
        if (kpoint->kpoint_mode == 2 && anharmonic_core->use_triplet_symmetry) {
            anharmonic_core->disable_triplet_symmetry();
            if (run.my_rank == 0 && run.verbosity > 0) {
                std::cout << "\n TRISYM was automatically set to 0.\n\n";
            }
        }

        if (anharmonic_core->quartic_mode > 0) {
            // This is for quartic vertexes.

            if (run.my_rank == 0 && run.verbosity > 0) {
                std::cout << " QUARTIC = 1 : Frequency shift due to the loop diagram associated with\n";
                std::cout << "               quartic anharmonicity will be calculated.\n";
                std::cout << "               Please check the accuracy of the quartic IFCs \n";
                std::cout << "               before doing serious calculations.\n\n";
            }

            if (kpoint->kpoint_mode == 2 && anharmonic_core->use_quartet_symmetry) {
                anharmonic_core->disable_quartet_symmetry();
                if (run.my_rank == 0 && run.verbosity > 0) {
                    std::cout << "\n QUADRISYM was automatically set to 0.\n\n";
                }
            }
        }

        if (calc_realpart && integration->ismear != 0) {
            exit("setup_mode_analysis", "Sorry. REALPART = 1 can be used only with ISMEAR = 0");
        }

        if (spectral_func && integration->ismear != -1) {
            exit("setup_mode_analysis",
                 "Sorry. SELF_W = 1 can be used only with the tetrahedron method (ISMEAR = -1).");
        }

        dynamical->modify_eigenvectors(*dos->kmesh_dos, *dos->dymat_dos);
    }
}

void ModeAnalysis::run_mode_analysis()
{
    const auto Tmax = system->Tmax;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    NDArray<double, 1> T_arr;

    unsigned int NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;
    T_arr.resize(NT);
    for (unsigned int i = 0; i < NT; ++i) T_arr[i] = Tmin + static_cast<double>(i) * dT;

    {
        if (calc_selfenergy) print_selfenergy(NT, T_arr);

        if (print_V3 == 1) {
            print_V3_elements();
        } else if (print_V3 == 2) {
            print_Phi3_elements();
        }

        if (print_V4 == 1) {
            print_V4_elements();
        } else if (print_V4 == 2) {
            print_Phi4_elements();
        }

        if (calc_fstate_omega) print_frequency_resolved_final_state(NT, T_arr);

        if (spectral_func) print_spectral_function(NT, T_arr);

        if (selfenergy_mode && interpolate) run_interpolated_spectrum(NT, T_arr);

        if (run.my_rank == 0 && selfenergy_mode && run.use_hdf5_io) write_results_hdf5(NT, T_arr);
    }

    T_arr.clear();
}

void ModeAnalysis::run_interpolated_spectrum(const unsigned int NT, const double *T_arr)
{
    // INTERPOLATE: Pi(q, omega) = -2 E W Sigma W E^+ on the coarse mesh (bubble_matrix),
    // Fourier interpolated to the target q, A(q, omega) = -(2 omega / pi) Im Tr G with
    // G = (omega^2 - D - Pi)^-1; conventions in SELFENERGY_MODE_PLAN.md.
    const auto ns = dynamical->neval;
    const auto kmesh = dos->kmesh_dos.get();
    for (auto i = 0; i < 3; ++i) {
        if (kmesh_coarse[i] == 0 || kmesh->nk_i[i] % kmesh_coarse[i] != 0) {
            exit("run_interpolated_spectrum", "KMESH_COARSE must be given and divide KMESH.");
        }
    }
    const unsigned int nk_c = kmesh_coarse[0] * kmesh_coarse[1] * kmesh_coarse[2];

    // Coarse points in the DFT order of fourier_dymat_k_to_r; all are dense-mesh points.
    std::vector<int> knum_c(nk_c);
    for (unsigned int ic = 0; ic < nk_c; ++ic) {
        const unsigned int i1 = ic / (kmesh_coarse[1] * kmesh_coarse[2]);
        const unsigned int i2 = (ic / kmesh_coarse[2]) % kmesh_coarse[1];
        const unsigned int i3 = ic % kmesh_coarse[2];
        double xk[3] = {static_cast<double>(i1) / kmesh_coarse[0],
                        static_cast<double>(i2) / kmesh_coarse[1],
                        static_cast<double>(i3) / kmesh_coarse[2]};
        knum_c[ic] = kmesh->get_knum(xk);
        if (knum_c[ic] < 0) exit("run_interpolated_spectrum", "A coarse-mesh point is not on KMESH.");
    }
    NDArray<MinimumDistList, 3> mindist_list;
    system->get_minimum_distances(kmesh_coarse, mindist_list);

    // Frequency grid: OMEGA_RANGE, or the SELF_W grid (0 .. 2 omega_max in steps of DELTA_E).
    double omega_min = 0.0, delta_omega = dos->delta_e;
    unsigned int nomega;
    if (omega_range[2] > 0.0) {
        omega_min = omega_range[0];
        delta_omega = omega_range[2];
        nomega = static_cast<unsigned int>((omega_range[1] - omega_range[0]) / delta_omega) + 1;
    } else {
        double emax_now = 0.0;
        for (auto ik = 0; ik < kmesh->nk_irred; ++ik)
            for (unsigned int is = 0; is < ns; ++is)
                emax_now =
                    std::max(emax_now,
                             in_kayser(dos->dymat_dos->get_eigenvalues()[kmesh->kpoint_irred_all[ik][0].knum][is]));
        nomega = static_cast<unsigned int>((emax_now + delta_omega) * 2.0 / delta_omega) + 1;
    }
    std::vector<double> omega_ry(nomega);
    spectrum_omega.resize(nomega);
    for (unsigned int io = 0; io < nomega; ++io) {
        spectrum_omega[io] = omega_min + delta_omega * io;
        omega_ry[io] = spectrum_omega[io] * time_ry / Hz_to_kayser;
    }

    // Distinct target q in input order (branches collapse; a path revisiting a q keeps both occurrences).
    spectrum_xk.clear();
    spectrum_kaxis.clear();
    for (const auto &r: results) {
        bool seen = false;
        for (size_t iq = 0; iq < spectrum_xk.size(); ++iq) {
            const auto &x = spectrum_xk[iq];
            if (std::abs(x[0] - r.xk[0]) + std::abs(x[1] - r.xk[1]) + std::abs(x[2] - r.xk[2]) < 1.0e-10 &&
                std::abs(spectrum_kaxis[iq] - r.kaxis) < 1.0e-10)
                seen = true;
        }
        if (seen) continue;
        spectrum_xk.push_back({r.xk[0], r.xk[1], r.xk[2]});
        spectrum_kaxis.push_back(r.kaxis);
    }
    unsigned int nq = spectrum_xk.size();
    MPI_Bcast(&nq, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (run.my_rank > 0) {
        spectrum_xk.assign(nq, std::vector<double>(3, 0.0));
        spectrum_kaxis.assign(nq, -1.0);
    }
    for (unsigned int iq = 0; iq < nq; ++iq) MPI_Bcast(spectrum_xk[iq].data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << "\n INTERPOLATE = 1: bubble self-energy matrix on the " << kmesh_coarse[0] << "x"
                  << kmesh_coarse[1] << "x" << kmesh_coarse[2] << " coarse mesh, spectral function on " << nq
                  << " target q points.\n";
    }

    // Harmonic D(q) and eigenpairs at the targets (rank 0).
    std::vector<Eigen::MatrixXcd> dmat_q(nq), evec_q(nq);
    std::vector<Eigen::VectorXd> omega_q(nq);
    if (run.my_rank == 0) {
        NDArray<double, 2> eval_tmp(1, ns);
        NDArray<std::complex<double>, 3> evec_tmp(1, ns, ns);
        for (unsigned int iq = 0; iq < nq; ++iq) {
            eigen_at(spectrum_xk[iq].data(), eval_tmp, evec_tmp);
            evec_q[iq].resize(ns, ns);
            omega_q[iq].resize(ns);
            for (unsigned int j = 0; j < ns; ++j) {
                omega_q[iq](j) = eval_tmp[0][j];
                for (unsigned int a = 0; a < ns; ++a) evec_q[iq](a, j) = evec_tmp[0][j][a];
            }
            Eigen::VectorXd w2 = omega_q[iq].array().abs() * omega_q[iq].array(); // omega^2 with the sign of omega
            dmat_q[iq] = evec_q[iq] * w2.asDiagonal() * evec_q[iq].adjoint();
        }
    }

    spectrum_total.assign(NT, std::vector<std::vector<double>>(nq, std::vector<double>(nomega, 0.0)));
    spectrum_branch.assign(NT,
                           std::vector<std::vector<std::vector<double>>>(
                               nq,
                               std::vector<std::vector<double>>(ns, std::vector<double>(nomega, 0.0))));

    NDArray<std::complex<double>, 3> sig(nomega, ns, ns);
    std::vector<NDArray<std::complex<double>, 3>> pi_k(nomega), pi_r(nomega);
    for (unsigned int io = 0; io < nomega; ++io) {
        pi_k[io].resize(ns, ns, nk_c);
        pi_r[io].resize(ns, ns, nk_c);
    }
    NDArray<std::complex<double>, 2> pi_q(ns, ns);

    for (unsigned int iT = 0; iT < NT; ++iT) {
        for (unsigned int ic = 0; ic < nk_c; ++ic) {
            selfenergy->bubble_matrix(T_arr[iT],
                                      knum_c[ic],
                                      kmesh,
                                      dos->dymat_dos->get_eigenvalues(),
                                      dos->dymat_dos->get_eigenvectors(),
                                      nomega,
                                      omega_ry.data(),
                                      sig);
            if (run.my_rank != 0) continue;
            // Pi = -2 E W Sigma W E^+
            Eigen::MatrixXcd E(ns, ns);
            Eigen::VectorXd W(ns);
            for (unsigned int j = 0; j < ns; ++j) {
                const double w = dos->dymat_dos->get_eigenvalues()[knum_c[ic]][j];
                W(j) = w < eps8 ? 0.0 : std::sqrt(w);
                for (unsigned int a = 0; a < ns; ++a) E(a, j) = dos->dymat_dos->get_eigenvectors()[knum_c[ic]][j][a];
            }
            Eigen::MatrixXcd S(ns, ns);
            for (unsigned int io = 0; io < nomega; ++io) {
                for (unsigned int j = 0; j < ns; ++j)
                    for (unsigned int jp = 0; jp < ns; ++jp) S(j, jp) = sig[io][j][jp];
                const Eigen::MatrixXcd P = -2.0 * E * W.asDiagonal() * S * W.asDiagonal() * E.adjoint();
                for (unsigned int a = 0; a < ns; ++a)
                    for (unsigned int b = 0; b < ns; ++b) pi_k[io][a][b][ic] = P(a, b);
            }
        }
        if (run.my_rank != 0) continue;
        for (unsigned int io = 0; io < nomega; ++io) {
            fourier_dymat_k_to_r(kmesh_coarse[0], kmesh_coarse[1], kmesh_coarse[2], ns, pi_k[io], pi_r[io]);
        }
        for (unsigned int iq = 0; iq < nq; ++iq) {
            Eigen::MatrixXcd P(ns, ns), G(ns, ns);
            for (unsigned int io = 0; io < nomega; ++io) {
                r2q(spectrum_xk[iq].data(),
                    kmesh_coarse[0],
                    kmesh_coarse[1],
                    kmesh_coarse[2],
                    ns,
                    mindist_list,
                    pi_r[io],
                    pi_q);
                for (unsigned int a = 0; a < ns; ++a)
                    for (unsigned int b = 0; b < ns; ++b) P(a, b) = pi_q[a][b];
                const double w = omega_ry[io];
                G = (w * w * Eigen::MatrixXcd::Identity(ns, ns) - dmat_q[iq] - P).inverse();
                const double pref = -2.0 * w / pi;
                spectrum_total[iT][iq][io] = pref * G.trace().imag();
                // Branch projections averaged over each degenerate block (basis independent).
                for (unsigned int j = 0; j < ns;) {
                    const auto block = degenerate_block(omega_q[iq].data(), ns, j);
                    double gsum = 0.0;
                    for (int m = block.first; m < block.second; ++m) {
                        const std::complex<double> gmm = evec_q[iq].col(m).adjoint() * G * evec_q[iq].col(m);
                        gsum += gmm.imag();
                    }
                    for (int m = block.first; m < block.second; ++m)
                        spectrum_branch[iT][iq][m][io] = pref * gsum / (block.second - block.first);
                    j = block.second;
                }
            }
        }
    }
    // Spectral functions in 1/cm^-1 (Ry^-1 grid -> per cm^-1).
    const double to_kayser_inv = time_ry / Hz_to_kayser;
    for (auto &a: spectrum_total)
        for (auto &b: a)
            for (auto &v: b) v *= to_kayser_inv;
    for (auto &a: spectrum_branch)
        for (auto &b: a)
            for (auto &c: b)
                for (auto &v: c) v *= to_kayser_inv;

    if (run.my_rank == 0 && write_text()) {
        const auto file = run.job_title + ".spectrum";
        std::ofstream ofs(file);
        if (!ofs) exit("run_interpolated_spectrum", "Cannot open the spectrum file");
        ofs << "## Interpolated spectral function A(q, omega) [1/cm^-1] from the bubble self-energy matrix\n";
        ofs << "## T[K], q index, kaxis, omega (cm^-1), A_total, A_1 ... A_" << ns << '\n';
        for (unsigned int iT = 0; iT < NT; ++iT) {
            for (unsigned int iq = 0; iq < nq; ++iq) {
                for (unsigned int io = 0; io < nomega; ++io) {
                    ofs << std::setw(10) << T_arr[iT] << std::setw(6) << iq + 1 << std::setw(15) << std::setprecision(6)
                        << std::fixed << spectrum_kaxis[iq] << std::setw(12) << spectrum_omega[io];
                    ofs << std::scientific << std::setw(15) << spectrum_total[iT][iq][io];
                    for (unsigned int j = 0; j < ns; ++j) ofs << std::setw(15) << spectrum_branch[iT][iq][j][io];
                    ofs << '\n';
                }
                ofs << '\n';
            }
        }
        ofs << std::defaultfloat;
        if (run.verbosity > 0) std::cout << "  Spectral functions are printed in " << file << '\n';
    }
}

bool ModeAnalysis::write_text() const
{
    return !(selfenergy_mode && run.use_hdf5_io);
}

void ModeAnalysis::write_results_hdf5(const unsigned int NT, const double *T_arr) const
{
    using namespace H5Easy;
    const auto filename = run.job_title + ".selfenergy.h5";
    HighFive::File fh(filename, HighFive::File::Overwrite);
    write_input_variables_h5(fh, run.input_variables);

    dump(fh, "/metadata/version", 1);
    dump(fh, "/metadata/mode", std::string("selfenergy"));
    dump(fh, "/metadata/temperatures", std::vector<double>(T_arr, T_arr + NT));
    dumpAttribute(fh, "/metadata/temperatures", "unit", std::string("K"));
    dump(fh, "/metadata/ismear", integration->ismear);
    dump(fh, "/metadata/smearing_width", in_kayser(integration->epsilon));
    dumpAttribute(fh, "/metadata/smearing_width", "unit", std::string("cm^-1"));
    dump(fh, "/metadata/kmesh", std::vector<unsigned int>(dos->kmesh_dos->nk_i, dos->kmesh_dos->nk_i + 3));
    dump(fh, "/metadata/target_mode", kpoint->target_mode);
    dump(fh, "/metadata/number_of_targets", results.size());
    if (kpoint->target_mode == 1) {
        const auto &bs = *kpoint->kpoint_bs;
        std::vector<double> kaxis(bs.nk);
        std::vector<std::vector<double>> xk(bs.nk, std::vector<double>(3));
        for (unsigned int ik = 0; ik < bs.nk; ++ik) {
            kaxis[ik] = bs.kaxis[ik];
            for (auto j = 0; j < 3; ++j) xk[ik][j] = bs.xk[ik][j];
        }
        dump(fh, "/path/kaxis", kaxis);
        dumpAttribute(fh,
                      "/path/kaxis",
                      "description",
                      std::string("path coordinate, same convention as PREFIX.bands"));
        dump(fh, "/path/xk", xk);
    }

    for (size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        std::ostringstream name;
        name << "/targets/" << std::setw(5) << std::setfill('0') << i + 1;
        const auto g = name.str();
        dump(fh, g + "/xk", std::vector<double>(r.xk, r.xk + 3));
        dumpAttribute(fh, g + "/xk", "description", std::string("fractional coordinates in the reciprocal basis"));
        dump(fh, g + "/branch", r.branch);
        dump(fh, g + "/on_mesh", static_cast<int>(r.on_mesh));
        if (r.kaxis >= 0.0) dump(fh, g + "/kaxis", r.kaxis);
        dump(fh, g + "/frequency", r.omega);
        dumpAttribute(fh, g + "/frequency", "unit", std::string("cm^-1"));
        if (!r.linewidth.empty()) {
            dump(fh, g + "/linewidth", r.linewidth);
            dumpAttribute(fh, g + "/linewidth", "unit", std::string("cm^-1"));
            dumpAttribute(fh,
                          g + "/linewidth",
                          "description",
                          std::string("2*Gamma (FWHM) of the bubble diagram per temperature"));
        }
        if (!r.shift_bubble.empty()) {
            dump(fh, g + "/shift_tadpole", r.shift_tadpole);
            dump(fh, g + "/shift_bubble", r.shift_bubble);
            dumpAttribute(fh, g + "/shift_bubble", "unit", std::string("cm^-1"));
            if (!r.shift_loop.empty()) dump(fh, g + "/shift_loop", r.shift_loop);
        }
        if (!r.self_omega.empty()) {
            dump(fh, g + "/self_omega", r.self_omega);
            dumpAttribute(fh, g + "/self_omega", "unit", std::string("cm^-1"));
            dump(fh, g + "/self_real", r.self_real);
            dump(fh, g + "/self_imag", r.self_imag);
            dumpAttribute(fh, g + "/self_imag", "unit", std::string("cm^-1"));
            dumpAttribute(fh, g + "/self_imag", "layout", std::string("[temperature][omega]"));
        }
        if (!r.fstate_energy.empty()) {
            dump(fh, g + "/fstate_energy", r.fstate_energy);
            dumpAttribute(fh, g + "/fstate_energy", "unit", std::string("cm^-1"));
            dump(fh, g + "/fstate_absorption", r.fstate_absorption);
            dump(fh, g + "/fstate_emission", r.fstate_emission);
            dumpAttribute(fh, g + "/fstate_absorption", "layout", std::string("[temperature][energy]"));
        }
    }
    if (!spectrum_omega.empty()) {
        dump(fh, "/spectrum/omega", spectrum_omega);
        dumpAttribute(fh, "/spectrum/omega", "unit", std::string("cm^-1"));
        dump(fh, "/spectrum/xk", spectrum_xk);
        dump(fh, "/spectrum/kaxis", spectrum_kaxis);
        dump(fh, "/spectrum/kmesh_coarse", std::vector<unsigned int>(kmesh_coarse, kmesh_coarse + 3));
        for (unsigned int iT = 0; iT < NT; ++iT) {
            std::ostringstream g;
            g << "/spectrum/T" << std::setw(3) << std::setfill('0') << iT + 1;
            dump(fh, g.str() + "/temperature", T_arr[iT]);
            dump(fh, g.str() + "/total", spectrum_total[iT]);
            dumpAttribute(fh, g.str() + "/total", "layout", std::string("[q][omega], unit 1/cm^-1"));
            for (size_t j = 0; j < spectrum_branch[iT][0].size(); ++j) {
                std::vector<std::vector<double>> a(spectrum_branch[iT].size());
                for (size_t iq = 0; iq < a.size(); ++iq) a[iq] = spectrum_branch[iT][iq][j];
                std::ostringstream d;
                d << g.str() << "/branch" << std::setw(3) << std::setfill('0') << j + 1;
                dump(fh, d.str(), a);
            }
        }
    }
    if (run.verbosity > 0) {
        std::cout << "\n Mode-analysis results are written to " << filename << '\n';
    }
}

void ModeAnalysis::print_selfenergy(const unsigned int NT, double *T_arr)
{
    auto ns = dynamical->neval;
    int j;

    std::ofstream ofs_linewidth, ofs_shift;

    NDArray<double, 1> damping_a;
    NDArray<std::complex<double>, 1> self_tadpole;
    NDArray<std::complex<double>, 1> self_a;
    NDArray<std::complex<double>, 1> self_b;
    NDArray<std::complex<double>, 1> self_c;
    NDArray<std::complex<double>, 1> self_d;
    NDArray<std::complex<double>, 1> self_e;
    NDArray<std::complex<double>, 1> self_f;
    NDArray<std::complex<double>, 1> self_g;
    NDArray<std::complex<double>, 1> self_h;
    NDArray<std::complex<double>, 1> self_i;
    NDArray<std::complex<double>, 1> self_j;

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << "\n Calculate the line width (FWHM) of phonons\n";
        std::cout << " due to 3-phonon interactions for given " << kslist.size() << " modes.\n";

        if (calc_realpart) {
            if (anharmonic_core->quartic_mode == 1) {
                std::cout << " REALPART = 1 and \n";
                std::cout << " QUARTIC  = 1     : Additionally, frequency shift of phonons due to 3-phonon\n";
                std::cout << "                    and 4-phonon interactions will be calculated.\n";
            } else {
                std::cout << " REALPART = 1 : Additionally, frequency shift of phonons due to 3-phonon\n";
                std::cout << "                interactions will be calculated.\n";
            }
        }

        if (anharmonic_core->quartic_mode == 2) {
            std::cout << '\n';
            std::cout << " QUARTIC = 2 : Additionally, phonon line width due to 4-phonon\n";
            std::cout << "               interactions will be calculated.\n";
            std::cout << " WARNING     : This is very very expensive.\n";
        }
    }

    damping_a.resize(NT);
    self_a.resize(NT);
    self_b.resize(NT);
    self_tadpole.resize(NT);

    if (anharmonic_core->quartic_mode == 2) {
        self_c.resize(NT);
        self_d.resize(NT);
        self_e.resize(NT);
        self_f.resize(NT);
        self_g.resize(NT);
        self_h.resize(NT);
        self_i.resize(NT);
        self_j.resize(NT);
    }

    for (int i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << "\n Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];
        // Degenerate targets: average over the whole block (gauge independent). Without
        // REALPART the smearing kernel of MODE = kappa is used (it honours ISMEAR); the
        // complex-frequency kernel selfenergy_a, needed for the bubble shift, is Lorentzian.
        const auto eval_k = dos->dymat_dos->get_eigenvalues()[knum];
        const auto block = degenerate_block(eval_k, ns, snum);
        const double nblock = block.second - block.first;
        NDArray<double, 1> damping_tmp(NT);
        NDArray<std::complex<double>, 1> self_tmp(NT);
        for (j = 0; j < NT; ++j) {
            damping_a[j] = 0.0;
            self_a[j] = 0.0;
        }
        for (auto s = block.first; s < block.second; ++s) {
            if (integration->ismear == -1) {
                anharmonic_core->calc_damping_tetrahedron(NT,
                                                          T_arr,
                                                          eval_k[s],
                                                          ik_irred,
                                                          s,
                                                          dos->kmesh_dos.get(),
                                                          dos->dymat_dos->get_eigenvalues(),
                                                          dos->dymat_dos->get_eigenvectors(),
                                                          damping_tmp);
            } else if (!calc_realpart) {
                anharmonic_core->calc_damping_smearing(NT,
                                                       T_arr,
                                                       eval_k[s],
                                                       ik_irred,
                                                       s,
                                                       dos->kmesh_dos.get(),
                                                       dos->dymat_dos->get_eigenvalues(),
                                                       dos->dymat_dos->get_eigenvectors(),
                                                       damping_tmp);
            } else {
                selfenergy->selfenergy_a(NT,
                                         T_arr,
                                         eval_k[s],
                                         knum,
                                         s,
                                         dos->kmesh_dos.get(),
                                         dos->dymat_dos->get_eigenvalues(),
                                         dos->dymat_dos->get_eigenvectors(),
                                         self_tmp);
                for (j = 0; j < NT; ++j) {
                    self_a[j] += self_tmp[j] / nblock;
                    damping_tmp[j] = self_tmp[j].imag();
                }
            }
            for (j = 0; j < NT; ++j) damping_a[j] += damping_tmp[j] / nblock;
        }
        if (anharmonic_core->quartic_mode == 2) {
            selfenergy->selfenergy_c(NT,
                                     T_arr,
                                     omega,
                                     knum,
                                     snum,
                                     dos->kmesh_dos.get(),
                                     dos->dymat_dos->get_eigenvalues(),
                                     dos->dymat_dos->get_eigenvectors(),
                                     self_c);
            //            selfenergy->selfenergy_d(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_d);
            //            selfenergy->selfenergy_e(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_e);
            //            selfenergy->selfenergy_f(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_f);
            //            selfenergy->selfenergy_g(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_g);
            //            selfenergy->selfenergy_h(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_h);
            //            selfenergy->selfenergy_i(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_i);
            //            selfenergy->selfenergy_j(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos.get(),
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_j);
        }

        if (run.my_rank == 0) {
            auto &r = results[kslist_id[i]];
            r.omega = in_kayser(omega);
            r.linewidth.resize(NT);
            for (j = 0; j < NT; ++j) r.linewidth[j] = in_kayser(2.0 * damping_a[j]);
        }
        if (run.my_rank == 0 && write_text()) {
            auto file_linewidth = run.job_title + ".Gamma." + std::to_string(i + 1);
            ofs_linewidth.open(file_linewidth.c_str(), std::ios::out);
            if (!ofs_linewidth) exit("print_selfenergy", "Cannot open file file_linewidth");

            ofs_linewidth << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_linewidth << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_linewidth << '\n';
            ofs_linewidth << "# mode = " << snum + 1 << '\n';
            ofs_linewidth << "# Frequency = " << in_kayser(omega) << '\n';
            ofs_linewidth << "## Temperature dependence of 2*Gamma (FWHM) for the given mode\n";
            ofs_linewidth << "## T[K], 2*Gamma3 (cm^-1) (bubble)";
            if (anharmonic_core->quartic_mode == 2) ofs_linewidth << ", 2*Gamma4(cm^-1) <-- specific diagram only";
            ofs_linewidth << '\n';

            for (j = 0; j < NT; ++j) {
                ofs_linewidth << std::setw(10) << T_arr[j] << std::setw(15) << in_kayser(2.0 * damping_a[j]);

                if (anharmonic_core->quartic_mode == 2) {
                    //							ofs_mode_tau << std::setw(15) << in_kayser(damp4[j]);
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_c[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_d[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_e[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_f[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_g[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_h[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_i[j].imag());
                    ofs_linewidth << std::setw(15) << in_kayser(2.0 * self_j[j].imag());
                }

                ofs_linewidth << '\n';
            }
            ofs_linewidth.close();
            if (run.verbosity > 0) {
                std::cout << "  Phonon line-width is printed in " << file_linewidth << '\n';
            }
        }

        if (calc_realpart) {
            for (j = 0; j < NT; ++j) {
                self_tadpole[j] = 0.0;
                self_b[j] = 0.0;
            }
            for (auto s = block.first; s < block.second; ++s) {
                selfenergy->selfenergy_tadpole(NT,
                                               T_arr,
                                               eval_k[s],
                                               knum,
                                               s,
                                               dos->kmesh_dos.get(),
                                               dos->dymat_dos->get_eigenvalues(),
                                               dos->dymat_dos->get_eigenvectors(),
                                               self_tmp);
                for (j = 0; j < NT; ++j) self_tadpole[j] += self_tmp[j] / nblock;
                if (anharmonic_core->quartic_mode == 1) {
                    selfenergy->selfenergy_b(NT,
                                             T_arr,
                                             eval_k[s],
                                             knum,
                                             s,
                                             dos->kmesh_dos.get(),
                                             dos->dymat_dos->get_eigenvalues(),
                                             dos->dymat_dos->get_eigenvectors(),
                                             self_tmp);
                    for (j = 0; j < NT; ++j) self_b[j] += self_tmp[j] / nblock;
                }
            }
            //                selfenergy->selfenergy_a(NT, T_arr, omega, knum, snum, self_a);

            if (run.my_rank == 0) {
                auto &r = results[kslist_id[i]];
                r.shift_tadpole.resize(NT);
                r.shift_bubble.resize(NT);
                if (anharmonic_core->quartic_mode == 1) r.shift_loop.resize(NT);
                for (j = 0; j < NT; ++j) {
                    r.shift_tadpole[j] = in_kayser(-self_tadpole[j].real());
                    r.shift_bubble[j] = in_kayser(-self_a[j].real());
                    if (anharmonic_core->quartic_mode == 1) r.shift_loop[j] = in_kayser(-self_b[j].real());
                }
            }
            if (run.my_rank == 0 && write_text()) {
                auto file_shift = run.job_title + ".Shift." + std::to_string(i + 1);
                ofs_shift.open(file_shift.c_str(), std::ios::out);
                if (!ofs_shift) exit("print_selfenergy", "Cannot open file file_shift");

                ofs_shift << "# xk = ";

                for (j = 0; j < 3; ++j) {
                    ofs_shift << std::setw(15) << dos->kmesh_dos->xk[knum][j];
                }
                ofs_shift << '\n';
                ofs_shift << "# mode = " << snum + 1 << '\n';
                ofs_shift << "# Frequency = " << in_kayser(omega) << '\n';
                ofs_shift << "## T[K], Shift3 (cm^-1) (tadpole), Shift3 (cm^-1) (bubble)";
                if (anharmonic_core->quartic_mode == 1) ofs_shift << ", Shift4 (cm^-1) (loop)";
                ofs_shift << ", Shifted frequency (cm^-1)\n";

                for (j = 0; j < NT; ++j) {
                    ofs_shift << std::setw(10) << T_arr[j];
                    ofs_shift << std::setw(15) << in_kayser(-self_tadpole[j].real());
                    ofs_shift << std::setw(15) << in_kayser(-self_a[j].real());

                    auto omega_shift = omega - self_tadpole[j].real() - self_a[j].real();

                    if (anharmonic_core->quartic_mode == 1) {
                        ofs_shift << std::setw(15) << in_kayser(-self_b[j].real());
                        omega_shift -= self_b[j].real();
                    }
                    ofs_shift << std::setw(15) << in_kayser(omega_shift);
                    ofs_shift << '\n';
                }

                ofs_shift.close();
                if (run.verbosity > 0) {
                    std::cout << "  Phonon frequency shift is printed in " << file_shift << '\n';
                }
            }
        }
    }

    if (damping_a) {
        damping_a.clear();
    }
    if (self_tadpole) {
        self_tadpole.clear();
    }
    if (self_a) {
        self_a.clear();
    }
    if (self_b) {
        self_b.clear();
    }
    if (self_c) {
        self_c.clear();
    }
    if (self_d) {
        self_d.clear();
    }
    if (self_e) {
        self_e.clear();
    }
    if (self_f) {
        self_f.clear();
    }
    if (self_g) {
        self_g.clear();
    }
    if (self_h) {
        self_h.clear();
    }
    if (self_i) {
        self_i.clear();
    }
    if (self_j) {
        self_j.clear();
    }

    print_selfenergy_offmesh(NT, T_arr, kslist.size());
}

namespace
{
bool same_target_k(const std::vector<PHON_NS::ModeAnalysis::OffMeshTarget> &list, const size_t i)
{
    if (i == 0) return false;
    for (auto j = 0; j < 3; ++j) {
        if (std::abs(list[i].xk[j] - list[i - 1].xk[j]) > 1.0e-12) return false;
    }
    return true;
}
} // namespace

void ModeAnalysis::print_selfenergy_offmesh(const unsigned int NT, const double *T_arr, const size_t number_offset)
{
    // Off-mesh text files are numbered after the on-mesh targets.
    if (kslist_offmesh.empty()) return;
    const auto ns = dynamical->neval;
    NDArray<double, 2> xq(1, 3), eval_q(1, ns);
    NDArray<std::complex<double>, 3> evec_q(1, ns, ns);
    NDArray<double, 1> damping(NT);
    AnharmonicCore::ShiftedGrid sg;

    for (size_t i = 0; i < kslist_offmesh.size(); ++i) {
        const auto &target = kslist_offmesh[i];
        const auto snum = target.snum;
        for (auto j = 0; j < 3; ++j) xq[0][j] = target.xk[j];
        if (!same_target_k(kslist_offmesh, i)) {
            eigen_at(xq[0], eval_q, evec_q);
            anharmonic_core->build_shifted_grid(xq[0], dos->kmesh_dos.get(), sg);
        }
        const auto omega = eval_q[0][snum];

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << "\n Number : " << std::setw(5) << number_offset + i + 1 << " (off the k-point grid)\n";
            std::cout << "  Phonon at k = (";
            for (auto j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << xq[0][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        // Degenerate targets: average over the whole block (gauge independent).
        const auto block = degenerate_block(eval_q[0], ns, snum);
        const double nblock = block.second - block.first;
        NDArray<double, 1> damping_tmp(NT);
        NDArray<std::complex<double>, 1> self_tmp(NT), self_a(NT), self_tadpole(NT), self_b(NT);
        for (unsigned int j = 0; j < NT; ++j) {
            damping[j] = 0.0;
            self_a[j] = self_tadpole[j] = self_b[j] = 0.0;
        }
        for (auto s = block.first; s < block.second; ++s) {
            if (calc_realpart) {
                // Bubble with the complex frequency (real and imaginary parts), tadpole, loop.
                selfenergy->selfenergy_a_at(NT,
                                            T_arr,
                                            eval_q[0][s],
                                            xq[0],
                                            eval_q[0][s],
                                            evec_q[0][s],
                                            dos->kmesh_dos.get(),
                                            dos->dymat_dos->get_eigenvalues(),
                                            dos->dymat_dos->get_eigenvectors(),
                                            sg,
                                            self_tmp);
                for (unsigned int j = 0; j < NT; ++j) {
                    self_a[j] += self_tmp[j] / nblock;
                    damping_tmp[j] = self_tmp[j].imag();
                }
                selfenergy->selfenergy_tadpole_at(NT,
                                                  T_arr,
                                                  xq[0],
                                                  eval_q[0][s],
                                                  evec_q[0][s],
                                                  dos->kmesh_dos.get(),
                                                  dos->dymat_dos->get_eigenvalues(),
                                                  dos->dymat_dos->get_eigenvectors(),
                                                  self_tmp);
                for (unsigned int j = 0; j < NT; ++j) self_tadpole[j] += self_tmp[j] / nblock;
                if (anharmonic_core->quartic_mode == 1) {
                    selfenergy->selfenergy_b_at(NT,
                                                T_arr,
                                                xq[0],
                                                eval_q[0][s],
                                                evec_q[0][s],
                                                dos->kmesh_dos.get(),
                                                dos->dymat_dos->get_eigenvalues(),
                                                dos->dymat_dos->get_eigenvectors(),
                                                self_tmp);
                    for (unsigned int j = 0; j < NT; ++j) self_b[j] += self_tmp[j] / nblock;
                }
            } else if (integration->ismear == -1) {
                anharmonic_core->calc_damping_tetrahedron_at(NT,
                                                             T_arr,
                                                             eval_q[0][s],
                                                             xq[0],
                                                             eval_q[0][s],
                                                             evec_q[0][s],
                                                             dos->kmesh_dos.get(),
                                                             dos->dymat_dos->get_eigenvalues(),
                                                             dos->dymat_dos->get_eigenvectors(),
                                                             sg,
                                                             damping_tmp);
            } else {
                anharmonic_core->calc_damping_smearing_at(NT,
                                                          T_arr,
                                                          eval_q[0][s],
                                                          xq[0],
                                                          eval_q[0][s],
                                                          evec_q[0][s],
                                                          dos->kmesh_dos.get(),
                                                          dos->dymat_dos->get_eigenvalues(),
                                                          dos->dymat_dos->get_eigenvectors(),
                                                          sg,
                                                          damping_tmp);
            }
            for (unsigned int j = 0; j < NT; ++j) damping[j] += damping_tmp[j] / nblock;
        }

        if (run.my_rank == 0) {
            auto &r = results[target.id];
            r.omega = in_kayser(omega);
            r.linewidth.resize(NT);
            for (unsigned int j = 0; j < NT; ++j) r.linewidth[j] = in_kayser(2.0 * damping[j]);
        }
        if (run.my_rank == 0 && write_text()) {
            const auto file_linewidth = run.job_title + ".Gamma." + std::to_string(number_offset + i + 1);
            std::ofstream ofs(file_linewidth);
            if (!ofs) exit("print_selfenergy_offmesh", "Cannot open file file_linewidth");
            ofs << "# xk = ";
            for (auto j = 0; j < 3; ++j) ofs << std::setw(15) << xq[0][j];
            ofs << '\n';
            ofs << "# mode = " << snum + 1 << '\n';
            ofs << "# Frequency = " << in_kayser(omega) << '\n';
            ofs << "## Temperature dependence of 2*Gamma (FWHM) for the given mode\n";
            ofs << "## T[K], 2*Gamma3 (cm^-1) (bubble)\n";
            for (unsigned int j = 0; j < NT; ++j) {
                ofs << std::setw(10) << T_arr[j] << std::setw(15) << in_kayser(2.0 * damping[j]) << '\n';
            }
            if (run.verbosity > 0) {
                std::cout << "  Phonon line-width is printed in " << file_linewidth << '\n';
            }
        }
        if (calc_realpart && run.my_rank == 0) {
            auto &r = results[target.id];
            r.shift_tadpole.resize(NT);
            r.shift_bubble.resize(NT);
            if (anharmonic_core->quartic_mode == 1) r.shift_loop.resize(NT);
            for (unsigned int j = 0; j < NT; ++j) {
                r.shift_tadpole[j] = in_kayser(-self_tadpole[j].real());
                r.shift_bubble[j] = in_kayser(-self_a[j].real());
                if (anharmonic_core->quartic_mode == 1) r.shift_loop[j] = in_kayser(-self_b[j].real());
            }
            if (write_text()) {
                const auto file_shift = run.job_title + ".Shift." + std::to_string(number_offset + i + 1);
                std::ofstream ofs(file_shift);
                if (!ofs) exit("print_selfenergy_offmesh", "Cannot open file file_shift");
                ofs << "# xk = ";
                for (auto j = 0; j < 3; ++j) ofs << std::setw(15) << xq[0][j];
                ofs << '\n';
                ofs << "# mode = " << snum + 1 << '\n';
                ofs << "# Frequency = " << in_kayser(omega) << '\n';
                ofs << "## T[K], Shift3 (cm^-1) (tadpole), Shift3 (cm^-1) (bubble)";
                if (anharmonic_core->quartic_mode == 1) ofs << ", Shift4 (cm^-1) (loop)";
                ofs << ", Shifted frequency (cm^-1)\n";
                for (unsigned int j = 0; j < NT; ++j) {
                    ofs << std::setw(10) << T_arr[j];
                    ofs << std::setw(15) << in_kayser(-self_tadpole[j].real());
                    ofs << std::setw(15) << in_kayser(-self_a[j].real());
                    auto omega_shift = omega - self_tadpole[j].real() - self_a[j].real();
                    if (anharmonic_core->quartic_mode == 1) {
                        ofs << std::setw(15) << in_kayser(-self_b[j].real());
                        omega_shift -= self_b[j].real();
                    }
                    ofs << std::setw(15) << in_kayser(omega_shift) << '\n';
                }
                if (run.verbosity > 0)
                    std::cout << "  Phonon frequency shift is printed in " << file_shift << '\n';
            }
        }
    }
}

void ModeAnalysis::eigen_at(const double *xk, NDArray<double, 2> &eval, NDArray<std::complex<double>, 3> &evec) const
{
    NDArray<double, 2> xq(1, 3), kvec(1, 3);
    double xf[3];
    for (auto j = 0; j < 3; ++j) {
        xq[0][j] = xk[j];
        xf[j] = xk[j] - std::floor(xk[j] + 0.5); // same non-analytic direction convention as the mesh
    }
    rotvec(kvec[0], xf, system->get_primcell().reciprocal_lattice_vector, 'T');
    const auto norm = kvec[0][0] * kvec[0][0] + kvec[0][1] * kvec[0][1] + kvec[0][2] * kvec[0][2];
    for (auto j = 0; j < 3; ++j) kvec[0][j] = norm > eps ? kvec[0][j] / std::sqrt(norm) : 0.0;
    dynamical->get_eigenvalues_dymat(1,
                                     xq,
                                     kvec,
                                     fcs_phonon->force_constant_with_cell[0],
                                     ewald->fc2_without_dipole,
                                     true,
                                     eval,
                                     evec);
}

void ModeAnalysis::print_vertex_offmesh(const int kind, const size_t number_offset) const
{
    // |V3|^2 (kind 0) and Phi3 (1): partners q' over the mesh, q'' = q - q' (0) or -q - q' (1).
    // |V4|^2 (2) and Phi4 (3): (q1, q2) over the mesh, q3 = q - q1 - q2 (2) or -q - q1 - q2 (3).
    // q3 is a mesh point shifted by q, so its eigenpair is the shifted-grid entry of q1 + q2.
    if (kslist_offmesh.empty()) return;
    const auto ns = dynamical->neval;
    const int nk = dos->kmesh_dos->nk;
    const auto &xk = dos->kmesh_dos->xk;
    const auto eval = dos->dymat_dos->get_eigenvalues();
    const auto evec = dos->dymat_dos->get_eigenvectors();
    const bool squared = (kind == 0 || kind == 2);
    const bool quartic = (kind >= 2);
    const double sign = (kind == 0 || kind == 2) ? 1.0 : -1.0; // external leg -q (squared) or +q (Phi)
    const double factor = kind == 0   ? std::pow(0.5, 3) * pow2(Hz_to_kayser / time_ry)
                          : kind == 2 ? std::pow(0.5, 4) * pow2(Hz_to_kayser / time_ry)
                          : kind == 1 ? std::pow(amu_ry, 1.5)
                                      : pow2(amu_ry);
    const char *tag = kind == 0 ? "V3" : kind == 1 ? "Phi3" : kind == 2 ? "V4" : "Phi4";

    NDArray<double, 2> xq(1, 3), eval_q(1, ns);
    NDArray<std::complex<double>, 3> evec_q(1, ns, ns);
    AnharmonicCore::ShiftedGrid sg;
    const int ngroup = anharmonic_core->get_ngroup_fcs(quartic ? 4 : 3);

    for (size_t i = 0; i < kslist_offmesh.size(); ++i) {
        const auto &target = kslist_offmesh[i];
        const auto snum = target.snum;
        for (auto j = 0; j < 3; ++j) xq[0][j] = target.xk[j];
        if (!same_target_k(kslist_offmesh, i)) {
            eigen_at(xq[0], eval_q, evec_q);
            // Grid of the last internal leg: {sign*q - k}.
            double xs[3];
            for (auto j = 0; j < 3; ++j) xs[j] = sign * xq[0][j];
            anharmonic_core->build_shifted_grid(xs, dos->kmesh_dos.get(), sg);
        }
        const auto omega_q = eval_q[0][snum];
        if (run.my_rank == 0) results[target.id].omega = in_kayser(omega_q);
        // External leg: e(-q) = e(q)^* for the squared listings, e(q) for Phi.
        std::vector<std::complex<double>> e0(ns);
        for (auto a = 0; a < ns; ++a) e0[a] = squared ? std::conj(evec_q[0][snum][a]) : evec_q[0][snum][a];

        const int npair = quartic ? nk * nk : nk;
        const size_t nb = quartic ? static_cast<size_t>(ns) * ns * ns : static_cast<size_t>(ns) * ns;
        NDArray<std::complex<double>, 2> val_loc(npair, nb), val(npair, nb);
        for (int ip = 0; ip < npair; ++ip)
            for (size_t ib = 0; ib < nb; ++ib) val_loc[ip][ib] = std::complex<double>(0.0, 0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<std::complex<double>> work(ngroup);
            double xk3[3];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
            for (int ip = 0; ip < npair; ++ip) {
                if (ip % run.nprocs != run.my_rank) continue;
                const int k1 = quartic ? ip / nk : ip;
                const int k2 = quartic ? ip % nk : -1;
                if (!quartic) {
                    anharmonic_core->phi3_reciprocal_at(xk[k1], sg.xk[k1], work.data());
                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            const auto w1 = eval[k1][is], w2 = sg.eval[k1][js];
                            auto v =
                                anharmonic_core->contract_phi3(e0.data(), evec[k1][is], sg.evec[k1][js], work.data());
                            if (squared) {
                                v = (omega_q < eps8 || w1 < eps8 || w2 < eps8) ? 0.0
                                                                               : std::norm(v) / (omega_q * w1 * w2);
                            }
                            val_loc[ip][is * ns + js] = v * factor;
                        }
                    }
                } else {
                    double xsum[3];
                    for (auto j = 0; j < 3; ++j) {
                        xsum[j] = xk[k1][j] + xk[k2][j];
                        xk3[j] = sign * xq[0][j] - xsum[j];
                    }
                    const int k3 = dos->kmesh_dos->get_knum(xsum); // sg entry of q1 + q2 is at sign*q - (q1 + q2) + G
                    anharmonic_core->phi4_reciprocal_at(xk[k1], xk[k2], xk3, work.data());
                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            for (auto ks = 0; ks < ns; ++ks) {
                                const auto w1 = eval[k1][is], w2 = eval[k2][js], w3 = sg.eval[k3][ks];
                                auto v = anharmonic_core->contract_phi4(e0.data(),
                                                                        evec[k1][is],
                                                                        evec[k2][js],
                                                                        sg.evec[k3][ks],
                                                                        work.data());
                                if (squared) {
                                    v = (omega_q < eps8 || w1 < eps8 || w2 < eps8 || w3 < eps8)
                                            ? 0.0
                                            : std::norm(v) / (omega_q * w1 * w2 * w3);
                                }
                                val_loc[ip][(is * ns + js) * ns + ks] = v * factor;
                            }
                        }
                    }
                }
            }
        }
        MPI_Reduce(&val_loc[0][0], &val[0][0], npair * nb, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

        if (run.my_rank == 0) {
            const auto file = run.job_title + "." + tag + "." + std::to_string(number_offset + i + 1);
            std::ofstream ofs(file);
            if (!ofs) exit("print_vertex_offmesh", "Cannot open the output file");
            ofs << "# xk = ";
            for (auto j = 0; j < 3; ++j) ofs << std::setw(15) << xq[0][j];
            ofs << '\n';
            ofs << "# mode = " << snum + 1 << '\n';
            ofs << "# Frequency = " << in_kayser(omega_q) << '\n';
            ofs << "## Matrix elements "
                << (kind == 0   ? "|V3|^2"
                    : kind == 1 ? "Phi3"
                    : kind == 2 ? "|V4|^2"
                                : "Phi4")
                << " for given mode (k point off the mesh: partners keyed by fractional coordinates, "
                   "every partner listed once)\n";
            if (!quartic) {
                ofs << "## q'(3), j', omega(q'j') (cm^-1), q''(3), j'', omega(q''j'') (cm^-1), "
                    << (kind == 0 ? "|V3(-qj,q'j',q''j'')|^2 (cm^-2)" : "Re Phi3, Im Phi3 (Ry/(u^1/2 Bohr)^3)")
                    << ", multiplicity\n";
            } else {
                ofs << "## q1(3), j1, omega(q1j1) (cm^-1), q2(3), j2, omega(q2j2) (cm^-1), q3(3), j3, omega(q3j3) (cm^-1), "
                    << (kind == 2 ? "|V4(-qj,q1j1,q2j2,q3j3)|^2 (cm^-2)" : "Re Phi4, Im Phi4 (Ry/(u^1/2 Bohr)^4)")
                    << ", multiplicity\n";
            }
            auto put_leg = [&](const double *x, const int js, const double w) {
                for (auto j = 0; j < 3; ++j) ofs << std::setw(12) << std::fixed << std::setprecision(6) << x[j];
                ofs << std::setw(5) << js + 1 << std::setw(15) << std::setprecision(6) << in_kayser(w);
            };
            auto put_val = [&](const std::complex<double> &v) {
                ofs << std::scientific << std::setprecision(6);
                if (squared) ofs << std::setw(15) << v.real();
                else
                    ofs << std::setw(15) << v.real() << std::setw(15) << v.imag();
                ofs << std::setw(5) << 1 << '\n';
            };
            for (int ip = 0; ip < npair; ++ip) {
                const int k1 = quartic ? ip / nk : ip;
                const int k2 = quartic ? ip % nk : -1;
                if (!quartic) {
                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            put_leg(xk[k1], is, eval[k1][is]);
                            put_leg(sg.xk[k1], js, sg.eval[k1][js]);
                            put_val(val[ip][is * ns + js]);
                        }
                        ofs << '\n';
                    }
                } else {
                    double xsum[3], xk3[3];
                    for (auto j = 0; j < 3; ++j) {
                        xsum[j] = xk[k1][j] + xk[k2][j];
                        xk3[j] = sign * xq[0][j] - xsum[j];
                    }
                    const int k3 = dos->kmesh_dos->get_knum(xsum);
                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            for (auto ks = 0; ks < ns; ++ks) {
                                put_leg(xk[k1], is, eval[k1][is]);
                                put_leg(xk[k2], js, eval[k2][js]);
                                put_leg(xk3, ks, sg.eval[k3][ks]);
                                put_val(val[ip][(is * ns + js) * ns + ks]);
                            }
                            ofs << '\n';
                        }
                    }
                }
            }
            ofs << std::defaultfloat;
            if (run.verbosity > 0) std::cout << "  Matrix elements are printed in " << file << '\n';
        }
    }
}

void ModeAnalysis::print_frequency_resolved_final_state(const unsigned int NT, double *T_arr)
{
    int i, j;
    NDArray<double, 3> gamma_final;
    NDArray<double, 1> freq_array;
    std::ofstream ofs_omega;
    const auto ns = dynamical->neval;

    gamma_final.resize(NT, dos->n_energy, 2);
    freq_array.resize(dos->n_energy);

    for (i = 0; i < dos->n_energy; ++i) {
        freq_array[i] = dos->energy_dos[i] * time_ry / Hz_to_kayser;
    }

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << '\n';
        std::cout << " FSTATE_W = 1 : Calculate the frequency-resolved final state amplitude\n";
        std::cout << "                due to 3-phonon interactions.\n";
    }

    for (i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega0 = dos->dymat_dos->get_eigenvalues()[knum][snum];

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega0) << '\n';
        }

        if (integration->ismear == -1) {
            calc_frequency_resolved_final_state_tetrahedron(NT,
                                                            T_arr,
                                                            omega0,
                                                            dos->n_energy,
                                                            freq_array,
                                                            dos->kmesh_dos->kmap_to_irreducible[knum],
                                                            snum,
                                                            dos->kmesh_dos.get(),
                                                            dos->dymat_dos->get_eigenvalues(),
                                                            dos->dymat_dos->get_eigenvectors(),
                                                            gamma_final);
        } else {
            calc_frequency_resolved_final_state(NT,
                                                T_arr,
                                                omega0,
                                                dos->n_energy,
                                                freq_array,
                                                dos->kmesh_dos->kmap_to_irreducible[knum],
                                                snum,
                                                dos->kmesh_dos.get(),
                                                dos->dymat_dos->get_eigenvalues(),
                                                dos->dymat_dos->get_eigenvectors(),
                                                gamma_final);
        }

        if (run.my_rank == 0) {
            auto &r = results[kslist_id[i]];
            r.omega = in_kayser(omega0);
            r.fstate_energy = dos->energy_dos;
            r.fstate_absorption.assign(NT, std::vector<double>(dos->n_energy));
            r.fstate_emission.assign(NT, std::vector<double>(dos->n_energy));
            for (j = 0; j < NT; ++j) {
                for (int ie = 0; ie < dos->n_energy; ++ie) {
                    r.fstate_absorption[j][ie] = gamma_final[j][ie][1];
                    r.fstate_emission[j][ie] = gamma_final[j][ie][0];
                }
            }
        }
        if (run.my_rank == 0 && write_text()) {
            std::string file_omega = run.job_title + ".fw." + std::to_string(i + 1);
            ofs_omega.open(file_omega.c_str(), std::ios::out);
            if (!ofs_omega) exit("print_frequency_resolved_final_state", "Cannot open file file_omega");

            ofs_omega << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_omega << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_omega << '\n';
            ofs_omega << "# mode = " << snum + 1 << '\n';
            ofs_omega << "# Frequency = " << in_kayser(omega0) << '\n';

            ofs_omega << "## Frequency-resolved final state amplitude for given modes\n";
            ofs_omega << "## Gamma[omega][temperature] (absorption, emission)\n";
            ofs_omega << "## ";
            for (j = 0; j < NT; ++j) {
                ofs_omega << std::setw(10) << T_arr[j];
            }
            ofs_omega << '\n';
            for (int ienergy = 0; ienergy < dos->n_energy; ++ienergy) {
                const auto omega = dos->energy_dos[ienergy];

                ofs_omega << std::setw(10) << omega;
                for (j = 0; j < NT; ++j) {
                    ofs_omega << std::setw(15) << gamma_final[j][ienergy][1];
                    ofs_omega << std::setw(15) << gamma_final[j][ienergy][0];
                }
                ofs_omega << '\n';
            }
            ofs_omega.close();
            if (run.verbosity > 0) {
                std::cout << "  Frequency-resolved final state amplitude is printed in " << file_omega << '\n';
            }
        }
    }

    print_frequency_resolved_final_state_offmesh(NT, T_arr, kslist.size(), freq_array);
    freq_array.clear();
    gamma_final.clear();
}

void ModeAnalysis::calc_frequency_resolved_final_state_offmesh(const unsigned int ntemp, const double *temperature,
                                                               const unsigned int nomegas, const double *omega,
                                                               const double *xq, const double omega_q,
                                                               const std::complex<double> *evec_q,
                                                               const AnharmonicCore::ShiftedGrid &sg,
                                                               double ***ret) const
{
    // ret[T][omega][0/1]: emission/absorption, resolved by the mesh phonon's frequency.
    // The partner is q - k. Tetrahedron energy conservation uses Gaussian frequency
    // resolution of width EPSILON, as in the on-mesh routines.
    const auto kmesh_in = dos->kmesh_dos.get();
    const auto eval_in = dos->dymat_dos->get_eigenvalues();
    const auto evec_in = dos->dymat_dos->get_eigenvectors();
    const int nk = kmesh_in->nk;
    const int ns = dynamical->neval;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    const auto epsilon = integration->epsilon;
    const int ismear = integration->ismear;
    const bool classical = thermodynamics->classical;

    NDArray<double, 3> ret_loc(ntemp, nomegas, 2);
    for (unsigned int i = 0; i < ntemp; ++i)
        for (unsigned int j = 0; j < nomegas; ++j) ret_loc[i][j][0] = ret_loc[i][j][1] = 0.0;

    // Energy-conservation weights delta[k][ib][0/1] (sum / difference channel).
    NDArray<double, 3> delta_arr(nk, ns2, 2);
    if (ismear == -1) {
        NDArray<unsigned int, 1> kmap_identity(nk);
        for (auto i = 0; i < nk; ++i) kmap_identity[i] = i;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            NDArray<double, 2> energy_tmp(3, nk), weight_tetra(3, nk);
#ifdef _OPENMP
#pragma omp for
#endif
            for (int ib = 0; ib < static_cast<int>(ns2); ++ib) {
                const int is = ib / ns, js = ib % ns;
                for (auto k = 0; k < nk; ++k) {
                    energy_tmp[0][k] = eval_in[k][is] + sg.eval[k][js];
                    energy_tmp[1][k] = eval_in[k][is] - sg.eval[k][js];
                    energy_tmp[2][k] = -energy_tmp[1][k];
                }
                for (auto i = 0; i < 3; ++i) {
                    integration->calc_weight_tetrahedron(nk,
                                                         kmap_identity,
                                                         energy_tmp[i],
                                                         omega_q,
                                                         dos->tetra_nodes_dos->get_ntetra(),
                                                         dos->tetra_nodes_dos->get_tetras(),
                                                         weight_tetra[i]);
                }
                for (auto k = 0; k < nk; ++k) {
                    delta_arr[k][ib][0] = weight_tetra[0][k];
                    delta_arr[k][ib][1] = weight_tetra[1][k] - weight_tetra[2][k];
                }
            }
        }
    } else {
        for (auto k = 0; k < nk; ++k) {
            AnharmonicCore::bubble_delta_smearing(ns,
                                                  omega_q,
                                                  eval_in[k],
                                                  sg.eval[k],
                                                  ismear,
                                                  epsilon,
                                                  nullptr,
                                                  nullptr,
                                                  0.0,
                                                  &delta_arr[k][0][0]);
        }
    }

    std::vector<std::complex<double>> e0(ns);
    for (auto a = 0; a < ns; ++a) e0[a] = std::conj(evec_q[a]);
    const int ngroup = anharmonic_core->get_ngroup_fcs(3);
    std::vector<std::complex<double>> work(ngroup);
    std::vector<double> v3sq(ns2);

    for (int k = run.my_rank; k < nk; k += run.nprocs) {
        anharmonic_core->phi3_reciprocal_at(kmesh_in->xk[k], sg.xk[k], work.data());
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                const auto w1 = eval_in[k][is], w2 = sg.eval[k][js];
                v3sq[is * ns + js] =
                    (omega_q < eps8 || w1 < eps8 || w2 < eps8)
                        ? 0.0
                        : std::norm(anharmonic_core
                                        ->contract_phi3(e0.data(), evec_in[k][is], sg.evec[k][js], work.data(), true)) /
                              (omega_q * w1 * w2);
            }
        }
        for (auto is = 0; is < ns; ++is) {
            const auto w1 = eval_in[k][is];
            for (auto js = 0; js < ns; ++js) {
                const auto w2 = sg.eval[k][js];
                const auto v = v3sq[is * ns + js];
                if (v <= eps) continue;
                const double d0 = delta_arr[k][is * ns + js][0], d1 = delta_arr[k][is * ns + js][1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int i = 0; i < static_cast<int>(ntemp); ++i) {
                    const auto f1 =
                        classical ? Thermodynamics::fC(w1, temperature[i]) : Thermodynamics::fB(w1, temperature[i]);
                    const auto f2 =
                        classical ? Thermodynamics::fC(w2, temperature[i]) : Thermodynamics::fB(w2, temperature[i]);
                    const auto n1 = classical ? f1 + f2 : f1 + f2 + 1.0;
                    const auto n2 = f1 - f2;
                    const auto p0 = v * n1 * d0;
                    const auto p1 = -v * n2 * d1;
                    for (unsigned int j = 0; j < nomegas; ++j) {
                        // Frequency resolution: Lorentzian/Gaussian for ISMEAR 0/1, Gaussian with the tetrahedron.
                        const auto r =
                            ismear == 0 ? delta_lorentz(omega[j] - w1, epsilon) : delta_gauss(omega[j] - w1, epsilon);
                        ret_loc[i][j][0] += p0 * r;
                        ret_loc[i][j][1] += p1 * r;
                    }
                }
            }
        }
    }
    const double factor = pi * std::pow(0.5, 4) / (ismear == -1 ? 1.0 : static_cast<double>(nk));
    for (unsigned int i = 0; i < ntemp; ++i)
        for (unsigned int j = 0; j < nomegas; ++j) {
            ret_loc[i][j][0] *= factor;
            ret_loc[i][j][1] *= factor;
        }
    MPI_Reduce(&ret_loc[0][0][0], &ret[0][0][0], 2 * ntemp * nomegas, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void ModeAnalysis::print_frequency_resolved_final_state_offmesh(const unsigned int NT, double *T_arr,
                                                                const size_t number_offset, const double *freq_array)
{
    if (kslist_offmesh.empty()) return;
    const auto ns = dynamical->neval;
    NDArray<double, 2> xq(1, 3), eval_q(1, ns);
    NDArray<std::complex<double>, 3> evec_q(1, ns, ns);
    NDArray<double, 3> gamma_final(NT, dos->n_energy, 2);
    AnharmonicCore::ShiftedGrid sg;

    for (size_t i = 0; i < kslist_offmesh.size(); ++i) {
        const auto &target = kslist_offmesh[i];
        const auto snum = target.snum;
        for (auto j = 0; j < 3; ++j) xq[0][j] = target.xk[j];
        if (!same_target_k(kslist_offmesh, i)) {
            eigen_at(xq[0], eval_q, evec_q);
            anharmonic_core->build_shifted_grid(xq[0], dos->kmesh_dos.get(), sg);
        }
        const auto omega0 = eval_q[0][snum];
        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << "\n Number : " << std::setw(5) << number_offset + i + 1 << " (off the k-point grid)\n";
            std::cout << "  Phonon at k = (";
            for (auto j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << xq[0][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega0) << '\n';
        }
        calc_frequency_resolved_final_state_offmesh(NT,
                                                    T_arr,
                                                    dos->n_energy,
                                                    freq_array,
                                                    xq[0],
                                                    omega0,
                                                    evec_q[0][snum],
                                                    sg,
                                                    gamma_final);
        if (run.my_rank == 0) {
            auto &r = results[target.id];
            r.omega = in_kayser(omega0);
            r.fstate_energy = dos->energy_dos;
            r.fstate_absorption.assign(NT, std::vector<double>(dos->n_energy));
            r.fstate_emission.assign(NT, std::vector<double>(dos->n_energy));
            for (unsigned int j = 0; j < NT; ++j) {
                for (int ie = 0; ie < dos->n_energy; ++ie) {
                    r.fstate_absorption[j][ie] = gamma_final[j][ie][1];
                    r.fstate_emission[j][ie] = gamma_final[j][ie][0];
                }
            }
        }
        if (run.my_rank == 0 && write_text()) {
            const auto file_omega = run.job_title + ".fw." + std::to_string(number_offset + i + 1);
            std::ofstream ofs_omega(file_omega);
            if (!ofs_omega) exit("print_frequency_resolved_final_state_offmesh", "Cannot open file file_omega");
            ofs_omega << "# xk = ";
            for (auto j = 0; j < 3; ++j) ofs_omega << std::setw(15) << xq[0][j];
            ofs_omega << '\n';
            ofs_omega << "# mode = " << snum + 1 << '\n';
            ofs_omega << "# Frequency = " << in_kayser(omega0) << '\n';
            ofs_omega << "## Frequency-resolved final state amplitude for given modes\n";
            ofs_omega << "## Gamma[omega][temperature] (absorption, emission)\n";
            ofs_omega << "## ";
            for (unsigned int j = 0; j < NT; ++j) ofs_omega << std::setw(10) << T_arr[j];
            ofs_omega << '\n';
            for (int ienergy = 0; ienergy < dos->n_energy; ++ienergy) {
                ofs_omega << std::setw(10) << dos->energy_dos[ienergy];
                for (unsigned int j = 0; j < NT; ++j) {
                    ofs_omega << std::setw(15) << gamma_final[j][ienergy][1];
                    ofs_omega << std::setw(15) << gamma_final[j][ienergy][0];
                }
                ofs_omega << '\n';
            }
            if (run.verbosity > 0) {
                std::cout << "  Frequency-resolved final state amplitude is printed in " << file_omega << '\n';
            }
        }
    }
}

void ModeAnalysis::calc_frequency_resolved_final_state(
    const unsigned int ntemp, const double *temperature, const double omega0, const unsigned int nomegas,
    const double *omega, const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in,
    const double *const *eval_in, const std::complex<double> *const *const *evec_in, double ***ret) const
{
    int i, j;

    unsigned int arr[3];
    double omega_inner[2];
    double n1, n2;
    double f1, f2;
    double prod_tmp[2];
    NDArray<double, 3> ret_mpi;
    const auto nk = kmesh_in->nk;
    const auto ns = dynamical->neval;

    const auto epsilon = integration->epsilon;

    std::vector<KsListGroup> triplet;

    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, anharmonic_core->use_triplet_symmetry, false, triplet);

    ret_mpi.resize(ntemp, nomegas, 2);

    for (i = 0; i < ntemp; ++i) {
        for (j = 0; j < nomegas; ++j) {
            ret_mpi[i][j][0] = 0.0;
            ret_mpi[i][j][1] = 0.0;
        }
    }

    for (int ik = run.my_rank; ik < triplet.size(); ik += run.nprocs) {
        const auto multi = static_cast<double>(triplet[ik].group.size());
        const auto knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
        const auto knum_minus = kmesh_in->kindex_minus_xk[knum];

        arr[0] = ns * knum_minus + is_in;
        const auto k1 = triplet[ik].group[0].ks[0];
        const auto k2 = triplet[ik].group[0].ks[1];

        for (int is = 0; is < ns; ++is) {
            for (int js = 0; js < ns; ++js) {
                arr[1] = ns * k1 + is;
                arr[2] = ns * k2 + js;

                omega_inner[0] = eval_in[k1][is];
                omega_inner[1] = eval_in[k2][js];

                const auto v3_tmp = std::norm(anharmonic_core->V3(arr));

                for (i = 0; i < ntemp; ++i) {
                    const auto T_tmp = temperature[i];

                    if (thermodynamics->classical) {
                        f1 = thermodynamics->fC(omega_inner[0], T_tmp);
                        f2 = thermodynamics->fC(omega_inner[1], T_tmp);
                        n1 = f1 + f2;
                        n2 = f1 - f2;
                    } else {
                        f1 = thermodynamics->fB(omega_inner[0], T_tmp);
                        f2 = thermodynamics->fB(omega_inner[1], T_tmp);
                        n1 = f1 + f2 + 1.0;
                        n2 = f1 - f2;
                    }

                    if (integration->ismear == 0) {
                        prod_tmp[0] = n1 * (delta_lorentz(omega0 - omega_inner[0] - omega_inner[1], epsilon) -
                                            delta_lorentz(omega0 + omega_inner[0] + omega_inner[1], epsilon));
                        prod_tmp[1] = n2 * (delta_lorentz(omega0 + omega_inner[0] - omega_inner[1], epsilon) -
                                            delta_lorentz(omega0 - omega_inner[0] + omega_inner[1], epsilon));

                        for (j = 0; j < nomegas; ++j) {
                            ret_mpi[i][j][0] +=
                                v3_tmp * multi * delta_lorentz(omega[j] - omega_inner[0], epsilon) * prod_tmp[0];
                            ret_mpi[i][j][1] +=
                                v3_tmp * multi * delta_lorentz(omega[j] - omega_inner[0], epsilon) * prod_tmp[1];
                        }
                    } else if (integration->ismear == 1) {
                        prod_tmp[0] = n1 * (delta_gauss(omega0 - omega_inner[0] - omega_inner[1], epsilon) -
                                            delta_gauss(omega0 + omega_inner[0] + omega_inner[1], epsilon));
                        prod_tmp[1] = n2 * (delta_gauss(omega0 + omega_inner[0] - omega_inner[1], epsilon) -
                                            delta_gauss(omega0 - omega_inner[0] + omega_inner[1], epsilon));

                        for (j = 0; j < nomegas; ++j) {
                            ret_mpi[i][j][0] +=
                                v3_tmp * multi * delta_gauss(omega[j] - omega_inner[0], epsilon) * prod_tmp[0];
                            ret_mpi[i][j][1] +=
                                v3_tmp * multi * delta_gauss(omega[j] - omega_inner[0], epsilon) * prod_tmp[1];
                        }
                    }
                }
            }
        }
    }
    for (i = 0; i < ntemp; ++i) {
        for (j = 0; j < nomegas; ++j) {
            ret_mpi[i][j][0] *= pi * std::pow(0.5, 4) / static_cast<double>(nk);
            ret_mpi[i][j][1] *= pi * std::pow(0.5, 4) / static_cast<double>(nk);
        }
    }

    MPI_Reduce(&ret_mpi[0][0][0], &ret[0][0][0], 2 * ntemp * nomegas, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    ret_mpi.clear();
    triplet.clear();
}

void ModeAnalysis::calc_frequency_resolved_final_state_tetrahedron(
    const unsigned int ntemp, double *temperature, const double omega0, const unsigned int nomegas, const double *omega,
    const unsigned int ik_in, const unsigned int is_in, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
    const std::complex<double> *const *const *evec_in, double ***ret) const
{
    int i, j;
    int ik;
    unsigned int jk;

    unsigned int is, js;
    unsigned int k1, k2;
    unsigned int arr[3];

    double omega_inner[2];
    double n1, n2;
    double f1, f2;
    double xk_tmp[3];
    double v3_tmp;
    const auto nk = kmesh_in->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;

    NDArray<unsigned int, 1> kmap_identity;
    NDArray<double, 2> energy_tmp;
    NDArray<double, 2> weight_tetra;
    NDArray<double, 2> v3_arr;
    NDArray<double, 3> delta_arr;
    double prod_tmp[2];

    const auto epsilon = integration->epsilon;

    const auto &xk = kmesh_in->xk;

    std::vector<KsListGroup> triplet;

    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, anharmonic_core->use_triplet_symmetry, false, triplet);

    for (i = 0; i < ntemp; ++i) {
        for (j = 0; j < nomegas; ++j) {
            ret[i][j][0] = 0.0;
            ret[i][j][1] = 0.0;
        }
    }

    const auto npair_uniq = triplet.size();

    v3_arr.resize(npair_uniq, ns2);
    delta_arr.resize(npair_uniq, ns2, 2);

    const auto knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const auto knum_minus = kmesh_in->kindex_minus_xk[knum];

    kmap_identity.resize(nk);

    for (i = 0; i < nk; ++i) kmap_identity[i] = i;

#ifdef _OPENMP
#pragma omp parallel private(is, js, k1, k2, xk_tmp, energy_tmp, i, weight_tetra, ik, jk, arr)
#endif
    {
        energy_tmp.resize(3, nk);
        weight_tetra.resize(3, nk);
        // Thread-local FC3 cache: the V3 overload without work buffers is not thread-safe.
        std::vector<std::complex<double>> phi3_work(anharmonic_core->get_ngroup_fcs(3));
        int kindex_work[2] = {-1, -1};

#ifdef _OPENMP
#pragma omp for
#endif
        for (int ib = 0; ib < ns2; ++ib) {
            is = ib / ns;
            js = ib % ns;

            for (k1 = 0; k1 < nk; ++k1) {
                // Prepare two-phonon frequency for the tetrahedron method

                for (i = 0; i < 3; ++i) xk_tmp[i] = xk[knum][i] - xk[k1][i];

                k2 = kmesh_in->get_knum(xk_tmp);

                energy_tmp[0][k1] = eval_in[k1][is] + eval_in[k2][js];
                energy_tmp[1][k1] = eval_in[k1][is] - eval_in[k2][js];
                energy_tmp[2][k1] = -energy_tmp[1][k1];
            }

            for (i = 0; i < 3; ++i) {
                //                integration->calc_weight_tetrahedron(nk_3ph,
                //                                                     kmap_identity,
                //                                                     weight_tetra[i],
                //                                                     energy_tmp[i],
                //                                                     omega0);

                integration->calc_weight_tetrahedron(nk,
                                                     kmap_identity,
                                                     energy_tmp[i],
                                                     omega0,
                                                     dos->tetra_nodes_dos->get_ntetra(),
                                                     dos->tetra_nodes_dos->get_tetras(),
                                                     weight_tetra[i]);
            }

            // Loop for irreducible k points
            for (ik = 0; ik < npair_uniq; ++ik) {
                delta_arr[ik][ib][0] = 0.0;
                delta_arr[ik][ib][1] = 0.0;

                for (i = 0; i < triplet[ik].group.size(); ++i) {
                    jk = triplet[ik].group[i].ks[0];
                    delta_arr[ik][ib][0] += weight_tetra[0][jk];
                    delta_arr[ik][ib][1] += weight_tetra[1][jk] - weight_tetra[2][jk];
                }

                // Calculate the matrix element V3 only when the weight is nonzero.
                if (delta_arr[ik][ib][0] > 0.0 || std::abs(delta_arr[ik][ib][1]) > 0.0) {
                    k1 = triplet[ik].group[0].ks[0];
                    k2 = triplet[ik].group[0].ks[1];

                    arr[0] = ns * knum_minus + is_in;
                    arr[1] = ns * k1 + is;
                    arr[2] = ns * k2 + js;
                    v3_arr[ik][ib] = std::norm(
                        anharmonic_core->V3(arr, kmesh_in->xk, eval_in, evec_in, phi3_work.data(), kindex_work));
                } else {
                    v3_arr[ik][ib] = 0.0;
                }
            }
        }

        energy_tmp.clear();
        weight_tetra.clear();
    }

    for (ik = 0; ik < npair_uniq; ++ik) {
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                v3_tmp = v3_arr[ik][ns * is + js];

                if (v3_tmp > eps) {
                    k1 = triplet[ik].group[0].ks[0];
                    k2 = triplet[ik].group[0].ks[1];
                    omega_inner[0] = eval_in[k1][is];
                    omega_inner[1] = eval_in[k2][js];

#ifdef _OPENMP
#pragma omp parallel for private(f1, f2, n1, n2, prod_tmp, j)
#endif
                    for (i = 0; i < ntemp; ++i) {
                        if (thermodynamics->classical) {
                            f1 = thermodynamics->fC(omega_inner[0], temperature[i]);
                            f2 = thermodynamics->fC(omega_inner[1], temperature[i]);

                            n1 = f1 + f2;
                            n2 = f1 - f2;
                        } else {
                            f1 = thermodynamics->fB(omega_inner[0], temperature[i]);
                            f2 = thermodynamics->fB(omega_inner[1], temperature[i]);

                            n1 = f1 + f2 + 1.0;
                            n2 = f1 - f2;
                        }

                        prod_tmp[0] = v3_tmp * n1 * delta_arr[ik][ns * is + js][0];
                        prod_tmp[1] = -v3_tmp * n2 * delta_arr[ik][ns * is + js][1];

                        for (j = 0; j < nomegas; ++j) {
                            ret[i][j][0] += prod_tmp[0] * delta_gauss(omega[j] - omega_inner[0], epsilon);
                            ret[i][j][1] += prod_tmp[1] * delta_gauss(omega[j] - omega_inner[0], epsilon);
                        }
                    }
                }
            }
        }
    }

    for (i = 0; i < ntemp; ++i) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (j = 0; j < nomegas; ++j) {
            ret[i][j][0] *= pi * std::pow(0.5, 4);
            ret[i][j][1] *= pi * std::pow(0.5, 4);
        }
    }

    v3_arr.clear();
    delta_arr.clear();
    kmap_identity.clear();

    triplet.clear();
}

void ModeAnalysis::print_V3_elements() const
{
    int j;
    const auto ns = dynamical->neval;
    std::ofstream ofs_V3;

    std::vector<KsListGroup> triplet;
    const auto eval_tmp = dos->dymat_dos->get_eigenvalues();

    for (auto i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega = eval_tmp[knum][snum];

        if (run.my_rank == 0) results[kslist_id[i]].omega = in_kayser(omega);

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        // Enumerate at the requested q to match the external leg and file header.
        dos->kmesh_dos->get_triplets_at_k(knum, true, triplet);
        const auto nk_size = triplet.size();

        std::vector<std::vector<double>> v3norm(nk_size, std::vector<double>(ns * ns));

        calc_V3norm2(knum, snum, triplet, v3norm);

        if (run.my_rank == 0) {
            auto file_V3 = run.job_title + ".V3." + std::to_string(i + 1);
            ofs_V3.open(file_V3.c_str(), std::ios::out);
            if (!ofs_V3) exit("run_mode_analysis", "Cannot open file file_V3");

            ofs_V3 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V3 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V3 << '\n';
            ofs_V3 << "# mode = " << snum + 1 << '\n';
            ofs_V3 << "# Frequency = " << in_kayser(omega) << '\n';
            ofs_V3 << "## Matrix elements |V3|^2 for given mode\n";
            ofs_V3 << "## q', j', omega(q'j') (cm^-1), q'', j'', ";
            ofs_V3 << "omega(q''j'') (cm^-1), |V3(-qj,q'j',q''j'')|^2 (cm^-2), multiplicity\n";

            for (j = 0; j < nk_size; ++j) {
                const int multi = triplet[j].group.size();
                const unsigned int k1 = triplet[j].group[0].ks[0];
                const unsigned int k2 = triplet[j].group[0].ks[1];

                unsigned int ib = 0;

                for (unsigned int is = 0; is < ns; ++is) {
                    for (unsigned int js = 0; js < ns; ++js) {
                        ofs_V3 << std::setw(5) << k1 + 1 << std::setw(5) << is + 1;
                        ofs_V3 << std::setw(15) << in_kayser(eval_tmp[k1][is]);
                        ofs_V3 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                        ofs_V3 << std::setw(15) << in_kayser(eval_tmp[k2][js]);
                        ofs_V3 << std::setw(15) << v3norm[j][ib];
                        ofs_V3 << std::setw(5) << multi;
                        ofs_V3 << '\n';

                        ++ib;
                    }
                    ofs_V3 << '\n';
                }
            }

            ofs_V3.close();
        }
    }
    print_vertex_offmesh(0, kslist.size());
}

void ModeAnalysis::print_V4_elements() const
{
    int j;
    const auto ns = dynamical->neval;
    std::ofstream ofs_V4;

    std::vector<KsListGroup> quartet;
    const auto eval_tmp = dos->dymat_dos->get_eigenvalues();

    for (int i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        double omega = eval_tmp[knum][snum];

        if (run.my_rank == 0) results[kslist_id[i]].omega = in_kayser(omega);

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        // Enumerate at the requested q to conserve momentum with the -q external leg.
        dos->kmesh_dos->get_quartets_at_k(knum, true, quartet);
        const auto nk_size = quartet.size();

        std::vector<std::vector<double>> v4norm(nk_size, std::vector<double>(ns * ns * ns));

        calc_V4norm2(knum, snum, quartet, v4norm);

        if (run.my_rank == 0) {
            std::string file_V4 = run.job_title + ".V4." + std::to_string(i + 1);
            ofs_V4.open(file_V4.c_str(), std::ios::out);
            if (!ofs_V4) exit("run_mode_analysis", "Cannot open file file_V4");

            ofs_V4 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V4 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V4 << '\n';
            ofs_V4 << "# mode = " << snum + 1 << '\n';
            ofs_V4 << "# Frequency = " << in_kayser(omega) << '\n';
            ofs_V4 << "## Matrix elements |V4|^2 for given mode\n";
            ofs_V4 << "## q1, j1, omega(q1j1) (cm^-1), "
                      "q2, j2, omega(q2j2) (cm^-1), "
                      "q3, j3, omega(q3j3) (cm^-1), "
                      "|V4(-qj,q1j1,q2j2,q3j3)|^2 (cm^-2), multiplicity\n";

            for (j = 0; j < nk_size; ++j) {
                int multi = quartet[j].group.size();
                unsigned int k1 = quartet[j].group[0].ks[0];
                unsigned int k2 = quartet[j].group[0].ks[1];
                unsigned int k3 = quartet[j].group[0].ks[2];

                unsigned int ib = 0;

                for (unsigned int is = 0; is < ns; ++is) {
                    for (unsigned int js = 0; js < ns; ++js) {
                        for (unsigned int ks = 0; ks < ns; ++ks) {
                            ofs_V4 << std::setw(5) << k1 + 1 << std::setw(5) << is + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k1][is]);
                            ofs_V4 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k2][js]);
                            ofs_V4 << std::setw(5) << k3 + 1 << std::setw(5) << ks + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k3][ks]);
                            ofs_V4 << std::setw(15) << v4norm[j][ib];
                            ofs_V4 << std::setw(5) << multi;
                            ofs_V4 << '\n';

                            ++ib;
                        }
                        ofs_V4 << '\n';
                    }
                    ofs_V4 << '\n';
                }
            }
            ofs_V4.close();
        }
    }
    print_vertex_offmesh(2, kslist.size());
}

void ModeAnalysis::calc_V3norm2(const unsigned int knum, const unsigned int snum,
                                const std::vector<KsListGroup> &triplet, std::vector<std::vector<double>> &ret) const
{
    unsigned int is, js;
    unsigned int k1, k2;
    unsigned int arr[3];
    const auto ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const auto factor = std::pow(0.5, 3) * pow2(Hz_to_kayser / time_ry);

    const auto knum_minus = dos->kmesh_dos->kindex_minus_xk[knum];
    const auto ntriplet = triplet.size();

    NDArray<double, 2> ret_loc;
    NDArray<double, 2> ret_sum;

    ret_loc.resize(ntriplet, ns2);
    ret_sum.resize(ntriplet, ns2);

    for (size_t ik = 0; ik < ntriplet; ++ik) {
        for (size_t ib = 0; ib < ns2; ++ib) {
            ret_loc[ik][ib] = 0.0;
            ret_sum[ik][ib] = 0.0;
        }
    }

    for (size_t ik = 0; ik < triplet.size(); ++ik) {
        k1 = triplet[ik].group[0].ks[0];
        k2 = triplet[ik].group[0].ks[1];

        for (size_t ib = run.my_rank; ib < ns2; ib += run.nprocs) {
            is = ib / ns;
            js = ib % ns;

            arr[0] = ns * knum_minus + snum;
            arr[1] = ns * k1 + is;
            arr[2] = ns * k2 + js;

            ret_loc[ik][ib] = std::norm(anharmonic_core->V3(arr)) * factor;
        }
    }

    const size_t count = ntriplet * ns2;
    MPI_Reduce(&ret_loc[0][0], &ret_sum[0][0], count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (run.my_rank == 0) {
        for (size_t ik = 0; ik < ntriplet; ++ik) {
            for (size_t ib = 0; ib < ns2; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    ret_loc.clear();
    ret_sum.clear();
}

void ModeAnalysis::calc_V4norm2(const unsigned int knum, const unsigned int snum,
                                const std::vector<KsListGroup> &quartet, std::vector<std::vector<double>> &ret) const
{
    unsigned int is, js, ks;
    unsigned int k1, k2, k3;
    unsigned int arr[4];
    const auto ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns3 = ns2 * ns;

    const double factor = std::pow(0.5, 4) * pow2(Hz_to_kayser / time_ry);
    const auto nquartet = quartet.size();

    NDArray<double, 2> ret_loc;
    NDArray<double, 2> ret_sum;

    ret_loc.resize(nquartet, ns3);
    ret_sum.resize(nquartet, ns3);

    for (size_t ik = 0; ik < nquartet; ++ik) {
        for (size_t ib = 0; ib < ns3; ++ib) {
            ret_loc[ik][ib] = 0.0;
            ret_sum[ik][ib] = 0.0;
        }
    }

    unsigned int knum_minus = dos->kmesh_dos->kindex_minus_xk[knum];

    for (size_t ik = 0; ik < nquartet; ++ik) {
        k1 = quartet[ik].group[0].ks[0];
        k2 = quartet[ik].group[0].ks[1];
        k3 = quartet[ik].group[0].ks[2];

        for (size_t ib = run.my_rank; ib < ns3; ib += run.nprocs) {
            is = ib / ns2;
            js = (ib - is * ns2) / ns;
            ks = ib % ns;

            arr[0] = ns * knum_minus + snum;
            arr[1] = ns * k1 + is;
            arr[2] = ns * k2 + js;
            arr[3] = ns * k3 + ks;

            ret_loc[ik][ib] = std::norm(anharmonic_core->V4(arr)) * factor;
        }
    }

    const size_t count = nquartet * ns3;
    MPI_Reduce(&ret_loc[0][0], &ret_sum[0][0], count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (run.my_rank == 0) {
        for (size_t ik = 0; ik < nquartet; ++ik) {
            for (size_t ib = 0; ib < ns3; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    ret_loc.clear();
    ret_sum.clear();
}

void ModeAnalysis::print_Phi3_elements() const
{
    int j;
    const auto ns = dynamical->neval;
    std::ofstream ofs_V3;

    std::vector<KsListGroup> triplet;
    const auto eval_tmp = dos->dymat_dos->get_eigenvalues();

    for (auto i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega = eval_tmp[knum][snum];

        if (run.my_rank == 0) results[kslist_id[i]].omega = in_kayser(omega);

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        dos->kmesh_dos->get_triplets_at_k(knum, true, triplet, 1); // q + q' + q'' = G
        const auto nk_size = triplet.size();
        std::vector<std::vector<std::complex<double>>> phi3(nk_size, std::vector<std::complex<double>>(ns * ns));

        calc_Phi3(knum, snum, triplet, phi3);

        if (run.my_rank == 0) {
            auto file_V3 = run.job_title + ".Phi3." + std::to_string(i + 1);
            ofs_V3.open(file_V3.c_str(), std::ios::out);
            if (!ofs_V3) exit("print_phi3_element", "Cannot open file file_V3");

            ofs_V3 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V3 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V3 << '\n';
            ofs_V3 << "# mode = " << snum + 1 << '\n';
            ofs_V3 << "# Frequency = " << in_kayser(omega) << '\n';
            ofs_V3 << "## Matrix elements Phi3 for given mode\n";
            ofs_V3 << "## q', j', omega(q'j') (cm^-1), q'', j'', omega(q''j'') (cm^-1), ";
            ofs_V3 << "Phi3(qj,q'j',q''j'') (Ry/(u^{1/2}Bohr)^{3}), multiplicity\n";

            for (j = 0; j < nk_size; ++j) {
                const auto multi = triplet[j].group.size();
                const unsigned int k1 = triplet[j].group[0].ks[0];
                const unsigned int k2 = triplet[j].group[0].ks[1];

                unsigned int ib = 0;

                for (unsigned int is = 0; is < ns; ++is) {
                    for (unsigned int js = 0; js < ns; ++js) {
                        ofs_V3 << std::setw(5) << k1 + 1 << std::setw(5) << is + 1;
                        ofs_V3 << std::setw(15) << in_kayser(eval_tmp[k1][is]);
                        ofs_V3 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                        ofs_V3 << std::setw(15) << in_kayser(eval_tmp[k2][js]);
                        ofs_V3 << std::setw(15) << phi3[j][ib].real();
                        ofs_V3 << std::setw(15) << phi3[j][ib].imag();
                        ofs_V3 << std::setw(5) << multi;
                        ofs_V3 << '\n';

                        ++ib;
                    }
                    ofs_V3 << '\n';
                }
            }

            ofs_V3.close();
        }
    }
    print_vertex_offmesh(1, kslist.size());
}

void ModeAnalysis::print_Phi4_elements() const
{
    int j;
    auto ns = dynamical->neval;
    std::ofstream ofs_V4;
    std::vector<KsListGroup> quartet;
    const auto eval_tmp = dos->dymat_dos->get_eigenvalues();

    for (int i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega = eval_tmp[knum][snum];

        if (run.my_rank == 0) results[kslist_id[i]].omega = in_kayser(omega);

        if (run.my_rank == 0 && run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
        }

        dos->kmesh_dos->get_quartets_at_k(knum, true, quartet, 1); // q + q1 + q2 + q3 = G
        unsigned int nk_size = quartet.size();

        std::vector<std::vector<std::complex<double>>> phi4(nk_size, std::vector<std::complex<double>>(ns * ns * ns));
        calc_Phi4(knum, snum, quartet, phi4);

        if (run.my_rank == 0) {
            std::string file_V4 = run.job_title + ".Phi4." + std::to_string(i + 1);
            ofs_V4.open(file_V4.c_str(), std::ios::out);
            if (!ofs_V4) exit("print_phi4_element", "Cannot open file file_V3");

            ofs_V4 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V4 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V4 << '\n';
            ofs_V4 << "# mode = " << snum + 1 << '\n';
            ofs_V4 << "# Frequency = " << in_kayser(omega) << '\n';
            ofs_V4 << "# List of k-point coordinates\n";
            for (j = 0; j < dos->kmesh_dos->nk; ++j) {
                ofs_V4 << "# ik = " << std::setw(4) << j + 1;
                ofs_V4 << ": xk = ";
                for (auto k = 0; k < 3; ++k) {
                    ofs_V4 << std::setw(15) << dos->kmesh_dos->xk[j][k];
                }
                ofs_V4 << '\n';
            }
            ofs_V4 << "## Matrix elements Phi4 for given mode\n";
            ofs_V4 << "## q1, j1, omega(q1j1) (cm^-1), "
                      "q2, j2, omega(q2j2) (cm^-1), "
                      "q3, j3, omega(q3j3) (cm^-1), "
                      "Phi4(qj,q1j1,q2j2,q3j3) (Ry/(u^{1/2}Bohr)^{4}), "
                      "multiplicity\n";

            for (j = 0; j < nk_size; ++j) {
                const auto multi = quartet[j].group.size();
                unsigned int k1 = quartet[j].group[0].ks[0];
                unsigned int k2 = quartet[j].group[0].ks[1];
                unsigned int k3 = quartet[j].group[0].ks[2];

                unsigned int ib = 0;

                for (unsigned int is = 0; is < ns; ++is) {
                    for (unsigned int js = 0; js < ns; ++js) {
                        for (unsigned int ks = 0; ks < ns; ++ks) {
                            ofs_V4 << std::setw(5) << k1 + 1 << std::setw(5) << is + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k1][is]);
                            ofs_V4 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k2][js]);
                            ofs_V4 << std::setw(5) << k3 + 1 << std::setw(5) << ks + 1;
                            ofs_V4 << std::setw(15) << in_kayser(eval_tmp[k3][ks]);
                            ofs_V4 << std::setw(15) << phi4[j][ib].real();
                            ofs_V4 << std::setw(15) << phi4[j][ib].imag();
                            ofs_V4 << std::setw(5) << multi;
                            ofs_V4 << '\n';

                            ++ib;
                        }
                        ofs_V4 << '\n';
                    }
                    ofs_V4 << '\n';
                }
            }
            ofs_V4.close();
        }
    }
    print_vertex_offmesh(3, kslist.size());
}

void ModeAnalysis::calc_Phi3(const unsigned int knum, const unsigned int snum, const std::vector<KsListGroup> &triplet,
                             std::vector<std::vector<std::complex<double>>> &ret) const
{
    unsigned int is, js;
    unsigned int k1, k2;
    unsigned int arr[3];
    const auto ns = dynamical->neval;
    const size_t ns2 = ns * ns;

    const auto factor = std::pow(amu_ry, 1.5);
    const auto ntriplet = triplet.size();

    NDArray<std::complex<double>, 2> ret_loc;
    NDArray<std::complex<double>, 2> ret_sum;

    ret_loc.resize(ntriplet, ns2);
    ret_sum.resize(ntriplet, ns2);

    for (size_t ik = 0; ik < ntriplet; ++ik) {
        for (size_t ib = 0; ib < ns2; ++ib) {
            ret_loc[ik][ib] = std::complex<double>(0.0, 0.0);
            ret_sum[ik][ib] = std::complex<double>(0.0, 0.0);
        }
    }

    for (size_t ik = 0; ik < ntriplet; ++ik) {

        k1 = triplet[ik].group[0].ks[0];
        k2 = triplet[ik].group[0].ks[1];

        for (size_t ib = run.my_rank; ib < ns2; ib += run.nprocs) {

            is = ib / ns;
            js = ib % ns;

            arr[0] = ns * knum + snum;
            arr[1] = ns * k1 + is;
            arr[2] = ns * k2 + js;

            ret_loc[ik][ib] = anharmonic_core->Phi3(arr) * factor;
        }
    }

    const size_t count = ntriplet * ns2;
    MPI_Reduce(&ret_loc[0][0], &ret_sum[0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    if (run.my_rank == 0) {
        for (size_t ik = 0; ik < ntriplet; ++ik) {
            for (size_t ib = 0; ib < ns2; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    ret_loc.clear();
    ret_sum.clear();
}

void ModeAnalysis::calc_Phi4(const unsigned int knum, const unsigned int snum, const std::vector<KsListGroup> &quartet,
                             std::vector<std::vector<std::complex<double>>> &ret) const
{
    unsigned int is, js, ks;
    unsigned int k1, k2, k3;
    unsigned int arr[4];
    const auto ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns3 = ns2 * ns;

    double factor = pow2(amu_ry);
    const auto nquartet = quartet.size();

    NDArray<std::complex<double>, 2> ret_loc;
    NDArray<std::complex<double>, 2> ret_sum;

    ret_loc.resize(nquartet, ns3);
    ret_sum.resize(nquartet, ns3);

    for (size_t ik = 0; ik < nquartet; ++ik) {
        for (size_t ib = 0; ib < ns3; ++ib) {
            ret_loc[ik][ib] = std::complex<double>(0.0, 0.0);
            ret_sum[ik][ib] = std::complex<double>(0.0, 0.0);
        }
    }

    for (size_t ik = 0; ik < nquartet; ++ik) {
        k1 = quartet[ik].group[0].ks[0];
        k2 = quartet[ik].group[0].ks[1];
        k3 = quartet[ik].group[0].ks[2];

        for (size_t ib = run.my_rank; ib < ns3; ib += run.nprocs) {
            is = ib / ns2;
            js = (ib - is * ns2) / ns;
            ks = ib % ns;

            arr[0] = ns * knum + snum;
            arr[1] = ns * k1 + is;
            arr[2] = ns * k2 + js;
            arr[3] = ns * k3 + ks;

            ret_loc[ik][ib] = anharmonic_core->Phi4(arr) * factor;
        }
    }

    const size_t count = nquartet * ns3;
    MPI_Reduce(&ret_loc[0][0], &ret_sum[0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    if (run.my_rank == 0) {
        for (size_t ik = 0; ik < nquartet; ++ik) {
            for (size_t ib = 0; ib < ns3; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    ret_loc.clear();
    ret_sum.clear();
}

void ModeAnalysis::print_spectral_function(const unsigned int NT, const double *T_arr)
{
    auto ns = dynamical->neval;
    int i, j;
    int iomega;
    NDArray<double, 2> self3_imag;
    NDArray<double, 2> self3_real;
    std::ofstream ofs_self;
    NDArray<double, 1> omega_array;
    //    const auto Omega_min = dos->emin;
    //    const auto Omega_max = dos->emax;
    const auto delta_omega = dos->delta_e;

    const double Omega_min = 0.0;
    double Omega_max;

    //    auto emin_now = std::numeric_limits<double>::max();
    auto emax_now = std::numeric_limits<double>::min();
    double omega_tmp;

    for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            omega_tmp = in_kayser(dos->dymat_dos->get_eigenvalues()[dos->kmesh_dos->kpoint_irred_all[ik][0].knum][is]);
            //            emin_now = std::min(emin_now, omega_tmp);
            emax_now = std::max(emax_now, omega_tmp);
        }
    }
    Omega_max = (emax_now + delta_omega) * 2.0;


    const auto nomega = static_cast<unsigned int>((Omega_max - Omega_min) / delta_omega) + 1;

    omega_array.resize(nomega);
    self3_imag.resize(NT, nomega);
    self3_real.resize(NT, nomega);

    for (i = 0; i < nomega; ++i) {
        omega_array[i] = Omega_min + delta_omega * static_cast<double>(i);
        omega_array[i] *= time_ry / Hz_to_kayser;
    }

    for (i = 0; i < kslist.size(); ++i) {
        auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;
        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        if (run.my_rank == 0) {
            if (run.verbosity > 0) {
                std::cout << '\n';
                std::cout << " SELF_W = 1: Calculate bubble selfenergy with frequency dependency\n";
                std::cout << " for given " << kslist.size() << " modes.\n\n";
                std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
                std::cout << "  Phonon at k = (";
                for (j = 0; j < 3; ++j) {
                    std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                    if (j < 2) std::cout << ",";
                }
                std::cout << ")\n";
                std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            }

            if (write_text()) {
                std::string file_self = run.job_title + ".Self." + std::to_string(i + 1);
                ofs_self.open(file_self.c_str(), std::ios::out);
                if (!ofs_self) exit("run_mode_analysis", "Cannot open file file_shift");
                ofs_self << "# xk = ";
                for (j = 0; j < 3; ++j) {
                    ofs_self << std::setw(15) << dos->kmesh_dos->xk[knum][j];
                }
                ofs_self << '\n';
                ofs_self << "# mode = " << snum + 1 << '\n';
                ofs_self << "## T[K], Freq (cm^-1), omega (cm^-1), Self.real (cm^-1), Self.imag (cm^-1)\n";
            }
        }

        for (int iT = 0; iT < NT; ++iT) {
            const auto T_now = T_arr[iT];
            const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];

            if (run.my_rank == 0 && run.verbosity > 0) {
                std::cout << "  Temperature (K) : " << std::setw(15) << T_now << '\n';
                std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
            }

            {
                // Degenerate targets: average Im Sigma(omega) over the block.
                const auto eval_k = dos->dymat_dos->get_eigenvalues()[knum];
                const auto block = degenerate_block(eval_k, ns, snum);
                const double nblock = block.second - block.first;
                NDArray<double, 1> imag_tmp(nomega);
                for (iomega = 0; iomega < nomega; ++iomega) self3_imag[iT][iomega] = 0.0;
                for (auto s = block.first; s < block.second; ++s) {
                    anharmonic_core->calc_self3omega_tetrahedron(T_now,
                                                                 dos->kmesh_dos.get(),
                                                                 dos->dymat_dos->get_eigenvalues(),
                                                                 dos->dymat_dos->get_eigenvectors(),
                                                                 ik_irred,
                                                                 s,
                                                                 nomega,
                                                                 omega_array,
                                                                 imag_tmp);
                    for (iomega = 0; iomega < nomega; ++iomega) self3_imag[iT][iomega] += imag_tmp[iomega] / nblock;
                }
            }

            // Calculate real part of the self-energy by Kramers-Kronig relation
            kramers_kronig_real(nomega, omega_array, delta_omega, self3_imag[iT], self3_real[iT]);
            if (run.my_rank == 0) {
                auto &r = results[kslist_id[i]];
                if (iT == 0) {
                    r.omega = in_kayser(omega);
                    r.self_omega.resize(nomega);
                    for (iomega = 0; iomega < nomega; ++iomega) r.self_omega[iomega] = in_kayser(omega_array[iomega]);
                    r.self_real.assign(NT, std::vector<double>(nomega));
                    r.self_imag.assign(NT, std::vector<double>(nomega));
                }
                for (iomega = 0; iomega < nomega; ++iomega) {
                    r.self_real[iT][iomega] = in_kayser(self3_real[iT][iomega]);
                    r.self_imag[iT][iomega] = in_kayser(self3_imag[iT][iomega]);
                }
            }
            if (run.my_rank == 0 && write_text()) {
                for (iomega = 0; iomega < nomega; ++iomega) {
                    ofs_self << std::setw(10) << T_now << std::setw(15) << in_kayser(omega);
                    ofs_self << std::setw(10) << in_kayser(omega_array[iomega]) << std::setw(15)
                             << in_kayser(self3_real[iT][iomega]) << std::setw(15) << in_kayser(self3_imag[iT][iomega])
                             << '\n';
                }
                ofs_self << '\n';
            }
        }
        if (run.my_rank == 0 && write_text()) ofs_self.close();
    }
    print_spectral_function_offmesh(NT, T_arr, kslist.size(), nomega, omega_array, delta_omega);

    omega_array.clear();
    self3_imag.clear();
    self3_real.clear();
}

void ModeAnalysis::kramers_kronig_real(const unsigned int nomega, const double *omega_array, const double delta_omega,
                                       const double *imag, double *real)
{
    // Re Sigma(omega) from Im Sigma on the finite uniform grid (principal value by
    // skipping the diagonal); delta_omega in cm^-1, omega_array and imag in Ry.
    for (unsigned int iomega = 0; iomega < nomega; ++iomega) {
        auto self_tmp = 0.0;
        const auto w2 = omega_array[iomega] * omega_array[iomega];
        for (unsigned int jomega = 0; jomega < nomega; ++jomega) {
            if (jomega == iomega) continue;
            self_tmp += omega_array[jomega] * imag[jomega] / (omega_array[jomega] * omega_array[jomega] - w2);
        }
        real[iomega] = 2.0 * delta_omega * time_ry * self_tmp / (pi * Hz_to_kayser);
    }
}

void ModeAnalysis::print_spectral_function_offmesh(const unsigned int NT, const double *T_arr,
                                                   const size_t number_offset, const unsigned int nomega,
                                                   const double *omega_array, const double delta_omega)
{
    // Off-mesh text files use the on-mesh format and follow them in numbering.
    if (kslist_offmesh.empty()) return;
    const auto ns = dynamical->neval;
    NDArray<double, 2> xq(1, 3), eval_q(1, ns);
    NDArray<std::complex<double>, 3> evec_q(1, ns, ns);
    NDArray<double, 1> self_imag(nomega), self_real(nomega);
    AnharmonicCore::ShiftedGrid sg;

    for (size_t i = 0; i < kslist_offmesh.size(); ++i) {
        const auto &target = kslist_offmesh[i];
        const auto snum = target.snum;
        for (auto j = 0; j < 3; ++j) xq[0][j] = target.xk[j];
        if (!same_target_k(kslist_offmesh, i)) {
            eigen_at(xq[0], eval_q, evec_q);
            anharmonic_core->build_shifted_grid(xq[0], dos->kmesh_dos.get(), sg);
        }
        const auto omega = eval_q[0][snum];

        std::ofstream ofs_self;
        if (run.my_rank == 0) {
            if (run.verbosity > 0) {
                std::cout << "\n Number : " << std::setw(5) << number_offset + i + 1 << " (off the k-point grid)\n";
                std::cout << "  Phonon at k = (";
                for (auto j = 0; j < 3; ++j) {
                    std::cout << std::setw(10) << std::fixed << xq[0][j];
                    if (j < 2) std::cout << ",";
                }
                std::cout << ")\n";
                std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            }
            if (write_text()) {
                const auto file_self = run.job_title + ".Self." + std::to_string(number_offset + i + 1);
                ofs_self.open(file_self);
                if (!ofs_self) exit("print_spectral_function_offmesh", "Cannot open file file_self");
                ofs_self << "# xk = ";
                for (auto j = 0; j < 3; ++j) ofs_self << std::setw(15) << xq[0][j];
                ofs_self << '\n';
                ofs_self << "# mode = " << snum + 1 << '\n';
                ofs_self << "## T[K], Freq (cm^-1), omega (cm^-1), Self.real (cm^-1), Self.imag (cm^-1)\n";
            }
        }
        for (unsigned int iT = 0; iT < NT; ++iT) {
            const auto T_now = T_arr[iT];
            if (run.my_rank == 0 && run.verbosity > 0) {
                std::cout << "  Temperature (K) : " << std::setw(15) << T_now << '\n';
                std::cout << "  Frequency (cm^-1) : " << std::setw(15) << in_kayser(omega) << '\n';
            }
            {
                const auto block = degenerate_block(eval_q[0], ns, snum);
                const double nblock = block.second - block.first;
                NDArray<double, 1> imag_tmp(nomega);
                for (unsigned int iomega = 0; iomega < nomega; ++iomega) self_imag[iomega] = 0.0;
                for (auto s = block.first; s < block.second; ++s) {
                    anharmonic_core->calc_self3omega_tetrahedron_at(T_now,
                                                                    xq[0],
                                                                    eval_q[0][s],
                                                                    evec_q[0][s],
                                                                    dos->kmesh_dos.get(),
                                                                    dos->dymat_dos->get_eigenvalues(),
                                                                    dos->dymat_dos->get_eigenvectors(),
                                                                    sg,
                                                                    nomega,
                                                                    omega_array,
                                                                    imag_tmp);
                    for (unsigned int iomega = 0; iomega < nomega; ++iomega)
                        self_imag[iomega] += imag_tmp[iomega] / nblock;
                }
            }
            kramers_kronig_real(nomega, omega_array, delta_omega, self_imag, self_real);
            if (run.my_rank == 0) {
                auto &r = results[target.id];
                if (iT == 0) {
                    r.omega = in_kayser(omega);
                    r.self_omega.resize(nomega);
                    for (unsigned int iomega = 0; iomega < nomega; ++iomega)
                        r.self_omega[iomega] = in_kayser(omega_array[iomega]);
                    r.self_real.assign(NT, std::vector<double>(nomega));
                    r.self_imag.assign(NT, std::vector<double>(nomega));
                }
                for (unsigned int iomega = 0; iomega < nomega; ++iomega) {
                    r.self_real[iT][iomega] = in_kayser(self_real[iomega]);
                    r.self_imag[iT][iomega] = in_kayser(self_imag[iomega]);
                }
            }
            if (run.my_rank == 0 && write_text()) {
                for (unsigned int iomega = 0; iomega < nomega; ++iomega) {
                    ofs_self << std::setw(10) << T_now << std::setw(15) << in_kayser(omega);
                    ofs_self << std::setw(10) << in_kayser(omega_array[iomega]) << std::setw(15)
                             << in_kayser(self_real[iomega]) << std::setw(15) << in_kayser(self_imag[iomega]) << '\n';
                }
                ofs_self << '\n';
            }
        }
        if (run.my_rank == 0 && write_text()) ofs_self.close();
    }
}
