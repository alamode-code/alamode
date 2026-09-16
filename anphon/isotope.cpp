/*
 isotope.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "isotope.h"
#include <algorithm>
#include <complex>
#include <iomanip>
#include <utility>
#include <vector>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "isotope_kernel.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "system.h"

using namespace PHON_NS;

Isotope::Isotope()
{
    set_default_variables();
};

Isotope::~Isotope()
{
    deallocate_variables();
};

void Isotope::set_default_variables()
{
    include_isotope = false;
}

void Isotope::deallocate_variables()
{
    isotope_factor.clear();
    if (gamma_isotope) {
        gamma_isotope.clear();
    }
}

void Isotope::setup_isotope_scattering(const System &system_in, const unsigned int nk_irred_in,
                                       const unsigned int ns_in, const int my_rank_in, const unsigned int verbosity)
{
    const int nkd = system_in.get_primcell().number_of_elems;

    MPI_Bcast(&include_isotope, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (include_isotope) {

        if (my_rank_in == 0) {
            if (isotope_factor.empty()) {
                isotope_factor.resize(nkd);
                set_isotope_factor_from_database(system_in, nkd, &system_in.symbol_kd[0], isotope_factor);
            } else {
                if (isotope_factor.size() != nkd) {
                    exit("setup_isotope_scattering",
                         "The number of elements in ISOFACT is inconsistent with the number of elements in KD.");
                }
            }
        } else {
            isotope_factor.resize(nkd);
        }

        MPI_Bcast(&isotope_factor[0], nkd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (my_rank_in == 0) {
            if (verbosity > 0) {
                std::cout << " ISOTOPE >= 1: Isotope scattering effects will be considered\n";
                std::cout << "               with the following scattering factors.\n";

                for (int i = 0; i < nkd; ++i) {
                    std::cout << std::setw(5) << system_in.symbol_kd[i] << ":";
                    std::cout << std::scientific << std::setw(17) << isotope_factor[i] << '\n';
                }
                std::cout << '\n';
            }

            gamma_isotope.resize(nk_irred_in, ns_in);
        }
    }
}

void Isotope::calc_isotope_selfenergy(const unsigned int knum, const unsigned int snum, const double omega,
                                      const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                      const std::complex<double> *const *const *evec_in, const System &system_in,
                                      Integration &integration_in, const unsigned int ns_in, double &ret) const
{
    // Compute phonon selfenergy of phonon (knum, snum)
    // due to phonon-isotope scatterings.
    // Delta functions are replaced by smearing functions with width EPSILON.

    const auto nk = kmesh_in->nk;
    const auto ns = static_cast<int>(ns_in);
    const auto natmin = system_in.get_primcell().number_of_atoms;
    const auto epsilon = integration_in.epsilon;

    ret = 0.0;

#pragma omp parallel for reduction(+ : ret)
    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {

            const auto prod = tamura_overlap(natmin,
                                             evec_in[ik][is],
                                             evec_in[knum][snum],
                                             &isotope_factor[0],
                                             &system_in.get_primcell().kind[0]);

            const auto omega1 = eval_in[ik][is];

            if (integration_in.ismear == 0) {
                ret += omega1 * delta_lorentz(omega - omega1, epsilon) * prod;
            } else if (integration_in.ismear == 1) {
                ret += omega1 * delta_gauss(omega - omega1, epsilon) * prod;
            } else if (integration_in.ismear == 2) {
                double eps;
                integration_in.adaptive_sigma->get_sigma(ik, is, eps);
                //integration_in.adaptive_smearing(ik, is, eps);
                //std::cout << eps << std::endl;
                ret += omega1 * delta_gauss(omega - omega1, eps) * prod;
            }
        }
    }

    ret *= pi * omega * 0.25 / static_cast<double>(nk);
}

namespace
{
// Tetrahedra of band `is` whose frequency range [min corner, max corner) can
// contain a given omega, stored as CSR lists per frequency bin. Built once per
// calc_isotope_selfenergy_all; lets each mode touch only O(ntetra/nbin)
// tetrahedra instead of all of them.
struct TetraBins
{
    double emin = 0.0;
    double de_inv = 0.0;
    int nbin = 0;
    std::vector<std::vector<unsigned int>> start; // [ns][nbin+1]
    std::vector<std::vector<unsigned int>> list;  // [ns]

    int bin(const double e) const
    {
        return std::min(std::max(static_cast<int>((e - emin) * de_inv), 0), nbin - 1);
    }
};

TetraBins build_tetra_bins(const int nk, const int ns, const double *const *eval_tetra, const unsigned int ntetra,
                           const unsigned int *const *tetras)
{
    TetraBins bins;
    auto emin = eval_tetra[0][0];
    auto emax = eval_tetra[0][0];
    for (int is = 0; is < ns; ++is) {
        for (int ik = 0; ik < nk; ++ik) {
            emin = std::min(emin, eval_tetra[is][ik]);
            emax = std::max(emax, eval_tetra[is][ik]);
        }
    }
    // ponytail: bin width ~ 1/4 of the mesh spacing in frequency; ~3 bins per tetrahedron on average.
    bins.nbin = std::max(1, static_cast<int>(4.0 * std::cbrt(static_cast<double>(nk))));
    bins.emin = emin;
    bins.de_inv = emax > emin ? bins.nbin / (emax - emin) : 0.0;
    bins.start.assign(ns, std::vector<unsigned int>(bins.nbin + 1, 0));
    bins.list.resize(ns);

    std::vector<int> lo(ntetra), hi(ntetra);
    for (int is = 0; is < ns; ++is) {
        auto &start = bins.start[is];
        for (unsigned int t = 0; t < ntetra; ++t) {
            auto e_lo = eval_tetra[is][tetras[t][0]];
            auto e_hi = e_lo;
            for (int j = 1; j < 4; ++j) {
                const auto e = eval_tetra[is][tetras[t][j]];
                e_lo = std::min(e_lo, e);
                e_hi = std::max(e_hi, e);
            }
            lo[t] = bins.bin(e_lo);
            hi[t] = bins.bin(e_hi);
            for (auto ib = lo[t]; ib <= hi[t]; ++ib) ++start[ib + 1];
        }
        for (int ib = 0; ib < bins.nbin; ++ib) start[ib + 1] += start[ib];
        auto &list = bins.list[is];
        list.resize(start[bins.nbin]);
        std::vector<unsigned int> fill(start.begin(), start.end() - 1);
        for (unsigned int t = 0; t < ntetra; ++t) {
            for (auto ib = lo[t]; ib <= hi[t]; ++ib) list[fill[ib]++] = t; // keeps ascending t within a bin
        }
    }
    return bins;
}

double averaged_omega(const double *const *eval, const int ns, const double tol_degenerate, const unsigned int knum,
                      const unsigned int snum)
{
    auto begin = snum;
    while (begin > 0 && std::abs(eval[knum][begin] - eval[knum][begin - 1]) < tol_degenerate) {
        --begin;
    }
    auto end = snum + 1;
    while (end < ns && std::abs(eval[knum][end] - eval[knum][end - 1]) < tol_degenerate) {
        ++end;
    }
    auto omega_sum = 0.0;
    for (auto is = begin; is < end; ++is) {
        omega_sum += eval[knum][is];
    }
    return omega_sum / static_cast<double>(end - begin);
}
} // namespace

void Isotope::calc_isotope_selfenergy_tetra_all(const KpointMeshUniform &kmesh_in, const DymatEigenValue &dymat_in,
                                                const TetraNodes &tetra_nodes_in, const System &system_in,
                                                const unsigned int ns_in, const int my_rank_in, const int nprocs_in,
                                                double *gamma_loc) const
{
    // Phonon selfenergy due to phonon-isotope scatterings, tetrahedron method.
    // Same arithmetic (and summation order) as summing over every tetrahedron
    // and every k point, but only the tetrahedra that can contain omega and
    // the k points they touch are visited.

    const auto nk = static_cast<int>(kmesh_in.nk);
    const auto ns = static_cast<int>(ns_in);
    const auto nks = kmesh_in.nk_irred * ns;
    const auto natmin = system_in.get_primcell().number_of_atoms;
    const int *kind = &system_in.get_primcell().kind[0];
    const auto tol_degenerate = 1.0e-7 * time_ry / Hz_to_kayser;
    const auto eval_in = dymat_in.get_eigenvalues();
    const auto evec_in = dymat_in.get_eigenvectors();
    const auto ntetra = tetra_nodes_in.get_ntetra();
    const auto tetras = tetra_nodes_in.get_tetras();
    const auto tetra_factor = 1.0 / static_cast<double>(ntetra);

    NDArray<double, 2> eval_tetra(ns, nk);
    average_degenerate_frequencies_transposed(nk, ns, eval_in, tol_degenerate, eval_tetra);
    const auto bins = build_tetra_bins(nk, ns, eval_tetra, ntetra, tetras);

    std::vector<unsigned int> kmap_identity(nk);
    for (int ik = 0; ik < nk; ++ik) kmap_identity[ik] = ik;

    std::vector<int> my_modes;
    for (int i = my_rank_in; i < nks; i += nprocs_in) my_modes.push_back(i);

#pragma omp parallel
    {
        // ponytail: ns*nk doubles per thread; switch to a sparse map if this ever matters.
        NDArray<double, 2> weight(ns, nk);
        for (int i = 0; i < ns * nk; ++i) weight.ptr()[0][i] = 0.0;
        std::vector<char> flag(nk, 0);
        std::vector<unsigned int> touched;

#pragma omp for schedule(dynamic, 4)
        for (int im = 0; im < static_cast<int>(my_modes.size()); ++im) {
            const auto i = my_modes[im];
            const auto knum = kmesh_in.kpoint_irred_all[i / ns][0].knum;
            const auto snum = i % ns;
            const auto omega = averaged_omega(eval_in, ns, tol_degenerate, knum, snum);
            const auto ib = bins.bin(omega);

            touched.clear();
            for (int is = 0; is < ns; ++is) {
                const auto &list = bins.list[is];
                for (auto it = bins.start[is][ib]; it < bins.start[is][ib + 1]; ++it) {
                    const auto *tetra = tetras[list[it]];
                    if (Integration::add_tetrahedron_weight(&kmap_identity[0],
                                                            eval_tetra[is],
                                                            omega,
                                                            tetra,
                                                            weight[is]))
                    {
                        for (int j = 0; j < 4; ++j) {
                            if (!flag[tetra[j]]) {
                                flag[tetra[j]] = 1;
                                touched.push_back(tetra[j]);
                            }
                        }
                    }
                }
            }
            std::sort(touched.begin(), touched.end());

            for (const auto ik: touched) {
                for (int is = 0; is < ns; ++is) weight[is][ik] *= tetra_factor;
                average_tetra_weights_over_degenerate_modes(ns, ik, eval_tetra, weight, tol_degenerate);
            }

            auto ret = 0.0;
            for (int is = 0; is < ns; ++is) {
                for (const auto ik: touched) {
                    if (weight[is][ik] == 0.0) continue; // overlap only where this band has weight
                    const auto prod =
                        tamura_overlap(natmin, evec_in[ik][is], evec_in[knum][snum], &isotope_factor[0], kind);
                    ret += weight[is][ik] * (prod * eval_tetra[is][ik]);
                }
            }
            gamma_loc[i] = ret * pi * omega * 0.25;

            for (const auto ik: touched) {
                flag[ik] = 0;
                for (int is = 0; is < ns; ++is) weight[is][ik] = 0.0;
            }
        }
    }
}

void Isotope::calc_isotope_selfenergy_all(const KpointMeshUniform &kmesh_dos_in, const DymatEigenValue &dymat_dos_in,
                                          const TetraNodes &tetra_nodes_dos_in, const System &system_in,
                                          Integration &integration_in, const unsigned int ns_in, const int my_rank_in,
                                          const int nprocs_in, const unsigned int verbosity)
{
    int i;
    const auto ns = static_cast<int>(ns_in);
    const auto nks = kmesh_dos_in.nk_irred * ns;
    double tmp;
    NDArray<double, 1> gamma_tmp;
    NDArray<double, 1> gamma_loc;

    if (include_isotope) {

        if (my_rank_in == 0 && verbosity > 0) {
            if (integration_in.ismear == -1) {
                std::cout << " Calculating self-energies from isotope scatterings (tetra)... ";
            } else if (integration_in.ismear == 0) {
                std::cout << " Calculating self-energies from isotope scatterings (lorentz)... ";
            } else if (integration_in.ismear == 1) {
                std::cout << " Calculating self-energies from isotope scatterings (gaussian)... ";
            } else if (integration_in.ismear == 2) {
                std::cout << " Calculating self-energies from isotope scatterings (adaptive)... ";
            }
        }

        if (my_rank_in == 0) {
            gamma_tmp.resize(nks);
        } else {
            gamma_tmp.resize(1);
        }
        gamma_loc.resize(nks);

        for (i = 0; i < nks; ++i) gamma_loc[i] = 0.0;

        const auto tol_degenerate = 1.0e-7 * time_ry / Hz_to_kayser;
        const auto eval_dos = dymat_dos_in.get_eigenvalues();

        if (integration_in.ismear == -1) {
            calc_isotope_selfenergy_tetra_all(kmesh_dos_in,
                                              dymat_dos_in,
                                              tetra_nodes_dos_in,
                                              system_in,
                                              ns_in,
                                              my_rank_in,
                                              nprocs_in,
                                              gamma_loc);
        } else {
            for (i = my_rank_in; i < nks; i += nprocs_in) {
                const auto knum = kmesh_dos_in.kpoint_irred_all[i / ns][0].knum;
                const auto snum = i % ns;
                const auto omega = averaged_omega(eval_dos, ns, tol_degenerate, knum, snum);
                calc_isotope_selfenergy(knum,
                                        snum,
                                        omega,
                                        &kmesh_dos_in,
                                        eval_dos,
                                        dymat_dos_in.get_eigenvectors(),
                                        system_in,
                                        integration_in,
                                        ns_in,
                                        tmp);
                gamma_loc[i] = tmp;
            }
        }

        MPI_Reduce(&gamma_loc[0], &gamma_tmp[0], nks, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (my_rank_in == 0) {
            for (i = 0; i < kmesh_dos_in.nk_irred; ++i) {
                for (int j = 0; j < ns; ++j) {
                    gamma_isotope[i][j] = gamma_tmp[ns * i + j];
                }
            }

            for (i = 0; i < kmesh_dos_in.nk_irred; ++i) {
                const auto knum = kmesh_dos_in.kpoint_irred_all[i][0].knum;
                auto begin = 0;
                auto omega_ref = eval_dos[knum][0];

                for (auto is = 1; is <= ns; ++is) {
                    if (is < ns && std::abs(eval_dos[knum][is] - omega_ref) < tol_degenerate) {
                        continue;
                    }

                    if (is - begin > 1) {
                        auto gamma_sum = 0.0;
                        for (auto js = begin; js < is; ++js) {
                            gamma_sum += gamma_isotope[i][js];
                        }
                        const auto gamma_avg = gamma_sum / static_cast<double>(is - begin);
                        for (auto js = begin; js < is; ++js) {
                            gamma_isotope[i][js] = gamma_avg;
                        }
                    }

                    if (is < ns) {
                        begin = is;
                        omega_ref = eval_dos[knum][is];
                    }
                }
            }
        }

        gamma_tmp.clear();
        gamma_loc.clear();

        if (my_rank_in == 0 && verbosity > 0) {
            std::cout << "done!\n";
        }
    }
}

void Isotope::set_isotope_factor_from_database(const System &system_in, const int nkd, const std::string *symbol_in,
                                               std::vector<double> &isofact_out)
{
    for (int i = 0; i < nkd; ++i) {
        const auto atom_number = system_in.get_atomic_number_by_name(symbol_in[i]);
        if (atom_number >= isotope_factors.size() || atom_number == -1) {
            exit("set_isotope_factor_from_database",
                 "The isotope factor for the given element doesn't exist in the database.\n"
                 "Therefore, please input ISOFACT manually.");
        }
        const auto isofact_tmp = isotope_factors[atom_number];
        if (isofact_tmp < -0.5) {
            exit("set_isotope_factor_from_database",
                 "One of the elements in the KD-tag is unstable. "
                 "Therefore, please input ISOFACT manually.");
        }
        isofact_out[i] = isofact_tmp;
    }
}
