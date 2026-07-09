/*
 isotope.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "isotope.h"
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

void Isotope::calc_isotope_selfenergy_tetra(const unsigned int knum, const unsigned int snum, const double omega,
                                            const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                            const std::complex<double> *const *const *evec_in, const System &system_in,
                                            Integration &integration_in, const TetraNodes &tetra_nodes_in,
                                            const unsigned int ns_in, double &ret) const
{
    // Compute phonon selfenergy of phonon (knum, snum)
    // due to phonon-isotope scatterings.
    // This version employs the tetrahedron method.

    int ik, is;
    const auto nk = kmesh_in->nk;
    const auto ns = static_cast<int>(ns_in);
    const auto natmin = system_in.get_primcell().number_of_atoms;
    const auto tol_degenerate = 1.0e-7 * time_ry / Hz_to_kayser;

    ret = 0.0;

    NDArray<double, 1> eval;
    NDArray<double, 2> eval_tetra;
    NDArray<double, 2> prod_omega;
    NDArray<double, 2> weight_tetra;
    NDArray<unsigned int, 1> kmap_identity;

    eval.resize(nk);
    eval_tetra.resize(ns, nk);
    prod_omega.resize(ns, nk);
    weight_tetra.resize(ns, nk);
    kmap_identity.resize(nk);

    for (ik = 0; ik < nk; ++ik) {
        kmap_identity[ik] = ik;
    }
    average_degenerate_frequencies_transposed(nk, ns, eval_in, tol_degenerate, eval_tetra);

    for (is = 0; is < ns; ++is) {
#pragma omp parallel for
        for (ik = 0; ik < nk; ++ik) {

            const auto prod = tamura_overlap(natmin,
                                             evec_in[ik][is],
                                             evec_in[knum][snum],
                                             &isotope_factor[0],
                                             &system_in.get_primcell().kind[0]);

            prod_omega[is][ik] = prod * eval_tetra[is][ik];
            eval[ik] = eval_tetra[is][ik];
        }
        integration_in.calc_weight_tetrahedron(nk,
                                               kmap_identity,
                                               eval,
                                               omega,
                                               tetra_nodes_in.get_ntetra(),
                                               tetra_nodes_in.get_tetras(),
                                               weight_tetra[is]);
    }

    for (ik = 0; ik < nk; ++ik) {
        average_tetra_weights_over_degenerate_modes(ns, ik, eval_tetra, weight_tetra, tol_degenerate);
    }

    for (is = 0; is < ns; ++is) {
        for (ik = 0; ik < nk; ++ik) {
            ret += weight_tetra[is][ik] * prod_omega[is][ik];
        }
    }

    ret *= pi * omega * 0.25;

    eval.clear();
    eval_tetra.clear();
    prod_omega.clear();
    weight_tetra.clear();
    kmap_identity.clear();
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
        auto get_averaged_omega = [&](const unsigned int knum, const unsigned int snum) {
            auto begin = snum;
            while (begin > 0 && std::abs(eval_dos[knum][begin] - eval_dos[knum][begin - 1]) < tol_degenerate) {
                --begin;
            }

            auto end = snum + 1;
            while (end < ns && std::abs(eval_dos[knum][end] - eval_dos[knum][end - 1]) < tol_degenerate) {
                ++end;
            }

            auto omega_sum = 0.0;
            for (auto is = begin; is < end; ++is) {
                omega_sum += eval_dos[knum][is];
            }
            return omega_sum / static_cast<double>(end - begin);
        };

        for (i = my_rank_in; i < nks; i += nprocs_in) {
            const auto knum = kmesh_dos_in.kpoint_irred_all[i / ns][0].knum;
            const auto snum = i % ns;
            const auto omega = get_averaged_omega(knum, snum);
            if (integration_in.ismear == -1) {
                calc_isotope_selfenergy_tetra(knum,
                                              snum,
                                              omega,
                                              &kmesh_dos_in,
                                              eval_dos,
                                              dymat_dos_in.get_eigenvectors(),
                                              system_in,
                                              integration_in,
                                              tetra_nodes_dos_in,
                                              ns_in,
                                              tmp);
            } else {
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
            }
            gamma_loc[i] = tmp;
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
