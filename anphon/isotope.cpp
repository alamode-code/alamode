/*
 isotope.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "isotope.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "memory.h"
#include "phonon_dos.h"
#include "system.h"
#include <iomanip>
#include <complex>

using namespace PHON_NS;

Isotope::Isotope(PHON *phon) :
    Pointers(phon)
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
    gamma_isotope = nullptr;
}

void Isotope::deallocate_variables()
{
    isotope_factor.clear();
    if (gamma_isotope) {
        deallocate(gamma_isotope);
    }
}

void Isotope::setup_isotope_scattering()
{
    const int nkd = system->get_primcell().number_of_elems;

    MPI_Bcast(&include_isotope, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (include_isotope) {

        if (mympi->my_rank == 0) {
            if (isotope_factor.empty()) {
                isotope_factor.resize(nkd);
                set_isotope_factor_from_database(nkd,
                                                 &system->symbol_kd[0],
                                                 isotope_factor);
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

        if (mympi->my_rank == 0) {
            std::cout << " ISOTOPE >= 1: Isotope scattering effects will be considered\n";
            std::cout << "               with the following scattering factors.\n";

            for (int i = 0; i < nkd; ++i) {
                std::cout << std::setw(5) << system->symbol_kd[i] << ":";
                std::cout << std::scientific << std::setw(17) << isotope_factor[i] << '\n';
            }
            std::cout << '\n';

            allocate(gamma_isotope, dos->kmesh_dos->nk_irred, dynamical->neval);
        }
    }
}

void Isotope::calc_isotope_selfenergy(const unsigned int knum,
                                      const unsigned int snum,
                                      const double omega,
                                      const KpointMeshUniform *kmesh_in,
                                      const double *const *eval_in,
                                      const std::complex<double> *const *const *evec_in,
                                      double &ret) const
{
    // Compute phonon selfenergy of phonon (knum, snum)
    // due to phonon-isotope scatterings.
    // Delta functions are replaced by smearing functions with width EPSILON.

    const auto nk = kmesh_in->nk;
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto epsilon = integration->epsilon;

    ret = 0.0;

#pragma omp parallel for reduction(+: ret)
    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {

            auto prod = 0.0;

            for (auto iat = 0; iat < natmin; ++iat) {

                auto dprod = std::complex<double>(0.0, 0.0);
                for (auto icrd = 0; icrd < 3; ++icrd) {
                    dprod += std::conj(evec_in[ik][is][3 * iat + icrd])
                        * evec_in[knum][snum][3 * iat + icrd];
                }
                prod += isotope_factor[system->get_primcell().kind[iat]] * std::norm(dprod);
            }

            const auto omega1 = eval_in[ik][is];

            if (integration->ismear == 0) {
                ret += omega1 * delta_lorentz(omega - omega1, epsilon) * prod;
            } else if (integration->ismear == 1) {
                ret += omega1 * delta_gauss(omega - omega1, epsilon) * prod;
            } else if (integration->ismear == 2) {
                double eps;
                integration->adaptive_sigma->get_sigma(ik, is, eps);
                //integration->adaptive_smearing(ik, is, eps);
                //std::cout << eps << std::endl;
                ret += omega1 * delta_gauss(omega - omega1, eps) * prod;
            }
        }
    }

    ret *= pi * omega * 0.25 / static_cast<double>(nk);
}

void Isotope::calc_isotope_selfenergy_tetra(const unsigned int knum,
                                            const unsigned int snum,
                                            const double omega,
                                            const KpointMeshUniform *kmesh_in,
                                            const double *const *eval_in,
                                            const std::complex<double> *const *const *evec_in,
                                            double &ret) const
{
    // Compute phonon selfenergy of phonon (knum, snum)
    // due to phonon-isotope scatterings.
    // This version employs the tetrahedron method.

    int ik, is;
    const auto nk = kmesh_in->nk;
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;

    ret = 0.0;

    double *eval;
    double *weight;

    allocate(eval, nk);
    allocate(weight, nk);

    for (is = 0; is < ns; ++is) {
#pragma omp parallel for
        for (ik = 0; ik < nk; ++ik) {

            auto prod = 0.0;

            for (auto iat = 0; iat < natmin; ++iat) {

                auto dprod = std::complex<double>(0.0, 0.0);
                for (auto icrd = 0; icrd < 3; ++icrd) {
                    dprod += std::conj(evec_in[ik][is][3 * iat + icrd])
                        * evec_in[knum][snum][3 * iat + icrd];
                }
                prod += isotope_factor[system->get_primcell().kind[iat]] * std::norm(dprod);
            }

            weight[ik] = prod * eval_in[ik][is];
            eval[ik] = eval_in[ik][is];
        }
        ret += integration->do_tetrahedron(eval,
                                           weight,
                                           dos->tetra_nodes_dos->get_ntetra(),
                                           dos->tetra_nodes_dos->get_tetras(),
                                           omega);
    }

    ret *= pi * omega * 0.25;

    deallocate(eval);
    deallocate(weight);
}

void Isotope::calc_isotope_selfenergy_all() const
{
    int i;
    const auto ns = dynamical->neval;
    const auto nks = dos->kmesh_dos->nk_irred * ns;
    double tmp;
    double *gamma_tmp = nullptr;
    double *gamma_loc = nullptr;

    if (include_isotope) {

        if (mympi->my_rank == 0) {
            if (integration->ismear == -1) {
                std::cout << " Calculating self-energies from isotope scatterings (tetra)... ";
            } else if (integration->ismear == 0) {
                std::cout << " Calculating self-energies from isotope scatterings (lorentz)... ";
            } else if (integration->ismear == 1) {
                std::cout << " Calculating self-energies from isotope scatterings (gaussian)... ";
            } else if (integration->ismear == 2) {
                std::cout << " Calculating self-energies from isotope scatterings (adaptive)... ";
            }
        }

        if (mympi->my_rank == 0) {
            allocate(gamma_tmp, nks);
        } else {
            allocate(gamma_tmp, 1);
        }
        allocate(gamma_loc, nks);

        for (i = 0; i < nks; ++i) gamma_loc[i] = 0.0;

        for (i = mympi->my_rank; i < nks; i += mympi->nprocs) {
            const auto knum = dos->kmesh_dos->kpoint_irred_all[i / ns][0].knum;
            const auto snum = i % ns;
            const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];
            if (integration->ismear == -1) {
                calc_isotope_selfenergy_tetra(knum,
                                              snum,
                                              omega,
                                              dos->kmesh_dos,
                                              dos->dymat_dos->get_eigenvalues(),
                                              dos->dymat_dos->get_eigenvectors(),
                                              tmp);
            } else {
                calc_isotope_selfenergy(knum,
                                        snum,
                                        omega,
                                        dos->kmesh_dos,
                                        dos->dymat_dos->get_eigenvalues(),
                                        dos->dymat_dos->get_eigenvectors(),
                                        tmp);
            }
            gamma_loc[i] = tmp;
        }

        MPI_Reduce(&gamma_loc[0],
                   &gamma_tmp[0],
                   nks,
                   MPI_DOUBLE,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        if (mympi->my_rank == 0) {
            for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
                for (int j = 0; j < ns; ++j) {
                    gamma_isotope[i][j] = gamma_tmp[ns * i + j];
                }
            }
        }

        deallocate(gamma_tmp);
        deallocate(gamma_loc);

        if (mympi->my_rank == 0) {
            std::cout << "done!\n";
        }
    }
}

void Isotope::set_isotope_factor_from_database(const int nkd,
                                               const std::string *symbol_in,
                                               std::vector<double> &isofact_out)
{
    for (int i = 0; i < nkd; ++i) {
        const auto atom_number = system->get_atomic_number_by_name(symbol_in[i]);
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
