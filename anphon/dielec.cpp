/*
 dielec.cpp

 Copyright (c) 2019 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "dielec.h"
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "system.h"
#include "write_phonons.h"

using namespace PHON_NS;

Dielec::Dielec(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Dielec::~Dielec()
{
    deallocate_variables();
}

void Dielec::set_default_variables()
{
    calc_dielectric_constant = 0;
    dielec = nullptr;
    omega_grid = nullptr;
    emin = 0.0;
    emax = 1.0;
    delta_e = 1.0;
    nomega = 1;
    file_born = "";
    borncharge = nullptr;
    symmetrize_borncharge = 0;
    dielec_tensor.setZero();
}

void Dielec::deallocate_variables()
{
    if (dielec) {
        deallocate(dielec);
    }
    if (omega_grid) {
        deallocate(omega_grid);
    }
    if (borncharge) {
        deallocate(borncharge);
    }
}

void Dielec::init()
{
    // This should be called after Dos::setup()

    if (mympi->my_rank == 0) {
        emax = dos->emax;
        emin = dos->emin;
        delta_e = dos->delta_e;
        nomega = static_cast<int>((emax - emin) / delta_e);
    }

    MPI_Bcast(&calc_dielectric_constant, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nomega, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&emin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&emax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (dynamical->nonanalytic || calc_dielectric_constant) {
        if (mympi->my_rank == 0) {
            if (file_born.empty()) {
                if (calc_dielectric_constant) {
                    exitall("Dielec::init()", "BORNINFO must be set when DIELEC = 1.");
                }
                exitall("Dielec::init()", "BORNINFO must be set when NONANALYTIC>0.");
            }
        }
        setup_dielectric(1);
    }

    if (calc_dielectric_constant) {
        allocate(omega_grid, nomega);

        for (auto i = 0; i < nomega; ++i) {
            omega_grid[i] = emin + delta_e * static_cast<double>(i);
        }
    }
}

void Dielec::setup_dielectric(const unsigned int verbosity)
{
    if (borncharge) deallocate(borncharge);

    allocate(borncharge, system->get_primcell().number_of_atoms, 3, 3);
    if (mympi->my_rank == 0) load_born(symmetrize_borncharge, verbosity);

    MPI_Bcast(dielec_tensor.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&borncharge[0][0][0], 9 * system->get_primcell().number_of_atoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Dielec::load_born(const unsigned int flag_symmborn, const unsigned int verbosity)
{
    // Read the dielectric tensor and born effective charges from file_born

    unsigned int i, j, k;
    double sum_born[3][3];
    std::ifstream ifs_born;

    const auto natmin_tmp = system->get_primcell().number_of_atoms;

    ifs_born.open(file_born.c_str(), std::ios::in);
    if (!ifs_born) exit("load_born", "cannot open file_born");

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ifs_born >> dielec_tensor(i, j);
        }
    }

    for (i = 0; i < natmin_tmp; ++i) {
        for (j = 0; j < 3; ++j) {
            for (k = 0; k < 3; ++k) {
                ifs_born >> borncharge[i][j][k];
            }
        }
    }
    ifs_born.close();

    if (verbosity > 0) {
        std::cout << "  Dielectric constants and Born effective charges are read from " << file_born << ".\n\n";
        std::cout << "  Dielectric constant tensor in Cartesian coordinate : \n";
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << dielec_tensor(i, j);
            }
            std::cout << '\n';
        }
        std::cout << '\n';

        std::cout << "  Born effective charge tensor in Cartesian coordinate\n";
        for (i = 0; i < natmin_tmp; ++i) {
            std::cout << "  Atom" << std::setw(5) << i + 1 << "(" << std::setw(3)
                      << system->symbol_kd[system->get_supercell(0).kind[system->get_map_p2s(0)[i][0]]] << ") :\n";

            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    std::cout << std::setw(15) << std::fixed << std::setprecision(6) << borncharge[i][j][k];
                }
                std::cout << '\n';
            }
        }
    }


    // Check if the ASR is satisfied. If not, enforce it.

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            sum_born[i][j] = 0.0;
            for (k = 0; k < natmin_tmp; ++k) {
                sum_born[i][j] += borncharge[k][i][j];
            }
        }
    }

    double res = 0.0;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            res += std::pow(sum_born[i][j], 2);
        }
    }

    if (res > eps10) {
        if (verbosity > 0) {
            std::cout << '\n';
            std::cout << "  WARNING: Born effective charges do not satisfy the acoustic sum rule.\n";
            std::cout << "           The born effective charges are modified to satisfy the ASR.\n";
        }

        for (i = 0; i < natmin_tmp; ++i) {
            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    borncharge[i][j][k] -= sum_born[j][k] / static_cast<double>(system->get_primcell().number_of_atoms);
                }
            }
        }
    }

    if (flag_symmborn) {

        // Symmetrize Born effective charges. Necessary to avoid the violation of ASR
        // particularly for NONANALYTIC=3 (Ewald summation).

        int iat;
        double ***born_sym;
        double rot[3][3];

        allocate(born_sym, natmin_tmp, 3, 3);

        for (iat = 0; iat < natmin_tmp; ++iat) {
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    born_sym[iat][i][j] = 0.0;
                }
            }
        }

        for (auto isym = 0; isym < symmetry->SymmListWithMap.size(); ++isym) {
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    rot[i][j] = symmetry->SymmListWithMap[isym].rot[3 * i + j];
                }
            }

            for (iat = 0; iat < natmin_tmp; ++iat) {
                int iat_sym = symmetry->SymmListWithMap[isym].mapping[iat];

                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        for (k = 0; k < 3; ++k) {
                            for (int m = 0; m < 3; ++m) {
                                born_sym[iat_sym][i][j] += rot[i][k] * rot[j][m] * borncharge[iat][k][m];
                            }
                        }
                    }
                }
            }
        }

        for (iat = 0; iat < natmin_tmp; ++iat) {
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    born_sym[iat][i][j] /= static_cast<double>(symmetry->SymmListWithMap.size());
                }
            }
        }

        // Check if the Born effective charges given by the users satisfy the symmetry.

        auto diff_sym = 0.0;
        for (iat = 0; iat < natmin_tmp; ++iat) {
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    diff_sym = std::max<double>(diff_sym, std::abs(borncharge[iat][i][j] - born_sym[iat][i][j]));
                }
            }
        }

        if (diff_sym > 0.5 && verbosity > 0) {
            std::cout << '\n';
            std::cout << "  WARNING: Born effective charges are inconsistent with the crystal symmetry.\n";
        }

        for (iat = 0; iat < natmin_tmp; ++iat) {
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    borncharge[iat][i][j] = born_sym[iat][i][j];
                }
            }
        }
        deallocate(born_sym);

        if (verbosity > 0) {
            if (diff_sym > eps8 || res > eps10) {
                std::cout << '\n';
                std::cout << "  Symmetrized Born effective charge tensor in Cartesian coordinate." << '\n';
                for (i = 0; i < natmin_tmp; ++i) {
                    std::cout << "  Atom" << std::setw(5) << i + 1 << "(" << std::setw(3)
                              << system->symbol_kd[system->get_primcell().kind[system->get_map_p2s(0)[i][0]]]
                              << ") :" << '\n';

                    for (j = 0; j < 3; ++j) {
                        for (k = 0; k < 3; ++k) {
                            std::cout << std::setw(15) << borncharge[i][j][k];
                        }
                        std::cout << '\n';
                    }
                }
            }
        }
    }
    std::cout << std::scientific;
}

double *Dielec::get_omega_grid(unsigned int &nomega_in) const
{
    nomega_in = nomega;
    return omega_grid;
}

double ***Dielec::get_dielectric_func() const
{
    return dielec;
}

void Dielec::run_dielec_calculation()
{
    double *xk;
    double *eval;
    std::complex<double> **evec;
    const auto ns = dynamical->neval;

    allocate(xk, 3);
    allocate(eval, ns);
    allocate(evec, ns, ns);
    allocate(dielec, nomega, 3, 3);

    for (auto i = 0; i < 3; ++i)
        xk[i] = 0.0;

    dynamical->eval_k(xk, xk, fcs_phonon->force_constant_with_cell[0], eval, evec, true);

    compute_dielectric_function(nomega, omega_grid, eval, evec, dielec);

    deallocate(xk);
    deallocate(eval);
    deallocate(evec);
}

void Dielec::compute_dielectric_function(const unsigned int nomega_in, double *omega_grid_in, double *eval_in,
                                         std::complex<double> **evec_in, double ***dielec_out)
{
    const auto ns = dynamical->neval;
    const auto zstar = borncharge;

#ifdef _DEBUG
    for (auto i = 0; i < ns; ++i) {
        std::cout << "eval = " << eval_in[i] << '\n';
        for (auto j = 0; j < ns; ++j) {
            std::cout << std::setw(15) << evec_in[i][j].real();
            std::cout << std::setw(15) << evec_in[i][j].imag();
            std::cout << '\n';
        }
        std::cout << '\n';
    }
#endif

    for (auto i = 0; i < ns; ++i) {
        for (auto j = 0; j < ns; ++j) {
            evec_in[i][j] /= std::sqrt(system->get_mass_super()[system->get_map_p2s(0)[j / 3][0]]);
        }
    }

#ifdef _DEBUG
    for (auto i = 0; i < ns; ++i) {
        std::cout << "U = " << eval_in[i] << '\n';
        for (auto j = 0; j < ns; ++j) {
            std::cout << std::setw(15) << real(evec_in[i][j]);
            std::cout << std::setw(15) << evec_in[i][j].imag();
            std::cout << '\n';
        }
        std::cout << '\n';
    }
#endif

    double ***s_born;
    double **zstar_u;

    allocate(zstar_u, 3, ns);
    allocate(s_born, 3, 3, ns);

    for (auto i = 0; i < 3; ++i) {
        for (auto is = 0; is < ns; ++is) {
            zstar_u[i][is] = 0.0;

            for (auto j = 0; j < ns; ++j) {
                zstar_u[i][is] += zstar[j / 3][i][j % 3] * evec_in[is][j].real();
            }
        }
    }

#ifdef _DEBUG
    std::cout << "Zstar_u:\n";
    for (auto is = 0; is < ns; ++is) {
        for (auto i = 0; i < 3; ++i) {
            std::cout << std::setw(15) << zstar_u[i][is];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    std::cout << "S_born:\n";
    for (auto is = 0; is < ns; ++is) {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << s_born[i][j][is];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

#endif

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto is = 0; is < ns; ++is) {
                s_born[i][j][is] = zstar_u[i][is] * zstar_u[j][is];
            }
        }
    }

    auto freq_conv_factor = time_ry * time_ry / (Hz_to_kayser * Hz_to_kayser);
    auto factor = 8.0 * pi / system->get_primcell().volume;
    double w2_tmp;
    for (auto iomega = 0; iomega < nomega_in; ++iomega) {
        w2_tmp = omega_grid_in[iomega] * omega_grid_in[iomega] * freq_conv_factor;

        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                dielec_out[iomega][i][j] = 0.0;

                for (auto is = 3; is < ns; ++is) {
                    dielec_out[iomega][i][j] += s_born[i][j][is] / (eval_in[is] - w2_tmp);
                }
                dielec_out[iomega][i][j] *= factor;
            }
        }
    }

    deallocate(zstar_u);
    deallocate(s_born);
}

std::vector<std::vector<double>> Dielec::get_zstar_mode() const
{
    const auto ns = dynamical->neval;
    std::vector<std::vector<double>> zstar_mode(ns, std::vector<double>(3));
    compute_mode_effective_charge(zstar_mode, false);
    return zstar_mode;
}

void Dielec::compute_mode_effective_charge(std::vector<std::vector<double>> &zstar_mode, const bool do_normalize) const
{
    // Compute the effective charges of normal coordinate at q = 0.

    std::vector<double> xk(3);
    double *eval;
    std::complex<double> **evec;
    const auto ns = dynamical->neval;
    const auto zstar_atom = borncharge;

    allocate(eval, ns);
    allocate(evec, ns, ns);

    for (auto i = 0; i < 3; ++i)
        xk[i] = 0.0;

    // Probably, I need to symmetrize the eigenvector here.
    std::vector<std::vector<double>> projectors;
    std::vector<double> vecs(3);

    if (!dynamical->get_projection_directions().empty()) {
        dynamical->project_degenerate_eigenvectors(system->get_primcell().lattice_vector,
                                                   fcs_phonon->force_constant_with_cell[0],
                                                   &xk[0],
                                                   dynamical->get_projection_directions(),
                                                   evec);
    } else {
        dynamical->eval_k(&xk[0], &xk[0], fcs_phonon->force_constant_with_cell[0], eval, evec, true);
    }

    // Divide by sqrt of atomic mass to get normal coordinate
    for (auto i = 0; i < ns; ++i) {
        for (auto j = 0; j < ns; ++j) {
            evec[i][j] /= std::sqrt(system->get_mass_super()[system->get_map_p2s(0)[j / 3][0]] / amu_ry);
            //            evec[i][j] /= std::sqrt(system->mass[system->map_trueprim_to_super[j / 3][0]]);
        }
    }

    // Compute the mode effective charges defined by Eq. (53) or its numerator of
    // Gonze & Lee, PRB 55, 10355 (1997).
    for (auto i = 0; i < 3; ++i) {
        for (auto is = 0; is < ns; ++is) {
            zstar_mode[is][i] = 0.0;
            auto normalization_factor = 0.0;

            for (auto j = 0; j < ns; ++j) {
                zstar_mode[is][i] += zstar_atom[j / 3][i][j % 3] * evec[is][j].real();
                normalization_factor += std::norm(evec[is][j]);
            }
            if (do_normalize) zstar_mode[is][i] /= std::sqrt(normalization_factor);
        }
    }

    deallocate(eval);
    deallocate(evec);
}

double ***Dielec::get_borncharge() const
{
    return borncharge;
}

Eigen::Matrix3d Dielec::get_dielec_tensor() const
{
    return dielec_tensor;
}
