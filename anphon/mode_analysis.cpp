/*
 mode_analysis.cpp

Copyright (c) 2018 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mode_analysis.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include "anharmonic_core.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "selfenergy.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

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
    calc_fstate_k = false;
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

    if (mympi->my_rank == 0) {
        if (!ks_input.empty()) {
            ks_analyze_mode = true;

            std::cout << "\n KS_INPUT-tag is given : Analysis on the specified phonon modes\n";
            std::cout << " will be performed instead of thermal conductivity calculation.\n\n";

            std::ifstream ifs_ks;
            ifs_ks.open(ks_input.c_str(), std::ios::in);
            if (!ifs_ks) exit("setup_mode_analysis", "Cannot open file KS_INPUT");

            unsigned int nlist;
            double ktmp[3];
            unsigned int snum_tmp;

            ifs_ks >> nlist;

            if (nlist <= 0) exit("setup_mode_analysis", "First line in KS_INPUT files should be a positive integer.");

            if (calc_fstate_k) {
                kslist_fstate_k.clear();

                for (i = 0; i < nlist; ++i) {
                    ifs_ks >> ktmp[0] >> ktmp[1] >> ktmp[2] >> snum_tmp;

                    if (snum_tmp <= 0 || snum_tmp > dynamical->neval) {
                        exit("setup_mode_analysis", "Mode index out of range.");
                    }

                    kslist_fstate_k.emplace_back(ktmp, snum_tmp - 1);
                }
                std::cout << " The number of entries = " << kslist_fstate_k.size() << '\n';
            } else {
                kslist.clear();
                for (i = 0; i < nlist; ++i) {
                    ifs_ks >> ktmp[0] >> ktmp[1] >> ktmp[2] >> snum_tmp;
                    const auto knum_tmp = dos->kmesh_dos->get_knum(ktmp);

                    if (knum_tmp == -1)
                        exit("setup_mode_analysis", "Given kpoint does not exist in given k-point grid.");
                    if (snum_tmp <= 0 || snum_tmp > dynamical->neval) {
                        exit("setup_mode_analysis", "Mode index out of range.");
                    }
                    kslist.push_back(knum_tmp * dynamical->neval + snum_tmp - 1);
                }
                std::cout << " The number of entries = " << kslist.size() << '\n';
            }

            ifs_ks.close();
        } else {
            ks_analyze_mode = false;
        }
    }

    MPI_Bcast(&ks_analyze_mode, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_realpart, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_fstate_k, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_fstate_omega, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_fstate_k, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_V3, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_V4, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_selfenergy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&spectral_func, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    unsigned int nlist;

    if (kpoint->kpoint_mode == 3) {

        if (!ks_analyze_mode) {
            exit("setup_mode_analysis", "KPMODE = 3 must be used with FSTATE_K = 1");
        }
        double **vec_tmp;
        unsigned int *mode_tmp;


        // Broadcast kslist_fstate_k

        nlist = kslist_fstate_k.size();
        MPI_Bcast(&nlist, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        allocate(vec_tmp, nlist, 3);
        allocate(mode_tmp, nlist);

        if (mympi->my_rank == 0) {
            for (i = 0; i < nlist; ++i) {
                for (int j = 0; j < 3; ++j) {
                    vec_tmp[i][j] = kslist_fstate_k[i].xk[j];
                }
                mode_tmp[i] = kslist_fstate_k[i].nmode;
            }
        }

        MPI_Bcast(&vec_tmp[0][0], 3 * nlist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mode_tmp[0], nlist, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (mympi->my_rank > 0) {
            kslist_fstate_k.clear();

            for (i = 0; i < nlist; ++i) {
                kslist_fstate_k.emplace_back(vec_tmp[i], mode_tmp[i]);
            }
        }

        deallocate(vec_tmp);
        deallocate(mode_tmp);

    } else {

        unsigned int *kslist_arr;
        nlist = kslist.size();

        // Broadcast kslist

        MPI_Bcast(&nlist, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        allocate(kslist_arr, nlist);

        if (mympi->my_rank == 0) {
            for (i = 0; i < nlist; ++i)
                kslist_arr[i] = kslist[i];
        }
        MPI_Bcast(&kslist_arr[0], nlist, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (mympi->my_rank > 0) {
            kslist.clear();
            for (i = 0; i < nlist; ++i)
                kslist.push_back(kslist_arr[i]);
        }
        deallocate(kslist_arr);
    }

    if (ks_analyze_mode) {
        if (kpoint->kpoint_mode == 2 && anharmonic_core->use_triplet_symmetry) {
            anharmonic_core->use_triplet_symmetry = false;
            if (mympi->my_rank == 0) {
                std::cout << "\n TRISYM was automatically set to 0.\n\n";
            }
        }

        if (anharmonic_core->quartic_mode > 0) {
            // This is for quartic vertexes.

            if (mympi->my_rank == 0) {
                std::cout << " QUARTIC = 1 : Frequency shift due to the loop diagram associated with\n";
                std::cout << "               quartic anharmonicity will be calculated.\n";
                std::cout << "               Please check the accuracy of the quartic IFCs \n";
                std::cout << "               before doing serious calculations.\n\n";
            }

            if (kpoint->kpoint_mode == 2 && anharmonic_core->use_quartet_symmetry) {
                anharmonic_core->use_quartet_symmetry = false;
                if (mympi->my_rank == 0) {
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

        if (calc_fstate_k && kpoint->kpoint_mode != 3) {
            exit("setup_mode_analysis", "KPMODE should be 3 when FSTATE_K = 1.");
        }
        if (!calc_fstate_k && kpoint->kpoint_mode == 3) {
            exit("setup_mode_analysis", "KPMODE = 3 works only when FSTATE_K = 1");
        }

        if (calc_fstate_k && (calc_fstate_omega || (print_V3 > 0) || spectral_func || calc_realpart)) {
            warn("setup_mode_analysis",
                 "FSTATE_K = 1 shouldn't be set with the followings: PRINTV3=1, REALPART=1, FSTATE_W=1, SELF_W=1");
        }

        dynamical->modify_eigenvectors();
    }
}

void ModeAnalysis::run_mode_analysis()
{
    const auto Tmax = system->Tmax;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    double *T_arr;

    unsigned int NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;
    allocate(T_arr, NT);
    for (unsigned int i = 0; i < NT; ++i)
        T_arr[i] = Tmin + static_cast<double>(i) * dT;

    const auto epsilon = integration->epsilon;

    if (calc_fstate_k) {

        // Momentum-resolved final state amplitude
        print_momentum_resolved_final_state(NT, T_arr, epsilon);

    } else {

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
    }

    deallocate(T_arr);
}

void ModeAnalysis::print_selfenergy(const unsigned int NT, double *T_arr)
{
    auto ns = dynamical->neval;
    int j;

    std::ofstream ofs_linewidth, ofs_shift;

    double *damping_a = nullptr;
    std::complex<double> *self_tadpole = nullptr;
    std::complex<double> *self_a = nullptr;
    std::complex<double> *self_b = nullptr;
    std::complex<double> *self_c = nullptr;
    std::complex<double> *self_d = nullptr;
    std::complex<double> *self_e = nullptr;
    std::complex<double> *self_f = nullptr;
    std::complex<double> *self_g = nullptr;
    std::complex<double> *self_h = nullptr;
    std::complex<double> *self_i = nullptr;
    std::complex<double> *self_j = nullptr;

    if (mympi->my_rank == 0) {
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

    allocate(damping_a, NT);
    allocate(self_a, NT);
    allocate(self_b, NT);
    allocate(self_tadpole, NT);

    if (anharmonic_core->quartic_mode == 2) {
        allocate(self_c, NT);
        allocate(self_d, NT);
        allocate(self_e, NT);
        allocate(self_f, NT);
        allocate(self_g, NT);
        allocate(self_h, NT);
        allocate(self_i, NT);
        allocate(self_j, NT);
    }

    for (int i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];

        if (mympi->my_rank == 0) {
            std::cout << "\n Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
        }

        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        if (integration->ismear == -1) {
            anharmonic_core->calc_damping_tetrahedron(NT,
                                                      T_arr,
                                                      omega,
                                                      ik_irred,
                                                      snum,
                                                      dos->kmesh_dos,
                                                      dos->dymat_dos->get_eigenvalues(),
                                                      dos->dymat_dos->get_eigenvectors(),
                                                      damping_a);
        } else {
            selfenergy->selfenergy_a(NT,
                                     T_arr,
                                     omega,
                                     knum,
                                     snum,
                                     dos->kmesh_dos,
                                     dos->dymat_dos->get_eigenvalues(),
                                     dos->dymat_dos->get_eigenvectors(),
                                     self_a);
            for (j = 0; j < NT; ++j)
                damping_a[j] = self_a[j].imag();
        }
        if (anharmonic_core->quartic_mode == 2) {
            selfenergy->selfenergy_c(NT,
                                     T_arr,
                                     omega,
                                     knum,
                                     snum,
                                     dos->kmesh_dos,
                                     dos->dymat_dos->get_eigenvalues(),
                                     dos->dymat_dos->get_eigenvectors(),
                                     self_c);
            //            selfenergy->selfenergy_d(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_d);
            //            selfenergy->selfenergy_e(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_e);
            //            selfenergy->selfenergy_f(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_f);
            //            selfenergy->selfenergy_g(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_g);
            //            selfenergy->selfenergy_h(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_h);
            //            selfenergy->selfenergy_i(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_i);
            //            selfenergy->selfenergy_j(NT, T_arr, omega, knum, snum,
            //                                     dos->kmesh_dos,
            //                                     dos->dymat_dos->get_eigenvalues(),
            //                                     dos->dymat_dos->get_eigenvectors(),
            //                                     self_j);
        }

        if (mympi->my_rank == 0) {
            auto file_linewidth = input->job_title + ".Gamma." + std::to_string(i + 1);
            ofs_linewidth.open(file_linewidth.c_str(), std::ios::out);
            if (!ofs_linewidth) exit("print_selfenergy", "Cannot open file file_linewidth");

            ofs_linewidth << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_linewidth << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_linewidth << '\n';
            ofs_linewidth << "# mode = " << snum + 1 << '\n';
            ofs_linewidth << "# Frequency = " << writes->in_kayser(omega) << '\n';
            ofs_linewidth << "## Temperature dependence of 2*Gamma (FWHM) for the given mode\n";
            ofs_linewidth << "## T[K], 2*Gamma3 (cm^-1) (bubble)";
            if (anharmonic_core->quartic_mode == 2) ofs_linewidth << ", 2*Gamma4(cm^-1) <-- specific diagram only";
            ofs_linewidth << '\n';

            for (j = 0; j < NT; ++j) {
                ofs_linewidth << std::setw(10) << T_arr[j] << std::setw(15) << writes->in_kayser(2.0 * damping_a[j]);

                if (anharmonic_core->quartic_mode == 2) {
                    //							ofs_mode_tau << std::setw(15) << writes->in_kayser(damp4[j]);
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_c[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_d[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_e[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_f[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_g[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_h[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_i[j].imag());
                    ofs_linewidth << std::setw(15) << writes->in_kayser(2.0 * self_j[j].imag());
                }

                ofs_linewidth << '\n';
            }
            ofs_linewidth.close();
            std::cout << "  Phonon line-width is printed in " << file_linewidth << '\n';
        }

        if (calc_realpart) {
            selfenergy->selfenergy_tadpole(NT,
                                           T_arr,
                                           omega,
                                           knum,
                                           snum,
                                           dos->kmesh_dos,
                                           dos->dymat_dos->get_eigenvalues(),
                                           dos->dymat_dos->get_eigenvectors(),
                                           self_tadpole);
            //                selfenergy->selfenergy_a(NT, T_arr, omega, knum, snum, self_a);

            if (anharmonic_core->quartic_mode == 1) {
                selfenergy->selfenergy_b(NT,
                                         T_arr,
                                         omega,
                                         knum,
                                         snum,
                                         dos->kmesh_dos,
                                         dos->dymat_dos->get_eigenvalues(),
                                         dos->dymat_dos->get_eigenvectors(),
                                         self_b);
            }

            if (mympi->my_rank == 0) {
                auto file_shift = input->job_title + ".Shift." + std::to_string(i + 1);
                ofs_shift.open(file_shift.c_str(), std::ios::out);
                if (!ofs_shift) exit("print_selfenergy", "Cannot open file file_shift");

                ofs_shift << "# xk = ";

                for (j = 0; j < 3; ++j) {
                    ofs_shift << std::setw(15) << dos->kmesh_dos->xk[knum][j];
                }
                ofs_shift << '\n';
                ofs_shift << "# mode = " << snum + 1 << '\n';
                ofs_shift << "# Frequency = " << writes->in_kayser(omega) << '\n';
                ofs_shift << "## T[K], Shift3 (cm^-1) (tadpole), Shift3 (cm^-1) (bubble)";
                if (anharmonic_core->quartic_mode == 1) ofs_shift << ", Shift4 (cm^-1) (loop)";
                ofs_shift << ", Shifted frequency (cm^-1)\n";

                for (j = 0; j < NT; ++j) {
                    ofs_shift << std::setw(10) << T_arr[j];
                    ofs_shift << std::setw(15) << writes->in_kayser(-self_tadpole[j].real());
                    ofs_shift << std::setw(15) << writes->in_kayser(-self_a[j].real());

                    auto omega_shift = omega - self_tadpole[j].real() - self_a[j].real();

                    if (anharmonic_core->quartic_mode == 1) {
                        ofs_shift << std::setw(15) << writes->in_kayser(-self_b[j].real());
                        omega_shift -= self_b[j].real();
                    }
                    ofs_shift << std::setw(15) << writes->in_kayser(omega_shift);
                    ofs_shift << '\n';
                }

                ofs_shift.close();
                std::cout << "  Phonon frequency shift is printed in " << file_shift << '\n';
            }
        }
    }

    if (damping_a) {
        deallocate(damping_a);
    }
    if (self_tadpole) {
        deallocate(self_tadpole);
    }
    if (self_a) {
        deallocate(self_a);
    }
    if (self_b) {
        deallocate(self_b);
    }
    if (self_c) {
        deallocate(self_c);
    }
    if (self_d) {
        deallocate(self_d);
    }
    if (self_e) {
        deallocate(self_e);
    }
    if (self_f) {
        deallocate(self_f);
    }
    if (self_g) {
        deallocate(self_g);
    }
    if (self_h) {
        deallocate(self_h);
    }
    if (self_i) {
        deallocate(self_i);
    }
    if (self_j) {
        deallocate(self_j);
    }
}

void ModeAnalysis::print_frequency_resolved_final_state(const unsigned int NT, double *T_arr)
{
    int i, j;
    double ***gamma_final;
    double *freq_array;
    std::ofstream ofs_omega;
    const auto ns = dynamical->neval;

    allocate(gamma_final, NT, dos->n_energy, 2);
    allocate(freq_array, dos->n_energy);

    for (i = 0; i < dos->n_energy; ++i) {
        freq_array[i] = dos->energy_dos[i] * time_ry / Hz_to_kayser;
    }

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " FSTATE_W = 1 : Calculate the frequency-resolved final state amplitude\n";
        std::cout << "                due to 3-phonon interactions.\n";
    }

    for (i = 0; i < kslist.size(); ++i) {
        const auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;

        const auto omega0 = dos->dymat_dos->get_eigenvalues()[knum][snum];

        if (mympi->my_rank == 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega0) << '\n';
        }

        if (integration->ismear == -1) {
            calc_frequency_resolved_final_state_tetrahedron(NT,
                                                            T_arr,
                                                            omega0,
                                                            dos->n_energy,
                                                            freq_array,
                                                            dos->kmesh_dos->kmap_to_irreducible[knum],
                                                            snum,
                                                            dos->kmesh_dos,
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
                                                dos->kmesh_dos,
                                                dos->dymat_dos->get_eigenvalues(),
                                                dos->dymat_dos->get_eigenvectors(),
                                                gamma_final);
        }

        if (mympi->my_rank == 0) {
            std::string file_omega = input->job_title + ".fw." + std::to_string(i + 1);
            ofs_omega.open(file_omega.c_str(), std::ios::out);
            if (!ofs_omega) exit("print_frequency_resolved_final_state", "Cannot open file file_omega");

            ofs_omega << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_omega << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_omega << '\n';
            ofs_omega << "# mode = " << snum << '\n';
            ofs_omega << "# Frequency = " << writes->in_kayser(omega0) << '\n';

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
            std::cout << "  Frequency-resolved final state amplitude is printed in " << file_omega << '\n';
        }
    }

    deallocate(freq_array);
    deallocate(gamma_final);
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
    double ***ret_mpi;
    const auto nk = kmesh_in->nk;
    const auto ns = dynamical->neval;

    const auto epsilon = integration->epsilon;

    std::vector<KsListGroup> triplet;

    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, anharmonic_core->use_triplet_symmetry, false, triplet);

    allocate(ret_mpi, ntemp, nomegas, 2);

    for (i = 0; i < ntemp; ++i) {
        for (j = 0; j < nomegas; ++j) {
            ret_mpi[i][j][0] = 0.0;
            ret_mpi[i][j][1] = 0.0;
        }
    }

    for (int ik = mympi->my_rank; ik < triplet.size(); ik += mympi->nprocs) {
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

    deallocate(ret_mpi);
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

    unsigned int *kmap_identity;
    double **energy_tmp;
    double **weight_tetra;
    double **v3_arr;
    double ***delta_arr;
    double prod_tmp[2];

    const auto epsilon = integration->epsilon;

    const auto xk = kmesh_in->xk;

    std::vector<KsListGroup> triplet;

    kmesh_in->get_unique_triplet_k(ik_in, symmetry->SymmList, anharmonic_core->use_triplet_symmetry, false, triplet);

    for (i = 0; i < ntemp; ++i) {
        for (j = 0; j < nomegas; ++j) {
            ret[i][j][0] = 0.0;
            ret[i][j][1] = 0.0;
        }
    }

    const auto npair_uniq = triplet.size();

    allocate(v3_arr, npair_uniq, ns2);
    allocate(delta_arr, npair_uniq, ns2, 2);

    const auto knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
    const auto knum_minus = kmesh_in->kindex_minus_xk[knum];

    allocate(kmap_identity, nk);

    for (i = 0; i < nk; ++i)
        kmap_identity[i] = i;

#ifdef _OPENMP
#pragma omp parallel private(is, js, k1, k2, xk_tmp, energy_tmp, i, weight_tetra, ik, jk, arr)
#endif
    {
        allocate(energy_tmp, 3, nk);
        allocate(weight_tetra, 3, nk);

#ifdef _OPENMP
#pragma omp for
#endif
        for (int ib = 0; ib < ns2; ++ib) {
            is = ib / ns;
            js = ib % ns;

            for (k1 = 0; k1 < nk; ++k1) {
                // Prepare two-phonon frequency for the tetrahedron method

                for (i = 0; i < 3; ++i)
                    xk_tmp[i] = xk[knum][i] - xk[k1][i];

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

                    v3_arr[ik][ib] = std::norm(anharmonic_core->V3(arr));
                } else {
                    v3_arr[ik][ib] = 0.0;
                }
            }
        }

        deallocate(energy_tmp);
        deallocate(weight_tetra);
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

    deallocate(v3_arr);
    deallocate(delta_arr);
    deallocate(kmap_identity);

    triplet.clear();
}

void ModeAnalysis::print_momentum_resolved_final_state(const unsigned int NT, double *T_arr, double epsilon)
{
    int i, j, k, l;
    int iT;
    int is, js;
    int mode;
    double xk1[3], xk2[3], xk3[3];
    double kvec[3];
    double f1, f2, n1, n2;
    double norm;
    double **eval, **eval2;
    double **gamma_k;
    const auto ns = dynamical->neval;

    std::complex<double> ***evec;

    std::ofstream ofs_mode_tau;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        if (integration->ismear == -1) {
            std::cout << " ISMEAR = -1: Tetrahedron method will be used.\n";
            std::cout << " Sorry. Currently, ISMEAR = -1 cannot be used with FSTATE_K = 1.";
            exit("calc_momentum_resolved_final_state", "exit.");
        } else if (integration->ismear == 0) {
            std::cout << " ISMEAR = 0: Lorentzian broadening with epsilon = " << std::fixed << std::setprecision(2)
                      << epsilon << " (cm^-1)\n";
        } else if (integration->ismear == 1) {
            std::cout << " ISMEAR = 1: Gaussian broadening with epsilon = " << std::fixed << std::setprecision(2)
                      << epsilon << " (cm^-1)\n";
        } else {
            exit("print_momentum_resolved_final_state", "Invalid ISMEAR");
        }
        std::cout << '\n';
        std::cout << " FSTATE_K = 1 : Calculate the momentum-resolved final state amplitude\n";
        std::cout << "                due to 3-phonon interactions for given " << kslist_fstate_k.size()
                  << " entries.\n\n";
    }

    double **xk_plane, **xk_plane2;
    double **kvec_plane;
    double xk_vec1[3], xk_vec2[3];
    double *eval_tmp;
    double omega_sum[3];
    double frac;
    int knum_triangle[3];
    std::vector<std::vector<double>> ***kplist_conserved;
    std::vector<KpointListWithCoordinate> ***kplist_for_target_mode;
    std::vector<double> xk_vec;
    double xk_norm[3], xk_tmp[3];
    double norm1, dprod;
    double theta;

    allocate(kplist_conserved, ns, ns, 2);
    allocate(kplist_for_target_mode, ns, ns, kslist_fstate_k.size());

    double theta_ref = 0.0;

    // Loop over k point planes

    for (i = 0; i < kpoint->kp_plane_geometry.size(); ++i) {
        const auto nk1_plane = kpoint->kp_plane_geometry[i].npoints[0];
        const auto nk2_plane = kpoint->kp_plane_geometry[i].npoints[1];

        const auto div1 = 1.0 / static_cast<double>(nk1_plane - 1);
        const auto div2 = 1.0 / static_cast<double>(nk2_plane - 1);

        const auto nk_plane = nk1_plane * nk2_plane;

        for (j = 0; j < 3; ++j) {
            xk_vec1[j] = kpoint->kp_plane_geometry[i].xk_edges[0][j] - kpoint->kp_plane_geometry[i].xk_origin[j];
            xk_vec2[j] = kpoint->kp_plane_geometry[i].xk_edges[1][j] - kpoint->kp_plane_geometry[i].xk_origin[j];
        }

        for (j = 0; j < 3; ++j) {
            xk_norm[j] = xk_vec1[j];
        }

        rotvec(xk_norm, xk_norm, system->get_primcell().reciprocal_lattice_vector, 'T');
        const auto norm_ref = std::sqrt(xk_norm[0] * xk_norm[0] + xk_norm[1] * xk_norm[1] + xk_norm[2] * xk_norm[2]);

        allocate(xk_plane, nk_plane, 3);
        allocate(xk_plane2, nk_plane, 3);
        allocate(kvec_plane, nk_plane, 3);
        allocate(eval, nk_plane, ns);
        allocate(eval2, nk_plane, ns);
        allocate(eval_tmp, ns);
        allocate(evec, 1, 1, 1);

        // Constructing xk's for the plane
        int m = 0;
        for (j = 0; j < nk1_plane; ++j) {
            for (k = 0; k < nk2_plane; ++k) {
                for (l = 0; l < 3; ++l) {
                    xk_plane[m][l] = kpoint->kp_plane_geometry[i].xk_origin[l] +
                                     xk_vec1[l] * static_cast<double>(j) * div1 +
                                     xk_vec2[l] * static_cast<double>(k) * div2;
                }
                ++m;
            }
        }

        // Get frequencies of each k point

        for (j = 0; j < nk_plane; ++j) {
            for (k = 0; k < 3; ++k)
                kvec_plane[j][k] = dynamical->fold(xk_plane[j][k]);
            rotvec(kvec_plane[j], kvec_plane[j], system->get_primcell().reciprocal_lattice_vector, 'T');
            norm = std::sqrt(kvec_plane[j][0] * kvec_plane[j][0] + kvec_plane[j][1] * kvec_plane[j][1] +
                             kvec_plane[j][2] * kvec_plane[j][2]);

            if (norm > eps) {
                for (k = 0; k < 3; ++k)
                    kvec_plane[j][k] /= norm;
            }
        }

        for (j = 0; j < nk_plane; ++j) {
            dynamical
                ->eval_k(xk_plane[j], kvec_plane[j], fcs_phonon->force_constant_with_cell[0], eval[j], evec[0], false);
            for (k = 0; k < ns; ++k) {
                eval[j][k] = dynamical->freq(eval[j][k]);
            }
        }

        // Loop over k points to analyze the final state amplitude

        for (j = 0; j < kslist_fstate_k.size(); ++j) {
            for (k = 0; k < 3; ++k)
                xk1[k] = kslist_fstate_k[j].xk[k];
            mode = kslist_fstate_k[j].nmode;

            for (k = 0; k < 3; ++k)
                kvec[k] = kslist_fstate_k[j].xk[k];
            rotvec(kvec, kvec, system->get_primcell().reciprocal_lattice_vector, 'T');
            norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);

            if (norm > eps) {
                for (k = 0; k < 3; ++k)
                    kvec[k] /= norm;
            }

            // for (k = 0; k < 3; ++k) xk1[k] = dynamical->fold(xk1[k]);

            dynamical->eval_k(xk1, kvec, fcs_phonon->force_constant_with_cell[0], eval_tmp, evec[0], false);
            for (k = 0; k < ns; ++k)
                eval_tmp[k] = dynamical->freq(eval_tmp[k]);

            // Calculate xk's for the third index that satisfy the momentum conservation

            for (k = 0; k < nk_plane; ++k) {
                for (l = 0; l < 3; ++l) {
                    xk_plane2[k][l] = dynamical->fold(xk1[l] - xk_plane[k][l]);
                }
            }

            // Get frequencies of each k point

            for (k = 0; k < nk_plane; ++k) {
                for (l = 0; l < 3; ++l)
                    kvec_plane[k][l] = xk_plane2[k][l];
                rotvec(kvec_plane[k], kvec_plane[k], system->get_primcell().reciprocal_lattice_vector, 'T');
                norm = std::sqrt(kvec_plane[k][0] * kvec_plane[k][0] + kvec_plane[k][1] * kvec_plane[k][1] +
                                 kvec_plane[k][2] * kvec_plane[k][2]);

                if (norm > eps) {
                    for (l = 0; l < 3; ++l)
                        kvec_plane[k][l] /= norm;
                }
            }

            for (k = 0; k < nk_plane; ++k) {
                dynamical->eval_k(xk_plane2[k],
                                  kvec_plane[k],
                                  fcs_phonon->force_constant_with_cell[0],
                                  eval2[k],
                                  evec[0],
                                  false);
                for (l = 0; l < ns; ++l) {
                    eval2[k][l] = dynamical->freq(eval2[k][l]);
                }
            }

            // Find a list of k points which satisfy the energy conservation

            for (const auto &it: kpoint->kp_planes_tri[i]) {
                // K point indexes for each triangle
                for (k = 0; k < 3; ++k)
                    knum_triangle[k] = it.knum[k];

                for (is = 0; is < ns; ++is) {
                    for (js = 0; js < ns; ++js) {
                        // The case of delta(w1 - w2 - w3)

                        //     std::cout << "is = " << is << " js = " << js << '\n';
                        for (k = 0; k < 3; ++k) {
                            omega_sum[k] = eval_tmp[mode] - eval[knum_triangle[k]][is] - eval2[knum_triangle[k]][js];
                        }
                        if ((omega_sum[0] > 0.0 && omega_sum[1] > 0.0 && omega_sum[2] > 0.0) ||
                            (omega_sum[0] < 0.0 && omega_sum[1] < 0.0 && omega_sum[2] < 0.0))
                            continue;

                        if (omega_sum[0] * omega_sum[1] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[0] / (omega_sum[1] - omega_sum[0]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[0]][k] +
                                                 frac * xk_plane[knum_triangle[1]][k]);
                            }
                            kplist_conserved[is][js][0].push_back(xk_vec);
                        }

                        if (omega_sum[0] * omega_sum[2] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[0] / (omega_sum[2] - omega_sum[0]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[0]][k] +
                                                 frac * xk_plane[knum_triangle[2]][k]);
                            }
                            kplist_conserved[is][js][0].push_back(xk_vec);
                        }

                        if (omega_sum[1] * omega_sum[2] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[1] / (omega_sum[2] - omega_sum[1]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[1]][k] +
                                                 frac * xk_plane[knum_triangle[2]][k]);
                            }
                            kplist_conserved[is][js][0].push_back(xk_vec);
                        }

                        // The case of delta(w1 - w2 + w3)

                        for (k = 0; k < 3; ++k) {
                            omega_sum[k] = eval_tmp[mode] - eval[knum_triangle[k]][is] + eval2[knum_triangle[k]][js];
                        }
                        if ((omega_sum[0] > 0.0 && omega_sum[1] > 0.0 && omega_sum[2] > 0.0) ||
                            (omega_sum[0] < 0.0 && omega_sum[1] < 0.0 && omega_sum[2] < 0.0))
                            continue;

                        if (omega_sum[0] * omega_sum[1] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[0] / (omega_sum[1] - omega_sum[0]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[0]][k] +
                                                 frac * xk_plane[knum_triangle[1]][k]);
                            }
                            kplist_conserved[is][js][1].push_back(xk_vec);
                        }

                        if (omega_sum[0] * omega_sum[2] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[0] / (omega_sum[2] - omega_sum[0]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[0]][k] +
                                                 frac * xk_plane[knum_triangle[2]][k]);
                            }
                            kplist_conserved[is][js][1].push_back(xk_vec);
                        }

                        if (omega_sum[1] * omega_sum[2] < 0.0) {
                            xk_vec.clear();

                            frac = -omega_sum[1] / (omega_sum[2] - omega_sum[1]);

                            for (k = 0; k < 3; ++k) {
                                xk_vec.push_back((1.0 - frac) * xk_plane[knum_triangle[1]][k] +
                                                 frac * xk_plane[knum_triangle[2]][k]);
                            }
                            kplist_conserved[is][js][1].push_back(xk_vec);
                        }
                    }
                }
            }

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    for (auto it2 = kplist_conserved[is][js][0].begin(); it2 != kplist_conserved[is][js][0].end();
                         ++it2)
                    {
                        for (k = 0; k < 3; ++k) {
                            xk_tmp[k] = (*it2)[k];
                        }
                        rotvec(xk_tmp, xk_tmp, system->get_primcell().reciprocal_lattice_vector, 'T');

                        norm1 = 0.0;
                        dprod = 0.0;
                        for (k = 0; k < 3; ++k) {
                            norm1 += xk_tmp[k] * xk_tmp[k];
                            dprod += xk_tmp[k] * xk_norm[k];
                        }
                        theta = std::acos(dprod / (norm_ref * std::sqrt(norm1)));

                        kplist_for_target_mode[is][js][j].emplace_back(*it2,
                                                                       std::cos(theta + theta_ref) * std::sqrt(norm1),
                                                                       std::sin(theta + theta_ref) * std::sqrt(norm1),
                                                                       i,
                                                                       0);
                    }

                    for (auto it2 = kplist_conserved[is][js][1].begin(); it2 != kplist_conserved[is][js][1].end();
                         ++it2)
                    {
                        for (k = 0; k < 3; ++k) {
                            xk_tmp[k] = (*it2)[k];
                        }
                        rotvec(xk_tmp, xk_tmp, system->get_primcell().reciprocal_lattice_vector, 'T');

                        norm1 = 0.0;
                        dprod = 0.0;
                        for (k = 0; k < 3; ++k) {
                            norm1 += xk_tmp[k] * xk_tmp[k];
                            dprod += xk_tmp[k] * xk_norm[k];
                        }
                        theta = std::acos(dprod / (norm_ref * std::sqrt(norm1)));

                        kplist_for_target_mode[is][js][j].emplace_back(*it2,
                                                                       std::cos(theta + theta_ref) * std::sqrt(norm1),
                                                                       std::sin(theta + theta_ref) * std::sqrt(norm1),
                                                                       i,
                                                                       1);
                    }

                    kplist_conserved[is][js][0].clear();
                    kplist_conserved[is][js][1].clear();
                }
            }
        }

        deallocate(xk_plane);
        deallocate(xk_plane2);
        deallocate(kvec_plane);
        deallocate(eval);
        deallocate(eval2);
        deallocate(eval_tmp);
        deallocate(evec);

        rotvec(xk_vec1, xk_vec1, system->get_primcell().reciprocal_lattice_vector, 'T');
        rotvec(xk_vec2, xk_vec2, system->get_primcell().reciprocal_lattice_vector, 'T');

        norm1 = 0.0;
        double norm2 = 0.0;
        dprod = 0.0;
        for (j = 0; j < 3; ++j) {
            norm1 += xk_vec1[j] * xk_vec1[j];
            norm2 += xk_vec2[j] * xk_vec2[j];
            dprod += xk_vec1[j] * xk_vec2[j];
        }
        theta = std::acos(dprod / std::sqrt(norm1 * norm2));

        theta_ref += theta;
    }

    deallocate(kplist_conserved);

    std::vector<std::vector<double>> **final_state_xy;
    std::vector<double> triplet_xyG;
    std::vector<int> small_group_k;
    int selection_type;

    int isym;

    double srot[3][3];
    double xk_sym[3];
    double srot_inv[3][3], srot_inv_t[3][3];
    double ***symop_k;

    allocate(symop_k, symmetry->nsym, 3, 3);
    allocate(final_state_xy, kslist_fstate_k.size(), NT);

    allocate(eval, 3, ns);
    allocate(evec, 3, ns, ns);

    for (i = 0; i < kslist_fstate_k.size(); ++i) {
        for (j = 0; j < 3; ++j)
            xk1[j] = -kslist_fstate_k[i].xk[j];
        mode = kslist_fstate_k[i].nmode;

        for (j = 0; j < 3; ++j)
            kvec[j] = dynamical->fold(xk1[j]);
        rotvec(kvec, kvec, system->get_primcell().reciprocal_lattice_vector, 'T');
        norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);

        if (norm > eps)
            for (j = 0; j < 3; ++j)
                kvec[j] /= norm;
        for (j = 0; j < 3; ++j)
            xk1[j] = dynamical->fold(xk1[j]);

        dynamical->eval_k(xk1, kvec, fcs_phonon->force_constant_with_cell[0], eval[0], evec[0], true);
        for (j = 0; j < ns; ++j)
            eval[0][j] = dynamical->freq(eval[0][j]);

        small_group_k.clear();

        for (isym = 0; isym < symmetry->nsym; ++isym) {
            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    srot[j][k] = static_cast<double>(symmetry->SymmList[isym].rotation(j, k));
                }
            }

            invmat3(srot_inv, srot);
            transpose3(srot_inv_t, srot_inv);

            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    symop_k[isym][j][k] = srot_inv_t[j][k];
                }
            }

            rotvec(xk_sym, xk1, symop_k[isym]);

            for (j = 0; j < 3; ++j) {
                xk_sym[j] = xk_sym[j] - nint(xk_sym[j]);
                xk_sym[j] = std::fmod(xk_sym[j], 1.0);
            }

            double diff = 0.0;
            for (j = 0; j < 3; ++j) {
                diff += std::pow(std::fmod(xk_sym[j] - xk1[j], 1.0), 2);
            }
            if (std::sqrt(diff) < eps8) {
                small_group_k.push_back(isym);
            }
        }

        if (mympi->my_rank == 0) {
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << kslist_fstate_k[i].xk[j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << mode + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(eval[0][mode]) << '\n';

            int count_kp = 0;

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    count_kp += kplist_for_target_mode[is][js][i].size();
                }
            }
            std::cout << "  Number of k points satisfying the selection rule : " << count_kp << '\n';
            std::cout << "  Number of symmetry operations at k point : " << small_group_k.size() << '\n' << '\n';
        }

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                const int nklist = kplist_for_target_mode[is][js][i].size();

                if (nklist == 0) continue;

                allocate(gamma_k, nklist, NT);

                for (k = 0; k < nklist; ++k) {
                    for (l = 0; l < NT; ++l) {
                        gamma_k[k][l] = 0.0;
                    }
                }

                for (k = 0; k < nklist; ++k) {
                    for (l = 0; l < 3; ++l)
                        xk2[l] = dynamical->fold(kplist_for_target_mode[is][js][i][k].xk[l]);

                    for (isym = 0; isym < small_group_k.size(); ++isym) {
                        rotvec(xk_sym, xk2, symop_k[small_group_k[isym]]);

                        for (l = 0; l < 3; ++l)
                            xk3[l] = dynamical->fold(-xk1[l] - xk_sym[l]);

                        for (l = 0; l < 3; ++l)
                            kvec[l] = xk_sym[l];
                        rotvec(kvec, kvec, system->get_primcell().reciprocal_lattice_vector, 'T');
                        norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);

                        if (norm > eps)
                            for (l = 0; l < 3; ++l)
                                kvec[l] /= norm;

                        dynamical
                            ->eval_k(xk_sym, kvec, fcs_phonon->force_constant_with_cell[0], eval[1], evec[1], true);

                        for (l = 0; l < 3; ++l)
                            kvec[l] = xk3[l];
                        rotvec(kvec, kvec, system->get_primcell().reciprocal_lattice_vector, 'T');
                        norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);

                        if (norm > eps)
                            for (l = 0; l < 3; ++l)
                                kvec[l] /= norm;

                        dynamical->eval_k(xk3, kvec, fcs_phonon->force_constant_with_cell[0], eval[2], evec[2], true);

                        for (l = 0; l < ns; ++l) {
                            eval[1][l] = dynamical->freq(eval[1][l]);
                            eval[2][l] = dynamical->freq(eval[2][l]);
                        }

                        const auto V3norm = std::norm(anharmonic_core->V3_mode(mode, xk_sym, xk3, is, js, eval, evec));

                        for (iT = 0; iT < NT; ++iT) {
                            const auto T_tmp = T_arr[iT];

                            if (thermodynamics->classical) {
                                f1 = thermodynamics->fC(eval[1][is], T_tmp);
                                f2 = thermodynamics->fC(eval[2][js], T_tmp);
                                n1 = f1 + f2;
                                n2 = f1 - f2;
                            } else {
                                f1 = thermodynamics->fB(eval[1][is], T_tmp);
                                f2 = thermodynamics->fB(eval[2][js], T_tmp);
                                n1 = f1 + f2 + 1.0;
                                n2 = f1 - f2;
                            }

                            if (selection_type == 0) {
                                gamma_k[k][iT] += V3norm * n1;
                            } else if (selection_type == 1) {
                                gamma_k[k][iT] += V3norm * n2;
                            }
                        }
                    }

                    for (iT = 0; iT < NT; ++iT)
                        gamma_k[k][iT] *= pi * std::pow(0.5, 4) / static_cast<double>(small_group_k.size());

                    auto pos_x = kplist_for_target_mode[is][js][i][k].x;
                    auto pos_y = kplist_for_target_mode[is][js][i][k].y;
                    selection_type = kplist_for_target_mode[is][js][i][k].selection_type;

                    for (iT = 0; iT < NT; ++iT) {
                        triplet_xyG.clear();
                        triplet_xyG.push_back(pos_x);
                        triplet_xyG.push_back(pos_y);
                        triplet_xyG.push_back(gamma_k[k][iT]);
                        final_state_xy[i][iT].push_back(triplet_xyG);
                    }
                }
                deallocate(gamma_k);
            }
        }

        if (mympi->my_rank == 0) {
            auto file_mode_tau = input->job_title + ".fk." + std::to_string(i + 1);
            ofs_mode_tau.open(file_mode_tau.c_str(), std::ios::out);
            if (!ofs_mode_tau) exit("compute_mode_tau", "Cannot open file file_mode_tau");

            ofs_mode_tau << "## Momentum-resolved final state amplitude\n";

            ofs_mode_tau << "# " << "Gamma at ";
            for (l = 0; l < 3; ++l)
                ofs_mode_tau << std::setw(10) << -xk1[l];
            ofs_mode_tau << " , mode = " << mode + 1 << '\n';
            ofs_mode_tau << " # Temperature [K], coordinate in FBZ, final state amplitude\n";
            for (iT = 0; iT < NT; ++iT) {
                for (k = 0; k < final_state_xy[i][iT].size(); ++k) {
                    ofs_mode_tau << std::setw(10) << T_arr[iT];
                    ofs_mode_tau << std::setw(15) << final_state_xy[i][iT][k][0];
                    ofs_mode_tau << std::setw(15) << final_state_xy[i][iT][k][1];
                    ofs_mode_tau << std::setw(15) << final_state_xy[i][iT][k][2] << '\n';
                }
            }

            ofs_mode_tau.close();
            std::cout << "  The result is saved in " << file_mode_tau << "\n\n";
        }
    }

    deallocate(kplist_for_target_mode);
    deallocate(final_state_xy);
    deallocate(symop_k);
    deallocate(eval);
    deallocate(evec);
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

        if (mympi->my_rank == 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
        }

        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        dos->kmesh_dos->get_unique_triplet_k(ik_irred,
                                             symmetry->SymmList,
                                             anharmonic_core->use_triplet_symmetry,
                                             true,
                                             triplet);
        const auto nk_size = triplet.size();

        std::vector<std::vector<double>> v3norm(nk_size, std::vector<double>(ns * ns));

        calc_V3norm2(ik_irred, snum, triplet, v3norm);

        if (mympi->my_rank == 0) {
            auto file_V3 = input->job_title + ".V3." + std::to_string(i + 1);
            ofs_V3.open(file_V3.c_str(), std::ios::out);
            if (!ofs_V3) exit("run_mode_analysis", "Cannot open file file_V3");

            ofs_V3 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V3 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V3 << '\n';
            ofs_V3 << "# mode = " << snum + 1 << '\n';
            ofs_V3 << "# Frequency = " << writes->in_kayser(omega) << '\n';
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
                        ofs_V3 << std::setw(15) << writes->in_kayser(eval_tmp[k1][is]);
                        ofs_V3 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                        ofs_V3 << std::setw(15) << writes->in_kayser(eval_tmp[k2][js]);
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

        if (mympi->my_rank == 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
        }

        int ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        dos->kmesh_dos->get_unique_quartet_k(ik_irred,
                                             symmetry->SymmList,
                                             anharmonic_core->use_quartet_symmetry,
                                             true,
                                             quartet);
        const auto nk_size = quartet.size();

        std::vector<std::vector<double>> v4norm(nk_size, std::vector<double>(ns * ns * ns));

        calc_V4norm2(knum, snum, quartet, v4norm);

        if (mympi->my_rank == 0) {
            std::string file_V4 = input->job_title + ".V4." + std::to_string(i + 1);
            ofs_V4.open(file_V4.c_str(), std::ios::out);
            if (!ofs_V4) exit("run_mode_analysis", "Cannot open file file_V4");

            ofs_V4 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V4 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V4 << '\n';
            ofs_V4 << "# mode = " << snum + 1 << '\n';
            ofs_V4 << "# Frequency = " << writes->in_kayser(omega) << '\n';
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
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k1][is]);
                            ofs_V4 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k2][js]);
                            ofs_V4 << std::setw(5) << k3 + 1 << std::setw(5) << ks + 1;
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k3][ks]);
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
}

void ModeAnalysis::calc_V3norm2(const unsigned int ik_in, const unsigned int snum,
                                const std::vector<KsListGroup> &triplet, std::vector<std::vector<double>> &ret) const
{
    unsigned int is, js;
    unsigned int k1, k2;
    unsigned int arr[3];
    const auto ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const auto factor = std::pow(0.5, 3) * std::pow(Hz_to_kayser / time_ry, 2);

    const auto knum = dos->kmesh_dos->kpoint_irred_all[ik_in][0].knum;
    const auto knum_minus = dos->kmesh_dos->kindex_minus_xk[knum];
    const auto ntriplet = triplet.size();

    double **ret_loc = nullptr;
    double **ret_sum = nullptr;

    allocate(ret_loc, ntriplet, ns2);
    allocate(ret_sum, ntriplet, ns2);

    for (size_t ik = 0; ik < ntriplet; ++ik) {
        for (size_t ib = 0; ib < ns2; ++ib) {
            ret_loc[ik][ib] = 0.0;
            ret_sum[ik][ib] = 0.0;
        }
    }

    for (size_t ik = 0; ik < triplet.size(); ++ik) {
        k1 = triplet[ik].group[0].ks[0];
        k2 = triplet[ik].group[0].ks[1];

        for (size_t ib = mympi->my_rank; ib < ns2; ib += mympi->nprocs) {
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

    if (mympi->my_rank == 0) {
        for (size_t ik = 0; ik < ntriplet; ++ik) {
            for (size_t ib = 0; ib < ns2; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    deallocate(ret_loc);
    deallocate(ret_sum);
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

    const double factor = std::pow(0.5, 4) * std::pow(Hz_to_kayser / time_ry, 2);
    const auto nquartet = quartet.size();

    double **ret_loc = nullptr;
    double **ret_sum = nullptr;

    allocate(ret_loc, nquartet, ns3);
    allocate(ret_sum, nquartet, ns3);

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

        for (size_t ib = mympi->my_rank; ib < ns3; ib += mympi->nprocs) {
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

    if (mympi->my_rank == 0) {
        for (size_t ik = 0; ik < nquartet; ++ik) {
            for (size_t ib = 0; ib < ns3; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    deallocate(ret_loc);
    deallocate(ret_sum);
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

        if (mympi->my_rank == 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
        }

        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        dos->kmesh_dos->get_unique_triplet_k(ik_irred,
                                             symmetry->SymmList,
                                             anharmonic_core->use_triplet_symmetry,
                                             true,
                                             triplet,
                                             1);
        const auto nk_size = triplet.size();
        std::vector<std::vector<std::complex<double>>> phi3(nk_size, std::vector<std::complex<double>>(ns * ns));

        calc_Phi3(knum, snum, triplet, phi3);

        if (mympi->my_rank == 0) {
            auto file_V3 = input->job_title + ".Phi3." + std::to_string(i + 1);
            ofs_V3.open(file_V3.c_str(), std::ios::out);
            if (!ofs_V3) exit("print_phi3_element", "Cannot open file file_V3");

            ofs_V3 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V3 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V3 << '\n';
            ofs_V3 << "# mode = " << snum + 1 << '\n';
            ofs_V3 << "# Frequency = " << writes->in_kayser(omega) << '\n';
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
                        ofs_V3 << std::setw(15) << writes->in_kayser(eval_tmp[k1][is]);
                        ofs_V3 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                        ofs_V3 << std::setw(15) << writes->in_kayser(eval_tmp[k2][js]);
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

        if (mympi->my_rank == 0) {
            std::cout << '\n';
            std::cout << " Number : " << std::setw(5) << i + 1 << '\n';
            std::cout << "  Phonon at k = (";
            for (j = 0; j < 3; ++j) {
                std::cout << std::setw(10) << std::fixed << dos->kmesh_dos->xk[knum][j];
                if (j < 2) std::cout << ",";
            }
            std::cout << ")\n";
            std::cout << "  Mode index = " << std::setw(5) << snum + 1 << '\n';
            std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
        }

        int ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        dos->kmesh_dos->get_unique_quartet_k(ik_irred,
                                             symmetry->SymmList,
                                             anharmonic_core->use_quartet_symmetry,
                                             true,
                                             quartet,
                                             1);
        unsigned int nk_size = quartet.size();

        std::vector<std::vector<std::complex<double>>> phi4(nk_size, std::vector<std::complex<double>>(ns * ns * ns));
        calc_Phi4(knum, snum, quartet, phi4);

        if (mympi->my_rank == 0) {
            std::string file_V4 = input->job_title + ".Phi4." + std::to_string(i + 1);
            ofs_V4.open(file_V4.c_str(), std::ios::out);
            if (!ofs_V4) exit("print_phi4_element", "Cannot open file file_V3");

            ofs_V4 << "# xk = ";

            for (j = 0; j < 3; ++j) {
                ofs_V4 << std::setw(15) << dos->kmesh_dos->xk[knum][j];
            }
            ofs_V4 << '\n';
            ofs_V4 << "# mode = " << snum + 1 << '\n';
            ofs_V4 << "# Frequency = " << writes->in_kayser(omega) << '\n';
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
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k1][is]);
                            ofs_V4 << std::setw(5) << k2 + 1 << std::setw(5) << js + 1;
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k2][js]);
                            ofs_V4 << std::setw(5) << k3 + 1 << std::setw(5) << ks + 1;
                            ofs_V4 << std::setw(15) << writes->in_kayser(eval_tmp[k3][ks]);
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

    std::complex<double> **ret_loc = nullptr;
    std::complex<double> **ret_sum = nullptr;

    allocate(ret_loc, ntriplet, ns2);
    allocate(ret_sum, ntriplet, ns2);

    for (size_t ik = 0; ik < ntriplet; ++ik) {
        for (size_t ib = 0; ib < ns2; ++ib) {
            ret_loc[ik][ib] = std::complex<double>(0.0, 0.0);
            ret_sum[ik][ib] = std::complex<double>(0.0, 0.0);
        }
    }

    for (size_t ik = 0; ik < ntriplet; ++ik) {

        k1 = triplet[ik].group[0].ks[0];
        k2 = triplet[ik].group[0].ks[1];

        for (size_t ib = mympi->my_rank; ib < ns2; ib += mympi->nprocs) {

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

    if (mympi->my_rank == 0) {
        for (size_t ik = 0; ik < ntriplet; ++ik) {
            for (size_t ib = 0; ib < ns2; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    deallocate(ret_loc);
    deallocate(ret_sum);
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

    double factor = std::pow(amu_ry, 2.0);
    const auto nquartet = quartet.size();

    std::complex<double> **ret_loc = nullptr;
    std::complex<double> **ret_sum = nullptr;

    allocate(ret_loc, nquartet, ns3);
    allocate(ret_sum, nquartet, ns3);

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

        for (size_t ib = mympi->my_rank; ib < ns3; ib += mympi->nprocs) {
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

    if (mympi->my_rank == 0) {
        for (size_t ik = 0; ik < nquartet; ++ik) {
            for (size_t ib = 0; ib < ns3; ++ib) {
                ret[ik][ib] = ret_sum[ik][ib];
            }
        }
    }

    deallocate(ret_loc);
    deallocate(ret_sum);
}

void ModeAnalysis::print_spectral_function(const unsigned int NT, const double *T_arr)
{
    auto ns = dynamical->neval;
    int i, j;
    int iomega;
    double **self3_imag, **self3_real;
    std::ofstream ofs_self;
    double *omega_array;
    //    const auto Omega_min = dos->emin;
    //    const auto Omega_max = dos->emax;
    const auto delta_omega = dos->delta_e;
    double omega2[2];

    const double Omega_min = 0.0;
    double Omega_max;

    //    auto emin_now = std::numeric_limits<double>::max();
    auto emax_now = std::numeric_limits<double>::min();
    double omega_tmp;

    for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            omega_tmp =
                writes->in_kayser(dos->dymat_dos->get_eigenvalues()[dos->kmesh_dos->kpoint_irred_all[ik][0].knum][is]);
            //            emin_now = std::min(emin_now, omega_tmp);
            emax_now = std::max(emax_now, omega_tmp);
        }
    }
    Omega_max = (emax_now + delta_omega) * 2.0;


    const auto nomega = static_cast<unsigned int>((Omega_max - Omega_min) / delta_omega) + 1;

    allocate(omega_array, nomega);
    allocate(self3_imag, NT, nomega);
    allocate(self3_real, NT, nomega);

    for (i = 0; i < nomega; ++i) {
        omega_array[i] = Omega_min + delta_omega * static_cast<double>(i);
        omega_array[i] *= time_ry / Hz_to_kayser;
    }

    for (i = 0; i < kslist.size(); ++i) {
        auto knum = kslist[i] / ns;
        const auto snum = kslist[i] % ns;
        const auto ik_irred = dos->kmesh_dos->kmap_to_irreducible[knum];

        if (mympi->my_rank == 0) {
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

            std::string file_self = input->job_title + ".Self." + std::to_string(i + 1);
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

        for (int iT = 0; iT < NT; ++iT) {
            const auto T_now = T_arr[iT];
            const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];

            if (mympi->my_rank == 0) {
                std::cout << "  Temperature (K) : " << std::setw(15) << T_now << '\n';
                std::cout << "  Frequency (cm^-1) : " << std::setw(15) << writes->in_kayser(omega) << '\n';
            }

            anharmonic_core->calc_self3omega_tetrahedron(T_now,
                                                         dos->kmesh_dos,
                                                         dos->dymat_dos->get_eigenvalues(),
                                                         dos->dymat_dos->get_eigenvectors(),
                                                         ik_irred,
                                                         snum,
                                                         nomega,
                                                         omega_array,
                                                         self3_imag[iT]);

            // Calculate real part of the self-energy by Kramers-Kronig relation
            for (iomega = 0; iomega < nomega; ++iomega) {
                auto self_tmp = 0.0;
                omega2[0] = omega_array[iomega] * omega_array[iomega];
                for (int jomega = 0; jomega < nomega; ++jomega) {
                    if (jomega == iomega) continue;
                    omega2[1] = omega_array[jomega] * omega_array[jomega];
                    self_tmp += omega_array[jomega] * self3_imag[iT][jomega] / (omega2[1] - omega2[0]);
                }
                self3_real[iT][iomega] = 2.0 * delta_omega * time_ry * self_tmp / (pi * Hz_to_kayser);
            }

            if (mympi->my_rank == 0) {
                for (iomega = 0; iomega < nomega; ++iomega) {
                    ofs_self << std::setw(10) << T_now << std::setw(15) << writes->in_kayser(omega);
                    ofs_self << std::setw(10) << writes->in_kayser(omega_array[iomega]) << std::setw(15)
                             << writes->in_kayser(self3_real[iT][iomega]) << std::setw(15)
                             << writes->in_kayser(self3_imag[iT][iomega]) << '\n';
                }
                ofs_self << '\n';
            }
        }
        if (mympi->my_rank == 0) ofs_self.close();
    }

    deallocate(omega_array);
    deallocate(self3_imag);
    deallocate(self3_real);
}
