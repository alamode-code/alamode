/*
relaxation.cpp

Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "relaxation.h"
#include "ifc_derivative.h"
#include <Eigen/Core>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <fftw3.h>
#include <iomanip>
#include "dynamical.h"
#include "error.h"
#include "optimizers.h"
#include "parsephon.h"
#include "scph.h"
#include "system.h"
#include "timer.h"
// #include <boost/range/algorithm.hpp> // boost/range may induce compile error

using namespace PHON_NS;

Relaxation::Relaxation(PHON *phon) : Pointers(phon)
{
    set_default_variables();
    derivative_ifc = std::make_unique<DerivativeIFC>(phon);
}

Relaxation::~Relaxation()
{
    deallocate_variables();
}

void Relaxation::set_default_variables()
{
    // variables related to the structural optimization
    relax_str = 0;
    relax_algo = 2;
    max_str_iter = 100;
    coord_conv_tol = 1.0e-5;
    mixbeta_coord = 0.5;
    alpha_steepest_decent = 1.0e4;
    cell_conv_tol = 1.0e-5;
    mixbeta_cell = 0.5;

    set_init_str = 1;
    cooling_u0_index = 0;
    cooling_u0_thr = 0.001;

    add_hess_diag = 100.0; // [cm^{-1}]
    stat_pressure = 0.0;   // [GPa]
}

void Relaxation::deallocate_variables()
{}

void Relaxation::setup_relaxation()
{
    MPI_Bcast(&relax_str, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    V0.resize(NT);
    std::fill(V0.begin(), V0.end(), 0.0);
}

void Relaxation::create_optimizer(const size_t num_modes)
{
    if (relax_str == 1 && relax_algo == 2) {
        optimizer = std::make_unique<Newton_Optimizer>(mixbeta_coord);
    } else if (relax_str == 2 && relax_algo == 2) {
        optimizer = std::make_unique<CellCoord_Newton_Optimizer>(mixbeta_cell, mixbeta_coord);
    } else if (relax_str == 1 && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes - 3, num_modes - 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init);
    } else if (relax_str == 2 && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes + 3, num_modes + 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init);
    }
}

void Relaxation::load_V0_from_file()
{
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    if (mympi->my_rank == 0) {

        std::vector<double> Temp_array(NT);
        double temp;

        for (int i = 0; i < NT; ++i) {
            Temp_array[i] = Tmin + dT * static_cast<double>(i);
        }

        std::ifstream ifs_v0;
        const auto file_v0 = input->job_title + ".V0";
        ifs_v0.open(file_v0.c_str(), std::ios::in);

        if (!ifs_v0) {
            exit("load_V0_from_file", "Cannot open V0 file.");
        }

        for (int iT = 0; iT < NT; iT++) {
            ifs_v0 >> temp >> V0[iT];

            if (std::fabs(temp - Temp_array[iT]) > eps6) {
                exit("load_V0_from_file", "Temperature grid is not consistent.");
            }
        }

        ifs_v0.close();
    }
    MPI_Bcast(V0.data(), NT, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Relaxation::store_V0_to_file() const
{
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;
    std::vector<double> Temp_array(NT);

    for (int i = 0; i < NT; ++i) {
        Temp_array[i] = Tmin + dT * static_cast<double>(i);
    }

    std::ofstream ofs_v0;
    const auto file_v0 = input->job_title + ".V0";

    ofs_v0.open(file_v0.c_str(), std::ios::out);

    if (!ofs_v0) {
        exit("store_V0_to_file", "Cannot open V0 file");
    }

    for (int i = 0; i < NT; i++) {
        ofs_v0 << std::scientific << std::setprecision(15);
        ofs_v0 << std::setw(30) << Temp_array[i] << std::setw(30) << V0[i] << '\n';
    }

    ofs_v0.close();

    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_v0;
    std::cout << " : Renormalized static potential V0 (restart file)\n";
}

void Relaxation::set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array) const
{

    // if the shape of the unit cell is relaxed,
    // read elastic constants from file
    if (relax_str == 2 || relax_str == 3) {
        read_C1_array(C1_array);
        read_elastic_constants(C2_array, C3_array);
        return;
    }
    // if the unit cell is fixed,
    // dummy values are set in the elastic constants
    if (relax_str == 1) {
        int i1, i2;
        std::fill_n(C1_array, 9, 0.0);

        // The elastic constant should be positive-definite
        // except for the rotational degrees of freedom
        for (i1 = 0; i1 < 3; i1++) {
            for (i2 = 0; i2 < 3; i2++) {
                for (int i3 = 0; i3 < 3; i3++) {
                    for (int i4 = 0; i4 < 3; i4++) {
                        if ((i1 == i3 && i2 == i4) || (i1 == i4 && i2 == i3)) {
                            C2_array[i1 * 3 + i2][i3 * 3 + i4] = 10.0; // This dummy value can be any positiva value
                        } else {
                            C2_array[i1 * 3 + i2][i3 * 3 + i4] = 0.0;
                        }
                    }
                }
            }
        }

        for (i1 = 0; i1 < 9; i1++) {
            for (i2 = 0; i2 < 9; i2++) {
                std::fill_n(C3_array[i1][i2], 9, 0.0);
            }
        }
    }
}

void Relaxation::read_C1_array(double *const C1_array)
{
    std::fstream fin_C1_array;
    std::string str_tmp;

    // initialize elastic constants
    for (auto i1 = 0; i1 < 9; i1++) {
        C1_array[i1] = 0.0;
    }

    fin_C1_array.open("C1_array.in");

    if (!fin_C1_array) {
        std::cout << "  Warning: file C1_array.in could not be open.\n";
        std::cout << "  The stress tensor at the reference structure is set zero.\n";
        return;
    }

    fin_C1_array >> str_tmp;
    for (auto i1 = 0; i1 < 9; i1++) {
        fin_C1_array >> C1_array[i1];
    }
}

void Relaxation::read_elastic_constants(double *const *const C2_array, double *const *const *const C3_array) const
{
    std::fstream fin_elastic_constants;
    std::string str_tmp;
    int i1, i2;

    // read elastic_constants.in from strain_IFC_dir directory
    fin_elastic_constants.open(strain_IFC_dir + "elastic_constants.in");

    if (!fin_elastic_constants) {
        exit("read_elastic_constants", "could not open file elastic_constants.in");
    }

    fin_elastic_constants >> str_tmp;
    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            fin_elastic_constants >> C2_array[i1][i2];
        }
    }
    fin_elastic_constants >> str_tmp;
    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            for (int i3 = 0; i3 < 9; i3++) {
                fin_elastic_constants >> C3_array[i1][i2][i3];
            }
        }
    }
}


void Relaxation::set_init_structure_atT(double *q0, double **u_tensor, double *u0, bool &converged_prev,
                                        int &str_diverged, const int i_temp_loop, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic) const
{
    int i1, i2;

    optimizer->initialize_flag = 1;

    if (str_diverged) {
        std::cout << " The crystal structure at the previous temperature is divergent.\n";
        std::cout << " read initial structure from input files.\n\n";

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

        // set initial strain
        if (relax_str == 1) {
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }
        } else {
            set_initial_strain(u_tensor);
        }
        converged_prev = false;
        str_diverged = 0;

        // set the flag to initialize the optimizer
        optimizer->initialize_flag = 1;

        return;
    }

    std::cout << " SET_INIT_STR = " << set_init_str << ":";

    if (set_init_str == 1) {
        std::cout << " set initial structure from the input file.\n\n";

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
        if (relax_str == 1) {
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }
        } else {
            set_initial_strain(u_tensor);
        }
        converged_prev = false;
        optimizer->initialize_flag = 1;

    } else if (set_init_str == 2) {
        if (i_temp_loop == 0) {
            std::cout << " set initial structure from the input file.\n\n";

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_str == 1) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            converged_prev = false;
            optimizer->initialize_flag = 1;
        } else {
            std::cout << " start from structure from the previous temperature.\n\n";
        }

    } else if (set_init_str == 3) {
        // read initial structure at initial temperature
        if (i_temp_loop == 0) {
            std::cout << " read initial structure from input files.\n\n";

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_str == 1) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            optimizer->initialize_flag = 1;
        }
        // read initial DISPLACEMENT if the structure converges
        // to the high-symmetry one.
        else if (std::fabs(u0[cooling_u0_index]) < cooling_u0_thr)
        {
            std::cout << '\n';
            std::cout << " u0[" << cooling_u0_index << "] < " << std::setw(15) << std::setprecision(6) << cooling_u0_thr
                      << " is satisfied.\n";
            std::cout << " the structure is back to the high-symmetry phase.\n";
            std::cout << " set again initial displacement from input file.\n\n";

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            converged_prev = false;
            optimizer->initialize_flag = 1;
        } else {
            std::cout << " start from the structure at the previous temperature.\n\n";
        }
    }
}

void Relaxation::set_initial_q0(double *const q0, std::complex<double> ***evec_harmonic) const
{
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;

    for (int is = 0; is < ns; is++) {
        q0[is] = 0.0;
        for (int i_atm = 0; i_atm < natmin; i_atm++) {
            for (int ixyz = 0; ixyz < 3; ixyz++) {
                q0[is] += evec_harmonic[0][is][i_atm * 3 + ixyz].real() * std::sqrt(system->get_mass_prim()[i_atm]) *
                          init_u0[i_atm * 3 + ixyz];
            }
        }
    }
}

void Relaxation::set_initial_strain(double *const *const u_tensor) const
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            u_tensor[i][j] = init_u_tensor[i][j];
        }
    }
}

void Relaxation::calculate_u0(const double *const q0, double *const u0, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;

    for (int i_atm = 0; i_atm < natmin; i_atm++) {
        for (int ixyz = 0; ixyz < 3; ixyz++) {
            const auto is = i_atm * 3 + ixyz;
            u0[is] = 0.0;
            for (int is2 = 0; is2 < ns; is2++) {
                if (std::fabs(omega2_harmonic[0][is2]) < eps8) {
                    continue;
                }
                u0[is] += evec_harmonic[0][is2][is].real() * q0[is2];
            }
            u0[is] /= std::sqrt(system->get_mass_prim()[i_atm]);
        }
    }
}

void Relaxation::update_cell_coordinate(
    double *q0, double *u0, double **u_tensor, const std::complex<double> *const v1_array_atT,
    const double *const *const omega2_array, const std::complex<double> *const del_v0_strain_atT,
    const double *const *const C2_array, const std::complex<double> *const *const *const cmat_convert,
    const std::vector<int> &harm_optical_modes, double *delta_q0, double *delta_u0, double *delta_umn, double &du0,
    double &du_tensor, double **omega2_harmonic, std::complex<double> ***evec_harmonic) const
{
    using namespace Eigen;

    const auto ns = dynamical->neval;
    int is;

    MatrixXcd Cmat(ns, ns), v2_mat_full(ns, ns);
    std::vector<double> grad_vec;
    std::vector<double> delta_vec;
    std::vector<double> state_vec;
    std::vector<std::vector<double>> hessian_mat;

    MatrixXcd C2_mat_tmp(6, 6);
    VectorXcd del_v0_strain_vec(6);


    constexpr double Ry_to_kayser_tmp = Hz_to_kayser / time_ry;
    const double add_hess_diag_omega2 = std::pow(add_hess_diag / Ry_to_kayser_tmp, 2);

    if (relax_str == 1) {
        delta_vec.assign(ns - 3, 0.0);
        state_vec.assign(ns - 3, 0.0);
        grad_vec.assign(ns - 3, 0.0);
        hessian_mat.assign(ns - 3, std::vector<double>(ns - 3, 0.0));
    } else if (relax_str == 2) {
        delta_vec.assign(ns + 3, 0.0);
        state_vec.assign(ns + 3, 0.0);
        grad_vec.assign(ns + 3, 0.0);
        hessian_mat.assign(ns + 3, std::vector<double>(ns + 3, 0.0));
    }

    for (is = 0; is < ns; is++) {
        delta_q0[is] = 0.0;
    }
    for (is = 0; is < 6; is++) {
        delta_umn[is] = 0.0;
    }

    if (relax_algo == 1) {
        // steepest decent
        for (is = 0; is < ns; is++) {
            // skip acoustic mode
            if (std::fabs(omega2_harmonic[0][is]) < eps8) {
                continue;
            }
            delta_q0[is] = -alpha_steepest_decent * v1_array_atT[is].real();
            q0[is] += delta_q0[is];
        }
    } else if (relax_algo >= 2) {
        int js;
        // iterative solution of linear equation

        // prepare harmonic IFC matrix
        for (is = 0; is < ns; is++) {
            for (js = 0; js < ns; js++) {
                Cmat(js, is) = cmat_convert[0][is][js]; // transpose
                v2_mat_full(is, js) = 0.0;
            }
            v2_mat_full(is, is) = omega2_array[0][is];
        }
        v2_mat_full = Cmat.adjoint() * v2_mat_full * Cmat;

        // set gradient, hessian, and current state
        for (is = 0; is < ns - 3; is++) {
            for (js = 0; js < ns - 3; js++) {
                hessian_mat[is][js] = v2_mat_full(harm_optical_modes[is], harm_optical_modes[js]).real();
            }
            hessian_mat[is][is] += add_hess_diag_omega2;
        }

        for (is = 0; is < ns - 3; is++) {
            grad_vec[is] = v1_array_atT[harm_optical_modes[is]].real();
            state_vec[is] = q0[harm_optical_modes[is]];
        }


        if (relax_str == 1) {
            int i1, i2;
            // call optimizer
            optimizer->update_state(ns - 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
                q0[harm_optical_modes[is]] += delta_q0[harm_optical_modes[is]];
            }
            for (i1 = 0; i1 < 6; i1++) {
                delta_umn[i1] = 0.0;
            }
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }

            std::cout << "grad_vec = ";
            for (auto &val: grad_vec) {
                std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            }
            std::cout << '\n';
            std::cout << "state_vec = ";
            for (auto &val: state_vec) {
                std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            }
            std::cout << '\n';

        } else if (relax_str == 2) {
            int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;

            // prepare matrix of elastic constants and vector of del_v0_strain_atT
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                del_v0_strain_vec(itmp1) = del_v0_strain_atT[itmp1 * 3 + itmp1];

                itmp2 = (itmp1 + 1) % 3;
                itmp3 = (itmp1 + 2) % 3;
                del_v0_strain_vec(itmp1 + 3) = del_v0_strain_atT[itmp2 * 3 + itmp3];
            }

            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    C2_mat_tmp(itmp1, itmp2) = C2_array[itmp1 * 3 + itmp1][itmp2 * 3 + itmp2];
                }
            }
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    itmp3 = (itmp2 + 1) % 3;
                    itmp4 = (itmp2 + 2) % 3;
                    C2_mat_tmp(itmp1, itmp2 + 3) = 2.0 * C2_array[itmp1 * 3 + itmp1][itmp3 * 3 + itmp4];
                    C2_mat_tmp(itmp2 + 3, itmp1) = C2_array[itmp3 * 3 + itmp4][itmp1 * 3 + itmp1];
                }
            }
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    itmp3 = (itmp1 + 1) % 3;
                    itmp4 = (itmp1 + 2) % 3;
                    itmp5 = (itmp2 + 1) % 3;
                    itmp6 = (itmp2 + 2) % 3;
                    C2_mat_tmp(itmp1 + 3, itmp2 + 3) = 2.0 * C2_array[itmp3 * 3 + itmp4][itmp5 * 3 + itmp6];
                }
            }

            std::cout << "del_v0_strain_vec\n";
            std::cout << del_v0_strain_vec << '\n';
            std::cout << "C2_mat_tmp = \n";
            std::cout << C2_mat_tmp << '\n';

            // write C2mat to hessian matrix
            for (itmp1 = 0; itmp1 < 6; itmp1++) {
                for (itmp2 = 0; itmp2 < 6; itmp2++) {
                    hessian_mat[itmp1 + ns - 3][itmp2 + ns - 3] = C2_mat_tmp(itmp1, itmp2).real();
                }
            }
            // write to grad vector and state vector
            for (itmp1 = 0; itmp1 < 6; itmp1++) {
                grad_vec[itmp1 + ns - 3] = del_v0_strain_vec(itmp1).real();

                if (itmp1 < 3) {
                    state_vec[itmp1 + ns - 3] = u_tensor[itmp1][itmp1];
                } else {
                    itmp2 = (itmp1 + 1) % 3;
                    itmp3 = (itmp1 + 2) % 3;
                    state_vec[itmp1 + ns - 3] = u_tensor[itmp2][itmp3];
                }
            }

            std::cout << "grad_vec = ";
            for (auto &val: grad_vec) {
                std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            }
            std::cout << '\n';
            std::cout << "state_vec = ";
            for (auto &val: state_vec) {
                std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            }
            std::cout << '\n';


            // call optimizer
            optimizer->update_state(ns + 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            std::cout << "update state";
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
                std::cout << "delta_q0[" << harm_optical_modes[is] << "] = " << delta_q0[harm_optical_modes[is]]
                          << '\n';
                q0[harm_optical_modes[is]] += delta_q0[harm_optical_modes[is]];
            }
            // update u tensor
            for (is = 0; is < 6; is++) {
                delta_umn[is] = delta_vec[is + ns - 3];
                if (is < 3) {
                    u_tensor[is][is] += delta_umn[is];
                } else {
                    itmp1 = (is + 1) % 3;
                    itmp2 = (is + 2) % 3;
                    u_tensor[itmp1][itmp2] += delta_umn[is];
                    u_tensor[itmp2][itmp1] += delta_umn[is];
                }
            }
        }
    }

    calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

    du0 = 0.0;
    calculate_u0(delta_q0, delta_u0, omega2_harmonic, evec_harmonic);
    for (is = 0; is < ns; is++) {
        du0 += delta_u0[is] * delta_u0[is];
    }
    du0 = std::sqrt(du0);

    du_tensor = 0.0;
    for (is = 0; is < 6; is++) {
        du_tensor += delta_umn[is] * delta_umn[is];
        if (is >= 3) {
            du_tensor += delta_umn[is] * delta_umn[is];
        }
    }
    du_tensor = std::sqrt(du_tensor);
    std::cout << "delta_umn\n";
    for (is = 0; is < 6; is++) {
        std::cout << delta_umn[is] << '\n';
    }
    std::cout << "\n";
}

void Relaxation::check_str_divergence(int &diverged, const double *const q0, const double *const u0,
                                      const double *const *const u_tensor) const
{
    const auto ns = dynamical->neval;

    int i, j;

    int flag_diverged = 0;

    for (i = 0; i < ns; i++) {
        if (!std::isfinite(q0[i]) || !std::isfinite(u0[i])) {
            flag_diverged = 1;
            break;
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (!std::isfinite(u_tensor[i][j])) {
                flag_diverged = 1;
                i = 3;
                break;
            }
        }
    }

    diverged = flag_diverged;
}


void Relaxation::compute_del_v_strain(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                      std::complex<double> **del_v1_del_umn, std::complex<double> **del2_v1_del_umn2,
                                      std::complex<double> **del3_v1_del_umn3, std::complex<double> ***del_v2_del_umn,
                                      std::complex<double> ***del2_v2_del_umn2, std::complex<double> ****del_v3_del_umn,
                                      double **omega2_harmonic, std::complex<double> ***evec_harmonic, int relax_str,
                                      MinimumDistList ***mindist_list, const PhaseFactorStorage *phase_storage_in)
{
    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense->nk;

    // relax_str == 1: keep the unit cell fixed and relax internal coordinates
    // set renormalization from strain as zero
    if (relax_str == 1) {
        set_del_v_fixed_cell(nk,
                             ns,
                             del_v1_del_umn,
                             del2_v1_del_umn2,
                             del3_v1_del_umn3,
                             del_v2_del_umn,
                             del2_v2_del_umn2,
                             del_v3_del_umn);
        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // relax_str == 2: relax both the cell shape and the internal coordinates.
    if (relax_str == 2) {

        set_del_v_relax_cell(kmesh_coarse,
                             kmesh_dense,
                             ns,
                             del_v1_del_umn,
                             del2_v1_del_umn2,
                             del3_v1_del_umn3,
                             del_v2_del_umn,
                             del2_v2_del_umn2,
                             del_v3_del_umn,
                             omega2_harmonic,
                             evec_harmonic,
                             mindist_list,
                             phase_storage_in);

        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // relax_str == 3: calculate lowest-order linear equation of QHA.
    if (relax_str == 3) {

        set_del_v_relax_cell_linearQHA(kmesh_coarse,
                                       kmesh_dense,
                                       ns,
                                       del_v1_del_umn,
                                       del2_v1_del_umn2,
                                       del_v2_del_umn,
                                       omega2_harmonic,
                                       evec_harmonic,
                                       mindist_list);

        if (mympi->my_rank == 0) timer->print_elapsed();
    }
}

// relax_str == 1: keep the unit cell fixed and relax internal coordinates
// set renormalization from strain as zero
void Relaxation::set_del_v_fixed_cell(const size_t nk, const size_t ns, std::complex<double> **del_v1_del_umn,
                                      std::complex<double> **del2_v1_del_umn2, std::complex<double> **del3_v1_del_umn3,
                                      std::complex<double> ***del_v2_del_umn, std::complex<double> ***del2_v2_del_umn2,
                                      std::complex<double> ****del_v3_del_umn)
{
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    int i1, ik1;
    for (i1 = 0; i1 < 9; i1++) {
        std::fill_n(del_v1_del_umn[i1], ns, complex_zero);
    }

    for (i1 = 0; i1 < 81; i1++) {
        std::fill_n(del2_v1_del_umn2[i1], ns, complex_zero);
    }

    for (i1 = 0; i1 < 729; i1++) {
        std::fill_n(del3_v1_del_umn3[i1], ns, complex_zero);
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (ik1 = 0; ik1 < nk; ik1++) {
            std::fill_n(del_v2_del_umn[i1][ik1], ns * ns, complex_zero);
        }
    }

    for (i1 = 0; i1 < 81; i1++) {
        for (ik1 = 0; ik1 < nk; ik1++) {
            std::fill_n(del2_v2_del_umn2[i1][ik1], ns * ns, complex_zero);
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (ik1 = 0; ik1 < nk; ik1++) {
            for (int is1 = 0; is1 < ns; is1++) {
                std::fill_n(del_v3_del_umn[i1][ik1][is1], ns * ns, complex_zero);
            }
        }
    }
}

// relax_str == 2: relax both the cell shape and the internal coordinates.
void Relaxation::set_del_v_relax_cell(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                      const size_t ns, std::complex<double> **del_v1_del_umn,
                                      std::complex<double> **del2_v1_del_umn2, std::complex<double> **del3_v1_del_umn3,
                                      std::complex<double> ***del_v2_del_umn, std::complex<double> ***del2_v2_del_umn2,
                                      std::complex<double> ****del_v3_del_umn, double **omega2_harmonic,
                                      std::complex<double> ***evec_harmonic, MinimumDistList ***mindist_list,
                                      const PhaseFactorStorage *phase_storage_in)
{
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;

    // first-order derivative of first-order IFCs
    switch (renorm_2to1st) {
    case 0:
        if (mympi->my_rank == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        for (int i1 = 0; i1 < 9; i1++) {
            std::fill_n(del_v1_del_umn[i1], ns, complex_zero);
        }
        if (mympi->my_rank == 0) std::cout << "  done!\n";
        break;
    case 1:
        if (mympi->my_rank == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (from harmonic IFCs) ... ";
        derivative_ifc->compute_del_v1_del_umn(del_v1_del_umn, evec_harmonic);
        if (mympi->my_rank == 0) std::cout << "  done!\n";
        break;
    case 2:
        if (mympi->my_rank == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (finite difference method) ... ";
        calculate_delv1_delumn_finite_difference(del_v1_del_umn, evec_harmonic);
        if (mympi->my_rank == 0) std::cout << "  done!\n";
        break;
    default:
        break;
    }

    // second and third-order derivatives of first-order IFCs
    switch (renorm_34to1st) {
    case 0:
        if (mympi->my_rank == 0) std::cout << "  - second-order derivatives of first-order IFCs (set zero) ... ";
        for (int i1 = 0; i1 < 81; i1++) {
            std::fill_n(del2_v1_del_umn2[i1], ns, complex_zero);
        }
        if (mympi->my_rank == 0) {
            std::cout << "  done!\n";
            std::cout << "  - third-order derivatives of first-order IFCs (set zero) ... ";
        }
        for (int i1 = 0; i1 < 729; i1++) {
            std::fill_n(del3_v1_del_umn3[i1], ns, complex_zero);
        }
        if (mympi->my_rank == 0) std::cout << "  done!\n";
        break;
    case 1:
        if (mympi->my_rank == 0) std::cout << "  - second-order derivatives of first-order IFCs (from cubic IFCs) ... ";
        derivative_ifc->compute_del2_v1_del_umn2(del2_v1_del_umn2, evec_harmonic);

        if (mympi->my_rank == 0) {
            std::cout << "  done!\n";
            std::cout << "  - third-order derivatives of first-order IFCs (from quartic IFCs) ... ";
        }
        derivative_ifc->compute_del3_v1_del_umn3(del3_v1_del_umn3, evec_harmonic);

        if (mympi->my_rank == 0) std::cout << "  done!\n";
        break;
    default:
        break;
    }


    // first-order derivatives of harmonic IFCs
    switch (renorm_3to2nd) {
    case 1:
        if (mympi->my_rank == 0)
            std::cout << "  - first-order derivatives of harmonic IFCs (from cubic IFCs) ... " << std::flush;

        derivative_ifc->compute_del_v2_del_umn(del_v2_del_umn, evec_harmonic, nk_interpolate, kmesh_coarse->xk);
        break;
    case 2:
        if (mympi->my_rank == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    use inputs with all strain patterns ... " << std::flush;
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v2_del_umn,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 mindist_list);
        break;

    case 3:
        if (mympi->my_rank == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    use inputs with specified strain patterns ... " << std::flush;
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v2_del_umn,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 mindist_list);
        break;

    case 4:
        if (mympi->my_rank == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs\n";
            std::cout << "    (read from file in k-space representation) ... " << std::flush;
        }

        read_del_v2_del_umn_in_kspace(omega2_harmonic, evec_harmonic, del_v2_del_umn, nk);
        break;

    default:
        break;
    }
    if (mympi->my_rank == 0) std::cout << "  done!\n";

    // second order derivatives of harmonic IFCs
    if (mympi->my_rank == 0)
        std::cout << "  - second-order derivatives of harmonic IFCs (from quartic IFCs) ... " << std::flush;

    derivative_ifc->compute_del2_v2_del_umn2(del2_v2_del_umn2, evec_harmonic, nk, kmesh_dense->xk);

    if (mympi->my_rank == 0) {
        std::cout << "  done!\n";
        std::cout << "  - first-order derivatives of cubic IFCs (from quartic IFCs) ... " << std::flush;
    }

    // first order derivatives of cubic IFCs
    derivative_ifc->compute_del_v3_del_umn(del_v3_del_umn,
                                           omega2_harmonic,
                                           evec_harmonic,
                                           kmesh_coarse,
                                           kmesh_dense,
                                           phase_storage_in);

    if (mympi->my_rank == 0) {
        std::cout << "  done!\n";
    }
}

// relax_str == 3: calculate lowest-order linear equation of QHA.
void Relaxation::set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse,
                                                const KpointMeshUniform *kmesh_dense, const size_t ns,
                                                std::complex<double> **del_v1_del_umn,
                                                std::complex<double> **del2_v1_del_umn2,
                                                std::complex<double> ***del_v2_del_umn, double **omega2_harmonic,
                                                std::complex<double> ***evec_harmonic, MinimumDistList ***mindist_list)

{
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;

    int i1, is1;
    // first-order derivative of first-order IFCs
    if (renorm_2to1st == 0) {
        if (mympi->my_rank == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        for (i1 = 0; i1 < 9; i1++) {
            for (is1 = 0; is1 < ns; is1++) {
                del_v1_del_umn[i1][is1] = complex_zero;
            }
        }
    } else if (renorm_2to1st == 1) {
        if (mympi->my_rank == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (from harmonic IFCs) ... ";
        derivative_ifc->compute_del_v1_del_umn(del_v1_del_umn, evec_harmonic);

    } else if (renorm_2to1st == 2) {
        if (mympi->my_rank == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (finite difference method) ... ";
        calculate_delv1_delumn_finite_difference(del_v1_del_umn, evec_harmonic);
    }
    if (mympi->my_rank == 0) {
        std::cout << "  done!\n";
    }

    // second-order derivatives of 1st order IFCs
    if (renorm_34to1st == 0) {
        if (mympi->my_rank == 0) std::cout << "  - second-order derivatives of first-order IFCs (set zero) ... ";
        for (i1 = 0; i1 < 81; i1++) {
            for (is1 = 0; is1 < ns; is1++) {
                del2_v1_del_umn2[i1][is1] = complex_zero;
            }
        }

    } else if (renorm_34to1st == 1) {
        if (mympi->my_rank == 0) std::cout << "  - second-order derivatives of first-order IFCs (from cubic IFCs) ... ";
        derivative_ifc->compute_del2_v1_del_umn2(del2_v1_del_umn2, evec_harmonic);
    }
    if (mympi->my_rank == 0) {
        std::cout << "  done!\n";
    }

    // first-order derivatives of harmonic IFCs
    if (renorm_3to2nd == 1) {
        if (mympi->my_rank == 0) std::cout << "  - first-order derivatives of harmonic IFCs (from cubic IFCs) ... ";

        derivative_ifc->compute_del_v2_del_umn(del_v2_del_umn, evec_harmonic, nk, kmesh_coarse->xk);
    } else if (renorm_3to2nd == 2 || renorm_3to2nd == 3) {
        if (mympi->my_rank == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            if (renorm_3to2nd == 2) {
                std::cout << "   use inputs with all strain patterns ...\n";
            } else if (renorm_3to2nd == 3) {
                std::cout << "   use inputs with specified strain patterns ...\n";
            }
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v2_del_umn,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 mindist_list);
    } else if (renorm_3to2nd == 4) {
        if (mympi->my_rank == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs\n";
            std::cout << "    (read from file in k-space representation) ... ";
        }
        read_del_v2_del_umn_in_kspace(omega2_harmonic, evec_harmonic, del_v2_del_umn, nk);
    }
    if (mympi->my_rank == 0) {
        std::cout << "  done!\n";
    }
}


void Relaxation::read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                               const std::complex<double> *const *const *const evec_harmonic,
                                               std::complex<double> ***del_v2_del_umn, const unsigned int nk) const
{
    using namespace Eigen;

    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;

    int ixyz1, ixyz2;
    int ik, is, js;

    double re_tmp, im_tmp;

    MatrixXcd dymat_tmp_mode(ns, ns);
    MatrixXcd dymat_tmp_alphamu(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    // read input
    std::fstream fin_strain_mode_coupling_kspace;

    // temporary
    // (alpha,mu) representation in k-space
    std::complex<double> ***del_v2_del_umn_alphamu;
    allocate(del_v2_del_umn_alphamu, 9, nk, ns * ns);

    // read from file
    fin_strain_mode_coupling_kspace.open("B_array_kspace.txt");

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            for (ik = 0; ik < nk; ik++) {
                for (is = 0; is < ns * ns; is++) {
                    fin_strain_mode_coupling_kspace >> re_tmp >> im_tmp;
                    del_v2_del_umn_alphamu[ixyz1 * 3 + ixyz2][ik][is] = std::complex<double>(re_tmp, im_tmp);
                }
            }
        }
    }

    // transform to mode-representation
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            for (ik = 0; ik < nk; ik++) {

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        evec_tmp(is, js) = evec_harmonic[ik][js][is]; // transpose
                        dymat_tmp_alphamu(is, js) = del_v2_del_umn_alphamu[ixyz1 * 3 + ixyz2][ik][is * ns + js];
                    }
                }
                dymat_tmp_mode = evec_tmp.adjoint() * dymat_tmp_alphamu * evec_tmp;

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        del_v2_del_umn[ixyz1 * 3 + ixyz2][ik][is * ns + js] = dymat_tmp_mode(is, js);
                    }
                }
            }
        }
    }
    deallocate(del_v2_del_umn_alphamu);

    // assign ASR
    int *is_acoustic;
    allocate(is_acoustic, 3 * natmin);

    constexpr double threshold_acoustic = 1.0e-16;
    int count_acoustic = 0;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < threshold_acoustic) {
            is_acoustic[is] = 1;
            count_acoustic++;
        } else {
            is_acoustic[is] = 0;
        }
    }

    // check number of acoustic modes
    if (count_acoustic != 3) {
        std::cout << "Warning in calculate_del_v2_strain_from_cubic_by_finite_difference: ";
        std::cout << count_acoustic << " acoustic modes are detected in Gamma point.\n\n";
    }

    // set acoustic sum rule (ASR)
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is = 0; is < ns; is++) {
            // mode is is an acoustic mode at Gamma point
            if (is_acoustic[is] == 0) {
                continue;
            }
            for (js = 0; js < ns; js++) {
                del_v2_del_umn[ixyz1][0][is * ns + js] = complex_zero;
                del_v2_del_umn[ixyz1][0][js * ns + is] = complex_zero;
            }
        }
    }
}


// calculate strain-force coupling using finite-difference method.
// use finite difference for all 6 strains
void Relaxation::calculate_delv1_delumn_finite_difference(
    std::complex<double> **del_v1_del_umn, const std::complex<double> *const *const *const evec_harmonic) const
{

    // Read strain_force.in, symmetrize elements, and set it to del_v1_del_umn

    const auto natmin = system->get_primcell().number_of_atoms;
    auto ns = dynamical->neval;

    int ixyz1, ixyz2, ixyz3, ixyz12, ixyz22, ixyz32, i1, i2;
    int iat1, iat2, is1, isymm;
    double dtmp;
    std::string mode_tmp;
    double smag, weight;
    Eigen::Matrix3d weight_sum;
    std::fstream fin_strain_force_coupling;

    double **del_v1_del_umn_in_real_space;
    double **del_v1_del_umn_in_real_space_symm;
    allocate(del_v1_del_umn_in_real_space, 9, ns);
    allocate(del_v1_del_umn_in_real_space_symm, 9, ns);

    fin_strain_force_coupling.open(strain_IFC_dir + "strain_force.in");

    if (!fin_strain_force_coupling) {
        exit("calculate_delv1_delumn_finite_difference", "strain_force.in not found");
    }

    weight_sum.setZero();

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        std::fill_n(del_v1_del_umn_in_real_space[ixyz1], ns, 0.0);
    }

    // read input file
    while (true) {
        if (fin_strain_force_coupling >> mode_tmp >> smag >> weight) {
            if (mode_tmp == "xx") {
                ixyz1 = ixyz2 = 0;
            } else if (mode_tmp == "yy") {
                ixyz1 = ixyz2 = 1;
            } else if (mode_tmp == "zz") {
                ixyz1 = ixyz2 = 2;
            } else if (mode_tmp == "xy") {
                ixyz1 = 0;
                ixyz2 = 1;
            } else if (mode_tmp == "yz") {
                ixyz1 = 1;
                ixyz2 = 2;
            } else if (mode_tmp == "zx") {
                ixyz1 = 2;
                ixyz2 = 0;
            }

            for (iat1 = 0; iat1 < natmin; iat1++) {
                for (ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    fin_strain_force_coupling >> dtmp;
                    del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3] += dtmp * -1.0 / smag * weight;

                    if (ixyz1 != ixyz2) {
                        del_v1_del_umn_in_real_space[ixyz2 * 3 + ixyz1][iat1 * 3 + ixyz3] =
                            del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3];
                    }
                }
            }

            if (ixyz1 == ixyz2) {
                weight_sum(ixyz1, ixyz2) += weight;
            } else {
                weight_sum(ixyz1, ixyz2) += weight;
                weight_sum(ixyz2, ixyz1) += weight;
            }
        } else {
            break;
        }
    }

    // check weight_sum
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) > eps6) {
                exit("calculate_delv1_delumn_finite_difference", "Sum of weights must be 1.");
            }
        }
    }


    // convert from eV/Angst to Ry/Bohr
    constexpr double eV_to_Ry = 1.6021766208e-19 / Ryd;
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_in_real_space[ixyz1][is1] *= Bohr_in_Angstrom * eV_to_Ry;
        }
    }

    // for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
    //     for (int is = 0; is < ns; ++is) {
    //         std::cout << "del_v1_del_umn_in_real_space[" << ixyz1 << "][" << is << "] = "
    //             << del_v1_del_umn_in_real_space[ixyz1][is] << '\n';
    //     }
    // }

    // symmetrize
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        std::fill_n(del_v1_del_umn_in_real_space_symm[ixyz1], ns, 0.0);
    }

    for (isymm = 0; isymm < symmetry->SymmListWithMap_ref.size(); isymm++) {
        for (iat1 = 0; iat1 < natmin; iat1++) {
            iat2 = symmetry->SymmListWithMap_ref[isymm].mapping[iat1];

            for (i1 = 0; i1 < 27; i1++) {
                ixyz1 = i1 / 9;
                ixyz2 = (i1 % 9) / 3;
                ixyz3 = i1 % 3;
                for (i2 = 0; i2 < 27; i2++) {
                    ixyz12 = i2 / 9;
                    ixyz22 = (i2 % 9) / 3;
                    ixyz32 = i2 % 3;

                    del_v1_del_umn_in_real_space_symm[ixyz12 * 3 + ixyz22][iat2 * 3 + ixyz32] +=
                        del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3] *
                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz12 * 3 + ixyz1] *
                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz22 * 3 + ixyz2] *
                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz32 * 3 + ixyz3];
                }
            }
        }
    }

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_in_real_space_symm[ixyz1][is1] /= static_cast<double>(symmetry->SymmListWithMap_ref.size());
        }
    }

    // for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
    //     for (int is = 0; is < ns; ++is) {
    //         std::cout << "del_v1_del_umn_in_real_space_symm[" << ixyz1 << "][" << is << "] = "
    //             << del_v1_del_umn_in_real_space_symm[ixyz1][is] << '\n';
    //     }
    // }

    // transform to Fourier space
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn[ixyz1][is1] = 0.0;
            for (iat1 = 0; iat1 < natmin; iat1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    del_v1_del_umn[ixyz1][is1] += evec_harmonic[0][is1][iat1 * 3 + ixyz2] *
                                                  system->get_invsqrt_mass()[iat1] *
                                                  del_v1_del_umn_in_real_space_symm[ixyz1][iat1 * 3 + ixyz2];
                }
            }
        }
    }
    //
    // std::cout << std::scientific << std::setprecision(15);
    //
    // for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
    //     for (is1 = 0; is1 < ns; is1++) {
    //         std::cout << "del_v1_del_umn[" << ixyz1 << "][" << is1 << "] = "
    //             << del_v1_del_umn[ixyz1][is1] << '\n';
    //     }
    // }

    deallocate(del_v1_del_umn_in_real_space);
    deallocate(del_v1_del_umn_in_real_space_symm);
}

// calculate strain-force coupling using finite-difference method.
void Relaxation::calculate_delv2_delumn_finite_difference(double **omega2_harmonic,
                                                          const std::complex<double> *const *const *const evec_harmonic,
                                                          std::complex<double> ***del_v2_del_umn,
                                                          const KpointMeshUniform *kmesh_coarse,
                                                          const KpointMeshUniform *kmesh_dense,
                                                          MinimumDistList ***mindist_list) const
{
    using namespace Eigen;

    const auto natmin = system->get_primcell().number_of_atoms;
    const auto nat = system->get_supercell(0).number_of_atoms;
    const auto natmin3 = natmin * 3;
    const auto nat3 = nat * 3;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    int **symm_mapping_s;
    int **inv_translation_mapping;

    std::vector<FcsClassExtent> fc2_tmp;

    int ixyz1, ixyz2, ixyz3, ixyz4;
    int ixyz1_2, ixyz2_2, ixyz3_2, ixyz4_2;
    int ixyz_comb1, ixyz_comb2;
    int i1, i2;
    int iat1, iat2, iat1_2, iat2_2, iat2_2_prim;
    int itran1, itran2;
    int ik, is1, is2, is, js;
    int isymm, imode;
    int index2;

    std::complex<double> ***dymat_q, **dymat_tmp;
    std::complex<double> ***dymat_new;

    const auto nk1 = kmesh_coarse->nk_i[0];
    const auto nk2 = kmesh_coarse->nk_i[1];
    const auto nk3 = kmesh_coarse->nk_i[2];


    MatrixXcd dymat_tmp_mode(ns, ns);
    MatrixXcd dymat_tmp_alphamu(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    // read input
    std::fstream fin_strain_mode_coupling;
    int nmode;
    std::vector<std::string> mode_list;
    std::vector<double> smag_list;
    std::vector<double> weight_list;
    std::vector<std::string> filename_list;

    double smag_tmp, weight_tmp;
    std::string mode_tmp, filename_tmp;

    double ****dphi2_dumn_realspace_in;
    double ****dphi2_dumn_realspace_symm;
    MatrixXd dphi2_dumn_realspace_tmp(natmin3, nat3);
    Matrix3i exist_in;
    Matrix3d weight_sum;
    int ****count_tmp;

    allocate(dphi2_dumn_realspace_in, 3, 3, natmin3, nat3);
    allocate(dphi2_dumn_realspace_symm, 3, 3, natmin3, nat3);
    allocate(count_tmp, 3, 3, natmin3, nat3);

    // temporary
    // (alpha,mu) representation in k-space
    std::complex<double> ***del_v2_strain_from_cubic_alphamu;
    allocate(del_v2_strain_from_cubic_alphamu, 9, nk, ns * ns);

    exist_in.setZero();
    weight_sum.setZero();
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            for (i1 = 0; i1 < natmin3; i1++) {
                std::fill_n(dphi2_dumn_realspace_in[ixyz1][ixyz2][i1], nat3, 0.0);
                std::fill_n(dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1], nat3, 0.0);
                std::fill_n(count_tmp[ixyz1][ixyz2][i1], nat3, 0.0);
            }
        }
    }

    // read information on strain patterns and names of corresponding XML files.
    fin_strain_mode_coupling.open(strain_IFC_dir + "strain_harmonic.in");

    if (!fin_strain_mode_coupling) {
        exit("calculate_delv2_delumn_finite_difference", "strain_harmonic.in not found");
    }

    mode_list.clear();
    smag_list.clear();
    weight_list.clear();
    filename_list.clear();

    nmode = 0;
    while (true) {
        if (fin_strain_mode_coupling >> mode_tmp >> smag_tmp >> weight_tmp >> filename_tmp) {
            mode_list.push_back(mode_tmp);
            smag_list.push_back(smag_tmp);
            weight_list.push_back(weight_tmp);
            filename_list.push_back(filename_tmp);
            nmode++;
        } else {
            break;
        }
    }

    // read IFCs with strain.
    std::vector<std::vector<FcsArrayWithCell>> fc2_deformed(nmode);

    for (imode = 0; imode < nmode; imode++) {
        fcs_phonon->get_fcs_from_file(strain_IFC_dir + filename_list[imode], 0, fc2_deformed[imode]);
        fcs_phonon->replicate_force_constant(system, fc2_deformed[imode]);
    }

    weight_sum.setZero();

    for (imode = 0; imode < nmode; imode++) {
        if (mode_list[imode] == "xx") {
            ixyz1 = ixyz2 = 0;
        } else if (mode_list[imode] == "yy") {
            ixyz1 = ixyz2 = 1;
        } else if (mode_list[imode] == "zz") {
            ixyz1 = ixyz2 = 2;
        } else if (mode_list[imode] == "xy") {
            ixyz1 = 0;
            ixyz2 = 1;
        } else if (mode_list[imode] == "yz") {
            ixyz1 = 1;
            ixyz2 = 2;
        } else if (mode_list[imode] == "zx") {
            ixyz1 = 2;
            ixyz2 = 0;
        } else {
            exit("calculate_delv2_delumn_finite_difference", "Invalid name of strain mode in strain_harmonic.in.");
        }

        // calculate finite difference
        dphi2_dumn_realspace_tmp.setZero();

        // This part assumes that the supercell size of the deformed IFC2 calculation is the
        // same as the supercell size of the original harmonic IFC2 calculation.
        // TODO: improve this part to allow different (smaller) supercell sizes for deformed IFC2 calculations.
        for (const auto &it: fc2_deformed[imode]) {
            index2 = system->get_map_p2s(0)[it.pairs[1].index / 3][it.pairs[1].tran];
            dphi2_dumn_realspace_tmp(it.pairs[0].index, index2 * 3 + it.pairs[1].index % 3) += it.fcs_val;
        }
        for (const auto &it: fcs_phonon->force_constant_with_cell[0]) {
            index2 = system->get_map_p2s(0)[it.pairs[1].index / 3][it.pairs[1].tran];
            dphi2_dumn_realspace_tmp(it.pairs[0].index, index2 * 3 + it.pairs[1].index % 3) -= it.fcs_val;
        }

        if (ixyz1 == ixyz2) {
            for (i1 = 0; i1 < natmin3; i1++) {
                for (i2 = 0; i2 < nat3; i2++) {
                    dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                }
            }
            weight_sum(ixyz1, ixyz2) += weight_list[imode];
        } else {
            for (i1 = 0; i1 < natmin3; i1++) {
                for (i2 = 0; i2 < nat3; i2++) {
                    dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                    dphi2_dumn_realspace_in[ixyz2][ixyz1][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                }
            }
            weight_sum(ixyz1, ixyz2) += weight_list[imode];
            weight_sum(ixyz2, ixyz1) += weight_list[imode];
        }
    } // end imode


    // check weight_sum
    if (renorm_3to2nd == 2) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) < eps6) {
                    exist_in(ixyz1, ixyz2) = 1;
                } else {
                    exit("calculate_delv2_delumn_finite_difference", "Sum of weights must be 1.");
                }
            }
        }
    } else if (renorm_3to2nd == 3) {
        // finite difference method.
        // read input from specified strain modes.
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) < eps6) {
                    exist_in(ixyz1, ixyz2) = 1;
                } else if (std::fabs(weight_sum(ixyz1, ixyz2)) < eps6) {
                    exist_in(ixyz1, ixyz2) = 0;
                } else {
                    exit("calculate_delv2_delumn_finite_difference", "Sum of weights must be 1 or 0 for each mode.");
                }
            }
        }
    }

    // make mapping information
    allocate(symm_mapping_s, symmetry->SymmListWithMap_ref.size(), nat);
    make_supercell_mapping_by_symmetry_operations(symm_mapping_s);

    // Note: this part also assumes the same supercell size as the original harmonic IFC2 calculation
    // TODO: lift this assumption to allow different (smaller) supercell sizes for deformed IFC2 calculations.
    const auto ntran = system->get_map_p2s(0)[0].size();

    allocate(inv_translation_mapping, ntran, ntran);
    make_inverse_translation_mapping(inv_translation_mapping);

    // symmetrize and replicate
    if (renorm_3to2nd == 2) {
        // input is given for all strain modes.
        for (isymm = 0; isymm < symmetry->SymmListWithMap_ref.size(); isymm++) {

            for (iat1 = 0; iat1 < natmin; iat1++) {

                for (ixyz_comb1 = 0; ixyz_comb1 < 81; ixyz_comb1++) {
                    ixyz1 = ixyz_comb1 / 27;
                    ixyz2 = (ixyz_comb1 / 9) % 3;
                    ixyz3 = (ixyz_comb1 / 3) % 3;
                    ixyz4 = ixyz_comb1 % 3;
                    for (ixyz_comb2 = 0; ixyz_comb2 < 81; ixyz_comb2++) {
                        ixyz1_2 = ixyz_comb2 / 27;
                        ixyz2_2 = (ixyz_comb2 / 9) % 3;
                        ixyz3_2 = (ixyz_comb2 / 3) % 3;
                        ixyz4_2 = ixyz_comb2 % 3;

                        iat1_2 = symmetry->SymmListWithMap_ref[isymm].mapping[iat1];

                        for (i1 = 0; i1 < ntran; i1++) {
                            if (system->get_map_p2s(0)[iat1_2][i1] ==
                                symm_mapping_s[isymm][system->get_map_p2s(0)[iat1][0]])
                            {
                                itran1 = i1;
                            }
                        }

                        for (iat2 = 0; iat2 < nat; iat2++) {
                            iat2_2 = symm_mapping_s[isymm][iat2]; // temporary
                            iat2_2_prim = system->get_map_s2p(0)[iat2_2].atom_num;
                            itran2 = system->get_map_s2p(0)[iat2_2].tran_num;

                            iat2_2 = system->get_map_p2s(0)[iat2_2_prim][inv_translation_mapping[itran1][itran2]];

                            dphi2_dumn_realspace_symm[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2][iat2_2 * 3 + ixyz4_2] +=
                                dphi2_dumn_realspace_in[ixyz1][ixyz2][iat1 * 3 + ixyz3][iat2 * 3 + ixyz4] *
                                symmetry->SymmListWithMap_ref[isymm].rot[ixyz1_2 * 3 + ixyz1] *
                                symmetry->SymmListWithMap_ref[isymm].rot[ixyz2_2 * 3 + ixyz2] *
                                symmetry->SymmListWithMap_ref[isymm].rot[ixyz3_2 * 3 + ixyz3] *
                                symmetry->SymmListWithMap_ref[isymm].rot[ixyz4_2 * 3 + ixyz4];
                        }
                    }
                }
            }
        }

        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                for (i1 = 0; i1 < natmin * 3; i1++) {
                    for (i2 = 0; i2 < nat * 3; i2++) {
                        dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] /= symmetry->SymmListWithMap_ref.size();
                    }
                }
            }
        }
    } else if (renorm_3to2nd == 3) {
        int mapping_xyz[3];
        for (isymm = 0; isymm < symmetry->SymmListWithMap_ref.size(); isymm++) {

            // make mapping of xyz by the rotation matrix
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                mapping_xyz[ixyz1] = -1;
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    if (std::fabs(std::fabs(symmetry->SymmListWithMap_ref[isymm].rot[ixyz1 * 3 + ixyz2]) - 1.0) < eps6)
                    {
                        mapping_xyz[ixyz2] = ixyz1;
                    }
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                if (mapping_xyz[ixyz1] == -1) {
                    exit("calculate_delv2_delumn_finite_difference",
                         "RENORM_3TO2ND == 3 cannot be used for this material.");
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    // skip if dphi2_dumn_realspace[ixyz1][ixyz2] is not included in the input
                    if (exist_in(ixyz1, ixyz2) == 0) {
                        continue;
                    }

                    for (iat1 = 0; iat1 < natmin; iat1++) {

                        iat1_2 = symmetry->SymmListWithMap_ref[isymm].mapping[iat1];
                        ixyz1_2 = mapping_xyz[ixyz1];
                        ixyz2_2 = mapping_xyz[ixyz2];

                        for (i1 = 0; i1 < ntran; i1++) {
                            if (system->get_map_p2s(0)[iat1_2][i1] ==
                                symm_mapping_s[isymm][system->get_map_p2s(0)[iat1][0]])
                            {
                                itran1 = i1;
                            }
                        }

                        for (iat2 = 0; iat2 < nat; iat2++) {
                            iat2_2 = symm_mapping_s[isymm][iat2]; // temporary
                            iat2_2_prim = system->get_map_s2p(0)[iat2_2].atom_num;
                            itran2 = system->get_map_s2p(0)[iat2_2].tran_num;

                            iat2_2 = system->get_map_p2s(0)[iat2_2_prim][inv_translation_mapping[itran1][itran2]];

                            for (ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                                for (ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                                    ixyz3_2 = mapping_xyz[ixyz3];
                                    ixyz4_2 = mapping_xyz[ixyz4];

                                    dphi2_dumn_realspace_symm[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2]
                                                             [iat2_2 * 3 + ixyz4_2] +=
                                        dphi2_dumn_realspace_in[ixyz1][ixyz2][iat1 * 3 + ixyz3][iat2 * 3 + ixyz4] *
                                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz1_2 * 3 + ixyz1] *
                                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz2_2 * 3 + ixyz2] *
                                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz3_2 * 3 + ixyz3] *
                                        symmetry->SymmListWithMap_ref[isymm].rot[ixyz4_2 * 3 + ixyz4];

                                    count_tmp[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2][iat2_2 * 3 + ixyz4_2]++;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                for (i1 = 0; i1 < natmin * 3; i1++) {
                    for (i2 = 0; i2 < nat * 3; i2++) {
                        if (count_tmp[ixyz1][ixyz2][i1][i2] == 0) {
                            std::cout << "Warning: dphi2_dumn_realspace[" << ixyz1 << "][" << ixyz2 << "][" << i1
                                      << "][" << i2 << "] is not given\n";
                            std::cout << "The corresponding component is set zero.\n";
                            dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] = 0.0;
                        } else {
                            dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] /= count_tmp[ixyz1][ixyz2][i1][i2];
                        }
                    }
                }
            }
        }
    }

    deallocate(symm_mapping_s);
    deallocate(inv_translation_mapping);

    // for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
    //     for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
    //         for (i1 = 0; i1 < natmin3; i1++) {
    //             for (i2 = 0; i2 < nat3; i2++) {
    //                 std::cout << "dhi2     : " << std::setw(3) << ixyz1 << std::setw(3) << ixyz2 << std::setw(6) << i1
    //                           << std::setw(6) << i2 << " : " << std::setw(15) << std::scientific
    //                           << dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] << '\n';
    //                 std::cout << "dhi2_symm: " << std::setw(3) << ixyz1 << std::setw(3) << ixyz2 << std::setw(6) << i1
    //                           << std::setw(6) << i2 << " : " << std::setw(15) << std::scientific
    //                           << dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] << '\n';
    //             }
    //         }
    //     }
    // }

    // Fourier transform and interpolate
    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_new, ns, ns, nk_interpolate);
    allocate(dymat_tmp, ns, ns);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {

            // put dphi2_dumn_realspace_symm to std::vector<FcsClassExtent> format
            fc2_tmp.clear();
            for (i1 = 0; i1 < natmin * 3; i1++) {
                for (i2 = 0; i2 < nat * 3; i2++) {
                    FcsClassExtent fce_tmp;
                    fce_tmp.atm1 = i1 / 3;
                    fce_tmp.atm2 = i2 / 3;
                    fce_tmp.xyz1 = i1 % 3;
                    fce_tmp.xyz2 = i2 % 3;
                    fce_tmp.cell_s = 0;
                    fce_tmp.fcs_val = dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2];
                    fc2_tmp.push_back(fce_tmp);
                }
            }

            // Fourier transform
            for (ik = 0; ik < nk_interpolate; ik++) {

                dynamical->calc_analytic_k(kmesh_coarse->xk[ik], fc2_tmp, dymat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        dymat_q[is1][is2][ik] = dymat_tmp[is1][is2];
                    }
                }
            }

            // interpolation
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    fftw_plan plan = fftw_plan_dft_3d(nk1,
                                                      nk2,
                                                      nk3,
                                                      reinterpret_cast<fftw_complex *>(dymat_q[is1][is2]),
                                                      reinterpret_cast<fftw_complex *>(dymat_new[is1][is2]),
                                                      FFTW_FORWARD,
                                                      FFTW_ESTIMATE);
                    fftw_execute(plan);
                    fftw_destroy_plan(plan);

                    for (ik = 0; ik < nk_interpolate; ++ik) {
                        dymat_new[is1][is2][ik] /= static_cast<double>(nk_interpolate);
                    }
                }
            }

            for (ik = 0; ik < nk; ik++) {
                dynamical->r2q(kmesh_dense->xk[ik], nk1, nk2, nk3, ns, mindist_list, dymat_new, dymat_tmp);

                // (alpha,mu) representation in k-space (this is temporary)
                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        del_v2_strain_from_cubic_alphamu[ixyz1 * 3 + ixyz2][ik][is * ns + js] = dymat_tmp[is][js];
                    }
                }

                // transform to mode representation
                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        evec_tmp(is, js) = evec_harmonic[ik][js][is]; // transpose
                        dymat_tmp_alphamu(is, js) = dymat_tmp[is][js];
                    }
                }
                dymat_tmp_mode = evec_tmp.adjoint() * dymat_tmp_alphamu * evec_tmp;

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        del_v2_del_umn[ixyz1 * 3 + ixyz2][ik][is * ns + js] = dymat_tmp_mode(is, js);
                    }
                }
            }
        }
    }

    // Assign ASR to del_v2_del_umn from here.
    // set elements of acoustic mode at Gamma point exactly zero

    double threshold_acoustic = 1.0e-16;
    int count_acoustic = 0;

    const auto complex_zero = std::complex<double>(0.0, 0.0);

    int *is_acoustic;
    allocate(is_acoustic, ns);

    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < threshold_acoustic) {
            is_acoustic[is] = 1;
            count_acoustic++;
        } else {
            is_acoustic[is] = 0;
        }
    }

    // check number of acoustic modes
    if (count_acoustic != 3) {
        exit("calculate_delv2_delumn_finite_difference", "the number of detected acoustic modes is not three.");
    }

    // set acoustic sum rule (ASR)
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is = 0; is < ns; is++) {
            if (is_acoustic[is] == 0) {
                continue;
            }
            for (js = 0; js < ns; js++) {
                del_v2_del_umn[ixyz1][0][is * ns + js] = complex_zero;
                del_v2_del_umn[ixyz1][0][js * ns + is] = complex_zero;
            }
        }
    }

    deallocate(dphi2_dumn_realspace_symm);
    deallocate(dphi2_dumn_realspace_in);
    deallocate(count_tmp);

    deallocate(dymat_q);
    deallocate(dymat_tmp);
    deallocate(dymat_new);

    deallocate(is_acoustic);
}

void Relaxation::renormalize_v0_from_umn(double &v0_with_umn, double v0_ref, double **eta_tensor, double *C1_array,
                                         double **C2_array, double ***C3_array, double **u_tensor, const double pvcell)
{
    int ixyz1;

    constexpr double factor1 = 0.5;
    constexpr double factor2 = 1.0 / 6.0;

    v0_with_umn = v0_ref;

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            v0_with_umn += C1_array[ixyz1 * 3 + ixyz2] * eta_tensor[ixyz1][ixyz2];
            for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                for (int ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                    v0_with_umn += factor1 * C2_array[ixyz1 * 3 + ixyz2][ixyz3 * 3 + ixyz4] * eta_tensor[ixyz1][ixyz2] *
                                   eta_tensor[ixyz3][ixyz4];
                    for (int ixyz5 = 0; ixyz5 < 3; ixyz5++) {
                        for (int ixyz6 = 0; ixyz6 < 3; ixyz6++) {
                            v0_with_umn += factor2 * C3_array[ixyz1 * 3 + ixyz2][ixyz3 * 3 + ixyz4][ixyz5 * 3 + ixyz6] *
                                           eta_tensor[ixyz1][ixyz2] * eta_tensor[ixyz3][ixyz4] *
                                           eta_tensor[ixyz5][ixyz6];
                        }
                    }
                }
            }
        }
    }

    // add pV term
    Eigen::Vector3d vec_tmp1, vec_tmp2, vec_tmp3;
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        vec_tmp1[ixyz1] = u_tensor[0][ixyz1];
        vec_tmp2[ixyz1] = u_tensor[1][ixyz1];
        vec_tmp3[ixyz1] = u_tensor[2][ixyz1];
    }
    vec_tmp1[0] += 1.0;
    vec_tmp2[1] += 1.0;
    vec_tmp3[2] += 1.0;

    const double det_F_tensor = std::abs(vec_tmp1.dot(vec_tmp2.cross(vec_tmp3)));
    v0_with_umn += pvcell * det_F_tensor;
}

void Relaxation::renormalize_v1_from_umn(std::complex<double> *v1_with_umn, const std::complex<double> *const v1_ref,
                                         const std::complex<double> *const *const del_v1_del_umn,
                                         const std::complex<double> *const *const del2_v1_del_umn2,
                                         const std::complex<double> *const *const del3_v1_del_umn3,
                                         const double *const *const u_tensor) const
{
    const auto ns = dynamical->neval;

    constexpr double factor1 = 0.5;
    constexpr double factor2 = 1.0 / 6.0;

    for (int is = 0; is < ns; is++) {
        // original 1st-order IFCs
        v1_with_umn[is] = v1_ref[is];

        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                // renormalization from harmonic IFCs
                v1_with_umn[is] += del_v1_del_umn[ixyz1 * 3 + ixyz2][is] * u_tensor[ixyz1][ixyz2];

                for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    for (int ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                        // renormalization from cubic IFCs
                        int ixyz_comb = ixyz1 * 27 + ixyz2 * 9 + ixyz3 * 3 + ixyz4;
                        v1_with_umn[is] +=
                            factor1 * del2_v1_del_umn2[ixyz_comb][is] * u_tensor[ixyz1][ixyz2] * u_tensor[ixyz3][ixyz4];

                        for (int ixyz5 = 0; ixyz5 < 3; ixyz5++) {
                            for (int ixyz6 = 0; ixyz6 < 3; ixyz6++) {
                                // renormalization from quartic IFCs
                                ixyz_comb = ixyz1 * 243 + ixyz2 * 81 + ixyz3 * 27 + ixyz4 * 9 + ixyz5 * 3 + ixyz6;
                                v1_with_umn[is] += factor2 * del3_v1_del_umn3[ixyz_comb][is] * u_tensor[ixyz1][ixyz2] *
                                                   u_tensor[ixyz3][ixyz4] * u_tensor[ixyz5][ixyz6];
                            }
                        }
                    }
                }
            }
        }
    }
}

void Relaxation::renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse,
                                         const std::vector<int> &kmap_coarse_to_dense,
                                         std::complex<double> **delta_v2_renorm, std::complex<double> ***del_v2_del_umn,
                                         std::complex<double> ***del2_v2_del_umn2, double **u_tensor) const
{
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    unsigned int ik, knum;
    unsigned int is1, is2;
    int ixyz1, ixyz2;
    int ixyz, ixyz11, ixyz12, ixyz21, ixyz22, itmp;

    const auto ns2 = ns * ns;
    const auto nkns2 = nk_interpolate * ns2;

#pragma omp parallel for private(ik, is1, is2, knum, ixyz1, ixyz2, ixyz, ixyz11, ixyz12, ixyz21, ixyz22, itmp)
    for (int iks = 0; iks < nkns2; ++iks) {
        ik = iks / ns2;
        is1 = (iks % ns2) / ns;
        is2 = iks % ns;

        knum = kmap_coarse_to_dense[ik];

        // initialize delta_v2_renorm
        delta_v2_renorm[ik][is1 * ns + is2] = 0.0;

        // renormalization from cubic IFCs
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                delta_v2_renorm[ik][is1 * ns + is2] +=
                    del_v2_del_umn[ixyz1 * 3 + ixyz2][knum][is1 * ns + is2] * u_tensor[ixyz1][ixyz2];
            }
        }

        // renormalization from quartic IFCs
        for (ixyz = 0; ixyz < 81; ixyz++) {
            itmp = ixyz;
            ixyz22 = itmp % 3;
            itmp /= 3;
            ixyz21 = itmp % 3;
            itmp /= 3;
            ixyz12 = itmp % 3;
            ixyz11 = itmp / 3;

            delta_v2_renorm[ik][is1 * ns + is2] += 0.5 * del2_v2_del_umn2[ixyz][knum][is1 * ns + is2] *
                                                   u_tensor[ixyz11][ixyz12] * u_tensor[ixyz21][ixyz22];
        }
    }
}

void Relaxation::renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                         std::complex<double> ***v3_with_umn, std::complex<double> ***v3_ref,
                                         std::complex<double> ****del_v3_del_umn, double **u_tensor) const
{
    const auto nk_scph = kmesh_dense->nk;
    //    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    unsigned int ik;
    unsigned int is1, is2, is3;
    unsigned int ixyz1, ixyz2;

    const auto ns2 = ns * ns;
    const auto ns3 = ns * ns2;
    const auto nkns3 = nk_scph * ns3;

#pragma omp parallel for private(ik, is1, is2, is3, ixyz1, ixyz2)
    for (int iks = 0; iks < nkns3; ++iks) {
        ik = iks / ns3;
        is1 = (iks % ns3) / ns2;
        is2 = (iks % ns2) / ns;
        is3 = iks % ns;

        // initialize v3_with_umn
        v3_with_umn[ik][is1][is2 * ns + is3] = v3_ref[ik][is1][is2 * ns + is3];

        // renormalization from cubic IFCs
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                v3_with_umn[ik][is1][is2 * ns + is3] +=
                    del_v3_del_umn[ixyz1 * 3 + ixyz2][ik][is1][is2 * ns + is3] * u_tensor[ixyz1][ixyz2];
            }
        }
    }
}

void Relaxation::renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_coarse,
                                        const KpointMeshUniform *kmesh_dense, std::complex<double> *v1_renorm,
                                        std::complex<double> *v1_ref, std::complex<double> **delta_v2_array_original,
                                        std::complex<double> ***v3_ref, std::complex<double> ***v4_ref,
                                        double *q0) const
{
    int is1, is2;
    const auto ik_irred0 = kmesh_coarse->kpoint_map_symmetry[0].knum_irred_orig;
    const auto ns = dynamical->neval;
    const auto factor = 0.5 * 4.0 * kmesh_dense->nk;
    const auto factor2 = 1.0 / 6.0 * 4.0 * kmesh_dense->nk;

    // renormalize v1 array
    for (int is = 0; is < ns; is++) {

        v1_renorm[is] = v1_ref[is];
        v1_renorm[is] += omega2_harmonic[0][is] * q0[is]; // original v2 is assumed to be diagonal

        for (is1 = 0; is1 < ns; is1++) {
            v1_renorm[is] += delta_v2_array_original[0][is * ns + is1] * q0[is1];
        }

        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                v1_renorm[is] += factor * v3_ref[0][is][is1 * ns + is2] * q0[is1] * q0[is2];
            }
        }

        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                for (int is3 = 0; is3 < ns; is3++) {

                    v1_renorm[is] += factor2 * v4_ref[ik_irred0 * kmesh_dense->nk][is * ns + is1][is2 * ns + is3] *
                                     q0[is1] * q0[is2] * q0[is3];
                    // the factor 4.0 appears due to the definition of v4_array = 1.0/(4.0*N_scph) Phi_4
                }
            }
        }
    }
}

void Relaxation::renormalize_v2_from_q0(std::complex<double> ***evec_harmonic, const KpointMeshUniform *kmesh_coarse,
                                        const KpointMeshUniform *kmesh_dense,
                                        const std::vector<int> &kmap_coarse_to_dense,
                                        std::complex<double> ****mat_transform_sym,
                                        std::complex<double> **delta_v2_renorm,
                                        std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                        std::complex<double> ***v4_ref, double *q0) const
{
    using namespace Eigen;

    int ik;
    int is1, is2, js1, js2;
    unsigned int knum, knum_interpolate;
    const auto nk_scph = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto factor = 4.0 * nk_scph;
    const auto factor2 = 4.0 * nk_scph * 0.5;

    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    std::complex<double> ***dymat_q;
    MatrixXcd Dymat(ns, ns);
    MatrixXcd evec_tmp(ns, ns);
    allocate(dymat_q, ns, ns, nk_interpolate);

    for (ik = 0; ik < nk_irred_interpolate; ik++) {

        knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        knum = kmap_coarse_to_dense[knum_interpolate];

        // calculate renormalization
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                Dymat(is1, is2) = complex_zero;
                for (js1 = 0; js1 < ns; js1++) {
                    // cubic reormalization
                    Dymat(is1, is2) += factor * v3_ref[knum][js1][is2 * ns + is1] * q0[js1];
                    // quartic renormalization
                    for (js2 = 0; js2 < ns; js2++) {
                        Dymat(is1, is2) +=
                            factor2 * v4_ref[ik * nk_scph][is1 * ns + is2][js1 * ns + js2] * q0[js1] * q0[js2];
                    }
                }
            }
        }

        // unitary transform Dymat
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                evec_tmp(is1, is2) = evec_harmonic[knum][is2][is1]; // transpose
            }
        }
        Dymat = evec_tmp * Dymat * evec_tmp.adjoint();

        // symmetrize dynamical matrix
        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse, mat_transform_sym, Dymat);

        // store to dymat_q
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                dymat_q[is1][is2][knum_interpolate] = Dymat(is1, is2);
            }
        }
    }

    // replicate dymat_q to all q
    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym, dymat_q);

    // copy to delta_v2_renorm
    for (ik = 0; ik < nk_interpolate; ik++) {
        knum = kmap_coarse_to_dense[ik];
        // unitary transform Dymat
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                Dymat(is1, is2) = dymat_q[is1][is2][ik];
                evec_tmp(is1, is2) = evec_harmonic[knum][is2][is1]; // transpose
            }
        }
        Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                delta_v2_renorm[ik][is1 * ns + is2] = Dymat(is1, is2);
            }
        }
    }

    for (ik = 0; ik < nk_interpolate; ik++) {
        for (is1 = 0; is1 < ns * ns; is1++) {
            delta_v2_renorm[ik][is1] += delta_v2_array_original[ik][is1];
        }
    }

    deallocate(dymat_q);
}

void Relaxation::renormalize_v3_from_q0(const KpointMeshUniform *kmesh_dense, const KpointMeshUniform *kmesh_coarse,
                                        std::complex<double> ***v3_renorm, std::complex<double> ***v3_ref,
                                        std::complex<double> ***v4_ref, double *q0) const
{
    const auto ns = dynamical->neval;
    const auto ik_irred0 = kmesh_coarse->kpoint_map_symmetry[0].knum_irred_orig;
    const auto nk_scph = kmesh_dense->nk;

    const auto ns2 = ns * ns;
    const auto ns3 = ns * ns2;
    const auto nkns3 = nk_scph * ns3;

    unsigned int ik, is1, is2, is3, js;

#pragma omp parallel for private(ik, is1, is2, is3, js)
    for (int iks = 0; iks < nkns3; ++iks) {
        ik = iks / ns3;
        is1 = (iks % ns3) / ns2;
        is2 = (iks % ns2) / ns;
        is3 = iks % ns;
        v3_renorm[ik][is1][is2 * ns + is3] = v3_ref[ik][is1][is2 * ns + is3];
        for (js = 0; js < ns; js++) {
            v3_renorm[ik][is1][is2 * ns + is3] +=
                v4_ref[ik_irred0 * nk_scph + ik][js * ns + is1][is2 * ns + is3] * q0[js];
        }
    }
}

void Relaxation::renormalize_v0_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense,
                                        double &v0_renorm, double v0_ref, std::complex<double> *v1_ref,
                                        std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                        std::complex<double> ***v4_ref, double *q0) const
{
    int is1, is2;
    const auto ns = dynamical->neval;
    const auto nk_scph = kmesh_dense->nk;
    constexpr double factor2 = 1.0 / 2.0;
    const double factor3 = 1.0 / 6.0 * 4.0 * nk_scph;
    ;
    const double factor4 = 1.0 / 24.0 * 4.0 * nk_scph;
    ;

    std::complex<double> v0_renorm_tmp = v0_ref;
    // renormalize from the 1st order, harmonic IFC
    for (is1 = 0; is1 < ns; is1++) {
        v0_renorm_tmp += v1_ref[is1] * q0[is1];
        v0_renorm_tmp += factor2 * omega2_harmonic[0][is1] * q0[is1] * q0[is1]; // original v2 is assumed to be diagonal
    }
    // renormalize from the delta_v2_array
    for (is1 = 0; is1 < ns; is1++) {
        for (is2 = 0; is2 < ns; is2++) {
            v0_renorm_tmp += factor2 * delta_v2_array_original[0][is1 * ns + is2] * q0[is1] * q0[is2];
        }
    }
    // renormalize from the cubic, quartic IFC
    for (is1 = 0; is1 < ns; is1++) {
        for (is2 = 0; is2 < ns; is2++) {
            for (int is3 = 0; is3 < ns; is3++) {
                v0_renorm_tmp += factor3 * v3_ref[0][is1][is2 * ns + is3] * q0[is1] * q0[is2] * q0[is3];
                for (int is4 = 0; is4 < ns; is4++) {
                    v0_renorm_tmp +=
                        factor4 * v4_ref[0][is2 * ns + is1][is3 * ns + is4] * q0[is1] * q0[is2] * q0[is3] * q0[is4];
                }
            }
        }
    }

    v0_renorm = v0_renorm_tmp.real();
}

void Relaxation::make_supercell_mapping_by_symmetry_operations(int **symm_mapping_s) const
{
    const auto nat = system->get_supercell(0).number_of_atoms;
    const auto ntran = system->get_map_p2s()[0].size();

    Eigen::Matrix3d rotmat;
    Eigen::Vector3d shift;
    Eigen::MatrixXd xtmp(nat, 3);
    int i, j;
    int iat1;

    xtmp = system->get_supercell(0).x_cartesian;

    // make mapping
    int isymm = -1;
    for (const auto &it: symmetry->SymmListWithMap_ref) {

        isymm++;

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                //                rotmat[i][j] = it.rot[3 * i + j];
                rotmat(i, j) = it.rot[3 * i + j];
            }
        }
        for (i = 0; i < 3; ++i) {
            shift[i] = it.shift[i];
        }
        shift = system->get_primcell().lattice_vector * shift;
        //rotvec(shift, shift, system->lavec_p);

        for (iat1 = 0; iat1 < nat; iat1++) {
            // apply symmetry operation in cartesian coordinate
            Eigen::Vector3d xr_tmp = rotmat * xtmp.row(iat1).transpose() + shift;
            //rotvec(xr_tmp, xtmp[iat1], rotmat);
            //for (i = 0; i < 3; i++) {
            //    xr_tmp[i] += shift[i];
            //}

            // transform to fractional coordinate of supercell
            //rotvec(xr_tmp, xr_tmp, system->rlavec_s);
            //for (i = 0; i < 3; i++) {
            //   xr_tmp[i] /= 2.0 * pi;
            //}
            xr_tmp = system->get_supercell(0).reciprocal_lattice_vector * xr_tmp * inv_tpi;

            for (i = 0; i < 3; i++) {
                xr_tmp[i] = std::fmod(xr_tmp[i] + 1.0, 1.0);
            }

            // search for corresponding atom in the supercell
            int atm_found = 0;
            for (int itran1 = 0; itran1 < ntran; itran1++) {
                int jat1 = system->get_map_p2s(0)[it.mapping[system->get_map_s2p(0)[iat1].atom_num]][itran1];
                int iflag = 1;
                for (i = 0; i < 3; i++) {
                    // This is for robustness when numerical error is present.
                    double dtmp =
                        std::min(std::fabs(system->get_supercell(0).x_fractional(jat1, i) - xr_tmp[i]),
                                 std::min(std::fabs(system->get_supercell(0).x_fractional(jat1, i) - xr_tmp[i] + 1.0),
                                          std::fabs(system->get_supercell(0).x_fractional(jat1, i) - xr_tmp[i] - 1.0)));
                    if (dtmp > eps6) {
                        iflag = 0;
                    }
                }
                if (iflag == 1) {
                    atm_found = 1;
                    symm_mapping_s[isymm][iat1] = jat1;
                    break;
                }
            }
            if (atm_found == 0) {
                exit("make_supercell_mapping_by_symmetry_operations", "corresponding atom is not found.");
            }
        }
    }

    //    deallocate(xtmp);

    // check one-to-one correspondence
    int *map_tmp;
    allocate(map_tmp, nat);

    for (isymm = 0; isymm < symmetry->SymmListWithMap_ref.size(); isymm++) {
        // initialize
        for (iat1 = 0; iat1 < nat; iat1++) {
            map_tmp[iat1] = 0;
        }
        // check
        for (iat1 = 0; iat1 < nat; iat1++) {
            map_tmp[symm_mapping_s[isymm][iat1]] = 1;
        }
        for (iat1 = 0; iat1 < nat; iat1++) {
            if (map_tmp[iat1] == 0) {
                exit("make_supercell_mapping_by_symmetry_operations",
                     " the mapping of atoms is not a one-to-one mapping.");
            }
        }
    }

    deallocate(map_tmp);
}

void Relaxation::make_inverse_translation_mapping(int **inv_translation_mapping) const
{
    const auto ntran = system->get_map_p2s(0)[0].size();

    int ixyz1;
    double x_tran1[3], x_tran2[3];

    for (int i1 = 0; i1 < ntran; i1++) {
        // bring i1-th primitive cell to the original primitive cell
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            x_tran1[ixyz1] = system->get_supercell(0).x_fractional(system->get_map_p2s(0)[0][i1], ixyz1) -
                             system->get_supercell(0).x_fractional(system->get_map_p2s(0)[0][0], ixyz1);
            x_tran1[ixyz1] = std::fmod(x_tran1[ixyz1] + 1.0, 1.0);
        }

        // check i2-th primitive cell is moved to i3-th primitive cell
        for (int i2 = 0; i2 < ntran; i2++) {
            int is_found = 0;
            for (int i3 = 0; i3 < ntran; i3++) {
                // calculate translation vector
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    x_tran2[ixyz1] = system->get_supercell(0).x_fractional(system->get_map_p2s(0)[0][i2], ixyz1) -
                                     system->get_supercell(0).x_fractional(system->get_map_p2s(0)[0][i3], ixyz1);
                    x_tran2[ixyz1] = std::fmod(x_tran2[ixyz1] + 1.0, 1.0);
                }

                // compare translation vector
                int itmp = 1;
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    double dtmp = std::min(std::fabs(x_tran1[ixyz1] - x_tran2[ixyz1]),
                                           std::fabs(x_tran1[ixyz1] - x_tran2[ixyz1] + 1.0));
                    dtmp = std::min(dtmp, std::fabs(x_tran1[ixyz1] - x_tran2[ixyz1] - 1.0));

                    if (dtmp > eps6) {
                        itmp = 0;
                        break;
                    }
                }
                if (itmp == 1) {
                    inv_translation_mapping[i1][i2] = i3;
                    is_found = 1;
                    break;
                }
            }
            if (is_found == 0) {
                exit("make_inverse_translation_mapping",
                     "failed to find the mapping of primitive cells for inverse translation operations.");
            }
        }
    }
}


void Relaxation::write_resfile_header(std::ofstream &fout_q0, std::ofstream &fout_u0,
                                      std::ofstream &fout_u_tensor) const
{
    const auto ns = dynamical->neval;

    int is1, iat1, ixyz1, ixyz2;
    std::string str_tmp, str_tmp2;

    // atomic displacement (normal coordinate)
    fout_q0 << "#";
    fout_q0 << std::setw(14) << "temp [K]";
    for (is1 = 0; is1 < ns; is1++) {
        fout_q0 << std::setw(15) << ("q_{" + std::to_string(is1) + "}");
    }
    fout_q0 << '\n';

    // atomic displacement
    fout_u0 << "#";
    fout_u0 << std::setw(14) << "temp [K]";
    for (iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            get_xyz_string(ixyz1, str_tmp);
            fout_u0 << std::setw(15) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
        }
    }
    fout_u0 << '\n';

    // if the cell shape is relaxed
    if (fout_u_tensor) {
        fout_u_tensor << "#";
        fout_u_tensor << std::setw(14) << "temp [K]";
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                get_xyz_string(ixyz1, str_tmp);
                get_xyz_string(ixyz2, str_tmp2);
                fout_u_tensor << std::setw(15) << ("u_{" + str_tmp + str_tmp2 + "}");
            }
        }
        fout_u_tensor << '\n';
    }
}

auto Relaxation::write_resfile_atT(const double *const q0, const double *const *const u_tensor, const double *const u0,
                                   const double temp, std::ofstream &fout_q0, std::ofstream &fout_u0,
                                   std::ofstream &fout_u_tensor) const -> void
{
    int is;
    const auto ns = dynamical->neval;

    if (fout_q0) {
        fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < ns; is++) {
            fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_q0 << '\n';
    }

    if (fout_u0) {
        fout_u0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < ns; is++) {
            fout_u0 << std::scientific << std::setw(15) << std::setprecision(6) << u0[is];
        }
        fout_u0 << '\n';
    }

    if (fout_u_tensor) {
        fout_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < 9; is++) {
            fout_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[is / 3][is % 3];
        }
        fout_u_tensor << '\n';
    }
}


void Relaxation::write_stepresfile_header_atT(std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                              std::ofstream &fout_step_u_tensor, const double temp) const
{
    const auto ns = dynamical->neval;

    int ixyz1;
    std::string str_tmp;

    if (fout_step_q0) {
        fout_step_q0 << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp << " K\n";
        fout_step_q0 << std::setw(6) << "step";
        for (int is1 = 0; is1 < ns; is1++) {
            fout_step_q0 << std::setw(15) << ("q_{" + std::to_string(is1) + "}");
        }
        fout_step_q0 << '\n';
    }

    if (fout_step_u0) {
        fout_step_u0 << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp << " K\n";
        fout_step_u0 << std::setw(6) << "step";
        for (int iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                get_xyz_string(ixyz1, str_tmp);
                fout_step_u0 << std::setw(15) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
            }
        }
        fout_step_u0 << '\n';
    }

    if (fout_step_u_tensor) {
        std::string str_tmp2;
        fout_step_u_tensor << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp
                           << " K\n";
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                get_xyz_string(ixyz1, str_tmp);
                get_xyz_string(ixyz2, str_tmp2);
                fout_step_u_tensor << std::setw(15) << ("u_{" + str_tmp + str_tmp2 + "}");
            }
        }
        fout_step_u_tensor << '\n';
    }
}

void Relaxation::write_stepresfile(const double *const q0, const double *const *const u_tensor, const double *const u0,
                                   const int i_str_loop, std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                   std::ofstream &fout_step_u_tensor) const
{
    const auto ns = dynamical->neval;

    int is, i1;

    if (fout_step_q0) {
        fout_step_q0 << std::setw(6) << i_str_loop;
        for (is = 0; is < ns; is++) {
            fout_step_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_step_q0 << '\n';
    }

    if (fout_step_u0) {
        fout_step_u0 << std::setw(6) << i_str_loop;
        for (is = 0; is < ns; is++) {
            fout_step_u0 << std::scientific << std::setw(15) << std::setprecision(6) << u0[is];
        }
        fout_step_u0 << '\n';
    }

    if (fout_step_u_tensor) {
        fout_step_u_tensor << std::setw(6) << i_str_loop;
        for (i1 = 0; i1 < 9; i1++) {
            fout_step_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[i1 / 3][i1 % 3];
        }
        fout_step_u_tensor << '\n';
    }
}

int Relaxation::get_xyz_string(const int ixyz, std::string &xyz_str)
{
    if (ixyz == 0) {
        xyz_str = "x";
    } else if (ixyz == 1) {
        xyz_str = "y";
    } else {
        xyz_str = "z";
    }
    return 0;
}

void Relaxation::calculate_eta_tensor(double **eta_tensor, const double *const *const u_tensor)
{
    for (auto i1 = 0; i1 < 3; i1++) {
        for (auto i2 = 0; i2 < 3; i2++) {
            eta_tensor[i1][i2] = 0.5 * (u_tensor[i1][i2] + u_tensor[i2][i1]);
            for (auto j = 0; j < 3; j++) {
                eta_tensor[i1][i2] += u_tensor[i1][j] * u_tensor[i2][j];
            }
        }
    }
}

void Relaxation::setInitialDistortion(const double (*u_tensor_in)[3])
{
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            init_u_tensor[i][j] = u_tensor_in[i][j];
        }
    }
}
