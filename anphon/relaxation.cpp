/*
relaxation.cpp

Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "relaxation.h"
#include <Eigen/Core>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <iomanip>
#include "dynamical.h"
#include "error.h"
#include "ifc_derivative.h"
#include "memory.h"
#include "optimizers.h"
#include "parsephon.h"
#include "scph.h"
#include "symmetry_core.h"
#include "system.h"
#include "timer.h"

extern "C"
{
#include "spglib.h"
}
// #include <boost/range/algorithm.hpp> // boost/range may induce compile error

using namespace PHON_NS;

Relaxation::Relaxation(PHON *phon) : Pointers(phon)
{
    set_default_variables();
    derivative_ifc = std::make_unique<DerivativeIFC>(*system,
                                                     *symmetry,
                                                     *fcs_phonon,
                                                     *dynamical,
                                                     *anharmonic_core,
                                                     mympi->my_rank,
                                                     mympi->nprocs);
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
    gradient_conv_tol = 0.0;      // disabled by default (step-size convergence only)
    cell_gradient_conv_tol = 0.0; // disabled by default (step-size convergence only)
    gdiis_control = 1;            // controlled GDIIS by default; GDIIS_PLAIN = 1 disables it

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
}

void Relaxation::create_optimizer(const size_t num_modes)
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    if (relax_algo == 1) {
        // Steepest descent is independent of the relaxation mode and dimension.
        optimizer = std::make_unique<SteepestDescent_Optimizer>(alpha_steepest_decent);
    } else if (relax_mode == RelaxationStrMode::CoordinatesOnly && relax_algo == 2) {
        optimizer = std::make_unique<Newton_Optimizer>(mixbeta_coord);
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell && relax_algo == 2) {
        optimizer = std::make_unique<CellCoord_Newton_Optimizer>(mixbeta_cell, mixbeta_coord);
    } else if (relax_mode == RelaxationStrMode::CoordinatesOnly && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes - 3, num_modes - 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init, gdiis_control != 0);
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes + 3, num_modes + 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init, gdiis_control != 0);
    }
}

void Relaxation::set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array) const
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    // if the shape of the unit cell is relaxed,
    // read elastic constants from file
    if (relax_mode == RelaxationStrMode::CoordinatesAndCell || relax_mode == RelaxationStrMode::PerturbativeQha) {
        read_C1_array(C1_array);
        read_elastic_constants(C2_array, C3_array);
        return;
    }
    // if the unit cell is fixed,
    // dummy values are set in the elastic constants
    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
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


void Relaxation::set_init_structure_atT(RelaxationStructureState &structure_state, bool &converged_prev,
                                        int &str_diverged, const int i_temp_loop, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic) const
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);
    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    int i1, i2;

    // The optimizer is null for relaxation modes that do not use a coordinate
    // optimizer (e.g. PerturbativeQha), so guard every access to it here.
    if (optimizer) optimizer->reset();

    if (str_diverged) {
        std::cout << " The crystal structure at the previous temperature is divergent.\n";
        std::cout << " read initial structure from input files.\n\n";

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

        // set initial strain
        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
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
        if (optimizer) optimizer->reset();

        return;
    }

    std::cout << " SET_INIT_STR = " << set_init_str << ":";

    if (set_init_str == 1) {
        std::cout << " set initial structure from the input file.\n\n";

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }
        } else {
            set_initial_strain(u_tensor);
        }
        converged_prev = false;
        if (optimizer) optimizer->reset();

    } else if (set_init_str == 2) {
        if (i_temp_loop == 0) {
            std::cout << " set initial structure from the input file.\n\n";

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            converged_prev = false;
            if (optimizer) optimizer->reset();
        } else {
            std::cout << " start from structure from the previous temperature.\n\n";
        }

    } else if (set_init_str == 3) {
        // read initial structure at initial temperature
        if (i_temp_loop == 0) {
            std::cout << " read initial structure from input files.\n\n";

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            if (optimizer) optimizer->reset();
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
            if (optimizer) optimizer->reset();
        } else {
            std::cout << " start from the structure at the previous temperature.\n\n";
        }
    }
}

void Relaxation::set_initial_q0(std::vector<double> &q0, std::complex<double> ***evec_harmonic) const
{
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;
    if (q0.size() != static_cast<std::size_t>(ns)) {
        q0.resize(ns);
    }

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

void Relaxation::set_initial_strain(std::array<std::array<double, 3>, 3> &u_tensor) const
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

void Relaxation::calculate_u0(const std::vector<double> &q0, std::vector<double> &u0, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;

    if (u0.size() != q0.size()) {
        u0.resize(q0.size());
    }

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
    RelaxationStructureState &structure_state, const std::complex<double> *const v1_array_atT,
    const double *const *const omega2_array, const std::complex<double> *const del_v0_strain_atT,
    const double *const *const C2_array, const std::complex<double> *const *const *const cmat_convert,
    const std::vector<int> &harm_optical_modes, double **omega2_harmonic, std::complex<double> ***evec_harmonic) const
{
    using namespace Eigen;
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &delta_q0 = structure_state.delta_q0;
    auto &delta_u0 = structure_state.delta_u0;
    auto &delta_umn = structure_state.delta_umn;
    auto &du0 = structure_state.du0;
    auto &du_tensor = structure_state.du_tensor;

    const auto ns = dynamical->neval;
    int is;

    MatrixXcd Cmat(ns, ns), v2_mat_full(ns, ns);
    std::vector<double> grad_vec;
    std::vector<double> delta_vec;
    std::vector<double> state_vec;
    std::vector<std::vector<double>> hessian_mat;

    MatrixXcd C2_mat_tmp(6, 6);
    VectorXcd del_v0_strain_vec(6);


    const double add_hess_diag_omega2 = pow2(add_hess_diag / Ry_to_kayser);

    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
        delta_vec.assign(ns - 3, 0.0);
        state_vec.assign(ns - 3, 0.0);
        grad_vec.assign(ns - 3, 0.0);
        hessian_mat.assign(ns - 3, std::vector<double>(ns - 3, 0.0));
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
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

    if (relax_algo >= 1) {

        // Newton (relax_algo == 2) and BFGS+GDIIS (relax_algo == 3) use the harmonic IFC
        // matrix as the (initial) Hessian. Steepest descent (relax_algo == 1) ignores the
        // Hessian, so skip building it.
        if (relax_algo >= 2) {
            int js;
            // prepare harmonic IFC matrix
            for (is = 0; is < ns; is++) {
                for (js = 0; js < ns; js++) {
                    Cmat(js, is) = cmat_convert[0][is][js]; // transpose
                    v2_mat_full(is, js) = 0.0;
                }
                v2_mat_full(is, is) = omega2_array[0][is];
            }
            v2_mat_full = Cmat.adjoint() * v2_mat_full * Cmat;

            // set hessian
            for (is = 0; is < ns - 3; is++) {
                for (js = 0; js < ns - 3; js++) {
                    hessian_mat[is][js] = v2_mat_full(harm_optical_modes[is], harm_optical_modes[js]).real();
                }
                hessian_mat[is][is] += add_hess_diag_omega2;
            }
        }

        // set gradient and current state over the optical modes
        for (is = 0; is < ns - 3; is++) {
            grad_vec[is] = v1_array_atT[harm_optical_modes[is]].real();
            state_vec[is] = q0[harm_optical_modes[is]];
        }

        // Steepest descent (relax_algo == 1) historically froze near-acoustic modes using a
        // coarser cutoff (|omega2_harmonic| < eps8) than the eps10 used to build
        // harm_optical_modes. Zero their gradient so their steepest-descent step stays
        // exactly zero, reproducing the original behavior.
        if (relax_algo == 1) {
            for (is = 0; is < ns - 3; is++) {
                if (std::fabs(omega2_harmonic[0][harm_optical_modes[is]]) < eps8) {
                    grad_vec[is] = 0.0;
                }
            }
        }


        if (relax_mode == RelaxationStrMode::CoordinatesOnly || relax_algo == 1) {
            int i1, i2;
            // Relax the internal coordinates only. This covers CoordinatesOnly mode (any
            // algorithm) and steepest descent (relax_algo == 1), which keeps the cell shape
            // fixed even in CoordinatesAndCell mode.
            optimizer->update_state(ns - 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
                q0[harm_optical_modes[is]] += delta_q0[harm_optical_modes[is]];
            }
            // keep the cell fixed: no strain step
            for (i1 = 0; i1 < 6; i1++) {
                delta_umn[i1] = 0.0;
            }
            // reset the strain to zero only in CoordinatesOnly mode (in CoordinatesAndCell
            // mode the current strain is left untouched)
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            }

            // std::cout << "grad_vec = ";
            // for (auto &val: grad_vec) {
            //     std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            // }
            // std::cout << '\n';
            // std::cout << "state_vec = ";
            // for (auto &val: state_vec) {
            //     std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            // }
            // std::cout << '\n';

        } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
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

            // std::cout << "del_v0_strain_vec\n";
            // std::cout << del_v0_strain_vec << '\n';
            // std::cout << "C2_mat_tmp = \n";
            // std::cout << C2_mat_tmp << '\n';

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

            // call optimizer
            optimizer->update_state(ns + 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            // std::cout << "update state";
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
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
}

void Relaxation::rescue_step_after_scp_failure(RelaxationStructureState &structure_state,
                                               const std::complex<double> *const v1_array_atT,
                                               const std::vector<int> &harm_optical_modes, double **omega2_harmonic,
                                               std::complex<double> ***evec_harmonic) const
{
    // Called instead of update_cell_coordinate when the SCP equation did not
    // converge at the current structure. The forces and stress evaluated from an
    // unconverged SCP solution are unreliable, so they are not given to the
    // optimizer: its history keeps only data from converged SCP solutions.
    // The structure is moved back halfway along the last step, so repeated
    // failures bisect toward the last structure where the SCP equation was
    // solvable. If there is no previous step to undo (failure at the first
    // structure iteration), a strongly damped steepest-descent step on the
    // unreliable force is taken so that the optimization can leave the initial
    // structure; the cell is kept fixed in that case.

    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &delta_q0 = structure_state.delta_q0;
    auto &delta_u0 = structure_state.delta_u0;
    auto &delta_umn = structure_state.delta_umn;
    auto &du0 = structure_state.du0;
    auto &du_tensor = structure_state.du_tensor;

    const auto ns = dynamical->neval;
    int is, i1, i2;

    double last_step_norm = 0.0;
    for (is = 0; is < ns; is++) {
        last_step_norm += delta_q0[is] * delta_q0[is];
    }
    for (is = 0; is < 6; is++) {
        last_step_norm += delta_umn[is] * delta_umn[is];
    }
    last_step_norm = std::sqrt(last_step_norm);

    if (last_step_norm > eps12) {
        constexpr double backtrack_ratio = 0.5;
        for (is = 0; is < ns; is++) {
            delta_q0[is] *= -backtrack_ratio;
            q0[is] += delta_q0[is];
        }
        for (is = 0; is < 6; is++) {
            delta_umn[is] *= -backtrack_ratio;
            if (is < 3) {
                u_tensor[is][is] += delta_umn[is];
            } else {
                i1 = (is + 1) % 3;
                i2 = (is + 2) % 3;
                u_tensor[i1][i2] += delta_umn[is];
                u_tensor[i2][i1] += delta_umn[is];
            }
        }
        std::cout << " Moving back halfway along the last step and retrying.\n";
    } else {
        // No accepted step exists yet: use the unreliable force, but with a very
        // small weight.
        const double alpha_rescue = 0.1 * alpha_steepest_decent;
        for (is = 0; is < ns; is++) {
            delta_q0[is] = 0.0;
        }
        for (const auto mode: harm_optical_modes) {
            delta_q0[mode] = -alpha_rescue * v1_array_atT[mode].real();
            q0[mode] += delta_q0[mode];
        }
        for (is = 0; is < 6; is++) {
            delta_umn[is] = 0.0;
        }
        std::cout << " No accepted step exists yet: taking a strongly damped steepest-descent\n"
                     " step on the unreliable force (the cell shape is kept fixed).\n";
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
}

std::string Relaxation::print_structure_and_symmetry(const RelaxationStructureState &structure_state,
                                                     const std::complex<double> *del_v0_del_umn_atT) const
{
    // Print the crystal structure described by structure_state (strained lattice
    // and displaced atomic positions), the Cauchy stress tensor and pressure when
    // the strain gradient is available (del_v0_del_umn_atT != nullptr), and the
    // space group detected by spglib. Returns the space group label, e.g.
    // "P4mm (#99)", for use in summary tables.

    using namespace Eigen;

    const auto &primcell = system->get_primcell();
    const auto natmin = primcell.number_of_atoms;

    // deformation gradient F = I + u
    Matrix3d Fmat = Matrix3d::Identity();
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            Fmat(i, j) += structure_state.u_tensor[i][j];
        }
    }

    const Matrix3d lavec_new = Fmat * primcell.lattice_vector; // columns are a1, a2, a3
    const Matrix3d lavec_new_inv = lavec_new.inverse();

    std::cout << "  Lattice vectors [Bohr]:\n";
    for (auto j = 0; j < 3; ++j) {
        std::cout << "   a" << j + 1 << " :";
        for (auto i = 0; i < 3; ++i) {
            std::cout << std::setw(15) << std::setprecision(8) << std::fixed << lavec_new(i, j);
        }
        std::cout << '\n';
    }

    std::vector<Vector3d> xf_new(natmin);
    std::cout << "  Atomic positions (fractional):\n";
    for (size_t iat = 0; iat < natmin; ++iat) {
        const Vector3d x_ref = primcell.x_fractional.row(iat).transpose();
        Vector3d r_cart = Fmat * (primcell.lattice_vector * x_ref);
        for (auto i = 0; i < 3; ++i) {
            r_cart(i) += structure_state.u0[3 * iat + i];
        }
        xf_new[iat] = lavec_new_inv * r_cart;

        std::cout << "   " << std::setw(4) << std::left << system->symbol_kd[primcell.kind[iat]] << std::right << " :";
        for (auto i = 0; i < 3; ++i) {
            std::cout << std::setw(15) << std::setprecision(8) << std::fixed << xf_new[iat](i);
        }
        std::cout << '\n';
    }
    std::cout << std::scientific;

    if (del_v0_del_umn_atT) {
        // Cauchy stress sigma = (dE/dF) F^T / (V_ref det F). del_v0_del_umn_atT is the
        // gradient of the optimized potential including the external-pressure term
        // p V, so the p V gradient p V F^{-T} is subtracted to recover the internal
        // stress. At full equilibrium, the pressure printed here equals the applied
        // pressure (STAT_PRESSURE).
        const auto detF = Fmat.determinant();
        const auto vol = primcell.volume * detF; // Bohr^3
        const auto gpa_to_ry_bohr3 = 1.0e9 / Ryd * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30;
        const auto p_ext = stat_pressure * gpa_to_ry_bohr3; // Ry/Bohr^3

        Matrix3d grad_total;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                grad_total(i, j) = del_v0_del_umn_atT[3 * i + j].real();
            }
        }
        const Matrix3d grad_internal = grad_total - p_ext * vol * Fmat.inverse().transpose();
        const Matrix3d stress = grad_internal * Fmat.transpose() / vol; // Ry/Bohr^3, tension positive

        const auto pressure_gpa = -stress.trace() / 3.0 / gpa_to_ry_bohr3;

        std::cout << "  Stress tensor [GPa]:\n";
        for (auto i = 0; i < 3; ++i) {
            std::cout << "   ";
            for (auto j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << std::setprecision(6) << stress(i, j) / gpa_to_ry_bohr3;
            }
            std::cout << '\n';
        }
        std::cout << "  Pressure :" << std::setw(15) << std::setprecision(6) << pressure_gpa << " [GPa]\n";
    }

    // space group detection by spglib
    double aa[3][3];
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            aa[i][j] = lavec_new(i, j);
        }
    }
    // Kept raw: spg_get_dataset takes double(*)[3] (spglib C ABI boundary).
    double(*position)[3];
    NDArray<int, 1> types;
    allocate(position, natmin);
    types.resize(natmin);
    for (size_t iat = 0; iat < natmin; ++iat) {
        for (auto i = 0; i < 3; ++i) {
            position[iat][i] = xf_new[iat](i);
        }
        types[iat] = primcell.kind[iat];
    }

    std::string spg_label = "detection failed";
    const auto spgdataset = spg_get_dataset(aa, position, types, static_cast<int>(natmin), symmetry->tolerance);
    if (spgdataset && spgdataset->spacegroup_number > 0) {
        spg_label =
            std::string(spgdataset->international_symbol) + " (#" + std::to_string(spgdataset->spacegroup_number) + ")";
    }
    std::cout << "  Space group :  " << spg_label << '\n';
    if (spgdataset) spg_free_dataset(spgdataset);

    deallocate(position);
    types.clear();

    return spg_label;
}

void Relaxation::print_optimization_history(const std::vector<StructOptStepRecord> &step_history, const double temp,
                                            const bool with_cell, const bool show_scp_column)
{
    // At-a-glance summary table of a structural optimization at one temperature.
    // show_scp_column controls the column marking whether the SCP equation
    // converged at each step (meaningless for QHA, which has no inner
    // self-consistency loop).

    if (step_history.empty()) return;

    auto print_rule = [&]() {
        std::cout << " ------------------------------------------------------------";
        if (with_cell) std::cout << "-------------";
        std::cout << '\n';
    };

    std::cout << "\n Optimization history at " << temp << " K :\n";
    print_rule();
    std::cout << "   step  ";
    if (show_scp_column) std::cout << " SCP    ";
    std::cout << "  du0 [Bohr]    du_tensor      |force|   ";
    if (with_cell) std::cout << "    |stress|  ";
    std::cout << "  space group\n";
    print_rule();
    std::cout << std::scientific << std::setprecision(3);
    for (std::size_t istep = 0; istep < step_history.size(); ++istep) {
        const auto &rec = step_history[istep];
        std::cout << std::setw(7) << istep + 1;
        if (show_scp_column) std::cout << (rec.scp_ok ? "   conv " : "   FAIL ");
        std::cout << std::setw(14) << rec.du0 << std::setw(13) << rec.du_tensor;
        if (rec.grad_norm >= 0.0) {
            std::cout << std::setw(13) << rec.grad_norm;
        } else {
            std::cout << std::setw(13) << "-";
        }
        if (with_cell) {
            if (rec.cell_grad_norm >= 0.0) {
                std::cout << std::setw(14) << rec.cell_grad_norm;
            } else {
                std::cout << std::setw(14) << "-";
            }
        }
        std::cout << "    " << rec.spacegroup << '\n';
    }
    print_rule();
}

void Relaxation::check_str_divergence(int &diverged, const RelaxationStructureState &structure_state) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;

    int i, j;

    int flag_diverged = 0;

    for (i = 0; i < static_cast<int>(q0.size()); i++) {
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
                                      DelVStrainData &del_v_strain, double **omega2_harmonic,
                                      std::complex<double> ***evec_harmonic, const RelaxationStrMode relax_mode,
                                      MinimumDistList ***mindist_list, const PhaseFactorStorage *phase_storage_in)
{
    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense->nk;

    // CoordinatesOnly: keep the unit cell fixed and relax internal coordinates
    // set renormalization from strain as zero
    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
        derivative_ifc->set_del_v_fixed_cell(nk, ns, del_v_strain);
        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // CoordinatesAndCell: relax both the cell shape and the internal coordinates.
    if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {

        derivative_ifc->set_del_v_relax_cell(kmesh_coarse,
                                             kmesh_dense,
                                             ns,
                                             del_v_strain,
                                             omega2_harmonic,
                                             evec_harmonic,
                                             renorm_2to1st,
                                             renorm_34to1st,
                                             renorm_3to2nd,
                                             strain_IFC_dir,
                                             mindist_list,
                                             phase_storage_in);

        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // PerturbativeQha: calculate lowest-order linear equation of QHA.
    if (relax_mode == RelaxationStrMode::PerturbativeQha) {

        derivative_ifc->set_del_v_relax_cell_linearQHA(kmesh_coarse,
                                                       kmesh_dense,
                                                       ns,
                                                       del_v_strain,
                                                       omega2_harmonic,
                                                       evec_harmonic,
                                                       renorm_2to1st,
                                                       renorm_34to1st,
                                                       renorm_3to2nd,
                                                       strain_IFC_dir,
                                                       mindist_list);

        if (mympi->my_rank == 0) timer->print_elapsed();
    }
}

void Relaxation::renormalize_v0_from_umn(double &v0_with_umn, double v0_ref,
                                         std::array<std::array<double, 3>, 3> &eta_tensor, double *C1_array,
                                         double **C2_array, double ***C3_array,
                                         const std::array<std::array<double, 3>, 3> &u_tensor, const double pvcell)
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
                                         const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
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
                v1_with_umn[is] += del_v_strain.del_v1(ixyz1 * 3 + ixyz2, is) * u_tensor[ixyz1][ixyz2];

                for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    for (int ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                        // renormalization from cubic IFCs
                        int ixyz_comb = ixyz1 * 27 + ixyz2 * 9 + ixyz3 * 3 + ixyz4;
                        v1_with_umn[is] += factor1 * del_v_strain.del2_v1(ixyz_comb, is) * u_tensor[ixyz1][ixyz2] *
                                           u_tensor[ixyz3][ixyz4];

                        for (int ixyz5 = 0; ixyz5 < 3; ixyz5++) {
                            for (int ixyz6 = 0; ixyz6 < 3; ixyz6++) {
                                // renormalization from quartic IFCs
                                ixyz_comb = ixyz1 * 243 + ixyz2 * 81 + ixyz3 * 27 + ixyz4 * 9 + ixyz5 * 3 + ixyz6;
                                v1_with_umn[is] += factor2 * del_v_strain.del3_v1(ixyz_comb, is) *
                                                   u_tensor[ixyz1][ixyz2] * u_tensor[ixyz3][ixyz4] *
                                                   u_tensor[ixyz5][ixyz6];
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
                                         std::complex<double> **delta_v2_renorm, const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
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
                    del_v_strain.del_v2[ixyz1 * 3 + ixyz2](knum, is1 * ns + is2) * u_tensor[ixyz1][ixyz2];
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

            delta_v2_renorm[ik][is1 * ns + is2] += 0.5 * del_v_strain.del2_v2[ixyz](knum, is1 * ns + is2) *
                                                   u_tensor[ixyz11][ixyz12] * u_tensor[ixyz21][ixyz22];
        }
    }
}

void Relaxation::renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                         std::complex<double> ***v3_with_umn, std::complex<double> ***v3_ref,
                                         const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
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
                    del_v_strain.del_v3[ixyz1 * 3 + ixyz2][ik](is1, is2 * ns + is3) * u_tensor[ixyz1][ixyz2];
            }
        }
    }
}

void Relaxation::renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_coarse,
                                        const KpointMeshUniform *kmesh_dense, std::complex<double> *v1_renorm,
                                        std::complex<double> *v1_ref, std::complex<double> **delta_v2_array_original,
                                        std::complex<double> ***v3_ref, std::complex<double> ***v4_ref,
                                        const std::vector<double> &q0) const
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
                                        std::complex<double> ***v4_ref, const std::vector<double> &q0) const
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

    NDArray<std::complex<double>, 3> dymat_q;
    MatrixXcd Dymat(ns, ns);
    MatrixXcd evec_tmp(ns, ns);
    dymat_q.resize(ns, ns, nk_interpolate);

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

    dymat_q.clear();
}

void Relaxation::renormalize_v3_from_q0(const KpointMeshUniform *kmesh_dense, const KpointMeshUniform *kmesh_coarse,
                                        std::complex<double> ***v3_renorm, std::complex<double> ***v3_ref,
                                        std::complex<double> ***v4_ref, const std::vector<double> &q0) const
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
                                        std::complex<double> ***v4_ref, const std::vector<double> &q0) const
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

void Relaxation::write_resfile_atT(const RelaxationStructureState &structure_state, const double temp,
                                   std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;
    int is;

    if (fout_q0) {
        fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < static_cast<int>(q0.size()); is++) {
            fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_q0 << '\n';
    }

    if (fout_u0) {
        fout_u0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < static_cast<int>(u0.size()); is++) {
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

void Relaxation::write_stepresfile(const RelaxationStructureState &structure_state, const int i_str_loop,
                                   std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                   std::ofstream &fout_step_u_tensor) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;

    int is, i1;

    if (fout_step_q0) {
        fout_step_q0 << std::setw(6) << i_str_loop;
        for (is = 0; is < static_cast<int>(q0.size()); is++) {
            fout_step_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_step_q0 << '\n';
    }

    if (fout_step_u0) {
        fout_step_u0 << std::setw(6) << i_str_loop;
        for (is = 0; is < static_cast<int>(u0.size()); is++) {
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

void Relaxation::calculate_eta_tensor(std::array<std::array<double, 3>, 3> &eta_tensor,
                                      const std::array<std::array<double, 3>, 3> &u_tensor)
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
