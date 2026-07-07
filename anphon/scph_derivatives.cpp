/*
 scph_derivatives.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Functions for computing derivatives of the free energy with respect to
 strain and atomic displacements. These are used in quasi-harmonic
 approximation (QHA) and structural relaxation calculations.

 Functions included:
 - calculate_del_v0_del_umn_renorm: Renormalized free energy derivatives
 - compute_anharmonic_v1_array: First-order anharmonic contributions
 - compute_anharmonic_del_v0_del_umn: Anharmonic free energy derivatives
 - get_derivative_central_diff: Numerical derivatives using central difference
*/

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "relaxation.h"
#include "scph_qha_common.h"
#include "thermodynamics.h"

using namespace PHON_NS;

void ScphQhaCommon::calculate_del_v0_del_umn_renorm(std::complex<double> *del_v0_del_umn_renorm, double *C1_array,
                                                    double **C2_array, double ***C3_array,
                                                    std::array<std::array<double, 3>, 3> &eta_tensor,
                                                    const std::array<std::array<double, 3>, 3> &u_tensor,
                                                    const DelVStrainData &del_v_strain, const std::vector<double> &q0,
                                                    double pvcell, const KpointMeshUniform *kmesh_dense_in)
{

    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense_in->nk;
    NDArray<double, 2> del_eta_del_u;
    NDArray<double, 1> del_v0_del_eta;
    NDArray<double, 1> del_v0_strain_with_strain;

    NDArray<std::complex<double>, 2> del_v1_del_umn_with_umn;
    NDArray<std::complex<double>, 3> del_v2_del_umn_with_umn;

    del_eta_del_u.resize(9, 9);
    del_v0_del_eta.resize(9);
    del_v0_strain_with_strain.resize(9);

    del_v1_del_umn_with_umn.resize(9, ns);
    del_v2_del_umn_with_umn.resize(9, ns, ns);

    const double factor = 1.0 / 6.0 * 4.0 * nk;
    int i1, i2, i3, ixyz1, ixyz2, ixyz3, ixyz4;
    int is1, is2, is3;


    // calculate the derivative of eta_tensor by u_tensor
    for (i1 = 0; i1 < 9; i1++) {
        ixyz1 = i1 / 3;
        ixyz2 = i1 % 3;
        for (i2 = 0; i2 < 9; i2++) {
            ixyz3 = i2 / 3;
            ixyz4 = i2 % 3;

            del_eta_del_u[i1][i2] = 0.0;

            if (ixyz1 == ixyz3 && ixyz2 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz2 == ixyz3 && ixyz1 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz1 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz2][ixyz4];
            }
            if (ixyz2 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz1][ixyz4];
            }
        }
    }

    // calculate del_v0_del_eta
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_eta[i1] = C1_array[i1];
        for (i2 = 0; i2 < 9; i2++) {
            del_v0_del_eta[i1] += C2_array[i1][i2] * eta_tensor[i2 / 3][i2 % 3];
            for (i3 = 0; i3 < 9; i3++) {
                del_v0_del_eta[i1] +=
                    0.5 * C3_array[i1][i2][i3] * eta_tensor[i2 / 3][i2 % 3] * eta_tensor[i3 / 3][i3 % 3];
            }
        }
    }

    // calculate contribution from v0 without atomic displacements
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_strain_with_strain[i1] = 0.0;
        for (i2 = 0; i2 < 9; i2++) {
            del_v0_strain_with_strain[i1] += del_eta_del_u[i2][i1] * del_v0_del_eta[i2];
        }
    }

    // add pV term
    double F_tensor[3][3]; // F_{mu nu} = delta_{mu nu} + u_{mu nu}
    for (i1 = 0; i1 < 3; i1++) {
        for (i2 = 0; i2 < 3; i2++) {
            F_tensor[i1][i2] = u_tensor[i1][i2];
        }
        F_tensor[i1][i1] += 1.0;
    }
    for (i1 = 0; i1 < 9; i1++) {
        is1 = i1 / 3;
        is2 = i1 % 3;
        ixyz1 = (is1 + 1) % 3;
        ixyz2 = (is1 + 2) % 3;
        ixyz3 = (is2 + 1) % 3;
        ixyz4 = (is2 + 2) % 3;

        del_v0_strain_with_strain[i1] += pvcell * (F_tensor[ixyz1][ixyz3] * F_tensor[ixyz2][ixyz4] -
                                                   F_tensor[ixyz1][ixyz4] * F_tensor[ixyz2][ixyz3]);
    }

    // calculate del_v1_del_umn
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_with_umn[i1][is1] = del_v_strain.del_v1(i1, is1);
            for (i2 = 0; i2 < 9; i2++) {
                del_v1_del_umn_with_umn[i1][is1] += del_v_strain.del2_v1(i1 * 9 + i2, is1) * u_tensor[i2 / 3][i2 % 3];
                for (i3 = 0; i3 < 9; i3++) {
                    del_v1_del_umn_with_umn[i1][is1] += 0.5 * del_v_strain.del3_v1(i1 * 81 + i2 * 9 + i3, is1) *
                                                        u_tensor[i2 / 3][i2 % 3] * u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    // calculate del_v2_del_umn
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v2_del_umn_with_umn[i1][is1][is2] = del_v_strain.del_v2[i1](0, is1 * ns + is2);
                for (i2 = 0; i2 < 9; i2++) {
                    del_v2_del_umn_with_umn[i1][is1][is2] +=
                        del_v_strain.del2_v2[i1 * 9 + i2](0, is1 * ns + is2) * u_tensor[i2 / 3][i2 % 3];
                }
            }
        }
    }

    // calculate del_v0_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_renorm[i1] = del_v0_strain_with_strain[i1];
        for (is1 = 0; is1 < ns; is1++) {
            del_v0_del_umn_renorm[i1] += del_v1_del_umn_with_umn[i1][is1] * q0[is1];
            for (is2 = 0; is2 < ns; is2++) {
                del_v0_del_umn_renorm[i1] += 0.5 * del_v2_del_umn_with_umn[i1][is1][is2] * q0[is1] * q0[is2];
                for (is3 = 0; is3 < ns; is3++) {
                    del_v0_del_umn_renorm[i1] +=
                        factor * del_v_strain.del_v3[i1][0](is1, is2 * ns + is3) * q0[is1] * q0[is2] * q0[is3];
                }
            }
        }
    }


    del_eta_del_u.clear();
    del_v0_del_eta.clear();
    del_v0_strain_with_strain.clear();
    del_v1_del_umn_with_umn.clear();
    del_v2_del_umn_with_umn.clear();
}


void ScphQhaCommon::compute_anharmonic_v1_array(std::complex<double> *v1_SCP, std::complex<double> *v1_renorm,
                                                std::complex<double> ***v3_renorm, std::complex<double> ***cmat_convert,
                                                double **omega2_anharm_T, const double T_in,
                                                const KpointMeshUniform *kmesh_dense_in)
{
    using namespace Eigen;

    int is;
    std::complex<double> Qtmp;
    const auto ns = dynamical->neval;
    const auto nk_scph = kmesh_dense_in->nk;

    MatrixXcd Cmat(ns, ns);
    MatrixXcd v3mat_original_mode(ns, ns), v3mat_tmp(ns, ns);

    // get gradient of the BO surface
    for (is = 0; is < ns; is++) {
        v1_SCP[is] = v1_renorm[is];
    }

    // calculate SCP renormalization
    for (is = 0; is < ns; is++) {
        for (int ik = 0; ik < nk_scph; ik++) {
            // unitary transform phi3 to SCP mode
            for (int js1 = 0; js1 < ns; js1++) {
                for (int js2 = 0; js2 < ns; js2++) {
                    Cmat(js2, js1) = cmat_convert[ik][js1][js2]; // transpose
                    v3mat_original_mode(js1, js2) = v3_renorm[ik][is][js1 * ns + js2];
                }
            }
            v3mat_tmp = Cmat * v3mat_original_mode * Cmat.adjoint();

            // update v1_SCP
            int count_zero = 0;
            for (int js = 0; js < ns; js++) {
                double omega1_tmp = std::sqrt(std::fabs(omega2_anharm_T[ik][js]));
                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                    count_zero++;
                } else {
                    const auto factor = thermodynamics->disp_corr_factor(omega1_tmp, T_in);
                    Qtmp = std::complex<double>(factor, 0.0);
                }

                v1_SCP[is] += v3mat_tmp(js, js) * Qtmp;
            }
            if (ik == 0 && count_zero != 3) {
                std::cout << "Warning in compute_anharmonic_v1_array : ";
                std::cout << count_zero << " acoustic modes are detected in Gamma point.\n\n";
            } else if (ik != 0 && count_zero != 0) {
                std::cout << "Warning in compute_anharmonic_v1_array : ";
                std::cout << count_zero << " zero frequencies are detected in non-Gamma point (ik = " << ik << ").\n\n";
            }
        }
    }
}

void ScphQhaCommon::compute_anharmonic_del_v0_del_umn(std::complex<double> *del_v0_del_umn_SCP,
                                                      std::complex<double> *del_v0_del_umn_renorm,
                                                      const DelVStrainData &del_v_strain,
                                                      const std::array<std::array<double, 3>, 3> &u_tensor,
                                                      const std::vector<double> &q0,
                                                      std::complex<double> ***cmat_convert, double **omega2_anharm_T,
                                                      const double T_in, const KpointMeshUniform *kmesh_dense_in)
{

    using namespace Eigen;

    int nk = kmesh_dense_in->nk;
    int ns = dynamical->neval;
    double factor = 4.0 * static_cast<double>(nk);
    double factor2 = 1.0 / factor;
    std::vector<Eigen::MatrixXcd> del_v2_del_umn_renorm(9, Eigen::MatrixXcd::Zero(nk, ns * ns));

    int i1, i2;
    int ik;
    int is1, is2, is3, js, js1, js2;
    double omega1_tmp;
    std::complex<double> Qtmp;
    int count_zero;

    MatrixXcd Cmat(ns, ns);
    MatrixXcd del_v2_strain_mat_original_mode(ns, ns), del_v2_strain_mat(ns, ns);

    // calculate del_v2_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        for (ik = 0; ik < nk; ik++) {
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    del_v2_del_umn_renorm[i1](ik, is1 * ns + is2) = del_v_strain.del_v2[i1](ik, is1 * ns + is2);
                    // renormalization by strain
                    for (i2 = 0; i2 < 9; i2++) {
                        del_v2_del_umn_renorm[i1](ik, is1 * ns + is2) +=
                            del_v_strain.del2_v2[i1 * 9 + i2](ik, is1 * ns + is2) * u_tensor[i2 / 3][i2 % 3];
                    }
                    // renormalization by displace
                    for (is3 = 0; is3 < ns; is3++) {
                        del_v2_del_umn_renorm[i1](ik, is1 * ns + is2) +=
                            factor * del_v_strain.del_v3[i1][ik](is3, is2 * ns + is1) * q0[is3];
                    }
                }
            }
        }
    }

    // potential energy term
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_SCP[i1] = del_v0_del_umn_renorm[i1];
    }

    // SCP renormalization
    for (i1 = 0; i1 < 9; i1++) {
        for (ik = 0; ik < nk; ik++) {
            // unitary transform the derivative of harmonic IFCs to SCP mode
            for (js1 = 0; js1 < ns; js1++) {
                for (js2 = 0; js2 < ns; js2++) {
                    Cmat(js1, js2) = cmat_convert[ik][js1][js2];
                    del_v2_strain_mat_original_mode(js1, js2) = del_v2_del_umn_renorm[i1](ik, js1 * ns + js2);
                }
            }
            del_v2_strain_mat = Cmat.adjoint() * del_v2_strain_mat_original_mode * Cmat;

            // update del_v0_del_umn_SCP
            count_zero = 0;
            for (js = 0; js < ns; js++) {
                omega1_tmp = std::sqrt(std::fabs(omega2_anharm_T[ik][js]));
                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                    count_zero++;
                } else {
                    if (omega2_anharm_T[ik][js] < 0.0) {
                        std::cout
                            << "Warning in compute_anharmonic_del_v0_del_umn: squared SCP frequency is negative. ik = "
                            << ik << '\n';
                    }
                    const auto factor = thermodynamics->disp_corr_factor(omega1_tmp, T_in);
                    Qtmp = std::complex<double>(factor, 0.0);
                }

                del_v0_del_umn_SCP[i1] += factor2 * del_v2_strain_mat(js, js) * Qtmp;
            }
        }
    }
}

void ScphQhaCommon::get_derivative_central_diff(const double delta_t, const unsigned int nk, double **omega0,
                                                double **omega2, double **domega_dt)
{
    const auto ns = dynamical->neval;
    const auto inv_dt = 1.0 / (2.0 * delta_t);
    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            domega_dt[ik][is] = (omega2[ik][is] - omega0[ik][is]) * inv_dt;
            //    std::cout << "domega_dt = " << domega_dt[ik][is] << '\n';
        }
    }
}
