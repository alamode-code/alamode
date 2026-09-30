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
    int is1, is2;


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
    // EFIELD with e0 (fixed voltage): the constant -Omega_ref E0_i e0_imn
    if (relaxation->has_piezo_field_term()) {
        for (i1 = 0; i1 < 9; i1++) del_v0_strain_with_strain[i1] += relaxation->efield_strain_gradient[i1];
    }
    // EFIELD with B: -Omega_ref E0_i B_i,mn,pq u_pq
    if (relaxation->has_piezo2_field_term()) {
        for (i1 = 0; i1 < 9; i1++) {
            for (i2 = 0; i2 < 9; i2++) {
                del_v0_strain_with_strain[i1] +=
                    relaxation->efield_strain_curvature[i1 * 9 + i2] * u_tensor[i2 / 3][i2 % 3];
            }
        }
    }
    // EFIELD with Lambda: -E0_i sum_k Lambda_k,ib,mn u0_kb = -sum_s L_mn,s q0_s, from the
    // q0 argument (the Hessian check perturbs q0 independently of the workspace)
    if (!efield_lambda_mode.empty()) {
        for (i1 = 0; i1 < 9; i1++) {
            for (is1 = 0; is1 < ns; is1++) del_v0_strain_with_strain[i1] -= efield_lambda_mode[i1 * ns + is1] * q0[is1];
        }
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
    // cubic term sum_{is1,is2,is3} del_v3[i1](is1, is2*ns+is3) q0[is1] q0[is2] q0[is3]
    // = q0^T (D3 w) with w = q0 (x) q0: one GEMV per strain component instead of a
    // single-threaded ns^3 loop
    {
        Eigen::VectorXcd q0c(ns), w(ns * ns);
        for (is1 = 0; is1 < ns; is1++) {
            q0c(is1) = std::complex<double>(q0[is1], 0.0);
        }
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                w(is1 * ns + is2) = std::complex<double>(q0[is1] * q0[is2], 0.0);
            }
        }
        for (i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_renorm[i1] = del_v0_strain_with_strain[i1];
            for (is1 = 0; is1 < ns; is1++) {
                del_v0_del_umn_renorm[i1] += del_v1_del_umn_with_umn[i1][is1] * q0[is1];
                for (is2 = 0; is2 < ns; is2++) {
                    del_v0_del_umn_renorm[i1] += 0.5 * del_v2_del_umn_with_umn[i1][is1][is2] * q0[is1] * q0[is2];
                }
            }
            const Eigen::VectorXcd d3w = del_v_strain.del_v3[i1][0] * w; // (ns x ns^2) * ns^2
            del_v0_del_umn_renorm[i1] += factor * q0c.dot(d3w);          // q0 is real: dot() conjugates nothing
        }
    }

    // STRAIN_FC5: (1/24) dPhi4/du_mn q0^4 = (1/24) 4N sum_ab q4mn[Gamma](a,b) q0[a] q0[b]
    // (as the quartic term of Relaxation::renormalize_v0_from_q0)
    if (!fc5_q4mn.empty()) {
        const auto ns2 = static_cast<std::size_t>(ns) * ns;
        const double factor4 = 1.0 / 24.0 * 4.0 * nk;
        for (std::size_t c = 0; c < fc5_mn.size(); c++) {
            const auto *q4 = fc5_q4mn.data() + (c * nk + ik_gamma_dense) * ns2;
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    del_v0_del_umn_renorm[fc5_mn[c]] += factor4 * q4[is1 * ns + is2] * q0[is1] * q0[is2];
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


int ScphQhaCommon::scp_occupation_matrix(const int ik, const std::complex<double> *const *cmat_at_k,
                                         const double *omega2_at_k, const double T_in, Eigen::MatrixXcd &G,
                                         std::vector<bool> *is_acoustic_out) const
{
    using namespace Eigen;
    const auto ns = dynamical->neval;

    // The acoustic modes at Gamma are excluded by their eigenvector character
    // (overlap with the harmonic acoustic subspace), not by frequency magnitude,
    // so that a soft optical mode with a nearly zero SCP frequency keeps its
    // contribution.
    std::vector<bool> is_acoustic_now;
    if (ik == ik_gamma_dense) {
        is_acoustic_now = classify_acoustic_modes_from_cmat(cmat_at_k);
    }
    if (is_acoustic_out != nullptr) {
        *is_acoustic_out = is_acoustic_now;
    }

    VectorXcd fvec(ns);
    int count_zero = 0;
    for (auto js = 0; js < ns; js++) {
        if (ik == ik_gamma_dense && is_acoustic_now[js]) {
            fvec(js) = 0.0;
            continue;
        }
        auto omega1_tmp = std::sqrt(std::fabs(omega2_at_k[js]));
        if (omega1_tmp < eps8) {
            omega1_tmp = eps8;
            count_zero++;
        }
        fvec(js) = std::complex<double>(thermodynamics->disp_corr_factor(omega1_tmp, T_in), 0.0);
    }

    MatrixXcd Cmat(ns, ns);
    for (auto a = 0; a < ns; a++) {
        for (auto js = 0; js < ns; js++) {
            Cmat(a, js) = cmat_at_k[a][js];
        }
    }
    G.noalias() = Cmat * fvec.asDiagonal() * Cmat.adjoint();
    return count_zero;
}

void ScphQhaCommon::compute_anharmonic_v1_array(std::complex<double> *v1_SCP, std::complex<double> *v1_renorm,
                                                std::complex<double> ***v3_renorm, std::complex<double> ***cmat_convert,
                                                double **omega2_anharm_T, const double T_in,
                                                const KpointMeshUniform *kmesh_dense_in)
{
    // v1_SCP[is] = v1_renorm[is] + sum_k sum_js f_js (C^T V3_is conj(C))(js,js)
    //            = v1_renorm[is] + sum_k sum_ab V3_is[a][b] G_k(a,b)
    // with V3_is[a][b] = v3_renorm[k][is][a*ns+b] and G_k = scp_occupation_matrix.
    // (The former code formed the full C^T V3_is conj(C) product for every is and
    // kept its diagonal: nk ns^4; the trace form is nk ns^3.)
    using namespace Eigen;
    using MatrixXcdRowMajor = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;

    const auto ns = dynamical->neval;
    const auto nk_scph = kmesh_dense_in->nk;

    // get gradient of the BO surface
    for (auto is = 0; is < ns; is++) {
        v1_SCP[is] = v1_renorm[is];
    }

    MatrixXcd G(ns, ns);

    // calculate SCP renormalization
    for (auto ik = 0; ik < nk_scph; ik++) {
        const auto count_zero = scp_occupation_matrix(ik, cmat_convert[ik], omega2_anharm_T[ik], T_in, G);
        if (count_zero != 0) {
            std::cout << "Warning in compute_anharmonic_v1_array : ";
            std::cout << count_zero << " non-acoustic zero frequencies are detected at ik = " << ik << ".\n\n";
        }

#pragma omp parallel for
        for (int is = 0; is < ns; is++) {
            Map<const MatrixXcdRowMajor> V3(v3_renorm[ik][is], ns, ns);
            v1_SCP[is] += V3.cwiseProduct(G).sum();
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
    using MatrixXcdRowMajor = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;

    const int nk = kmesh_dense_in->nk;
    const int ns = dynamical->neval;
    const double factor = 4.0 * static_cast<double>(nk);
    const double factor2 = 1.0 / factor;

    // del_v2_del_umn_renorm[i1 * nk + ik]: the strain vertices
    std::vector<MatrixXcdRowMajor> del_v2_del_umn_renorm(9 * nk, MatrixXcdRowMajor(ns, ns));

#pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < 9; i1++) {
        for (int ik = 0; ik < nk; ik++) {
            del_v2_del_umn_renorm[i1 * nk + ik] = strain_vertex(del_v_strain, u_tensor, q0, i1, ik, nk);
        }
    }

    // potential energy term
    for (auto i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_SCP[i1] = del_v0_del_umn_renorm[i1];
    }

    // SCP renormalization: sum_js f_js (C^+ M C)(js,js) = sum_ab M(a,b) G(b,a)
    // with M = del_v2_del_umn_renorm[i1 * nk + ik] and G = scp_occupation_matrix
    // (the former code formed the full C^+ M C product for each of the 9 strain
    // components and kept its diagonal).
    MatrixXcd G(ns, ns);
    std::vector<bool> is_acoustic_now;
    const auto ns2 = static_cast<std::size_t>(ns) * ns;
    // STRAIN_FC5: the occupation matrices of every k (row-major), see below
    std::vector<std::complex<double>> gall(dv4_fc5 ? nk * ns2 : 0);
    for (auto ik = 0; ik < nk; ik++) {
        scp_occupation_matrix(ik, cmat_convert[ik], omega2_anharm_T[ik], T_in, G, &is_acoustic_now);
        if (dv4_fc5) Map<MatrixXcdRowMajor>(gall.data() + ik * ns2, ns, ns) = G;
        for (auto js = 0; js < ns; js++) {
            if (ik == ik_gamma_dense && is_acoustic_now[js]) {
                continue;
            }
            if (omega2_anharm_T[ik][js] < 0.0 && std::sqrt(std::fabs(omega2_anharm_T[ik][js])) >= eps8) {
                std::cout << "Warning in compute_anharmonic_del_v0_del_umn: squared SCP frequency is negative. ik = "
                          << ik << '\n';
            }
        }
        const MatrixXcd GT = G.transpose();
        for (auto i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_SCP[i1] += factor2 * del_v2_del_umn_renorm[i1 * nk + ik].cwiseProduct(GT).sum();
        }
    }

    // STRAIN_FC5: (1/8) dPhi4/du_mn G G. The contraction F_mn[k] = sum dV4/du_mn G is the
    // change of the SCP matrix, (1/2) dPhi4/du_mn G, so the term is half the harmonic-like
    // trace above: (1/2) sum_k tr[F_mn[k] G_k^T] / (4N).
    if (dv4_fc5) {
        std::vector<unsigned int> kall(nk);
        for (auto ik = 0; ik < nk; ik++) kall[ik] = static_cast<unsigned int>(ik);
        std::vector<std::complex<double>> f(fc5_mn.size() * nk * ns2);
        dv4_fc5->contract_channels(gall.data(), kall, f.data());
        for (std::size_t c = 0; c < fc5_mn.size(); c++) {
            std::complex<double> sum = 0.0;
            for (auto ik = 0; ik < nk; ik++) {
                Map<const MatrixXcdRowMajor> F(f.data() + (c * nk + ik) * ns2, ns, ns);
                Map<const MatrixXcdRowMajor> Gk(gall.data() + ik * ns2, ns, ns);
                sum += F.cwiseProduct(Gk.transpose()).sum();
            }
            del_v0_del_umn_SCP[fc5_mn[c]] += 0.5 * factor2 * sum;
        }
    }
}

Eigen::MatrixXcd ScphQhaCommon::strain_vertex(const DelVStrainData &del_v_strain,
                                              const std::array<std::array<double, 3>, 3> &u_tensor,
                                              const std::vector<double> &q0, const int i1, const int ik,
                                              const int nk) const
{
    using namespace Eigen;
    using MatrixXcdRowMajor = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;
    const int ns = dynamical->neval;
    const auto ns2 = static_cast<std::size_t>(ns) * ns;
    const double factor = 4.0 * static_cast<double>(nk);

    MatrixXcdRowMajor mat(ns, ns);
    // the (is1, is2) rows of del_v2 / del2_v2 are contiguous in is1*ns+is2 order
    Map<VectorXcd> flat(mat.data(), ns2);
    flat = del_v_strain.del_v2[i1].row(ik).transpose();
    // renormalization by strain
    for (auto i2 = 0; i2 < 9; i2++) {
        flat += u_tensor[i2 / 3][i2 % 3] * del_v_strain.del2_v2[i1 * 9 + i2].row(ik).transpose();
    }
    // renormalization by displacement: sum_is3 del_v3[i1][ik](is3, is2*ns+is1) q0[is3],
    // accumulated in the (is2, is1) order of del_v3 and added transposed
    VectorXcd q0c(ns);
    for (auto is = 0; is < ns; is++) q0c(is) = std::complex<double>(q0[is], 0.0);
    const VectorXcd acc = factor * (del_v_strain.del_v3[i1][ik].transpose() * q0c);
    Map<const MatrixXcdRowMajor> acc_mat(acc.data(), ns, ns); // acc_mat(is2, is1)
    mat += acc_mat.transpose();
    // STRAIN_FC5: (1/2) dPhi4/du_mn q0 q0, as the quartic term of Relaxation::renormalize_v2_from_q0
    if (!fc5_q4mn.empty() && fc5_channel_of_mn[i1] >= 0) {
        const auto c = static_cast<std::size_t>(fc5_channel_of_mn[i1]);
        mat += 0.5 * factor * Map<const MatrixXcdRowMajor>(fc5_q4mn.data() + (c * nk + ik) * ns2, ns, ns);
    }
    return mat;
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
