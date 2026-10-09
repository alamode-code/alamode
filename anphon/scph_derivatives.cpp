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
#include <cstdint>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "interpolation.h"
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

const std::vector<Eigen::MatrixXcd> &
ScphQhaCommon::coarse_occupation_matrices(std::complex<double> ***cmat_convert, double **omega2_anharm_T,
                                          const double T_in, const KpointMeshUniform *kmesh_dense_in) const
{
    // The renormalized harmonic matrix of the SCP (and QHA) solve is Delta(k) =
    // interp(Delta_c)(k): Delta is formed on the coarse points c (renormalize_v2_from_q0,
    // renormalize_v2_from_umn) and Fourier-interpolated to the dense k. Its derivative over
    // q0 or u is the interpolation of the coarse-point vertices M_c, not the vertex taken
    // directly at the dense k; the two differ off the coarse mesh when the q0- and
    // strain-renormalization has a longer range than the coarse supercell. With the
    // Cartesian forms Mt = E M E^+ and Gt = E G E^+ (E: harmonic eigenvectors),
    //   sum_k tr[interp(M)_k G_k] = sum_k sum_ij interp(Mt)_k(i,j) Gt_k(j,i)
    //                             = sum_c sum_ij Mt_c(i,j) Y_c(i,j)
    // with the adjoint Y_c(i,j) = (1/Nc) sum_R e^{-2 pi i c.R} sum_k phi_ij(k,R) Gt_k(j,i),
    // phi_ij(k,R) the minimum-image phase of r2q, and the coarse DFT of
    // fourier_dymat_k_to_r. Back in the mode basis: Geff_c = E_c^+ Y_c^T E_c.
    using namespace Eigen;
    const auto nk = kmesh_dense_in->nk;
    const auto nkc = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    if (nkc == nk) {
        geff_cache.clear();
        geff_cache_key = 0;
        return geff_cache;
    }

    // FNV-1a over the occupation inputs: the force and the stress (and the Hessian's
    // finite differences at fixed occupations) of one solved state share one adjoint
    std::uint64_t key = 1469598103934665603ULL;
    const auto mix = [&key](const void *p, const std::size_t n) {
        const auto *c = static_cast<const unsigned char *>(p);
        for (std::size_t i = 0; i < n; ++i) key = (key ^ c[i]) * 1099511628211ULL;
    };
    mix(&T_in, sizeof(T_in));
    mix(&kmesh_dense_in, sizeof(kmesh_dense_in));
    for (unsigned int ik = 0; ik < nk; ++ik) {
        mix(omega2_anharm_T[ik], ns * sizeof(double));
        for (unsigned int is = 0; is < ns; ++is) mix(cmat_convert[ik][is], ns * sizeof(std::complex<double>));
    }
    if (key == geff_cache_key && geff_cache.size() == nkc) return geff_cache;
    const auto nk1 = kmesh_coarse->nk_i[0];
    const auto nk2 = kmesh_coarse->nk_i[1];
    const auto nk3 = kmesh_coarse->nk_i[2];

    const auto evec_at = [&](const unsigned int knum) {
        MatrixXcd e(ns, ns);
        for (unsigned int is = 0; is < ns; ++is)
            for (unsigned int js = 0; js < ns; ++js) e(is, js) = evec_harmonic[knum][js][is];
        return e;
    };

    // X[i][j][R] = sum_k phi_ij(k,R) Gt_k(j,i)
    NDArray<std::complex<double>, 3> x(ns, ns, nkc), y(ns, ns, nkc);
    for (unsigned int i = 0; i < ns; ++i)
        for (unsigned int j = 0; j < ns; ++j)
            for (unsigned int r = 0; r < nkc; ++r) x[i][j][r] = 0.0;
    MatrixXcd G(ns, ns);
    for (unsigned int ik = 0; ik < nk; ++ik) {
        scp_occupation_matrix(static_cast<int>(ik), cmat_convert[ik], omega2_anharm_T[ik], T_in, G);
        const auto e = evec_at(ik);
        const MatrixXcd gt = e * G * e.adjoint();
        const auto *xk = kmesh_dense_in->xk[ik];
        // the phase depends on the atom pair only
        const auto nat = static_cast<int>(ns / 3);
#pragma omp parallel for
        for (int ab = 0; ab < nat * nat; ++ab) {
            const auto a = static_cast<unsigned int>(ab / nat);
            const auto b = static_cast<unsigned int>(ab % nat);
            for (unsigned int r = 0; r < nkc; ++r) {
                const auto &shifts = mindist_list[a][b][r].shift;
                std::complex<double> phase = 0.0;
                for (const auto &it: shifts) {
                    phase += std::exp(im * (2.0 * pi *
                                            (static_cast<double>(it.sx) * xk[0] + static_cast<double>(it.sy) * xk[1] +
                                             static_cast<double>(it.sz) * xk[2])));
                }
                phase /= static_cast<double>(shifts.size());
                for (unsigned int i = 3 * a; i < 3 * a + 3; ++i)
                    for (unsigned int j = 3 * b; j < 3 * b + 3; ++j) x[i][j][r] += phase * gt(j, i);
            }
        }
    }
    // the coarse DFT matrix of fourier_dymat_k_to_r is symmetric in (k, R)
    fourier_dymat_k_to_r(nk1, nk2, nk3, ns, x, y);

    geff_cache.assign(nkc, MatrixXcd(ns, ns));
    MatrixXcd yt(ns, ns);
    for (unsigned int ic = 0; ic < nkc; ++ic) {
        for (unsigned int i = 0; i < ns; ++i)
            for (unsigned int j = 0; j < ns; ++j) yt(j, i) = y[i][j][ic];
        const auto e = evec_at(kmap_coarse_to_dense[ic]);
        geff_cache[ic] = e.adjoint() * yt * e;
    }
    geff_cache_key = key;
    return geff_cache;
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

    // With interpolation, the vertices of the coarse points against the pulled-back
    // occupations (coarse_occupation_matrices); the 1/(4N) of v3 stays that of the dense mesh.
    const auto &geff = coarse_occupation_matrices(cmat_convert, omega2_anharm_T, T_in, kmesh_dense_in);
    for (std::size_t ic = 0; ic < geff.size(); ++ic) {
        const auto knum = kmap_coarse_to_dense[ic];
#pragma omp parallel for
        for (int is = 0; is < ns; is++) {
            Map<const MatrixXcdRowMajor> V3(v3_renorm[knum][is], ns, ns);
            v1_SCP[is] += V3.cwiseProduct(geff[ic]).sum();
        }
    }

    // calculate SCP renormalization
    for (auto ik = 0; ik < nk_scph; ik++) {
        const auto count_zero = scp_occupation_matrix(ik, cmat_convert[ik], omega2_anharm_T[ik], T_in, G);
        if (count_zero != 0) {
            std::cout << "Warning in compute_anharmonic_v1_array : ";
            std::cout << count_zero << " non-acoustic zero frequencies are detected at ik = " << ik << ".\n\n";
        }
        if (!geff.empty()) continue;

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
    const auto &geff = coarse_occupation_matrices(cmat_convert, omega2_anharm_T, T_in, kmesh_dense_in);
    int nnegative = 0;
    for (auto ik = 0; ik < nk; ik++) {
        scp_occupation_matrix(ik, cmat_convert[ik], omega2_anharm_T[ik], T_in, G, &is_acoustic_now);
        if (dv4_fc5) Map<MatrixXcdRowMajor>(gall.data() + ik * ns2, ns, ns) = G;
        for (auto js = 0; js < ns; js++) {
            if (ik == ik_gamma_dense && is_acoustic_now[js]) {
                continue;
            }
            if (omega2_anharm_T[ik][js] < 0.0 && std::sqrt(std::fabs(omega2_anharm_T[ik][js])) >= eps8) {
                ++nnegative;
            }
        }
        if (!geff.empty()) continue;
        const MatrixXcd GT = G.transpose();
        for (auto i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_SCP[i1] += factor2 * del_v2_del_umn_renorm[i1 * nk + ik].cwiseProduct(GT).sum();
        }
    }
    if (nnegative > 0 && run.my_rank == 0) {
        std::cout << " Warning in compute_anharmonic_del_v0_del_umn: " << nnegative
                  << " mode(s) with a negative squared frequency; their |omega| enters the stress.\n";
    }
    // With interpolation, the vertices of the coarse points against the pulled-back
    // occupations (coarse_occupation_matrices).
    for (std::size_t ic = 0; ic < geff.size(); ++ic) {
        const MatrixXcd GT = geff[ic].transpose();
        const auto knum = kmap_coarse_to_dense[ic];
        for (auto i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_SCP[i1] += factor2 * del_v2_del_umn_renorm[i1 * nk + knum].cwiseProduct(GT).sum();
        }
    }

    // STRAIN_FC5: (1/8) dPhi4/du_mn G G. The contraction F_mn[k] = sum dV4/du_mn G is the
    // change of the SCP matrix, (1/2) dPhi4/du_mn G, so the term is half the harmonic-like
    // trace above: (1/2) sum_k tr[F_mn[k] G_k^T] / (4N).
    if (dv4_fc5) {
        // The FC5 part of the SCP matrix is formed at the coarse points (RealSpaceV4::fmat)
        // and interpolated like the rest of it, so with interpolation its strain derivative
        // is contracted at the coarse points against the pulled-back occupations.
        const bool interp = !geff.empty();
        std::vector<unsigned int> kout;
        if (interp) {
            for (const auto knum: kmap_coarse_to_dense) kout.push_back(static_cast<unsigned int>(knum));
        } else {
            for (auto ik = 0; ik < nk; ik++) kout.push_back(static_cast<unsigned int>(ik));
        }
        const auto nout = kout.size();
        std::vector<std::complex<double>> f(fc5_mn.size() * nout * ns2);
        dv4_fc5->contract_channels(gall.data(), kout, f.data());
        for (std::size_t c = 0; c < fc5_mn.size(); c++) {
            std::complex<double> sum = 0.0;
            for (std::size_t i = 0; i < nout; i++) {
                Map<const MatrixXcdRowMajor> F(f.data() + (c * nout + i) * ns2, ns, ns);
                if (interp) {
                    sum += F.cwiseProduct(geff[i].transpose()).sum();
                } else {
                    Map<const MatrixXcdRowMajor> Gk(gall.data() + kout[i] * ns2, ns, ns);
                    sum += F.cwiseProduct(Gk.transpose()).sum();
                }
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
