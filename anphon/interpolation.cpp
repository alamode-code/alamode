/*
 interpolation.cpp

 Copyright (c) 2021 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "interpolation.h"
#include <cmath>
#include "constants.h"
#include "error.h"
#include "mathfunctions.h"
#include "symmetry_core.h"
#include "system.h"

using namespace PHON_NS;

// TriLinearInterpolator implementations

TriLinearInterpolator::TriLinearInterpolator(const unsigned int ngrid_coarse_in[3],
                                             const unsigned int ngrid_dense_in[3])
{
    for (auto i = 0; i < 3; ++i) {
        grid_c[i] = ngrid_coarse_in[i];
        grid_f[i] = ngrid_dense_in[i];
    }
    ngrid_c = grid_c[0] * grid_c[1] * grid_c[2];
    ngrid_f = grid_f[0] * grid_f[1] * grid_f[2];

    xf.resize(ngrid_f, 3);
    xc.resize(ngrid_c, 3);
}

void TriLinearInterpolator::setup()
{
    set_grid(grid_c, xc);
    set_grid(grid_f, xf);
}

TriLinearInterpolator::~TriLinearInterpolator()
{}

void TriLinearInterpolator::set_grid(const unsigned int ngrid_in[3], double **x_out)
{
    // gamma point will always be index 0
    size_t ik;
    double invn[3];
    for (auto i = 0; i < 3; ++i) {
        invn[i] = 1.0 / static_cast<double>(ngrid_in[i]);
    }
    for (size_t ix = 0; ix < ngrid_in[0]; ++ix) {
        for (size_t iy = 0; iy < ngrid_in[1]; ++iy) {
            for (size_t iz = 0; iz < ngrid_in[2]; ++iz) {
                ik = iz + iy * ngrid_in[2] + ix * ngrid_in[2] * ngrid_in[1];
                x_out[ik][0] = static_cast<double>(ix) * invn[0];
                x_out[ik][1] = static_cast<double>(iy) * invn[1];
                x_out[ik][2] = static_cast<double>(iz) * invn[2];
            }
        }
    }
}

void TriLinearInterpolator::get_corners_regular(double *xk_i, int *corner_index, double **corner_coord)
{
    // get the index of the corner, as well as coordinates[8][3]
    int iloc[2], jloc[2], kloc[2];
    double dn_c[3];
    double tmp[3];
    int n23 = static_cast<int>(grid_c[1] * grid_c[2]);
    int igrid[3];

    for (auto i = 0; i < 3; ++i) {
        dn_c[i] = static_cast<double>(grid_c[i]);
        igrid[i] = static_cast<int>(grid_c[i]);
    }

    for (auto j = 0; j < 3; ++j) tmp[j] = xk_i[j] * dn_c[j];
    iloc[0] = nint(std::floor(tmp[0]));
    iloc[1] = nint(std::ceil(tmp[0]));
    jloc[0] = nint(std::floor(tmp[1]));
    jloc[1] = nint(std::ceil(tmp[1]));
    kloc[0] = nint(std::floor(tmp[2]));
    kloc[1] = nint(std::ceil(tmp[2]));

    if (iloc[1] == iloc[0]) ++iloc[1];
    if (jloc[1] == jloc[0]) ++jloc[1];
    if (kloc[1] == kloc[0]) ++kloc[1];

    corner_coord[0][0] = static_cast<double>(iloc[0]) / dn_c[0];
    corner_coord[0][1] = static_cast<double>(jloc[0]) / dn_c[1];
    corner_coord[0][2] = static_cast<double>(kloc[0]) / dn_c[2];

    corner_coord[1][0] = static_cast<double>(iloc[1]) / dn_c[0];
    corner_coord[1][1] = static_cast<double>(jloc[0]) / dn_c[1];
    corner_coord[1][2] = static_cast<double>(kloc[0]) / dn_c[2];

    corner_coord[2][0] = static_cast<double>(iloc[0]) / dn_c[0];
    corner_coord[2][1] = static_cast<double>(jloc[1]) / dn_c[1];
    corner_coord[2][2] = static_cast<double>(kloc[0]) / dn_c[2];

    corner_coord[3][0] = static_cast<double>(iloc[1]) / dn_c[0];
    corner_coord[3][1] = static_cast<double>(jloc[1]) / dn_c[1];
    corner_coord[3][2] = static_cast<double>(kloc[0]) / dn_c[2];

    corner_coord[4][0] = static_cast<double>(iloc[0]) / dn_c[0];
    corner_coord[4][1] = static_cast<double>(jloc[0]) / dn_c[1];
    corner_coord[4][2] = static_cast<double>(kloc[1]) / dn_c[2];

    corner_coord[5][0] = static_cast<double>(iloc[1]) / dn_c[0];
    corner_coord[5][1] = static_cast<double>(jloc[0]) / dn_c[1];
    corner_coord[5][2] = static_cast<double>(kloc[1]) / dn_c[2];

    corner_coord[6][0] = static_cast<double>(iloc[0]) / dn_c[0];
    corner_coord[6][1] = static_cast<double>(jloc[1]) / dn_c[1];
    corner_coord[6][2] = static_cast<double>(kloc[1]) / dn_c[2];

    corner_coord[7][0] = static_cast<double>(iloc[1]) / dn_c[0];
    corner_coord[7][1] = static_cast<double>(jloc[1]) / dn_c[1];
    corner_coord[7][2] = static_cast<double>(kloc[1]) / dn_c[2];

    iloc[0] = iloc[0] % igrid[0];
    iloc[1] = iloc[1] % igrid[0];
    jloc[0] = jloc[0] % igrid[1];
    jloc[1] = jloc[1] % igrid[1];
    kloc[0] = kloc[0] % igrid[2];
    kloc[1] = kloc[1] % igrid[2];

    corner_index[0] = kloc[0] + jloc[0] * igrid[2] + iloc[0] * n23; // index of c000
    corner_index[1] = kloc[0] + jloc[0] * igrid[2] + iloc[1] * n23; // index of c100
    corner_index[2] = kloc[0] + jloc[1] * igrid[2] + iloc[0] * n23; // index of c010
    corner_index[3] = kloc[0] + jloc[1] * igrid[2] + iloc[1] * n23; // index of c110
    corner_index[4] = kloc[1] + jloc[0] * igrid[2] + iloc[0] * n23; // index of c001
    corner_index[5] = kloc[1] + jloc[0] * igrid[2] + iloc[1] * n23; // index of c101
    corner_index[6] = kloc[1] + jloc[1] * igrid[2] + iloc[0] * n23; // index of c011
    corner_index[7] = kloc[1] + jloc[1] * igrid[2] + iloc[1] * n23; // index of c111
}

// FourierInterpolator implementations

FourierInterpolator::FourierInterpolator(const KpointMeshUniform &kmesh_coarse_in,
                                         const KpointMeshUniform &kmesh_dense_in,
                                         const bool subtract_harmonic_term_in) :
    kmesh_coarse(kmesh_coarse_in), kmesh_dense(kmesh_dense_in)
{
    subtract_harmonic_term = subtract_harmonic_term_in;

    get_map_coarse_to_dense(kmesh_coarse, kmesh_dense, map_corase_to_dense);
}

void FourierInterpolator::get_map_coarse_to_dense(const KpointMeshUniform &kmesh_coarse,
                                                  const KpointMeshUniform &kmesh_dense, std::vector<int> &kmap)
{
    if (Kpoint::get_kmap_coarse_to_dense(&kmesh_coarse, &kmesh_dense, kmap) != 0) {
        exit("FourierInterpolator::get_map_coarse_to_dense",
             "Cannot find the corresponding kpoint in the dense mesh");
    }
}

void PHON_NS::r2q(const double *xk_in, const unsigned int nx, const unsigned int ny, const unsigned int nz,
                  const unsigned int ns, MinimumDistList ***mindist_list_in, std::complex<double> ***dymat_r_in,
                  std::complex<double> **dymat_k_out)
{
    const auto ncell = nx * ny * nz;
    const auto ns2 = ns * ns;

    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

#pragma omp parallel for
    for (int ij = 0; ij < ns2; ++ij) {
        const auto i = ij / ns;
        const auto j = ij % ns;
        const auto iat = i / 3;
        const auto jat = j / 3;

        dymat_k_out[i][j] = complex_zero;

        for (auto icell = 0; icell < ncell; ++icell) {
            auto exp_phase = complex_zero;
            // This operation is necessary for the Hermiticity of the dynamical matrix.
            for (const auto &it: mindist_list_in[iat][jat][icell].shift) {
                auto phase = 2.0 * pi *
                             (static_cast<double>(it.sx) * xk_in[0] + static_cast<double>(it.sy) * xk_in[1] +
                              static_cast<double>(it.sz) * xk_in[2]);

                exp_phase += std::exp(im * phase);
            }
            exp_phase /= static_cast<double>(mindist_list_in[iat][jat][icell].shift.size());
            dymat_k_out[i][j] += dymat_r_in[i][j][icell] * exp_phase;
        }
    }
}

void PHON_NS::fourier_dymat_k_to_r(const unsigned int nk1, const unsigned int nk2, const unsigned int nk3,
                                   const unsigned int ns, const std::complex<double> *const *const *dymat_k,
                                   std::complex<double> ***dymat_r)
{
    // Forward DFT (k -> r), including the 1/N normalization, of all (is, js)
    // components of a coarse-mesh dynamical-matrix array at once:
    //   dymat_r[is][js][r] = (1/N) sum_k dymat_k[is][js][k] e^{-2 pi i k.r}.
    //
    // This used to be done with one FFTW plan per (is, js) pair, re-created at
    // every call: fftw_execute() always transforms the arrays its plan was
    // created with, and fftw_execute_dft() on different arrays carries strict
    // alignment requirements, so the plan could not be hoisted safely. The
    // coarse mesh is tiny, so plan creation dominated the transform itself.
    // The explicit DFT matrix below reproduces the FFTW_FORWARD convention
    // (row-major multi-index, negative exponent) exactly, is alignment-free,
    // and applies to all ns^2 components as a single matrix product; the arrays
    // from allocate() are contiguous, with (is, js) blocks of length
    // nk1*nk2*nk3 each.

    using namespace Eigen;

    const auto nk_coarse = nk1 * nk2 * nk3;

    MatrixXcd dft_matrix(nk_coarse, nk_coarse);
    for (unsigned int ir = 0; ir < nk_coarse; ++ir) {
        const auto r1 = ir / (nk2 * nk3);
        const auto r2 = (ir / nk3) % nk2;
        const auto r3 = ir % nk3;
        for (unsigned int ik = 0; ik < nk_coarse; ++ik) {
            const auto k1 = ik / (nk2 * nk3);
            const auto k2 = (ik / nk3) % nk2;
            const auto k3 = ik % nk3;
            const auto phase = -2.0 * pi *
                               (static_cast<double>(r1 * k1) / static_cast<double>(nk1) +
                                static_cast<double>(r2 * k2) / static_cast<double>(nk2) +
                                static_cast<double>(r3 * k3) / static_cast<double>(nk3));
            dft_matrix(ir, ik) = std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }
    // 1/N normalization previously applied after the FFT
    dft_matrix /= static_cast<double>(nk_coarse);

    Map<const MatrixXcd> dymat_k_flat(dymat_k[0][0], nk_coarse, ns * ns);
    Map<MatrixXcd> dymat_r_flat(dymat_r[0][0], nk_coarse, ns * ns);
    dymat_r_flat.noalias() = dft_matrix * dymat_k_flat;
}

void PHON_NS::symmetrize_dynamical_matrix(const unsigned int ik, const KpointMeshUniform *kmesh_coarse,
                                          const unsigned int ns, std::complex<double> ****mat_transform_sym,
                                          Eigen::MatrixXcd &dymat)
{
    // Symmetrize the dynamical matrix of given index ik.
    using namespace Eigen;
    unsigned int i, isym;
    unsigned int is, js;
    MatrixXcd dymat_sym = MatrixXcd::Zero(ns, ns);
    MatrixXcd dymat_tmp(ns, ns), gamma(ns, ns);

    const auto nsym_small = kmesh_coarse->small_group_of_k[ik].size();
    const auto nsym_minus = kmesh_coarse->symop_minus_at_k[ik].size();

    for (i = 0; i < nsym_minus; ++i) {
        isym = kmesh_coarse->symop_minus_at_k[ik][i];

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                gamma(is, js) = mat_transform_sym[ik][isym][is][js];
            }
        }

        // Eq. (3.35) of Maradudin & Vosko
        dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
        dymat_sym += dymat_tmp.conjugate();
    }

    for (i = 0; i < nsym_small; ++i) {
        isym = kmesh_coarse->small_group_of_k[ik][i];

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                gamma(is, js) = mat_transform_sym[ik][isym][is][js];
            }
        }

        // Eq. (3.14) of Maradudin & Vosko
        dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
        dymat_sym += dymat_tmp;
    }

    dymat = dymat_sym / static_cast<double>(nsym_small + nsym_minus);
}


void PHON_NS::replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse, const unsigned int ns,
                                              std::complex<double> ****mat_transform_sym,
                                              std::complex<double> ***dymat_inout)
{
    using namespace Eigen;
    unsigned int i;
    unsigned int is, js;
    MatrixXcd dymat_tmp(ns, ns), gamma(ns, ns), dymat(ns, ns);

    NDArray<std::complex<double>, 3> dymat_all;

    dymat_all.resize(ns, ns, kmesh_coarse->nk);

    for (i = 0; i < kmesh_coarse->nk; ++i) {

        const auto ik_irred = kmesh_coarse->kpoint_map_symmetry[i].knum_irred_orig;
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym >= 0) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    gamma(is, js) = mat_transform_sym[ik_irred][isym][is][js];
                    dymat(is, js) = dymat_inout[is][js][ik_orig];
                }
            }
            dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_all[is][js][i] = dymat_tmp(is, js);
                }
            }
        }
    }

    // When the point group operation S_ which transforms k into -k, i.e., (S_)k = -k,
    // does not exist for k, we simply set D(k)=D(-k)^{*}.
    // (This should hold even when the time-reversal symmetry breaks.)
    for (i = 0; i < kmesh_coarse->nk; ++i) {
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;
        if (isym == -1) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_all[is][js][i] = std::conj(dymat_all[is][js][ik_orig]);
                }
            }
        }
    }

    for (is = 0; is < ns; ++is) {
        for (js = 0; js < ns; ++js) {
            for (i = 0; i < kmesh_coarse->nk; ++i) {
                dymat_inout[is][js][i] = dymat_all[is][js][i];
            }
        }
    }
    dymat_all.clear();
}

void PHON_NS::replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse, const unsigned int ns,
                                              std::complex<double> ****mat_transform_sym,
                                              std::vector<Eigen::MatrixXcd> &dymat_inout)
{
    // Eigen version: works directly with vector of MatrixXcd
    // Avoids temporary C-style array allocation

    using namespace Eigen;
    const auto nk = kmesh_coarse->nk;

    MatrixXcd gamma(ns, ns);

    // Temporary storage for replicated dynamical matrices
    std::vector<MatrixXcd> dymat_all;
    dymat_all.reserve(nk);
    for (unsigned int i = 0; i < nk; ++i) {
        dymat_all.emplace_back(ns, ns);
    }

    // Apply symmetry operations to replicate for all k-points
    for (unsigned int i = 0; i < nk; ++i) {

        const auto ik_irred = kmesh_coarse->kpoint_map_symmetry[i].knum_irred_orig;
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym >= 0) {
            // Copy transformation matrix and dynamical matrix
            for (unsigned int is = 0; is < ns; ++is) {
                for (unsigned int js = 0; js < ns; ++js) {
                    gamma(is, js) = mat_transform_sym[ik_irred][isym][is][js];
                }
            }

            // Apply symmetry transformation: D'(k) = Γ D(k) Γ^†
            dymat_all[i] = gamma * dymat_inout[ik_orig] * gamma.adjoint();
            ;
        }
    }

    // Handle time-reversal symmetry: D(k) = D(-k)^*
    // When point group operation S_ which transforms k into -k doesn't exist,
    // we set D(k) = D(-k)^* (holds even when time-reversal symmetry breaks)
    for (unsigned int i = 0; i < nk; ++i) {
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym == -1) {
            dymat_all[i] = dymat_all[ik_orig].conjugate();
        }
    }

    // Copy results back to input/output vector
    for (unsigned int i = 0; i < nk; ++i) {
        dymat_inout[i] = dymat_all[i];
    }
}


void PHON_NS::get_symmetry_gamma_dynamical(KpointMeshUniform *kmesh_in, const unsigned int natmin_in,
                                           const unsigned int ns, const Eigen::MatrixXd &x_fractional_in,
                                           const std::vector<SymmetryOperationWithMapping> &symmlist,
                                           NDArray<std::complex<double>, 4> &mat_transform_sym)
{
    // Construct the transformation matrix for the dynamical matrix.

    unsigned int ik;
    unsigned int is, js;
    unsigned int icrd, jcrd;
    double x1[3], x2[3], k[3], xtmp[3];
    double S_cart[3][3], S_frac[3][3], S_frac_inv[3][3];
    NDArray<std::complex<double>, 2> gamma_tmp;

    const auto natmin = natmin_in;
    const auto nk_irred_interpolate = kmesh_in->nk_irred;

    const auto nsym = symmlist.size();

    gamma_tmp.resize(ns, ns);

    mat_transform_sym.clear();
    mat_transform_sym.resize(nk_irred_interpolate, nsym, ns, ns);

    for (ik = 0; ik < nk_irred_interpolate; ++ik) {

        const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
        for (icrd = 0; icrd < 3; ++icrd) {
            k[icrd] = kmesh_in->xk[knum][icrd];
        }

        unsigned int isym = 0;

        for (const auto &it: symmlist) {

            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    S_cart[icrd][jcrd] = it.rot[3 * icrd + jcrd];
                    S_frac[icrd][jcrd] = it.rot_real[3 * icrd + jcrd];
                }
            }

            invmat3(S_frac_inv, S_frac);

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    gamma_tmp[is][js] = std::complex<double>(0.0, 0.0);
                }
            }

            for (unsigned int jat = 0; jat < natmin; ++jat) {
                const auto iat = it.mapping[jat];

                // Fractional coordinates of x1 and x2
                for (icrd = 0; icrd < 3; ++icrd) {
                    x1[icrd] = x_fractional_in(iat, icrd);
                    x2[icrd] = x_fractional_in(jat, icrd);
                }

                rotvec(xtmp, x1, S_frac_inv);
                for (icrd = 0; icrd < 3; ++icrd) {
                    xtmp[icrd] = xtmp[icrd] - x2[icrd];
                }

                auto phase = 2.0 * pi * (k[0] * xtmp[0] + k[1] * xtmp[1] + k[2] * xtmp[2]);

                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jcrd = 0; jcrd < 3; ++jcrd) {
                        gamma_tmp[3 * iat + icrd][3 * jat + jcrd] = S_cart[icrd][jcrd] * std::exp(im * phase);
                    }
                }
            }

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    mat_transform_sym[ik][isym][is][js] = gamma_tmp[is][js];
                }
            }

            ++isym;
        }
    }

    gamma_tmp.clear();
}
