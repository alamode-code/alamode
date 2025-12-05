/*
 scph_v3v4_elements.cpp
 Copyright (c) 2015 Terumasa Tadano
 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/
/*
 Functions for computing V3 and V4 phonon interaction elements.
 These are used by both SCPH and QHA calculations.
*/
#include "scph.h"
#include "anharmonic_core.h"
#include "dynamical.h"
#include "error.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "timer.h"
#include "relaxation.h"
#include <complex>
#include <iostream>
#include <cmath>
#include <vector>
using namespace PHON_NS;
void Scph::compute_V3_elements_mpi_over_kpoint(std::complex<double> ***v3_out, double **omega2_harmonic_in,
                                               const std::complex<double> *const *const *evec_in,
                                               const bool self_offdiag, const KpointMeshUniform *kmesh_coarse_in,
                                               const KpointMeshUniform *kmesh_dense_in,
                                               const PhaseFactorStorage *phase_storage_in,
                                               std::complex<double> *phi3_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    auto ns = dynamical->neval;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    unsigned int **ind;
    unsigned int i, j;

    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v3 = anharmonic_core->get_ngroup_fcs(3);
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v3_array_at_kpair;
    std::complex<double> ***v3_mpi;

    std::complex<double> **v3_tmp0, **v3_tmp1, **v3_tmp2, **v3_tmp3;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v3_array ... ";
        } else {
            std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v3_array ... ";
        }
    }

    allocate(v3_array_at_kpair, ngroup_v3);
    allocate(ind, ngroup_v3, 3);
    allocate(v3_mpi, nk_scph, ns, ns2);

    allocate(v3_tmp0, ns, ns2);
    allocate(v3_tmp1, ns, ns2);
    allocate(v3_tmp2, ns, ns2);
    allocate(v3_tmp3, ns, ns2);

    for (unsigned int ik = mympi->my_rank; ik < nk_scph; ik += mympi->nprocs) {

        anharmonic_core->calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[ik]],
                                              anharmonic_core->get_ngroup_fcs(3),
                                              anharmonic_core->get_fcs_group(3),
                                              anharmonic_core->get_relvec(3),
                                              phase_storage_in,
                                              phi3_reciprocal_inout);

#pragma omp parallel for private(j)
        for (ii = 0; ii < ngroup_v3; ++ii) {
            v3_array_at_kpair[ii] = phi3_reciprocal_inout[ii] * anharmonic_core->get_invmass_factor(3)[ii];
            for (j = 0; j < 3; ++j)
                ind[ii][j] = anharmonic_core->get_evec_index(3)[ii][j];
        }

#pragma omp parallel for private(is)
        for (ii = 0; ii < ns; ++ii) {
            for (is = 0; is < ns2; ++is) {
                v3_mpi[ik][ii][is] = complex_zero;
                v3_out[ik][ii][is] = complex_zero;
            }
        }

        if (self_offdiag) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (i.e., when considering polarization mixing).

            // initialize temporary matrices
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                    v3_tmp1[is][js] = complex_zero;
                    v3_tmp2[is][js] = complex_zero;
                    v3_tmp3[is][js] = complex_zero;
                }
            }

            // copy v3 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                for (is2 = 0; is2 < ns; ++is2) {
                    v3_tmp1[is][js2_1] += v3_tmp0[is2][js2_1] * evec_in[0][is][is2];
                }
            }

            // transform the second index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // second index
                ks = js2_1 % ns; // third index

                for (js2 = 0; js2 < ns; ++js2) {
                    js2_2 = js2 * ns + ks;
                    v3_tmp2[is][js2_1] += v3_tmp1[is][js2_2] * evec_in[ik][js][js2];
                }
            }

            // transform the third index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // third index
                ks = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = js * ns + ks2;
                    v3_tmp3[is][js2_1] += v3_tmp2[is][js2_2] * std::conj(evec_in[ik][ks][ks2]);
                }
            }

            // copy to the final matrix
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_mpi[ik][is][js2_1] = factor * v3_tmp3[is][js2_1];
            }

        } else {

            // Only diagonal elements will be computed when neglecting the polarization mixing.

            if (ik == 0) {
#pragma omp parallel for private(is, js, ks, ret, i)
                for (ii = 0; ii < ns3; ++ii) {
                    is = ii / ns2;
                    js = (ii - ns2 * is) / ns;
                    ks = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3; ++i) {

                        ret += v3_array_at_kpair[i] * evec_in[0][is][ind[i][0]] * evec_in[ik][js][ind[i][1]] *
                               std::conj(evec_in[ik][ks][ind[i][2]]);
                    }

                    v3_mpi[ik][is][ns * js + ks] = factor * ret;
                }
            } else {

#pragma omp parallel for private(is, js, ret, i)
                for (ii = 0; ii < ns2; ++ii) {
                    is = ii / ns;
                    js = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3; ++i) {

                        ret += v3_array_at_kpair[i] * evec_in[0][is][ind[i][0]] * evec_in[ik][js][ind[i][1]] *
                               std::conj(evec_in[ik][js][ind[i][2]]);
                    }

                    v3_mpi[ik][is][(ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    deallocate(v3_array_at_kpair);
    deallocate(ind);
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(&v3_mpi[0][0][0],
                  &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
    MPI_Allreduce(&v3_mpi[0][0][0],
                  &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    deallocate(v3_mpi);
    deallocate(v3_tmp0);
    deallocate(v3_tmp1);
    deallocate(v3_tmp2);
    deallocate(v3_tmp3);


    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v3_out, 3, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

// This function should be merged with void Scph::compute_V3_elements_mpi_over_kpoint
// after merged with dev2.0 because the implementation is redundant.
void Scph::compute_V3_elements_for_given_IFCs(std::complex<double> ***v3_out, double **omega2_harmonic_in,
                                              const int ngroup_v3_in, std::vector<double> *fcs_group_v3_in,
                                              std::vector<RelativeVector> *relvec_v3_in, double *invmass_v3_in,
                                              int **evec_index_v3_in, const std::complex<double> *const *const *evec_in,
                                              const bool self_offdiag, const KpointMeshUniform *kmesh_coarse_in,
                                              const KpointMeshUniform *kmesh_dense_in,
                                              const PhaseFactorStorage *phase_storage_in)
{
    auto ns = dynamical->neval;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    unsigned int **ind;
    unsigned int i, j;
    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    static auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v3_array_at_kpair;
    std::complex<double> ***v3_mpi;
    std::complex<double> *phi3_reciprocal_tmp;

    std::complex<double> **v3_tmp0, **v3_tmp1, **v3_tmp2, **v3_tmp3;

    allocate(phi3_reciprocal_tmp, ngroup_v3_in);
    allocate(v3_array_at_kpair, ngroup_v3_in);
    allocate(ind, ngroup_v3_in, 3);
    allocate(v3_mpi, nk_scph, ns, ns2);

    allocate(v3_tmp0, ns, ns2);
    allocate(v3_tmp1, ns, ns2);
    allocate(v3_tmp2, ns, ns2);
    allocate(v3_tmp3, ns, ns2);

    for (unsigned int ik = mympi->my_rank; ik < nk_scph; ik += mympi->nprocs) {

        anharmonic_core->calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[ik]],
                                              ngroup_v3_in,
                                              fcs_group_v3_in,
                                              relvec_v3_in,
                                              phase_storage_in,
                                              phi3_reciprocal_tmp);

#ifdef _OPENMP
#pragma omp parallel for private(j)
#endif
        for (ii = 0; ii < ngroup_v3_in; ++ii) {
            v3_array_at_kpair[ii] = phi3_reciprocal_tmp[ii] * invmass_v3_in[ii];
            for (j = 0; j < 3; ++j)
                ind[ii][j] = evec_index_v3_in[ii][j];
        }

#pragma omp parallel for private(is)
        for (ii = 0; ii < ns; ++ii) {
            for (is = 0; is < ns2; ++is) {
                v3_mpi[ik][ii][is] = complex_zero;
                v3_out[ik][ii][is] = complex_zero;
            }
        }

        if (self_offdiag) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (i.e., when considering polarization mixing).

            // initialize temporary matrices
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                    v3_tmp1[is][js] = complex_zero;
                    v3_tmp2[is][js] = complex_zero;
                    v3_tmp3[is][js] = complex_zero;
                }
            }

            // copy v3 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3_in; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                for (is2 = 0; is2 < ns; ++is2) {
                    v3_tmp1[is][js2_1] += v3_tmp0[is2][js2_1] * evec_in[0][is][is2];
                }
            }

            // transform the second index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // second index
                ks = js2_1 % ns; // third index

                for (js2 = 0; js2 < ns; ++js2) {
                    js2_2 = js2 * ns + ks;
                    v3_tmp2[is][js2_1] += v3_tmp1[is][js2_2] * evec_in[ik][js][js2];
                }
            }

            // transform the third index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // third index
                ks = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = js * ns + ks2;
                    v3_tmp3[is][js2_1] += v3_tmp2[is][js2_2] * std::conj(evec_in[ik][ks][ks2]);
                }
            }

            // copy to the final matrix
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_mpi[ik][is][js2_1] = factor * v3_tmp3[is][js2_1];
            }

        } else {

            // Only diagonal elements will be computed when neglecting the polarization mixing.

            if (ik == 0) {
#pragma omp parallel for private(is, js, ks, ret, i)
                for (ii = 0; ii < ns3; ++ii) {
                    is = ii / ns2;
                    js = (ii - ns2 * is) / ns;
                    ks = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3_in; ++i) {

                        ret += v3_array_at_kpair[i] * evec_in[0][is][ind[i][0]] * evec_in[ik][js][ind[i][1]] *
                               std::conj(evec_in[ik][ks][ind[i][2]]);
                    }

                    v3_mpi[ik][is][ns * js + ks] = factor * ret;
                }
            } else {

#pragma omp parallel for private(is, js, ret, i)
                for (ii = 0; ii < ns2; ++ii) {
                    is = ii / ns;
                    js = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3_in; ++i) {

                        ret += v3_array_at_kpair[i] * evec_in[0][is][ind[i][0]] * evec_in[ik][js][ind[i][1]] *
                               std::conj(evec_in[ik][js][ind[i][2]]);
                    }

                    v3_mpi[ik][is][(ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    deallocate(v3_array_at_kpair);
    deallocate(ind);
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(&v3_mpi[0][0][0],
                  &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
    MPI_Allreduce(&v3_mpi[0][0][0],
                  &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    deallocate(v3_mpi);
    deallocate(v3_tmp0);
    deallocate(v3_tmp1);
    deallocate(v3_tmp2);
    deallocate(v3_tmp3);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v3_out, 3, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);
}


void Scph::compute_V4_elements_mpi_over_kpoint(std::complex<double> ***v4_out, double **omega2_harmonic_in,
                                               std::complex<double> ***evec_in, const bool self_offdiag,
                                               const bool relax, const KpointMeshUniform *kmesh_coarse_in,
                                               const KpointMeshUniform *kmesh_dense_in,
                                               const std::vector<int> &kmap_coarse_to_dense,
                                               const PhaseFactorStorage *phase_storage_in,
                                               std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    const size_t nk_reduced_interpolate = kmesh_coarse_in->nk_irred;
    const size_t ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns3 = ns * ns * ns;
    const size_t ns4 = ns * ns * ns * ns;
    size_t is, js, ks, ls;
    size_t is2_1, js2_1, is2_2, js2_2;
    size_t is2, js2, ks2, ls2;
    unsigned int **ind;
    unsigned int j;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v4_array_at_kpair;
    std::complex<double> ***v4_mpi;
    std::complex<double> ***evec_conj;

    std::complex<double> **v4_tmp0, **v4_tmp1, **v4_tmp2, **v4_tmp3, **v4_tmp4;

    const size_t nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... " << std::flush;
        } else {
            std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v4_array ... " << std::flush;
        }
    }

    allocate(v4_array_at_kpair, ngroup_v4);
    allocate(ind, ngroup_v4, 4);
    allocate(v4_mpi, nk2_prod, ns2, ns2);
    allocate(evec_conj, kmesh_dense_in->nk, ns, ns);

    allocate(v4_tmp0, ns2, ns2);
    allocate(v4_tmp1, ns2, ns2);
    allocate(v4_tmp2, ns2, ns2);
    allocate(v4_tmp3, ns2, ns2);
    allocate(v4_tmp4, ns2, ns2);

    const long int nks2 = kmesh_dense_in->nk * ns2;

#pragma omp parallel for private(is, js)
    for (long int iks = 0; iks < nks2; ++iks) {
        size_t ik = iks / ns2;
        is = (iks - ik * ns2) / ns;
        js = iks % ns;
        evec_conj[ik][is][js] = std::conj(evec_in[ik][is][js]);
    }

    for (size_t ik_prod = mympi->my_rank; ik_prod < nk2_prod; ik_prod += mympi->nprocs) {
        const auto ik = ik_prod / nk_scph;
        const auto jk = ik_prod % nk_scph;

        const unsigned int knum = kmap_coarse_to_dense[kmesh_coarse_in->kpoint_irred_all[ik][0].knum];

        anharmonic_core->calc_phi4_reciprocal(kmesh_dense_in->xk[knum],
                                              kmesh_dense_in->xk[jk],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[jk]],
                                              phase_storage_in,
                                              phi4_reciprocal_inout);

#pragma omp parallel for private(j)
        for (ii = 0; ii < ngroup_v4; ++ii) {
            v4_array_at_kpair[ii] = phi4_reciprocal_inout[ii] * anharmonic_core->get_invmass_factor(4)[ii];
            for (j = 0; j < 4; ++j)
                ind[ii][j] = anharmonic_core->get_evec_index(4)[ii][j];
        }

#pragma omp parallel for private(is, js)
        for (ii = 0; ii < ns4; ++ii) {
            is = ii / ns2;
            js = ii % ns2;
            v4_mpi[ik_prod][is][js] = complex_zero;
            v4_out[ik_prod][is][js] = complex_zero;
        }

        // initialize temporary matrices
#pragma omp parallel for private(js)
        for (is = 0; is < ns2; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp0[is][js] = complex_zero;
                v4_tmp1[is][js] = complex_zero;
                v4_tmp2[is][js] = complex_zero;
                v4_tmp3[is][js] = complex_zero;
                v4_tmp4[is][js] = complex_zero;
            }
        }

        if (self_offdiag || relaxation->relax_str) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (loop diagram).


            // copy v4 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v4; ++ii) {

                is = ind[ii][0] * ns + ind[ii][1];
                js = ind[ii][2] * ns + ind[ii][3];
                v4_tmp0[is][js] = v4_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                is = is2_1 / ns; // first index
                js = is2_1 % ns; // second index

                for (is2 = 0; is2 < ns; ++is2) {
                    is2_2 = is2 * ns + js;
                    v4_tmp1[is2_1][js2_1] += v4_tmp0[is2_2][js2_1] * evec_conj[knum][is][is2];
                }
            }
            // transform the second index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                is = is2_1 / ns; // first index
                js = is2_1 % ns; // second index

                for (js2 = 0; js2 < ns; ++js2) {
                    is2_2 = is * ns + js2;
                    v4_tmp2[is2_1][js2_1] += v4_tmp1[is2_2][js2_1] * evec_in[knum][js][js2];
                }
            }
            // transform the third index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                ks = js2_1 / ns; // third index
                ls = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = ks2 * ns + ls;
                    v4_tmp3[is2_1][js2_1] += v4_tmp2[is2_1][js2_2] * evec_in[jk][ks][ks2];
                }
            }

            // transform the fourth index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                ks = js2_1 / ns; // third index
                ls = js2_1 % ns; // fourth index

                for (ls2 = 0; ls2 < ns; ++ls2) {
                    js2_2 = ks * ns + ls2;
                    v4_tmp4[is2_1][js2_1] += v4_tmp3[is2_1][js2_2] * evec_conj[jk][ls][ls2];
                }
            }

            // copy to the final matrix
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;

                v4_mpi[ik_prod][is2_1][js2_1] = factor * v4_tmp4[is2_1][js2_1];
            }

        } else {

            // copy v4 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v4; ++ii) {

                is = ind[ii][0] * ns + ind[ii][1];
                js = ind[ii][2] * ns + ind[ii][3];
                v4_tmp0[is][js] = v4_array_at_kpair[ii];
            }

            // transform the first and the second index
#pragma omp parallel for private(is, js, ks, is2_1, is2_2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                is2_1 = ii % ns2;
                for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                    // is2_2 = js*ns+ks
                    js = is2_2 / ns;
                    ks = is2_2 % ns;

                    v4_tmp1[(ns + 1) * is][is2_1] +=
                        v4_tmp0[is2_2][is2_1] * evec_conj[knum][is][js] * evec_in[knum][is][ks];
                }
            }
#pragma omp parallel for private(is, js, ks, ls, is2_2)
            // transform the third and the fourth index
            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;
                for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                    ks = is2_2 / ns;
                    ls = is2_2 % ns;

                    v4_tmp2[(ns + 1) * is][(ns + 1) * js] +=
                        v4_tmp1[(ns + 1) * is][is2_2] * evec_in[jk][js][ks] * evec_conj[jk][js][ls];
                }
            }
            // copy to the final matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ns2; ++ii) {
                is = ii / ns;
                js = ii % ns;

                v4_mpi[ik_prod][(ns + 1) * is][(ns + 1) * js] = factor * v4_tmp2[(ns + 1) * is][(ns + 1) * js];
            }
        }
    }


    deallocate(evec_conj);
    deallocate(v4_array_at_kpair);
    deallocate(ind);

    deallocate(v4_tmp0);
    deallocate(v4_tmp1);
    deallocate(v4_tmp2);
    deallocate(v4_tmp3);
    deallocate(v4_tmp4);

    // Now, communicate the calculated data.
    // When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(&v4_mpi[ik_prod][0][0],
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#else
            MPI_Allreduce(&v4_mpi[ik_prod][0][0],
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_COMPLEX16,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(&v4_mpi[ik_prod][is][0],
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#else
                MPI_Allreduce(&v4_mpi[ik_prod][is][0],
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_COMPLEX16,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#endif
            }
        }
    }

    deallocate(v4_mpi);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v4_out, 4, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void Scph::compute_V4_elements_mpi_over_band(std::complex<double> ***v4_out, double **omega2_harmonic_in,
                                             std::complex<double> ***evec_in, const bool self_offdiag,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const std::vector<int> &kmap_coarse_to_dense,
                                             const PhaseFactorStorage *phase_storage_in,
                                             std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    size_t ik_prod;
    const size_t nk_reduced_interpolate = kmesh_coarse_in->nk_irred;
    const size_t ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns4 = ns * ns * ns * ns;
    int is, js, ks, ls;
    size_t is2_1, js2_1, is2_2;
    size_t is2;
    int is4_1;
    unsigned int knum;
    unsigned int **ind;
    unsigned int i, j;
    long int *nset_mpi;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v4_array_at_kpair;
    std::complex<double> ***v4_mpi;

    std::complex<double> **v4_tmp0, **v4_tmp1, **v4_tmp2, **v4_tmp3, **v4_tmp4;

    std::vector<int> ik_vec, jk_vec, is_vec;

    auto nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " IALGO = 1 : Use different algorithm efficient when nbands >> nk_3ph\n";
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... \n";
        } else {
            exit("compute_V4_elements_mpi_over_kpoint", "This function can be used only when SELF_OFFDIAG = 1");
        }
    }

    allocate(nset_mpi, mympi->nprocs);

    const long int nset_tot = nk2_prod * ns;
    long int nset_each = nset_tot / mympi->nprocs;
    const long int nres = nset_tot - nset_each * mympi->nprocs;

    for (i = 0; i < mympi->nprocs; ++i) {
        nset_mpi[i] = nset_each;
        if (nres > i) {
            nset_mpi[i] += 1;
        }
    }

    MPI_Bcast(&nset_mpi[0], mympi->nprocs, MPI_LONG, 0, MPI_COMM_WORLD);
    long int nstart = 0;
    for (i = 0; i < mympi->my_rank; ++i) {
        nstart += nset_mpi[i];
    }
    long int nend = nstart + nset_mpi[mympi->my_rank];
    nset_each = nset_mpi[mympi->my_rank];
    deallocate(nset_mpi);

    ik_vec.clear();
    jk_vec.clear();
    is_vec.clear();

    long int icount = 0;
    for (ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
        for (is = 0; is < ns; ++is) {
            // if (is < js && relax_str == 0) continue;

            if (icount >= nstart && icount < nend) {
                ik_vec.push_back(ik_prod / nk_scph);
                jk_vec.push_back(ik_prod % nk_scph);
                is_vec.push_back(is);
            }
            ++icount;
        }
    }

    allocate(v4_array_at_kpair, ngroup_v4);
    allocate(ind, ngroup_v4, 4);
    allocate(v4_mpi, nk2_prod, ns2, ns2);
    allocate(v4_tmp0, ns2, ns2);
    allocate(v4_tmp1, ns, ns2);
    allocate(v4_tmp2, ns, ns2);
    allocate(v4_tmp3, ns, ns2);
    allocate(v4_tmp4, ns, ns2);

    for (ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#pragma omp parallel for private(js)
        for (is = 0; is < ns2; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_mpi[ik_prod][is][js] = complex_zero;
                v4_out[ik_prod][is][js] = complex_zero;
            }
        }
    }

    int ik_old = -1;
    int jk_old = -1;

    if (mympi->my_rank == 0) {
        std::cout << " Total number of sets to compute : " << nset_each << '\n';
    }

    for (long int ii = 0; ii < nset_each; ++ii) {

        auto ik_now = ik_vec[ii];
        auto jk_now = jk_vec[ii];
        auto is_now = is_vec[ii];

        if (!(ik_now == ik_old && jk_now == jk_old)) {

            // Update v4_array_at_kpair and ind

            knum = kmap_coarse_to_dense[kmesh_coarse_in->kpoint_irred_all[ik_now][0].knum];

            anharmonic_core->calc_phi4_reciprocal(kmesh_dense_in->xk[knum],
                                                  kmesh_dense_in->xk[jk_now],
                                                  kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[jk_now]],
                                                  phase_storage_in,
                                                  phi4_reciprocal_inout);

#ifdef _OPENMP
#pragma omp parallel for private(j)
#endif
            for (i = 0; i < ngroup_v4; ++i) {
                v4_array_at_kpair[i] = phi4_reciprocal_inout[i] * anharmonic_core->get_invmass_factor(4)[i];
                for (j = 0; j < 4; ++j)
                    ind[i][j] = anharmonic_core->get_evec_index(4)[i][j];
            }
            ik_old = ik_now;
            jk_old = jk_now;

            for (is4_1 = 0; is4_1 < ns4; is4_1++) {
                is2_1 = is4_1 / ns2;
                js2_1 = is4_1 % ns2;
                v4_tmp0[is2_1][js2_1] = complex_zero;
            }

            for (i = 0; i < ngroup_v4; ++i) {

                is = ind[i][0] * ns + ind[i][1];
                js = ind[i][2] * ns + ind[i][3];
                v4_tmp0[is][js] = v4_array_at_kpair[i];
            }
        }

        ik_prod = ik_now * nk_scph + jk_now;
        // int is_prod = ns * is_now + js_now;


        // initialize temporary matrices
#pragma omp parallel for private(js)
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp1[is][js] = complex_zero;
                v4_tmp2[is][js] = complex_zero;
                v4_tmp3[is][js] = complex_zero;
                v4_tmp4[is][js] = complex_zero;
            }
        }

        // transform the first index
#pragma omp parallel for private(is2_1, is, js, ks, ls)
        for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
            ks = is2_2 / ns;
            ls = is2_2 % ns;

            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;

                v4_tmp1[js][is2_2] += v4_tmp0[is2_1][is2_2] * std::conj(evec_in[knum][is_now][is]);
            }
        }

        // transform the second index
#pragma omp parallel for private(is2_1, is, js, ks, ls)
        for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
            ks = is2_2 / ns;
            ls = is2_2 % ns;

            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;

                v4_tmp2[is][is2_2] += v4_tmp1[js][is2_2] * evec_in[knum][is][js];
            }
        }


        // transform the third index
#pragma omp parallel for private(is2_2, is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                ks = is2_2 / ns;
                ls = is2_2 % ns;

                v4_tmp3[is][ks * ns + js] += v4_tmp2[is][ls * ns + js] * evec_in[jk_now][ks][ls];
            }
        }

        // transform the fourth index
#pragma omp parallel for private(is2_2, is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                ks = is2_2 / ns;
                ls = is2_2 % ns;

                v4_tmp4[is][js * ns + ks] += v4_tmp3[is][js * ns + ls] * std::conj(evec_in[jk_now][ks][ls]);
            }
        }

        // copy to the final matrix
#pragma omp parallel for private(is, js)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2 = 0; is2 < ns; is2++) {
                v4_mpi[ik_prod][is_now * ns + is2][is2_1] = factor * v4_tmp4[is2][is2_1];
            }
        }

        if (mympi->my_rank == 0) {
            std::cout << " SET " << ii + 1 << " done. \n";
        }

    } // loop over nk2_prod*ns

    deallocate(v4_array_at_kpair);
    deallocate(ind);

    // Now, communicate the calculated data.
    // When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (mympi->my_rank == 0) {
        std::cout << "Communicating v4_array over MPI ..." << std::flush;
    }
    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(&v4_mpi[ik_prod][0][0],
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#else
            MPI_Allreduce(&v4_mpi[ik_prod][0][0],
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_COMPLEX16,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(&v4_mpi[ik_prod][is][0],
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#else
                MPI_Allreduce(&v4_mpi[ik_prod][is][0],
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_COMPLEX16,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#endif
            }
        }
    }
    if (mympi->my_rank == 0) {
        std::cout << "done.\n";
    }

    deallocate(v4_mpi);
    deallocate(v4_tmp0);
    deallocate(v4_tmp1);
    deallocate(v4_tmp2);
    deallocate(v4_tmp3);
    deallocate(v4_tmp4);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v4_out, 4, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void Scph::zerofill_elements_acoustic_at_gamma(double **omega2, std::complex<double> ***v_elems, const int fc_order,
                                               const unsigned int nk_dense_in,
                                               const unsigned int nk_irred_coarse_in) const
{
    // Set V3 or V4 elements involving acoustic modes at Gamma point
    // exactly zero.

    int jk;
    int is, js, ks, ls;
    const auto ns = dynamical->neval;
    bool *is_acoustic;
    allocate(is_acoustic, ns);
    int nacoustic;
    auto threshould = 1.0e-24;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    if (!(fc_order == 3 || fc_order == 4)) {
        exit("zerofill_elements_acoustic_at_gamma", "The fc_order must be either 3 or 4.");
    }

    do {
        nacoustic = 0;
        for (is = 0; is < ns; ++is) {
            if (std::abs(omega2[0][is]) < threshould) {
                is_acoustic[is] = true;
                ++nacoustic;
            } else {
                is_acoustic[is] = false;
            }
        }
        if (nacoustic > 3) {
            exit("zerofill_elements_acoustic_at_gamma", "Could not assign acoustic modes at Gamma.");
        }
        threshould *= 2.0;
    } while (nacoustic < 3);


    if (fc_order == 3) {

        // Set V3 to zeros so as to avoid mixing with gamma acoustic modes
        // jk = 0;
        for (is = 0; is < ns; ++is) {
            for (ks = 0; ks < ns; ++ks) {
                for (ls = 0; ls < ns; ++ls) {
                    if (is_acoustic[ks] || is_acoustic[ls]) {
                        v_elems[0][is][ns * ks + ls] = complex_zero;
                    }
                }
            }
        }

        // ik = 0;
        for (jk = 0; jk < nk_dense_in; ++jk) {
            for (is = 0; is < ns; ++is) {
                if (is_acoustic[is]) {
                    for (ks = 0; ks < ns; ++ks) {
                        for (ls = 0; ls < ns; ++ls) {
                            v_elems[jk][is][ns * ks + ls] = complex_zero;
                        }
                    }
                }
            }
        }

    } else if (fc_order == 4) {
        // Set V4 to zeros so as to avoid mixing with gamma acoustic modes
        // jk = 0;
        for (int ik = 0; ik < nk_irred_coarse_in; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    for (ks = 0; ks < ns; ++ks) {
                        for (ls = 0; ls < ns; ++ls) {
                            if (is_acoustic[ks] || is_acoustic[ls]) {
                                v_elems[nk_dense_in * ik][ns * is + js][ns * ks + ls] = complex_zero;
                            }
                        }
                    }
                }
            }
        }
        // ik = 0;
        for (jk = 0; jk < nk_dense_in; ++jk) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    if (is_acoustic[is] || is_acoustic[js]) {
                        for (ks = 0; ks < ns; ++ks) {
                            for (ls = 0; ls < ns; ++ls) {
                                v_elems[jk][ns * is + js][ns * ks + ls] = complex_zero;
                            }
                        }
                    }
                }
            }
        }
    }

    deallocate(is_acoustic);
}
