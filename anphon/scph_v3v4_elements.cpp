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
#include "scph_v3v4_elements.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "dynamical.h"
#include "error.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "scph.h"
#include "timer.h"
#include "write_phonons.h"
using namespace PHON_NS;

namespace
{
// CSC-like skeleton of the quartic-IFC scatter pattern phi4[(a1,a2)][(a3,a4)].
// The row/column indices depend only on evec_index_v4, which is fixed once in
// AnharmonicCore::setup_quartic(), so the skeleton is built once per kernel call
// and only the values are refilled for each (k1,k2) pair. Duplicate (row,col)
// pairs are merged into one slot; refilling in increasing group order keeps the
// last group's value, reproducing the dense scatter's overwrite semantics (the
// quadruplets are in fact unique after grouping, so this is a safe no-op).
struct SparsePhi4Skeleton
{
    std::vector<size_t> row;               // size nnz: row index a1*ns+a2
    std::vector<size_t> col_ptr;           // size ns2+1: offsets over col a3*ns+a4
    std::vector<size_t> slot_of_group;     // group g -> slot in row/val
    std::vector<std::complex<double>> val; // size nnz, refilled per k-pair
};

// One V4 index transform as a complex GEMM (zgemm via blas_wrapper.h, cf.
// dynamical.cpp). All buffers are row-major contiguous: the input viewed as
// (ns x ns^3) is contracted over its OUTERMOST mode index with E[out][in], and
// the result is written as (ns^3 x ns) with the new index innermost, i.e. the
// flat layout rotates [a b c d] -> [b c d out]. In column-major BLAS terms this
// is C(ns x ns^3) = E_buf^T * In_buf^T.
inline void transform_v4_index_gemm(const std::complex<double> *evec_row_major, const std::complex<double> *buf_in,
                                    std::complex<double> *buf_out, const size_t ns, const std::complex<double> alpha_in)
{
    int m = static_cast<int>(ns);
    int n = static_cast<int>(ns * ns * ns);
    int k = static_cast<int>(ns);
    auto alpha = alpha_in;
    auto beta = std::complex<double>(0.0, 0.0);
    zgemm_cpx("T",
              "T",
              &m,
              &n,
              &k,
              &alpha,
              const_cast<std::complex<double> *>(evec_row_major),
              &m,
              const_cast<std::complex<double> *>(buf_in),
              &n,
              &beta,
              buf_out,
              &m);
}

auto build_phi4_skeleton(const int *const *evec_index, const long int ngroup, const size_t ns) -> SparsePhi4Skeleton
{
    struct Entry
    {
        size_t row, col, group;
    };
    std::vector<Entry> entries(ngroup);
    for (long int g = 0; g < ngroup; ++g) {
        entries[g].row = static_cast<size_t>(evec_index[g][0]) * ns + evec_index[g][1];
        entries[g].col = static_cast<size_t>(evec_index[g][2]) * ns + evec_index[g][3];
        entries[g].group = g;
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        if (a.col != b.col) return a.col < b.col;
        return a.row < b.row;
    });

    SparsePhi4Skeleton skeleton;
    skeleton.slot_of_group.resize(ngroup);
    skeleton.col_ptr.assign(ns * ns + 1, 0);
    // Merge duplicates while recording each group's slot (stable sort keeps the
    // original group order within equal (col,row), so later groups overwrite).
    size_t prev_col = 0;
    bool have_prev = false;
    size_t prev_row = 0;
    for (const auto &entry: entries) {
        const bool new_slot = !have_prev || entry.col != prev_col || entry.row != prev_row;
        if (new_slot) {
            skeleton.row.push_back(entry.row);
            ++skeleton.col_ptr[entry.col + 1];
            prev_col = entry.col;
            prev_row = entry.row;
            have_prev = true;
        }
        skeleton.slot_of_group[entry.group] = skeleton.row.size() - 1;
    }
    for (size_t c = 0; c < ns * ns; ++c) {
        skeleton.col_ptr[c + 1] += skeleton.col_ptr[c];
    }
    skeleton.val.resize(skeleton.row.size());
    return skeleton;
}
} // namespace
void ScphQhaCommon::compute_V3_elements_mpi_over_kpoint(
    std::complex<double> ***v3_out, const std::complex<double> *const *const *evec_in,
    const bool self_offdiag, const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
    const PhaseFactorCache *phase_cache_in, std::complex<double> *phi3_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    auto ns = dynamical->neval;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    NDArray<unsigned int, 2> ind;
    unsigned int i, j;

    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v3 = anharmonic_core->get_ngroup_fcs(3);
    const auto factor = pow2(0.5) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    NDArray<std::complex<double>, 1> v3_array_at_kpair;

    NDArray<std::complex<double>, 2> v3_tmp0;
    NDArray<std::complex<double>, 2> v3_tmp1;
    NDArray<std::complex<double>, 2> v3_tmp2;
    NDArray<std::complex<double>, 2> v3_tmp3;

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        if (self_offdiag) {
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v3_array ... ";
        } else {
            std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v3_array ... ";
        }
    }

    v3_array_at_kpair.resize(ngroup_v3);
    ind.resize(ngroup_v3, 3);

    v3_tmp0.resize(ns, ns2);
    v3_tmp1.resize(ns, ns2);
    v3_tmp2.resize(ns, ns2);
    v3_tmp3.resize(ns, ns2);

    // v3_out may come from STL-backed storage via pointer bridges and is not guaranteed
    // to be contiguous across the k-point dimension, so the MPI reduction cannot happen
    // in v3_out itself. Each rank writes its disjoint (strided) ik slices into this
    // zero-initialized contiguous buffer, which is then summed in place over MPI.
    std::vector<std::complex<double>> v3_allreduce_buffer(static_cast<std::size_t>(nk_scph) * ns3);

    for (unsigned int ik = mympi->my_rank; ik < nk_scph; ik += mympi->nprocs) {

        anharmonic_core->calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[ik]],
                                              anharmonic_core->get_ngroup_fcs(3),
                                              anharmonic_core->get_fcs_group(3),
                                              anharmonic_core->get_relvec(3),
                                              phase_cache_in,
                                              phi3_reciprocal_inout);

#pragma omp parallel for private(j)
        for (ii = 0; ii < ngroup_v3; ++ii) {
            v3_array_at_kpair[ii] = phi3_reciprocal_inout[ii] * anharmonic_core->get_invmass_factor(3)[ii];
            for (j = 0; j < 3; ++j) ind[ii][j] = anharmonic_core->get_evec_index(3)[ii][j];
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

            // copy to the reduction buffer
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + js2_1] =
                    factor * v3_tmp3[is][js2_1];
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

                    v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + ns * js + ks] = factor * ret;
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

                    v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + (ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    v3_array_at_kpair.clear();
    ind.clear();
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(MPI_IN_PLACE,
                  v3_allreduce_buffer.data(),
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
    MPI_Allreduce(MPI_IN_PLACE,
                  v3_allreduce_buffer.data(),
                  static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

#pragma omp parallel for collapse(3) schedule(static)
    for (unsigned int ik = 0; ik < nk_scph; ++ik) {
        for (unsigned int is_local = 0; is_local < ns; ++is_local) {
            for (unsigned int js_local = 0; js_local < ns2; ++js_local) {
                const auto idx = (static_cast<std::size_t>(ik) * ns + is_local) * ns2 + js_local;
                v3_out[ik][is_local][js_local] = v3_allreduce_buffer[idx];
            }
        }
    }

    v3_tmp0.clear();
    v3_tmp1.clear();
    v3_tmp2.clear();
    v3_tmp3.clear();


    zerofill_elements_acoustic_at_gamma(v3_out, 3, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

// A free function (all inputs explicit) so that DerivativeIFC can compute
// V3 elements of strain-derivative IFCs without a live Scph instance.
// The implementation shares its structure with
// ScphQhaCommon::compute_V3_elements_mpi_over_kpoint; merging the two is a
// possible future cleanup.
void PHON_NS::compute_V3_elements_for_given_IFCs(
    std::complex<double> ***v3_out, const std::vector<bool> &is_acoustic_gamma_in, const int ngroup_v3_in,
    std::vector<double> *fcs_group_v3_in, std::vector<RelativeVector> *relvec_v3_in, double *invmass_v3_in,
    int **evec_index_v3_in, const std::complex<double> *const *const *evec_in, const bool self_offdiag,
    const unsigned int ns_in, const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
    const PhaseFactorCache *phase_storage_in, AnharmonicCore &anharmonic_core_in, const int my_rank, const int nprocs)
{
    const auto ns = ns_in;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    NDArray<unsigned int, 2> ind;
    unsigned int i, j;
    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto factor = pow2(0.5) / static_cast<double>(nk_scph);
    static auto complex_zero = std::complex<double>(0.0, 0.0);
    NDArray<std::complex<double>, 1> v3_array_at_kpair;
    NDArray<std::complex<double>, 1> phi3_reciprocal_tmp;

    NDArray<std::complex<double>, 2> v3_tmp0;
    NDArray<std::complex<double>, 2> v3_tmp1;
    NDArray<std::complex<double>, 2> v3_tmp2;
    NDArray<std::complex<double>, 2> v3_tmp3;

    if (ngroup_v3_in == 0) {
#pragma omp parallel for collapse(3) schedule(static)
        for (unsigned int ik = 0; ik < nk_scph; ++ik) {
            for (unsigned int is_local = 0; is_local < ns; ++is_local) {
                for (unsigned int js_local = 0; js_local < ns2; ++js_local) {
                    v3_out[ik][is_local][js_local] = complex_zero;
                }
            }
        }
        zerofill_elements_acoustic_at_gamma(is_acoustic_gamma_in,
                                            v3_out,
                                            3,
                                            ns,
                                            kmesh_dense_in->nk,
                                            kmesh_coarse_in->nk_irred);
        return;
    }

    phi3_reciprocal_tmp.resize(ngroup_v3_in);
    v3_array_at_kpair.resize(ngroup_v3_in);
    ind.resize(ngroup_v3_in, 3);

    v3_tmp0.resize(ns, ns2);
    v3_tmp1.resize(ns, ns2);
    v3_tmp2.resize(ns, ns2);
    v3_tmp3.resize(ns, ns2);

    // v3_out may come from STL-backed storage via pointer bridges and is not guaranteed
    // to be contiguous across the k-point dimension, so the MPI reduction cannot happen
    // in v3_out itself. Each rank writes its disjoint (strided) ik slices into this
    // zero-initialized contiguous buffer, which is then summed in place over MPI.
    std::vector<std::complex<double>> v3_allreduce_buffer(static_cast<std::size_t>(nk_scph) * ns3);

    for (unsigned int ik = my_rank; ik < nk_scph; ik += nprocs) {

        anharmonic_core_in.calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
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
            for (j = 0; j < 3; ++j) ind[ii][j] = evec_index_v3_in[ii][j];
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

            // copy to the reduction buffer
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + js2_1] =
                    factor * v3_tmp3[is][js2_1];
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

                    v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + ns * js + ks] = factor * ret;
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

                    v3_allreduce_buffer[(static_cast<std::size_t>(ik) * ns + is) * ns2 + (ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    v3_array_at_kpair.clear();
    ind.clear();
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(MPI_IN_PLACE,
                  v3_allreduce_buffer.data(),
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
    MPI_Allreduce(MPI_IN_PLACE,
                  v3_allreduce_buffer.data(),
                  static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

#pragma omp parallel for collapse(3) schedule(static)
    for (unsigned int ik = 0; ik < nk_scph; ++ik) {
        for (unsigned int is_local = 0; is_local < ns; ++is_local) {
            for (unsigned int js_local = 0; js_local < ns2; ++js_local) {
                const auto idx = (static_cast<std::size_t>(ik) * ns + is_local) * ns2 + js_local;
                v3_out[ik][is_local][js_local] = v3_allreduce_buffer[idx];
            }
        }
    }

    v3_tmp0.clear();
    v3_tmp1.clear();
    v3_tmp2.clear();
    v3_tmp3.clear();

    zerofill_elements_acoustic_at_gamma(is_acoustic_gamma_in,
                                        v3_out,
                                        3,
                                        ns,
                                        kmesh_dense_in->nk,
                                        kmesh_coarse_in->nk_irred);
}


void ScphQhaCommon::compute_V4_elements_mpi_over_kpoint(std::complex<double> ***v4_out,
                                                        std::complex<double> ***evec_in, const bool self_offdiag,
                                                        const bool relax, const KpointMeshUniform *kmesh_coarse_in,
                                                        const KpointMeshUniform *kmesh_dense_in,
                                                        const std::vector<int> &kmap_coarse_to_dense,
                                                        const PhaseFactorCache *phase_storage_in,
                                                        std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    const size_t nk_reduced_interpolate = kmesh_coarse_in->nk_irred;
    const size_t ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns4 = ns * ns * ns * ns;
    size_t is, js, ks, ls;
    size_t is2_1, is2_2;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    const auto factor = pow2(0.5) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    NDArray<std::complex<double>, 3> evec_conj;

    NDArray<std::complex<double>, 2> v4_tmp1;
    NDArray<std::complex<double>, 2> v4_tmp2;

    const size_t nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        const auto nsize_dble =
            static_cast<double>((nk2_prod * ns4 + 2 * ns4) * sizeof(std::complex<double>)) / 1000000000.0;
        if (writes->getVerbosity() > 0) {
            std::cout << " Estimated memory usage for the V4 arrays per MPI process: " << std::setw(10) << std::fixed
                      << std::setprecision(4) << nsize_dble << " GByte.\n";
            if (self_offdiag) {
                std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... " << std::flush;
            } else {
                std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v4_array ... " << std::flush;
            }
        }
    }

    evec_conj.resize(kmesh_dense_in->nk, ns, ns);

    v4_tmp1.resize(ns2, ns2);
    v4_tmp2.resize(ns2, ns2);

    // Sparse representation of the phi4 scatter pattern; the indices are fixed for
    // the entire calculation, only the values are refilled per (k1,k2) pair.
    auto phi4_skeleton = build_phi4_skeleton(anharmonic_core->get_evec_index(4), ngroup_v4, ns);
    const double *invmass_v4 = anharmonic_core->get_invmass_factor(4);

    const long int nks2 = kmesh_dense_in->nk * ns2;

#pragma omp parallel for private(is, js)
    for (long int iks = 0; iks < nks2; ++iks) {
        size_t ik = iks / ns2;
        is = (iks - ik * ns2) / ns;
        js = iks % ns;
        evec_conj[ik][is][js] = std::conj(evec_in[ik][is][js]);
    }

    // Each rank computes a disjoint subset of the ik_prod slices of v4_out and the
    // result is summed in place over MPI, so v4_out must be zero everywhere first.
#pragma omp parallel for
    for (long int iel = 0; iel < static_cast<long int>(nk2_prod * ns4); ++iel) {
        v4_out[0][0][iel] = complex_zero;
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

        // Refill the phi4 values of the sparse skeleton. Every slot is refilled
        // (each slot belongs to at least one group), and increasing group order
        // keeps the last group's value on duplicates.
        for (ii = 0; ii < ngroup_v4; ++ii) {
            phi4_skeleton.val[phi4_skeleton.slot_of_group[ii]] = phi4_reciprocal_inout[ii] * invmass_v4[ii];
        }

        if (self_offdiag || relaxation->relax_str) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (loop diagram).

            // initialize the target of the first transform; the remaining transforms
            // overwrite every element of their target buffer.
#pragma omp parallel for private(js)
            for (is = 0; is < ns2; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v4_tmp1[is][js] = complex_zero;
                }
            }

            // scatter phi4 and transform the first index in one sparse pass into the
            // ROTATED layout v4_tmp1[(a2,a3,a4)][i]; threads own distinct columns, so
            // the shared (a2, col) target runs never race across threads
#pragma omp parallel for
            for (long int col = 0; col < static_cast<long int>(ns2); ++col) {
                for (size_t p = phi4_skeleton.col_ptr[col]; p < phi4_skeleton.col_ptr[col + 1]; ++p) {
                    const size_t a1 = phi4_skeleton.row[p] / ns;
                    const size_t a2 = phi4_skeleton.row[p] % ns;
                    const auto val = phi4_skeleton.val[p];
                    auto *dst = &v4_tmp1[0][0] + (a2 * ns2 + col) * ns;
                    for (size_t i = 0; i < ns; ++i) {
                        dst[i] += val * evec_conj[knum][i][a1];
                    }
                }
            }

            // The remaining transforms are matrix products: each one contracts the
            // outermost mode index of the (ns x ns^3) view of the current buffer and
            // appends the new index innermost, so the layout rotates as
            // [a2 a3 a4 i] -> [a3 a4 i j] -> [a4 i j k] -> [i j k m], and after the
            // fourth transform the flat layout coincides with v4_out[ik_prod].
            constexpr auto complex_one = std::complex<double>(1.0, 0.0);

            // transform the second index (v4_tmp1 -> v4_tmp2)
            transform_v4_index_gemm(&evec_in[knum][0][0], &v4_tmp1[0][0], &v4_tmp2[0][0], ns, complex_one);

            // transform the third index (v4_tmp2 -> v4_tmp1)
            transform_v4_index_gemm(&evec_in[jk][0][0], &v4_tmp2[0][0], &v4_tmp1[0][0], ns, complex_one);

            // transform the fourth index and store to the final matrix (v4_tmp1 -> v4_out)
            transform_v4_index_gemm(&evec_conj[jk][0][0],
                                    &v4_tmp1[0][0],
                                    &v4_out[ik_prod][0][0],
                                    ns,
                                    std::complex<double>(factor, 0.0));

        } else {

            // initialize the target of the first transform (only its first ns rows
            // are used in the diagonal-only branch)
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v4_tmp1[is][js] = complex_zero;
                }
            }

            // scatter phi4 and transform the first and the second index in one
            // sparse pass (-> v4_tmp1[i][(a3,a4)], diagonal in the first two indices)
#pragma omp parallel for
            for (long int col = 0; col < static_cast<long int>(ns2); ++col) {
                for (size_t p = phi4_skeleton.col_ptr[col]; p < phi4_skeleton.col_ptr[col + 1]; ++p) {
                    const size_t a1 = phi4_skeleton.row[p] / ns;
                    const size_t a2 = phi4_skeleton.row[p] % ns;
                    const auto val = phi4_skeleton.val[p];
                    for (size_t i = 0; i < ns; ++i) {
                        v4_tmp1[i][col] += val * evec_conj[knum][i][a1] * evec_in[knum][i][a2];
                    }
                }
            }

            // transform the third and the fourth index and store to the final matrix
#pragma omp parallel for private(is, js, ks, ls, is2_2)
            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;

                auto accumulator = complex_zero;
                for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                    ks = is2_2 / ns;
                    ls = is2_2 % ns;

                    accumulator += v4_tmp1[is][is2_2] * evec_in[jk][js][ks] * evec_conj[jk][js][ls];
                }
                v4_out[ik_prod][(ns + 1) * is][(ns + 1) * js] = factor * accumulator;
            }
        }
    }


    evec_conj.clear();

    v4_tmp1.clear();
    v4_tmp2.clear();

    // Now, communicate the calculated data. Each rank filled a disjoint subset of
    // v4_out (zero elsewhere), so an in-place summation assembles the full tensor
    // without a second staging buffer of the same size.
    // When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(MPI_IN_PLACE, &v4_out[0][0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
        MPI_Allreduce(MPI_IN_PLACE, &v4_out[0][0][0], count, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(MPI_IN_PLACE,
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#else
            MPI_Allreduce(MPI_IN_PLACE, &v4_out[ik_prod][0][0], count_sub, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(MPI_IN_PLACE,
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#else
                MPI_Allreduce(MPI_IN_PLACE, &v4_out[ik_prod][is][0], ns2, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
            }
        }
    }

    zerofill_elements_acoustic_at_gamma(v4_out, 4, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void ScphQhaCommon::compute_V4_elements_mpi_over_band(std::complex<double> ***v4_out,
                                                      std::complex<double> ***evec_in, const bool self_offdiag,
                                                      const KpointMeshUniform *kmesh_coarse_in,
                                                      const KpointMeshUniform *kmesh_dense_in,
                                                      const std::vector<int> &kmap_coarse_to_dense,
                                                      const PhaseFactorCache *phase_storage_in,
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
    size_t is2_1, is2_2;
    size_t is2;
    unsigned int knum;
    unsigned int i;
    NDArray<long int, 1> nset_mpi;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    auto factor = pow2(0.5) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    NDArray<std::complex<double>, 2> v4_tmp1;
    NDArray<std::complex<double>, 2> v4_tmp2;

    std::vector<int> ik_vec, jk_vec, is_vec;

    auto nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            if (writes->getVerbosity() > 0) {
                std::cout << " IALGO = 1 : Use different algorithm efficient when nbands >> nk_3ph\n";
            }
            const auto nsize_dble =
                static_cast<double>((nk2_prod * ns4 + 2 * ns * ns2) * sizeof(std::complex<double>)) / 1000000000.0;
            if (writes->getVerbosity() > 0) {
                std::cout << " Estimated memory usage for the V4 arrays per MPI process: " << std::setw(10)
                          << std::fixed << std::setprecision(4) << nsize_dble << " GByte.\n";
                std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... \n";
            }
        } else {
            exit("compute_V4_elements_mpi_over_kpoint", "This function can be used only when SELF_OFFDIAG = 1");
        }
    }

    nset_mpi.resize(mympi->nprocs);

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
    nset_mpi.clear();

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

    v4_tmp1.resize(ns, ns2);
    v4_tmp2.resize(ns, ns2);

    // Sparse representation of the phi4 scatter pattern; the indices are fixed for
    // the entire calculation, only the values are refilled per (k1,k2) pair.
    auto phi4_skeleton = build_phi4_skeleton(anharmonic_core->get_evec_index(4), ngroup_v4, ns);
    const double *invmass_v4 = anharmonic_core->get_invmass_factor(4);

    // Each rank computes a disjoint set of rows of the ik_prod slices of v4_out and
    // the result is summed in place over MPI, so v4_out must be zero everywhere first.
    for (ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#pragma omp parallel for private(js)
        for (is = 0; is < ns2; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_out[ik_prod][is][js] = complex_zero;
            }
        }
    }

    int ik_old = -1;
    int jk_old = -1;

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " Total number of sets to compute : " << nset_each << '\n';
    }

    for (long int ii = 0; ii < nset_each; ++ii) {

        auto ik_now = ik_vec[ii];
        auto jk_now = jk_vec[ii];
        auto is_now = is_vec[ii];

        if (!(ik_now == ik_old && jk_now == jk_old)) {

            // Update the phi4 values of the sparse skeleton

            knum = kmap_coarse_to_dense[kmesh_coarse_in->kpoint_irred_all[ik_now][0].knum];

            anharmonic_core->calc_phi4_reciprocal(kmesh_dense_in->xk[knum],
                                                  kmesh_dense_in->xk[jk_now],
                                                  kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[jk_now]],
                                                  phase_storage_in,
                                                  phi4_reciprocal_inout);

            // Every slot is refilled (each slot belongs to at least one group), and
            // increasing group order keeps the last group's value on duplicates.
            for (i = 0; i < ngroup_v4; ++i) {
                phi4_skeleton.val[phi4_skeleton.slot_of_group[i]] = phi4_reciprocal_inout[i] * invmass_v4[i];
            }
            ik_old = ik_now;
            jk_old = jk_now;
        }

        ik_prod = ik_now * nk_scph + jk_now;

        // initialize the target of the first transform; the remaining transforms
        // overwrite every element of their target buffer.
#pragma omp parallel for private(js)
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp1[is][js] = complex_zero;
            }
        }

        // transform the first index (sparse phi4 -> v4_tmp1); threads own distinct
        // columns, so the shared (row%ns, col) targets never race across threads
#pragma omp parallel for
        for (long int col = 0; col < static_cast<long int>(ns2); ++col) {
            for (size_t p = phi4_skeleton.col_ptr[col]; p < phi4_skeleton.col_ptr[col + 1]; ++p) {
                const size_t a1 = phi4_skeleton.row[p] / ns;
                const size_t a2 = phi4_skeleton.row[p] % ns;
                v4_tmp1[a2][col] += phi4_skeleton.val[p] * std::conj(evec_in[knum][is_now][a1]);
            }
        }

        // transform the second index (v4_tmp1 -> v4_tmp2)
#pragma omp parallel for private(is, js)
        for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
            for (is = 0; is < ns; ++is) {
                auto accumulator = complex_zero;
                for (js = 0; js < ns; ++js) {
                    accumulator += v4_tmp1[js][is2_2] * evec_in[knum][is][js];
                }
                v4_tmp2[is][is2_2] = accumulator;
            }
        }

        // transform the third index (v4_tmp2 -> v4_tmp1)
#pragma omp parallel for private(is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (ks = 0; ks < ns; ++ks) {
                auto accumulator = complex_zero;
                for (ls = 0; ls < ns; ++ls) {
                    accumulator += v4_tmp2[is][ls * ns + js] * evec_in[jk_now][ks][ls];
                }
                v4_tmp1[is][ks * ns + js] = accumulator;
            }
        }

        // transform the fourth index (v4_tmp1 -> v4_tmp2)
#pragma omp parallel for private(is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (ks = 0; ks < ns; ++ks) {
                auto accumulator = complex_zero;
                for (ls = 0; ls < ns; ++ls) {
                    accumulator += v4_tmp1[is][js * ns + ls] * std::conj(evec_in[jk_now][ks][ls]);
                }
                v4_tmp2[is][js * ns + ks] = accumulator;
            }
        }

        // copy to the final matrix
#pragma omp parallel for private(is, js)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2 = 0; is2 < ns; is2++) {
                v4_out[ik_prod][is_now * ns + is2][is2_1] = factor * v4_tmp2[is2][is2_1];
            }
        }

        // Report progress roughly 20 times over the whole loop.
        if (mympi->my_rank == 0) {
            const long int nreport = std::max(nset_each / 20, static_cast<long int>(1));
            if ((ii + 1) % nreport == 0 || ii + 1 == nset_each) {
                if (writes->getVerbosity() > 0) std::cout << " SET " << ii + 1 << " / " << nset_each << " done. \n";
            }
        }

    } // loop over nk2_prod*ns

    // Now, communicate the calculated data. Each rank filled a disjoint set of rows
    // of v4_out (zero elsewhere), so an in-place summation assembles the full tensor
    // without a second staging buffer of the same size.
    // When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "Communicating v4_array over MPI ..." << std::flush;
    }
    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(MPI_IN_PLACE, &v4_out[0][0][0], count, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
        MPI_Allreduce(MPI_IN_PLACE, &v4_out[0][0][0], count, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(MPI_IN_PLACE,
                          &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#else
            MPI_Allreduce(MPI_IN_PLACE, &v4_out[ik_prod][0][0], count_sub, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(MPI_IN_PLACE,
                              &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#else
                MPI_Allreduce(MPI_IN_PLACE, &v4_out[ik_prod][is][0], ns2, MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
            }
        }
    }
    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "done.\n";
    }

    v4_tmp1.clear();
    v4_tmp2.clear();

    zerofill_elements_acoustic_at_gamma(v4_out, 4, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void ScphQhaCommon::zerofill_elements_acoustic_at_gamma(std::complex<double> ***v_elems, const int fc_order,
                                                        const unsigned int nk_dense_in,
                                                        const unsigned int nk_irred_coarse_in) const
{
    PHON_NS::zerofill_elements_acoustic_at_gamma(is_acoustic_gamma_harm,
                                                 v_elems,
                                                 fc_order,
                                                 dynamical->neval,
                                                 nk_dense_in,
                                                 nk_irred_coarse_in);
}

void PHON_NS::zerofill_elements_acoustic_at_gamma(const std::vector<bool> &is_acoustic,
                                                  std::complex<double> ***v_elems, const int fc_order,
                                                  const unsigned int ns_in, const unsigned int nk_dense_in,
                                                  const unsigned int nk_irred_coarse_in)
{
    // Set V3 or V4 elements involving acoustic modes at Gamma point
    // exactly zero. The acoustic modes are assigned from the eigenvectors
    // (see Dynamical::detect_acoustic_modes_at_gamma), not from the
    // magnitude of the harmonic frequencies.

    int jk;
    int is, js, ks, ls;
    const auto ns = ns_in;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    if (!(fc_order == 3 || fc_order == 4)) {
        exit("zerofill_elements_acoustic_at_gamma", "The fc_order must be either 3 or 4.");
    }

    if (std::count(is_acoustic.begin(), is_acoustic.end(), true) != 3) {
        exit("zerofill_elements_acoustic_at_gamma", "Could not assign acoustic modes at Gamma.");
    }


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

}
