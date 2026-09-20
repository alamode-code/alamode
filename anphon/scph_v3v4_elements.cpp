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
#include "v4_distributed.h"
#include "v4_index_transform.h"
#include "write_phonons.h"
using namespace PHON_NS;
using PHON_NS::v4_index_transform::transform_index_gemm;

namespace
{
// Cache the CSC-like scatter pattern phi4[(a1,a2)][(a3,a4)] from
// evec_index_v4 and refill values per (k1,k2). Merge duplicate slots,
// keeping the last group's value to match dense-scatter semantics.
struct SparsePhi4Skeleton
{
    std::vector<size_t> row;               // size nnz: row index a1*ns+a2
    std::vector<size_t> col_ptr;           // size ns2+1: offsets over col a3*ns+a4
    std::vector<size_t> slot_of_group;     // group g -> slot in row/val
    std::vector<std::complex<double>> val; // size nnz, refilled per k-pair
};

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
    std::complex<double> ***v3_out, const std::complex<double> *const *const *evec_in, const bool self_offdiag,
    const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
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
    std::vector<std::complex<double>> evec_conj_ik(ns2);

    if (run.my_rank == 0 && run.verbosity > 0) {
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

    // v3_out may come from STL-backed storage via pointer bridges and is not guaranteed
    // to be contiguous across the k-point dimension, so the MPI reduction cannot happen
    // in v3_out itself. Each rank writes its disjoint (strided) ik slices into this
    // zero-initialized contiguous buffer, which is then summed in place over MPI.
    std::vector<std::complex<double>> v3_allreduce_buffer(static_cast<std::size_t>(nk_scph) * ns3);

    for (unsigned int ik = run.my_rank; ik < nk_scph; ik += run.nprocs) {

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

            // v3_tmp0 holds the (alpha,mu)-representation Phi(a,b,c), row-major [a][b*ns+c].
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                }
            }

#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // Three rotating index-transform GEMMs (v4_index_transform.h), each contracting
            // the outermost index: [a b c] -> [b c i] -> [c i j] -> [i j k], with
            // evec[0][i][a], evec[ik][j][b] and conj(evec[ik][k][c]). The last one writes
            // the ik slice of the reduction buffer directly, scaled by the prefactor.
            // evec_in[ik] must be a contiguous ns x ns row-major block (NDArray storage).
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    evec_conj_ik[is * ns + js] = std::conj(evec_in[ik][is][js]);
                }
            }
            constexpr auto complex_one = std::complex<double>(1.0, 0.0);
            transform_index_gemm(&evec_in[0][0][0], &v3_tmp0[0][0], &v3_tmp1[0][0], ns, ns2, complex_one);
            transform_index_gemm(&evec_in[ik][0][0], &v3_tmp1[0][0], &v3_tmp2[0][0], ns, ns2, complex_one);
            transform_index_gemm(evec_conj_ik.data(),
                                 &v3_tmp2[0][0],
                                 v3_allreduce_buffer.data() + static_cast<std::size_t>(ik) * ns3,
                                 ns,
                                 ns2,
                                 std::complex<double>(factor, 0.0));

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


    zerofill_elements_acoustic_at_gamma(v3_out, 3, kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (run.my_rank == 0 && run.verbosity > 0) {
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
    std::vector<std::complex<double>> evec_conj_ik(ns2);

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

            // v3_tmp0 holds the (alpha,mu)-representation Phi(a,b,c), row-major [a][b*ns+c].
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                }
            }

#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3_in; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // Three rotating index-transform GEMMs (v4_index_transform.h), each contracting
            // the outermost index: [a b c] -> [b c i] -> [c i j] -> [i j k], with
            // evec[0][i][a], evec[ik][j][b] and conj(evec[ik][k][c]). The last one writes
            // the ik slice of the reduction buffer directly, scaled by the prefactor.
            // evec_in[ik] must be a contiguous ns x ns row-major block (NDArray storage).
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    evec_conj_ik[is * ns + js] = std::conj(evec_in[ik][is][js]);
                }
            }
            constexpr auto complex_one = std::complex<double>(1.0, 0.0);
            transform_index_gemm(&evec_in[0][0][0], &v3_tmp0[0][0], &v3_tmp1[0][0], ns, ns2, complex_one);
            transform_index_gemm(&evec_in[ik][0][0], &v3_tmp1[0][0], &v3_tmp2[0][0], ns, ns2, complex_one);
            transform_index_gemm(evec_conj_ik.data(),
                                 &v3_tmp2[0][0],
                                 v3_allreduce_buffer.data() + static_cast<std::size_t>(ik) * ns3,
                                 ns,
                                 ns2,
                                 std::complex<double>(factor, 0.0));

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

    zerofill_elements_acoustic_at_gamma(is_acoustic_gamma_in,
                                        v3_out,
                                        3,
                                        ns,
                                        kmesh_dense_in->nk,
                                        kmesh_coarse_in->nk_irred);
}


void ScphQhaCommon::compute_V4_elements_mpi_over_kpoint(v4_distributed::V4RowBlock &v4_block,
                                                        std::complex<double> ***evec_in, const bool self_offdiag,
                                                        const bool relax, const KpointMeshUniform *kmesh_coarse_in,
                                                        const KpointMeshUniform *kmesh_dense_in,
                                                        const std::vector<int> &kmap_coarse_to_dense,
                                                        const PhaseFactorCache *phase_storage_in,
                                                        std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

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


    if (run.my_rank == 0) {
        const auto nsize_dble =
            static_cast<double>((v4_block.nrows_local() * ns2 + 2 * ns4) * sizeof(std::complex<double>)) / 1000000000.0;
        if (run.verbosity > 0) {
            std::cout << " Estimated memory usage for the V4 arrays on this process (local rows + 2 ns^4 scratch): "
                      << std::setw(10) << std::fixed << std::setprecision(4) << nsize_dble << " GByte.\n";
            if (self_offdiag || relax) {
                std::cout << " Calculating all components of v4_array ... " << std::flush;
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

    // This rank computes the whole slices it owns (the slice partition keeps
    // slices intact); the diagonal-only regime relies on the block being zero-filled
    // at setup, the full-tensor regime overwrites every owned element.
    if (v4_block.unit_begin % ns != 0 || v4_block.unit_end % ns != 0) {
        exit("compute_V4_elements_mpi_over_kpoint", "The owned V4 rows are not aligned to whole slices.");
    }
    const size_t slice_begin = v4_block.unit_begin / ns;
    const size_t slice_end = v4_block.unit_end / ns;

    for (size_t ik_prod = slice_begin; ik_prod < slice_end; ++ik_prod) {
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

        if (self_offdiag || relax) {

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

            // Contract outermost indices with GEMMs on ns x ns^3 views:
            // [a2 a3 a4 i] -> [a3 a4 i j] -> [a4 i j k] -> [i j k m].
            // The final layout matches the owned v4 slice.
            constexpr auto complex_one = std::complex<double>(1.0, 0.0);

            // transform the second index (v4_tmp1 -> v4_tmp2)
            transform_index_gemm(&evec_in[knum][0][0], &v4_tmp1[0][0], &v4_tmp2[0][0], ns, ns * ns2, complex_one);

            // transform the third index (v4_tmp2 -> v4_tmp1)
            transform_index_gemm(&evec_in[jk][0][0], &v4_tmp2[0][0], &v4_tmp1[0][0], ns, ns * ns2, complex_one);

            // transform the fourth index and store to the final matrix (v4_tmp1 -> v4 rows)
            transform_index_gemm(&evec_conj[jk][0][0],
                                 &v4_tmp1[0][0],
                                 v4_block.row(ik_prod * ns, 0), // the ns^2 owned rows of this slice are contiguous
                                 ns,
                                 ns * ns2,
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
                v4_block.row(ik_prod * ns + is, is)[(ns + 1) * js] = factor * accumulator;
            }
        }
    }


    evec_conj.clear();

    v4_tmp1.clear();
    v4_tmp2.clear();

    zerofill_v4_acoustic_at_gamma(v4_block);

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void ScphQhaCommon::compute_V4_elements_mpi_over_band(v4_distributed::V4RowBlock &v4_block,
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
    int is, js;
    unsigned int knum;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    auto factor = pow2(0.5) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    NDArray<std::complex<double>, 2> v4_tmp1;
    NDArray<std::complex<double>, 2> v4_tmp2;

    unsigned int i;
    std::vector<int> ik_vec, jk_vec, is_vec;

    auto nk2_prod = nk_reduced_interpolate * nk_scph;

    // Every element is computed, so this builder serves whenever the full tensor is
    // needed (SELF_OFFDIAG = 1 or structural relaxation).
    if (!(self_offdiag || relaxation->relax_str)) {
        exit("compute_V4_elements_mpi_over_band", "This function can be used only when the full V4 tensor is needed");
    }
    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << " IALGO = 1 : Use different algorithm efficient when nbands >> nk_3ph\n";
        const auto nsize_dble =
            static_cast<double>((v4_block.nrows_local() * ns2 + 2 * ns * ns2) * sizeof(std::complex<double>)) /
            1000000000.0;
        std::cout << " Estimated memory usage for the V4 arrays on this process: " << std::setw(10) << std::fixed
                  << std::setprecision(4) << nsize_dble << " GByte.\n";
        std::cout << " Calculating all components of v4_array ... \n";
    }

    // The sets (ik_prod, is) of this rank are exactly its owned units of the
    // row-distributed layout (unit u = ik_prod * ns + is).
    const long int nstart = static_cast<long int>(v4_block.unit_begin);
    const long int nend = static_cast<long int>(v4_block.unit_end);
    const long int nset_each = nend - nstart;

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

    int ik_old = -1;
    int jk_old = -1;
    // conj(E_j) as a contiguous ns x ns matrix for the fourth index transform;
    // filled on the first owned set and whenever the (ik, jk) pair changes
    std::vector<std::complex<double>> evec_conj_jk(ns2);

    if (run.my_rank == 0 && run.verbosity > 0) {
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
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    evec_conj_jk[is * ns + js] = std::conj(evec_in[jk_now][is][js]);
                }
            }
            ik_old = ik_now;
            jk_old = jk_now;
        }

        ik_prod = ik_now * nk_scph + jk_now;

        // initialize the target of the scatter (it accumulates with +=)
#pragma omp parallel for private(js)
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp1[is][js] = complex_zero;
            }
        }

        // scatter phi4 and transform the first index in one sparse pass
        // (-> v4_tmp1[a2][(a3 a4)]); threads own distinct columns, so the shared
        // (row%ns, col) targets never race across threads
#pragma omp parallel for
        for (long int col = 0; col < static_cast<long int>(ns2); ++col) {
            for (size_t p = phi4_skeleton.col_ptr[col]; p < phi4_skeleton.col_ptr[col + 1]; ++p) {
                const size_t a1 = phi4_skeleton.row[p] / ns;
                const size_t a2 = phi4_skeleton.row[p] % ns;
                v4_tmp1[a2][col] += phi4_skeleton.val[p] * std::conj(evec_in[knum][is_now][a1]);
            }
        }

        // Contract ns x ns^2 views: [a2 a3 a4] -> [a3 a4 j] -> [a4 j k] -> [j k l].
        // Write the scaled result to owned rows (is_now, j), columns k*ns + l.
        constexpr auto complex_one = std::complex<double>(1.0, 0.0);
        transform_index_gemm(&evec_in[knum][0][0], &v4_tmp1[0][0], &v4_tmp2[0][0], ns, ns2, complex_one);
        transform_index_gemm(&evec_in[jk_now][0][0], &v4_tmp2[0][0], &v4_tmp1[0][0], ns, ns2, complex_one);
        transform_index_gemm(evec_conj_jk.data(),
                             &v4_tmp1[0][0],
                             v4_block.row(ik_prod * ns + is_now, 0), // the ns rows of the unit are contiguous
                             ns,
                             ns2,
                             std::complex<double>(factor, 0.0));

        // Report progress roughly 20 times over the whole loop.
        if (run.my_rank == 0) {
            const long int nreport = std::max(nset_each / 20, static_cast<long int>(1));
            if ((ii + 1) % nreport == 0 || ii + 1 == nset_each) {
                if (run.verbosity > 0) std::cout << " SET " << ii + 1 << " / " << nset_each << " done. \n";
            }
        }

    } // loop over nk2_prod*ns

    v4_tmp1.clear();
    v4_tmp2.clear();

    zerofill_v4_acoustic_at_gamma(v4_block);

    if (run.my_rank == 0 && run.verbosity > 0) {
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

void PHON_NS::zerofill_elements_acoustic_at_gamma(const std::vector<bool> &is_acoustic, std::complex<double> ***v_elems,
                                                  const int fc_order, const unsigned int ns_in,
                                                  const unsigned int nk_dense_in, const unsigned int nk_irred_coarse_in)
{
    // Set V3 or V4 elements involving acoustic modes at Gamma point
    // exactly zero. The acoustic modes are assigned from the eigenvectors
    // (see Dynamical::detect_acoustic_modes_at_gamma), not from the
    // magnitude of the harmonic frequencies.

    int jk;
    int is, js, ks, ls;
    const auto ns = ns_in;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    if (fc_order != 3) {
        exit(
            "zerofill_elements_acoustic_at_gamma",
            "Only the cubic elements use this function; the quartic ones are row-distributed (zerofill_v4_acoustic_at_gamma_block).");
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
    }
}

void ScphQhaCommon::zerofill_v4_acoustic_at_gamma(v4_distributed::V4RowBlock &v4_block) const
{
    PHON_NS::zerofill_v4_acoustic_at_gamma_block(is_acoustic_gamma_harm, v4_block);
}

void PHON_NS::zerofill_v4_acoustic_at_gamma_block(const std::vector<bool> &is_acoustic,
                                                  v4_distributed::V4RowBlock &v4_block)
{
    // Row-distributed version of zerofill_elements_acoustic_at_gamma(fc_order = 4):
    // the columns of the Gamma-column slices (ik, jk = 0) and the rows of the
    // Gamma-row slices (ik_irred = 0, jk) that involve an acoustic mode at Gamma
    // are set to zero on the rows this rank owns.
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    const auto ns = v4_block.ns;
    const auto ns2 = v4_block.ns2;
    const auto nk = v4_block.nk_dense;

    if (std::count(is_acoustic.begin(), is_acoustic.end(), true) != 3) {
        exit("zerofill_v4_acoustic_at_gamma_block", "Could not assign acoustic modes at Gamma.");
    }

    std::vector<std::size_t> acoustic_columns;
    for (std::size_t ks = 0; ks < ns; ++ks) {
        for (std::size_t ls = 0; ls < ns; ++ls) {
            if (is_acoustic[ks] || is_acoustic[ls]) {
                acoustic_columns.push_back(ks * ns + ls);
            }
        }
    }

    // columns of the slices (ik, 0)
    for (std::size_t ik = 0; ik < v4_block.nk_irred; ++ik) {
        const std::size_t ik_prod = ik * nk;
        std::size_t a0, a1;
        v4_block.owned_a_range(ik_prod, a0, a1);
        for (std::size_t a = a0; a < a1; ++a) {
            const auto u = v4_distributed::V4RowBlock::unit_of(ik_prod, a, ns);
            for (std::size_t b = 0; b < ns; ++b) {
                auto *row = v4_block.row(u, b);
                for (const auto c: acoustic_columns) {
                    row[c] = complex_zero;
                }
            }
        }
    }
    // rows of the slices (0, jk)
    for (std::size_t jk = 0; jk < nk; ++jk) {
        const std::size_t ik_prod = jk;
        std::size_t a0, a1;
        v4_block.owned_a_range(ik_prod, a0, a1);
        for (std::size_t a = a0; a < a1; ++a) {
            const auto u = v4_distributed::V4RowBlock::unit_of(ik_prod, a, ns);
            for (std::size_t b = 0; b < ns; ++b) {
                if (is_acoustic[a] || is_acoustic[b]) {
                    std::fill(v4_block.row(u, b), v4_block.row(u, b) + ns2, complex_zero);
                }
            }
        }
    }
}
