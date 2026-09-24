/*
 v4_service.h

 Row-distributed V4 for the SCPH/QHA structural optimization: the local
 block of every MPI rank (v4_distributed.h) plus the two collective
 contractions the drivers need, organized so that rank 0 keeps the whole
 existing control flow (temperature loop, structure loop, optimizer, all I/O)
 while the other ranks only serve opcodes:

   FMAT  the anharmonic part of the SCP self-energy for all coarse
         irreducible k from the D matrices of the dense mesh (once per SCPH
         iteration),
   Q0    the q0 renormalization sweep (once per structure step),
   DONE  leave the worker loop.

 Rank 0 calls fmat() / q0_sweep() from the existing code and finish() at the
 end of its rank-0 block, before the drivers' final broadcasts of the
 results; the other ranks call worker_loop() instead of the rank-0 block.
 With one process everything runs locally and no MPI call is made.

 Fatal errors on any rank go through PHON_NS::exit -> MPI_Abort, so a
 waiting worker cannot be left behind by an error on rank 0.
*/

#pragma once

#include <complex>
#include <cstddef>
#include <vector>
#include "ndarray.h"
#include "v4_distributed.h"

namespace PHON_NS
{

class V4Service
{
public:
    enum class Partition
    {
        Units,
        Slices
    };

    V4Service(int my_rank, int nprocs);
    ~V4Service() = default;

    // Dimensions and the owned range of this rank; allocates the local block.
    // offdiag_fmat: SELF_OFFDIAG (lower-triangle rows of the Fmat contraction);
    // full_tensor: every element is built (SELF_OFFDIAG or structural
    // relaxation), otherwise only the on-site diagonal entries are written and
    // the block is zero-filled so that the rest reads as zero.
    void setup(std::size_t ns, std::size_t nk_dense, std::size_t nk_irred, std::size_t ik_gamma_irred,
               std::size_t jk_gamma_dense, bool full_tensor, bool offdiag_fmat, Partition kind);

    v4_distributed::V4RowBlock &block()
    {
        return block_;
    }

    const v4_distributed::V4RowBlock &block() const
    {
        return block_;
    }

    // Pointer table v4[ik_prod][a*ns+b] over the local block (nullptr for rows
    // this rank does not own), for the q0 sweep kernel.
    std::complex<double> ***row_table();

    // After the builders: the on-site diagonal v4[(ik_irred, knum)][(ns+1)a][(ns+1)a]
    // (knum = dense index of coarse irreducible k) gathered on every rank, and
    // the per-rank memory line (rank 0, verbosity > 0).
    void finalize_build(const std::vector<unsigned int> &knum_of_irred, unsigned int verbosity);

    const double *const *v4_diag() const
    {
        return v4_diag_;
    }

    std::size_t ik_gamma_irred() const
    {
        return ik_gamma_irred_;
    }

    std::size_t jk_gamma_dense() const
    {
        return jk_gamma_dense_;
    }

    // ---- rank 0 ----
    // fmat_inout[ik_irred][a][b] (nk_irred x ns x ns, contiguous) is seeded by the
    // caller with the harmonic F (the other ranks add their partial anharmonic
    // contributions); on return it holds F on rank 0 (lower triangle or diagonal
    // updated, see v4_distributed::accumulate_fmat).
    void fmat(const std::complex<double> *dvec, std::complex<double> ***fmat_inout);

    // ---- rank 0 ----
    // Batched full-matrix contraction for linear-response solves
    // (v4_distributed::accumulate_fmat_batch): fout (nrhs blocks of
    // nk_irred * ns * ns) is overwritten with V4 contracted with the nrhs D
    // blocks of dmat (nrhs blocks of nk_dense * ns * ns); complete on rank 0.
    // Needs the full tensor; dmat and fout must not overlap.
    void fmat_batch(const std::complex<double> *dmat, std::size_t nrhs, std::complex<double> *fout);

    // The q0 sweep of q0_contraction.h over the distributed rows; on return
    // v3_renorm and q4_q0 are complete on rank 0. A q0 that is exactly zero
    // takes the local fast path without involving the other ranks.
    void q0_sweep(const double *q0, const std::complex<double> *const *const *v3_with_umn,
                  std::complex<double> ***v3_renorm, std::complex<double> ***q4_q0);

    // Release the other ranks from worker_loop().
    void finish();

    // ---- other ranks ----
    void worker_loop();

    bool distributed() const
    {
        return nprocs_ > 1;
    }

private:
    enum Opcode : int
    {
        OP_FMAT = 1,
        OP_Q0 = 2,
        OP_DONE = 3,
        OP_FMAT_BATCH = 4
    };

    void broadcast_opcode(int op) const;
    void reduce_to_root(std::complex<double> *buf, std::size_t count) const;
    void broadcast_from_root(std::complex<double> *buf, std::size_t count) const;
    void fmat_batch_local_and_reduce(const std::complex<double> *dmat, std::size_t nrhs, std::complex<double> *fout);
    void fmat_local_and_reduce(const std::complex<double> *dvec, std::complex<double> ***fmat_inout);
    void q0_local_and_reduce(const double *q0, const std::complex<double> *const *const *v3_with_umn,
                             std::complex<double> ***v3_renorm, std::complex<double> ***q4_q0);

    int my_rank_, nprocs_;
    std::size_t ns_ = 0, ns2_ = 0, nk_dense_ = 0, nk_irred_ = 0;
    std::size_t ik_gamma_irred_ = 0, jk_gamma_dense_ = 0;
    bool full_tensor_ = true, offdiag_fmat_ = true;

    v4_distributed::V4RowBlock block_;
    std::vector<std::complex<double> *> rowtab_storage_;
    std::vector<std::complex<double> **> rowtab_;
    NDArray<double, 2> v4_diag_;

    // worker buffers
    std::vector<std::complex<double>> dvec_buf_;
    std::vector<double> q0_buf_;
    NDArray<std::complex<double>, 3> fmat_buf_;
    NDArray<std::complex<double>, 3> v3_buf_;
    NDArray<std::complex<double>, 3> q4_buf_;
    std::vector<std::complex<double>> dmat_batch_buf_, fmat_batch_buf_;
};

} // namespace PHON_NS
