/*
 v4_service.cpp

 See v4_service.h.
*/

#include "v4_service.h"
#include <algorithm>
#include <climits>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include "error.h"
#include "q0_contraction.h"

using namespace PHON_NS;

namespace
{
#ifdef MPI_CXX_DOUBLE_COMPLEX
const MPI_Datatype mpi_complex_type = MPI_CXX_DOUBLE_COMPLEX;
#else
const MPI_Datatype mpi_complex_type = MPI_COMPLEX16;
#endif
constexpr std::size_t mpi_chunk = static_cast<std::size_t>(1) << 30; // elements per MPI call

// Counts that go through a single (unchunked) MPI call must fit an int.
int mpi_count(const std::size_t n, const char *what)
{
    if (n > static_cast<std::size_t>(INT_MAX)) {
        PHON_NS::exit("V4Service", what);
    }
    return static_cast<int>(n);
}
} // namespace

V4Service::V4Service(const int my_rank, const int nprocs) : my_rank_(my_rank), nprocs_(nprocs)
{}

void V4Service::setup(const std::size_t ns, const std::size_t nk_dense, const std::size_t nk_irred,
                      const std::size_t ik_gamma_irred, const std::size_t jk_gamma_dense, const bool full_tensor,
                      const bool offdiag_fmat, const Partition kind)
{
    ns_ = ns;
    ns2_ = ns * ns;
    nk_dense_ = nk_dense;
    nk_irred_ = nk_irred;
    ik_gamma_irred_ = ik_gamma_irred;
    jk_gamma_dense_ = jk_gamma_dense;
    full_tensor_ = full_tensor;
    offdiag_fmat_ = offdiag_fmat;

    std::vector<std::size_t> bounds;
    if (kind == Partition::Units) {
        const auto weight =
            v4_distributed::unit_weights(ns, nk_dense, nk_irred, ik_gamma_irred, jk_gamma_dense, offdiag_fmat);
        bounds = v4_distributed::partition_units(weight, nprocs_);
    } else {
        bounds = v4_distributed::partition_slices(nk_irred * nk_dense, ns, nprocs_);
    }
    block_.allocate(ns, nk_dense, nk_irred, bounds[my_rank_], bounds[my_rank_ + 1]);

    if (!full_tensor_ && block_.nrows_local() > 0) {
        // only the on-site diagonal entries are written by the builder
        std::fill(&block_.rows[0][0], &block_.rows[0][0] + block_.nrows_local() * ns2_, std::complex<double>(0.0, 0.0));
    }

    // pointer table over the local rows
    const std::size_t nslices = nk_irred * nk_dense;
    rowtab_storage_.assign(nslices * ns2_, nullptr);
    rowtab_.assign(nslices, nullptr);
    for (std::size_t ik_prod = 0; ik_prod < nslices; ++ik_prod) {
        rowtab_[ik_prod] = rowtab_storage_.data() + ik_prod * ns2_;
        std::size_t a0, a1;
        block_.owned_a_range(ik_prod, a0, a1);
        for (std::size_t a = a0; a < a1; ++a) {
            const auto u = v4_distributed::V4RowBlock::unit_of(ik_prod, a, ns);
            for (std::size_t b = 0; b < ns; ++b) {
                rowtab_[ik_prod][a * ns + b] = block_.row(u, b);
            }
        }
    }

    v4_diag_.resize(nk_irred, ns);
}

std::complex<double> ***V4Service::row_table()
{
    return rowtab_.data();
}

void V4Service::finalize_build(const std::vector<unsigned int> &knum_of_irred, const unsigned int verbosity)
{
    const auto ns = ns_;
    for (std::size_t ik = 0; ik < nk_irred_; ++ik) {
        for (std::size_t a = 0; a < ns; ++a) {
            v4_diag_[ik][a] = 0.0;
        }
        const std::size_t ik_prod = ik * nk_dense_ + knum_of_irred[ik];
        std::size_t a0, a1;
        block_.owned_a_range(ik_prod, a0, a1);
        for (std::size_t a = a0; a < a1; ++a) {
            const auto u = v4_distributed::V4RowBlock::unit_of(ik_prod, a, ns);
            v4_diag_[ik][a] = block_.row(u, a)[(ns + 1) * a].real();
        }
    }
    if (distributed()) {
        MPI_Allreduce(MPI_IN_PLACE,
                      &v4_diag_[0][0],
                      mpi_count(nk_irred_ * ns, "the V4 diagonal exceeds INT_MAX elements"),
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }

    double bytes_min = block_.bytes_local(), bytes_max = block_.bytes_local();
    if (distributed()) {
        MPI_Allreduce(MPI_IN_PLACE, &bytes_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &bytes_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    if (my_rank_ == 0 && verbosity > 0) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(4);
        if (distributed()) {
            line << " V4 rows distributed over " << nprocs_ << " MPI processes: " << bytes_min / 1.0e9 << " - "
                 << bytes_max / 1.0e9 << " GByte per process.\n";
        } else {
            line << " V4 array: " << bytes_max / 1.0e9 << " GByte.\n";
        }
        std::cout << line.str();
    }
}

void V4Service::broadcast_opcode(const int op) const
{
    int code = op;
    MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void V4Service::reduce_to_root(std::complex<double> *buf, const std::size_t count) const
{
    for (std::size_t offset = 0; offset < count; offset += mpi_chunk) {
        const auto n = static_cast<int>(std::min(mpi_chunk, count - offset));
        if (my_rank_ == 0) {
            MPI_Reduce(MPI_IN_PLACE, buf + offset, n, mpi_complex_type, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(buf + offset, nullptr, n, mpi_complex_type, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }
}

void V4Service::broadcast_from_root(std::complex<double> *buf, const std::size_t count) const
{
    for (std::size_t offset = 0; offset < count; offset += mpi_chunk) {
        const auto n = static_cast<int>(std::min(mpi_chunk, count - offset));
        MPI_Bcast(buf + offset, n, mpi_complex_type, 0, MPI_COMM_WORLD);
    }
}

void V4Service::fmat_batch_local_and_reduce(const std::complex<double> *dmat, const std::size_t nrhs,
                                            std::complex<double> *fout)
{
    const auto count = nrhs * nk_irred_ * ns2_;
    std::fill(fout, fout + count, std::complex<double>(0.0, 0.0));
    v4_distributed::accumulate_fmat_batch(block_, dmat, nrhs, fout);
    if (distributed()) {
        reduce_to_root(fout, count);
    }
}

void V4Service::fmat_batch(const std::complex<double> *dmat, const std::size_t nrhs, std::complex<double> *fout)
{
    if (!full_tensor_) {
        PHON_NS::exit("V4Service::fmat_batch", "the batched contraction needs every V4 element (full tensor).");
    }
    if (distributed()) {
        broadcast_opcode(OP_FMAT_BATCH);
        auto n = static_cast<unsigned long long>(nrhs);
        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        broadcast_from_root(const_cast<std::complex<double> *>(dmat), nrhs * nk_dense_ * ns2_);
    }
    fmat_batch_local_and_reduce(dmat, nrhs, fout);
}

void V4Service::fmat_local_and_reduce(const std::complex<double> *dvec, std::complex<double> ***fmat_inout)
{
    v4_distributed::accumulate_fmat(block_, dvec, offdiag_fmat_, fmat_inout);
    if (distributed()) {
        reduce_to_root(&fmat_inout[0][0][0], nk_irred_ * ns2_);
    }
}

void V4Service::fmat(const std::complex<double> *dvec, std::complex<double> ***fmat_inout)
{
    if (distributed()) {
        broadcast_opcode(OP_FMAT);
        MPI_Bcast(
            const_cast<std::complex<double> *>(dvec),
            mpi_count(nk_dense_ * ns2_, "the D matrices exceed INT_MAX elements; the FMAT broadcast needs chunking"),
            mpi_complex_type,
            0,
            MPI_COMM_WORLD);
    }
    fmat_local_and_reduce(dvec, fmat_inout);
}

void V4Service::q0_local_and_reduce(const double *q0, const std::complex<double> *const *const *v3_with_umn,
                                    std::complex<double> ***v3_renorm, std::complex<double> ***q4_q0)
{
    q0_contraction::Options opt;
    opt.unit_begin = block_.unit_begin;
    opt.unit_end = block_.unit_end;
    opt.seed_v3_with_umn = (my_rank_ == 0);
    q0_contraction::contract_v4_with_q0(ns_,
                                        nk_dense_,
                                        nk_irred_,
                                        ik_gamma_irred_,
                                        jk_gamma_dense_,
                                        row_table(),
                                        q0,
                                        v3_with_umn,
                                        v3_renorm,
                                        q4_q0,
                                        opt);
    if (distributed()) {
        reduce_to_root(&v3_renorm[0][0][0], nk_dense_ * ns_ * ns2_);
        reduce_to_root(&q4_q0[0][0][0], nk_irred_ * ns2_);
    }
}

void V4Service::q0_sweep(const double *q0, const std::complex<double> *const *const *v3_with_umn,
                         std::complex<double> ***v3_renorm, std::complex<double> ***q4_q0)
{
    auto q0_is_zero = true;
    for (std::size_t a = 0; a < ns_; ++a) {
        if (q0[a] != 0.0) {
            q0_is_zero = false;
            break;
        }
    }
    if (!distributed() || q0_is_zero) {
        // local fast path (no quartic terms) or single process: the kernel does everything
        q0_contraction::Options opt;
        if (q0_is_zero) {
            q0_contraction::contract_v4_with_q0(ns_,
                                                nk_dense_,
                                                nk_irred_,
                                                ik_gamma_irred_,
                                                jk_gamma_dense_,
                                                nullptr,
                                                q0,
                                                v3_with_umn,
                                                v3_renorm,
                                                q4_q0,
                                                opt);
        } else {
            opt.unit_begin = block_.unit_begin;
            opt.unit_end = block_.unit_end;
            q0_contraction::contract_v4_with_q0(ns_,
                                                nk_dense_,
                                                nk_irred_,
                                                ik_gamma_irred_,
                                                jk_gamma_dense_,
                                                row_table(),
                                                q0,
                                                v3_with_umn,
                                                v3_renorm,
                                                q4_q0,
                                                opt);
        }
        return;
    }
    broadcast_opcode(OP_Q0);
    MPI_Bcast(const_cast<double *>(q0), mpi_count(ns_, "ns exceeds INT_MAX"), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    q0_local_and_reduce(q0, v3_with_umn, v3_renorm, q4_q0);
}

void V4Service::finish()
{
    if (distributed()) {
        broadcast_opcode(OP_DONE);
    }
}

void V4Service::worker_loop()
{
    if (!distributed() || my_rank_ == 0) {
        return;
    }
    constexpr std::complex<double> czero(0.0, 0.0);
    dvec_buf_.assign(nk_dense_ * ns2_, czero);
    q0_buf_.assign(ns_, 0.0);
    fmat_buf_.resize(nk_irred_, ns_, ns_);

    while (true) {
        int code = 0;
        MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (code == OP_DONE) {
            break;
        }
        if (code == OP_FMAT) {
            MPI_Bcast(dvec_buf_.data(),
                      mpi_count(dvec_buf_.size(),
                                "the D matrices exceed INT_MAX elements; the FMAT broadcast needs chunking"),
                      mpi_complex_type,
                      0,
                      MPI_COMM_WORLD);
            std::fill(&fmat_buf_[0][0][0], &fmat_buf_[0][0][0] + fmat_buf_.size(), czero);
            fmat_local_and_reduce(dvec_buf_.data(), fmat_buf_);
        } else if (code == OP_FMAT_BATCH) {
            unsigned long long n = 0;
            MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            const auto nrhs = static_cast<std::size_t>(n);
            dmat_batch_buf_.resize(nrhs * nk_dense_ * ns2_);
            fmat_batch_buf_.resize(nrhs * nk_irred_ * ns2_);
            broadcast_from_root(dmat_batch_buf_.data(), dmat_batch_buf_.size());
            fmat_batch_local_and_reduce(dmat_batch_buf_.data(), nrhs, fmat_batch_buf_.data());
        } else if (code == OP_Q0) {
            if (v3_buf_.size() == 0) {
                v3_buf_.resize(nk_dense_, ns_, ns2_);
                q4_buf_.resize(nk_irred_, ns_, ns_);
            }
            MPI_Bcast(q0_buf_.data(), mpi_count(ns_, "ns exceeds INT_MAX"), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            q0_local_and_reduce(q0_buf_.data(), nullptr, v3_buf_, q4_buf_);
        } else {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // release the worker buffers
    std::vector<std::complex<double>>().swap(dvec_buf_);
    std::vector<std::complex<double>>().swap(dmat_batch_buf_);
    std::vector<std::complex<double>>().swap(fmat_batch_buf_);
    fmat_buf_.clear();
    v3_buf_.clear();
    q4_buf_.clear();
}
