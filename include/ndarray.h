/*
 ndarray.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>

/*
 NDArray<T, N>: RAII owner for the N-dimensional numeric arrays that were
 historically managed with allocate()/deallocate() (memory.h). It reproduces
 that layout EXACTLY -- one contiguous payload of n1*...*nN elements plus
 row-pointer tables on top -- so that

   - arr[i][j][k] indexing compiles to the same pointer-table walk,
   - &arr[0][0][0] is a valid contiguous buffer for MPI collectives,
   - rows can be handed to BLAS/LAPACK or wrapped in Eigen::Map unchanged,
   - the payload is DEFAULT-initialized (new T[n]), not value-initialized,
     preserving both the allocation cost and the NUMA first-touch behavior
     of the parallel fill loops (std::vector/make_unique would memset).

 The implicit conversion operators vend the same T** (and deeper) views the
 raw arrays provided, so call sites do not change. Ownership mistakes become
 compile errors: allocate() and deallocate() take an lvalue reference to the
 pointer, which the conversion's prvalue cannot bind to, and a
 delete-expression is ambiguous with two pointer conversion functions in
 play.

 Deliberate looseness: because of the implicit conversions, pointer
 arithmetic and comparisons on the object itself are technically legal
 (they act on the top-level table). Do not rely on that.

 Intentionally NOT provided: operator[] (the built-in surrogate subscript
 through the conversion already resolves unambiguously; a member operator[]
 would create real ambiguities), copy construction/assignment (owners move).
*/

namespace ndarray_detail
{
inline void report_bad_alloc_and_exit(const char *what, const int rank, const std::size_t *dims,
                                      const std::size_t bytes_per_elem)
{
    std::size_t ntot = 1;
    std::cout << " Caught an exception when trying to allocate " << rank << "-dimensional array" << '\n';
    std::cout << " " << what << " : Array shape = ";
    for (int i = 0; i < rank; ++i) {
        ntot *= dims[i];
        std::cout << dims[i] << (i + 1 < rank ? "x" : "");
    }
    std::cout << '\n';
    std::cout << " " << what << " : Array size (MB) = " << ntot * bytes_per_elem / 1000000 << '\n';
    std::exit(EXIT_FAILURE);
}

template <typename U>
U *new_or_die(const std::size_t n, const int rank, const std::size_t *dims, const std::size_t bytes_per_elem)
{
    try {
        return new U[n]; // default-initialization, matching memory.h
    } catch (std::bad_alloc &ba) {
        report_bad_alloc_and_exit(ba.what(), rank, dims, bytes_per_elem);
        return nullptr; // unreachable
    }
}
} // namespace ndarray_detail

template <typename T, std::size_t N>
class NDArray;

template <typename T>
class NDArray<T, 1>
{
public:
    NDArray() = default;

    explicit NDArray(const std::size_t n1) { resize(n1); }

    NDArray(const NDArray &) = delete;

    NDArray &operator=(const NDArray &) = delete;

    NDArray(NDArray &&) noexcept = default;

    NDArray &operator=(NDArray &&) noexcept = default;

    ~NDArray() = default;

    void resize(const std::size_t n1)
    {
        if (payload_ && n1 == n_[0]) return;
        clear();
        n_ = {n1};
        payload_.reset(ndarray_detail::new_or_die<T>(n1, 1, n_.data(), sizeof(T)));
    }

    void clear() noexcept
    {
        payload_.reset();
        n_ = {0};
    }

    operator T *() noexcept { return payload_.get(); }

    operator const T *() const noexcept { return payload_.get(); }

    T *data() noexcept { return payload_.get(); }

    const T *data() const noexcept { return payload_.get(); }

    T *ptr() noexcept { return payload_.get(); }

    std::size_t size() const noexcept { return n_[0]; }

    std::size_t dim(const std::size_t i) const noexcept { return n_[i]; }

    const std::array<std::size_t, 1> &shape() const noexcept { return n_; }

    bool empty() const noexcept { return !payload_; }

private:
    std::array<std::size_t, 1> n_{};
    std::unique_ptr<T[]> payload_;
};

template <typename T>
class NDArray<T, 2>
{
public:
    NDArray() = default;

    NDArray(const std::size_t n1, const std::size_t n2) { resize(n1, n2); }

    NDArray(const NDArray &) = delete;

    NDArray &operator=(const NDArray &) = delete;

    NDArray(NDArray &&) noexcept = default;

    NDArray &operator=(NDArray &&) noexcept = default;

    ~NDArray() = default;

    void resize(const std::size_t n1, const std::size_t n2)
    {
        if (table_ && n1 == n_[0] && n2 == n_[1]) return;
        clear();
        n_ = {n1, n2};
        // A zero leading dimension has no rows to point into: keep the null
        // table so `if (arr)` behaves like the memory.h nullptr convention.
        if (n1 == 0) return;
        payload_.reset(ndarray_detail::new_or_die<T>(n1 * n2, 2, n_.data(), sizeof(T)));
        table_.reset(ndarray_detail::new_or_die<T *>(n1, 2, n_.data(), sizeof(T)));
        for (std::size_t i = 0; i < n1; ++i) {
            table_[i] = payload_.get() + i * n2;
        }
    }

    void clear() noexcept
    {
        table_.reset();
        payload_.reset();
        n_ = {0, 0};
    }

    operator T **() noexcept { return table_.get(); }

    operator const T *const *() const noexcept { return table_.get(); }

    T *data() noexcept { return payload_.get(); }

    const T *data() const noexcept { return payload_.get(); }

    T **ptr() noexcept { return table_.get(); }

    std::size_t size() const noexcept { return n_[0] * n_[1]; }

    std::size_t dim(const std::size_t i) const noexcept { return n_[i]; }

    const std::array<std::size_t, 2> &shape() const noexcept { return n_; }

    bool empty() const noexcept { return !table_; }

private:
    std::array<std::size_t, 2> n_{};
    std::unique_ptr<T[]> payload_;
    std::unique_ptr<T *[]> table_;
};

template <typename T>
class NDArray<T, 3>
{
public:
    NDArray() = default;

    NDArray(const std::size_t n1, const std::size_t n2, const std::size_t n3) { resize(n1, n2, n3); }

    NDArray(const NDArray &) = delete;

    NDArray &operator=(const NDArray &) = delete;

    NDArray(NDArray &&) noexcept = default;

    NDArray &operator=(NDArray &&) noexcept = default;

    ~NDArray() = default;

    void resize(const std::size_t n1, const std::size_t n2, const std::size_t n3)
    {
        if (table_ && n1 == n_[0] && n2 == n_[1] && n3 == n_[2]) return;
        clear();
        n_ = {n1, n2, n3};
        if (n1 == 0) return;
        payload_.reset(ndarray_detail::new_or_die<T>(n1 * n2 * n3, 3, n_.data(), sizeof(T)));
        mid_.reset(ndarray_detail::new_or_die<T *>(n1 * n2, 3, n_.data(), sizeof(T)));
        table_.reset(ndarray_detail::new_or_die<T **>(n1, 3, n_.data(), sizeof(T)));
        for (std::size_t i = 0; i < n1; ++i) {
            table_[i] = mid_.get() + i * n2;
            for (std::size_t j = 0; j < n2; ++j) {
                mid_[i * n2 + j] = payload_.get() + i * n2 * n3 + j * n3;
            }
        }
    }

    void clear() noexcept
    {
        table_.reset();
        mid_.reset();
        payload_.reset();
        n_ = {0, 0, 0};
    }

    operator T ***() noexcept { return table_.get(); }

    operator const T *const *const *() const noexcept { return table_.get(); }

    T *data() noexcept { return payload_.get(); }

    const T *data() const noexcept { return payload_.get(); }

    T ***ptr() noexcept { return table_.get(); }

    std::size_t size() const noexcept { return n_[0] * n_[1] * n_[2]; }

    std::size_t dim(const std::size_t i) const noexcept { return n_[i]; }

    const std::array<std::size_t, 3> &shape() const noexcept { return n_; }

    bool empty() const noexcept { return !table_; }

private:
    std::array<std::size_t, 3> n_{};
    std::unique_ptr<T[]> payload_;
    std::unique_ptr<T *[]> mid_;
    std::unique_ptr<T **[]> table_;
};

template <typename T>
class NDArray<T, 4>
{
public:
    NDArray() = default;

    NDArray(const std::size_t n1, const std::size_t n2, const std::size_t n3, const std::size_t n4)
    {
        resize(n1, n2, n3, n4);
    }

    NDArray(const NDArray &) = delete;

    NDArray &operator=(const NDArray &) = delete;

    NDArray(NDArray &&) noexcept = default;

    NDArray &operator=(NDArray &&) noexcept = default;

    ~NDArray() = default;

    void resize(const std::size_t n1, const std::size_t n2, const std::size_t n3, const std::size_t n4)
    {
        if (table_ && n1 == n_[0] && n2 == n_[1] && n3 == n_[2] && n4 == n_[3]) return;
        clear();
        n_ = {n1, n2, n3, n4};
        if (n1 == 0) return;
        payload_.reset(ndarray_detail::new_or_die<T>(n1 * n2 * n3 * n4, 4, n_.data(), sizeof(T)));
        mid2_.reset(ndarray_detail::new_or_die<T *>(n1 * n2 * n3, 4, n_.data(), sizeof(T)));
        mid1_.reset(ndarray_detail::new_or_die<T **>(n1 * n2, 4, n_.data(), sizeof(T)));
        table_.reset(ndarray_detail::new_or_die<T ***>(n1, 4, n_.data(), sizeof(T)));
        for (std::size_t i = 0; i < n1; ++i) {
            table_[i] = mid1_.get() + i * n2;
            for (std::size_t j = 0; j < n2; ++j) {
                mid1_[i * n2 + j] = mid2_.get() + i * n2 * n3 + j * n3;
                for (std::size_t k = 0; k < n3; ++k) {
                    mid2_[i * n2 * n3 + j * n3 + k] = payload_.get() + i * n2 * n3 * n4 + j * n3 * n4 + k * n4;
                }
            }
        }
    }

    void clear() noexcept
    {
        table_.reset();
        mid1_.reset();
        mid2_.reset();
        payload_.reset();
        n_ = {0, 0, 0, 0};
    }

    operator T ****() noexcept { return table_.get(); }

    operator const T *const *const *const *() const noexcept { return table_.get(); }

    T *data() noexcept { return payload_.get(); }

    const T *data() const noexcept { return payload_.get(); }

    T ****ptr() noexcept { return table_.get(); }

    std::size_t size() const noexcept { return n_[0] * n_[1] * n_[2] * n_[3]; }

    std::size_t dim(const std::size_t i) const noexcept { return n_[i]; }

    const std::array<std::size_t, 4> &shape() const noexcept { return n_; }

    bool empty() const noexcept { return !table_; }

private:
    std::array<std::size_t, 4> n_{};
    std::unique_ptr<T[]> payload_;
    std::unique_ptr<T *[]> mid2_;
    std::unique_ptr<T **[]> mid1_;
    std::unique_ptr<T ***[]> table_;
};
