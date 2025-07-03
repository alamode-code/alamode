/*
 memory.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <iostream>

// memsize calculator

inline auto memsize_in_MB(const size_t size_of_one, const size_t n1) -> size_t
{
    const auto n = n1 * size_of_one;
    return n / 1000000;
}

inline auto memsize_in_MB(const size_t size_of_one, const size_t n1, const size_t n2) -> size_t
{
    const auto n = n1 * n2 * size_of_one;
    return n / 1000000;
}

inline auto memsize_in_MB(const size_t size_of_one, const size_t n1, const size_t n2, const size_t n3) -> size_t
{
    const auto n = n1 * n2 * n3 * size_of_one;
    return n / 1000000;
}

inline auto memsize_in_MB(const size_t size_of_one, const size_t n1, const size_t n2, const size_t n3, const size_t n4)
    -> size_t
{
    const auto n = n1 * n2 * n3 * n4 * size_of_one;
    return n / 1000000;
}

// Declaration and definition must be located in the same file for template functions.

/* allocator */

template <typename T>
auto allocate(T *&arr, const size_t n1) -> T *
{
    try {
        arr = new T[n1];
    } catch (std::bad_alloc &ba) {
        std::cout << " Caught an exception when trying to allocate 1-dimensional array" << '\n';
        std::cout << " " << ba.what() << " : Array shape = " << n1 << '\n';
        std::cout << " " << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1) << '\n';
        std::exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T **&arr, const size_t n1, const size_t n2) -> T **
{
    try {
        arr = new T *[n1];
        arr[0] = new T[n1 * n2];
        for (size_t i = 1; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
        }
    } catch (std::bad_alloc &ba) {
        std::cout << " Caught an exception when trying to allocate 2-dimensional array" << '\n';
        std::cout << " " << ba.what() << " : Array shape = " << n1 << "x" << n2 << '\n';
        std::cout << " " << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2) << '\n';
        std::exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T ***&arr, const size_t n1, const size_t n2, const size_t n3) -> T ***
{
    try {
        arr = new T **[n1];
        arr[0] = new T *[n1 * n2];
        arr[0][0] = new T[n1 * n2 * n3];
        for (size_t i = 0; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
            for (size_t j = 0; j < n2; ++j) {
                arr[i][j] = arr[0][0] + i * n2 * n3 + j * n3;
            }
        }
    } catch (std::bad_alloc &ba) {
        std::cout << " Caught an exception when trying to allocate 3-dimensional array" << '\n';
        std::cout << " " << ba.what() << " : Array shape = " << n1 << "x" << n2 << "x" << n3 << '\n';
        std::cout << " " << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2, n3) << '\n';
        std::exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T ****&arr, const size_t n1, const size_t n2, const size_t n3, const size_t n4) -> T ****
{
    try {
        arr = new T ***[n1];
        arr[0] = new T **[n1 * n2];
        arr[0][0] = new T *[n1 * n2 * n3];
        arr[0][0][0] = new T[n1 * n2 * n3 * n4];

        for (size_t i = 0; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
            for (size_t j = 0; j < n2; ++j) {
                arr[i][j] = arr[0][0] + i * n2 * n3 + j * n3;
                for (size_t k = 0; k < n3; ++k) {
                    arr[i][j][k] = arr[0][0][0] + i * n2 * n3 * n4 + j * n3 * n4 + k * n4;
                }
            }
        }
    } catch (std::bad_alloc &ba) {
        std::cout << " Caught an exception when trying to allocate 4-dimensional array" << '\n';
        std::cout << " " << ba.what() << " : Array shape = " << n1 << "x" << n2 << "x" << n3 << "x" << n4 << '\n';
        std::cout << " " << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2, n3, n4)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
    return arr;
}


/* deallocator */


template <typename T>
auto deallocate(T *&arr) -> void
{
    delete[] arr;
    arr = nullptr;
}

template <typename T>
auto deallocate(T **&arr) -> void
{
    delete[] arr[0];
    delete[] arr;
    arr = nullptr;
}

template <typename T>
auto deallocate(T ***&arr) -> void
{
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
    arr = nullptr;
}

template <typename T>
auto deallocate(T ****&arr) -> void
{
    delete[] arr[0][0][0];
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
    arr = nullptr;
}
