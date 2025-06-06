/*
 memory.h

 Copyright (c) 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <iostream>

// memsize calculator

inline int memsize_in_MB(int size_of_one, int n1)
{
    unsigned long n = n1 * size_of_one;
    return n / 1000000;
}

inline int memsize_in_MB(int size_of_one, int n1, int n2)
{
    unsigned long n = n1 * n2 * size_of_one;
    return n / 1000000;
}

inline int memsize_in_MB(int size_of_one, int n1, int n2, int n3)
{
    unsigned long n = n1 * n2 * n3 * size_of_one;
    return n / 1000000;
}

inline int memsize_in_MB(int size_of_one, int n1, int n2, int n3, int n4)
{
    unsigned long n = n1 * n2 * n3 * n4 * size_of_one;
    return n / 1000000;
}


template <typename T>
auto allocate(T *&arr, int n1) -> T *
{
    try {
        arr = new T[n1];
    } catch (std::bad_alloc &ba) {
        std::cout << "Caught an exception when trying to allocate 1-dimensional array" << std::endl;
        std::cout << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1) << std::endl;
        exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T **&arr, int n1, int n2) -> T **
{
    try {
        arr = new T *[n1];
        arr[0] = new T[n1 * n2];
        for (int i = 1; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
        }
    } catch (std::bad_alloc &ba) {
        std::cout << "Caught an exception when trying to allocate 2-dimensional array" << std::endl;
        std::cout << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2) << std::endl;
        exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T ***&arr, int n1, int n2, int n3) -> T ***
{
    try {
        arr = new T **[n1];
        arr[0] = new T *[n1 * n2];
        arr[0][0] = new T[n1 * n2 * n3];
        for (int i = 0; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
            for (int j = 0; j < n2; ++j) {
                arr[i][j] = arr[0][0] + i * n2 * n3 + j * n3;
            }
        }
    } catch (std::bad_alloc &ba) {
        std::cout << "Caught an exception when trying to allocate 3-dimensional array" << std::endl;
        std::cout << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2, n3) << std::endl;
        exit(EXIT_FAILURE);
    }
    return arr;
}

template <typename T>
auto allocate(T ****&arr, int n1, int n2, int n3, int n4) -> T ****
{
    try {
        arr = new T ***[n1];
        arr[0] = new T **[n1 * n2];
        arr[0][0] = new T *[n1 * n2 * n3];
        arr[0][0][0] = new T[n1 * n2 * n3 * n4];

        for (int i = 0; i < n1; ++i) {
            arr[i] = arr[0] + i * n2;
            for (int j = 0; j < n2; ++j) {
                arr[i][j] = arr[0][0] + i * n2 * n3 + j * n3;
                for (int k = 0; k < n3; ++k) {
                    arr[i][j][k] = arr[0][0][0] + i * n2 * n3 * n4 + j * n3 * n4 + k * n4;
                }
            }
        }
    } catch (std::bad_alloc &ba) {
        std::cout << "Caught an exception when trying to allocate 3-dimensional array" << std::endl;
        std::cout << ba.what() << " : Array size (MB) = " << memsize_in_MB(sizeof(T), n1, n2, n3, n4) << std::endl;
        exit(EXIT_FAILURE);
    }
    return arr;
}

// deallocator

template <typename T>
auto deallocate(T *&arr) -> void
{
    delete[] arr;
}

template <typename T>
auto deallocate(T **&arr) -> void
{
    delete[] arr[0];
    delete[] arr;
}

template <typename T>
auto deallocate(T ***&arr) -> void
{
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
}

template <typename T>
auto deallocate(T ****&arr) -> void
{
    delete[] arr[0][0][0];
    delete[] arr[0][0];
    delete[] arr[0];
    delete[] arr;
}
