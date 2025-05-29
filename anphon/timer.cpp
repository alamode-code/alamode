/*
 timer.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "timer.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>

using namespace PHON_NS;

Timer::Timer(PHON *phon) :
    Pointers(phon)
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&time_ref);
    QueryPerformanceFrequency(&frequency);
#else
    gettimeofday(&time_ref, NULL);
#endif
}

Timer::~Timer() {}

void Timer::reset()
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&time_ref);
#else
    gettimeofday(&time_ref, NULL);
#endif
}

double Timer::elapsed() const
{
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER time_now;
    QueryPerformanceCounter(&time_now);
    return static_cast<double>(time_now.QuadPart - time_ref.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
    timeval time_now;
    gettimeofday(&time_now, NULL);
    return (time_now.tv_sec - time_ref.tv_sec) + (time_now.tv_usec - time_ref.tv_usec) * 1.0e-6;
#endif
}

void Timer::print_elapsed() const
{
    std::cout << "\n Time Elapsed: " << elapsed() << " sec.\n\n";
}

std::string Timer::DateAndTime() const
{
    time_t current;
    std::time(&current);

#if defined(WIN32) || defined(_WIN32)
    struct tm local;

    char str_now[32];

    errno_t err_t = localtime_s(&local, &current);
    err_t = asctime_s(str_now, 32, &local);
    return str_now;
#else
    struct tm *local;
    local = std::localtime(&current);

    return asctime(local);
#endif
}
