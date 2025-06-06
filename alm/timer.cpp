/*
 timer.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "timer.h"
#include <ctime>
#include <iostream>
#include <string>

using namespace ALM_NS;

Timer::Timer()
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&walltime_ref);
    QueryPerformanceFrequency(&frequency);
    cputime_ref = get_cputime();
#else
    gettimeofday(&walltime_ref, nullptr);
    cputime_ref = static_cast<double>(clock());
#endif
    lock = false;
}

Timer::~Timer()
{
    walltime.clear();
    cputime.clear();
}

void Timer::reset()
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&walltime_ref);
#else
    gettimeofday(&walltime_ref, nullptr);
#endif
}

auto Timer::elapsed_walltime() const -> double
{
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER time_now;
    QueryPerformanceCounter(&time_now);
    return static_cast<double>(time_now.QuadPart - walltime_ref.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
    timeval time_now;
    gettimeofday(&time_now, nullptr);
    return (time_now.tv_sec - walltime_ref.tv_sec) + (time_now.tv_usec - walltime_ref.tv_usec) * 1.0e-6;
#endif
}

#if defined(WIN32) || defined(_WIN32)
double Timer::get_cputime() const
{
    FILETIME createTime;
    FILETIME exitTime;
    FILETIME kernelTime;
    FILETIME userTime;
    if (GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime) != -1) {
        SYSTEMTIME userSystemTime;
        if (FileTimeToSystemTime(&userTime, &userSystemTime) != -1)
            return static_cast<double>(userSystemTime.wHour) * 3600.0 +
                   static_cast<double>(userSystemTime.wMinute) * 60.0 + static_cast<double>(userSystemTime.wSecond) +
                   static_cast<double>(userSystemTime.wMilliseconds) / 1000.0;
    }
    return 0.0;
}
#endif

auto Timer::elapsed_cputime() const -> double
{
#if defined(WIN32) || defined(_WIN32)
    return get_cputime() - cputime_ref;
#else
    return (static_cast<double>(clock()) - cputime_ref) / CLOCKS_PER_SEC;
#endif
}

auto Timer::print_elapsed() const -> void
{
    std::cout << "  Time Elapsed: " << elapsed_walltime() << " sec.\n\n" << std::flush;
}


auto Timer::DateAndTime() -> std::string
{
    time_t current;
    std::time(&current);

#if defined(WIN32) || defined(_WIN32)
    struct tm local{};
    char str_now[32];

    auto err_t = localtime_s(&local, &current);
    err_t = asctime_s(str_now, 32, &local);
    return str_now;
#else
    struct tm *local;
    local = std::localtime(&current);

    return asctime(local);
#endif
}


auto Timer::start_clock(const std::string &str_tag) -> void
{
    if (lock) {
        std::cout << "Error: cannot start clock because it's occupied." << '\n';
        exit(1);
    }
    // Initialize the counter if the key is new
    if (walltime.find(str_tag) == walltime.end()) {
        walltime[str_tag] = 0.0;
    }
    if (cputime.find(str_tag) == cputime.end()) {
        cputime[str_tag] = 0.0;
    }

    wtime_tmp = elapsed_walltime();
    ctime_tmp = elapsed_cputime();

    lock = true;
}

auto Timer::stop_clock(const std::string &str_tag) -> void
{
    if (!lock) {
        std::cout << "Error: cannot stop clock because it's not initialized." << '\n';
        exit(1);
    }

    auto it = walltime.find(str_tag);

    if (it == walltime.end()) {
        std::cout << "Error: invalid tag for clock" << '\n';
        exit(1);
    }

    auto time_tmp = (*it).second;
    time_tmp += elapsed_walltime() - wtime_tmp;
    walltime[str_tag] = time_tmp;

    it = cputime.find(str_tag);
    if (it == cputime.end()) {
        std::cout << "Error: invalid tag for clock" << '\n';
        exit(1);
    }

    time_tmp = (*it).second;
    time_tmp += elapsed_cputime() - ctime_tmp;
    cputime[str_tag] = time_tmp;

    lock = false;
}

auto Timer::get_walltime(const std::string &str_tag) -> double
{
    const auto it = walltime.find(str_tag);

    if (it == walltime.end()) {
        std::cout << "Error: invalid tag for clock" << '\n';
        exit(1);
    }
    return (*it).second;
}


auto Timer::get_cputime(const std::string &str_tag) -> double
{
    const auto it = cputime.find(str_tag);

    if (it == cputime.end()) {
        std::cout << "Error: invalid tag for clock" << '\n';
        exit(1);
    }
    return (*it).second;
}
