/*
 timer.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <map>
#include <string>

#if defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#else

#include <sys/time.h>
#include <time.h>

#endif

namespace ALM_NS
{
class Timer
{
public:
    Timer();

    ~Timer();

    auto print_elapsed() const -> void;

    auto start_clock(const std::string &) -> void;

    auto stop_clock(const std::string &) -> void;

    [[nodiscard]] auto get_walltime(const std::string &) -> double;

    [[nodiscard]] auto get_cputime(const std::string &) -> double;

    [[nodiscard]] static auto DateAndTime() -> std::string;

private:
    auto reset() -> void;

    [[nodiscard]] auto elapsed_walltime() const -> double;

    [[nodiscard]] auto elapsed_cputime() const -> double;

    std::map<std::string, double> walltime;
    std::map<std::string, double> cputime;
    double wtime_tmp, ctime_tmp;
    bool lock;

#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER walltime_ref;
    LARGE_INTEGER frequency;
    double get_cputime() const;
    double cputime_ref;
#else
    timeval walltime_ref;
    double cputime_ref;
#endif
};
} // namespace ALM_NS
