/*
 timer.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <string>
#include <map>

#if defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#else

#include <time.h>
#include <sys/time.h>

#endif

namespace ALM_NS
{
class Timer
{
public:
    Timer();

    ~Timer();

    void print_elapsed() const;

    void start_clock(std::string);

    void stop_clock(std::string);

    [[nodiscard]] double get_walltime(std::string);

    [[nodiscard]] double get_cputime(std::string);

    [[nodiscard]] static std::string DateAndTime();

private:
    void reset();

    [[nodiscard]] double elapsed_walltime() const;

    [[nodiscard]] double elapsed_cputime() const;

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
}
