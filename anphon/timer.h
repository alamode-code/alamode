/*
 timer.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <string>
#include "pointers.h"

#if defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#else

#include <sys/time.h>
#include <time.h>

#endif

namespace PHON_NS
{
class Timer: protected Pointers
{
public:
    Timer(class PHON *);

    ~Timer();

    void reset();

    double elapsed() const;

    void print_elapsed() const;

    std::string DateAndTime() const;

private:
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER time_ref;
    LARGE_INTEGER frequency;
#else
    timeval time_ref;
#endif
};
} // namespace PHON_NS
