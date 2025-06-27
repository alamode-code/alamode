//
// Created by Terumasa Tadano on 25/06/28.
//

#pragma once

#include <iostream>
#include <utility>

namespace util
{
template <typename... Args>
inline void log_if(int verbosity, int level, const char* func, Args&&... args)
{
    if (verbosity > level) {
        std::cout << "   [" << func << "] ";
        (std::cout << ... << std::forward<Args>(args));
    }
}
} // namespace util

#define LOG_IF(verbosity, level, ...) util::log_if(verbosity, level, __func__, __VA_ARGS__)
