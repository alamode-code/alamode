//
// Created by Terumasa Tadano on 25/06/28.
//

#pragma once

#include <iostream>
#include <utility>

namespace util
{

enum class LogStream
{
    Cout,
    Cerr
};

template <typename... Args>
inline void log_if(int verbosity, int level, LogStream stream, const char *func, Args &&...args)
{
    if (verbosity > level) {
        auto &os = (stream == LogStream::Cerr ? std::cerr : std::cout);
        os << "   [" << func << "] ";
        (os << ... << std::forward<Args>(args));
    }
}
} // namespace util

#define LOG_IF(verbosity, level, ...) util::log_if(verbosity, level, util::LogStream::Cout, __func__, __VA_ARGS__)
#define LOG_ERR_IF(verbosity, level, ...) util::log_if(verbosity, level, util::LogStream::Cerr, __func__, __VA_ARGS__)
