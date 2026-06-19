/*
 error.h

 Copyright (c) 2014-2018 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <cstdlib>
#include <iostream>


namespace ALM_NS
{
inline auto warn(const char *file, const char *message) -> void
{
    std::cout << '\n' << " WARNING in " << file << "  MESSAGE: " << message << '\n';
}

[[noreturn]] inline auto exit(const char *file, const char *message) -> void
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

template <typename T>
[[noreturn]] auto exit(const char *file, const char *message, const T info) -> void
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << info << '\n';
    std::exit(EXIT_FAILURE);
}

[[noreturn]] inline auto exit(const char *file, const char *message, const char *info) -> void
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << info << '\n';
    std::exit(EXIT_FAILURE);
}
} // namespace ALM_NS
