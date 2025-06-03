/*
 error.h

 Copyright (c) 2014-2018 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <iostream>
#include <cstdlib>


namespace ALM_NS
{
inline void warn(const char *file,
                 const char *message)
{
    std::cout << '\n' << " WARNING in " << file << "  MESSAGE: " << message << '\n';
}

inline void exit(const char *file,
                 const char *message)
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

template <typename T>
void exit(const char *file,
          const char *message,
          const T info)
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << info << '\n';
    std::exit(EXIT_FAILURE);
}

inline void exit(const char *file,
                 const char *message,
                 const char *info)
{
    std::cout << '\n' << " ERROR in " << file << "  MESSAGE: " << message << info << '\n';
    std::exit(EXIT_FAILURE);
}
}
