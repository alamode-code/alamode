/*
 main.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include <cstdlib>
#include "alm_cui.h"

using namespace ALM_NS;

auto main(const int argc, char **argv) -> int
{
    const auto alm_cui = new ALMCUI();

    alm_cui->run(argc, argv);

    delete alm_cui;

    return EXIT_SUCCESS;
}
