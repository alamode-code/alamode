/*
 phonon_cui.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#ifdef _WIN32
#include <mpi.h>
#else

#include "mpi.h"

#endif

namespace PHON_NS
{
class PhononCUI
{
public:
    PhononCUI();

    ~PhononCUI();

    void run(int narg, char **arg, MPI_Comm comm) const;
};
} // namespace PHON_NS
