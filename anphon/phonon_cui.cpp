/*
 phonon_cui.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon_cui.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include "input_parser.h"
#include "mpi_common.h"
#include "phonon.h"
#include "timer.h"
#include "version.h"
#include "write_phonons.h"

#ifdef _OPENMP

#include <omp.h>

#endif

using namespace PHON_NS;

PhononCUI::PhononCUI()
{}

PhononCUI::~PhononCUI()
{}

void PhononCUI::run(const int narg, char **arg, MPI_Comm comm) const
{
    auto phon = std::make_unique<PHON>(comm);

    if (phon->mympi->my_rank == 0) {
        std::cout << " +-----------------------------------------------------------------+\n";
        std::cout << " +                         Program ANPHON                          +\n";
        std::cout << " +                             Ver.";
        std::cout << std::setw(7) << ALAMODE_VERSION;
        std::cout << "                         +\n";
        std::cout << " +-----------------------------------------------------------------+\n\n";
        std::cout << " Job started at " << phon->timer->DateAndTime() << '\n';
        std::cout << " The number of MPI processes: " << phon->mympi->nprocs << '\n';
#ifdef _OPENMP
        std::cout << " The number of OpenMP threads: " << omp_get_max_threads() << '\n';
#endif
        std::cout << '\n';

        auto input_parser = std::make_unique<InputParser>();
        input_parser->run(phon.get(), narg, arg);
        phon->writes->writeInputVars();
    }

    phon->mympi->MPI_Bcast_string(phon->job_title, 0, MPI_COMM_WORLD);
    phon->mympi->MPI_Bcast_string(phon->mode, 0, MPI_COMM_WORLD);

    // VERBOSITY is parsed on rank 0 only; broadcast it so every rank honors
    // the user's setting (Part B threads it into all-rank code paths).
    auto verbosity_tmp = phon->get_verbosity();
    MPI_Bcast(&verbosity_tmp, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    phon->set_verbosity(verbosity_tmp);

    phon->run();

    if (phon->mympi->my_rank == 0) {
        std::cout << "\n Job finished at " << phon->timer->DateAndTime() << '\n';
    }
}
