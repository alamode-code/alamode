/*
 alm_cui.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "alm_cui.h"
#include "alm.h"
#include "input_parser.h"
#include "version.h"
#include <iostream>
#include <iomanip>
#include <memory>

#ifdef _OPENMP

#include <omp.h>

#endif

using namespace ALM_NS;

ALMCUI::ALMCUI() {}

ALMCUI::~ALMCUI() {}

void ALMCUI::run(const int narg,
                 char **arg) const
{
    auto alm = new ALM();

    // alm->mode is set herein.
    auto input_parser = std::make_unique<InputParser>();
    input_parser->run(alm, narg, arg);
    auto run_mode = input_parser->get_run_mode();

    if (alm->get_verbosity() > 0) {
        std::cout << " +-----------------------------------------------------------------+\n";
        std::cout << " +                         Program ALM                             +\n";
        std::cout << " +                             Ver.";
        std::cout << std::setw(7) << ALAMODE_VERSION;
        std::cout << "                         +\n";
        std::cout << " +-----------------------------------------------------------------+\n\n";
#ifdef _OPENMP
        std::cout << " Number of OpenMP threads = "
            << omp_get_max_threads() << "\n\n";
#endif

        std::cout << " Job started at " << alm->timer->DateAndTime() << '\n';
    }

    if (alm->get_verbosity() > 0) {
        alm->writer->write_input_vars(alm->system,
                                      alm->symmetry,
                                      alm->cluster,
                                      alm->displace,
                                      alm->fcs,
                                      alm->constraint,
                                      alm->optimize,
                                      alm->files,
                                      run_mode);
    }

    alm->init_fc_table();

    if (run_mode == "optimize") {
        alm->run_optimize();
        if (alm->get_optimizer_control().linear_model == 1 ||
            (alm->get_optimizer_control().linear_model >= 2
             && alm->get_optimizer_control().cross_validation == 0)) {
            alm->writer->writeall(alm->system,
                                  alm->symmetry,
                                  alm->cluster,
                                  alm->constraint,
                                  alm->fcs,
                                  alm->optimize,
                                  alm->files,
                                  alm->get_verbosity());
        }
    } else if (run_mode == "suggest") {
        alm->run_suggest();
        alm->writer->write_displacement_pattern(alm->system,
                                                alm->cluster,
                                                alm->displace,
                                                alm->files->get_prefix(),
                                                alm->get_verbosity());
    }

    if (alm->get_verbosity() > 0) {
        std::cout << '\n' << " Job finished at "
            << alm->timer->DateAndTime() << '\n';
    }

    delete alm;
}
