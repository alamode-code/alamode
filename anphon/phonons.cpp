/*
 phonons.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include <iostream>
#include <iomanip>
#include "phonons.h"
#include "timer.h"
#include "parsephon.h"
#include "error.h"
#include "gruneisen.h"
#include "system.h"
#include "symmetry_core.h"
#include "kpoint.h"
#include "fcs_phonon.h"
#include "dynamical.h"
#include "phonon_velocity.h"
#include "thermodynamics.h"
#include "write_phonons.h"
#include "phonon_dos.h"
#include "integration.h"
#include "anharmonic_core.h"
#include "mode_analysis.h"
#include "conductivity.h"
#include "iterativebte.h"
#include "isotope.h"
#include "selfenergy.h"
#include "version.h"
#include "scph.h"
#include "ewald.h"
#include "dielec.h"
#include "qha.h"
#include "relaxation.h"

#ifdef _OPENMP

#include <omp.h>

#endif

using namespace PHON_NS;

PHON::PHON(int narg,
           char **arg,
           MPI_Comm comm)
{
    mympi = new MyMPI(this, comm);
    input = new Input(this);

    create_pointers();

    if (mympi->my_rank == 0) {
        std::cout << " +-----------------------------------------------------------------+\n";
        std::cout << " +                         Program ANPHON                          +\n";
        std::cout << " +                             Ver.";
        std::cout << std::setw(7) << ALAMODE_VERSION;
        std::cout << "                         +\n";
        std::cout << " +-----------------------------------------------------------------+\n\n";
        std::cout << " Job started at " << timer->DateAndTime() << '\n';
        std::cout << " The number of MPI processes: " << mympi->nprocs << '\n';
#ifdef _OPENMP
        std::cout << " The number of OpenMP threads: "
            << omp_get_max_threads() << '\n';
#endif
        std::cout << '\n';

        input->parce_input(narg, arg);
        writes->writeInputVars();
    }

    mympi->MPI_Bcast_string(input->job_title, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_string(mode, 0, MPI_COMM_WORLD);

    if (mode == "PHONONS") {

        execute_phonons();

    } else if (mode == "RTA") {

        execute_RTA();

    } else if (mode == "SCPH" || mode == "QHA") {

        execute_self_consistent_phonon();

    } else {
        exit("phonons", "invalid mode: ", mode.c_str());
    }

    if (mympi->my_rank == 0) {
        std::cout << "\n Job finished at "
            << timer->DateAndTime() << '\n';
    }
    destroy_pointers();
}

PHON::~PHON()
{
    delete input;
    delete mympi;
}

void PHON::create_pointers()
{
    timer = new Timer(this);
    system = new System(this);
    symmetry = new Symmetry(this);
    kpoint = new Kpoint(this);
    fcs_phonon = new Fcs_phonon(this);
    dynamical = new Dynamical(this);
    integration = new Integration(this);
    phonon_velocity = new PhononVelocity(this);
    thermodynamics = new Thermodynamics(this);
    anharmonic_core = new AnharmonicCore(this);
    mode_analysis = new ModeAnalysis(this);
    selfenergy = new Selfenergy(this);
    conductivity = new Conductivity(this);
    writes = new Writes(this);
    dos = new Dos(this);
    gruneisen = new Gruneisen(this);
    isotope = new Isotope(this);
    scph = new Scph(this);
    ewald = new Ewald(this);
    dielec = new Dielec(this);
    qha = new Qha(this);
    iterativebte = new Iterativebte(this);
    relaxation = new Relaxation(this);
}

void PHON::destroy_pointers() const
{
    delete timer;
    delete system;
    delete symmetry;
    delete kpoint;
    delete fcs_phonon;
    delete dynamical;
    delete integration;
    delete phonon_velocity;
    delete thermodynamics;
    delete anharmonic_core;
    delete mode_analysis;
    delete selfenergy;
    delete conductivity;
    delete writes;
    delete dos;
    delete gruneisen;
    delete isotope;
    delete scph;
    delete ewald;
    delete dielec;
    delete iterativebte;
    delete qha;
    delete relaxation;
}

void PHON::setup_base() const
{
    system->setup();
    symmetry->setup_symmetry();
    kpoint->kpoint_setups(mode);
    dynamical->setup_dynamical();
    fcs_phonon->setup(mode);
    phonon_velocity->setup_velocity();
    integration->setup_integration();
    dos->setup();
    thermodynamics->setup();
    anharmonic_core->setup();
    dielec->init();
    ewald->init();

    if (mympi->my_rank == 0) {
        std::cout << " \n -----------------------------------------------------------------\n\n";
        if (thermodynamics->classical) {
            std::cout << "\n CLASSICAL = 1: Classical approximations will be used\n";
            std::cout << "                for all thermodynamic functions.\n\n";
        }
    }
}

void PHON::execute_phonons() const
{
    if (mympi->my_rank == 0) {
        std::cout << "                      MODE = phonons                         \n";
        std::cout << "                                                             \n";
        std::cout << "      Phonon calculation within harmonic approximation       \n";
        std::cout << "      Harmonic force constants will be used.                 \n";

        if (gruneisen->print_gruneisen) {
            std::cout << "\n      GRUNEISEN = 1 : Cubic force constants are necessary.\n";
        }
        std::cout << '\n';
    }

    setup_base();

    dynamical->diagonalize_dynamical_all();

    if (dos->flag_dos) {
        dos->calc_dos_all();
    }

    gruneisen->setup();
    if (gruneisen->print_gruneisen) {
        gruneisen->calc_gruneisen();
    }
    if (dielec->calc_dielectric_constant) {
        dielec->run_dielec_calculation();
    }

    if (thermodynamics->calc_FE_bubble) {
        thermodynamics->compute_free_energy_bubble();
    }

    if (mympi->my_rank == 0) {
        writes->printPhononEnergies();
        writes->writePhononInfo();
        if (gruneisen->print_newfcs) {
            gruneisen->write_new_fcsxml_all();
        }
    }
}

void PHON::execute_RTA() const
{
    if (mympi->my_rank == 0) {
        std::cout << "                        MODE = RTA                           \n";
        std::cout << "                                                             \n";
        std::cout << "      Calculation of phonon line width (lifetime) and        \n";
        std::cout << "      lattice thermal conductivity within the RTA            \n";
        std::cout << "      (relaxation time approximation).                       \n";
        std::cout << "      Harmonic and anharmonic force constants will be used.  \n\n";
    }

    setup_base();

    if (kpoint->kpoint_mode < 3) {
        dynamical->diagonalize_dynamical_all();
    }

    isotope->setup_isotope_scattering();
    isotope->calc_isotope_selfenergy_all();

    mode_analysis->setup_mode_analysis();
    selfenergy->setup_selfenergy();

    MPI_Bcast(&iterativebte->do_iterative, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&conductivity->fph_rta, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mode_analysis->ks_analyze_mode) {

        mode_analysis->run_mode_analysis();

    } else if (iterativebte->do_iterative) {

        iterativebte->setup_iterative();
        iterativebte->do_iterativebte();

    } else {

        conductivity->setup_kappa();
        conductivity->calc_anharmonic_imagself();
        conductivity->compute_kappa();
        writes->writeKappa();
        writes->writeSelfenergyIsotope();

    }
}

void PHON::execute_self_consistent_phonon() const
{
    if (mympi->my_rank == 0) {
        if (mode == "SCPH" && relaxation->relax_str == 0) {
            std::cout << "                        MODE = SCPH                          \n";
            std::cout << "                                                             \n";
            std::cout << "      Self-consistent phonon calculation to estimate         \n";
            std::cout << "      anharmonic phonon frequencies.                         \n";
            std::cout << "      Harmonic and quartic force constants will be used.     \n\n";
        } else if (mode == "SCPH" && relaxation->relax_str != 0) {
            std::cout << "                        MODE = SCPH                          \n";
            std::cout << "                                                             \n";
            std::cout << "      Self-consistent phonon calculation to compute          \n";
            std::cout << "      anharmonic phonon frequencies and crystal structure    \n";
            std::cout << "      at finite temperatures.                                \n";
            std::cout << "      Harmonic to quartic force constants will be used.      \n\n";
        } else if (mode == "QHA") {
            std::cout << "                        MODE = QHA                           \n";
            std::cout << "                                                             \n";
            std::cout << "      QHA calculation to compute crystal structure           \n";
            std::cout << "      at finite temperatures.                                \n";
            std::cout << "      Harmonic to quartic force constants will be used.      \n\n";
        }
    }

    setup_base();

    dynamical->diagonalize_dynamical_all();
    relaxation->setup_relaxation();

    if (mode == "SCPH") {
        scph->setup_scph();
        scph->exec_scph();
    } else if (mode == "QHA") {
        qha->setup_qha();
        qha->exec_qha_optimization();
    }
}
