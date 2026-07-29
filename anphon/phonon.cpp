/*
 phonon.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon.h"
#include <iostream>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "gruneisen.h"
#include "integration.h"
#include "isotope.h"
#include "iterativebte.h"
#include "kpoint.h"
#include "mode_analysis.h"
#include "mode_symmetry.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "phonon_velocity.h"
#include "qha.h"
#include "relaxation.h"
#include "scph.h"
#include "selfenergy.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

PHON::PHON(MPI_Comm comm)
{
    mympi = std::make_unique<MyMPI>(this, comm);

    create_pointers();
}

PHON::~PHON()
{
    destroy_pointers();
    mympi.reset();
}

void PHON::create_pointers()
{
    timer = std::make_unique<Timer>(this);
    system = std::make_unique<System>(this);
    symmetry = std::make_unique<Symmetry>(this);
    kpoint = std::make_unique<Kpoint>(this);
    fcs_phonon = std::make_unique<Fcs_phonon>(this);
    dynamical = std::make_unique<Dynamical>(this);
    integration = std::make_unique<Integration>();
    phonon_velocity = std::make_unique<PhononVelocity>(this);
    thermodynamics = std::make_unique<Thermodynamics>();
    anharmonic_core = std::make_unique<AnharmonicCore>(this);
    mode_analysis = std::make_unique<ModeAnalysis>(this);
    mode_symmetry = std::make_unique<ModeSymmetry>(this);
    selfenergy = std::make_unique<Selfenergy>();
    conductivity = std::make_unique<Conductivity>(this);
    writes = std::make_unique<Writes>(this);
    dos = std::make_unique<Dos>(this);
    gruneisen = std::make_unique<Gruneisen>(this);
    isotope = std::make_unique<Isotope>();
    scph = std::make_unique<Scph>(this);
    ewald = std::make_unique<Ewald>(this);
    dielec = std::make_unique<Dielec>(this);
    qha = std::make_unique<Qha>(this);
    iterativebte = std::make_unique<Iterativebte>(this);
    relaxation = std::make_unique<Relaxation>(this);
}

void PHON::destroy_pointers()
{
    timer.reset();
    system.reset();
    symmetry.reset();
    kpoint.reset();
    fcs_phonon.reset();
    dynamical.reset();
    integration.reset();
    phonon_velocity.reset();
    thermodynamics.reset();
    anharmonic_core.reset();
    mode_analysis.reset();
    mode_symmetry.reset();
    selfenergy.reset();
    conductivity.reset();
    writes.reset();
    dos.reset();
    gruneisen.reset();
    isotope.reset();
    scph.reset();
    ewald.reset();
    dielec.reset();
    iterativebte.reset();
    qha.reset();
    relaxation.reset();
}

void PHON::set_verbosity(const unsigned int verbosity_in)
{
    // Clamp to the supported range [0, 2] so an out-of-range value (e.g. a
    // negative Python int wrapping to a huge unsigned via a future nanobind
    // binding) can never reach the output guards.
    verbosity = verbosity_in > 2 ? 2 : verbosity_in;
}

unsigned int PHON::get_verbosity() const
{
    return verbosity;
}

void PHON::run() const
{
    if (mode == "PHONONS") {

        execute_phonons();

    } else if (mode == "KAPPA") {

        execute_kappa();

    } else if (mode == "SCPH" || mode == "QHA") {

        execute_self_consistent_phonon();

    } else {
        exit("run", "invalid mode: ", mode.c_str());
    }
}

void PHON::setup_base() const
{
    system->setup();
    symmetry->setup_symmetry();
    kpoint->kpoint_setups(mode);
    // Broadcasts the IRREPS flag; must precede dielec->init(), which uses it
    // to decide whether Born charges are loaded.
    mode_symmetry->setup();
    dynamical->setup_dynamical();
    fcs_phonon->setup(mode);
    phonon_velocity->setup_velocity();
    integration->setup_integration(dos->kmesh_dos.get(),
                                   phonon_velocity.get(),
                                   dynamical->neval,
                                   system->get_primcell().lattice_vector,
                                   system->get_primcell().reciprocal_lattice_vector,
                                   anharmonic_core->quartic_mode,
                                   mympi->my_rank,
                                   get_verbosity());
    dos->setup();
    thermodynamics->setup();
    anharmonic_core->setup();
    dielec->init();
    ewald->init();

    if (mympi->my_rank == 0 && get_verbosity() > 0) {
        std::cout << " \n -----------------------------------------------------------------\n\n";
        if (thermodynamics->classical) {
            std::cout << "\n CLASSICAL = 1: Classical approximations will be used\n";
            std::cout << "                for all thermodynamic functions.\n\n";
        }
    }
}

void PHON::execute_phonons() const
{
    if (mympi->my_rank == 0 && get_verbosity() > 0) {
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

    if (mode_symmetry->print_irreps && mympi->my_rank == 0) {
        mode_symmetry->analyze_irreps_at_gamma();
    }

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
        thermodynamics->compute_free_energy_bubble(*system,
                                                   *dos->kmesh_dos.get(),
                                                   *dos->dymat_dos.get(),
                                                   symmetry->SymmList,
                                                   *anharmonic_core,
                                                   dynamical->neval,
                                                   mympi->my_rank,
                                                   mympi->nprocs,
                                                   get_verbosity());
    }

    if (mympi->my_rank == 0) {
        writes->printPhononEnergies();
        if (mode_symmetry->print_irreps) {
            writes->printModeIrrepsSummary();
        }
        writes->writePhononInfo();
        if (gruneisen->print_newfcs) {
            gruneisen->write_new_fcsxml_all();
        }
    }
}

void PHON::execute_kappa() const
{
    if (mympi->my_rank == 0 && get_verbosity() > 0) {
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

    isotope->setup_isotope_scattering(*system,
                                      dos->kmesh_dos->nk_irred,
                                      dynamical->neval,
                                      mympi->my_rank,
                                      get_verbosity());
    isotope->calc_isotope_selfenergy_all(*dos->kmesh_dos.get(),
                                         *dos->dymat_dos.get(),
                                         *dos->tetra_nodes_dos.get(),
                                         *system,
                                         *integration,
                                         dynamical->neval,
                                         mympi->my_rank,
                                         mympi->nprocs,
                                         get_verbosity());

    mode_analysis->setup_mode_analysis();
    selfenergy->setup_selfenergy(dynamical->neval,
                                 integration->epsilon,
                                 thermodynamics->classical,
                                 symmetry->SymmList,
                                 *anharmonic_core,
                                 mympi->my_rank,
                                 mympi->nprocs);

    if (mode_analysis->ks_analyze_mode) {
        mode_analysis->run_mode_analysis();
    } else {
        conductivity->run_kappa();
    }
}

void PHON::execute_self_consistent_phonon() const
{
    if (mympi->my_rank == 0 && get_verbosity() > 0) {
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
