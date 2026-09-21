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
#include "stage_timer.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

PHON::PHON(MPI_Comm comm)
{
    run_info.comm = comm;
    MPI_Comm_rank(comm, &run_info.my_rank);
    MPI_Comm_size(comm, &run_info.nprocs);

    create_pointers();
}

PHON::~PHON()
{
    destroy_pointers();
}

void PHON::create_pointers()
{
    // Providers first: an object is constructed after everything it depends on
    // (see POINTERS_REMOVAL_PLAN.md). Writes is application shell: it keeps only
    // the PHON pointer, so it can come first and be handed to the classes that emit results.
    timer = std::make_unique<Timer>();
    writes = std::make_unique<Writes>(this);
    system = std::make_unique<System>(run_info);
    symmetry = std::make_unique<Symmetry>(run_info, system.get());
    kpoint = std::make_unique<Kpoint>(run_info, system.get());
    fcs_phonon = std::make_unique<Fcs_phonon>(run_info, system.get());
    dielec = std::make_unique<Dielec>(run_info, system.get());
    ewald = std::make_unique<Ewald>(run_info, system.get());
    dynamical = std::make_unique<Dynamical>(run_info, system.get(), dielec.get(), ewald.get());
    integration = std::make_unique<Integration>();
    thermodynamics = std::make_unique<Thermodynamics>();
    dos = std::make_unique<Dos>(run_info, system.get());
    phonon_velocity = std::make_unique<PhononVelocity>(run_info, system.get());
    anharmonic_core = std::make_unique<AnharmonicCore>(run_info,
                                                       timer.get(),
                                                       system.get(),
                                                       symmetry.get(),
                                                       fcs_phonon.get(),
                                                       integration.get(),
                                                       thermodynamics.get(),
                                                       dos.get());
    selfenergy = std::make_unique<Selfenergy>();
    isotope = std::make_unique<Isotope>();
    mode_symmetry =
        std::make_unique<ModeSymmetry>(run_info, system.get(), symmetry.get(), dielec.get(), dynamical.get());
    gruneisen = std::make_unique<Gruneisen>(run_info, system.get());
    relaxation = std::make_unique<Relaxation>(run_info, system.get());
    conductivity = std::make_unique<Conductivity>(run_info,
                                                  system.get(),
                                                  symmetry.get(),
                                                  fcs_phonon.get(),
                                                  dielec.get(),
                                                  ewald.get(),
                                                  dynamical.get(),
                                                  integration.get(),
                                                  thermodynamics.get(),
                                                  dos.get(),
                                                  phonon_velocity.get(),
                                                  anharmonic_core.get(),
                                                  isotope.get());
    iterativebte = std::make_unique<Iterativebte>(run_info,
                                                  system.get(),
                                                  symmetry.get(),
                                                  fcs_phonon.get(),
                                                  ewald.get(),
                                                  dynamical.get(),
                                                  integration.get(),
                                                  thermodynamics.get(),
                                                  dos.get(),
                                                  phonon_velocity.get(),
                                                  anharmonic_core.get(),
                                                  isotope.get(),
                                                  writes.get(),
                                                  conductivity.get());
    mode_analysis = std::make_unique<ModeAnalysis>(run_info,
                                                   system.get(),
                                                   symmetry.get(),
                                                   kpoint.get(),
                                                   fcs_phonon.get(),
                                                   ewald.get(),
                                                   dynamical.get(),
                                                   integration.get(),
                                                   thermodynamics.get(),
                                                   dos.get(),
                                                   anharmonic_core.get(),
                                                   selfenergy.get());
    const ScphQhaCollaborators scph_qha_collaborators{run_info,
                                                      timer.get(),
                                                      writes.get(),
                                                      system.get(),
                                                      symmetry.get(),
                                                      kpoint.get(),
                                                      fcs_phonon.get(),
                                                      ewald.get(),
                                                      dielec.get(),
                                                      dynamical.get(),
                                                      integration.get(),
                                                      thermodynamics.get(),
                                                      dos.get(),
                                                      anharmonic_core.get(),
                                                      selfenergy.get(),
                                                      relaxation.get()};
    scph = std::make_unique<Scph>(scph_qha_collaborators);
    qha = std::make_unique<Qha>(scph_qha_collaborators);
}

void PHON::destroy_pointers()
{
    // Reverse of create_pointers(): an object never outlives the collaborators it observes.
    qha.reset();
    scph.reset();
    mode_analysis.reset();
    iterativebte.reset();
    conductivity.reset();
    relaxation.reset();
    gruneisen.reset();
    mode_symmetry.reset();
    isotope.reset();
    selfenergy.reset();
    anharmonic_core.reset();
    phonon_velocity.reset();
    dos.reset();
    thermodynamics.reset();
    integration.reset();
    dynamical.reset();
    ewald.reset();
    dielec.reset();
    fcs_phonon.reset();
    kpoint.reset();
    symmetry.reset();
    system.reset();
    writes.reset();
    timer.reset();
}

void PHON::set_verbosity(const unsigned int verbosity_in)
{
    // Clamp to the supported range [0, 2] so an out-of-range value (e.g. a
    // negative Python int wrapping to a huge unsigned via a future nanobind
    // binding) can never reach the output guards.
    run_info.verbosity = verbosity_in > 2 ? 2 : verbosity_in;
}

unsigned int PHON::get_verbosity() const
{
    return run_info.verbosity;
}

void PHON::run() const
{
    if (run_info.mode == "PHONONS") {

        execute_phonons();

    } else if (run_info.mode == "KAPPA") {

        execute_kappa();

    } else if (run_info.mode == "SCPH" || run_info.mode == "QHA") {

        execute_self_consistent_phonon();

    } else {
        exit("run", "invalid mode: ", run_info.mode.c_str());
    }
}

void PHON::setup_base() const
{
    system->setup({fcs_phonon->file_fcs, fcs_phonon->file_fc2, fcs_phonon->file_fc3, fcs_phonon->file_fc4},
                  relaxation->init_u_tensor,
                  relaxation->init_u0);

    // Analysis flags that decide the required IFC orders (set on rank 0 by the
    // parser). print_newfcs must be synchronized here: Gruneisen::setup()
    // broadcasts it only after the IFCs are loaded.
    MPI_Bcast(&anharmonic_core->quartic_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gruneisen->gruneisen_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gruneisen->print_newfcs, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&thermodynamics->calc_FE_bubble, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    // RELAX_STR is set on rank 0 by the parser but read below on all ranks
    // (relaxing_structure); Relaxation::setup_relaxation() broadcasts it too late.
    MPI_Bcast(&relaxation->relax_str, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const auto setup_fcs = [this]() {
        fcs_phonon->setup(run_info.mode,
                          anharmonic_core->quartic_mode,
                          gruneisen->gruneisen_mode > 0 || thermodynamics->calc_FE_bubble,
                          gruneisen->print_newfcs);
    };

    const auto relaxing_structure = (run_info.mode == "SCPH" || run_info.mode == "QHA") && relaxation->relax_str != 0;

    // &displace DISPMODE = 2 resolves the initial displacements from the harmonic IFCs and
    // the symmetry of the reference cell, so both are prepared before the distorted cell
    // (and its symmetry, which drives the k-point reduction) is set up.
    int init_u0_from_modes = relaxation->init_disp_modes.empty() ? 0 : 1; // set on rank 0 by the parser
    MPI_Bcast(&init_u0_from_modes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (init_u0_from_modes) {
        setup_fcs();
        symmetry->setup_symmetry(relaxing_structure, false); // reference-cell operations only (init_u0 is still empty)
        relaxation->set_init_u0_from_modes(fcs_phonon->force_constant_with_cell[0], symmetry->SymmListWithMap_ref);
        system->initialize_distorted_primitive_cell(relaxation->init_u_tensor, relaxation->init_u0);
    }

    symmetry->setup_symmetry(relaxing_structure);
    kpoint->kpoint_setups(run_info.mode);
    if (kpoint->kpoint_mode == 2) {
        dos->create_kmesh_dos(kpoint->nk_mesh,
                              symmetry->SymmList,
                              system->get_primcell().reciprocal_lattice_vector,
                              symmetry->use_time_reversal && symmetry->time_reversal_sym);
        kpoint->print_uniform_mesh_info(*dos->kmesh_dos);
    }
    // Broadcasts the IRREPS flag; must precede dielec->init(), which uses it
    // to decide whether Born charges are loaded.
    mode_symmetry->setup();
    dynamical->setup_dynamical(kpoint->kpoint_bs.get(), kpoint->kpoint_general.get());
    if (!init_u0_from_modes) setup_fcs();
    phonon_velocity->setup_velocity();
    integration->setup_integration(anharmonic_core->quartic_mode, run_info.my_rank, get_verbosity());
    // ismear is broadcast inside setup_integration, so the adaptive smearing table can
    // only be built here, on every rank, as prepare_adaptivesmearing used to do.
    if (integration->ismear == 2) {
        NDArray<double, 3> vel_adaptive;
        vel_adaptive.resize(dos->kmesh_dos->nk, dynamical->neval, 3);
        phonon_velocity->get_phonon_group_velocity_mesh(*dos->kmesh_dos.get(),
                                                        system->get_primcell().lattice_vector,
                                                        *dynamical,
                                                        fcs_phonon->force_constant_with_cell[0],
                                                        *ewald,
                                                        vel_adaptive);
        integration->create_adaptive_sigma(dos->kmesh_dos.get(),
                                           system->get_primcell().reciprocal_lattice_vector,
                                           dynamical->neval,
                                           std::move(vel_adaptive));
    }
    dos->setup(*integration, dynamical->require_eigenvectors);
    thermodynamics->setup();
    anharmonic_core->setup();
    dielec->init(dos->emin,
                 dos->emax,
                 dos->delta_e,
                 dynamical->nonanalytic,
                 writes->print_zmode,
                 mode_symmetry->print_irreps,
                 symmetry->SymmListWithMap);
    ewald->init(*dielec, fcs_phonon->force_constant_with_cell[0]);

    if (run_info.my_rank == 0 && get_verbosity() > 0) {
        std::cout << " \n -----------------------------------------------------------------\n\n";
        if (thermodynamics->classical) {
            std::cout << "\n CLASSICAL = 1: Classical approximations will be used\n";
            std::cout << "                for all thermodynamic functions.\n\n";
        }
    }
}

void PHON::execute_phonons() const
{
    if (run_info.my_rank == 0 && get_verbosity() > 0) {
        std::cout << "                      MODE = phonons                         \n";
        std::cout << "                                                             \n";
        std::cout << "      Phonon calculation within harmonic approximation       \n";
        std::cout << "      Harmonic force constants will be used.                 \n";

        if (gruneisen->gruneisen_mode > 0) {
            std::cout << "\n      GRUNEISEN = " << gruneisen->gruneisen_mode
                      << " : Cubic force constants are necessary.\n";
        }
        std::cout << '\n';
    }

    setup_base();

    dynamical->diagonalize_dynamical_all(kpoint->kpoint_bs.get(),
                                         kpoint->kpoint_general.get(),
                                         dos->kmesh_dos.get(),
                                         dos->dymat_dos.get(),
                                         fcs_phonon->force_constant_with_cell[0],
                                         *ewald);

    if (mode_symmetry->print_irreps && run_info.my_rank == 0) {
        mode_symmetry->analyze_irreps_at_gamma(fcs_phonon->force_constant_with_cell[0], *ewald);
    }

    if (dos->flag_dos) {
        dos->calc_dos_all(*integration, thermodynamics->classical);
    }

    gruneisen->setup(anharmonic_core->quartic_mode, fcs_phonon->force_constant_with_cell);
    if (gruneisen->gruneisen_mode > 0) {
        gruneisen->calc_gruneisen(kpoint->kpoint_bs.get(),
                                  dynamical->dymat_band.get(),
                                  dos->kmesh_dos.get(),
                                  dos->dymat_dos.get());
    }
    if (dielec->calc_dielectric_constant) {
        dielec->run_dielec_calculation(*dynamical, fcs_phonon->force_constant_with_cell[0], *ewald);
    }

    if (thermodynamics->calc_FE_bubble) {
        thermodynamics->compute_free_energy_bubble(*system,
                                                   *dos->kmesh_dos.get(),
                                                   *dos->dymat_dos.get(),
                                                   symmetry->SymmList,
                                                   *anharmonic_core,
                                                   dynamical->neval,
                                                   run_info.my_rank,
                                                   run_info.nprocs,
                                                   get_verbosity());
    }

    if (run_info.my_rank == 0) {
        writes->printPhononEnergies();
        if (mode_symmetry->print_irreps) {
            writes->printModeIrrepsSummary();
        }
        writes->writePhononInfo();
        if (gruneisen->print_newfcs) {
            gruneisen->write_new_fcsxml_all(*writes, fcs_phonon->update_fc2, !fcs_phonon->file_fc3.empty());
        }
    }
}

void PHON::execute_kappa() const
{
    if (run_info.my_rank == 0 && get_verbosity() > 0) {
        std::cout << "                        MODE = RTA                           \n";
        std::cout << "                                                             \n";
        std::cout << "      Calculation of phonon line width (lifetime) and        \n";
        std::cout << "      lattice thermal conductivity within the RTA            \n";
        std::cout << "      (relaxation time approximation).                       \n";
        std::cout << "      Harmonic and anharmonic force constants will be used.  \n\n";
    }

    setup_base();

    if (kpoint->kpoint_mode < 3) {
        dynamical->diagonalize_dynamical_all(kpoint->kpoint_bs.get(),
                                             kpoint->kpoint_general.get(),
                                             dos->kmesh_dos.get(),
                                             dos->dymat_dos.get(),
                                             fcs_phonon->force_constant_with_cell[0],
                                             *ewald);
    }

    isotope->setup_isotope_scattering(*system,
                                      dos->kmesh_dos->nk_irred,
                                      dynamical->neval,
                                      run_info.my_rank,
                                      get_verbosity());
    isotope->calc_isotope_selfenergy_all(*dos->kmesh_dos.get(),
                                         *dos->dymat_dos.get(),
                                         *dos->tetra_nodes_dos.get(),
                                         *system,
                                         *integration,
                                         dynamical->neval,
                                         run_info.my_rank,
                                         run_info.nprocs,
                                         get_verbosity());

    mode_analysis->setup_mode_analysis();
    selfenergy->setup_selfenergy(dynamical->neval,
                                 integration->epsilon,
                                 thermodynamics->classical,
                                 run_info.my_rank,
                                 run_info.nprocs);

    if (mode_analysis->ks_analyze_mode) {
        mode_analysis->run_mode_analysis();
    } else {
        conductivity->run_kappa(); // broadcasts the solver flags; the RTA solver runs here
        if (conductivity->solver_ibte) {
            iterativebte->setup_iterative();
            iterativebte->do_iterativebte();
        } else {
            writes->writeKappa();
            writes->writeSelfenergyIsotope();
        }
    }
}

void PHON::execute_self_consistent_phonon() const
{
    if (run_info.my_rank == 0 && get_verbosity() > 0) {
        if (run_info.mode == "SCPH" && relaxation->relax_str == 0) {
            std::cout << "                        MODE = SCPH                          \n";
            std::cout << "                                                             \n";
            std::cout << "      Self-consistent phonon calculation to estimate         \n";
            std::cout << "      anharmonic phonon frequencies.                         \n";
            std::cout << "      Harmonic and quartic force constants will be used.     \n\n";
        } else if (run_info.mode == "SCPH" && relaxation->relax_str != 0) {
            std::cout << "                        MODE = SCPH                          \n";
            std::cout << "                                                             \n";
            std::cout << "      Self-consistent phonon calculation to compute          \n";
            std::cout << "      anharmonic phonon frequencies and crystal structure    \n";
            std::cout << "      at finite temperatures.                                \n";
            std::cout << "      Harmonic to quartic force constants will be used.      \n\n";
        } else if (run_info.mode == "QHA") {
            std::cout << "                        MODE = QHA                           \n";
            std::cout << "                                                             \n";
            std::cout << "      QHA calculation to compute crystal structure           \n";
            std::cout << "      at finite temperatures.                                \n";
            std::cout << "      Harmonic to quartic force constants will be used.      \n\n";
        }
    }

    auto t_stage = timer->elapsed();
    setup_base();
    print_stage_line("setup (IFCs, symmetry, k points, ...)",
                     timer->elapsed() - t_stage,
                     run_info.my_rank,
                     get_verbosity());

    t_stage = timer->elapsed();
    dynamical->diagonalize_dynamical_all(kpoint->kpoint_bs.get(),
                                         kpoint->kpoint_general.get(),
                                         dos->kmesh_dos.get(),
                                         dos->dymat_dos.get(),
                                         fcs_phonon->force_constant_with_cell[0],
                                         *ewald);
    print_stage_line("harmonic diagonalization, all k", timer->elapsed() - t_stage, run_info.my_rank, get_verbosity());
    relaxation->setup_relaxation(symmetry->tolerance);

    if (run_info.mode == "SCPH") {
        scph->setup_scph();
        scph->exec_scph();
    } else if (run_info.mode == "QHA") {
        qha->setup_qha();
        qha->exec_qha_optimization();
    }
}
