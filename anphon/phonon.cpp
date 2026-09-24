/*
 phonon.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "gruneisen.h"
#include "ifc_derivative.h"
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
#include "scph_result_io.h"
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
    dynamical = std::make_unique<Dynamical>(run_info, system.get());
    integration = std::make_unique<Integration>();
    thermodynamics = std::make_unique<Thermodynamics>();
    dos = std::make_unique<Dos>(run_info, system.get());
    phonon_velocity = std::make_unique<PhononVelocity>(run_info, system.get());
    anharmonic_core = std::make_unique<AnharmonicCore>(run_info, system.get());
    selfenergy = std::make_unique<Selfenergy>();
    isotope = std::make_unique<Isotope>();
    mode_symmetry = std::make_unique<ModeSymmetry>(run_info, system.get());
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
                                                  dielec.get(),
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
                                                   dielec.get(),
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

void PHON::apply_relaxed_structure() const
{
    // The state file is the one that also supplies the renormalized FC2,
    // in the order FC2_TEMPERATURE itself resolves: DFC2FILE carries the
    // correction when given, otherwise the FC2 comes from FC2FILE, or from
    // FCSFILE when no separate FC2 file was named.
    const auto &statefile = !fcs_phonon->file_dfc2.empty()  ? fcs_phonon->file_dfc2
                            : !fcs_phonon->file_fc2.empty() ? fcs_phonon->file_fc2
                                                            : fcs_phonon->file_fcs;

    std::vector<double> u_tensor(9, 0.0), u0;
    std::string spg_label;
    int natmin_file = 0;

    if (run_info.my_rank == 0) {
        const ScphResultIOH5 io(statefile);
        Eigen::Matrix3d lavec_file;
        Eigen::MatrixXd xf_file;
        if (!io.load_structure(fcs_phonon->fc2_temperature, u_tensor, u0, spg_label, lavec_file, xf_file)) {
            exit("apply_relaxed_structure",
                 "RELAXED_STRUCTURE = 1, but the state file carries no /structure group.\n"
                 " It must come from a run with RELAX_STR != 0; a fixed-cell SCPH run relaxes\n"
                 " nothing and stores no structure.");
        }
        io.check_convergence({fcs_phonon->fc2_temperature}, run_info.allow_unconverged);

        // u0 is indexed by the producer's primitive-cell atoms. The atom
        // order need not match this run's -- a state file whose FC2
        // correction is mapped positionally by read_delta_fc2_from_scph
        // may well be ordered differently -- so map by position and reorder,
        // rather than trusting the index. This also rejects a state file
        // recorded for a different cell, which would otherwise displace the
        // wrong atoms with no complaint.
        const auto &primcell = system->get_primcell();
        if (static_cast<size_t>(xf_file.rows()) != primcell.number_of_atoms) {
            exit("apply_relaxed_structure",
                 "The relaxed structure was recorded for a different number of primitive-cell atoms\n"
                 " than this run uses. The two runs must share the same primitive cell.");
        }
        if ((lavec_file - primcell.lattice_vector).cwiseAbs().maxCoeff() > 1.0e-4) {
            exit("apply_relaxed_structure",
                 "The reference cell of the relaxed structure differs from the cell of this run.\n"
                 " The deformation is relative to the producer's reference cell, so the two must match.");
        }

        std::vector<double> u0_mapped(u0.size(), 0.0);
        for (Eigen::Index i = 0; i < xf_file.rows(); ++i) {
            auto match = -1;
            for (size_t p = 0; p < primcell.number_of_atoms; ++p) {
                Eigen::Vector3d xdiff = xf_file.row(i).transpose() - primcell.x_fractional.row(p).transpose();
                xdiff = xdiff.unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });
                if ((primcell.lattice_vector * xdiff).norm() < 1.0e-3) {
                    match = static_cast<int>(p);
                    break;
                }
            }
            if (match < 0) {
                exit("apply_relaxed_structure",
                     "An atom of the relaxed structure has no counterpart in the primitive cell of this run.");
            }
            for (auto j = 0; j < 3; ++j) u0_mapped[3 * match + j] = u0[3 * i + j];
        }
        u0.swap(u0_mapped);
        natmin_file = static_cast<int>(u0.size() / 3);

        // The deformation was determined by the IFCs the relaxation loaded
        // -- above all by FC4, which drove the optimization that produced
        // u0 and u_tensor and which the Phi4 : d correction below is built
        // from. A different FC4 pairs a deformed FC3 with a structure it
        // does not belong to, and nothing downstream would notice.
        ScphProvenanceH5 current;
        current.fcs_nrows = fcs_phonon->fcs_nrows;
        current.fcs_sum_abs = fcs_phonon->fcs_sum_abs;
        current.fcs_sum_signed = fcs_phonon->fcs_sum_signed;
        current.fcs_sum_sq = fcs_phonon->fcs_sum_sq;
        for (const auto order: io.compare_provenance(current)) {
            const auto what = "order " + std::to_string(order + 2) +
                              " differs from the force constants the"
                              " relaxation used";
            if (order + 2 >= 4) {
                exit("apply_relaxed_structure",
                     (what + ".\n The relaxed structure and the quartic force constants belong together;"
                             " supply the\n same FC4 the SCPH/QHA run did, or rerun the relaxation.")
                         .c_str());
            }
            // A better FC3 is a legitimate thing to bring to a deformed run,
            // so this one only warns.
            warn("apply_relaxed_structure", (what + "; continuing.").c_str());
        }
    }

    MPI_Bcast(u_tensor.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&natmin_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (run_info.my_rank != 0) u0.assign(3 * natmin_file, 0.0);
    MPI_Bcast(u0.data(), 3 * natmin_file, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (run_info.my_rank == 0 && run_info.verbosity > 0) {
        std::cout << "\n RELAXED_STRUCTURE = 1: adopting the structure relaxed at " << fcs_phonon->fc2_temperature
                  << " K,\n  read from " << statefile;
        if (!spg_label.empty()) {
            std::cout << "\n  space group of that structure, as the relaxation found it: " << spg_label
                      << "\n  (TOLERANCE must match, or this run may find a different one)";
        }
        std::cout << '\n';
    }

    // FC3 of the deformed structure, Phi3 + Phi4 : d. This must run before
    // the cells move: the helper needs the reference lattice.
    if (fcs_phonon->maxorder >= 3) {
        std::vector<FcsArrayWithCell> fc3_deformed;
        DerivativeIFC::compute_deformed_cubic_ifcs(fcs_phonon->force_constant_with_cell,
                                                   u_tensor.data(),
                                                   u0,
                                                   system->get_primcell().lattice_vector,
                                                   fc3_deformed);
        const auto nfc3_before = fcs_phonon->force_constant_with_cell[1].size();
        const auto nfc3_after = fc3_deformed.size();
        fcs_phonon->force_constant_with_cell[1] = std::move(fc3_deformed);
        if (run_info.my_rank == 0 && run_info.verbosity > 0) {
            std::cout << "  Deformed FC3 (Phi3 + Phi4 : d): " << nfc3_before << " -> " << nfc3_after << " entries\n";
        }
    }
    // maxorder < 3 here means the run does not use the cubic IFCs either
    // (a band structure, say), so there is nothing to deform: Fcs_phonon
    // ::setup raises maxorder to 3 whenever they are required.

    const auto volume_ref = system->get_primcell().volume;
    system->apply_deformation(u_tensor, u0);
    fcs_phonon->deform_relative_vectors(u0);

    // The crystal-structure block was printed by System::setup(), before the
    // deformation, so report what the run actually uses.
    if (run_info.my_rank == 0 && run_info.verbosity > 0) {
        const auto &cell = system->get_primcell();
        std::cout << "\n  Lattice vectors of the relaxed cell [Bohr]:\n";
        for (auto j = 0; j < 3; ++j) {
            std::cout << "   a" << j + 1 << " :";
            for (auto i = 0; i < 3; ++i) {
                std::cout << std::setw(15) << std::setprecision(8) << std::fixed << cell.lattice_vector(i, j);
            }
            std::cout << '\n';
        }
        std::cout << std::defaultfloat << "  Volume : " << cell.volume << " (a.u.)^3, " << cell.volume / volume_ref
                  << " x the reference cell\n\n";
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
    // Also set on rank 0 only, and read inside Fcs_phonon::setup, where it
    // raises maxorder: every rank must agree before any setup_fcs() call,
    // or they would load a different number of IFC orders and diverge.
    MPI_Bcast(&fcs_phonon->relaxed_structure, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

    // RELAXED_STRUCTURE: move onto the relaxed crystal before anything reads
    // the geometry. The IFCs have to be loaded first, because the
    // reference-geometry relative vectors that replicate_force_constant
    // builds are what deform_relative_vectors then corrects; and this has to
    // precede setup_symmetry, whose space group drives the k-point
    // reduction, and setup_dynamical, which builds mindist_list.
    const auto fcs_loaded_early = init_u0_from_modes || fcs_phonon->relaxed_structure;
    if (fcs_phonon->relaxed_structure) {
        if (!init_u0_from_modes) setup_fcs();
        apply_relaxed_structure();
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
    // Not when it was loaded early: reloading would replicate the IFCs
    // against the already-deformed cells, and the relative vectors that
    // came out would be neither the reference nor the deformed ones.
    if (!fcs_loaded_early) setup_fcs();
    phonon_velocity->setup_velocity();
    integration->setup_integration(anharmonic_core->quartic_mode, run_info.my_rank, get_verbosity());
    dos->setup(*integration, dynamical->require_eigenvectors);
    thermodynamics->setup();
    anharmonic_core->setup(fcs_phonon->maxorder, fcs_phonon->force_constant_with_cell, dos->kmesh_dos.get());
    dielec->init(dos->emin,
                 dos->emax,
                 dos->delta_e,
                 dynamical->nonanalytic,
                 writes->print_zmode,
                 mode_symmetry->print_irreps,
                 symmetry->SymmListWithMap);
    ewald->init(*dielec, fcs_phonon->force_constant_with_cell[0]);

    // The adaptive smearing widths come from the group velocities, which need the
    // non-analytic term, so this must follow dielec->init() and ewald->init().  ismear
    // is broadcast inside setup_integration, so the table cannot be built before that
    // either; every rank runs this block, as prepare_adaptivesmearing used to do.
    if (integration->ismear == 2) {
        NDArray<double, 3> vel_adaptive;
        vel_adaptive.resize(dos->kmesh_dos->nk, dynamical->neval, 3);
        phonon_velocity->get_phonon_group_velocity_mesh(*dos->kmesh_dos.get(),
                                                        system->get_primcell().lattice_vector,
                                                        *dynamical,
                                                        fcs_phonon->force_constant_with_cell[0],
                                                        *dielec,
                                                        *ewald,
                                                        vel_adaptive);
        integration->create_adaptive_sigma(dos->kmesh_dos.get(),
                                           system->get_primcell().reciprocal_lattice_vector,
                                           dynamical->neval,
                                           std::move(vel_adaptive));
    }

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
                                         *dielec,
                                         *ewald);

    if (mode_symmetry->print_irreps && run_info.my_rank == 0) {
        mode_symmetry->analyze_irreps_at_gamma(*symmetry,
                                               *dynamical,
                                               fcs_phonon->force_constant_with_cell[0],
                                               *dielec,
                                               *ewald);
    }

    if (dos->flag_dos) {
        dos->calc_dos_all(*integration, thermodynamics->classical);
    }

    gruneisen->setup(anharmonic_core->quartic_mode, fcs_phonon->force_constant_with_cell);
    if (gruneisen->gruneisen_mode > 0) {
        gruneisen->calc_gruneisen(kpoint->kpoint_bs.get(),
                                  dynamical->dymat_band.get(),
                                  dos->kmesh_dos.get(),
                                  dos->dymat_dos.get(),
                                  kpoint->kpoint_general.get(),
                                  dynamical->dymat_general.get());
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
                                             *dielec,
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
                                         *dielec,
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
