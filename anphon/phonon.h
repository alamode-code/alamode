/*
 phonon.h

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

#include <memory>
#include <string>

namespace PHON_NS
{
class PHON
{
public:
    PHON(MPI_Comm comm);

    virtual ~PHON();

    std::unique_ptr<class Timer> timer;

    std::unique_ptr<class System> system;

    std::unique_ptr<class Symmetry> symmetry;

    std::unique_ptr<class Kpoint> kpoint;

    std::unique_ptr<class Integration> integration;

    std::unique_ptr<class Fcs_phonon> fcs_phonon;

    std::unique_ptr<class Dynamical> dynamical;

    std::unique_ptr<class PhononVelocity> phonon_velocity;

    std::unique_ptr<class Thermodynamics> thermodynamics;

    std::unique_ptr<class AnharmonicCore> anharmonic_core;

    std::unique_ptr<class ModeAnalysis> mode_analysis;

    std::unique_ptr<class Selfenergy> selfenergy;

    std::unique_ptr<class Conductivity> conductivity;

    std::unique_ptr<class Writes> writes;

    std::unique_ptr<class Dos> dos;

    std::unique_ptr<class Iterativebte> iterativebte;

    std::unique_ptr<class Gruneisen> gruneisen;

    std::unique_ptr<class MyMPI> mympi;

    std::unique_ptr<class Isotope> isotope;

    std::unique_ptr<class Scph> scph;

    std::unique_ptr<class Ewald> ewald;

    std::unique_ptr<class Dielec> dielec;

    std::unique_ptr<class Qha> qha;

    std::unique_ptr<class Relaxation> relaxation;

    void create_pointers();

    void destroy_pointers();

    std::string mode;

    std::string job_title;

    // FILE_FORMAT tag: true (default) routes restart/state files through
    // the unified HDF5 formats; false forces the legacy text files.
    bool use_hdf5_io = true;

    // ALLOW_UNCONVERGED tag: accept renormalized IFC/structure data from an
    // SCPH/QHA state file even when its iterations did not converge
    // (FC2_TEMPERATURE reads and RESTART_SCPH/RESTART_QHA refuse them by
    // default).
    bool allow_unconverged = false;

    // VERBOSITY tag: controls how much progress output is printed.
    // 0 = silent, 1 = normal progress/banners (default), 2 = extra detail.
    // Canonical owner of the setting (mirrors ALM::verbosity); the Writes
    // accessors forward here. Set on rank 0 during input parsing and
    // broadcast to all ranks in PhononCUI::run.
    unsigned int verbosity = 1;

    void set_verbosity(unsigned int verbosity_in);

    [[nodiscard]] unsigned int get_verbosity() const;

    // Dispatch to the executor selected by the current mode.
    void run() const;

    void execute_phonons() const;

    void execute_kappa() const;

    void execute_self_consistent_phonon() const;

    void setup_base() const;
};
} // namespace PHON_NS
