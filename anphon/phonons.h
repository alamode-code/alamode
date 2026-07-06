/*
 phonons.h

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
    PHON(int, char **, MPI_Comm);

    virtual ~PHON();

    std::unique_ptr<class Timer> timer;

    std::unique_ptr<class Input> input;

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
    //bool restart_flag;

    void execute_phonons() const;

    void execute_kappa() const;

    void execute_self_consistent_phonon() const;

    void setup_base() const;
};
} // namespace PHON_NS
