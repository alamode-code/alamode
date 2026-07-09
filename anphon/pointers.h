/*
 pointers.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <memory>
#include "phonon.h"

namespace PHON_NS
{
class Pointers
{
public:
    Pointers(PHON *ptr) :
        phon(ptr), system(ptr->system), symmetry(ptr->symmetry), kpoint(ptr->kpoint), integration(ptr->integration),
        fcs_phonon(ptr->fcs_phonon), dynamical(ptr->dynamical), phonon_velocity(ptr->phonon_velocity),
        thermodynamics(ptr->thermodynamics), anharmonic_core(ptr->anharmonic_core), mode_analysis(ptr->mode_analysis),
        selfenergy(ptr->selfenergy), conductivity(ptr->conductivity), iterativebte(ptr->iterativebte),
        writes(ptr->writes), dos(ptr->dos), gruneisen(ptr->gruneisen), mympi(ptr->mympi), isotope(ptr->isotope),
        scph(ptr->scph), ewald(ptr->ewald), dielec(ptr->dielec), qha(ptr->qha), relaxation(ptr->relaxation),
        timer(ptr->timer)
    {}

    virtual ~Pointers()
    {}

protected:
    PHON *phon;
    std::unique_ptr<System> &system;
    std::unique_ptr<Symmetry> &symmetry;
    std::unique_ptr<Kpoint> &kpoint;
    std::unique_ptr<Integration> &integration;
    std::unique_ptr<Fcs_phonon> &fcs_phonon;
    std::unique_ptr<Dynamical> &dynamical;
    std::unique_ptr<PhononVelocity> &phonon_velocity;
    std::unique_ptr<Thermodynamics> &thermodynamics;
    std::unique_ptr<AnharmonicCore> &anharmonic_core;
    std::unique_ptr<ModeAnalysis> &mode_analysis;
    std::unique_ptr<Selfenergy> &selfenergy;
    std::unique_ptr<Conductivity> &conductivity;
    std::unique_ptr<Iterativebte> &iterativebte;
    std::unique_ptr<Writes> &writes;
    std::unique_ptr<Dos> &dos;
    std::unique_ptr<Gruneisen> &gruneisen;
    std::unique_ptr<MyMPI> &mympi;
    std::unique_ptr<Isotope> &isotope;
    std::unique_ptr<Scph> &scph;
    std::unique_ptr<Ewald> &ewald;
    std::unique_ptr<Dielec> &dielec;
    std::unique_ptr<Qha> &qha;
    std::unique_ptr<Relaxation> &relaxation;
    std::unique_ptr<Timer> &timer;
};
} // namespace PHON_NS
