/*
 gruneisen.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "fcs_phonon.h"
#include "pointers.h"
#include <string>
#include <complex>
#include <vector>

namespace PHON_NS
{

class Gruneisen: protected Pointers
{
public:
    Gruneisen(class PHON *);

    ~Gruneisen();

    double delta_a;
    bool print_gruneisen;
    bool print_newfcs;

    void setup();

    std::complex<double> **gruneisen_bs;
    std::complex<double> **gruneisen_dos;

    void calc_gruneisen();

    void write_new_fcsxml_all();

private:
    void set_default_variables();

    void deallocate_variables();

    double **xshift_s;
    std::vector<FcsArrayWithCell> delta_fc2, delta_fc3;

    void prepare_delta_fcs(const std::vector<FcsArrayWithCell> &,
                           std::vector<FcsArrayWithCell> &,
                           const int) const;

    // void impose_ASR_on_harmonic_IFC(std::vector<FcsArrayWithCell> &,
    //                    int);

    void calc_dfc2_reciprocal(std::complex<double> **,
                              const double *);

    void write_new_fcsxml(const std::string &,
                          double);

    std::string double2string(double) const;

    //  double calc_stress_energy2(const std::vector<FcsArrayWithCell>);
    void calc_stress_energy3(std::vector<FcsArrayWithCell>,
                             double ****);

    void print_stress_energy();
};
}
