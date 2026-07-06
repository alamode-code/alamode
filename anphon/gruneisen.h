/*
 gruneisen.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>
#include "fcs_phonon.h"
#include "ndarray.h"
#include "pointers.h"

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

    NDArray<std::complex<double>, 2> gruneisen_bs;
    NDArray<std::complex<double>, 2> gruneisen_dos;

    void calc_gruneisen();

    void write_new_fcsxml_all() const;

private:
    void set_default_variables();

    void deallocate_variables();

    NDArray<double, 2> xshift_s;
    std::vector<FcsArrayWithCell> delta_fc2, delta_fc3;

    void prepare_delta_fcs(const std::vector<FcsArrayWithCell> &, std::vector<FcsArrayWithCell> &) const;

    // void impose_ASR_on_harmonic_IFC(std::vector<FcsArrayWithCell> &,
    //                    int);

    void write_new_fcsxml(const std::string &, double) const;

    static std::string double2string(double);

    //  double calc_stress_energy2(const std::vector<FcsArrayWithCell>);
    void calc_stress_energy3(std::vector<FcsArrayWithCell>, double ****);

    void print_stress_energy();
};
} // namespace PHON_NS
