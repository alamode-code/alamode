/*
 isotope.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <vector>
#include "kpoint.h"

namespace PHON_NS
{
class System;
class Integration;
class TetraNodes;
class DymatEigenValue;

// Phonon-isotope (mass-disorder) scattering rates via the Tamura formula.
// No Pointers base: the entry points take their inputs explicitly from the
// caller (PHON::execute_kappa); the parser fills the public config members.
class Isotope
{
public:
    Isotope();

    ~Isotope();

    int include_isotope;
    std::vector<double> isotope_factor;
    double **gamma_isotope;

    // Broadcast the parsed configuration, fill missing ISOFACT entries from
    // the built-in database, and allocate gamma_isotope (rank 0).
    void setup_isotope_scattering(const System &system_in, unsigned int nk_irred_in, unsigned int ns_in,
                                  int my_rank_in);

    void calc_isotope_selfenergy_all(const KpointMeshUniform &kmesh_dos_in, const DymatEigenValue &dymat_dos_in,
                                     const TetraNodes &tetra_nodes_dos_in, const System &system_in,
                                     Integration &integration_in, unsigned int ns_in, int my_rank_in,
                                     int nprocs_in) const;

private:
    void set_default_variables();

    void deallocate_variables();

    void calc_isotope_selfenergy(const unsigned int knum, const unsigned int snum, const double omega,
                                 const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                 const std::complex<double> *const *const *evec_in, const System &system_in,
                                 Integration &integration_in, const unsigned int ns_in, double &ret) const;

    void calc_isotope_selfenergy_tetra(const unsigned int knum, const unsigned int snum, const double omega,
                                       const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                       const std::complex<double> *const *const *evec_in, const System &system_in,
                                       Integration &integration_in, const TetraNodes &tetra_nodes_in,
                                       const unsigned int ns_in, double &ret) const;

    void set_isotope_factor_from_database(const System &system_in, const int, const std::string *,
                                          std::vector<double> &);

    std::vector<double> isotope_factors{0.00011460742534756168,
                                        1.2150768298148375e-07,
                                        0.0014588200485501472,
                                        0.0,
                                        0.0013539149846334955,
                                        7.387230453507936e-05,
                                        1.8376657935622062e-05,
                                        3.358802509356315e-05,
                                        0.0,
                                        0.0008278312013487067,
                                        0.0,
                                        0.0007398827329803543,
                                        0.0,
                                        0.00020070043414117884,
                                        0.0,
                                        0.00016512811706321465,
                                        0.0005827012669402452,
                                        3.4803742792164614e-05,
                                        0.00016400010165430582,
                                        0.0002975638213562891,
                                        0.0,
                                        0.00028645552741825937,
                                        9.54832036799122e-07,
                                        0.00013287040458747333,
                                        0.0,
                                        8.244415886624584e-05,
                                        0.0,
                                        0.0004307094643900411,
                                        0.00021093334605961256,
                                        0.0005931532093279201,
                                        0.0001971273025933254,
                                        0.0005867117795429008,
                                        0.0,
                                        0.00046053773717786,
                                        0.00015627684094360946,
                                        0.0002488075418011238,
                                        0.00010969638953101122,
                                        6.099698033035131e-05,
                                        0.0,
                                        0.0003426292948175251,
                                        0.0,
                                        0.0005968576646618982,
                                        -1,
                                        0.0004066665857603728,
                                        0.0,
                                        0.0003094783537612192,
                                        8.579867520082529e-05,
                                        0.00027153572115266693,
                                        1.2430903342376982e-05,
                                        0.0003340852255990917,
                                        6.607546197957849e-05,
                                        0.0002839333322460794,
                                        0.0,
                                        0.00026754775528758446,
                                        0.0,
                                        6.247981695145255e-05,
                                        4.591735999752819e-08,
                                        2.2495450855209385e-05,
                                        0.0,
                                        0.0002323717843347356,
                                        -1,
                                        0.0003348897750118124,
                                        4.3279432991825585e-05,
                                        0.00012767490913313887,
                                        0.0,
                                        5.1980730746625334e-05,
                                        0.0,
                                        7.232481416362226e-05,
                                        0.0,
                                        8.560789645205211e-05,
                                        8.30080231546816e-07,
                                        5.2516616775106e-05,
                                        3.595087509564231e-09,
                                        6.966808158951864e-05,
                                        2.708493881651469e-05,
                                        7.452357784116751e-05,
                                        2.5359000205138314e-05,
                                        3.42867052115942e-05,
                                        0.0,
                                        6.522800081772828e-05,
                                        1.9964347198006125e-05,
                                        1.9437783811894164e-05,
                                        0.0,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        -1,
                                        1.4928612482562733e-08,
                                        0.0,
                                        1.1564620560663096e-06,
                                        -1,
                                        -1}; // Precalculated isotope factors.
    // They are computed using the isotope abundance information reported by CIAAW.
    // Some recent changes are not considered because of the presence of uncertainty interval.
    // For unstable elements, the value is set to -1.
};
} // namespace PHON_NS
