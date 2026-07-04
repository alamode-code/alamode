/*
 conductivity.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/


/*
implementation of the iterative bte
additional parameter:
    ITERATIVE = 1 -> do iterative calculation
    MAX_CYCLE = 20 (default)
    ITER_THRESHOLD = 0.005 (default)
    
we need to calculate and store: 
- k points in the irreducible BZ are divided amoung processors.
- we calculate L absorb and L emitt, the transition probability of absorption and emission process
- we go through each temperature, for each temperature, we calculate Q, W is calculated for each iteration
- for each temperature, we check for convergence

Improvement:
- how to implement a restart? 
    iterative part is fast, the time consuming part is the calculation of L
    so for restart, we should write down L, this means we should change the for so that L contains 
    the iteration of ik and s1, which we then write out at each iteration.
    we also need to implement the read and write part.
    
- add the direct solution method

*/

#pragma once

#include <array>
#include <complex>
#include <fstream>
#include <set>
#include <vector>
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{
class Iterativebte: protected Pointers
{
public:
    Iterativebte(class PHON *);

    ~Iterativebte();

    void setup_iterative(); // initialize variables

    void do_iterativebte(); // wrapper

    double *Temperature; // non-owning alias of conductivity->temperature
    unsigned int ntemp;

    int max_cycle;
    int min_cycle;
    double mixing_factor;

    double convergence_criteria; // dF(i+1) - dF(i) < cc

    double ***kappa;

    std::fstream fs_result;

private:
    void set_default_variables();

    void deallocate_variables();

    bool use_triplet_symmetry;
    bool sym_permutation;

    int kplength_emitt;
    int kplength_absorb;
    int nk_3ph, nklocal, ns, ns2;

    // calculated at equilibrium
    double ***L_absorb; // L q0 + q1 -> q2
    double ***L_emitt;  // L q0 -> q1 + q2
    double ***vel;

    // linear response to deltaT
    double ***dFold;
    double ***dFnew;

    double ***damping4; // four phonon selfenergy

    void calc_damping4();

    std::vector<std::vector<KsListGroup>> localnk_triplets_emitt;
    std::vector<std::vector<KsListGroup>> localnk_triplets_absorb;

    std::vector<int> nk_l;

    // Flattened (local irreducible k, triplet) index built once in
    // get_triplets(): the row of a triplet in L_emitt/L_absorb is
    // offset_*[ik] + j, and pairs_* lists all rows as (ik, j) for loops
    // that parallelize over the flat index directly.
    std::vector<std::array<int, 2>> pairs_emitt, pairs_absorb;
    std::vector<int> offset_emitt, offset_absorb;

    void iterative_solver(); // calculate kappa iteratively

    void calc_kappa(int, double ***&, double **&); // calculate kappa with off equilibrium part

    void get_triplets(); // set up all triplets

    void setup_L_smear();

    void setup_L_tetra();

    void calc_Q_from_L(double **&, double **&); // calculate Q

    void calc_boson(int, double **&, double **&);

    bool check_convergence(double **&, double **&); // check if convergence cirteria is meet
    bool check_convergence(const std::vector<double> &);

    //void write_result_gamma(unsigned int,unsigned int,double ***,double **) const;
    void write_result();

    void write_Q_dF(int, double **&, double ***&);

    void write_kappa_iterative();
};
} // namespace PHON_NS
