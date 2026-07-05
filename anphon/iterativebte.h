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

#include <fstream>
#include <memory>
#include <vector>
#include "collision_operator.h"
#include "pointers.h"

namespace PHON_NS
{
class KappaResultIOH5;

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

    int nk_3ph, nklocal, ns, ns2;

    // The distributed 3ph collision operator (L matrices, wedge
    // distribution, symmetry expansion); this class is the Jacobi solver
    // on top of it.
    std::unique_ptr<CollisionOperator> collision_op;

    // Persistent per-temperature state in the /iterativebte group of
    // PREFIX.kappa.h5 (borrowed from Conductivity; rank 0 and h5 mode
    // only). t_computed marks temperatures restored from a previous run,
    // which the solver skips.
    KappaResultIOH5 *ibte_io = nullptr;
    std::vector<unsigned char> t_computed;
    std::vector<unsigned char> t_converged;

    double ***vel;

    // linear response to deltaT
    double ***dFold;

    double ***damping4; // four phonon selfenergy

    void calc_damping4();

    // Local copy of the operator's wedge distribution (irreducible k
    // indices handled by this rank).
    std::vector<int> nk_l;

    void iterative_solver(); // calculate kappa iteratively

    void calc_kappa(int, double ***&, double **&); // calculate kappa with off equilibrium part

    void calc_boson(int, double **&, double **&);

    bool check_convergence(double **&, double **&); // check if convergence cirteria is meet

    //void write_result_gamma(unsigned int,unsigned int,double ***,double **) const;
    void write_result();

    void write_Q_dF(int, double **&, double ***&, bool converged);

    void write_kappa_iterative();
};
} // namespace PHON_NS
