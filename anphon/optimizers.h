/*
 relaxation.h

 Copyright (c) 2025 Takumi Chida, Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "mpi_common.h"
#include "dynamical.h"
#include "gruneisen.h"
#include "system.h"
#include "constants.h"
#include "scph.h"
#include "parsephon.h"
#include "error.h"
#include "timer.h"
#include "mathfunctions.h"
#include <fftw3.h>
#include <iomanip>
#include <Eigen/Core>
#include <iomanip>

using namespace PHON_NS;

class Optimizer{
public:

    Optimizer();
    ~Optimizer();

    virtual void update_state(const int dim,
                              const std::vector<double> &grad_vec, 
                              std::vector<double> &state_vec,
                              const std::vector<std::vector<double>> &hessian,
                              std::vector<double> &delta)
                            {std::cout << "Parent class" << std::endl;};
};

class Newton_Optimizer : public Optimizer{
    public:
    double mixbeta = 1.0;

    Newton_Optimizer();
    ~Newton_Optimizer();

    Newton_Optimizer(double mixbeta);
    void update_state(const int dim,
                      const std::vector<double> &grad_vec, 
                      std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian,
                      std::vector<double> &delta);
};