/*
 constants.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <cfloat>
#include <cmath>
#include <complex>

constexpr double pi = M_PI;
constexpr double inverse_pi = 1.0 / pi; // pi^{-1}
constexpr double tpi = 2.0 * pi;        // (2*pi)
constexpr double inv_tpi = 1.0 / tpi;   // (2*pi)^{-1}
constexpr double amu = 1.660538782e-27;
constexpr double electron_mass = 9.10938215e-31;
constexpr double amu_ry = amu / electron_mass / 2.0;
constexpr double c_light = 299792458;
constexpr double h_planck = 6.62606896e-34;
constexpr double Ryd = 4.35974394e-18 / 2.0;
constexpr double time_ry = h_planck * inv_tpi / Ryd;
constexpr double Hz_to_kayser = 1.0e-2 * inv_tpi / c_light;
constexpr double Bohr_in_Angstrom = 0.52917721092;
constexpr double Ryd_in_eV = 13.605693122994;
constexpr double k_Boltzmann = 1.3806488e-23; // J/K
constexpr auto im = std::complex<double>(0.0, 1.0);
constexpr double eps = DBL_EPSILON;
constexpr double eps15 = 1.0e-15;
constexpr double eps12 = 1.0e-12;
constexpr double eps10 = 1.0e-10;
constexpr double eps8 = 1.0e-8;
constexpr double eps6 = 1.0e-6;
constexpr double eps4 = 1.0e-4;
