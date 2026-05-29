/*
 analyze_phonons.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <string>
#include "constants.h"


int locate_tag(std::string);

std::ifstream ifs;

int i, j, k, l;

int nat, nkd, ns;
int nkx, nky, nkz;
int nk;
int nt;
double *temp, tmin, tmax, dt;
double volume;

double **omega, ***tau;
double ****vel;
int *n_weight;

std::string calc, file_isotope;

int beg_k, end_k;
int beg_s, end_s;
int average_gamma;
int isotope;
bool classical;

void calc_tau(int);

void calc_tau_temp(int, int);

void calc_kappa();

void calc_kappa_cumulative(double max_length, double delta_length, int itemp, const int nsample,
                           const std::string &gridtype);

void calc_kappa_cumulative2(double max_length, double delta_length, int itemp, int flag[3], const int nsample,
                            const std::string &gridtype);

void calc_kappa_boundary(const double);

void calc_kappa_boundary2(double, double, int, int[3]);

double Cv(double, double);

void average_gamma_at_degenerate_point(double **, double ***, const int, const int, const int);

void average_gamma_isotope_at_degenerate_point(double **, double **, const int, const int);

void update_tau_isotope(const std::string, double **, double ***, const int, const int, const int);

static const double Ryd_to_kayser = Hz_to_kayser / time_ry;
static const double kayser_to_Ryd = 1.0 / Ryd_to_kayser;
static const double T_to_Ryd = k_Boltzmann / Ryd;


// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}
