/*
 qe2alm.h

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <fstream>
#include <iostream>
#include <vector>

std::string file_fc2qe, file_xmlout;
std::string flag_asr;
std::ifstream ifs_fc2qe;

int i, j, k;
int natmin, nkd, ibrav;
int nsize, nat;
double celldm[6];
double lavec[3][3], rlavec[3][3];

int *kd, **kd_super;
int **map_p2s;
double **xcrd, **xcrd_super;
std::string *kd_symbol;

bool include_na;
bool consider_asr;
;
int nq[3];
double dielec[3][3];
double ***born;
double ***fc2;

class DistInfo
{
public:
    int cell;
    double dist;
    double relvec[3];

    DistInfo();

    DistInfo(const int n, const double d, const double x[3])
    {
        cell = n;
        dist = d;
        for (int i = 0; i < 3; ++i)
            relvec[i] = x[i];
    }

    DistInfo(const DistInfo &obj)
    {
        cell = obj.cell;
        dist = obj.dist;
        for (int i = 0; i < 3; ++i)
            relvec[i] = obj.relvec[i];
    }
};

inline bool operator<(const DistInfo a, const DistInfo b)
{
    return a.dist < b.dist;
}

std::vector<DistInfo> **mindist;

void calc_lattice_vector(const int, double[6], double[3][3]);

void recips(double[3][3], double[3][3]);

std::string double2string(const double);

void get_pairs_of_minimum_distance(const int, const int, int **, double **, std::vector<DistInfo> **);
