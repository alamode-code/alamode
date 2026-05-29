/*
dfc2.h

Copyright (c) 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/


#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

class Maps
{
public:
    unsigned int atom_num;
    unsigned int tran_num;
};


class FcsClassExtent
{
public:
    unsigned int atm1, atm2;
    unsigned int xyz1, xyz2;
    unsigned int cell_s;
    double fcs_val;

    FcsClassExtent() {};

    FcsClassExtent(const FcsClassExtent &obj)
    {
        atm1 = obj.atm1;
        atm2 = obj.atm2;
        xyz1 = obj.xyz1;
        xyz2 = obj.xyz2;
        cell_s = obj.cell_s;
        fcs_val = obj.fcs_val;
    }
};

class DeltaFcs
{
public:
    int sx, sy, sz;
    int atm1, xyz1;
    int atm2, xyz2;
    double dfc2;

    DeltaFcs(const int sx, const int sy, const int sz, const int atm1, const int xyz1, const int atm2, const int xyz2,
             const double dfc2) : sx(sx), sy(sy), sz(sz), atm1(atm1), xyz1(xyz1), atm2(atm2), xyz2(xyz2), dfc2(dfc2) {};
};

class FcsTrans
{
public:
    int fcs_index;
    std::vector<int> arr;

    FcsTrans(const std::vector<int> &arr_in, const int index_in)
    {
        std::copy(arr_in.begin(), arr_in.end(), std::back_inserter(arr));
        fcs_index = index_in;
    }

    bool operator==(const FcsTrans &obj) const
    {
        return arr == obj.arr;
    }

    bool operator<(const FcsTrans &obj) const
    {
        return std::lexicographical_compare(arr.begin(), arr.end(), obj.arr.begin(), obj.arr.end());
    }
};


std::string original_xml, new_xml;
std::string file_fc2_correction;

unsigned int nat, nkd, ntran, natmin;
unsigned int nat_p, nkd_p, natmin_p;
unsigned int *kd, *kd_p;

std::string *kd_symbol;

double temp;
double lavec_s[3][3], lavec_p[3][3];
double rlavec_p[3][3];
double **xr_s, **xr_p;
unsigned int is_periodic[3];

void load_fc2_xml(const std::string);

void load_delta_fc2(const std::string, const double);

void calculate_new_fc2(const std::vector<FcsClassExtent> &, std::vector<DeltaFcs>, std::vector<FcsClassExtent> &);

void recips(double[3][3], double[3][3]);

void write_new_xml(const std::vector<FcsClassExtent>, const std::string);

std::string double2string(const double, const int nprec = 15);


unsigned int **map_p2s;
Maps *map_s2p;

std::vector<FcsClassExtent> fc2_orig, fc2_new;
std::vector<DeltaFcs> delta_fc2;
