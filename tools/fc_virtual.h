/*
 fc_virtual.h

 Copyright (c) 2018 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class AtomProperty
{
public:
    double x, y, z;
    int kind;
    int atom, tran;

    AtomProperty() {};

    AtomProperty(const AtomProperty &other) :
        x(other.x), y(other.y), z(other.z), kind(other.kind), atom(other.atom), tran(other.tran) {};

    AtomProperty(const double *pos, const int kind_in, const int atom_in, const int tran_in)
    {
        x = pos[0];
        y = pos[1];
        z = pos[2];
        kind = kind_in;
        atom = atom_in;
        tran = tran_in;
    }
};

class StructureProperty
{
public:
    double lattice_vector[3][3];
    std::vector<AtomProperty> atoms;
    int nat, natmin, ntran;
    int nspecies;
    int is_periodic[3];
    std::vector<std::string> kd_symbol;

    StructureProperty() {};
};

struct AtomCellSuper
{
    unsigned int index;
    unsigned int tran;
    unsigned int cell_s;
};

inline bool operator<(const AtomCellSuper &a, const AtomCellSuper &b)
{
    return a.index < b.index;
}

class FcsArrayWithCell
{
public:
    std::vector<AtomCellSuper> pairs;
    double fcs_val;

    FcsArrayWithCell() {};

    FcsArrayWithCell(const double fcs_in, const std::vector<AtomCellSuper> &pairs_in) :
        pairs(pairs_in), fcs_val(fcs_in) {};

    bool operator==(const FcsArrayWithCell &obj) const
    {
        if (pairs.size() != obj.pairs.size()) return false;

        std::vector<unsigned int> index_a, index_b;
        index_a.clear();
        index_b.clear();
        for (int i = 0; i < pairs.size(); ++i) {
            index_a.push_back(pairs[i].index);
            index_b.push_back(obj.pairs[i].index);
        }
        for (int i = 0; i < pairs.size(); ++i) {
            index_a.push_back(pairs[i].tran);
            index_a.push_back(pairs[i].cell_s);
            index_b.push_back(obj.pairs[i].tran);
            index_b.push_back(obj.pairs[i].cell_s);
        }
        return index_a == index_b;
    }

    bool operator<(const FcsArrayWithCell &obj) const
    {
        if (pairs.size() != obj.pairs.size()) return pairs.size() < obj.pairs.size();

        std::vector<unsigned int> index_a, index_b;
        index_a.clear();
        index_b.clear();
        for (int i = 0; i < pairs.size(); ++i) {
            index_a.push_back(pairs[i].index);
            index_b.push_back(obj.pairs[i].index);
        }
        for (int i = 0; i < pairs.size(); ++i) {
            index_a.push_back(pairs[i].tran);
            index_a.push_back(pairs[i].cell_s);
            index_b.push_back(obj.pairs[i].tran);
            index_b.push_back(obj.pairs[i].cell_s);
        }
        return std::lexicographical_compare(index_a.begin(), index_a.end(), index_b.begin(), index_b.end());
    }
};

void load_fcs_xml(const std::string, const int, StructureProperty &, std::vector<FcsArrayWithCell> *);

void write_new_xml(const std::string, const std::string, const std::string, const int, const double,
                   const StructureProperty &, std::vector<FcsArrayWithCell> *);

void mix_structure(const StructureProperty &, const StructureProperty &, StructureProperty &, const double);

void mix_forceconstant(std::vector<FcsArrayWithCell> *, std::vector<FcsArrayWithCell> *,
                       std::vector<FcsArrayWithCell> *, const double, const int);

std::string double2string(const double, const int nprec = 15);
