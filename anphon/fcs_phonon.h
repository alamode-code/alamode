/*
 fcs_phonon.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <set>
#include <string>
#include <vector>
#include "mathfunctions.h"
#include "pointers.h"

namespace PHON_NS
{

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

    bool operator==(const FcsClassExtent &a) const
    {
        return (this->atm1 == a.atm1) && (this->atm2 == a.atm2) && (this->xyz1 == a.xyz1) && (this->xyz2 == a.xyz2) &&
               (this->cell_s == a.cell_s);
    }
};

struct AtomCellSuper
{
    unsigned int index; // flattened array
    unsigned int tran;
    unsigned int cell_s;
} __attribute__((aligned(16)));

inline bool operator<(const AtomCellSuper &a, const AtomCellSuper &b)
{
    return a.index < b.index;
}

class FcsArrayWithCell
{
public:
    std::vector<AtomCellSuper> pairs;
    //std::vector<unsigned int> atoms_p; // atom index in the primitive cell (not used?)
    std::vector<unsigned int> atoms_s; // atom index in the supercell
    std::vector<unsigned int> coords;  // xyz components
    double fcs_val;
    std::vector<Eigen::Vector3d> relvecs;          // For computing phase factor in exp
    std::vector<Eigen::Vector3d> relvecs_velocity; // For computing group velocity matrix

    FcsArrayWithCell() {};

    FcsArrayWithCell(const double fcs_in, const std::vector<AtomCellSuper> &pairs_in,
                     const std::vector<unsigned int> &atoms_s_in) :
        pairs(pairs_in), atoms_s(atoms_s_in), fcs_val(fcs_in)
    {
        coords.clear();
        for (const auto &it: pairs_in) {
            coords.push_back(it.index % 3);
        }
    };

    FcsArrayWithCell(const double fcs_in, const std::vector<AtomCellSuper> &pairs_in,
                     const std::vector<unsigned int> &atoms_s_in, const std::vector<Eigen::Vector3d> &relvecs_vel_in) :
        pairs(pairs_in), atoms_s(atoms_s_in), fcs_val(fcs_in), relvecs_velocity(relvecs_vel_in)
    {
        coords.clear();
        for (const auto &it: pairs_in) {
            coords.push_back(it.index % 3);
        }
    };

    FcsArrayWithCell(const double fcs_in, const std::vector<AtomCellSuper> &pairs_in,
                     const std::vector<unsigned int> &atoms_s_in, const std::vector<Eigen::Vector3d> &relvecs_in,
                     const std::vector<Eigen::Vector3d> &relvecs_vel_in) :
        pairs(pairs_in), atoms_s(atoms_s_in), fcs_val(fcs_in), relvecs(relvecs_in), relvecs_velocity(relvecs_vel_in)
    {
        coords.clear();
        for (const auto &it: pairs_in) {
            coords.push_back(it.index % 3);
        }
    };

    bool operator<(const FcsArrayWithCell &obj) const
    {
        const auto n = pairs.size();
        for (int i = 0; i < n; ++i) {
            if (pairs[i].index != obj.pairs[i].index) {
                return pairs[i].index < obj.pairs[i].index;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (pairs[i].tran != obj.pairs[i].tran) {
                return pairs[i].tran < obj.pairs[i].tran;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (pairs[i].cell_s != obj.pairs[i].cell_s) {
                return pairs[i].cell_s < obj.pairs[i].cell_s;
            }
        }
        return false;
    }

    static bool sort_with_index(const FcsArrayWithCell &a, const FcsArrayWithCell &b)
    {
        const auto n = a.pairs.size();
        std::vector<size_t> index_a(n), index_b(n);
        for (auto i = 0; i < n; ++i) {
            if (a.pairs[i].index != b.pairs[i].index) {
                return a.pairs[i].index < b.pairs[i].index;
            }
        }
        return false;
    }
};


struct sort_by_heading_indices
{
    unsigned int number_of_tails; // number of indices at the tail of the array
    // It should be 1 when renormalizing 2nd-order force constants by 3rd-order force constants,
    // and should be 2 when renormalizing 2nd-order force constants by 4th-order force constants.

    sort_by_heading_indices(const unsigned int n) : number_of_tails(n)
    {}

    inline bool operator()(const FcsArrayWithCell &a, const FcsArrayWithCell &b) const
    {
        std::vector<int> array_a, array_b;
        array_a.clear();
        array_b.clear();
        const auto len = a.pairs.size();

        for (auto i = 0; i < len - number_of_tails; ++i) {
            array_a.push_back(a.pairs[i].index);
            array_b.push_back(b.pairs[i].index);
        }
        // The components of relvec should be integers,
        // so let's convert their types into int for sort.
        for (auto i = 0; i < len - number_of_tails - 1; ++i) {
            for (auto j = 0; j < 3; ++j) {
                array_a.push_back(nint(a.relvecs[i][j]));
                array_b.push_back(nint(b.relvecs[i][j]));
            }
        }
        // Register the last index
        for (auto i = len - number_of_tails; i < len; ++i) {
            array_a.push_back(a.pairs[i].index);
            array_b.push_back(b.pairs[i].index);
        }
        for (auto i = len - number_of_tails - 1; i < len - 1; ++i) {
            for (auto j = 0; j < 3; ++j) {
                array_a.push_back(nint(a.relvecs[i][j]));
                array_b.push_back(nint(b.relvecs[i][j]));
            }
        }

        return std::lexicographical_compare(array_a.begin(), array_a.end(), array_b.begin(), array_b.end());
    }
};

class Fcs_phonon: protected Pointers
{
public:
    Fcs_phonon(class PHON *);

    ~Fcs_phonon();

    void setup(const std::string &);

    unsigned int maxorder;
    std::string file_fcs, file_fc2, file_fc3, file_fc4;

    std::vector<FcsArrayWithCell> *force_constant_with_cell;

    bool update_fc2;

    // FC2_TEMPERATURE tag: when >= 0 and the FC2 HDF5 file carries
    // temperature-dependent force constants (an SCPH/QHA state file), the
    // row matching this temperature is loaded instead of the base values.
    double fc2_temperature = -1.0;

    void get_fcs_from_file(const std::string &fname_fcs, const int order, std::vector<FcsArrayWithCell> &fcs_out) const;

    void replicate_force_constant(const System *system_in, std::vector<FcsArrayWithCell> &fcs_inout) const;

private:
    bool require_cubic;
    bool require_quartic;

    void set_default_variables();

    void deallocate_variables();

    void load_fc2_xml();

    void load_fcs_xml(const std::string &fname_fcs, const int order, std::vector<FcsArrayWithCell> &fcs_out) const;

    void parse_fcs_from_h5(const std::string &fname_fcs, const int order,
                           std::vector<FcsArrayWithCell> &fcs_out) const;

    void load_fcs_from_file(const int maxorder_in) const;


    static double examine_translational_invariance(int order, unsigned int nat, unsigned int natmin,
                                                   const std::vector<FcsArrayWithCell> &fc_in);

    void replicate_force_constants(const int maxorder_in) const;

    void MPI_Bcast_fc_class(unsigned int) const;

    void MPI_Bcast_fcs_array(unsigned int) const;

    void MPI_Bcast_fc2_ext();
};
} // namespace PHON_NS
