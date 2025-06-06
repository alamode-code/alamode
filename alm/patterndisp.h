/*
 patterndisp.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

//#include "pointers.h"
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "cluster.h"
#include "constraint.h"
#include "fcs.h"
#include "symmetry.h"
#include "system.h"

namespace ALM_NS
{
class DispAtomSet
{
public:
    std::vector<int> atomset;

    DispAtomSet();

    DispAtomSet(std::vector<int> atomset_in) : atomset(std::move(atomset_in))
    {}
};

class DirectionVec
{
public:
    double direction[3]{};

    DirectionVec();

    DirectionVec(const double vec_in[3])
    {
        for (auto i = 0; i < 3; ++i)
            direction[i] = vec_in[i];
    }
};

class DispDirectionHarmonic
{
public:
    int atom;
    std::vector<DirectionVec> directionlist;

    DispDirectionHarmonic();

    DispDirectionHarmonic(const int n, const std::vector<DirectionVec> &list_in)
    {
        atom = n;
        for (auto &it: list_in) {
            directionlist.push_back(it);
        }
    }
};

class AtomWithDirection
{
public:
    std::vector<int> atoms;
    std::vector<double> directions;

    AtomWithDirection();

    AtomWithDirection(std::vector<int> a, std::vector<double> b) : atoms(std::move(a)), directions(std::move(b))
    {}

    AtomWithDirection(const int n, const int *a, const double **b)
    {
        for (auto i = 0; i < n; ++i) {
            atoms.push_back(a[i]);
            for (auto j = 0; j < 3; ++j) {
                directions.push_back(b[i][j]);
            }
        }
    }
};


inline auto operator<(const DispAtomSet &a, const DispAtomSet &b) -> bool
{
    return std::lexicographical_compare(a.atomset.begin(), a.atomset.end(), b.atomset.begin(), b.atomset.end());
}


class IndexWithSign
{
public:
    int ind, sign;

    IndexWithSign();

    IndexWithSign(const int ind_in, const int sign_in)
    {
        ind = ind_in;
        sign = sign_in;
    }
};

inline auto operator<(const IndexWithSign &a, const IndexWithSign &b) -> bool
{
    return a.ind < b.ind;
}

class Displace
{
public:
    Displace();

    ~Displace();

    auto gen_displacement_pattern(const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Symmetry> &symmetry,
                                  const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                  const std::unique_ptr<System> &system, const int verbosity) -> void;

    auto set_trim_dispsign_for_evenfunc(const bool) -> void;

    [[nodiscard]] auto get_disp_basis() const -> std::string;

    auto set_disp_basis(const std::string &) -> void;

    [[nodiscard]] auto get_pattern_all(const int) const -> const std::vector<AtomWithDirection> &;

private:
    bool trim_dispsign_for_evenfunc;
    std::string disp_basis;
    std::vector<AtomWithDirection> *pattern_all;

    std::vector<DispDirectionHarmonic> disp_harm, disp_harm_best;

    auto set_default_variables() -> void;

    auto deallocate_variables() -> void;

    auto generate_pattern_all(const int maxorder, const size_t nat, const Eigen::Matrix3d &lavec,
                              const std::unique_ptr<Symmetry> &symmetry, const std::set<DispAtomSet> *dispset_in,
                              const std::string &preferred_basis) const -> void;

    auto generate_signvecs(const int, std::vector<std::vector<int>> &, std::vector<int>) const -> void;

    auto find_unique_sign_pairs(const int natom_disp_in, const size_t nat, const std::unique_ptr<Symmetry> &symmetry,
                                const std::vector<std::vector<int>> &sign_in, const std::vector<int> &pair_in,
                                std::vector<std::vector<int>> &sign_out, const std::string &preferred_basis) const
        -> void;
};
} // namespace ALM_NS
