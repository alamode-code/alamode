/*
 cluster.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "constants.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"

#include "mathfunctions.h"


namespace ALM_NS
{
class IntList
{
public:
    std::vector<int> iarray;

    IntList() = default;

    ~IntList() = default;

    IntList(const IntList &a) = default;

    IntList &operator=(const IntList &a) = default;

    IntList(const int n, const int *arr)
    {
        for (auto i = 0; i < n; ++i) {
            iarray.push_back(arr[i]);
        }
    }

    // Constructor from std::vector for modern C++ style
    explicit IntList(const std::vector<int> &vec) : iarray(vec)
    {}
    explicit IntList(std::vector<int> &&vec) : iarray(std::move(vec))
    {}

    auto operator<(const IntList &a) const -> bool
    {
        return std::lexicographical_compare(iarray.begin(), iarray.end(), a.iarray.begin(), a.iarray.end());
    }

    auto operator==(const IntList &a) const -> bool
    {
        // Use vector's built-in equality operator for consistency
        // This is required for std::unordered_set<IntList> in fcs.cpp
        // and ensures consistent behavior across platforms
        // std::cout << "== operator called\n";
        return iarray == a.iarray;
    }
};

class RelativeVectors
{
public:
    int order;
    std::vector<std::vector<double>> relvecs_cartesian;
    std::vector<std::vector<double>> relvecs_fractional;

    RelativeVectors() = default;

    RelativeVectors(const int order_in)
    {
        relvecs_cartesian.clear();
        order = order_in;
    }

    auto clear_relvecs() -> void
    {
        for (int i = 0; i < order + 2; i++) {
            relvecs_fractional[i].clear();
            relvecs_cartesian[i].clear();
        }
    }

    auto make_fractional_from_cartesian(const Eigen::Matrix3d &reciprocal_lat) -> void
    {
        std::vector<double> vectmp;
        relvecs_fractional.clear();

        for (int i = 0; i < order + 1; i++) {

            vectmp.clear();
            for (int xyz = 0; xyz < 3; xyz++) {
                vectmp.push_back(0.0);
                for (int xyz2 = 0; xyz2 < 3; xyz2++) {
                    vectmp[xyz] += reciprocal_lat(xyz, xyz2) / (2.0 * pi) * relvecs_cartesian[i][xyz2];
                }
            }

            relvecs_fractional.push_back(vectmp);
        }
    }

    auto is_equal(const RelativeVectors &another, const double threshold) const -> bool
    {
        std::vector<int> is_checked(order + 1);
        int is_found = 0;

        for (int i = 0; i < order + 1; i++) {
            is_found = 0;
            // find corresponding relative vector
            for (int j = 0; j < order + 1; j++) {
                if (is_checked[j] == 1) {
                    continue;
                }
                int is_equal_flg = 1;
                for (int xyz = 0; xyz < 3; xyz++) {
                    if (std::fabs(relvecs_cartesian[i][xyz] - another.relvecs_cartesian[j][xyz]) > threshold) {
                        is_equal_flg = 0;
                        break;
                    }
                }
                if (is_equal_flg == 1) {
                    is_found = 1;
                    is_checked[j] = 1;
                    break;
                }
            }
            // if not found
            if (is_found == 0) {
                return false;
            }
        }
        return true;
    }
};

class PairDistances
{
public:
    std::vector<int> cells;
    std::vector<double> distances;
    std::vector<Eigen::Vector3d> relative_vectors; // Cartesian frame
    int ncells_minimum_distance{0};
    double tol_distance;

    PairDistances() = default;

    ~PairDistances() = default;

    PairDistances(const std::vector<int> &cells_, const std::vector<double> &distances_,
                  const std::vector<Eigen::Vector3d> &relative_vectors_, const double tolerance = 1.0e-3)
    {
        std::vector<size_t> indices(distances_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&distances_](int left, int right) -> bool {
            return distances_[left] < distances_[right];
        });
        cells.resize(cells_.size());
        distances.resize(distances_.size());
        relative_vectors.resize(relative_vectors_.size());

        for (auto i = 0; i < cells_.size(); ++i) {
            cells[i] = cells_[indices[i]];
            distances[i] = distances_[indices[i]];
            relative_vectors[i] = relative_vectors_[indices[i]];
        }
        const auto dist_min = distances[0];
        ncells_minimum_distance = 0;
        for (const auto &it: distances) {
            if (std::abs(it - dist_min) < tolerance) {
                ++ncells_minimum_distance;
            } else {
                break;
            }
        }
        tol_distance = tolerance;
    }
};

class MinDistList
{
public:
    std::vector<int> cell;
    std::vector<double> dist;

    MinDistList() = default;

    MinDistList(std::vector<int> cell_in, std::vector<double> dist_in) :
        cell(std::move(cell_in)), dist(std::move(dist_in)) {};

    static auto compare_sum_distance(const MinDistList &a, const MinDistList &b) -> bool
    {
        double dist_a = 0;
        double dist_b = 0;

        for (const auto i: a.dist) {
            dist_a += i;
        }
        for (const auto i: b.dist) {
            dist_b += i;
        }
        return dist_a < dist_b;
    }

    static auto compare_max_distance(const MinDistList &a, const MinDistList &b) -> bool
    {
        // This function works properly when dvec_a.size() > 0 and dvec_b.size() > 0
        const auto max_dist_a = *std::max_element(a.dist.begin(), a.dist.end());
        const auto max_dist_b = *std::max_element(b.dist.begin(), b.dist.end());

        return max_dist_a < max_dist_b;
    }
};

class InteractionCluster
{
public:
    std::vector<int> atom;
    std::vector<std::vector<int>> cell;
    double distmax;

    InteractionCluster() = default;

    InteractionCluster(std::vector<int> atom_in, std::vector<std::vector<int>> cell_in, const double dist_in) :
        atom(std::move(atom_in)), cell(std::move(cell_in)), distmax(dist_in) {};

    InteractionCluster(std::vector<int> atom_in, std::vector<std::vector<int>> cell_in) :
        atom(std::move(atom_in)), cell(std::move(cell_in)), distmax(0.0) {};

    InteractionCluster(std::vector<int> atom_in) : atom(std::move(atom_in))
    {
        distmax = 0.0;
        cell.clear();
    }

    auto operator<(const InteractionCluster &a) const -> bool
    {
        return lexicographical_compare(atom.begin(), atom.end(), a.atom.begin(), a.atom.end());
    }
};

class Cluster
{
public:
    Cluster();

    ~Cluster();

    auto init(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
              const int periodic_image_conv, const int verbosity, std::unique_ptr<Timer> &timer) -> void;

    auto satisfy_nbody_rule(const int nelem, const int *arr, const int order) const -> bool;

    auto is_incutoff(const int n, const int *atomnumlist, const size_t order, const std::vector<int> &kd) const -> bool;

    auto define(const int maxorder_in, const size_t nkd, const int *nbody_include_in,
                const double *cutoff_radii_in) -> void;

    [[nodiscard]] auto get_maxorder() const -> int;

    [[nodiscard]] auto get_nbody_include() const -> int *;

    [[nodiscard]] auto get_ordername(const unsigned int order) const -> std::string;

    [[nodiscard]] auto get_unique_clusters(const unsigned int order) const -> const std::set<IntList> &;

    [[nodiscard]] auto get_atoms_in_cutoff(const unsigned int order,
                                           const size_t atom_index) const -> const std::vector<int> &;

    [[nodiscard]] auto get_interaction_cluster(const unsigned int order,
                                               const size_t atom_index) const -> const std::set<InteractionCluster> &;

private:
    int maxorder;
    int *nbody_include;
    double ***cutoff_radii;
    std::vector<std::set<IntList>> unique_clusters;
    std::vector<std::vector<std::vector<int>>> atoms_in_cutoff; // List of atoms inside the cutoff radius for each order
    std::vector<std::vector<std::set<InteractionCluster>>> interaction_cluster;
    std::vector<std::vector<PairDistances>> distance_table; // Distance of all pairs (i,j) under the PBC.
    // The distances and the corresponding cell indices are sorted in the ascending order in distance

    auto set_default_variables() -> void;

    auto deallocate_variables() -> void;

    auto get_pairs_of_minimum_distance(const size_t nat, const std::vector<Eigen::MatrixXd> &xc_in, const int *exist,
                                       std::vector<std::vector<PairDistances>> &dist_test_out) const -> void;

    auto generate_interaction_information_by_cutoff(const size_t nat, const size_t natmin, const std::vector<int> &kd,
                                                    const std::vector<std::vector<int>> &map_p2s,
                                                    const double *const *rc,
                                                    std::vector<std::vector<int>> &interaction_list) const -> void;

    auto set_interaction_by_cutoff(const size_t nat, const std::vector<int> &kd, const size_t nat_prim,
                                   const std::vector<std::vector<int>> &map_p2s,
                                   std::vector<std::vector<std::vector<int>>> &interaction_pair_out) const -> void;

    auto print_neighborlist(const size_t, const size_t, const std::vector<std::vector<int>> &, const std::vector<int> &,
                            const std::vector<std::string> &) const -> void;

    auto
    print_interaction_information(const size_t natmin, const std::vector<std::vector<int>> &map_p2s,
                                  const std::vector<int> &kd, const std::vector<std::string> &kdname,
                                  const std::vector<std::vector<std::vector<int>>> &interaction_list) const -> void;

    [[nodiscard]] auto distance(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2) const -> double;

    static auto nbody(const int, const int *) -> int;

    auto calc_interaction_clusters(const size_t natmin, const std::vector<int> &kd,
                                   const std::vector<std::vector<int>> &map_p2s,
                                   const std::vector<Eigen::MatrixXd> &x_image, const int *exist,
                                   const int periodic_image_conv) -> void;

    auto set_interaction_cluster(const int order, const size_t natmin, const std::vector<int> &kd,
                                 const std::vector<std::vector<int>> &map_p2s,
                                 const std::vector<std::vector<int>> &interaction_pair_in,
                                 const std::vector<Eigen::MatrixXd> &x_image, const int *exist,
                                 const int periodic_image_conv,
                                 std::vector<std::set<InteractionCluster>> &interaction_cluster_out) const -> void;

    auto cell_combination(const std::vector<std::vector<int>> &, const size_t, const std::vector<int> &,
                          std::vector<std::vector<int>> &) const -> void;

    auto generate_unique_clusters(const size_t natmin, const std::vector<std::vector<int>> &map_p2s,
                                  std::vector<std::set<IntList>> &cluster_out) const -> void;

    auto check_permutation_symmetry(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                                    int order) -> void;

    auto make_symnum_tran_to_prim(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                                  std::vector<int> &symnum_tran_to_prim) -> void;

    [[nodiscard]] auto is_inprim(const int iat, // atom index in supercell
                                 const size_t natmin, const std::vector<std::vector<int>> &map_p2s) const -> bool;
};
} // namespace ALM_NS

namespace std
{
template <>
struct hash<ALM_NS::IntList>
{
    auto operator()(ALM_NS::IntList const &obj) const -> std::size_t
    {
        hash<int> const hasher;
        size_t seed = 0;
        for (const auto i: obj.iarray) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
} // namespace std
