/*
 writer.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "cluster.h"
#include "constraint.h"
#include "fcs.h"
#include "files.h"
#include "optimize.h"
#include "patterndisp.h"
#include "symmetry.h"
#include "system.h"

#define H5_USE_EIGEN 1

#include <highfive/H5Easy.hpp>


namespace ALM_NS
{
class AtomProperty
{
public:
    double x, y, z;
    int kind;
    size_t atom, tran;

    AtomProperty() = default;

    AtomProperty(const AtomProperty &other) = default;

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

class SystemInfo
{
public:
    double lattice_vector[3][3];
    std::vector<AtomProperty> atoms;
    size_t nat, natmin, ntran;
    size_t nspecies;

    SystemInfo() = default;
    ;
};

class ForceConstantsWithShifts
{
public:
    std::vector<int> atoms_p, atoms_s;
    std::vector<int> coords;
    std::vector<Eigen::Vector3d> shifts;
    double fcs_value;

    ForceConstantsWithShifts() = default;

    ForceConstantsWithShifts(const std::vector<int> &atoms_p_, const std::vector<int> &atoms_s_,
                             const std::vector<int> &coords_, const std::vector<Eigen::Vector3d> shifts_,
                             const double fcs_value_)
    {
        coords = coords_;
        atoms_p = atoms_p_;
        atoms_s = atoms_s_;
        shifts = shifts_;
        fcs_value = fcs_value_;
    }

    bool operator<(const ForceConstantsWithShifts &obj) const
    {
        std::vector<int> flatternarray(coords.size()), flatternarray_(coords.size());
        for (auto i = 0; i < coords.size(); ++i) {
            flatternarray[i] = 3 * atoms_p[i] + coords[i];
            flatternarray_[i] = 3 * obj.atoms_p[i] + obj.coords[i];
        }
        return std::lexicographical_compare(flatternarray.begin(),
                                            flatternarray.end(),
                                            flatternarray_.begin(),
                                            flatternarray_.end());
    }
};

class Writer
{
public:
    Writer();

    ~Writer();

    auto writeall(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                  const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Constraint> &constraint,
                  const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Optimize> &optimize,
                  const std::unique_ptr<Files> &files, const int verbosity) const -> void;

    auto write_input_vars(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                          const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Displace> &displace,
                          const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                          const std::unique_ptr<Optimize> &optimize, const std::unique_ptr<Files> &files,
                          const std::string &run_mode) const -> void;

    auto write_displacement_pattern(const std::unique_ptr<System> &system, const std::unique_ptr<Cluster> &cluster,
                                    const std::unique_ptr<Displace> &displace, const std::string &prefix,
                                    const int verbosity) const -> void;

    auto save_fcs_with_specific_format(const std::string &fcs_format, const std::unique_ptr<System> &system,
                                       const std::unique_ptr<Symmetry> &symmetry,
                                       const std::unique_ptr<Cluster> &cluster,
                                       const std::unique_ptr<Constraint> &constraint, const std::unique_ptr<Fcs> &fcs,
                                       const std::unique_ptr<Optimize> &optimize, const std::unique_ptr<Files> &files,
                                       const int verbosity) const -> void;

    auto set_fcs_save_flag(const std::string &key_str, const int val) -> void;

    [[nodiscard]] auto get_fcs_save_flag(const std::string &key_str) -> int;

    auto set_filename_fcs(const std::string &filename_in) -> void;

    [[nodiscard]] auto get_filename_fcs() const -> std::string;

    auto set_output_maxorder(const int maxorder) -> void;

    [[nodiscard]] auto get_output_maxorder() const -> int;

    auto set_compression_level(const int level) -> void;

    [[nodiscard]] auto get_compression_level() const -> int;

    auto set_input_vars(const std::map<std::string, std::string> &input_var_dict) -> void;

    [[nodiscard]] auto get_input_var(const std::string &key) const -> std::string;

    auto set_format_patternfile(const std::string &format_name) -> void;

    [[nodiscard]] auto get_format_patternfile() const -> std::string;

private:
    auto write_force_constants(const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                               const std::unique_ptr<Symmetry> &symmetry, const double *fcs_vals, const int verbosity,
                               const std::string &fname_save) const -> void;

    auto save_fcs_alamode_oldformat(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                                    const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                    const std::unique_ptr<Constraint> &constraint, const double *fcs_vals,
                                    const std::string &fname_dfset, const std::string fname_fcs,
                                    const int verbosity) const -> void;

    auto save_fcs_alamode(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                          const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                          const double *fcs_vals, const std::string &fname_fcs, int verbosity) const -> void;

    static auto write_structures_h5(H5Easy::File &file, const Cell &cell, const Spin &spin, const std::string &celltype,
                                    const std::vector<std::string> &kdnames, const size_t ntran,
                                    const std::vector<std::vector<int>> &mapping_info) -> void;

    static auto write_forceconstant_at_given_order_h5(H5Easy::File &file, const int order,
                                                      const std::vector<ForceConstantTable> &fc_cart,
                                                      const std::vector<Eigen::MatrixXd> &x_image,
                                                      const std::vector<Maps> &map_s2tp,
                                                      const std::unique_ptr<Cluster> &cluster,
                                                      const int compression_level = 9) -> void;

    auto write_hessian(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                       const std::unique_ptr<Fcs> &fcs, const std::string &fname_out, const int verbosity) const
        -> void;

    static auto save_fc2_QEfc_format(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                                     const std::unique_ptr<Fcs> &fcs, const std::string fname_out, const int verbosity)
        -> void;

    static auto save_fc3_shengbte_format(const std::unique_ptr<System> &system,
                                         const std::unique_ptr<Symmetry> &symmetry,
                                         const std::unique_ptr<Cluster> &cluster,
                                         const std::unique_ptr<Constraint> &constraint, const std::unique_ptr<Fcs> &fcs,
                                         const std::string &fname_out, const int verbosity) -> void;

    static auto save_fc4_shengbte_format(const std::unique_ptr<System> &system,
                                         const std::unique_ptr<Symmetry> &symmetry,
                                         const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                         const std::string &fname_out, int verbosity) -> void;

    [[nodiscard]] static auto easyvizint(int) -> std::string;

    [[nodiscard]] static auto double2string(double, int nprec = 15) -> std::string;

    std::map<std::string, int> save_format_flags;
    int output_maxorder, compression_level;
    std::string file_fcs, file_hes;
    std::string filename_fcs;
    std::string format_pattern;

    std::map<std::string, std::string> input_variables;
};
} // namespace ALM_NS

// Helper function to combine hash values
template <class T>
inline auto hash_combine(std::size_t &seed, const T &value) -> void
{
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for tuple
namespace std
{
template <typename... Types>
struct hash<std::tuple<Types...>>
{
    auto operator()(const std::tuple<Types...> &t) const -> size_t
    {
        return std::apply(
            [](auto &&...args) {
                size_t seed = 0;
                (hash_combine(seed, args), ...);
                return seed;
            },
            t);
    }
};
} // namespace std
