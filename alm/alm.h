/*
 alm.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <memory>
#include <string>
#include "cluster.h"
#include "constraint.h"
#include "fcs.h"
#include "files.h"
#include "optimize.h"
#include "patterndisp.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"
#include "writer.h"

namespace ALM_NS
{
class ALM
{
public:
    ALM();

    ~ALM();

    std::unique_ptr<Cluster> cluster;

    std::unique_ptr<Fcs> fcs;

    std::unique_ptr<System> system;

    std::unique_ptr<Symmetry> symmetry;

    std::unique_ptr<Optimize> optimize;

    std::unique_ptr<Constraint> constraint;

    std::unique_ptr<Files> files;

    std::unique_ptr<Displace> displace;

    std::unique_ptr<Timer> timer;

    std::unique_ptr<Writer> writer;

    auto set_verbosity(int verbosity_in) -> void;

    [[nodiscard]] auto get_verbosity() const -> int;

    auto set_output_filename_prefix(std::string prefix) const -> void;

    auto set_print_symmetry(int printsymmetry) const -> void;

    auto set_datfile_train(const DispForceFile &dat_in) const -> void;

    auto set_datfile_validation(const DispForceFile &dat_in) const -> void;

    auto set_symmetry_tolerance(double tolerance) const -> void;

    auto set_displacement_param(bool trim_dispsign_for_evenfunc) const -> void;

    auto set_displacement_basis(std::string str_disp_basis) const -> void;

    auto set_periodicity(const int is_periodic[3]) const -> void;

    auto set_cell(const size_t nat, const double lavec[3][3], const double xcoord[][3], const int kind[]) const -> void;

    auto set_element_names(const std::vector<std::string> &kdname_in) const -> void;

    auto set_transformation_matrices(const double transmat_to_super[3][3], const double transmat_to_prim[3][3],
                                     const int autoset_primcell_in) const -> void;

    auto set_magnetic_params(const size_t nat, const double (*magmom)[3], const bool lspin, const int noncollinear,
                             const int trev_sym_mag, const std::string &str_magmom) const -> void;

    auto set_u_train(const std::vector<std::vector<double>> &u) const -> void;

    auto set_f_train(const std::vector<std::vector<double>> &f) const -> void;

    auto set_validation_data(const std::vector<std::vector<double>> &u, const std::vector<std::vector<double>> &f) const
        -> void;

    auto set_optimizer_control(const OptimizerControl &optcontrol_in) const -> void;

    auto set_constraint_mode(const int constraint_flag) const -> void;

    auto set_algebraic_constraint(const int use_algebraic_flag) const -> void;

    auto set_tolerance_constraint(const double tolerance_constraint) const -> void;

    auto set_rotation_axis(const std::string rotation_axis) const -> void;

    auto set_fc_file(const int order, const std::string fc_file) const -> void;

    auto set_fc_fix(const int order, const bool fc_fix) const -> void;

    [[nodiscard]] auto ready_all_constraints() const -> bool;

    auto set_forceconstants_to_fix(const std::vector<std::vector<int>> &intpair_fix,
                                   const std::vector<double> &values_fix) const -> void;

    auto set_sparse_mode(const int sparse_mode) const -> void;

    auto set_forceconstant_basis(const std::string preferred_basis) const -> void;

    [[nodiscard]] auto get_forceconstant_basis() const -> std::string;

    auto set_nmaxsave(const int nmaxsave) const -> void; // NMAXSAVE

    [[nodiscard]] auto get_nmaxsave() const -> int;

    auto set_compression_level(const int level) const -> void; // COMPRESSION

    [[nodiscard]] auto get_compression_level() const -> int;

    //void set_fitting_filenames(std::string dfile,
    //                           std::string ffile) const;
    auto define(const int maxorder, const size_t nkd, const int *nbody_include, const double *cutoff_radii) const
        -> void;

    //int get_ndata_used() const;
    [[nodiscard]] auto get_optimizer_control() const -> OptimizerControl;

    [[nodiscard]] auto get_u_train() const -> std::vector<std::vector<double>>;

    [[nodiscard]] auto get_f_train() const -> std::vector<std::vector<double>>;

    [[nodiscard]] auto get_number_of_data() const -> size_t;

    [[nodiscard]] auto get_nrows_sensing_matrix() const -> size_t;

    [[nodiscard]] auto get_cv_l1_alpha() const -> double;

    [[nodiscard]] auto get_symmetry_tolerance() const -> double;

    [[nodiscard]] auto get_supercell() const -> Cell;

    [[nodiscard]] auto get_kdname() const -> std::vector<std::string>;

    [[nodiscard]] auto get_spin() const -> Spin;

    static auto set_str_magmom(std::string) -> void;

    [[nodiscard]] auto get_str_magmom() const -> std::string;

    [[nodiscard]] auto get_x_image() const -> const std::vector<Eigen::MatrixXd> &;

    [[nodiscard]] auto get_periodicity() const -> int *;

    [[nodiscard]] auto get_atom_mapping_by_pure_translations() const -> const std::vector<std::vector<int>> &;

    [[nodiscard]] auto get_maxorder() const -> int;

    [[nodiscard]] auto get_nbody_include() const -> int *;

    [[nodiscard]] auto get_number_of_displacement_patterns(const int fc_order) const -> size_t; // harmonic=1, ...
    auto get_number_of_displaced_atoms(int *numbers,
                                       int fc_order) const -> void; // harmonic=1, ...
    auto get_displacement_patterns(int *atom_indices, double *disp_patterns,
                                   int fc_order) const -> int;                        // harmonic=1, ...
    [[nodiscard]] auto get_number_of_fc_elements(const int fc_order) const -> size_t; // harmonic=1, ...
    [[nodiscard]] auto get_number_of_irred_fc_elements(const int fc_order) -> size_t; // harmonic=1, ...

    [[nodiscard]] auto get_number_of_fc_origin(const int fc_order, // harmonic = 1
                                               const int permutation) const -> size_t;

    auto get_fc_origin(double *fc_values,
                       int *elem_indices, // (len(fc_value), fc_order) is flatten.
                       int fc_order,      // harmonic=1, ...
                       int permutation = 1) const -> void;


    auto get_fc_irreducible(double *fc_values,
                            int *elem_indices,     // (len(fc_value), fc_order) is flatten.
                            int fc_order) -> void; // harmonic=1, ...


    auto get_fc_all(double *fc_values,
                    int *elem_indices, // (len(fc_value), fc_order) is flatten.
                    int fc_order,      // harmonic=1, ...
                    int permutation = 1) const -> void;

    auto set_fc(double *fc_in) const -> void;

    auto set_fc_zero_threshold(const double threshold_in) const -> void;

    [[nodiscard]] auto get_fc_zero_threshold() const -> double;

    auto get_matrix_elements(double *amat, double *bvec) -> void;

    auto run_optimize() -> int;

    auto run_suggest() const -> void;

    auto init_fc_table() -> void;

    auto save_fc(const std::string &filename, const std::string fc_format, const int maxorder_to_save) const -> void;

    auto set_fcs_save_flag(const std::string &fcs_format, const int val) const -> void;

    auto get_fcs_save_flag(const std::string &fcs_format) const -> int;

    auto set_input_vars(const std::map<std::string, std::string> &input_var_dict) const -> void;

    [[nodiscard]] auto get_input_var(const std::string &key) const -> std::string;

    auto set_pattern_format(const std::string &format_name) const -> void;

    [[nodiscard]] auto get_format_pattern() const -> std::string;

private:
    int verbosity;
    bool structure_initialized;
    bool initialized_constraint_class;
    std::ofstream *ofs_alm;
    std::streambuf *coutbuf;

    auto init_instances() -> void;
};
} // namespace ALM_NS
