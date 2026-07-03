/*
 input_setter.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "input_setter.h"
#include <string>
#include "alm.h"
#include "constraint.h"
#include "error.h"
#include "files.h"
#include "memory.h"
#include "optimize.h"
#include "patterndisp.h"
#include "symmetry.h"

using namespace ALM_NS;

InputSetter::InputSetter()
{
    nat_base = 0;
    nkd = 0;
    maxorder = 0;

    is_periodic[0] = 1;
    is_periodic[1] = 1;
    is_periodic[2] = 1;

    lspin = false;
    noncollinear = 0;
    trevsym = 1;
    str_magmom = "";

    nbody_include = nullptr;
    cutoff_radii = nullptr;
}

InputSetter::~InputSetter()
{
    if (nbody_include) {
        deallocate(nbody_include);
    }
    if (cutoff_radii) {
        deallocate(cutoff_radii);
    }
}

auto InputSetter::set_cell_parameter(const Eigen::Matrix3d &lavec_in) -> void
{
    lavec_base_mat = lavec_in;
}

auto InputSetter::set_interaction_vars(const int maxorder_in, const int *nbody_include_in) -> void
{
    maxorder = maxorder_in;
    if (nbody_include) {
        deallocate(nbody_include);
    }
    allocate(nbody_include, maxorder);
    for (auto i = 0; i < maxorder; i++) {
        nbody_include[i] = nbody_include_in[i];
    }
}

auto InputSetter::set_cutoff_radii(const int maxorder_in, const size_t nkd_in,
                                   const std::vector<double> &cutoff_radii_in) -> void
{
    if (cutoff_radii_in.size() != (nkd_in * nkd_in * maxorder_in)) {
        exit("set_cutoff_radii", "Incorrect size of the input array cutoff_radii_in");
    }
    if (cutoff_radii) {
        deallocate(cutoff_radii);
    }
    allocate(cutoff_radii, maxorder_in * nkd_in * nkd_in);
    auto counter = 0;
    for (auto i = 0; i < maxorder_in; i++) {
        for (size_t j = 0; j < nkd_in; j++) {
            for (size_t k = 0; k < nkd_in; k++) {
                cutoff_radii[counter] = cutoff_radii_in[counter];
                ++counter;
            }
        }
    }
}

auto InputSetter::set_general_vars(ALM *alm, const std::string &prefix, const std::string &mode, const int verbosity,
                                   const std::string &str_disp_basis, const int printsymmetry,
                                   const int is_periodic_in[3], const bool trim_dispsign_for_evenfunc,
                                   const int print_hessian, const int print_fcs_alamode, const int print_fc3_shengbte,
                                   const int print_fc4_shengbte, const int print_fc2_qefc, const double tolerance,
                                   const double tolerance_constraint, const std::string &basis_force_constant,
                                   const int nmaxsave, const double fc_zero_threshold, const int compression_level,
                                   const std::string &format_pattern, const std::string &length_unit,
                                   const std::string &force_unit, const std::string &fcs_unit_output) -> void
{
    size_t i;

    alm->set_output_filename_prefix(prefix);
    alm->set_verbosity(verbosity);
    alm->set_print_symmetry(printsymmetry);
    alm->set_symmetry_tolerance(tolerance);
    // Must precede the cell/DFSET/cutoff setters, which convert their input
    // from these units to the internal canonical units (bohr, Ry/bohr).
    alm->set_input_units(length_unit, force_unit);
    alm->set_fcs_unit_output(fcs_unit_output);

    for (i = 0; i < 3; i++) {
        is_periodic[i] = is_periodic_in[i];
    }

    alm->set_fcs_save_flag("hessian", print_hessian);
    alm->set_fcs_save_flag("alamode", print_fcs_alamode);
    alm->set_fcs_save_flag("shengbte", print_fc3_shengbte);
    alm->set_fcs_save_flag("shengbte4", print_fc4_shengbte);
    alm->set_fcs_save_flag("qefc", print_fc2_qefc);
    alm->set_fc_zero_threshold(fc_zero_threshold);
    alm->set_tolerance_constraint(tolerance_constraint);
    alm->set_forceconstant_basis(basis_force_constant);
    alm->set_nmaxsave(nmaxsave);
    alm->set_compression_level(compression_level);
    alm->set_pattern_format(format_pattern);

    if (mode == "suggest") {
        alm->set_displacement_basis(str_disp_basis);
        alm->set_displacement_param(trim_dispsign_for_evenfunc);
    }
}

auto InputSetter::define(ALM *alm) const -> void
{
    alm->define(maxorder, nkd, nbody_include, cutoff_radii);
}


auto InputSetter::set_optimize_vars(ALM *alm, const std::vector<std::vector<double>> &u_train_in,
                                    const std::vector<std::vector<double>> &f_train_in,
                                    const std::vector<std::vector<double>> &u_validation_in,
                                    const std::vector<std::vector<double>> &f_validation_in,
                                    const std::vector<double> &e_train_in,
                                    const std::vector<double> &e_validation_in,
                                    const OptimizerControl &optcontrol_in) const -> void
{
    alm->set_u_train(u_train_in);
    alm->set_f_train(f_train_in);
    alm->set_e_train(e_train_in);
    alm->set_e_validation(e_validation_in);
    alm->set_validation_data(u_validation_in, f_validation_in);
    alm->set_optimizer_control(optcontrol_in);
}

auto InputSetter::set_file_vars(ALM *alm, const DispForceFile &datfile_train_in,
                                const DispForceFile &datfile_validation_in) const -> void
{
    alm->set_datfile_train(datfile_train_in);
    alm->set_datfile_validation(datfile_validation_in);
}

auto InputSetter::set_constraint_vars(ALM *alm, const int constraint_flag, const std::string &rotation_axis,
                                      const std::string &fc2_file, const std::string &fc3_file, const bool fix_harmonic,
                                      const bool fix_cubic, const int ialgo_reduce) const -> void
{
    alm->set_constraint_mode(constraint_flag);
    alm->set_rotation_axis(rotation_axis);
    alm->set_fc_file(2, fc2_file);
    alm->set_fc_fix(2, fix_harmonic);
    alm->set_fc_file(3, fc3_file);
    alm->set_fc_fix(3, fix_cubic);
    const auto use_algebraic_constraint = constraint_flag / 10;
    alm->set_algebraic_constraint(use_algebraic_constraint);
    alm->set_reduction_algo(ialgo_reduce);
}


auto InputSetter::set_atomic_positions(const Eigen::MatrixXd &positions_in) -> void
{
    nat_base = positions_in.col(0).size();
    xcoord_base_mat = positions_in;
}

auto InputSetter::set_element_types(const std::vector<int> &kd_in, const std::vector<std::string> &kdnames_in) -> void
{
    kd_base_vec = kd_in;
    kdnames_vec = kdnames_in;
    nkd = kdnames_in.size();
}

auto InputSetter::set_transformation_matrices(const Eigen::Matrix3d &transmat_super_in,
                                              const Eigen::Matrix3d &transmat_prim_in, const int autoset_primcell_in,
                                              const bool transpose) -> void
{
    // if the input transformation matrices are defined by (a_s, b_s, c_s)^T = M (a_p, b_p, c_p)^T,
    // which is more understandable for human, we need to transpose the matrices to make it consistent
    // with the definition of the lattice vectors used in the code.
    if (transpose) {
        transmat_super = transmat_super_in.transpose();
        transmat_prim = transmat_prim_in.transpose();
    } else {
        transmat_super = transmat_super_in;
        transmat_prim = transmat_prim_in;
    }

    if (autoset_primcell_in) {
        // temporary set the matrix to identity matrix
        transmat_prim = Eigen::Matrix3d::Identity();
    }
    autoset_primcell = autoset_primcell_in;
}

auto InputSetter::set_magnetic_vars(const int lspin_in, const Eigen::MatrixXd &magmom_in, const int noncollinear_in,
                                    const int time_reversal_symm_in) -> void
{
    lspin = lspin_in;
    magmom_base_mat = magmom_in;
    noncollinear = noncollinear_in;
    trevsym = time_reversal_symm_in;
}


auto InputSetter::set_geometric_structure(ALM *alm) -> void
{
    double(*xcoord_base)[3]; // fractional coordinate
    double(*magmom_base)[3];
    int *kd_base;
    double lavec_base[3][3];

    allocate(xcoord_base, nat_base);
    allocate(magmom_base, nat_base);
    allocate(kd_base, nat_base);

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            lavec_base[i][j] = lavec_base_mat(i, j);
        }
    }

    for (auto i = 0; i < nat_base; ++i) {
        for (auto j = 0; j < 3; ++j) {
            xcoord_base[i][j] = xcoord_base_mat(i, j);
            magmom_base[i][j] = magmom_base_mat(i, j);
        }
        kd_base[i] = kd_base_vec[i];
    }

    alm->set_cell(nat_base, lavec_base, xcoord_base, kd_base);
    alm->set_element_names(kdnames_vec);
    alm->set_periodicity(is_periodic);

    deallocate(xcoord_base);
    deallocate(kd_base);

    double transmat_super_tmp[3][3], transmat_prim_tmp[3][3];

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            transmat_super_tmp[i][j] = transmat_super(i, j);
            transmat_prim_tmp[i][j] = transmat_prim(i, j);
        }
    }
    alm->set_transformation_matrices(transmat_super_tmp, transmat_prim_tmp, autoset_primcell);

    alm->set_magnetic_params(nat_base, magmom_base, lspin, noncollinear, trevsym, str_magmom);
    deallocate(magmom_base);
}

auto InputSetter::set_input_var_dict(ALM *alm, const std::map<std::string, std::string> &dict_input_vars) const -> void
{
    alm->set_input_vars(dict_input_vars);
}
