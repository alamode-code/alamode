/*
 input_setter.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "alm.h"
#include <string>

namespace ALM_NS
{
class InputSetter
{
public:
    InputSetter();

    ~InputSetter();

    void set_cell_parameter(const double a,
                            const double lavec_in[3][3]);

    void set_cell_parameter(const Eigen::Matrix3d &lavec_in);

    void set_atomic_positions(const size_t nat_in,
                              const int *kd_in,
                              const double (*xcoord_in)[3]);

    void set_atomic_positions(const Eigen::MatrixXd &positions_in);

    void set_element_types(const std::vector<int> &kd_in,
                           const std::vector<std::string> &kdnames_in);

    void set_transformation_matrices(const Eigen::Matrix3d &transmat_super_in,
                                     const Eigen::Matrix3d &transmat_prim_in,
                                     const int autoset_primcell_in,
                                     const bool transpose = false);

    void set_magnetic_vars(const int lspin_in,
                           const Eigen::MatrixXd &magmom_in,
                           const int noncollinear_in,
                           const int time_reversal_symm_in);

    void set_geometric_structure(ALM *alm);

    void set_interaction_vars(const int maxorder_in,
                              const int *nbody_include_in);

    void set_cutoff_radii(const int maxorder_in,
                          const size_t nkd_in,
                          const std::vector<double> &cutoff_radii_in);

    void define(ALM *alm) const;

    void set_general_vars(ALM *alm,
                          const std::string &prefix,
                          const std::string &mode,
                          int verbosity,
                          const std::string &str_disp_basis,
                          int printsymmetry,
                          const int is_periodic_in[3],
                          bool trim_dispsign_for_evenfunc,
                          int print_hessian,
                          int print_fcs_alamode,
                          int print_fc3_shengbte,
                          int print_fc4_shengbte,
                          int print_fc2_qefc,
                          double tolerance,
                          double tolerance_constraint,
                          const std::string &basis_force_constant,
                          const int nmaxsave,
                          const double fc_zero_threshold,
                          const int compression_level,
                          const std::string &format_pattern);

    void set_optimize_vars(ALM *alm,
                           const std::vector<std::vector<double>> &u_train_in,
                           const std::vector<std::vector<double>> &f_train_in,
                           const std::vector<std::vector<double>> &u_validation_in,
                           const std::vector<std::vector<double>> &f_validation_in,
                           const OptimizerControl &optcontrol_in) const;

    void set_file_vars(ALM *alm,
                       const DispForceFile &datfile_train_in,
                       const DispForceFile &datfile_validation_in) const;

    void set_constraint_vars(ALM *alm,
                             int constraint_flag,
                             const std::string &rotation_axis,
                             const std::string &fc2_file,
                             const std::string &fc3_file,
                             bool fix_harmonic,
                             bool fix_cubic) const;

    void set_input_var_dict(ALM *alm, const std::map<std::string, std::string> &dict_in) const;

private:
    size_t nat_base, nkd;
    Eigen::Matrix3d lavec_base_mat;
    Eigen::MatrixXd xcoord_base_mat;
    std::vector<int> kd_base_vec;
    std::vector<std::string> kdnames_vec;
    int is_periodic[3];

    bool lspin;
    Eigen::MatrixXd magmom_base_mat;
    int noncollinear;
    int trevsym;
    std::string str_magmom;

    int maxorder;
    int *nbody_include;
    double *cutoff_radii;

    Eigen::Matrix3d transmat_super, transmat_prim;
    int autoset_primcell;
};
}
