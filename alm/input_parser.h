/*
 input_parser.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "alm.h"
#include "input_setter.h"

#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <Eigen/Core>


namespace ALM_NS
{
class InputParser
{
public:
    InputParser();

    ~InputParser();

    void run(ALM *alm,
             const int narg,
             const char *const *arg);

    std::string get_run_mode() const;

private:
    std::ifstream ifs_input;
    bool from_stdin{};
    std::string mode;
    int maxorder{};
    size_t nat_in{};
    size_t nkd_in{};
    std::unique_ptr<InputSetter> input_setter;

    Eigen::MatrixXd xf_input, xf_poscar;
    Eigen::Matrix3d lavec_input, lavec_poscar;
    Eigen::Matrix3d transmat_to_super, transmat_to_prim;
    int autoset_primcell;

    std::vector<int> atomic_types_input, atomic_types_poscar;
    std::vector<std::string> kdname_vec, kdname_vec_poscar;
    std::map<std::string, std::string> dict_input_vars;

    void parse_input(ALM *alm);

    void parse_general_vars(ALM *alm);

    void parse_transformation_matrix_string(const std::string &string_celldim,
                                            const std::vector<std::string> &celldim_v,
                                            Eigen::Matrix3d &transform_matrix,
                                            const int checkmode_determinant = 0);

    void parse_cell_parameter();

    void parse_atomic_positions();

    void parse_structure_poscar(const std::string &fname_poscar,
                                Eigen::Matrix3d &lavec_out,
                                Eigen::MatrixXd &coordinates_out,
                                std::vector<std::string> &kdname_vec_out,
                                std::vector<int> &atomic_types_out);

    void get_magnetic_params(std::map<std::string, std::string> &dict_input_in,
                             const size_t nat_in,
                             int &lspin_out,
                             Eigen::MatrixXd &magmom_out,
                             int &noncollinear_out,
                             int &time_reversal_symm_out);

    void parse_interaction_vars();

    void parse_cutoff_radii();

    void parse_optimize_vars(ALM *alm);

    [[nodiscard]] int locate_tag(const std::string);

    static void split_str_by_space(const std::string,
                                   std::vector<std::string> &);

    [[nodiscard]] static bool is_endof_entry(const std::string);

    void get_var_dict(const std::vector<std::string> &,
                      std::map<std::string,
                               std::string> &);

    [[nodiscard]] bool is_data_range_consistent(const DispForceFile &datfile_in) const;

    template <typename T>
    void assign_val(T &,
                    const std::string &,
                    std::map<std::string, std::string>);

    void parse_displacement_and_force_files(std::vector<std::vector<double>> &u,
                                            std::vector<std::vector<double>> &f,
                                            DispForceFile &datfile_in) const;
};
}
