/*
 input_parser.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "input_parser.h"
#include <Eigen/Core>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <sys/stat.h>
#include "alm.h"
#include "error.h"
#include "files.h"
#include "input_setter.h"
#include "memory.h"
#include "optimize.h"

using namespace ALM_NS;

InputParser::InputParser()
{
    input_setter = std::make_unique<InputSetter>();
}

InputParser::~InputParser()
{}

auto InputParser::run(ALM *alm, const int narg, const char *const *arg) -> void
{
    if (narg == 1) {
        from_stdin = true;
    } else {
        from_stdin = false;
        ifs_input.open(arg[1], std::ios::in);
        if (!ifs_input) {
            std::cout << "No such file or directory: " << arg[1] << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    parse_input(alm);
}

auto InputParser::get_run_mode() const -> std::string
{
    return mode;
}

auto InputParser::parse_displacement_and_force_files(std::vector<std::vector<double>> &u,
                                                     std::vector<std::vector<double>> &f,
                                                     DispForceFile &datfile_in) const -> void
{
    int nrequired;

    const auto nat = nat_in * static_cast<int>(transmat_to_super.determinant());

    if (datfile_in.ndata == 0) {
        nrequired = -1;
    } else {
        // Total number of data entries (displacement + force)
        nrequired = 6 * nat * datfile_in.ndata;
    }

    std::vector<double> value_arr;
    std::string line;
    double val;
    auto nline_u = 0;

    // Open the target file and copy the data to 1D temporary vector
    std::ifstream ifs_data;
    ifs_data.open(datfile_in.filename.c_str(), std::ios::in);
    if (!ifs_data) exit("openfiles", "cannot open DFSET file");
    auto reach_end = false;
    while (std::getline(ifs_data >> std::ws, line)) {
        if (line[0] != '#') {

            std::istringstream iss(line);

            while (iss >> val) {
                value_arr.push_back(val);
                ++nline_u;

                if (nline_u == nrequired) {
                    reach_end = true;
                    break;
                }
            }
        }

        if (reach_end) break;
    }
    ifs_data.close();

    // Check if the length of the vector is correct.
    // Also, estimate ndata if it is not set.
    const auto n_entries = value_arr.size();

    if (nrequired == -1) {
        if (n_entries % (6 * nat) == 0) {
            datfile_in.ndata = n_entries / (6 * nat);
        } else {
            exit("parse_displacement_and_force_files", "The number of lines in DFSET is indivisible by NAT");
        }
    } else {
        if (n_entries < nrequired) {
            exit("parse_displacement_and_force_files",
                 "The number of lines in DFSET is too small for the given NDATA = ",
                 datfile_in.ndata);
        }
    }

    if (datfile_in.nstart == 0) datfile_in.nstart = 1;
    if (datfile_in.nend == 0) datfile_in.nend = datfile_in.ndata;

    // Copy the data into 2D array
    const auto ndata_used = datfile_in.nend - datfile_in.nstart + 1 - datfile_in.skip_e + datfile_in.skip_s;

    u.resize(ndata_used, std::vector<double>(3 * nat));
    f.resize(ndata_used, std::vector<double>(3 * nat));

    auto idata = 0;
    for (size_t i = 0; i < datfile_in.ndata; ++i) {
        if (i < datfile_in.nstart - 1) continue;
        if (i >= datfile_in.skip_s - 1 && i < datfile_in.skip_e - 1) continue; // When skip_s == skip_e, skip nothing.
        if (i > datfile_in.nend - 1) break;

        for (auto j = 0; j < nat; ++j) {
            for (auto k = 0; k < 3; ++k) {
                u[idata][3 * j + k] = value_arr[6 * nat * i + 6 * j + k];
                f[idata][3 * j + k] = value_arr[6 * nat * i + 6 * j + k + 3];
            }
        }
        ++idata;
    }
    value_arr.clear();
}

auto InputParser::parse_input(ALM *alm) -> void
{
    // The order of calling methods in this method is important.
    // Since following methods rely on variables those already
    // parsed.
    // Those below are set as the private class variables. See input_parser.h.
    //  std::string mode;
    //  int maxorder;
    //  int nat_base;
    //  int nkd;

    // Parse &general field
    if (!locate_tag("&general")) {
        exit("parse_input", "&general entry not found in the input file");
    }
    parse_general_vars(alm);

    if (dict_input_vars["STRUCTURE_FILE"].empty()) {
        // Read &cell and &position fields and get structure information
        if (!locate_tag("&cell")) {
            exit("parse_input", "&cell entry not found in the input file");
        }
        parse_cell_parameter();

        if (!locate_tag("&position")) {
            exit("parse_input", "&position entry not found in the input file");
        }
        parse_atomic_positions();

        input_setter->set_cell_parameter(lavec_input);
        input_setter->set_atomic_positions(xf_input);
        input_setter->set_element_types(atomic_types_input, kdname_vec);
        nat_in = atomic_types_input.size();
        nkd_in = kdname_vec.size();
    } else {
        // If STRUCTURE_FILE is given, use the structure parameters defined in this file.
        // In this case, the &cell and &position entries are ignored.
        lavec_poscar /= Bohr_in_Angstrom;
        input_setter->set_cell_parameter(lavec_poscar);
        input_setter->set_atomic_positions(xf_poscar);
        input_setter->set_element_types(atomic_types_poscar, kdname_vec_poscar);
        nat_in = atomic_types_poscar.size();
        nkd_in = kdname_vec_poscar.size();
        kdname_vec = kdname_vec_poscar; // Copy for use in parse_cutoff_radii
    }

    input_setter->set_transformation_matrices(transmat_to_super, transmat_to_prim, autoset_primcell, true);

    int noncollinear, time_reversal_symm, lspin;
    Eigen::MatrixXd magmom_vec;

    get_magnetic_params(dict_input_vars, nat_in, lspin, magmom_vec, noncollinear, time_reversal_symm);

    input_setter->set_magnetic_vars(lspin, magmom_vec, noncollinear, time_reversal_symm);

    // This method should be called after the structural and magnetic parameters are set.
    input_setter->set_geometric_structure(alm);

    if (!locate_tag("&interaction")) {
        exit("parse_input", "&interaction entry not found in the input file");
    }
    parse_interaction_vars();

    if (!locate_tag("&cutoff")) {
        exit("parse_input", "&cutoff entry not found in the input file");
    }
    parse_cutoff_radii();
    input_setter->define(alm);

    if (mode == "optimize") {
        if (!locate_tag("&optimize")) {
            exit("parse_input", "&optimize entry not found in the input file");
        }
        parse_optimize_vars(alm);
    }

    input_setter->set_input_var_dict(alm, dict_input_vars);
}

auto InputParser::parse_general_vars(ALM *alm) -> void
{
    // Parse variables defined in the &general field of an input file.
    size_t i;
    std::string str_tmp, str_disp_basis, basis_force_constant;
    int printsymmetry, is_periodic[3];
    auto trim_dispsign_for_evenfunc = true;
    int print_hessian, print_fcs_alamode, print_fc2_qefc;
    int print_fc3_shengbte, print_fc4_shengbte;
    int noncollinear, trevsym;
    double **magmom, magmag{0.0};
    double tolerance;
    double tolerance_constraint;
    double fc_zero_threshold;
    int verbosity, nmaxsave, compression_level;

    std::vector<std::string> kdname_v, periodic_v;
    std::vector<std::string> supercell_v, primcell_v;
    std::string structure_file{};
    std::string format_pattern;

    const std::vector<std::string> input_list{
        "PREFIX",       "MODE",           "NAT",         "NKD",           "KD",          "PERIODIC",
        "PRINTSYM",     "TOLERANCE",      "DBASIS",      "TRIMEVEN",      "VERBOSITY",   "MAGMOM",
        "NONCOLLINEAR", "TREVSYM",        "HESSIAN",     "TOL_CONST",     "FCSYM_BASIS", "NMAXSAVE",
        "FC3_SHENGBTE", "FC4_SHENGBTE",   "FC2_QEFC",    "FCS_ALAMODE",   "FC_ZERO_THR", "SUPERCELL",
        "PRIMCELL",     "STRUCTURE_FILE", "COMPRESSION", "FORMAT_PATTERN"};
    std::vector<std::string> no_defaults{"PREFIX", "MODE"};
    std::map<std::string, std::string> general_var_dict;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    // parse all input parameters in &general field
    get_var_dict(input_list, general_var_dict);

    for (const auto &it: general_var_dict) {
        dict_input_vars.insert(it);
    }

    // Parse PREFIX and MODE (which are mandatory to be given in inputs)
    for (const auto &it: no_defaults) {
        if (general_var_dict.find(it) == general_var_dict.end()) {
            exit("parse_general_vars", "The following variable is not found in &general input region: ", it.c_str());
        }
    }
    const auto prefix = general_var_dict["PREFIX"];
    mode = general_var_dict["MODE"];
    std::transform(mode.begin(), mode.end(), mode.begin(), tolower);
    if (mode != "suggest" && mode != "optimize" && mode != "opt") {
        exit("parse_general_vars", "Invalid MODE variable");
    }
    if (mode == "opt") mode = "optimize";

    // We first check if STRUCTURE_FILE field is empty or not.
    // If not, the structure data is read from the given file (in a POSCAR format)
    // and copy the lattice vectors, element types, and coordinates to
    // the corresponding private variables of this class.
    if (!general_var_dict["STRUCTURE_FILE"].empty()) {
        structure_file = general_var_dict["STRUCTURE_FILE"];
        struct stat buffer;
        if (stat(structure_file.c_str(), &buffer) != 0) {
            const std::string str_message =
                "STRUCTURE_FILE is given but the target file (" + structure_file + ") does not exist.";
            exit("parse_general_vars", str_message.c_str());
        }
    }
    if (!structure_file.empty()) {
        parse_structure_poscar(structure_file, lavec_poscar, xf_poscar, kdname_vec_poscar, atomic_types_poscar);
    }

    // Parse NAT, NKD, and KD variables if they are given.
    if (!general_var_dict["NAT"].empty()) {
        assign_val(nat_in, "NAT", general_var_dict);
    }
    if (!general_var_dict["NKD"].empty()) {
        assign_val(nkd_in, "NKD", general_var_dict);
    }
    if (!general_var_dict["KD"].empty()) {
        split_str_by_space(general_var_dict["KD"], kdname_vec);
    }
    if (nkd_in > 0) {
        if (kdname_vec.size() != nkd_in) {
            exit("parse_general_vars", "The number of entries for KD is inconsistent with NKD");
        }
    } else {
        nkd_in = kdname_vec.size();
    }


    if (general_var_dict["VERBOSITY"].empty()) {
        verbosity = 1;
    } else {
        assign_val(verbosity, "VERBOSITY", general_var_dict);
    }

    if (general_var_dict["PRINTSYM"].empty()) {
        printsymmetry = 0;
    } else {
        assign_val(printsymmetry, "PRINTSYM", general_var_dict);
    }

    // PERIODIC tag
    split_str_by_space(general_var_dict["PERIODIC"], periodic_v);
    if (periodic_v.empty()) {
        for (i = 0; i < 3; ++i) {
            is_periodic[i] = 1;
        }
    } else if (periodic_v.size() == 3) {
        for (i = 0; i < 3; ++i) {
            try {
                is_periodic[i] = boost::lexical_cast<int>(periodic_v[i]);
            } catch (std::exception &e) {
                std::cout << e.what() << '\n';
                exit("parse_general_vars", "The PERIODIC tag must be a set of integers.");
            }
        }
    } else {
        exit("parse_general_vars", "Invalid number of entries for PERIODIC");
    }

    // parse SUPERCELL
    split_str_by_space(general_var_dict["SUPERCELL"], supercell_v);

    parse_transformation_matrix_string("SUPERCELL", supercell_v, transmat_to_super);
    // parse PRIMCELL
    split_str_by_space(general_var_dict["PRIMCELL"], primcell_v);
    autoset_primcell = 0;
    if (primcell_v.size() == 1) {
        if (std::tolower(primcell_v[0][0]) == 'a') {
            autoset_primcell = 1;
        }
    }
    if (!autoset_primcell) {
        parse_transformation_matrix_string("PRIMCELL", primcell_v, transmat_to_prim, 1);
    }

    if (general_var_dict["TOLERANCE"].empty()) {
        tolerance = 1.0e-3;
    } else {
        assign_val(tolerance, "TOLERANCE", general_var_dict);
    }
    if (general_var_dict["TOL_CONST"].empty()) {
        tolerance_constraint = 1.0e-6;
    } else {
        assign_val(tolerance_constraint, "TOL_CONST", general_var_dict);
    }

    if (general_var_dict["FCSYM_BASIS"].empty()) {
        basis_force_constant = "Lattice";
    } else {
        basis_force_constant = general_var_dict["FCSYM_BASIS"];
        boost::to_lower(basis_force_constant);

        if (basis_force_constant[0] != 'c' && basis_force_constant[0] != 'l') {
            exit("parse_general_vars", "Invalid FCSYM_BASIS.", basis_force_constant.c_str());
        }
    }

    if (general_var_dict["NMAXSAVE"].empty()) {
        nmaxsave = 5;
    } else {
        assign_val(nmaxsave, "NMAXSAVE", general_var_dict);
    }
    if (general_var_dict["COMPRESSION"].empty()) {
        compression_level = 1;
    } else {
        assign_val(compression_level, "COMPRESSION", general_var_dict);
    }
    if (general_var_dict["HESSIAN"].empty()) {
        print_hessian = 0;
    } else {
        assign_val(print_hessian, "HESSIAN", general_var_dict);
    }
    if (general_var_dict["FCS_ALAMODE"].empty()) {
        print_fcs_alamode = 1;
    } else {
        assign_val(print_fcs_alamode, "FCS_ALAMODE", general_var_dict);
    }
    if (general_var_dict["FC3_SHENGBTE"].empty()) {
        print_fc3_shengbte = 0;
    } else {
        assign_val(print_fc3_shengbte, "FC3_SHENGBTE", general_var_dict);
    }
    if (general_var_dict["FC4_SHENGBTE"].empty()) {
        print_fc4_shengbte = 0;
    } else {
        assign_val(print_fc4_shengbte, "FC4_SHENGBTE", general_var_dict);
    }
    if (general_var_dict["FC2_QEFC"].empty()) {
        print_fc2_qefc = 0;
    } else {
        assign_val(print_fc2_qefc, "FC2_QEFC", general_var_dict);
    }
    if (general_var_dict["FC_ZERO_THR"].empty()) {
        fc_zero_threshold = eps12;
    } else {
        assign_val(fc_zero_threshold, "FC_ZERO_THR", general_var_dict);
    }
    if (!general_var_dict["FORMAT_PATTERN"].empty()) {
        assign_val(format_pattern, "FORMAT_PATTERN", general_var_dict);
    } else {
        format_pattern = "yaml";
    }

    if (mode == "suggest") {
        if (general_var_dict["DBASIS"].empty()) {
            str_disp_basis = "Cart";
        } else {
            str_disp_basis = general_var_dict["DBASIS"];
        }
        std::transform(str_disp_basis.begin(), str_disp_basis.end(), str_disp_basis.begin(), toupper);
        if ((str_disp_basis[0] != 'C') && (str_disp_basis[0] != 'F')) {
            exit("parse_general_vars", "Invalid DBASIS");
        }

        if (!general_var_dict["TRIMEVEN"].empty()) {
            assign_val(trim_dispsign_for_evenfunc, "TRIMEVEN", general_var_dict);
        }
    }

    input_setter->set_general_vars(alm,
                                   prefix,
                                   mode,
                                   verbosity,
                                   str_disp_basis,
                                   printsymmetry,
                                   is_periodic,
                                   trim_dispsign_for_evenfunc,
                                   print_hessian,
                                   print_fcs_alamode,
                                   print_fc3_shengbte,
                                   print_fc4_shengbte,
                                   print_fc2_qefc,
                                   tolerance,
                                   tolerance_constraint,
                                   basis_force_constant,
                                   nmaxsave,
                                   fc_zero_threshold,
                                   compression_level,
                                   format_pattern);

    kdname_v.clear();
    periodic_v.clear();
    no_defaults.clear();
    general_var_dict.clear();
}

auto InputParser::parse_transformation_matrix_string(const std::string &string_celldim,
                                                     const std::vector<std::string> &celldim_v,
                                                     Eigen::Matrix3d &transform_matrix, const int checkmode_determinant)
    -> void
{
    // Split the SUPERCELL or PRIMCELL entry by space and convert the data into
    // 3x3 double matrix in the Eigen::Matrix3d type.
    const Eigen::Matrix3d mat_identity = Eigen::Matrix3d::Identity();

    if (celldim_v.empty()) {
        // if not given, use identity matrix
        transform_matrix = mat_identity;
    } else if (celldim_v.size() == 1) {
        std::vector<std::string> str_vec;
        boost::split(str_vec, celldim_v[0], boost::is_any_of("/"));

        if (str_vec.size() == 2) {
            transform_matrix = std::stod(str_vec[0]) / std::stod(str_vec[1]) * mat_identity;
        } else {
            transform_matrix = std::stod(str_vec[0]) * mat_identity;
        }

    } else if (celldim_v.size() == 3) {
        transform_matrix = mat_identity;
        for (auto i = 0; i < 3; ++i) {
            std::vector<std::string> str_vec;
            boost::split(str_vec, celldim_v[i], boost::is_any_of("/"));

            if (str_vec.size() == 2) {
                transform_matrix(i, i) = std::stod(str_vec[0]) / std::stod(str_vec[1]);
            } else {
                transform_matrix(i, i) = std::stod(str_vec[0]);
            }
        }
    } else if (celldim_v.size() == 9) {
        auto k = 0;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                std::vector<std::string> str_vec;
                boost::split(str_vec, celldim_v[k++], boost::is_any_of("/"));
                if (str_vec.size() == 2) {
                    transform_matrix(i, j) = std::stod(str_vec[0]) / std::stod(str_vec[1]);
                } else {
                    transform_matrix(i, j) = std::stod(str_vec[0]);
                }
            }
        }
    } else {
        std::string str_message =
            "Invalid number of entries for " + string_celldim + ".\n" + "The size should be either 1, 3, or 9.";
        exit("parse_transformation_matrix", str_message.c_str());
    }

    // Check if the transformation matrix is

    const double det = transform_matrix.determinant();

    if (std::abs(det) < eps) {
        std::string str_message = "The input matrix in " + string_celldim + " is singular.\n";
        exit("parse_transformation_matrix", str_message.c_str());
    }

    if (checkmode_determinant == 0) {
        // check if the determinant of the transformation matrix is integer for SUPERCELL
        if (std::abs(det - static_cast<double>(nint(det))) > eps) {
            std::string str_message = "The determinant of the matrix in " + string_celldim + " is not an integer.\n";
            exit("parse_transformation_matrix", str_message.c_str());
        }
    } else {
        // check if the determinant of the transformation matrix is (1/a) for PRIMCELL,
        // where a is an integer.

        if (std::abs(1 / det - static_cast<double>(nint(1 / det))) > eps) {
            std::string str_message =
                "The determinant of the inverse matrix of " + string_celldim + " is not an integer.\n";
            exit("parse_transformation_matrix", str_message.c_str());
        }
    }

    if (det < 0.0) {
        std::string str_message = "The determinant of the matrix in " + string_celldim + " is negative.\n";
        warn("parse_transformation_matrix", str_message.c_str());
    }
}

auto InputParser::parse_cell_parameter() -> void
{
    auto a = 0.0;
    double lavec_tmp[3][3];
    std::string line;
    std::string line_wo_comment, line_tmp;
    std::vector<std::string> line_vec, line_split;
    std::string::size_type pos_first_comment_tag;

    line_vec.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_if(line_wo_comment, boost::is_any_of("\t\n\r "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }

    } else {
        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_if(line_wo_comment, boost::is_any_of("\t\n\r "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }
    }

    if (line_vec.size() != 4) {
        exit("parse_cell_parameter",
             "Too few or too much lines for the &cell field.\n"
             "The number of valid lines for the &cell field should be 4.");
    }

    for (auto i = 0; i < 4; ++i) {

        line = line_vec[i];
        boost::split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

        if (i == 0) {
            // Lattice factor a
            if (line_split.size() == 1) {
                a = boost::lexical_cast<double>(line_split[0]);
            } else {
                exit("parse_cell_parameter", "Unacceptable format for &cell field.");
            }

        } else {
            // Lattice vectors a1, a2, a3
            if (line_split.size() == 3) {
                for (auto j = 0; j < 3; ++j) {
                    lavec_tmp[j][i - 1] = boost::lexical_cast<double>(line_split[j]);
                }
            } else {
                exit("parse_cell_parameter", "Unacceptable format for &cell field.");
            }
        }
    }

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            lavec_input(i, j) = a * lavec_tmp[i][j];
        }
    }
}

auto InputParser::parse_atomic_positions() -> void
{
    std::string line, line_wo_comment;
    std::string str_tmp;
    std::string::size_type pos_first_comment_tag;
    std::vector<std::string> str_v, pos_line;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    str_v.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_if(line_wo_comment, boost::is_any_of("\t\n\r "));
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            str_v.push_back(line_wo_comment);
        }

    } else {

        while (std::getline(ifs_input, line)) {

            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_if(line_wo_comment, boost::is_any_of("\t\n\r "));
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            str_v.push_back(line_wo_comment);
        }
    }

    if (nat_in > 0) {
        if (str_v.size() != nat_in) {
            exit("parse_atomic_positions", "The number of entries for atomic positions should be NAT");
        }
    } else {
        nat_in = str_v.size();
    }

    atomic_types_input.resize(nat_in);
    xf_input.resize(nat_in, 3);

    for (size_t i = 0; i < nat_in; ++i) {

        split_str_by_space(str_v[i], pos_line);

        if (pos_line.size() == 4) {
            try {
                atomic_types_input[i] = boost::lexical_cast<int>(pos_line[0]);
            } catch (std::exception &e) {
                std::cout << e.what() << '\n';
                exit("parse_atomic_positions", "Invalid entry for the &position field at line ", i + 1);
            }

            for (auto j = 0; j < 3; ++j) {
                xf_input(i, j) = boost::lexical_cast<double>(pos_line[j + 1]);
            }

        } else {
            exit("parse_atomic_positions", "Bad format for &position region");
        }
    }

    pos_line.clear();
    str_v.clear();
}


auto InputParser::parse_structure_poscar(const std::string &fname_poscar, Eigen::Matrix3d &lavec_out,
                                         Eigen::MatrixXd &coordinates_out, std::vector<std::string> &kdname_vec_out,
                                         std::vector<int> &atomic_types_out) -> void
{
    // Parse structure data from a file in the POSCAR format and
    // copy the read data to the corresponding private variables of
    // this class.

    std::ifstream ifs;
    std::string dummy;
    double aval;

    ifs.open(fname_poscar, std::ios::in);

    // Skip first row and parse lattice vectors from the 2nd-5th rows
    std::getline(ifs, dummy);
    ifs >> aval;
    ifs >> lavec_out(0, 0) >> lavec_out(1, 0) >> lavec_out(2, 0);
    ifs >> lavec_out(0, 1) >> lavec_out(1, 1) >> lavec_out(2, 1);
    ifs >> lavec_out(0, 2) >> lavec_out(1, 2) >> lavec_out(2, 2);
    lavec_out *= aval;
    ifs.ignore(); // ignore newline char

    std::string str_species;
    std::vector<std::string> species_v;
    std::set<std::string> unique_species;
    std::set<std::string>::iterator it_set;
    std::map<std::string, int> map_kindname_to_index;

    std::getline(ifs, str_species);
    split_str_by_space(str_species, species_v);

    int counter = 0;
    for (auto &it: species_v) {
        it_set = unique_species.find(it);
        if (it_set == unique_species.end()) {
            unique_species.insert(it);
            kdname_vec_out.push_back(it);
            map_kindname_to_index[it] = counter++;
        }
    }

    std::string str_num_kinds;
    std::vector<std::string> num_kinds_split;
    std::getline(ifs, str_num_kinds);
    split_str_by_space(str_num_kinds, num_kinds_split);

    if (num_kinds_split.size() != species_v.size()) {
        exit("parse_structure_poscar", "The numbers for ion species and numbers are inconsistent.");
    }

    std::vector<int> num_kinds_vec;
    for (auto &it: num_kinds_split) {
        num_kinds_vec.push_back(std::stoi(it));
    }
    counter = 0;
    atomic_types_out.clear();
    for (auto i = 0; i < num_kinds_vec.size(); ++i) {
        const auto ikd_now = map_kindname_to_index[species_v[i]];
        for (auto j = 0; j < num_kinds_vec[i]; ++j) {
            atomic_types_out.push_back(ikd_now + 1);
        }
    }

    std::getline(ifs, dummy);
    std::transform(dummy.begin(), dummy.end(), dummy.begin(), tolower);

    if (dummy[0] == 's') {
        // Selective dynamics
        // read the next line
        std::getline(ifs, dummy);
        std::transform(dummy.begin(), dummy.end(), dummy.begin(), tolower);
    }
    bool structure_given_in_cartesian = false;
    if (dummy[0] == 'k' || dummy[0] == 'c') {
        structure_given_in_cartesian = true;
    }

    const auto nat_tmp = std::accumulate(num_kinds_vec.begin(), num_kinds_vec.end(), 0);
    coordinates_out.resize(nat_tmp, 3);

    std::vector<std::string> coordinate_entry_split;
    // Read coordinates
    for (auto i = 0; i < nat_tmp; ++i) {
        std::getline(ifs, dummy);
        split_str_by_space(dummy, coordinate_entry_split);
        if (coordinate_entry_split.size() < 3) {
            exit("parse_structure_poscar", "The number of entries for the position is too few.");
        }
        for (auto j = 0; j < 3; ++j) {
            coordinates_out(i, j) = std::stod(coordinate_entry_split[j]);
        }
    }
    ifs.close();

    // If the coordinates are given in Cartesian frame, convert them to the fractional basis.
    if (structure_given_in_cartesian) {
        coordinates_out = coordinates_out * lavec_out.transpose().inverse();
    }
}

auto InputParser::get_magnetic_params(std::map<std::string, std::string> &dict_input_in, const size_t nat_in,
                                      int &lspin_out, Eigen::MatrixXd &magmom_out, int &noncollinear_out,
                                      int &time_reversal_symm_out) -> void
{
    double magmag{0.0};
    std::vector<std::string> magmom_v, str_split;
    size_t icount, ncount;

    magmom_out = Eigen::MatrixXd::Zero(nat_in, 3);
    lspin_out = 0;

    if (dict_input_in["NONCOLLINEAR"].empty()) {
        noncollinear_out = 0;
    } else {
        assign_val(noncollinear_out, "NONCOLLINEAR", dict_input_in);
    }
    if (dict_input_in["TREVSYM"].empty()) {
        time_reversal_symm_out = 1;
    } else {
        assign_val(time_reversal_symm_out, "TREVSYM", dict_input_in);
    }
    if (!dict_input_in["MAGMOM"].empty()) {
        lspin_out = 1;

        if (noncollinear_out) {
            icount = 0;
            split_str_by_space(dict_input_in["MAGMOM"], magmom_v);
            for (const auto &it: magmom_v) {
                if (it.find('*') != std::string::npos) {
                    exit("get_magnetic_params", "Wild card '*' is not supported when NONCOLLINEAR = 1.");
                } else {
                    magmag = boost::lexical_cast<double>(it);
                    if (icount / 3 >= nat_in) {
                        exit("get_magnetic_params", "Too many entries for MAGMOM.");
                    }
                    magmom_out(icount / 3, icount % 3) = magmag;
                    ++icount;
                }
            }

            if (icount != 3 * nat_in) {
                exit("get_magnetic_params", "Number of entries for MAGMOM must be 3*NAT when NONCOLLINEAR = 1.");
            }
        } else {
            icount = 0;
            split_str_by_space(dict_input_in["MAGMOM"], magmom_v);
            for (const auto &it: magmom_v) {

                if (it.find('*') != std::string::npos) {
                    if (it == "*") {
                        exit("get_magnetic_params", "Please place '*' without space for the MAGMOM-tag.");
                    }
                    boost::split(str_split, it, boost::is_any_of("*"));
                    if (str_split.size() != 2) {
                        exit("get_magnetic_params", "Invalid format for the MAGMOM-tag.");
                    } else {
                        if (str_split[0].empty() || str_split[1].empty()) {
                            exit("get_magnetic_params", "Please place '*' without space for the MAGMOM-tag.");
                        }
                        ncount = 0;
                        try {
                            magmag = boost::lexical_cast<double>(str_split[1]);
                            ncount = static_cast<size_t>(boost::lexical_cast<double>(str_split[0]));
                        } catch (std::exception) {
                            exit("get_magnetic_params", "Bad format for MAGMOM.");
                        }

                        for (auto i = icount; i < icount + ncount; ++i) {
                            if (i >= nat_in) {
                                exit("get_magnetic_params", "Too many entries for MAGMOM.");
                            }
                            magmom_out(i, 2) = magmag;
                        }
                        icount += ncount;
                    }

                } else {
                    magmag = boost::lexical_cast<double>(it);
                    if (icount == nat_in) {
                        icount = 0;
                        break;
                    }
                    magmom_out(icount++, 2) = magmag;
                }
            }
            if (icount != nat_in) {
                exit("get_magnetic_params", "Number of entries for MAGMOM must be NAT.");
            }
        }
    }
}

auto InputParser::parse_interaction_vars() -> void
{
    int i;
    int *nbody_include;

    std::vector<std::string> nbody_v;
    const std::vector<std::string> input_list{"NORDER", "NBODY"};
    std::vector<std::string> no_defaults{"NORDER"};
    std::map<std::string, std::string> interaction_var_dict;


    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, interaction_var_dict);

    for (const auto &it: no_defaults) {
        if (interaction_var_dict.find(it) == interaction_var_dict.end()) {
            exit("parse_interaction_vars",
                 "The following variable is not found in &interaction input region: ",
                 it.c_str());
        }
    }

    for (const auto &it: interaction_var_dict) {
        dict_input_vars.insert(it);
    }

    assign_val(maxorder, "NORDER", interaction_var_dict);
    if (maxorder < 1) exit("parse_interaction_vars", "maxorder has to be a positive integer");

    allocate(nbody_include, maxorder);

    boost::split(nbody_v, interaction_var_dict["NBODY"], boost::is_space());

    if (nbody_v[0].empty()) {
        for (i = 0; i < maxorder; ++i) {
            nbody_include[i] = i + 2;
        }
    } else if (nbody_v.size() == static_cast<size_t>(maxorder)) {
        for (i = 0; i < maxorder; ++i) {
            try {
                nbody_include[i] = boost::lexical_cast<int>(nbody_v[i]);
            } catch (std::exception &e) {
                std::cout << e.what() << '\n';
                exit("parse_interaction_vars", "NBODY must be an integer.");
            }
        }
    } else {
        exit("parse_interaction_vars", "The number of entry of NBODY has to be equal to NORDER");
    }

    if (nbody_include[0] != 2) {
        warn("parse_interaction_vars", "Harmonic interaction is always 2 body (except on-site 1 body)");
    }

    input_setter->set_interaction_vars(maxorder, nbody_include);

    deallocate(nbody_include);

    nbody_v.clear();
    no_defaults.clear();
}


auto InputParser::parse_optimize_vars(ALM *alm) -> void
{
    // This method must not be called before setting NAT by parse_general_vars.

    int constraint_flag;
    auto flag_sparse = 0;
    std::string rotation_axis;

    OptimizerControl optcontrol;
    std::vector<std::vector<double>> u_tmp1, f_tmp1;
    std::vector<std::vector<double>> u_tmp2, f_tmp2;

    const std::vector<std::string> input_list{"LMODEL",
                                              "SPARSE",
                                              "SPARSESOLVER",
                                              "ICONST",
                                              "ROTAXIS",
                                              "FC2XML",
                                              "FC3XML",
                                              "FC2FIX",
                                              "FC3FIX",
                                              "NDATA",
                                              "NSTART",
                                              "NEND",
                                              "SKIP",
                                              "DFSET",
                                              "NDATA_CV",
                                              "NSTART_CV",
                                              "NEND_CV",
                                              "DFSET_CV",
                                              "L1_RATIO",
                                              "STANDARDIZE",
                                              "ENET_DNORM",
                                              "L1_ALPHA",
                                              "CV_MAXALPHA",
                                              "CV_MINALPHA",
                                              "CV_MINALPHA_RATIO",
                                              "CV_NALPHA",
                                              "CV",
                                              "MAXITER",
                                              "CONV_TOL",
                                              "NWRITE",
                                              "SOLUTION_PATH",
                                              "DEBIAS_OLS",
                                              "PERIODIC_IMAGE_CONV",
                                              "STOP_CRITERION",
                                              "USE_CHOLESKY",
                                              "CHUNKSIZE",
                                              "ALGO_REDUCTION"};

    std::map<std::string, std::string> optimize_var_dict;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    for (const auto &it: optimize_var_dict) {
        dict_input_vars.insert(it);
    }

    get_var_dict(input_list, optimize_var_dict);

    if (!optimize_var_dict["LMODEL"].empty()) {
        auto str_LMODEL = optimize_var_dict["LMODEL"];
        boost::to_lower(str_LMODEL);

        if (str_LMODEL == "ols" || str_LMODEL == "ls" || str_LMODEL == "least-squares" || str_LMODEL == "1") {
            optcontrol.linear_model = 1;
        } else if (str_LMODEL == "enet" || str_LMODEL == "elastic-net" || str_LMODEL == "2") {
            optcontrol.linear_model = 2;
        } else if (str_LMODEL == "adaptive-lasso" || str_LMODEL == "3") {
            optcontrol.linear_model = 3;
        } else {
            exit("parse_optimize_vars", "Invalid OPTIMIZER-tag");
        }
    }

    if (!optimize_var_dict["SPARSE"].empty()) {
        assign_val(flag_sparse, "SPARSE", optimize_var_dict);
        optcontrol.use_sparse_solver = flag_sparse;
    }
    if (!optimize_var_dict["SPARSESOLVER"].empty()) {
        std::string str_sparsesolver;
        assign_val(str_sparsesolver, "SPARSESOLVER", optimize_var_dict);
        const auto str_lower = boost::algorithm::to_lower_copy(str_sparsesolver);

        if (str_lower != "simplicialldlt" && str_lower != "sparseqr" && str_lower != "conjugategradient" &&
            str_lower != "leastsquaresconjugategradient" && str_lower != "bicgstab")
        {
            exit("parse_optimize_vars", "Unsupported SPARSESOLVER :", str_sparsesolver.c_str());
        }
        optcontrol.sparsesolver = str_sparsesolver;
    }

    if (!optimize_var_dict["ENET_DNORM"].empty()) {
        optcontrol.displacement_normalization_factor = boost::lexical_cast<double>(optimize_var_dict["ENET_DNORM"]);
    }
    if (!optimize_var_dict["L1_ALPHA"].empty()) {
        optcontrol.l1_alpha = boost::lexical_cast<double>(optimize_var_dict["L1_ALPHA"]);
    }
    if (!optimize_var_dict["CV_MINALPHA"].empty()) {
        optcontrol.l1_alpha_min = boost::lexical_cast<double>(optimize_var_dict["CV_MINALPHA"]);
    }
    if (!optimize_var_dict["CV_MAXALPHA"].empty()) {
        optcontrol.l1_alpha_max = boost::lexical_cast<double>(optimize_var_dict["CV_MAXALPHA"]);
    }
    if (!optimize_var_dict["CV_MINALPHA_RATIO"].empty()) {
        optcontrol.l1_alpha_min_ratio = boost::lexical_cast<double>(optimize_var_dict["CV_MINALPHA_RATIO"]);
    }
    if (!optimize_var_dict["CV_NALPHA"].empty()) {
        optcontrol.num_l1_alpha = boost::lexical_cast<int>(optimize_var_dict["CV_NALPHA"]);
    }
    if (!optimize_var_dict["CONV_TOL"].empty()) {
        optcontrol.tolerance_iteration = boost::lexical_cast<double>(optimize_var_dict["CONV_TOL"]);
    }
    if (!optimize_var_dict["MAXITER"].empty()) {
        optcontrol.maxnum_iteration = boost::lexical_cast<int>(optimize_var_dict["MAXITER"]);
    }
    if (!optimize_var_dict["CV"].empty()) {
        optcontrol.cross_validation = boost::lexical_cast<int>(optimize_var_dict["CV"]);
        if (optcontrol.cross_validation == 1) {
            exit("parse_optimize_vars", "CV must not be 1.");
        }
    }
    if (!optimize_var_dict["NWRITE"].empty()) {
        optcontrol.output_frequency = boost::lexical_cast<int>(optimize_var_dict["NWRITE"]);
    }
    if (!optimize_var_dict["STANDARDIZE"].empty()) {
        optcontrol.standardize = boost::lexical_cast<int>(optimize_var_dict["STANDARDIZE"]);
    }
    if (!optimize_var_dict["SOLUTION_PATH"].empty()) {
        optcontrol.save_solution_path = boost::lexical_cast<int>(optimize_var_dict["SOLUTION_PATH"]);
    }
    if (!optimize_var_dict["DEBIAS_OLS"].empty()) {
        optcontrol.debiase_after_l1opt = boost::lexical_cast<int>(optimize_var_dict["DEBIAS_OLS"]);
    }
    if (!optimize_var_dict["L1_RATIO"].empty()) {
        optcontrol.l1_ratio = boost::lexical_cast<double>(optimize_var_dict["L1_RATIO"]);
    }
    if (!optimize_var_dict["STOP_CRITERION"].empty()) {
        optcontrol.stop_criterion = boost::lexical_cast<int>(optimize_var_dict["STOP_CRITERION"]);
    }
    if (!optimize_var_dict["PERIODIC_IMAGE_CONV"].empty()) {
        optcontrol.periodic_image_conv = boost::lexical_cast<int>(optimize_var_dict["PERIODIC_IMAGE_CONV"]);
    }
    if (!optimize_var_dict["USE_CHOLESKY"].empty()) {
        optcontrol.use_cholesky = boost::lexical_cast<int>(optimize_var_dict["USE_CHOLESKY"]);
    }
    if (!optimize_var_dict["CHUNKSIZE"].empty()) {
        optcontrol.chunk_size = boost::lexical_cast<int>(optimize_var_dict["CHUNKSIZE"]);
    }

    DispForceFile datfile_train, datfile_validation;

    if (optimize_var_dict["DFSET"].empty()) {
        exit("parse_optimize_vars", "DFSET tag must be given.");
    }
    datfile_train.filename = optimize_var_dict["DFSET"];

    if (!optimize_var_dict["NDATA"].empty()) {
        assign_val(datfile_train.ndata, "NDATA", optimize_var_dict);
    }
    if (!optimize_var_dict["NSTART"].empty()) {
        assign_val(datfile_train.nstart, "NSTART", optimize_var_dict);
    }
    if (!optimize_var_dict["NEND"].empty()) {
        assign_val(datfile_train.nend, "NEND", optimize_var_dict);
    }

    if (!optimize_var_dict["SKIP"].empty()) {
        std::vector<std::string> str_entry;
        std::string str_skip;
        assign_val(str_skip, "SKIP", optimize_var_dict);
        boost::split(str_entry, str_skip, boost::is_any_of("-"));
        if (str_entry.size() == 1) {
            datfile_train.skip_s = boost::lexical_cast<int>(str_entry[0]);
            datfile_train.skip_e = datfile_train.skip_s + 1;
        } else if (str_entry.size() == 2) {
            datfile_train.skip_s = boost::lexical_cast<int>(str_entry[0]);
            datfile_train.skip_e = boost::lexical_cast<int>(str_entry[1]) + 1;
        } else {
            exit("parse_optimize_vars", "Invalid format for the SKIP-tag.");
        }
    }

    if (!is_data_range_consistent(datfile_train)) {
        exit("parse_optimize_vars", "NDATA, NSTART, NEND and SKIP tags are inconsistent.");
    }

    // Parse u_tmp1 and f_tmp1 from DFSET and set ndata if it's not.
    parse_displacement_and_force_files(u_tmp1, f_tmp1, datfile_train);

    // Check consistency again
    if (!is_data_range_consistent(datfile_train)) {
        exit("parse_optimize_vars", "NDATA, NSTART, NEND and SKIP tags are inconsistent.");
    }

    datfile_validation.skip_s = 0;
    datfile_validation.skip_e = 0;

    if (!optimize_var_dict["NDATA_CV"].empty()) {
        datfile_validation.ndata = boost::lexical_cast<int>(optimize_var_dict["NDATA_CV"]);
    }

    if (!optimize_var_dict["NSTART_CV"].empty()) {
        datfile_validation.nstart = boost::lexical_cast<int>(optimize_var_dict["NSTART_CV"]);
    }
    if (!optimize_var_dict["NEND_CV"].empty()) {
        datfile_validation.nend = boost::lexical_cast<int>(optimize_var_dict["NEND_CV"]);
    }

    if (!optimize_var_dict["DFSET_CV"].empty()) {
        datfile_validation.filename = optimize_var_dict["DFSET_CV"];
    } else {
        datfile_validation.filename = datfile_train.filename;
    }

    if (!is_data_range_consistent(datfile_validation)) {
        exit("parse_optimize_vars", "NDATA_CV, NSTART_CV and NEND_CV tags are inconsistent.");
    }

    if (optcontrol.cross_validation == -1) {
        parse_displacement_and_force_files(u_tmp2, f_tmp2, datfile_validation);

        if (!is_data_range_consistent(datfile_validation)) {
            exit("parse_optimize_vars", "NDATA_CV, NSTART_CV and NEND_CV tags are inconsistent.");
        }
    }

    input_setter->set_optimize_vars(alm, u_tmp1, f_tmp1, u_tmp2, f_tmp2, optcontrol);

    input_setter->set_file_vars(alm, datfile_train, datfile_validation);


    if (optimize_var_dict["ICONST"].empty()) {
        constraint_flag = 11;
    } else {
        assign_val(constraint_flag, "ICONST", optimize_var_dict);
    }

    std::string fc2_file{}, fc3_file{};

    if (!optimize_var_dict["FC2XML"].empty()) {
        warn("parse_optimize_vars",
             "FC2XML is replaced with FC2FIX and will be deprecated in a future release.\n"
             " So, please use FC2FIX instead.");
        fc2_file = optimize_var_dict["FC2XML"];
    } else {
        fc2_file = optimize_var_dict["FC2FIX"];
    }

    if (!optimize_var_dict["FC3XML"].empty()) {
        warn("parse_optimize_vars",
             "FC3XML is replaced with FC3FIX and will be deprecated in a future release.\n"
             " So, please use FC3FIX instead.");
        fc3_file = optimize_var_dict["FC3XML"];
    } else {
        fc3_file = optimize_var_dict["FC3FIX"];
    }
    const auto fix_harmonic = !fc2_file.empty();
    const auto fix_cubic = !fc3_file.empty();

    if (constraint_flag % 10 >= 2) {
        rotation_axis = optimize_var_dict["ROTAXIS"];
        if (rotation_axis.empty()) {
            exit("parse_optimize_vars", "ROTAXIS has to be given when ICONST=2 or 3");
        }
    }

    int ialgo_reduce;
    if (optimize_var_dict["ALGO_REDUCTION"].empty()) {
        ialgo_reduce = 1;
    } else {
        assign_val(ialgo_reduce, "ALGO_REDUCTION", optimize_var_dict);
    }

    input_setter->set_constraint_vars(alm,
                                      constraint_flag,
                                      rotation_axis,
                                      fc2_file,
                                      fc3_file,
                                      fix_harmonic,
                                      fix_cubic,
                                      ialgo_reduce);

    optimize_var_dict.clear();
}


auto InputParser::parse_cutoff_radii() -> void
{
    std::string line, line_wo_comment;
    std::string::size_type pos_first_comment_tag;
    std::vector<std::string> str_cutoff;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    str_cutoff.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_left(line_wo_comment);
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            str_cutoff.push_back(line_wo_comment);
        }
    } else {

        while (std::getline(ifs_input, line)) {

            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_left(line_wo_comment);
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            str_cutoff.push_back(line_wo_comment);
        }
    }

    size_t i, j, k;
    int order;
    std::vector<std::string> cutoff_line;
    std::set<std::string> element_allowed;
    std::vector<std::string> str_pair;
    std::map<std::string, int> kd_map;

    double cutoff_tmp;
    double ***cutoff_radii_tmp;
    bool ***undefined_cutoff;

    allocate(undefined_cutoff, maxorder, nkd_in, nkd_in);

    for (order = 0; order < maxorder; ++order) {
        for (i = 0; i < nkd_in; ++i) {
            for (j = 0; j < nkd_in; ++j) {
                undefined_cutoff[order][i][j] = true;
            }
        }
    }

    allocate(cutoff_radii_tmp, maxorder, nkd_in, nkd_in);

    element_allowed.clear();

    for (i = 0; i < nkd_in; ++i) {
        element_allowed.insert(kdname_vec[i]);
        kd_map.insert(std::map<std::string, int>::value_type(kdname_vec[i], i));
    }

    element_allowed.insert("*");
    kd_map.insert(std::map<std::string, int>::value_type("*", -1));

    for (const auto &it: str_cutoff) {

        split_str_by_space(it, cutoff_line);

        if (cutoff_line.size() < maxorder + 1) {
            exit("parse_cutoff_radii", "Invalid format for &cutoff entry");
        }

        boost::split(str_pair, cutoff_line[0], boost::is_any_of("-"));

        if (str_pair.size() != 2) {
            exit("parse_cutoff_radii2", "Invalid format for &cutoff entry");
        }

        for (i = 0; i < 2; ++i) {
            if (element_allowed.find(str_pair[i]) == element_allowed.end()) {
                exit("parse_cutoff_radii2", "Invalid format for &cutoff entry");
            }
        }

        const auto ikd = kd_map[str_pair[0]];
        const auto jkd = kd_map[str_pair[1]];

        for (order = 0; order < maxorder; ++order) {
            // Accept any strings starting with 'N' or 'n' as 'None'
            if ((cutoff_line[order + 1][0] == 'N') || (cutoff_line[order + 1][0] == 'n')) {
                // Minus value for cutoff radius.
                // This is a flag for neglecting cutoff radius
                cutoff_tmp = -1.0;
            } else {
                cutoff_tmp = boost::lexical_cast<double>(cutoff_line[order + 1]);
            }

            if (ikd == -1 && jkd == -1) {
                for (i = 0; i < nkd_in; ++i) {
                    for (j = 0; j < nkd_in; ++j) {
                        cutoff_radii_tmp[order][i][j] = cutoff_tmp;
                        undefined_cutoff[order][i][j] = false;
                    }
                }
            } else if (ikd == -1) {
                for (i = 0; i < nkd_in; ++i) {
                    cutoff_radii_tmp[order][i][jkd] = cutoff_tmp;
                    cutoff_radii_tmp[order][jkd][i] = cutoff_tmp;
                    undefined_cutoff[order][i][jkd] = false;
                    undefined_cutoff[order][jkd][i] = false;
                }
            } else if (jkd == -1) {
                for (j = 0; j < nkd_in; ++j) {
                    cutoff_radii_tmp[order][j][ikd] = cutoff_tmp;
                    cutoff_radii_tmp[order][ikd][j] = cutoff_tmp;
                    undefined_cutoff[order][j][ikd] = false;
                    undefined_cutoff[order][ikd][j] = false;
                }
            } else {
                cutoff_radii_tmp[order][ikd][jkd] = cutoff_tmp;
                cutoff_radii_tmp[order][jkd][ikd] = cutoff_tmp;
                undefined_cutoff[order][ikd][jkd] = false;
                undefined_cutoff[order][jkd][ikd] = false;
            }
        }
    }
    element_allowed.clear();
    str_cutoff.clear();

    for (order = 0; order < maxorder; ++order) {
        for (j = 0; j < nkd_in; ++j) {
            for (k = 0; k < nkd_in; ++k) {
                if (undefined_cutoff[order][j][k]) {
                    std::cout << " Cutoff radius for " << std::setw(3) << order + 2 << "th-order terms\n";
                    std::cout << " are not defined between elements " << std::setw(3) << j + 1 << " and "
                              << std::setw(3) << k + 1 << '\n';
                    exit("parse_cutoff_radii", "Incomplete cutoff radii");
                }
            }
        }
    }
    deallocate(undefined_cutoff);

    std::vector<double> cutoff_information_flatten;
    cutoff_information_flatten.resize(maxorder * nkd_in * nkd_in);

    i = 0;
    for (order = 0; order < maxorder; ++order) {
        for (j = 0; j < nkd_in; ++j) {
            for (k = 0; k < nkd_in; ++k) {
                cutoff_information_flatten[i++] = cutoff_radii_tmp[order][j][k];
            }
        }
    }
    deallocate(cutoff_radii_tmp);

    input_setter->set_cutoff_radii(maxorder, nkd_in, cutoff_information_flatten);
}

auto InputParser::get_var_dict(const std::vector<std::string> &input_list, std::map<std::string, std::string> &var_dict)
    -> void
{
    std::string line, key, val;
    std::string line_wo_comment;
    std::string::size_type pos_first_comment_tag;
    std::vector<std::string> str_entry, str_varval;

    std::set<std::string> keyword_set;

    for (const auto &it: input_list) {
        keyword_set.insert(it);
    }

    var_dict.clear();

    if (from_stdin) {
        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_left(line_wo_comment);
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            // Split the input line by ';'

            boost::split(str_entry, line_wo_comment, boost::is_any_of(";"));

            for (auto &it: str_entry) {

                // Split the input entry by '='

                auto str_tmp = boost::trim_copy(it);

                if (!str_tmp.empty()) {

                    boost::split(str_varval, str_tmp, boost::is_any_of("="));

                    if (str_varval.size() != 2) {
                        std::cout << "Failed to parse :";
                        for (auto &it2: str_varval) {
                            std::cout << it2 << ' ';
                        }
                        std::cout << '\n';
                        exit("get_var_dict", "Unacceptable format");
                    }

                    key = boost::to_upper_copy(boost::trim_copy(str_varval[0]));
                    val = boost::trim_copy(str_varval[1]);

                    if (keyword_set.find(key) == keyword_set.end()) {
                        std::cout << "Could not recognize the variable " << key << '\n';
                        exit("get_var_dict", "Invalid variable found");
                    }

                    if (var_dict.find(key) != var_dict.end()) {
                        std::cout << "Variable " << key << " appears twice in the input file.\n";
                        exit("get_var_dict", "Redundant input parameter");
                    }

                    // If everything is OK, add the variable and the corresponding value
                    // to the dictionary.

                    var_dict.insert(std::map<std::string, std::string>::value_type(key, val));
                }
            }
        }

    } else {

        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            boost::trim_left(line_wo_comment);
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            // Split the input line by ';'

            boost::split(str_entry, line_wo_comment, boost::is_any_of(";"));

            for (auto &it: str_entry) {

                // Split the input entry by '='

                auto str_tmp = boost::trim_copy(it);

                if (!str_tmp.empty()) {

                    boost::split(str_varval, str_tmp, boost::is_any_of("="));

                    if (str_varval.size() != 2) {
                        std::cout << " Failed to parse : ";
                        for (auto &it2: str_varval) {
                            std::cout << it2 << ' ';
                        }
                        std::cout << '\n';
                        exit("get_var_dict", "Unacceptable format");
                    }

                    key = boost::to_upper_copy(boost::trim_copy(str_varval[0]));
                    val = boost::trim_copy(str_varval[1]);

                    if (keyword_set.find(key) == keyword_set.end()) {
                        std::cout << " Could not recognize the variable " << key << '\n';
                        exit("get_var_dict", "Invalid variable found");
                    }

                    if (var_dict.find(key) != var_dict.end()) {
                        std::cout << " Variable " << key << " appears twice in the input file.\n";
                        exit("get_var_dict", "Redundant input parameter");
                    }

                    // If everything is OK, add the variable and the corresponding value
                    // to the dictionary.

                    var_dict.insert(std::map<std::string, std::string>::value_type(key, val));
                }
            }
        }
    }

    keyword_set.clear();
}

auto InputParser::is_data_range_consistent(const DispForceFile &datfile_in) const -> bool
{
    const auto ndata = datfile_in.ndata;
    const auto nstart = datfile_in.nstart;
    const auto nend = datfile_in.nend;
    const auto skip_s = datfile_in.skip_s;
    const auto skip_e = datfile_in.skip_e;

    if (nstart > 0 && skip_s > 0 && skip_s < nstart) return false;
    if (nend > 0 && skip_e > 0 && (skip_e - 1) > nend) return false;
    if (nstart > 0 && nend > 0 && nstart > nend) return false;
    if (skip_s > 0 && skip_e > 0 && skip_s >= skip_e) return false;

    if (ndata > 0) {
        if (nstart > 0 && nstart > ndata) return false;
        if (nend > 0 && nend > ndata) return false;
        if (skip_s > 0 && skip_s > ndata) return false;
        if (skip_e > 0 && (skip_e - 1) > ndata) return false;
    }

    return true;
}


auto InputParser::locate_tag(const std::string key) -> int
{
    auto ret = 0;
    std::string line;

    if (from_stdin) {

        std::cin.clear();
        std::cin.seekg(0, std::ios_base::beg);

        while (std::cin >> line) {
            boost::to_lower(line);
            if (line == key) {
                ret = 1;
                break;
            }
        }
        return ret;
    }

    ifs_input.clear();
    ifs_input.seekg(0, std::ios_base::beg);

    while (ifs_input >> line) {
        boost::to_lower(line);
        if (line == key) {
            ret = 1;
            break;
        }
    }
    return ret;
}

auto InputParser::is_endof_entry(const std::string str) -> bool
{
    return str[0] == '/';
}

auto InputParser::split_str_by_space(const std::string str, std::vector<std::string> &str_vec) -> void
{
    std::string str_tmp;
    std::istringstream is(str);

    str_vec.clear();

    while (true) {
        str_tmp.clear();
        is >> str_tmp;
        if (str_tmp.empty()) {
            break;
        }
        str_vec.push_back(str_tmp);
    }
    str_tmp.clear();
}

template <typename T>
auto InputParser::assign_val(T &val, const std::string &key, std::map<std::string, std::string> dict) -> void
{
    // Assign a value to the variable "key" using the boost::lexica_cast.

    if (!dict[key].empty()) {
        try {
            val = boost::lexical_cast<T>(dict[key]);
        } catch (std::exception &e) {
            std::cout << e.what() << '\n';
            auto str_tmp = "Invalid entry for the " + key + " tag.\n";
            str_tmp += " Please check the input value.";
            exit("assign_val", str_tmp.c_str());
        }
    }
}
