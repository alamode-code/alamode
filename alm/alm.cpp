/*
 alm.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "alm.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cluster.h"
#include "constraint.h"
#include "fcs.h"
#include "files.h"
#include "optimize.h"
#include "patterndisp.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"
#include "units.h"

namespace
{
// Scale every entry of a 2D training-data array; returns the input untouched
// when the factor is exactly 1 (the default-unit fast path).
auto scaled_copy(const std::vector<std::vector<double>> &data, const double factor) -> std::vector<std::vector<double>>
{
    if (factor == 1.0) return data;
    auto scaled = data;
    for (auto &row: scaled) {
        for (auto &val: row)
            val *= factor;
    }
    return scaled;
}
} // namespace

using namespace ALM_NS;

ALM::ALM()
{
    init_instances();
    verbosity = 1;
    structure_initialized = false;
    initialized_constraint_class = false;
    ofs_alm = nullptr;
    coutbuf = nullptr;
}

ALM::~ALM()
{}

auto ALM::init_instances() -> void
{
    files = std::make_unique<Files>();
    system = std::make_unique<System>();
    cluster = std::make_unique<Cluster>();
    fcs = std::make_unique<Fcs>();
    symmetry = std::make_unique<Symmetry>();
    optimize = std::make_unique<Optimize>();
    constraint = std::make_unique<Constraint>();
    displace = std::make_unique<Displace>();
    timer = std::make_unique<Timer>();
    writer = std::make_unique<Writer>();
}

auto ALM::set_verbosity(const int verbosity_in) -> void
{
    verbosity = verbosity_in;
}

auto ALM::get_verbosity() const -> int
{
    return verbosity;
}

auto ALM::set_output_filename_prefix(const std::string prefix) const -> void // PREFIX
{
    files->set_prefix(prefix);
}

auto ALM::set_print_symmetry(const int printsymmetry) const -> void // PRINTSYM
{
    symmetry->set_print_symmetry(printsymmetry);
}

auto ALM::set_datfile_train(const DispForceFile &dat_in) const -> void
{
    files->set_datfile_train(dat_in);
}

auto ALM::set_datfile_validation(const DispForceFile &dat_in) const -> void
{
    files->set_datfile_validation(dat_in);
}

auto ALM::set_symmetry_tolerance(const double tolerance) const -> void // TOLERANCE
{
    symmetry->set_tolerance(tolerance);
    system->set_tolerance(tolerance); // copy the same value to the system class
}

auto ALM::set_displacement_param(const bool trim_dispsign_for_evenfunc) const -> void // TRIMEVEN
{
    displace->set_trim_dispsign_for_evenfunc(trim_dispsign_for_evenfunc);
}

auto ALM::set_displacement_basis(const std::string str_disp_basis) const -> void // DBASIS
{
    displace->set_disp_basis(str_disp_basis);
}

auto ALM::set_periodicity(const int is_periodic[3]) const -> void // PERIODIC
{
    system->set_periodicity(is_periodic);
}

auto ALM::set_input_units(const std::string &length_unit, const std::string &force_unit) -> void
{
    const auto lu = units::parse_length_unit(length_unit); // throws std::invalid_argument
    const auto fu = units::parse_force_unit(force_unit);
    const auto factor_l = units::length_to_bohr(lu);
    const auto factor_f = units::force_to_ry_bohr(fu);
    if (unit_sensitive_setter_called_ && (factor_l != factor_length_to_bohr_ || factor_f != factor_force_to_ry_bohr_)) {
        throw std::logic_error("set_input_units must be called before the cell, "
                               "displacement-force data, or cutoff radii are set.");
    }
    factor_length_to_bohr_ = factor_l;
    factor_force_to_ry_bohr_ = factor_f;
    length_unit_name_ = units::canonical_name(lu);
    force_unit_name_ = units::canonical_name(fu);
}

auto ALM::get_input_units() const -> std::pair<std::string, std::string>
{
    return {length_unit_name_, force_unit_name_};
}

auto ALM::set_fcs_unit_output(const std::string &unit_system) const -> void // FCS_UNIT_OUTPUT
{
    writer->set_fcs_unit_output(unit_system);
}

auto ALM::get_fcs_unit_output() const -> std::string
{
    return writer->get_fcs_unit_output();
}

auto ALM::set_cell(const size_t nat, const double lavec[3][3], const double xcoord[][3], const int kind[]) const -> void
{
    unit_sensitive_setter_called_ = true;
    if (factor_length_to_bohr_ != 1.0) {
        double lavec_bohr[3][3];
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                lavec_bohr[i][j] = lavec[i][j] * factor_length_to_bohr_;
            }
        }
        system->set_basecell(lavec_bohr, nat, kind, xcoord);
    } else {
        system->set_basecell(lavec, nat, kind, xcoord);
    }
}

auto ALM::set_element_names(const std::vector<std::string> &kdname_in) const -> void
{
    system->set_kdname(kdname_in);
}

auto ALM::set_transformation_matrices(const double transmat_to_super[3][3], const double transmat_to_prim[3][3],
                                      const int autoset_primcell_in) const -> void
{
    system->set_transformation_matrices(transmat_to_super, transmat_to_prim, autoset_primcell_in);
}

auto ALM::set_magnetic_params(const size_t nat,
                              const double (*magmom)[3], // MAGMOM
                              const bool lspin,
                              const int noncollinear,                      // NONCOLLINEAR
                              const int trev_sym_mag,                      // TREVSYM
                              const std::string &str_magmom) const -> void // MAGMOM
{
    system->set_spin_variables(nat, lspin, noncollinear, trev_sym_mag, magmom);
    system->set_str_magmom(str_magmom);
}

auto ALM::set_u_train(const std::vector<std::vector<double>> &u) const -> void
{
    unit_sensitive_setter_called_ = true;
    optimize->set_u_train(scaled_copy(u, factor_length_to_bohr_));
}

auto ALM::set_f_train(const std::vector<std::vector<double>> &f) const -> void
{
    unit_sensitive_setter_called_ = true;
    optimize->set_f_train(scaled_copy(f, factor_force_to_ry_bohr_));
}

auto ALM::set_e_train(const std::vector<double> &e) const -> void
{
    optimize->set_e_train(e);
}

auto ALM::set_e_validation(const std::vector<double> &e) const -> void
{
    optimize->set_e_validation(e);
}

auto ALM::set_validation_data(const std::vector<std::vector<double>> &u,
                              const std::vector<std::vector<double>> &f) const -> void
{
    unit_sensitive_setter_called_ = true;
    optimize->set_validation_data(scaled_copy(u, factor_length_to_bohr_), scaled_copy(f, factor_force_to_ry_bohr_));
}

auto ALM::set_optimizer_control(const OptimizerControl &optcontrol_in) const -> void
{
    optimize->set_optimizer_control(optcontrol_in);
}

auto ALM::set_constraint_mode(const int constraint_flag) const -> void // ICONST
{
    constraint->set_constraint_mode(constraint_flag);
}

auto ALM::set_algebraic_constraint(const int use_algebraic_flag) const -> void // ICONST / 10
{
    constraint->set_constraint_algebraic(use_algebraic_flag);
}

auto ALM::set_tolerance_constraint(const double tolerance_constraint) const -> void // TOL_CONST
{
    constraint->set_tolerance_constraint(tolerance_constraint);
}

auto ALM::set_rotation_axis(const std::string rotation_axis) const -> void // ROTAXIS
{
    constraint->set_rotation_axis(rotation_axis);
}

auto ALM::set_fc_file(const int order, const std::string fc_file) const -> void
{
    constraint->set_fc_file(order, fc_file);
}

auto ALM::set_fc_fix(const int order, const bool fc_fix) const -> void
{
    if (order == 2) {
        constraint->set_fix_harmonic(fc_fix);
    }
    if (order == 3) {
        constraint->set_fix_cubic(fc_fix);
    }
}

auto ALM::set_reduction_algo(const int ialgo_reduce) const -> void
{
    constraint->set_reduction_algorithm(ialgo_reduce);
}

auto ALM::ready_all_constraints() const -> bool
{
    return constraint->ready_all_constraints();
}

auto ALM::set_forceconstants_to_fix(const std::vector<std::vector<int>> &intpair_fix,
                                    const std::vector<double> &values_fix) const -> void
{
    constraint->set_forceconstants_to_fix(intpair_fix, values_fix);
}

auto ALM::set_sparse_mode(const int sparse_mode) const -> void // SPARSE
{
    auto optctrl = optimize->get_optimizer_control();
    optctrl.use_sparse_solver = sparse_mode;
    optimize->set_optimizer_control(optctrl);
}

auto ALM::set_forceconstant_basis(const std::string preferred_basis) const -> void // FCSYM_BASIS
{
    fcs->set_forceconstant_basis(preferred_basis);
}

auto ALM::get_forceconstant_basis() const -> std::string
{
    return fcs->get_forceconstant_basis();
}

auto ALM::set_nmaxsave(const int nmaxsave) const -> void // NMAXSAVE
{
    writer->set_output_maxorder(nmaxsave);
}

auto ALM::get_nmaxsave() const -> int
{
    return writer->get_output_maxorder();
}

auto ALM::set_compression_level(const int level) const -> void
{
    writer->set_compression_level(level);
}

auto ALM::get_compression_level() const -> int
{
    return writer->get_compression_level();
}

auto ALM::define(const int maxorder, const size_t nkd, const int *nbody_include,
                 const double *cutoff_radii) const -> void
{
    // nkd = 0 means cutoff_radii undefined (hopefully nullptr).
    unit_sensitive_setter_called_ = true;
    if (cutoff_radii && factor_length_to_bohr_ != 1.0) {
        std::vector<double> cutoff_bohr(cutoff_radii, cutoff_radii + maxorder * nkd * nkd);
        for (auto &r: cutoff_bohr) {
            if (r > 0.0) r *= factor_length_to_bohr_; // negative = "no cutoff" sentinel
        }
        cluster->define(maxorder, nkd, nbody_include, cutoff_bohr.data());
    } else {
        cluster->define(maxorder, nkd, nbody_include, cutoff_radii);
    }
}

auto ALM::get_optimizer_control() const -> OptimizerControl
{
    return optimize->get_optimizer_control();
}

auto ALM::get_u_train() const -> std::vector<std::vector<double>>
{
    return optimize->get_u_train();
}

auto ALM::get_f_train() const -> std::vector<std::vector<double>>
{
    return optimize->get_f_train();
}

auto ALM::get_number_of_data() const -> size_t
{
    return optimize->get_number_of_data();
}

auto ALM::get_nrows_sensing_matrix() const -> size_t
{
    return optimize->get_number_of_rows_sensing_matrix();
}

auto ALM::get_cv_l1_alpha() const -> double
{
    return optimize->get_cv_l1_alpha();
}

auto ALM::get_symmetry_tolerance() const -> double
{
    return symmetry->get_tolerance();
}

auto ALM::get_supercell() const -> Cell
{
    return system->get_supercell();
}

auto ALM::get_kdname() const -> std::vector<std::string>
{
    return system->get_kdname();
}

auto ALM::get_spin() const -> Spin
{
    return system->get_spin();
}

auto ALM::set_str_magmom(std::string) -> void
{}

auto ALM::get_str_magmom() const -> std::string
{
    return system->get_str_magmom();
}

auto ALM::get_x_image() const -> const std::vector<Eigen::MatrixXd> &
{
    return system->get_x_image();
}

auto ALM::get_periodicity() const -> int *
{
    return system->get_periodicity();
}

auto ALM::get_atom_mapping_by_pure_translations() const -> const std::vector<std::vector<int>> &
{
    return symmetry->get_map_trueprim_to_super();
}

auto ALM::get_maxorder() const -> int
{
    return cluster->get_maxorder();
}

auto ALM::get_nbody_include() const -> int *
{
    return cluster->get_nbody_include();
}

auto ALM::get_number_of_displacement_patterns(const int fc_order) const -> size_t
// harmonic=1, ...
{
    const auto order = fc_order - 1;
    return displace->get_pattern_all(order).size();
}

auto ALM::get_number_of_displaced_atoms(int *numbers, const int fc_order) const -> void
// harmonic=1, ...
{
    const auto order = fc_order - 1;

    for (size_t i = 0; i < displace->get_pattern_all(order).size(); ++i) {
        numbers[i] = static_cast<int>(displace->get_pattern_all(order)[i].atoms.size());
    }
}

auto ALM::get_displacement_patterns(int *atom_indices, double *disp_patterns, const int fc_order) const -> int
// harmonic=1, ...
{
    const auto order = fc_order - 1;

    auto i_atom = 0;
    auto i_disp = 0;
    for (const auto &displacements: displace->get_pattern_all(order)) {
        for (size_t j = 0; j < displacements.atoms.size(); ++j) {
            atom_indices[i_atom] = displacements.atoms[j];
            ++i_atom;
            for (auto k = 0; k < 3; ++k) {
                disp_patterns[i_disp] = displacements.directions[3 * j + k];
                ++i_disp;
            }
        }
    }

    // 0:Cartesian or 1:Fractional. -1 means something wrong.
    if (displace->get_disp_basis()[0] == 'C') {
        return 0;
    }
    if (displace->get_disp_basis()[0] == 'F') {
        return 1;
    }
    return -1;
}

auto ALM::get_number_of_fc_elements(const int fc_order) const -> size_t
// harmonic=1, ...
{
    const auto order = fc_order - 1;

    if (fcs->get_nequiv()[order].empty()) {
        return 0;
    }
    size_t id = 0;
    const auto num_unique_elems = fcs->get_nequiv()[order].size();

    for (size_t iuniq = 0; iuniq < num_unique_elems; ++iuniq) {
        const auto num_equiv_elems = fcs->get_nequiv()[order][iuniq];
        id += num_equiv_elems;
    }
    return id;
}

auto ALM::get_number_of_irred_fc_elements(const int fc_order) -> size_t // harmonic=1, ...
{
    // Returns the number of irreducible force constants for the given order.
    // The irreducible force constant means a set of independent force constants
    // reduced by using all available symmetry operations and
    // constraints for translational invariance. Rotational invariance is not considered.

    const auto order = fc_order - 1;
    if (!initialized_constraint_class) {
        constraint->setup(system,
                          fcs,
                          cluster,
                          symmetry,
                          get_optimizer_control().linear_model,
                          get_optimizer_control().periodic_image_conv,
                          verbosity,
                          timer);
        initialized_constraint_class = true;
    }
    if (!ready_all_constraints()) {
        constraint->update_constraint_matrix(system,
                                             symmetry,
                                             cluster,
                                             fcs,
                                             verbosity,
                                             get_optimizer_control().periodic_image_conv,
                                             constraint->get_reduction_algorithm());
    }

    return constraint->get_index_bimap(order).size();
}

auto ALM::get_number_of_fc_origin(const int fc_order, const int permutation) const -> size_t
{
    if (fc_order <= 0) {
        std::cout << "fc_order must be larger than 0." << '\n';
        exit(EXIT_FAILURE);
    }
    const auto maxorder = cluster->get_maxorder();
    if (fc_order > maxorder) {
        std::cout << "fc_order must not be larger than maxorder" << '\n';
        exit(EXIT_FAILURE);
    }
    auto nfc_cart = fcs->get_nfc_cart(1);

    if (nfc_cart.size() < fc_order) {
        std::cout << "fc has not yet been computed or set." << '\n';
        exit(EXIT_FAILURE);
    }

    if (permutation) {
        return fcs->get_nfc_cart(1)[fc_order - 1];
    } else {
        return fcs->get_nfc_cart(0)[fc_order - 1];
    }
}

auto ALM::get_fc_origin(double *fc_values,
                        int *elem_indices,  // (len(fc_values), fc_order + 1) is flatten.
                        const int fc_order, // harmonic=1, ...
                        const int permutation) const -> void
{
    // Return a set of force constants Phi(i,j,k,...) where i is an atom
    // inside the primitive cell at origin.

    const auto maxorder = cluster->get_maxorder();
    if (fc_order > maxorder) {
        std::cout << "fc_order must not be larger than maxorder" << '\n';
        exit(EXIT_FAILURE);
    }
    if (!fcs->get_fc_cart()) {
        std::cout << "fc has not yet been computed." << '\n';
        exit(EXIT_FAILURE);
    }

    auto id = 0;

    if (permutation) {
        for (const auto &it: fcs->get_fc_cart()[fc_order - 1]) {
            fc_values[id] = it.fc_value;
            for (auto i = 0; i < fc_order + 1; ++i) {
                elem_indices[id * (fc_order + 1) + i] = it.flattenarray[i];
            }
            ++id;
        }
    } else {
        for (const auto &it: fcs->get_fc_cart()[fc_order - 1]) {
            if (it.is_ascending_order) {
                fc_values[id] = it.fc_value;
                for (auto i = 0; i < fc_order + 1; ++i) {
                    elem_indices[id * (fc_order + 1) + i] = it.flattenarray[i];
                }
                ++id;
            }
        }
    }
}


auto ALM::get_fc_irreducible(double *fc_values,
                             int *elem_indices,          // (len(fc_values), fc_order + 1) is flatten.
                             const int fc_order) -> void // harmonic=1, ...
{
    // Return an irreducible set of force constants.

    double fc_elem;

    const auto maxorder = cluster->get_maxorder();
    if (fc_order > maxorder) {
        std::cout << "fc_order must not be larger than maxorder" << '\n';
        exit(EXIT_FAILURE);
    }
    if (!optimize->get_params()) {
        std::cout << "fc has not yet been computed." << '\n';
        exit(EXIT_FAILURE);
    }

    if (!initialized_constraint_class) {
        constraint->setup(system,
                          fcs,
                          cluster,
                          symmetry,
                          get_optimizer_control().linear_model,
                          get_optimizer_control().periodic_image_conv,
                          verbosity,
                          timer);
        initialized_constraint_class = true;
    }
    if (!ready_all_constraints()) {
        constraint->update_constraint_matrix(system,
                                             symmetry,
                                             cluster,
                                             fcs,
                                             verbosity,
                                             get_optimizer_control().periodic_image_conv,
                                             constraint->get_reduction_algorithm());
    }

    size_t ishift = 0;
    size_t inew, iold;

    for (auto order = 0; order < fc_order; ++order) {

        if (constraint->get_index_bimap(order).empty()) {
            continue;
        }

        if (order == fc_order - 1) {
            for (const auto &it: constraint->get_index_bimap(order)) {
                inew = it.left;
                iold = it.right + ishift;

                fc_elem = optimize->get_params()[iold];
                fc_values[inew] = fc_elem;
                for (auto i = 0; i < fc_order + 1; ++i) {
                    elem_indices[inew * (fc_order + 1) + i] = fcs->get_fc_table()[order][it.right].elems[i];
                }
            }
        }
        ishift += fcs->get_nequiv()[order].size();
    }
}


auto ALM::get_fc_all(double *fc_values,
                     int *elem_indices,  // (len(fc_values), fc_order + 1) is flatten.
                     const int fc_order, // harmonic=1, ...
                     const int permutation) const -> void
{
    int i;
    const auto ntran = symmetry->get_ntran();
    const auto maxorder = cluster->get_maxorder();

    if (fc_order > maxorder) {
        std::cout << "fc_order must not be larger than maxorder" << '\n';
        exit(EXIT_FAILURE);
    }
    if (!fcs->get_fc_cart()) {
        std::cout << "fc has not yet been computed." << '\n';
        exit(EXIT_FAILURE);
    }

    std::vector<int> pair_tran(fc_order + 1);
    size_t id = 0;

    if (permutation) {
        for (const auto &it: fcs->get_fc_cart()[fc_order - 1]) {

            for (size_t itran = 0; itran < ntran; ++itran) {
                for (i = 0; i < fc_order + 1; ++i) {
                    pair_tran[i] = symmetry->get_map_sym()[it.atoms[i]][symmetry->get_symnum_tran()[itran]];
                }
                fc_values[id] = it.fc_value;
                for (i = 0; i < fc_order + 1; ++i) {
                    elem_indices[id * (fc_order + 1) + i] = 3 * pair_tran[i] + it.coords[i];
                }
                ++id;
            }
        }
    } else {
        for (const auto &it: fcs->get_fc_cart()[fc_order - 1]) {
            if (it.is_ascending_order) {
                for (size_t itran = 0; itran < ntran; ++itran) {
                    for (i = 0; i < fc_order + 1; ++i) {
                        pair_tran[i] = symmetry->get_map_sym()[it.atoms[i]][symmetry->get_symnum_tran()[itran]];
                    }
                    fc_values[id] = it.fc_value;
                    for (i = 0; i < fc_order + 1; ++i) {
                        elem_indices[id * (fc_order + 1) + i] = 3 * pair_tran[i] + it.coords[i];
                    }
                    ++id;
                }
            }
        }
    }
}

auto ALM::set_fc(double *fc_in) const -> void
{
    optimize->set_fcs_values(cluster->get_maxorder(), fc_in, fcs->get_nequiv(), constraint);

    fcs->set_forceconstant_cartesian(cluster->get_maxorder(), optimize->get_params());
}

void ALM::set_fc_zero_threshold(const double threshold_in) const
{
    fcs->set_fc_zero_threshold(threshold_in);
}

auto ALM::get_fc_zero_threshold() const -> double
{
    return fcs->get_fc_zero_threshold();
}

auto ALM::get_number_of_free_parameters() -> size_t
{
    // Ensure the (algebraic) constraints are set up, mirroring get_matrix_elements's prologue,
    // then count the free parameters per order from the algebraic index map.
    if (!initialized_constraint_class) {
        constraint->setup(system,
                          fcs,
                          cluster,
                          symmetry,
                          get_optimizer_control().linear_model,
                          get_optimizer_control().periodic_image_conv,
                          verbosity,
                          timer);
        initialized_constraint_class = true;
    }
    if (!ready_all_constraints()) {
        constraint->update_constraint_matrix(system,
                                             symmetry,
                                             cluster,
                                             fcs,
                                             verbosity,
                                             get_optimizer_control().periodic_image_conv,
                                             constraint->get_reduction_algorithm());
    }
    size_t n = 0;
    const int maxorder = cluster->get_maxorder();
    for (int i = 0; i < maxorder; ++i)
        n += constraint->get_index_bimap(i).size();
    return n;
}

auto ALM::get_matrix_elements(double *amat, double *bvec) -> void
{
    const auto maxorder = cluster->get_maxorder();
    double fnorm;

    std::vector<double> amat_vec;
    std::vector<double> bvec_vec;

    if (!initialized_constraint_class) {
        constraint->setup(system,
                          fcs,
                          cluster,
                          symmetry,
                          get_optimizer_control().linear_model,
                          get_optimizer_control().periodic_image_conv,
                          verbosity,
                          timer);
        initialized_constraint_class = true;
    }
    if (!ready_all_constraints()) {
        constraint->update_constraint_matrix(system,
                                             symmetry,
                                             cluster,
                                             fcs,
                                             verbosity,
                                             get_optimizer_control().periodic_image_conv,
                                             constraint->get_reduction_algorithm());
    }

    std::unique_ptr<SensingMatrix> matrix_out = std::make_unique<SensingMatrix>();

    optimize->get_matrix_elements_unified(maxorder,
                                          matrix_out,
                                          optimize->get_u_train(),
                                          optimize->get_f_train(),
                                          symmetry,
                                          fcs,
                                          constraint,
                                          true,
                                          false,
                                          false,
                                          0);
    //    optimize->get_matrix_elements_algebraic_constraint(maxorder,
    //                                                       amat_vec,
    //                                                       bvec_vec,
    //                                                       optimize->get_u_train(),
    //                                                       optimize->get_f_train(),
    //                                                       fnorm,
    //                                                       symmetry,
    //                                                       fcs,
    //                                                       constraint);
    // This may be inefficient.
    auto i = 0;
    for (const auto it: matrix_out->amat_dense) {
        amat[i++] = it;
    }
    i = 0;
    for (const auto it: matrix_out->bvec) {
        bvec[i++] = it;
    }
    //amat = amat_vec.data();
    //bvec = bvec_vec.data();
}

auto ALM::run_optimize() -> int
{
    if (!structure_initialized) {
        std::cout << "initialize_structure must be called beforehand." << '\n';
        exit(EXIT_FAILURE);
    }

    if (!initialized_constraint_class) {
        constraint->setup(system,
                          fcs,
                          cluster,
                          symmetry,
                          get_optimizer_control().linear_model,
                          get_optimizer_control().periodic_image_conv,
                          verbosity,
                          timer);
        initialized_constraint_class = true;
    }
    if (!ready_all_constraints()) {
        constraint->update_constraint_matrix(system,
                                             symmetry,
                                             cluster,
                                             fcs,
                                             verbosity,
                                             get_optimizer_control().periodic_image_conv,
                                             constraint->get_reduction_algorithm());
    }

    const auto maxorder = cluster->get_maxorder();
    std::vector<std::string> str_order(maxorder);
    for (auto i = 0; i < maxorder; ++i) {
        str_order[i] = cluster->get_ordername(i);
    }
    const auto info = optimize->optimize_main(symmetry,
                                              constraint,
                                              fcs,
                                              maxorder,
                                              files->get_prefix(),
                                              str_order,
                                              verbosity,
                                              files->get_datfile_train(),
                                              files->get_datfile_validation(),
                                              writer->get_output_maxorder(),
                                              timer);
    return info;
}

auto ALM::run_suggest() const -> void
{
    displace->gen_displacement_pattern(cluster, symmetry, fcs, constraint, system, verbosity);
}

auto ALM::init_fc_table() -> void
{
    // Initialization of structure information.
    // Perform initialization only once.

    if (!structure_initialized) {
        system->init(verbosity, timer);
        symmetry->init(system, verbosity, timer);
        structure_initialized = true;
    }

    // Build cluster & force constant table
    cluster->init(system, symmetry, get_optimizer_control().periodic_image_conv, verbosity, timer);
    fcs->init(cluster, symmetry, system->get_supercell(), verbosity, timer);

    // Switch off the initialized_constraint_class flag
    // because the force constants are updated
    // but corresponding constranits are not.
    initialized_constraint_class = false;
}

auto ALM::save_fc(const std::string &filename, const std::string fcs_format, const int maxorder_to_save) const -> void
{
    writer->set_output_maxorder(maxorder_to_save);
    writer->set_filename_fcs(filename);
    writer->save_fcs_with_specific_format(fcs_format,
                                          system,
                                          symmetry,
                                          cluster,
                                          constraint,
                                          fcs,
                                          optimize,
                                          files,
                                          verbosity);
}

auto ALM::set_fcs_save_flag(const std::string &fcs_format, const int val) const -> void
{
    writer->set_fcs_save_flag(fcs_format, val);
}

auto ALM::get_fcs_save_flag(const std::string &fcs_format) const -> int
{
    return writer->get_fcs_save_flag(fcs_format);
}

auto ALM::set_input_vars(const std::map<std::string, std::string> &input_var_dict) const -> void
{
    writer->set_input_vars(input_var_dict);
}

auto ALM::get_input_var(const std::string &key) const -> std::string
{
    return writer->get_input_var(key);
}

auto ALM::set_pattern_format(const std::string &format_name) const -> void
{
    writer->set_format_patternfile(format_name);
}

auto ALM::get_format_pattern() const -> std::string
{
    return writer->get_format_patternfile();
}
