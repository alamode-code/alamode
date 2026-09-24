/*
fcs_phonon.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "fcs_phonon.h"
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include "cell_shift_table.h"
#include "constants.h"
#include "error.h"
#include "hdf5_parser.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon.h"
#include "stage_timer.h"
#include "system.h"

using namespace PHON_NS;

Fcs_phonon::Fcs_phonon(const RunInfo &run_in, const System *system_in) : run(run_in), system(system_in)
{
    set_default_variables();
}

Fcs_phonon::~Fcs_phonon()
{
    deallocate_variables();
}

void Fcs_phonon::set_default_variables()
{
    maxorder = 0;
    file_fcs = "";
    file_fc2 = "";
    file_fc3 = "";
    file_fc4 = "";

    update_fc2 = false;
}

void Fcs_phonon::deallocate_variables()
{}

void Fcs_phonon::setup(const std::string &mode, const int quartic_mode, const bool cubic_for_phonons,
                       const bool print_newfcs)
{
    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << " =================\n";
        std::cout << "  Force Constants \n";
        std::cout << " =================\n\n";
    }

    if (mode == "PHONONS") {
        require_cubic = false;
        require_quartic = false;
        maxorder = 1;

        if (cubic_for_phonons) {
            require_cubic = true;
            maxorder = 2;
        }
        if (print_newfcs) {
            require_cubic = true;
            maxorder = 2;

            if (quartic_mode > 0) {
                require_quartic = true;
                maxorder = 3;
            }
        }

    } else if (mode == "KAPPA") {
        require_cubic = true;

        if (quartic_mode > 0) {
            maxorder = 3;
            require_quartic = true;
        } else {
            maxorder = 2;
            require_quartic = false;
        }
    } else if (mode == "SCPH" || mode == "QHA") {
        require_cubic = true;
        require_quartic = true;
        maxorder = 3;
        // quartic_mode == 1 is guaranteed for these modes by the input
        // parser (parse_analysis_vars).
    }

    // RELAXED_STRUCTURE deforms the cubic IFCs as Phi3 + Phi4 : d, so a run
    // that uses them at all needs the quartic ones, whether or not
    // four-phonon scattering is wanted. Without this a 3ph KAPPA run would
    // adopt the relaxed cell but silently keep the reference FC3, which is
    // the very inconsistency the tag exists to remove. The producer had to
    // load FC4 anyway, so this asks for nothing the user does not have.
    if (relaxed_structure && require_cubic) {
        maxorder = 3;
        require_quartic = true;
    }

    force_constant_with_cell.resize(maxorder);

    if (run.my_rank == 0) {

        const auto t_stage = stage_clock();
        load_fcs_from_file(maxorder);
        print_stage_line("IFCs: read from file", stage_clock() - t_stage, run.my_rank, run.verbosity);

        if (run.verbosity > 0) {
            for (auto i = 0; i < maxorder; ++i) {
                std::cout << "  Number of non-zero IFCs for " << i + 2 << " order: ";
                std::cout << force_constant_with_cell[i].size() << '\n';
            }
            std::cout << '\n';

            std::cout << "  Maximum deviation from the translational invariance: \n";
        }
        for (auto i = 0; i < maxorder; ++i) {
            // Before replication, the first atom index runs over the primitive cell of
            // the FCS file, which can be larger than the &cell primitive cell.
            const auto maxdev =
                examine_translational_invariance(i,
                                                 system->get_supercell(i).number_of_atoms,
                                                 system->get_mapping_super_alm(i).from_true_primitive.size(),
                                                 force_constant_with_cell[i]);
            if (run.verbosity > 0)
                std::cout << "   Order " << i + 2 << " : " << std::setw(12) << std::scientific << maxdev << '\n';
        }
        if (run.verbosity > 0) std::cout << '\n';
    }

    auto t_stage = stage_clock();
    MPI_Bcast_fcs_array(maxorder);
    print_stage_line("IFCs: MPI broadcast", stage_clock() - t_stage, run.my_rank, run.verbosity);
    // Fingerprint the IFCs as loaded. Every rank now holds the same arrays,
    // so this needs no further communication, and it must precede
    // replicate_force_constants, which multiplies the entry count by the
    // number of translations of the user cell.
    fcs_nrows.assign(maxorder, 0);
    fcs_sum_abs.assign(maxorder, 0.0);
    fcs_sum_signed.assign(maxorder, 0.0);
    fcs_sum_sq.assign(maxorder, 0.0);
    for (unsigned int order = 0; order < maxorder; ++order) {
        const auto &fcs = force_constant_with_cell[order];
        fcs_nrows[order] = fcs.size();
        auto sum_abs = 0.0, sum_signed = 0.0, sum_sq = 0.0;
        for (const auto &it: fcs) {
            sum_abs += std::abs(it.fcs_val);
            sum_signed += it.fcs_val;
            sum_sq += it.fcs_val * it.fcs_val;
        }
        fcs_sum_abs[order] = sum_abs;
        fcs_sum_signed[order] = sum_signed;
        fcs_sum_sq[order] = sum_sq;
    }

    t_stage = stage_clock();
    replicate_force_constants(maxorder);
    // Collective on every rank: DFC2FILE is known on rank 0 only, and a run
    // without it broadcasts zero rows.
    append_delta_fc2_rows(force_constant_with_cell[0]);
    print_stage_line("IFCs: replicate to the unit cell", stage_clock() - t_stage, run.my_rank, run.verbosity);

    // Sort the anharmonic IFCs using the operator defined in fcs_phonon.h.
    // This sorting is necessary. It is done once here because AnharmonicCore,
    // Gruneisen and DerivativeIFC all read these arrays.
    if (maxorder >= 2) {
        std::sort(force_constant_with_cell[1].begin(), force_constant_with_cell[1].end());
    }
    if (maxorder >= 3) {
        std::sort(force_constant_with_cell[2].begin(), force_constant_with_cell[2].end());
    }
}

void Fcs_phonon::replicate_force_constants(const int maxorder_in)
{
    for (auto order = 0; order < maxorder_in; ++order) {
        replicate_force_constant(system, force_constant_with_cell[order]);
    }
}

void Fcs_phonon::deform_relative_vectors(const std::vector<double> &u0)
{
    if (u0.empty()) return;

    const auto natmin = system->get_primcell().number_of_atoms;
    if (u0.size() != 3 * natmin) {
        exit("deform_relative_vectors", "The displacement field does not match the primitive cell of this run.");
    }

    const Eigen::Matrix3d lavec_inv = system->get_primcell().lattice_vector.inverse();
    std::vector<Eigen::Vector3d> du0_frac(natmin);
    for (size_t kappa = 0; kappa < natmin; ++kappa) {
        du0_frac[kappa] = lavec_inv * Eigen::Vector3d(u0[3 * kappa], u0[3 * kappa + 1], u0[3 * kappa + 2]);
    }

    for (auto order = 0; order < maxorder; ++order) {
        for (auto &it: force_constant_with_cell[order]) {
            const auto kappa_first = it.pairs[0].index / 3;
            for (size_t leg = 0; leg < it.relvecs_velocity.size(); ++leg) {
                const auto kappa_leg = it.pairs[leg + 1].index / 3;
                it.relvecs_velocity[leg] += du0_frac[kappa_leg] - du0_frac[kappa_first];
            }
        }
    }
}

void Fcs_phonon::replicate_force_constant(const System *system_in, std::vector<FcsArrayWithCell> &fcs_inout)
{
    // Replicate IFCs from the true primitive cell to the user-defined cell,
    // convert relative vectors to its lattice basis, and derive relvec
    // from relvec_velocity.

    std::vector<FcsArrayWithCell> force_constant_replicate;
    std::vector<Eigen::Vector3d> relvecs, relvecs_vel;
    Eigen::Vector3d relvec_tmp, relvec_tmp2;
    std::vector<std::vector<unsigned int>> map_trans;
    Eigen::Vector3d xshift;
    Eigen::Vector3d xdiff, xdiff_cart;
    std::vector<unsigned int> map_now;

    const int order = static_cast<int>(fcs_inout[0].pairs.size()) - 2;

    if (order < 0) return;

    force_constant_replicate.clear();
    map_trans.clear();

    const auto [to_true_primitive, from_true_primitive] = system_in->get_mapping_super_alm(order);
    const auto &cell_tmp = system_in->get_supercell(order);
    const auto ntran_tmp = from_true_primitive[0].size();

    // Generate the atom index mapping table for all translations
    for (auto itran = 0; itran < ntran_tmp; ++itran) {
        xshift = cell_tmp.x_fractional.row(from_true_primitive[0][itran]) -
                 cell_tmp.x_fractional.row(from_true_primitive[0][0]);
        map_now.clear();

        for (auto iat = 0; iat < cell_tmp.number_of_atoms; ++iat) {
            auto kat = -1;
            for (auto jat = 0; jat < cell_tmp.number_of_atoms; ++jat) {
                xdiff = cell_tmp.x_fractional.row(jat) - cell_tmp.x_fractional.row(iat);
                xdiff = (xdiff + xshift).unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });
                xdiff_cart = cell_tmp.lattice_vector * xdiff;
                if (xdiff_cart.norm() < 1.0e-3) {
                    kat = jat;
                    break;
                }
            }
            if (kat == -1) {
                exit("replicate_force_constants", "Equivalent atom could not be found.");
            } else {
                map_now.emplace_back(kat);
            }
        }
        map_trans.emplace_back(map_now);
    }

    std::vector<AtomCellSuper> pairs_tmp(order + 2);
    std::vector<unsigned int> atom_super(order + 2), atom_super_tran(order + 2);
    std::vector<unsigned int> atom_new_prim(order + 2), atom_new_super(order + 2);

    const auto convmat = system_in->get_primcell().lattice_vector.inverse();

    for (const auto &it_trans: map_trans) {
        for (const auto &it: fcs_inout) {

            for (auto i = 0; i < order + 2; ++i) {
                atom_super[i] = it.atoms_s[i];
                atom_super_tran[i] = it_trans[atom_super[i]];
            }

            if (system_in->get_map_s2p(order)[atom_super_tran[0]].tran_num != 0) continue;

            for (auto i = 0; i < order + 2; ++i) {
                atom_new_prim[i] = system_in->get_map_s2p(order)[atom_super_tran[i]].atom_num;
                pairs_tmp[i].index = 3 * atom_new_prim[i] + it.pairs[i].index % 3;
                pairs_tmp[i].tran = system_in->get_map_s2p(order)[atom_super_tran[i]].tran_num;
                pairs_tmp[i].cell_s = it.pairs[i].cell_s;
            }

            relvecs.clear();
            relvecs_vel.clear();
            for (auto i = 0; i < order + 1; ++i) {
                for (auto j = 0; j < 3; ++j) {
                    relvec_tmp[j] = it.relvecs_velocity[i][j] + cell_tmp.x_cartesian(atom_super_tran[0], j) -
                                    cell_tmp.x_cartesian(system_in->get_map_p2s(order)[atom_new_prim[i + 1]][0], j);
                    relvec_tmp2[j] = it.relvecs_velocity[i][j];
                }
                relvec_tmp = convmat * relvec_tmp;
                relvec_tmp2 = convmat * relvec_tmp2;
                relvecs.emplace_back(relvec_tmp);
                relvecs_vel.emplace_back(relvec_tmp2);
            }
            force_constant_replicate.emplace_back(it.fcs_val, pairs_tmp, atom_super_tran, relvecs, relvecs_vel);
        }
    }

    fcs_inout.clear();
    std::copy(force_constant_replicate.begin(), force_constant_replicate.end(), std::back_inserter(fcs_inout));
}


void Fcs_phonon::load_fcs_from_file(const int maxorder_in)
{
    std::vector filename_list{file_fc2, file_fc3, file_fc4};

    std::vector load_flags{true, require_cubic, require_quartic};

    if (file_fc2.empty()) {
        filename_list[0] = file_fcs;
    }

    if (require_cubic) {
        if (file_fc3.empty()) {
            if (!file_fcs.empty()) {
                filename_list[1] = file_fcs;
            } else {
                exit("load_fcs_from_file",
                     "Either FCSFILE or FC3FILE must be given in the "
                     "&general section of the input file.");
            }
        }
    }

    if (require_quartic) {
        if (file_fc4.empty()) {
            if (!file_fcs.empty()) {
                filename_list[2] = file_fcs;
            } else {
                exit("load_fcs_from_file",
                     "Either FCSFILE or FC4FILE must be given in the "
                     "&general section of the input file.");
            }
        }
    }

    // RELAXED_STRUCTURE raises require_quartic on its own, so the file the
    // user named for the other orders may simply not carry FC4. Say so
    // here: letting the loader fall through produces a bare HDF5 "cannot
    // open dataset" that explains nothing.
    if (relaxed_structure && require_quartic && filename_list[2].size() > 3 &&
        filename_list[2].compare(filename_list[2].size() - 3, 3, ".h5") == 0)
    {
        const HighFive::File probe(filename_list[2], HighFive::File::ReadOnly);
        if (!probe.exist("/ForceConstants/Order4")) {
            exit("load_fcs_from_file",
                 ("RELAXED_STRUCTURE = 1 needs the quartic force constants, to deform FC3 as\n"
                  " Phi3 + Phi4 : d, but " +
                  filename_list[2] +
                  " carries none.\n Give FC4FILE, or an FCSFILE that includes the quartic order -- the same ones\n"
                  " the SCPH/QHA run used.")
                     .c_str());
        }
    }

    if (run.verbosity > 0) {
        std::cout << "  Reading force constants from the following file(s):\n";
        for (auto i = 0; i < filename_list.size(); ++i) {
            if (!load_flags[i]) continue;
            std::cout << "   Order " << i + 2 << " : " << filename_list[i] << '\n';
        }
        std::cout << "  ... ";
    }

    for (auto i = 0; i < filename_list.size(); ++i) {

        if (!load_flags[i]) continue;

        get_fcs_from_file(filename_list[i], i, force_constant_with_cell[i]);

        // Legacy dfc2.py workflow, native: the (short-ranged) anharmonic FC2
        // correction of an SCPH/QHA state file, added onto the harmonic FC2 of
        // a possibly larger supercell after replication (append_delta_fc2_rows).
        if (i == 0 && !file_dfc2.empty()) read_delta_fc2_from_scph(file_dfc2);
    }

    if (run.verbosity > 0) std::cout << "done.\n\n";
}

void Fcs_phonon::get_fcs_from_file(const std::string &fname_fcs, const int order,
                                   std::vector<FcsArrayWithCell> &fcs_out) const
{
    const auto file_extension = fname_fcs.substr(fname_fcs.find_last_of('.') + 1);
    if (file_extension == "xml" || file_extension == "XML") {
        load_fcs_xml(fname_fcs, order, fcs_out);
    } else if (file_extension == "h5" || file_extension == "hdf5") {
        parse_fcs_from_h5(fname_fcs, order, fcs_out);
    } else {
        const auto str_error =
            "Unsupported file extension of " + fname_fcs + " (only .xml, .h5, and .hdf5 are accepted).";
        exit("get_fcs_from_file", str_error.c_str());
    }
}


void Fcs_phonon::load_fcs_xml(const std::string &fname_fcs, const int order,
                              std::vector<FcsArrayWithCell> &fcs_out) const
{
    using namespace boost::property_tree;
    ptree pt;
    std::string str_tag;
    unsigned int atmn, xyz, cell_s;

    std::stringstream ss;

    std::vector<AtomCellSuper> ivec_with_cell, ivec_copy;
    std::vector<Eigen::Vector3d> relvecs, relvecs_velocity;

    Eigen::Vector3d relvec_tmp;
    std::vector<unsigned int> atoms_s_tmp;

    NDArray<double, 2> xf_image;
    build_27cell_shift_table(xf_image);
    const auto [to_true_primitive, from_true_primitive] = system->get_mapping_super_alm(order);
    const auto xf_tmp = system->get_supercell(order).x_fractional;

    fcs_out.clear();

    try {
        read_xml(fname_fcs, pt);
    } catch (std::exception &e) {
        auto str_error = "Cannot open file FCSFILE ( " + fname_fcs + " )";
        exit("load_fcs_xml", str_error.c_str());
    }

    if (order == 0) {
        str_tag = "Data.ForceConstants.HARMONIC";
    } else {
        str_tag = "Data.ForceConstants.ANHARM" + std::to_string(order + 2);
    }

    if (auto child_ = pt.get_child_optional(str_tag); !child_) {
        auto str_tmp = str_tag + " flag not found in the FCSFILE file";
        exit("load_fcs_xml", str_tmp.c_str());
    }

    BOOST_FOREACH (const ptree::value_type &child_, pt.get_child(str_tag)) {
        AtomCellSuper ivec_tmp{};
        const auto &child = child_.second;

        auto fcs_val = boost::lexical_cast<double>(child.data());

        ivec_with_cell.clear();

        for (auto i = 0; i < order + 2; ++i) {
            auto str_attr = "<xmlattr>.pair" + std::to_string(i + 1);
            auto str_pairs = child.get<std::string>(str_attr);

            ss.str("");
            ss.clear();
            ss << str_pairs;

            if (i == 0) {
                ss >> atmn >> xyz;
                ivec_tmp.index = 3 * from_true_primitive[atmn - 1][0] + xyz - 1;
                ivec_tmp.cell_s = 0;
                ivec_tmp.tran = 0; // dummy
                ivec_with_cell.push_back(ivec_tmp);
            } else {
                ss >> atmn >> xyz >> cell_s;
                ivec_tmp.index = 3 * (atmn - 1) + xyz - 1;
                ivec_tmp.cell_s = cell_s - 1;
                ivec_tmp.tran = 0; // dummy
                ivec_with_cell.push_back(ivec_tmp);
            }
        }

        if (std::abs(fcs_val) > eps) {
            do {
                ivec_copy.clear();
                atoms_s_tmp.clear();

                for (auto &i: ivec_with_cell) {
                    atmn = i.index / 3;
                    xyz = i.index % 3;
                    ivec_tmp.index = 3 * to_true_primitive[atmn].atom_num + xyz;
                    ivec_tmp.cell_s = i.cell_s;
                    ivec_tmp.tran = to_true_primitive[atmn].tran_num;
                    ivec_copy.push_back(ivec_tmp);
                    atoms_s_tmp.emplace_back(atmn);
                }
                fcs_out.emplace_back(fcs_val, ivec_copy, atoms_s_tmp);
            } while (std::next_permutation(ivec_with_cell.begin() + 1, ivec_with_cell.end()));
        }
    }

    // Register relative vector information for later use.
    // The relative vectors computed here are on the Cartesian basis, which will be converted to
    // the fractional basis of the user-defined unit cell later.
    for (auto &it: fcs_out) {
        relvecs.clear();
        relvecs_velocity.clear();
        const auto atom1_s = from_true_primitive[it.pairs[0].index / 3][0];
        for (auto i = 1; i < order + 2; ++i) {
            const auto atom2_s = from_true_primitive[it.pairs[i].index / 3][it.pairs[i].tran];
            const auto atom2_s_mod = from_true_primitive[it.pairs[i].index / 3][0];
            for (auto j = 0; j < 3; ++j) {
                relvec_tmp[j] = xf_tmp(atom2_s_mod, j) + xf_image[it.pairs[i].cell_s][j] - xf_tmp(atom1_s, j);
            }
            relvec_tmp = system->get_supercell(order).lattice_vector * relvec_tmp;
            relvecs.emplace_back(relvec_tmp);

            for (auto j = 0; j < 3; ++j) {
                relvec_tmp[j] = xf_tmp(atom2_s, j) + xf_image[it.pairs[i].cell_s][j] - xf_tmp(atom1_s, j);
            }

            relvec_tmp = system->get_supercell(order).lattice_vector * relvec_tmp;
            relvecs_velocity.emplace_back(relvec_tmp);
        }
        it.relvecs = relvecs;
        it.relvecs_velocity = relvecs_velocity;
    }
}

void Fcs_phonon::parse_fcs_from_h5(const std::string &fname_fcs, const int order,
                                   std::vector<FcsArrayWithCell> &fcs_out) const
{
    // Parse the force constants from the HDF5 file.
    // The relative vectors in the Cartesian basis are loaded and set to relvec_velocity.
    // The relvec member variable is not set here, and it will be set later.

    using namespace H5Easy;
    const File file(fname_fcs, File::ReadOnly);

    // FC2_TEMPERATURE: pick one temperature row of the renormalized FC2
    // stored in an SCPH/QHA state file instead of the base values. When
    // DFC2FILE is given, FC2_TEMPERATURE refers to that correction file
    // instead and the main FC2 file is read as-is.
    if (order == 0 && !file_dfc2.empty() &&
        file.exist("/ForceConstants/Order2_temperature_dependent/force_constant_values"))
    {
        warn("parse_fcs_from_h5",
             "The harmonic FC2 source is itself an SCPH/QHA state file while DFC2FILE is also given:\n"
             " only the (coarse-mesh folded) base FC2 of this file is used here, and the anharmonic\n"
             " correction is taken from DFC2FILE. This is probably not what you want — give the\n"
             " original harmonic FC2 file instead, or drop DFC2FILE to read the total FC2 directly.");
    }
    int temperature_index = -1;
    if (order == 0 && fc2_temperature >= 0.0 && file_dfc2.empty()) {
        if (!file.exist("/ForceConstants/Order2_temperature_dependent/force_constant_values")) {
            exit("parse_fcs_from_h5",
                 "FC2_TEMPERATURE was given, but the FC2 file carries no "
                 "temperature-dependent force constants.");
        }
        temperature_index = h5_resolve_temperature_index(file, fc2_temperature, eps6, "/settings/temperatures");

        // Refuse renormalized FC2 from unconverged SCPH/structural
        // iterations unless the user opted in (absent /convergence data,
        // e.g. a legacy import, cannot be checked and is accepted).
        const auto iteration_converged = [&file, temperature_index](const std::string &name) {
            if (!file.exist("/convergence/" + name)) return true;
            std::vector<unsigned char> flags;
            file.getDataSet("/convergence/" + name).read(flags);
            return static_cast<size_t>(temperature_index) >= flags.size() || flags[temperature_index] != 0;
        };
        if (!iteration_converged("scph") || !iteration_converged("structure")) {
            if (run.allow_unconverged) {
                warn("parse_fcs_from_h5",
                     "The iterations at FC2_TEMPERATURE did not converge;\n"
                     " using the renormalized FC2 anyway because ALLOW_UNCONVERGED = 1.");
            } else {
                exit("parse_fcs_from_h5",
                     "The SCPH iteration or structural optimization at FC2_TEMPERATURE did not converge\n"
                     " in the run that produced this state file. Reconverge it (MAXITER, MAX_STR_ITER, ...)\n"
                     " or set ALLOW_UNCONVERGED = 1 in &general to use the data anyway.");
            }
        }

        if (run.verbosity > 0)
            std::cout << "\n  FC2_TEMPERATURE = " << fc2_temperature
                      << " K : loading the renormalized FC2 at this temperature from " << fname_fcs << "\n  ";
    }

    parse_fcs_from_h5(file, "", order, fcs_out, temperature_index);
}

void Fcs_phonon::parse_fcs_from_h5(const HighFive::File &file, const std::string &group_prefix, const int order,
                                   std::vector<FcsArrayWithCell> &fcs_out, const int temperature_index) const
{
    Eigen::MatrixXi atom_indices, atom_indices_super, coord_indices;
    Eigen::MatrixXd shift_vectors;
    Eigen::ArrayXd fcs_values;
    std::string unit_shift, unit_fc;

    get_force_constants_from_h5(file,
                                order,
                                atom_indices,
                                atom_indices_super,
                                coord_indices,
                                shift_vectors,
                                fcs_values,
                                &unit_shift,
                                &unit_fc,
                                temperature_index,
                                group_prefix);

    // The helper already converted the data; report only when the stored unit
    // differs from the internal one to keep the common case quiet.
    const std::string unit_fc_internal = "Ry/bohr^" + std::to_string(order + 2);
    if (!unit_fc.empty() && unit_fc != unit_fc_internal) {
        if (run.verbosity > 0)
            std::cout << "\n  " << file.getName() << group_prefix << " [Order " << order + 2 << "]: stored unit "
                      << unit_fc << " -> converted to " << unit_fc_internal << '\n';
    }

    const auto nentries = fcs_values.size();

    std::vector<AtomCellSuper> ivec_with_cell, ivec_copy;
    std::vector<Eigen::Vector3d> relvecs_tmp;
    std::vector<unsigned int> atoms_s_tmp;

    struct IndexAndRelvecs
    {
        unsigned int index_super;
        unsigned int index_prim;
        Eigen::Vector3d relvec_vel;
    };

    const Eigen::Vector3d zerovec = Eigen::Vector3d::Zero();

    const auto nelems = order + 2;
    IndexAndRelvecs index_tmp{};
    std::vector<IndexAndRelvecs> vec_index(nelems);

    for (auto i = 0; i < nentries; ++i) {

        if (std::abs(fcs_values[i]) < eps) continue;

        vec_index.clear();

        index_tmp.index_prim = 3 * atom_indices(i, 0) + coord_indices(i, 0);
        index_tmp.index_super = 3 * atom_indices_super(i, 0) + coord_indices(i, 0);
        index_tmp.relvec_vel = zerovec;
        vec_index.emplace_back(index_tmp);

        for (auto j = 1; j < nelems; ++j) {
            index_tmp.index_prim = 3 * atom_indices(i, j) + coord_indices(i, j);
            index_tmp.index_super = 3 * atom_indices_super(i, j) + coord_indices(i, j);
            for (auto k = 0; k < 3; ++k) {
                index_tmp.relvec_vel[k] = shift_vectors(i, 3 * (j - 1) + k);
            }
            vec_index.emplace_back(index_tmp);
        }

        do {
            ivec_copy.clear();
            relvecs_tmp.clear();
            atoms_s_tmp.clear();

            for (auto &j: vec_index) {
                AtomCellSuper ivec_tmp{};
                ivec_tmp.index = j.index_prim;
                ivec_tmp.cell_s = 0; // no information about the cell shift
                ivec_tmp.tran = 0;   // no information about the translation
                ivec_copy.push_back(ivec_tmp);
                atoms_s_tmp.emplace_back(j.index_super / 3);
            }
            for (auto j = 1; j < vec_index.size(); ++j) {
                relvecs_tmp.emplace_back(vec_index[j].relvec_vel);
            }
            fcs_out.emplace_back(fcs_values[i], ivec_copy, atoms_s_tmp, relvecs_tmp);
        } while (std::next_permutation(
            vec_index.begin() + 1,
            vec_index.end(),
            [](const IndexAndRelvecs &a, const IndexAndRelvecs &b) { return a.index_super < b.index_super; }));
    }
}


void Fcs_phonon::read_delta_fc2_from_scph(const std::string &fname_dfc2)
{
    using namespace H5Easy;
    const File file(fname_dfc2, File::ReadOnly);

    check_h5_schema(file, h5_schema_scph_state, h5_version_scph_state);

    if (fc2_temperature < 0.0) {
        exit("read_delta_fc2_from_scph", "FC2_TEMPERATURE must be given together with DFC2FILE.");
    }
    const auto itemp = h5_resolve_temperature_index(file, fc2_temperature, eps6, "/settings/temperatures");

    // Same convergence guard as the direct FC2_TEMPERATURE read.
    const auto iteration_converged = [&file, itemp](const std::string &name) {
        if (!file.exist("/convergence/" + name)) return true;
        std::vector<unsigned char> flags;
        file.getDataSet("/convergence/" + name).read(flags);
        return static_cast<size_t>(itemp) >= flags.size() || flags[itemp] != 0;
    };
    if (!iteration_converged("scph") || !iteration_converged("structure")) {
        if (run.allow_unconverged) {
            warn("read_delta_fc2_from_scph",
                 "The iterations at FC2_TEMPERATURE did not converge;\n"
                 " using the FC2 correction anyway because ALLOW_UNCONVERGED = 1.");
        } else {
            exit("read_delta_fc2_from_scph",
                 "The SCPH iteration or structural optimization at FC2_TEMPERATURE did not converge\n"
                 " in the run that produced DFC2FILE. Reconverge it (MAXITER, MAX_STR_ITER, ...)\n"
                 " or set ALLOW_UNCONVERGED = 1 in &general to use the data anyway.");
        }
    }

    // Fold SCPH correction atoms onto the current primitive cell and count
    // translationally equivalent rows once. The SCPH cell may be an integer
    // supercell of the current cell.
    std::vector<int> map_dfc2_to_prim;
    {
        Eigen::Matrix3d lavec_dfc2;
        Eigen::MatrixXd xf_dfc2;
        std::vector<int> kinds_dfc2;
        std::vector<std::string> elems_dfc2;
        get_structures_from_h5(file, "PrimitiveCell", lavec_dfc2, xf_dfc2, kinds_dfc2, elems_dfc2);
        const auto &primcell = system->get_primcell();
        const Eigen::Matrix3d lavec_prim_inv = primcell.lattice_vector.inverse();
        const Eigen::Matrix3d transmat = lavec_prim_inv * lavec_dfc2;
        const Eigen::Matrix3d transmat_int =
            transmat.unaryExpr([](const double x) { return static_cast<double>(nint(x)); });
        if ((transmat - transmat_int).cwiseAbs().maxCoeff() > eps4) {
            exit("read_delta_fc2_from_scph",
                 "The cell of DFC2FILE is not an integer supercell of the present primitive cell.");
        }
        const auto ncopy = nint(std::abs(transmat_int.determinant()));
        if (static_cast<size_t>(xf_dfc2.rows()) != ncopy * primcell.number_of_atoms) {
            exit("read_delta_fc2_from_scph",
                 "The number of atoms in the DFC2FILE cell is inconsistent with the present primitive cell.");
        }
        map_dfc2_to_prim.assign(xf_dfc2.rows(), -1);
        for (Eigen::Index i = 0; i < xf_dfc2.rows(); ++i) {
            const Eigen::Vector3d xf = lavec_prim_inv * (lavec_dfc2 * xf_dfc2.row(i).transpose());
            for (size_t p = 0; p < primcell.number_of_atoms; ++p) {
                Eigen::Vector3d xdiff = xf - primcell.x_fractional.row(p).transpose();
                xdiff = xdiff.unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });
                if ((primcell.lattice_vector * xdiff).norm() < 1.0e-3) {
                    map_dfc2_to_prim[i] = static_cast<int>(p);
                    break;
                }
            }
            if (map_dfc2_to_prim[i] < 0) {
                exit("read_delta_fc2_from_scph",
                     "An atom of the DFC2FILE cell has no counterpart in the present primitive cell.");
            }
        }
        if (ncopy > 1 && run.verbosity > 0) {
            std::cout << "\n  DFC2FILE: the SCPH cell holds " << ncopy
                      << " copies of the present primitive cell; corrections are folded.";
        }
    }

    // Delta = total(T) - base, both converted to internal units by the reader.
    Eigen::MatrixXi atom_indices, atom_indices_super, coord_indices;
    Eigen::MatrixXd shift_vectors;
    Eigen::ArrayXd fcs_base, fcs_total;
    get_force_constants_from_h5(file, 0, atom_indices, atom_indices_super, coord_indices, shift_vectors, fcs_base);
    get_force_constants_from_h5(file,
                                0,
                                atom_indices,
                                atom_indices_super,
                                coord_indices,
                                shift_vectors,
                                fcs_total,
                                nullptr,
                                nullptr,
                                itemp);
    const Eigen::ArrayXd delta = fcs_total - fcs_base;

    // Keep one row per (atom1, coord1, atom2, coord2, relvec) of the *present*
    // primitive cell. Copies of that cell inside the SCPH cell must agree;
    // the rows are turned into force constants only after replication
    // (append_delta_fc2_rows), because replicate_force_constant would impose
    // the translations of the FCS-file primitive cell, which a relaxed SCPH
    // cell (a cell-doubling distortion, say) need not respect.
    std::map<std::tuple<int, int, int, int, long, long, long>, double> seen;

    dfc2_rows.clear();
    size_t nfolded = 0;
    for (Eigen::Index irow = 0; irow < delta.size(); ++irow) {
        const auto iat = map_dfc2_to_prim[atom_indices(irow, 0)];
        const auto jat = map_dfc2_to_prim[atom_indices(irow, 1)];

        Eigen::Vector3d relvec;
        for (auto k = 0; k < 3; ++k) relvec[k] = shift_vectors(irow, k);

        const auto key = std::make_tuple(iat,
                                         coord_indices(irow, 0),
                                         jat,
                                         coord_indices(irow, 1),
                                         static_cast<long>(nint(relvec[0] * 1.0e4)),
                                         static_cast<long>(nint(relvec[1] * 1.0e4)),
                                         static_cast<long>(nint(relvec[2] * 1.0e4)));
        const auto it = seen.find(key);
        if (it != seen.end()) {
            if (std::abs(it->second - delta[irow]) > 1.0e-8 * std::max(1.0, std::abs(it->second))) {
                exit("read_delta_fc2_from_scph",
                     "Translationally equivalent correction rows of DFC2FILE carry different values;\n"
                     " the SCPH cell does not respect the translations of the present primitive cell.");
            }
            ++nfolded;
            continue;
        }
        seen.emplace(key, delta[irow]);
        // Zero rows are compared above (a zero copy against a nonzero one is
        // an error) but not kept.
        if (std::abs(delta[irow]) < eps) continue;
        dfc2_rows.push_back({iat, coord_indices(irow, 0), jat, coord_indices(irow, 1), relvec, delta[irow]});
    }

    if (run.verbosity > 0) {
        std::cout << "\n  DFC2FILE: " << dfc2_rows.size() << " anharmonic FC2 correction rows at " << fc2_temperature
                  << " K from " << fname_dfc2;
        if (nfolded > 0) std::cout << " (" << nfolded << " translational duplicates folded)";
        std::cout << "\n  ";
    }
}

void Fcs_phonon::append_delta_fc2_rows(std::vector<FcsArrayWithCell> &fc2_inout)
{
    // Broadcast the rows read on rank 0 (flattened), then build replicated FC2
    // entries on every rank, in the layout replicate_force_constant produces:
    // pairs[].index over the present primitive cell, pairs[].tran of the
    // supercell image, relvecs and relvecs_velocity in its lattice basis.
    int nrows = static_cast<int>(dfc2_rows.size());
    MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (nrows == 0) return;

    std::vector<int> ints(4 * static_cast<size_t>(nrows));
    std::vector<double> reals(4 * static_cast<size_t>(nrows));
    if (run.my_rank == 0) {
        for (size_t i = 0; i < dfc2_rows.size(); ++i) {
            const auto &r = dfc2_rows[i];
            ints[4 * i] = r.iat;
            ints[4 * i + 1] = r.coord1;
            ints[4 * i + 2] = r.jat;
            ints[4 * i + 3] = r.coord2;
            for (auto k = 0; k < 3; ++k) reals[4 * i + k] = r.relvec[k];
            reals[4 * i + 3] = r.value;
        }
    }
    MPI_Bcast(ints.data(), 4 * nrows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(reals.data(), 4 * nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const auto &scell = system->get_supercell(0);
    const auto &map_p2s = system->get_map_p2s(0);
    const auto &map_s2p = system->get_map_s2p(0);
    const Eigen::Matrix3d lavec_super_inv = scell.lattice_vector.inverse();
    const Eigen::Matrix3d convmat = system->get_primcell().lattice_vector.inverse();

    std::vector<AtomCellSuper> pairs(2);
    std::vector<unsigned int> atoms_s(2);
    std::vector<Eigen::Vector3d> relvecs(1), relvecs_vel(1);

    for (auto i = 0; i < nrows; ++i) {
        const auto iat = ints[4 * i];
        const auto jat = ints[4 * i + 2];
        const Eigen::Vector3d relvec(reals[4 * i], reals[4 * i + 1], reals[4 * i + 2]);

        // First atom: the translation-0 image of iat, as after replication.
        const auto atom1_s = map_p2s[iat][0];
        const Eigen::Vector3d xf_target = lavec_super_inv * (scell.x_cartesian.row(atom1_s).transpose() + relvec);
        int atom2_s = -1;
        for (const auto &cand: map_p2s[jat]) {
            Eigen::Vector3d xdiff = xf_target - scell.x_fractional.row(cand).transpose();
            xdiff = xdiff.unaryExpr([](const double x) { return x - static_cast<double>(nint(x)); });
            if ((scell.lattice_vector * xdiff).norm() < 1.0e-3) {
                atom2_s = static_cast<int>(cand);
                break;
            }
        }
        if (atom2_s == -1) {
            exit("append_delta_fc2_rows",
                 "A correction row of DFC2FILE has no matching atom in the present supercell.\n"
                 " The supercells of DFC2FILE and the harmonic FC2 file are probably incommensurate.");
        }

        pairs[0].index = 3 * iat + ints[4 * i + 1];
        pairs[0].tran = map_s2p[atom1_s].tran_num;
        pairs[0].cell_s = 0;
        pairs[1].index = 3 * jat + ints[4 * i + 3];
        pairs[1].tran = map_s2p[atom2_s].tran_num;
        pairs[1].cell_s = 0;
        atoms_s[0] = atom1_s;
        atoms_s[1] = static_cast<unsigned int>(atom2_s);
        // Same conventions as replicate_force_constant.
        relvecs[0] = convmat * (relvec + scell.x_cartesian.row(atom1_s).transpose() -
                                scell.x_cartesian.row(map_p2s[jat][0]).transpose());
        relvecs_vel[0] = convmat * relvec;
        fc2_inout.emplace_back(reals[4 * i + 3], pairs, atoms_s, relvecs, relvecs_vel);
    }
    dfc2_rows.clear();
}


double Fcs_phonon::examine_translational_invariance(const int order, const unsigned int nat, const unsigned int natmin,
                                                    const std::vector<FcsArrayWithCell> &fc_in)
{
    size_t j, k, l, m;

    double dev;

    double ret = 0.0;

    const auto nat3 = 3 * nat;
    const auto natmin3 = 3 * natmin;

    switch (order) {
    case 0:
        {
            NDArray<double, 2> sum2;
            sum2.resize(natmin3, 3);

            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < 3; ++k) {
                    sum2[j][k] = 0.0;
                }
            }

            for (const auto &it: fc_in) {
                j = it.pairs[0].index;
                k = it.pairs[1].index % 3;
                sum2[j][k] += it.fcs_val;
            }

            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < 3; ++k) {
                    dev = std::abs(sum2[j][k]);
                    ret = std::max(ret, dev);
                }
            }
            sum2.clear();
            break;
        }
    case 1:
        {
            NDArray<double, 3> sum3;
            sum3.resize(3 * natmin, 3 * nat, 3);

            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < nat3; ++k) {
                    for (l = 0; l < 3; ++l) {
                        sum3[j][k][l] = 0.0;
                    }
                }
            }

            for (const auto &it: fc_in) {
                j = it.pairs[0].index;
                k = 3 * it.atoms_s[1] + it.pairs[1].index % 3;
                l = it.pairs[2].index % 3;
                sum3[j][k][l] += it.fcs_val;
            }
            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < nat3; ++k) {
                    for (l = 0; l < 3; ++l) {
                        dev = std::abs(sum3[j][k][l]);
                        ret = std::max(ret, dev);
                    }
                }
            }
            sum3.clear();
            break;
        }
    case 2:
        {
            NDArray<double, 4> sum4;
            sum4.resize(natmin3, nat3, nat3, 3);

            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < nat3; ++k) {
                    for (l = 0; l < nat3; ++l) {
                        for (m = 0; m < 3; ++m) {
                            sum4[j][k][l][m] = 0.0;
                        }
                    }
                }
            }

            for (const auto &it: fc_in) {
                j = it.pairs[0].index;
                k = 3 * it.atoms_s[1] + it.pairs[1].index % 3;
                l = 3 * it.atoms_s[2] + it.pairs[2].index % 3;
                m = it.pairs[3].index % 3;
                sum4[j][k][l][m] += it.fcs_val;
            }

            for (j = 0; j < natmin3; ++j) {
                for (k = 0; k < nat3; ++k) {
                    for (l = 0; l < nat3; ++l) {
                        for (m = 0; m < 3; ++m) {
                            dev = std::abs(sum4[j][k][l][m]);
                            ret = std::max(ret, dev);
                        }
                    }
                }
            }
            sum4.clear();
            break;
        }
    default:
        break;
    }

    return ret;
}

void Fcs_phonon::MPI_Bcast_fcs_array(const unsigned int N)
{
    int j, k;
    NDArray<double, 1> fcs_tmp;
    NDArray<unsigned int, 3> ind;
    NDArray<double, 3> relative_vector_tmp;

    std::vector<AtomCellSuper> ivec_array;
    std::vector<unsigned int> atoms_s_tmp;

    std::vector<Eigen::Vector3d> relvecs_vel;
    Eigen::Vector3d relvec_tmp;

    for (unsigned int i = 0; i < N; ++i) {

        int len = force_constant_with_cell[i].size();
        int const nelem = i + 2;

        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (len == 0) continue;

        fcs_tmp.resize(len);
        ind.resize(len, nelem, 4);
        relative_vector_tmp.resize(len, nelem - 1, 3);

        if (run.my_rank == 0) {
            for (j = 0; j < len; ++j) {
                fcs_tmp[j] = force_constant_with_cell[i][j].fcs_val;
                for (k = 0; k < nelem; ++k) {
                    ind[j][k][0] = force_constant_with_cell[i][j].pairs[k].index;
                    ind[j][k][1] = force_constant_with_cell[i][j].pairs[k].tran;
                    ind[j][k][2] = force_constant_with_cell[i][j].pairs[k].cell_s;
                    ind[j][k][3] = force_constant_with_cell[i][j].atoms_s[k];
                }
                for (k = 0; k < nelem - 1; ++k) {
                    for (auto l = 0; l < 3; ++l) {
                        relative_vector_tmp[j][k][l] = force_constant_with_cell[i][j].relvecs_velocity[k][l];
                    }
                }
            }
        }

        MPI_Bcast(&fcs_tmp[0], len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ind[0][0][0], 4 * nelem * len, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Bcast(&relative_vector_tmp[0][0][0], 3 * len * (nelem - 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (run.my_rank > 0) {
            force_constant_with_cell[i].clear();

            for (j = 0; j < len; ++j) {

                ivec_array.clear();
                atoms_s_tmp.clear();
                for (k = 0; k < nelem; ++k) {
                    AtomCellSuper ivec_tmp{};
                    ivec_tmp.index = ind[j][k][0];
                    ivec_tmp.tran = ind[j][k][1];
                    ivec_tmp.cell_s = ind[j][k][2];
                    ivec_array.push_back(ivec_tmp);
                    atoms_s_tmp.emplace_back(ind[j][k][3]);
                }

                relvecs_vel.clear();
                for (k = 0; k < nelem - 1; ++k) {
                    for (auto l = 0; l < 3; ++l) {
                        relvec_tmp[l] = relative_vector_tmp[j][k][l];
                    }
                    relvecs_vel.emplace_back(relvec_tmp);
                }
                force_constant_with_cell[i].emplace_back(fcs_tmp[j], ivec_array, atoms_s_tmp, relvecs_vel);
            }
        }

        fcs_tmp.clear();
        ind.clear();
        relative_vector_tmp.clear();
    }
}
