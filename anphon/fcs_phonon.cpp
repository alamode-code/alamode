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
#include <string>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "gruneisen.h"
#include "hdf5_parser.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonons.h"
#include "system.h"
#include "thermodynamics.h"

using namespace PHON_NS;

Fcs_phonon::Fcs_phonon(PHON *phon) : Pointers(phon)
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

void Fcs_phonon::setup(const std::string &mode)
{
    if (mympi->my_rank == 0) {
        std::cout << " =================\n";
        std::cout << "  Force Constants \n";
        std::cout << " =================\n\n";
    }

    MPI_Bcast(&anharmonic_core->quartic_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gruneisen->print_gruneisen, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&thermodynamics->calc_FE_bubble, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (mode == "PHONONS") {
        require_cubic = false;
        require_quartic = false;
        maxorder = 1;

        if (gruneisen->print_gruneisen || thermodynamics->calc_FE_bubble) {
            require_cubic = true;
            maxorder = 2;
        }
        if (gruneisen->print_newfcs) {
            require_cubic = true;
            maxorder = 2;

            if (anharmonic_core->quartic_mode > 0) {
                require_quartic = true;
                maxorder = 3;
            }
        }

    } else if (mode == "KAPPA") {
        require_cubic = true;

        if (anharmonic_core->quartic_mode > 0) {
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

    force_constant_with_cell.resize(maxorder);

    if (mympi->my_rank == 0) {

        load_fcs_from_file(maxorder);

        for (auto i = 0; i < maxorder; ++i) {
            std::cout << "  Number of non-zero IFCs for " << i + 2 << " order: ";
            std::cout << force_constant_with_cell[i].size() << '\n';
        }
        std::cout << '\n';

        std::cout << "  Maximum deviation from the translational invariance: \n";
        for (auto i = 0; i < maxorder; ++i) {
            const auto maxdev = examine_translational_invariance(i,
                                                                 system->get_supercell(i).number_of_atoms,
                                                                 system->get_primcell().number_of_atoms,
                                                                 force_constant_with_cell[i]);
            std::cout << "   Order " << i + 2 << " : " << std::setw(12) << std::scientific << maxdev << '\n';
        }
        std::cout << '\n';
    }

    MPI_Bcast_fcs_array(maxorder);
    replicate_force_constants(maxorder);
}

void Fcs_phonon::replicate_force_constants(const int maxorder_in)
{
    for (auto order = 0; order < maxorder_in; ++order) {
        replicate_force_constant(system.get(), force_constant_with_cell[order]);
    }
}

void Fcs_phonon::replicate_force_constant(const System *system_in, std::vector<FcsArrayWithCell> &fcs_inout) const
{
    // This function does the following tasks:
    // 1. Replicates the force constants originally computed for \Phi_{ij}, \Phi_{ijk}, ...,
    //    where atom i belongs to the (true) primitive cell to all pairs where atom i belongs to
    //    the user-defined unit cell.
    // 2. Relative vector basis is transformed from the Cartesian to the lattice vector basis
    //    of the user-defined unit cell.
    // 3. Relative vector (relvec) member function is computed from relvec_velocity.

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
                                    cell_tmp.x_cartesian(system->get_map_p2s(order)[atom_new_prim[i + 1]][0], j);
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

    std::cout << "  Reading force constants from the FCSFILE ... ";

    for (auto i = 0; i < filename_list.size(); ++i) {

        if (!load_flags[i]) continue;

        const auto &filename = filename_list[i];
        const auto file_extension = filename.substr(filename.find_last_of('.') + 1);
        if (file_extension == "xml" || file_extension == "XML") {

            load_fcs_xml(filename, i, force_constant_with_cell[i]);

        } else if (file_extension == "h5" || file_extension == "hdf5") {

            parse_fcs_from_h5(filename, i, force_constant_with_cell[i]);
        }

        // Legacy dfc2.py workflow, native: add the (short-ranged) anharmonic
        // FC2 correction of an SCPH/QHA state file onto the harmonic FC2 of
        // a possibly larger supercell.
        if (i == 0 && !file_dfc2.empty()) {
            append_delta_fc2_from_scph(file_dfc2, force_constant_with_cell[i]);
        }
    }

    std::cout << "done.\n\n";
}

void Fcs_phonon::get_fcs_from_file(const std::string &fname_fcs, const int order,
                                   std::vector<FcsArrayWithCell> &fcs_out) const
{
    const auto file_extension = fname_fcs.substr(fname_fcs.find_last_of('.') + 1);
    if (file_extension == "xml" || file_extension == "XML") {
        load_fcs_xml(fname_fcs, order, fcs_out);
    } else if (file_extension == "h5" || file_extension == "hdf5") {
        parse_fcs_from_h5(fname_fcs, order, fcs_out);
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

    const auto xf_image = dynamical->get_xrs_image();
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

    Eigen::MatrixXi atom_indices, atom_indices_super, coord_indices;
    Eigen::MatrixXd shift_vectors;
    Eigen::ArrayXd fcs_values;
    std::string unit_shift, unit_fc;

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
            if (input->allow_unconverged) {
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

        std::cout << "\n  FC2_TEMPERATURE = " << fc2_temperature
                  << " K : loading the renormalized FC2 at this temperature from " << fname_fcs << "\n  ";
    }

    get_force_constants_from_h5(file,
                                order,
                                atom_indices,
                                atom_indices_super,
                                coord_indices,
                                shift_vectors,
                                fcs_values,
                                &unit_shift,
                                &unit_fc,
                                temperature_index);

    // The helper already converted the data; report only when the stored unit
    // differs from the internal one to keep the common case quiet.
    const std::string unit_fc_internal = "Ry/bohr^" + std::to_string(order + 2);
    if (!unit_fc.empty() && unit_fc != unit_fc_internal) {
        std::cout << "\n  " << fname_fcs << " [Order " << order + 2 << "]: stored unit " << unit_fc
                  << " -> converted to " << unit_fc_internal << '\n';
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


void Fcs_phonon::append_delta_fc2_from_scph(const std::string &fname_dfc2, std::vector<FcsArrayWithCell> &fcs_out) const
{
    using namespace H5Easy;
    const File file(fname_dfc2, File::ReadOnly);

    check_h5_schema(file, h5_schema_scph_state, h5_version_scph_state);

    if (fc2_temperature < 0.0) {
        exit("append_delta_fc2_from_scph", "FC2_TEMPERATURE must be given together with DFC2FILE.");
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
        if (input->allow_unconverged) {
            warn("append_delta_fc2_from_scph",
                 "The iterations at FC2_TEMPERATURE did not converge;\n"
                 " using the FC2 correction anyway because ALLOW_UNCONVERGED = 1.");
        } else {
            exit("append_delta_fc2_from_scph",
                 "The SCPH iteration or structural optimization at FC2_TEMPERATURE did not converge\n"
                 " in the run that produced DFC2FILE. Reconverge it (MAXITER, MAX_STR_ITER, ...)\n"
                 " or set ALLOW_UNCONVERGED = 1 in &general to use the data anyway.");
        }
    }

    // The correction rows are indexed in the primitive cell of the SCPH run;
    // it must be the same primitive cell as the present calculation.
    {
        Eigen::Matrix3d lavec_dfc2;
        Eigen::MatrixXd xf_dfc2;
        std::vector<int> kinds_dfc2;
        std::vector<std::string> elems_dfc2;
        get_structures_from_h5(file, "PrimitiveCell", lavec_dfc2, xf_dfc2, kinds_dfc2, elems_dfc2);
        const auto &primcell = system->get_primcell();
        if (static_cast<size_t>(xf_dfc2.rows()) != primcell.number_of_atoms) {
            exit("append_delta_fc2_from_scph", "The primitive cell of DFC2FILE differs from the present one.");
        }
        if ((lavec_dfc2 - primcell.lattice_vector).cwiseAbs().maxCoeff() > eps4) {
            warn("append_delta_fc2_from_scph",
                 "The primitive lattice vectors of DFC2FILE differ from the present ones.");
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

    // Re-express each row in the supercell of the harmonic FC2 file: the
    // Cartesian relative vector is physical and carries over unchanged; the
    // second atom is located in the present supercell by position matching
    // (both cells tile the same primitive cell, so a match must exist when
    // the two supercells are commensurate).
    const auto &scell = system->get_supercell(0);
    const auto &map_p2s = system->get_map_p2s(0);
    const Eigen::Matrix3d lavec_super_inv = scell.lattice_vector.inverse();

    std::vector<AtomCellSuper> ivec_pair(2);
    std::vector<unsigned int> atoms_s_pair(2);
    std::vector<Eigen::Vector3d> relvecs_pair(1);

    size_t nadded = 0;
    for (Eigen::Index irow = 0; irow < delta.size(); ++irow) {
        if (std::abs(delta[irow]) < eps) continue;

        const auto iat = atom_indices(irow, 0);
        const auto jat = atom_indices(irow, 1);
        const auto atom1_s = map_p2s[iat][0];

        Eigen::Vector3d relvec;
        for (auto k = 0; k < 3; ++k) relvec[k] = shift_vectors(irow, k);

        const Eigen::Vector3d target = scell.x_cartesian.row(atom1_s).transpose() + relvec;
        const Eigen::Vector3d xf_target = lavec_super_inv * target;

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
            exit("append_delta_fc2_from_scph",
                 "A correction row of DFC2FILE has no matching atom in the present supercell.\n"
                 " The supercells of DFC2FILE and the harmonic FC2 file are probably incommensurate.");
        }

        ivec_pair[0].index = 3 * iat + coord_indices(irow, 0);
        ivec_pair[0].cell_s = 0;
        ivec_pair[0].tran = 0;
        ivec_pair[1].index = 3 * jat + coord_indices(irow, 1);
        ivec_pair[1].cell_s = 0;
        ivec_pair[1].tran = 0;
        atoms_s_pair[0] = atom1_s;
        atoms_s_pair[1] = static_cast<unsigned int>(atom2_s);
        relvecs_pair[0] = relvec;

        fcs_out.emplace_back(delta[irow], ivec_pair, atoms_s_pair, relvecs_pair);
        ++nadded;
    }

    std::cout << "\n  DFC2FILE: added " << nadded << " anharmonic FC2 correction rows at " << fc2_temperature
              << " K from " << fname_dfc2 << "\n  ";
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

        if (mympi->my_rank == 0) {
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

        if (mympi->my_rank > 0) {
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
