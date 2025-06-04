/*
 constraint.cpp

 Copyright (c) 2014-2022 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "constraint.h"
#include "combination.h"
#include "error.h"
#include "fcs.h"
#include "cluster.h"
#include "memory.h"
#include "rref.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"
#include "xml_parser.h"
#include "hdf5_parser.h"
#include "svd.h"
#include <iostream>
#include <iomanip>
#include <boost/bimap.hpp>
#include <algorithm>
#include <unordered_set>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <highfive/H5Easy.hpp>

using namespace ALM_NS;

Constraint::Constraint()
{
    set_default_variables();
}

Constraint::~Constraint()
{
    deallocate_variables();
}

void Constraint::set_default_variables()
{
    constraint_mode = 11;
    rotation_axis = "";
    fix_harmonic = false;
    fix_cubic = false;
    constraint_algebraic = 1;
    fc2_file = "";
    fc3_file = "";
    const_mat = nullptr;
    const_rhs = nullptr;
    index_bimap = nullptr;
    number_of_constraints = 0;
    tolerance_constraint = eps8;
    status_constraint_subset["symmetry"] = 0;
    status_constraint_subset["translation"] = -1;
    status_constraint_subset["rotation"] = -1;
    status_constraint_subset["rotation_extra"] = -1;
    status_constraint_subset["fix2"] = -1;
    status_constraint_subset["fix3"] = -1;
    status_constraint_subset["huang"] = -1;
}

void Constraint::deallocate_variables()
{
    if (index_bimap) {
        deallocate(index_bimap);
    }
    if (const_mat) {
        deallocate(const_mat);
    }
    if (const_rhs) {
        deallocate(const_rhs);
    }
}

void Constraint::setup(const std::unique_ptr<System> &system,
                       const std::unique_ptr<Fcs> &fcs,
                       const std::unique_ptr<Cluster> &cluster,
                       const std::unique_ptr<Symmetry> &symmetry,
                       const int linear_model,
                       const int periodic_image_conv,
                       const int verbosity,
                       std::unique_ptr<Timer> &timer)
{
    timer->start_clock("constraint");

    if (verbosity > 0) {
        std::cout << " ============\n";
        std::cout << "  CONSTRAINT \n";
        std::cout << " ============\n\n";
    }

    constraint_mode = constraint_mode % 10;

    if (linear_model >= 2) {
        if (constraint_mode > 1) {
            warn("Constraint::setup",
                 "Sorry, only ICONST = 11 is supported \n"
                 "                      when LMODEL = enet. We set ICONST = 11 in this run.\n");
            constraint_mode = 1;
        }
        constraint_algebraic = 1;
    }

    switch (constraint_mode) {
    case 0: // do nothing
        impose_inv_T = false;
        impose_inv_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 0);
        set_constraint_flag("rotation", 0);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 0: Constraint for translational/rotational invariance\n";
            std::cout << "              will NOT be considered.\n";
        }
        break;
    case 1:
        impose_inv_T = true;
        impose_inv_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 0);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 1: Constraints for translational invariance\n";
            std::cout << "              will be considered.\n";
        }
        break;
    case 2:
        impose_inv_T = true;
        impose_inv_R = true;
        exclude_last_R = true;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 2: Constraints for translational and rotational invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
            std::cout << "              Rotational invariance of the maximum order will be neglected\n";
        }
        break;
    case 3:
        impose_inv_T = true;
        impose_inv_R = true;
        exclude_last_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 1);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 3: Constraints for translational and rotational invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
        }
        break;
    case 4:
        impose_inv_T = true;
        impose_inv_R = true;
        impose_inv_Huang = true;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 1);
        if (verbosity > 0) {
            std::cout << "  ICONST = 4: Constraints for translational, rotational, and Huang invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
        }
        break;
    default:
        exit("Constraint::setup", "invalid constraint_mode", constraint_mode);
        break;
    }

    if (fcs->get_forceconstant_basis() == "Lattice" && impose_inv_R) {
        exit("Constraint::setup()",
             "Sorry, rotational invariance with FCSYM_BASIS = Lattice is "
             "not supported.\n Use FCSYM_BASIS = Cartesian instead.");
    }
    if (verbosity > 0) std::cout << '\n';

    if (fix_harmonic) {
        if (verbosity > 0) {
            std::cout << "  FC2FIX is given : Harmonic force constants will be \n";
            std::cout << "                    fixed to the values given in " << fc2_file << "\n\n";
        }
        std::vector<std::vector<int>> intpair_fix;
        std::vector<double> values_fix;
        get_forceconstants_from_file(0,
                                     symmetry,
                                     fcs,
                                     fc2_file,
                                     intpair_fix,
                                     values_fix);

        set_forceconstants_to_fix(intpair_fix, values_fix);
    }

    fix_cubic = fix_cubic & (cluster->get_maxorder() > 1);
    if (fix_cubic) {
        if (verbosity > 0) {
            std::cout << "  FC3FIX is given : Cubic force constants will be \n";
            std::cout << "                    fixed to the values given in " << fc3_file << "\n\n";
        }
        std::vector<std::vector<int>> intpair_fix;
        std::vector<double> values_fix;
        get_forceconstants_from_file(1,
                                     symmetry,
                                     fcs,
                                     fc3_file,
                                     intpair_fix,
                                     values_fix);

        set_forceconstants_to_fix(intpair_fix, values_fix);
    }

    status_constraint_subset["symmetry"] = 0;

    update_constraint_matrix(system,
                             symmetry,
                             cluster,
                             fcs,
                             verbosity,
                             periodic_image_conv);

    if (verbosity > 0) {
        print_constraint_information(cluster);
        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------" << '\n';
        std::cout << '\n';
    }

    timer->stop_clock("constraint");
}

void Constraint::update_constraint_matrix(const std::unique_ptr<System> &system,
                                          const std::unique_ptr<Symmetry> &symmetry,
                                          const std::unique_ptr<Cluster> &cluster,
                                          const std::unique_ptr<Fcs> &fcs,
                                          const int verbosity,
                                          const int periodic_image_conv)
{
    const auto maxorder = cluster->get_maxorder();
    // const_symmetry is updated.
    if (const_symmetry.size() != maxorder) const_symmetry.resize(maxorder);

    if (status_constraint_subset["symmetry"] == 0) {
        generate_symmetry_constraint(system->get_supercell().number_of_atoms,
                                     symmetry,
                                     cluster,
                                     fcs,
                                     verbosity);
    }

    // const_translation is updated.
    if (const_translation.size() != maxorder) const_translation.resize(maxorder);

    if (status_constraint_subset["translation"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_translation[order].clear();
            const_translation[order].shrink_to_fit();
        }
    }
    if (status_constraint_subset["translation"] == 0) {
        generate_translational_constraint(system->get_supercell(),
                                          symmetry,
                                          cluster,
                                          fcs,
                                          periodic_image_conv,
                                          verbosity);
    }

    // const_rotation_self and const_rotation_cross are updated.
    if (const_rotation_self.size() != maxorder) const_rotation_self.resize(maxorder);
    if (const_rotation_cross.size() != maxorder) const_rotation_cross.resize(maxorder);

    if (status_constraint_subset["rotation"] == -1
        or status_constraint_subset["rotation_extra"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_rotation_self[order].clear();
            const_rotation_cross[order].clear();
            const_rotation_self[order].shrink_to_fit();
            const_rotation_cross[order].shrink_to_fit();
        }
    }

    if (status_constraint_subset["rotation"] == 0
        or status_constraint_subset["rotation_extra"] == 0) {
        generate_rotational_constraint(system,
                                       symmetry,
                                       cluster,
                                       fcs,
                                       verbosity,
                                       tolerance_constraint);
    }

    // Huang constraint only for the harmonic IFCs
    if (const_huang.size() != 1) const_huang.resize(1);
    if (status_constraint_subset["huang"] == -1) {
        const_huang[0].clear();
        const_huang[0].shrink_to_fit();
    }

    if (status_constraint_subset["huang"] == 0) {
        // Implement a function to compute the huang constraint

        generate_huang_constraint(system->get_supercell(),
                                  symmetry,
                                  cluster,
                                  fcs,
                                  system->get_x_image(),
                                  verbosity);
    }

    // const_fix is updated.
    if (const_fix.size() != maxorder) const_fix.resize(maxorder);
    if (status_constraint_subset["fix2"] == -1 or status_constraint_subset["fix3"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_fix[order].clear();
            const_fix[order].shrink_to_fit();
        }
    }
    if (status_constraint_subset["fix2"] == 0
        or status_constraint_subset["fix3"] == 0) {
        generate_fix_constraint(symmetry,
                                fcs);
    }


    // Merge intra-order constraints and do reduction
    if (const_self.size() != maxorder) const_self.resize(maxorder);
    if (const_relate.size() != maxorder) const_relate.resize(maxorder);
    if (index_bimap) {
        deallocate(index_bimap);
        index_bimap = nullptr;
    }
    allocate(index_bimap, maxorder);

    for (auto order = 0; order < maxorder; ++order) {
        const_self[order].clear();
        const_self[order].shrink_to_fit();
        const_relate[order].clear();
        const_relate[order].shrink_to_fit();
        index_bimap[order].clear();
    }

    for (auto order = 0; order < maxorder; ++order) {
        const auto nparam = fcs->get_nequiv()[order].size();

        auto nlen_const = const_translation[order].size()
                          + const_rotation_self[order].size()
                          + const_symmetry[order].size();

        if (order == 0) {
            nlen_const += const_huang[order].size();
        }

        const_self[order].reserve(nlen_const);

        // The order of const_symmetry and const_translation
        // should not be changed since the rref_sparse is sensitive to
        // the numerical accuracy of the input matrix.

        const_self[order].insert(const_self[order].end(),
                                 const_symmetry[order].begin(),
                                 const_symmetry[order].end());

        const_self[order].insert(const_self[order].end(),
                                 const_translation[order].begin(),
                                 const_translation[order].end());

        if (order == 0) {
            const_self[order].insert(const_self[order].end(),
                                     const_huang[0].begin(),
                                     const_huang[0].end());
        }

        const_self[order].insert(const_self[order].end(),
                                 const_rotation_self[order].begin(),
                                 const_rotation_self[order].end());

        //        size_t nparams = 0;
        //        for (auto order2 = 0; order2 < maxorder; ++order2) {
        //            nparams += fcs->get_nequiv()[order2].size();
        //        }
        //test_svd(const_self[order], nparams);
        rref_sparse(nparam, const_self[order], tolerance_constraint);
    }


    if (!constraint_algebraic) {
        size_t Pmax = 0;
        size_t nparams = 0;
        for (auto order = 0; order < maxorder; ++order) {
            Pmax += const_self[order].size()
                + const_rotation_cross[order].size();
        }
        if (fix_harmonic) {
            Pmax -= const_self[0].size();
            Pmax += fcs->get_nequiv()[0].size();
        }
        if (fix_cubic) {
            Pmax -= const_self[1].size();
            Pmax += fcs->get_nequiv()[1].size();
        }
        for (auto order = 0; order < maxorder; ++order) {
            nparams += fcs->get_nequiv()[order].size();
        }

        if (const_mat) {
            deallocate(const_mat);
        }
        allocate(const_mat, Pmax, nparams);

        if (const_rhs) {
            deallocate(const_rhs);
        }
        allocate(const_rhs, Pmax);

        // const_mat and const_rhs are updated.
        number_of_constraints = calc_constraint_matrix(maxorder,
                                                       fcs->get_nequiv(),
                                                       nparams);
    }

    get_mapping_constraint(maxorder,
                           fcs->get_nequiv(),
                           const_self.data(),
                           const_fix.data(),
                           const_relate.data(),
                           index_bimap);

#ifdef _DEBUG
    for (auto order = 0; order < maxorder; ++order) {
        std::cout << "const_relate:\n";

        for (const auto &it: const_relate[order]) {
            std::cout << std::setw(5) << it.p_index_target;
            std::cout << " : ";
            for (auto m = 0; m < it.alpha.size(); ++m) {
                std::cout << "(" << std::setw(15) << it.alpha[m] << ", " << std::setw(5) << it.p_index_orig[m] << ") ";
            }
            std::cout << '\n';
        }

        std::cout << "\nindex_bimap:\n";

        for (const auto &it: index_bimap[order]) {
            std::cout << std::setw(5) << it.left << " <--> " << it.right << '\n';
        }

    }

#endif
}

void Constraint::print_constraint_information(const std::unique_ptr<Cluster> &cluster) const
{
    const auto maxorder = cluster->get_maxorder();
    auto extra_constraint_from_symmetry = false;
    for (auto order = 0; order < cluster->get_maxorder(); ++order) {
        if (!const_symmetry[order].empty()) extra_constraint_from_symmetry = true;
    }

    const auto exist_constraint = get_exist_constraint();

    if (exist_constraint) {
        int order;

        if (impose_inv_T || impose_inv_R) {
            std::cout << "  Number of constraints [T-inv, R-inv (self), R-inv (cross)]:\n";
            for (order = 0; order < maxorder; ++order) {
                std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
                std::cout << " " << std::setw(6) << const_translation[order].size();
                std::cout << std::setw(5) << const_rotation_self[order].size();
                std::cout << std::setw(5) << const_rotation_cross[order].size();
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        if (extra_constraint_from_symmetry) {
            std::cout << "  There are constraints from crystal symmetry.\n";
            std::cout << "  The number of such constraints for each order:\n";
            for (order = 0; order < maxorder; ++order) {
                std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
                std::cout << " " << std::setw(6) << const_symmetry[order].size();
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        if (extra_constraint_from_symmetry) {
            std::cout << "  Constraints of T-inv, R-inv (self), and those from crystal symmetry are merged.\n";
        } else {
            std::cout << "  Constraints of T-inv and R-inv (self) are merged.\n";
        }
        std::cout << "  If there are redundant constraints, they are removed in this process.\n\n";
        std::cout << "  Number of inequivalent constraints (self, cross) : \n";

        for (order = 0; order < maxorder; ++order) {
            std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
            std::cout << " " << std::setw(6) << const_self[order].size();
            std::cout << std::setw(5) << const_rotation_cross[order].size();
            std::cout << '\n';
        }
        std::cout << '\n';

        if (constraint_algebraic) {
            std::cout << "  ICONST >= 10 : Constraints will be considered algebraically.\n\n";

            if (impose_inv_R) {
                std::cout << "  WARNING : Inter-order constraints for rotational invariance will be neglected.\n";
            }

            for (order = 0; order < maxorder; ++order) {
                std::cout << "  Number of free" << std::setw(9) << cluster->get_ordername(order)
                    << " FCs : " << index_bimap[order].size() << '\n';
            }
            std::cout << '\n';
        } else {
            std::cout << "  Total number of constraints = " << number_of_constraints << "\n\n";
        }
    }
}


size_t Constraint::calc_constraint_matrix(const int maxorder,
                                          const std::vector<size_t> *nequiv,
                                          const size_t nparams) const
{
    size_t i, j;
    int order;
    double *arr_tmp;
    std::vector<ConstraintClass> const_total;

    const_total.clear();
    allocate(arr_tmp, nparams);

    size_t nshift = 0;

    for (order = 0; order < maxorder; ++order) {
        const auto nelems = nequiv[order].size();
        if (const_fix[order].empty()) {
            for (auto &p: const_self[order]) {
                for (i = 0; i < nparams; ++i) arr_tmp[i] = 0.0;
                for (const auto &it: p) {
                    arr_tmp[nshift + it.first] = it.second;
                }
                const_total.emplace_back(nparams, arr_tmp);
            }
        }
        nshift += nelems;
    }

    const auto nconst1 = const_total.size();

    // Inter-order constraints
    size_t nshift2 = 0;
    for (order = 0; order < maxorder; ++order) {
        if (order > 0) {
            if (const_fix[order - 1].empty() && const_fix[order].empty()) {
                for (auto &p: const_rotation_cross[order]) {
                    for (i = 0; i < nparams; ++i) arr_tmp[i] = 0.0;
                    for (const auto &it: p) {
                        arr_tmp[nshift2 + it.first] = it.second;
                    }
                    const_total.emplace_back(nparams, arr_tmp);
                }
            }

            nshift2 += nequiv[order - 1].size();
        }
    }
    deallocate(arr_tmp);

    if (nconst1 != const_total.size())
        remove_redundant_rows(nparams, const_total, tolerance_constraint);

    auto nconst = const_total.size();

    if (fix_harmonic) nconst += nequiv[0].size();
    if (fix_cubic) nconst += nequiv[1].size();

    for (i = 0; i < nconst; ++i) {
        for (j = 0; j < nparams; ++j) {
            const_mat[i][j] = 0.0;
        }
        const_rhs[i] = 0.0;
    }

    size_t irow = 0;
    //    size_t icol = 0;
    size_t ishift = 0;

    if (fix_harmonic) {
        for (const auto &p: const_fix[0]) {
            i = p.p_index_target;
            const_mat[i][i] = 1.0;
            const_rhs[i] = p.val_to_fix;
        }

        irow += const_fix[0].size();
        //        icol += const_fix[0].size();
        ishift += const_fix[0].size();
    }

    if (fix_cubic && maxorder > 1) {
        const auto ishift2 = nequiv[0].size();

        for (const auto &p: const_fix[1]) {
            i = p.p_index_target;
            const_mat[i + ishift][i + ishift2] = 1.0;
            const_rhs[i + ishift] = p.val_to_fix;
        }

        irow += const_fix[1].size();
        //icol += const_fix[1].size();
    }

    for (auto &p: const_total) {
        for (i = 0; i < nparams; ++i) {
            const_mat[irow][i] = p.w_const[i];
        }
        ++irow;
    }
    const_total.clear();

    return nconst;
}


void Constraint::get_mapping_constraint(const int nmax,
                                        const std::vector<size_t> *nequiv,
                                        const ConstraintSparseForm *const_in,
                                        std::vector<ConstraintTypeFix> *const_fix_out,
                                        std::vector<ConstraintTypeRelate> *const_relate_out,
                                        boost::bimap<size_t, size_t> *index_bimap_out) const
{
    // If const_fix_out[order] is not empty as input, it assumes that fix_forceconstant[order] is true.
    // In this case, const_fix_out[order] is not updated.

    int order;
    size_t i;

    std::vector<ConstraintDoubleElement> ConstVec;

    for (order = 0; order < nmax; ++order) {
        if (const_fix_out[order].empty()) {
            size_t p_index_target;
            std::vector<double> alpha_tmp;
            std::vector<size_t> p_index_tmp;

            for (auto p = const_in[order].rbegin(); p != const_in[order].rend(); ++p) {
                alpha_tmp.clear();
                p_index_tmp.clear();

#ifndef _USE_MAP_FOR_CONSTRAINT
                ConstVec.clear();
                ConstVec.reserve(p->size());
                for (const auto &[fst, snd]: (*p)) {
                    ConstVec.emplace_back(fst, snd);
                }
                std::sort(ConstVec.begin(), ConstVec.end());

                p_index_target = ConstVec[0].col;

                const auto nsize = ConstVec.size();
                alpha_tmp.resize(nsize - 1);
                p_index_tmp.resize(nsize - 1);

                for (i = 1; i < nsize; ++i) {
                    alpha_tmp[i - 1] = ConstVec[i].val;
                    p_index_tmp[i - 1] = ConstVec[i].col;
                }

#else
                auto counter = 0;
                for (const auto &p2 : (*p)) {
                    if (counter == 0) {
                        p_index_target = p2.first;
                    } else {
                        alpha_tmp.push_back(p2.second);
                        p_index_tmp.push_back(p2.first);
                }
                    ++counter;
                    }
#endif

                if (!alpha_tmp.empty()) {
                    const_relate_out[order].emplace_back(p_index_target,
                                                         alpha_tmp,
                                                         p_index_tmp);
                } else {
                    const_fix_out[order].emplace_back(p_index_target, 0.0);
                }
            }
        }
    }

    std::vector<int> *has_constraint;
    allocate(has_constraint, nmax);
    size_t nparam;
    for (order = 0; order < nmax; ++order) {
        nparam = nequiv[order].size();
        has_constraint[order].resize(nparam, 0);

        for (i = 0; i < const_fix_out[order].size(); ++i) {
            has_constraint[order][const_fix_out[order][i].p_index_target] = 1;
        }

        for (i = 0; i < const_relate_out[order].size(); ++i) {
            has_constraint[order][const_relate_out[order][i].p_index_target] = 2;
        }
    }

    for (order = 0; order < nmax; ++order) {
        nparam = nequiv[order].size();

        size_t icount = 0;
        for (i = 0; i < nparam; ++i) {
            if (has_constraint[order][i] == 0) {
                index_bimap_out[order].insert(
                    boost::bimap<size_t, size_t>::value_type(icount, i));
                ++icount;
            }
        }
    }

    deallocate(has_constraint);
}

bool Constraint::ready_all_constraints() const
{
    for (const auto &[fst, snd]: status_constraint_subset) {
        if (snd == 0) return false;
    }

    return true;
}

int Constraint::get_constraint_mode() const
{
    return constraint_mode;
}

void Constraint::set_constraint_mode(const int constraint_mode_in)
{
    constraint_mode = constraint_mode_in;
}

size_t Constraint::get_number_of_constraints() const
{
    return number_of_constraints;
}

std::string Constraint::get_fc_file(const int order) const
{
    switch (order) {
    case 2:
        return fc2_file;
    case 3:
        return fc3_file;
    default:
        return "";
    }
}

void Constraint::set_fc_file(const int order,
                             const std::string &fc_file)
{
    switch (order) {
    case 2:
        fc2_file = fc_file;
        break;
    case 3:
        fc3_file = fc_file;
        break;
    default:
        break;
    }
}

bool Constraint::get_fix_harmonic() const
{
    return fix_harmonic;
}

void Constraint::set_fix_harmonic(const bool fix_harmonic_in)
{
    fix_harmonic = fix_harmonic_in;
}

bool Constraint::get_fix_cubic() const
{
    return fix_cubic;
}

void Constraint::set_fix_cubic(const bool fix_cubic_in)
{
    fix_cubic = fix_cubic_in;
}

void Constraint::set_constraint_algebraic(const int constraint_algebraic_in)
{
    constraint_algebraic = constraint_algebraic_in;
}

int Constraint::get_constraint_algebraic() const
{
    return constraint_algebraic;
}

double **Constraint::get_const_mat() const
{
    return const_mat;
}

double *Constraint::get_const_rhs() const
{
    return const_rhs;
}

double Constraint::get_tolerance_constraint() const
{
    return tolerance_constraint;
}

void Constraint::set_tolerance_constraint(const double tol)
{
    tolerance_constraint = tol;
}

bool Constraint::get_exist_constraint() const
{
    if (constraint_algebraic) {
        if (!const_self.empty()) {
            const auto n = const_self.size();

            for (auto i = 0; i < n; ++i) {
                if (!const_self[i].empty()) return true;
            }
        }

        if (!const_fix.empty()) {
            const auto n = const_fix.size();
            for (auto i = 0; i < n; ++i) {
                if (!const_fix[i].empty()) return true;
            }
        }

        if (!const_relate.empty()) {
            const auto n = const_relate.size();
            for (auto i = 0; i < n; ++i) {
                if (!const_relate[i].empty()) return true;
            }
        }
    } else {
        if (number_of_constraints > 0) return true;
    }

    return false;
}

std::string Constraint::get_rotation_axis() const
{
    return rotation_axis;
}

void Constraint::set_rotation_axis(const std::string &rotation_axis_in)
{
    rotation_axis = rotation_axis_in;
}

const ConstraintSparseForm &Constraint::get_const_symmetry(const int order) const
{
    return const_symmetry[order];
}

const std::vector<ConstraintTypeFix> &Constraint::get_const_fix(const int order) const
{
    return const_fix[order];
}

void Constraint::set_const_fix_val_to_fix(const int order,
                                          const size_t idx,
                                          const double val)
{
    const_fix[order][idx].val_to_fix = val;
}

const std::vector<ConstraintTypeRelate> &Constraint::get_const_relate(const int order) const
{
    return const_relate[order];
}

const boost::bimap<size_t, size_t> &Constraint::get_index_bimap(const int order) const
{
    return index_bimap[order];
}

void Constraint::set_constraint_flag(const std::string &const_name,
                                     const int use_constraint)
{
    auto it = status_constraint_subset.find(const_name);

    if (it != status_constraint_subset.end()) {
        if (use_constraint == 0) {
            it->second = -1;
        } else {
            it->second = 0;
        }
    } else {
        exit("set_constraint_flag",
             "Invalid constraint name");
    }
}

void Constraint::generate_symmetry_constraint(const size_t nat,
                                              const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Cluster> &cluster,
                                              const std::unique_ptr<Fcs> &fcs,
                                              const int verbosity)
{
    // Create constraint matrices arising from the crystal symmetry.
    // This function clears and updates const_symmetry.

    const auto maxorder = cluster->get_maxorder();
    auto has_constraint_from_symm = false;

    if (fcs->get_forceconstant_basis() == "Cartesian") {
        for (auto isym = 0; isym < symmetry->get_nsym(); ++isym) {
            if (!symmetry->get_symmetry_data()[isym].compatible_with_cartesian) {
                has_constraint_from_symm = true;
                break;
            }
        }
    } else {
        for (auto isym = 0; isym < symmetry->get_nsym(); ++isym) {
            if (!symmetry->get_symmetry_data()[isym].compatible_with_lattice) {
                has_constraint_from_symm = true;
                break;
            }
        }
    }

    has_constraint_from_symm = has_constraint_from_symm & (verbosity > 0);

    if (has_constraint_from_symm) {
        std::cout << "  Generating constraints from crystal symmetry\n";
        if (fcs->get_forceconstant_basis() == "Lattice") {
            std::cout << "  in crystallographic (fractional) coordinates ...\n";
        } else {
            std::cout << "  in Cartesian coordinates ...\n";
        }
    }

    for (auto order = 0; order < maxorder; ++order) {
        if (has_constraint_from_symm) {
            std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
        }

        if (fcs->get_forceconstant_basis() == "Lattice") {
            fcs->get_constraint_symmetry_in_integer(nat,
                                                    symmetry,
                                                    order,
                                                    fcs->get_forceconstant_basis(),
                                                    fcs->get_fc_table()[order],
                                                    fcs->get_nequiv()[order].size(),
                                                    tolerance_constraint,
                                                    const_symmetry[order],
                                                    true);
        } else {
            fcs->get_constraint_symmetry(nat,
                                         symmetry,
                                         order,
                                         fcs->get_forceconstant_basis(),
                                         fcs->get_fc_table()[order],
                                         fcs->get_nequiv()[order].size(),
                                         tolerance_constraint,
                                         const_symmetry[order],
                                         true);
        }

        if (has_constraint_from_symm) {
            std::cout << " done.\n";
        }
    }

    if (has_constraint_from_symm) {
        std::cout << "  Finished !\n\n";
    }

    status_constraint_subset["symmetry"] = 1;
}


void Constraint::generate_translational_constraint(const Cell &supercell,
                                                   const std::unique_ptr<Symmetry> &symmetry,
                                                   const std::unique_ptr<Cluster> &cluster,
                                                   const std::unique_ptr<Fcs> &fcs,
                                                   const int periodic_image_conv,
                                                   const int verbosity)
{
    // Create constraint matrix for the translational invariance (aka acoustic sum rule).
    const auto maxorder = cluster->get_maxorder();

    if (const_translation.empty()) {
        const_translation.resize(maxorder);
    }

    if (status_constraint_subset["translation"] == -1) return;

    if (verbosity > 0) {
        std::cout << "  Generating constraints for translational invariance ...\n";
    }

    for (auto order = 0; order < maxorder; ++order) {
        if (verbosity > 0)
            std::cout << "   " << std::setw(8) << cluster->get_ordername(order) << " ...";

        const_translation[order].clear();

        const auto nparams = fcs->get_nequiv()[order].size();

        if (nparams == 0) {
            if (verbosity > 0) std::cout << "  No parameters! Skipped.\n";
            continue;
        }

        if (periodic_image_conv == 0 || order == 0) {
            get_constraint_translation(supercell,
                                       symmetry,
                                       cluster,
                                       fcs,
                                       order,
                                       fcs->get_fc_table()[order],
                                       fcs->get_nequiv()[order].size(),
                                       const_translation[order],
                                       true);
        }
        // make translation constraint for each periodic image combination
        // if periodic_image_conv == 0 or order == 0, there is no need to impose additional ASR constraints.
        else {
            // if(periodic_image_conv > 0 && order > 0)
            get_constraint_translation_for_periodic_images(supercell,
                                                           symmetry,
                                                           cluster,
                                                           order,
                                                           fcs->get_fc_table()[order],
                                                           fcs->get_nequiv()[order].size(),
                                                           const_translation[order],
                                                           true);
        }

        if (verbosity > 0) std::cout << " done.\n" << std::flush;
    }
    status_constraint_subset["translation"] = 1;
    if (verbosity > 0) std::cout << "  Finished !\n\n";
}


void Constraint::get_constraint_translation(const Cell &supercell,
                                            const std::unique_ptr<Symmetry> &symmetry,
                                            const std::unique_ptr<Cluster> &cluster,
                                            const std::unique_ptr<Fcs> &fcs,
                                            const int order,
                                            const std::vector<FcProperty> &fc_table,
                                            const size_t nparams,
                                            ConstraintSparseForm &const_out,
                                            const bool do_rref) const
{
    // Generate equality constraint for the acoustic sum rule.

    int i, j;
    int iat, jat, icrd, jcrd;
    int idata;
    int loc_nonzero;

    int *ind = nullptr;
    int *intarr = nullptr;
    int *intarr_copy = nullptr;
    int **xyzcomponent = nullptr;

    int ixyz;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;

    unsigned int isize;

    std::vector<int> data;
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty>::iterator iter_found;
    std::vector<std::vector<int>> data_vec;
    std::vector<int> const_now;

    using ConstEntry = std::vector<ConstraintIntegerElement>;
    std::vector<ConstEntry> constraint_all;

    ConstEntry const_tmp;

    if (order < 0) return;

    if (nparams == 0) return;

    allocate(ind, order + 2);

    // Create force constant table for search

    list_found.clear();

    for (const auto &p: fc_table) {
        for (i = 0; i < order + 2; ++i) {
            ind[i] = p.elems[i];
        }
        if (list_found.find(FcProperty(order + 2,
                                       p.sign,
                                       ind,
                                       p.mother)) != list_found.end()) {
            exit("get_constraint_translation", "Duplicate interaction list found");
        }
        list_found.insert(FcProperty(order + 2,
                                     p.sign,
                                     ind,
                                     p.mother));
    }

    deallocate(ind);
    ind = nullptr;

    // Generate xyz component for each order

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    fcs->get_xyzcomponent(order + 1, xyzcomponent);

    allocate(intarr, order + 2);
    allocate(intarr_copy, order + 2);

    const_now.resize(nparams);

    for (i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        // Generate atom pairs for each order

        if (order == 0) {
            for (icrd = 0; icrd < 3; ++icrd) {
                intarr[0] = 3 * iat + icrd;

                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    // Reset the temporary array for another constraint
                    for (j = 0; j < nparams; ++j) const_now[j] = 0;

                    for (jat = 0; jat < 3 * nat; jat += 3) {
                        intarr[1] = jat + jcrd;

                        iter_found = list_found.find(FcProperty(order + 2,
                                                                1.0,
                                                                intarr,
                                                                1));

                        //  If found an IFC
                        if (iter_found != list_found.end()) {
                            // Round the coefficient to integer
                            const_now[iter_found->mother] += nint(iter_found->sign);
                        }
                    }
                    // Add to the constraint list
                    if (!is_allzero(const_now, loc_nonzero)) {
                        if (const_now[loc_nonzero] < 0) {
                            for (j = 0; j < nparams; ++j) const_now[j] *= -1;
                        }
                        const_tmp.clear();
                        for (j = 0; j < nparams; ++j) {
                            if (std::abs(const_now[j]) > 0) {
                                const_tmp.emplace_back(j, const_now[j]);
                            }
                        }
                        constraint_all.emplace_back(const_tmp);
                    }
                }
            }
        } else {
            // Anharmonic cases

            auto intlist(cluster->get_atoms_in_cutoff(order, i));
            std::sort(intlist.begin(), intlist.end());

            data_vec.clear();
            // Generate data_vec that contains possible interacting clusters.
            // Each cluster contains (order + 1) atoms, and the last atom index
            // will be treated separately below.
            CombinationWithRepetition<int> g2(intlist.begin(), intlist.end(), order);
            do {
                data = g2.now();

                intarr[0] = iat;

                for (isize = 0; isize < data.size(); ++isize) {
                    intarr[isize + 1] = data[isize];
                }

                if (cluster->satisfy_nbody_rule(order + 1, intarr, order)) {
                    if (cluster->is_incutoff(order + 1, intarr, order, supercell.kind)) {
                        // Add to list if the atoms interact with each other.
                        data_vec.push_back(data);
                    }
                }
            } while (g2.next());

            const auto ndata = data_vec.size();

            // Use openmp for acceleration if possible
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int *intarr_omp = nullptr;
                int *intarr_copy_omp = nullptr;

                allocate(intarr_omp, order + 2);
                allocate(intarr_copy_omp, order + 2);

                std::vector<int> data_omp;
                std::vector<int> const_now_omp;

                ConstEntry const_tmp_omp;
                std::vector<ConstEntry> constraint_list_omp;

                const_now_omp.resize(nparams);
#ifdef _OPENMP
#pragma omp for private(isize, ixyz, jcrd, j, jat, iter_found, loc_nonzero), nowait
#endif
                for (idata = 0; idata < ndata; ++idata) {
                    data_omp = data_vec[idata];

                    intarr_omp[0] = iat;
                    for (isize = 0; isize < data_omp.size(); ++isize) {
                        intarr_omp[isize + 1] = data_omp[isize];
                    }

                    // Loop for xyz component
                    for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                        // Loop for the xyz index of the last atom
                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                            // Reset the temporary array for another constraint
                            for (j = 0; j < nparams; ++j) const_now_omp[j] = 0;

                            // Loop for the last atom index
                            for (jat = 0; jat < 3 * nat; jat += 3) {
                                intarr_omp[order + 1] = jat / 3;

                                if (cluster->satisfy_nbody_rule(order + 2, intarr_omp, order)) {
                                    for (j = 0; j < order + 1; ++j) {
                                        intarr_copy_omp[j] = 3 * intarr_omp[j] + xyzcomponent[ixyz][j];
                                    }
                                    intarr_copy_omp[order + 1] = jat + jcrd;

                                    sort_tail(order + 2, intarr_copy_omp);

                                    iter_found = list_found.find(FcProperty(order + 2,
                                                                            1.0,
                                                                            intarr_copy_omp,
                                                                            1));
                                    if (iter_found != list_found.end()) {
                                        const_now_omp[iter_found->mother] += nint(iter_found->sign);
                                    }
                                }
                            } // close loop jat

                            // Add the constraint to the private array
                            if (!is_allzero(const_now_omp, loc_nonzero)) {
                                if (const_now_omp[loc_nonzero] < 0) {
                                    for (j = 0; j < nparams; ++j) const_now_omp[j] *= -1;
                                }

                                const_tmp_omp.clear();
                                for (j = 0; j < nparams; ++j) {
                                    if (std::abs(const_now_omp[j]) > 0) {
                                        const_tmp_omp.emplace_back(j, const_now_omp[j]);
                                    }
                                }
                                if (const_tmp_omp.empty()) {
                                    std::cout << "This cannot happen\n";
                                }
                                constraint_list_omp.emplace_back(const_tmp_omp);
                            }
                        }
                    }
                } // close idata (openmp main loop)

                if (intarr_omp) {
                    deallocate(intarr_omp);
                    intarr_omp = nullptr;
                }
                if (intarr_copy_omp) {
                    deallocate(intarr_copy_omp);
                    intarr_copy_omp = nullptr;
                }

                // Merge vectors
#pragma omp critical
                {
                    for (const auto &it: constraint_list_omp) {
                        constraint_all.emplace_back(it);
                    }
                }
                constraint_list_omp.clear();
            } // close openmp

            intlist.clear();
        } // close if
    }     // close loop i

    if (xyzcomponent) {
        deallocate(xyzcomponent);
        xyzcomponent = nullptr;
    }
    if (intarr) {
        deallocate(intarr);
        intarr = nullptr;
    }
    if (intarr_copy) {
        deallocate(intarr_copy);
        intarr_copy = nullptr;
    }

    std::sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(),
                                     constraint_all.end()),
                         constraint_all.end());

    MapConstraintElement const_tmp2;
    auto division_factor = 1.0;
    int counter;
    const_out.clear();

    for (const auto &it: constraint_all) {
        const_tmp2.clear();
        counter = 0;
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_tmp2[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_out.emplace_back(const_tmp2);
    }
    constraint_all.clear();
    if (do_rref) rref_sparse(nparams, const_out, eps8);
}

void Constraint::get_constraint_translation_for_periodic_images(const Cell &supercell,
                                                                const std::unique_ptr<Symmetry> &symmetry,
                                                                const std::unique_ptr<Cluster> &cluster,
                                                                const int order,
                                                                const std::vector<FcProperty> &fc_table,
                                                                const size_t nparams,
                                                                ConstraintSparseForm &const_out,
                                                                const bool do_rref) const
{
    // Generate equality constraint for the acoustic sum rule.

    int i, j;
    int iat, jat, jcrd;
    int idata;
    int loc_nonzero;

    int *ind;
    int *intarr, *intarr_copy;
    int **xyzcomponent;

    int ixyz;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;

    // generate combinations of periodic images
    //long int n_mirror_images = nint(std::pow(static_cast<double>(27), order));

    unsigned int isize;

    std::vector<int> data;
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty>::iterator iter_found;
    std::vector<std::vector<int>> data_vec;
    std::vector<double> const_now;

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    std::vector<ConstEntry> constraint_all;

    ConstEntry const_tmp;

    if (order < 0) return;

    if (nparams == 0) return;

    allocate(ind, order + 2);

    // Create force constant table for search
    list_found.clear();

    for (const auto &p: fc_table) {
        for (i = 0; i < order + 2; ++i) {
            ind[i] = p.elems[i];
        }
        if (list_found.find(FcProperty(order + 2,
                                       p.sign,
                                       ind,
                                       p.mother)) != list_found.end()) {
            exit("get_constraint_translation", "Duplicate interaction list found");
        }
        list_found.insert(FcProperty(order + 2,
                                     p.sign,
                                     ind,
                                     p.mother));
    }

    deallocate(ind);

    // Generate xyz component for each order

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    Fcs::get_xyzcomponent(order + 1, xyzcomponent);

    allocate(intarr, order + 2);
    allocate(intarr_copy, order + 2);

    const_now.resize(nparams);

    for (i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        // Generate atom pairs for each order

        if (order == 0) {
            continue; // there is no new translational invariance
        } else {
            // Anharmonic cases

            auto intlist(cluster->get_atoms_in_cutoff(order, i));
            std::sort(intlist.begin(), intlist.end());

            data_vec.clear();
            // Generate data_vec that contains possible interacting clusters.
            // Each cluster contains (order + 1) atoms, and the last atom index
            // will be treated separately below.
            CombinationWithRepetition<int> g2(intlist.begin(), intlist.end(), order);
            do {
                data = g2.now();

                intarr[0] = iat;

                for (isize = 0; isize < data.size(); ++isize) {
                    intarr[isize + 1] = data[isize];
                }

                if (cluster->satisfy_nbody_rule(order + 1, intarr, order)) {
                    if (cluster->is_incutoff(order + 1, intarr, order, supercell.kind)) {
                        // Add to list if the atoms interact with each other.
                        data_vec.push_back(data);
                    }
                }
            } while (g2.next());

            const auto ndata = data_vec.size();

            // Use openmp for acceleration if possible
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int *intarr_omp, *intarr_copy_omp;

                double weight;

                allocate(intarr_omp, order + 2);
                allocate(intarr_copy_omp, order + 2);

                std::vector<int> data_omp;
                std::vector<int> atom_tmp;
                std::vector<int> sort_table, sort_table_tmp;
                std::vector<std::vector<double>> consts_now_omp;

                std::vector<long long int> periodic_images_found;

                ConstEntry const_tmp_omp;
                std::vector<ConstEntry> constraint_list_omp;
                long long int i_periodic_images;
                long long int i_tmp, j_tmp, i_tmp2;
                long int i_mi_tmp;

#ifdef _OPENMP
#pragma omp for private(isize, ixyz, jcrd, j, jat, iter_found, loc_nonzero), nowait
#endif
                for (idata = 0; idata < ndata; ++idata) {
                    data_omp = data_vec[idata];

                    intarr_omp[0] = iat;
                    for (isize = 0; isize < data_omp.size(); ++isize) {
                        intarr_omp[isize + 1] = data_omp[isize];
                    }

                    // Loop for xyz component
                    for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                        // Loop for the xyz index of the last atom
                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                            // Reset the temporary array for another constraint
                            consts_now_omp.clear();
                            periodic_images_found.clear();

                            // Loop for the last atom index
                            for (jat = 0; jat < 3 * nat; jat += 3) {
                                intarr_omp[order + 1] = jat / 3;
                                atom_tmp = data_omp;
                                atom_tmp.push_back(jat / 3);
                                // sort atom_tmp and get corresponding sort_table
                                sort_table_tmp.resize(atom_tmp.size());
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    sort_table_tmp[i_tmp] = i_tmp;
                                }
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    for (j_tmp = i_tmp + 1; j_tmp < atom_tmp.size(); j_tmp++) {
                                        if (atom_tmp[i_tmp] > atom_tmp[j_tmp]) {
                                            // swap atom numbers
                                            i_tmp2 = atom_tmp[i_tmp];
                                            atom_tmp[i_tmp] = atom_tmp[j_tmp];
                                            atom_tmp[j_tmp] = i_tmp2;
                                            // write on sort table
                                            i_tmp2 = sort_table_tmp[i_tmp];
                                            sort_table_tmp[i_tmp] = sort_table_tmp[j_tmp];
                                            sort_table_tmp[j_tmp] = i_tmp2;
                                        }
                                    }
                                }
                                // make sort table
                                sort_table.resize(atom_tmp.size());
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    sort_table[sort_table_tmp[i_tmp]] = i_tmp;
                                }

                                if (cluster->satisfy_nbody_rule(order + 2, intarr_omp, order)) {
                                    for (j = 0; j < order + 1; ++j) {
                                        intarr_copy_omp[j] = 3 * intarr_omp[j] + xyzcomponent[ixyz][j];
                                    }
                                    intarr_copy_omp[order + 1] = jat + jcrd;

                                    sort_tail(order + 2, intarr_copy_omp);

                                    iter_found = list_found.find(FcProperty(order + 2,
                                                                            1.0,
                                                                            intarr_copy_omp,
                                                                            1));

                                    auto cluster_found = cluster->get_interaction_cluster(order, i).find(
                                        InteractionCluster(atom_tmp));

                                    if (iter_found != list_found.end()) {
                                        if (cluster_found == cluster->get_interaction_cluster(order, i).end()) {
                                            std::cout << "Warning: cluster corresponding to the IFC is NOT found.\n";
                                        } else {
                                            // get weight
                                            weight = 1.0 / static_cast<double>((cluster_found->cell).size());
                                            for (auto cellvec: cluster_found->cell) {
                                                // get number of the combination of the cell
                                                i_periodic_images = 0;

                                                for (i_tmp = 0; i_tmp < order; i_tmp++) {
                                                    i_periodic_images *= 27;
                                                    i_periodic_images += cellvec[sort_table[i_tmp]];
                                                }

                                                // check if the same periodic image has already been found.
                                                for (i_mi_tmp = 0;
                                                     i_mi_tmp < periodic_images_found.size(); i_mi_tmp++) {
                                                    if (periodic_images_found[i_mi_tmp] == i_periodic_images) {
                                                        break;
                                                    }
                                                }
                                                // if not found
                                                if (i_mi_tmp == periodic_images_found.size()) {
                                                    periodic_images_found.push_back(i_periodic_images);
                                                    consts_now_omp.push_back(std::vector<double>(nparams, 0.0));
                                                }

                                                // add to the constraint
                                                consts_now_omp[i_mi_tmp][iter_found->mother] +=
                                                    weight * iter_found->sign;
                                            }
                                        }
                                    }
                                }
                            } // close loop jat

                            // Add the constraint to the private array
                            for (i_mi_tmp = 0; i_mi_tmp < periodic_images_found.size(); i_mi_tmp++) {
                                if (!is_allzero(consts_now_omp[i_mi_tmp], eps8, loc_nonzero, 0)) {
                                    if (consts_now_omp[i_mi_tmp][loc_nonzero] < 0) {
                                        for (j = 0; j < nparams; ++j) consts_now_omp[i_mi_tmp][j] *= -1.0;
                                    }

                                    const_tmp_omp.clear();
                                    for (j = 0; j < nparams; ++j) {
                                        if (std::abs(consts_now_omp[i_mi_tmp][j]) > 0) {
                                            const_tmp_omp.emplace_back(j, consts_now_omp[i_mi_tmp][j]);
                                        }
                                    }
                                    if (const_tmp_omp.empty()) {
                                        std::cout << "This cannot happen\n";
                                    }
                                    constraint_list_omp.emplace_back(const_tmp_omp);
                                }
                            }
                        }
                    }
                } // close idata (openmp main loop)

                deallocate(intarr_omp);
                deallocate(intarr_copy_omp);

                // Merge vectors
#pragma omp critical
                {
                    for (const auto &it: constraint_list_omp) {
                        constraint_all.emplace_back(it);
                    }
                }
                constraint_list_omp.clear();
            } // close openmp

            intlist.clear();
        } // close if
    }     // close loop i

    deallocate(xyzcomponent);
    deallocate(intarr);
    deallocate(intarr_copy);

    std::sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(),
                                     constraint_all.end()),
                         constraint_all.end());

    MapConstraintElement const_tmp2;
    auto division_factor = 1.0;
    int counter;
    const_out.clear();

    for (const auto &it: constraint_all) {
        const_tmp2.clear();
        counter = 0;
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_tmp2[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_out.emplace_back(const_tmp2);
    }
    constraint_all.clear();
    if (do_rref) rref_sparse(nparams, const_out, eps8);
}

void Constraint::generate_rotational_constraint(const std::unique_ptr<System> &system,
                                                const std::unique_ptr<Symmetry> &symmetry,
                                                const std::unique_ptr<Cluster> &cluster,
                                                const std::unique_ptr<Fcs> &fcs,
                                                const int verbosity,
                                                const double tolerance)
{
    // Create constraints for the rotational invariance
    const auto maxorder = cluster->get_maxorder();

    if (const_rotation_self.empty()) {
        const_rotation_self.resize(maxorder);
    }
    if (const_rotation_cross.empty()) {
        const_rotation_cross.resize(maxorder);
    }

    if (status_constraint_subset["rotation"] == -1
        and status_constraint_subset["rotation_extra"] == -1)
        return;


    if (verbosity > 0)
        std::cout << "  Generating constraints for rotational invariance ...\n";


    int order;
    bool valid_rotation_axis[3][3];
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty> list_found_last;

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    std::vector<ConstEntry> *const_self_vec, *const_cross_vec;

    allocate(const_self_vec, maxorder);
    allocate(const_cross_vec, maxorder);

    setup_rotation_axis(valid_rotation_axis);

    std::vector<size_t> nparams;
    nparams.resize(maxorder);

    for (order = 0; order < maxorder; ++order) {
        nparams[order] = fcs->get_nequiv()[order].size();

        const_rotation_self[order].clear();
        const_rotation_cross[order].clear();

        if (order == 0) {
            if (verbosity > 0) {
                std::cout << "   Constraints between " << std::setw(8)
                    << "1st-order IFCs (which are zero) and "
                    << std::setw(8) << cluster->get_ordername(order) << " ...";
            }
        } else {
            if (verbosity > 0) {
                std::cout << "   Constraints between " << std::setw(8)
                    << cluster->get_ordername(order - 1) << " and "
                    << std::setw(8) << cluster->get_ordername(order) << " ...";
            }
        }

        const_self_vec[order].clear();
        const_cross_vec[order].clear();

        if (order > 0) {
            list_found_last = list_found;
        }

        list_found.clear();

        // Accumulate sets of non-zero force constants.
        for (auto p = fcs->get_fc_table()[order].begin();
             p != fcs->get_fc_table()[order].end(); ++p) {
            list_found.insert(FcProperty(order + 2,
                                         p->sign,
                                         p->elems.data(),
                                         p->mother));
        }

        set_rotation_constraints(system,
                                 symmetry,
                                 cluster,
                                 fcs,
                                 order,
                                 valid_rotation_axis,
                                 list_found,
                                 list_found_last,
                                 tolerance,
                                 const_self_vec,
                                 const_cross_vec);

        set_rotation_constraints_extra(system,
                                       symmetry,
                                       cluster,
                                       fcs,
                                       order,
                                       valid_rotation_axis,
                                       list_found,
                                       tolerance,
                                       const_self_vec,
                                       const_cross_vec);

        if (verbosity > 0) std::cout << " done.\n" << std::flush;
    } // order

    int counter;
    MapConstraintElement const_copy;
    auto division_factor = 1.0;

    for (order = 0; order < maxorder; ++order) {
        // Sort & unique
        std::sort(const_self_vec[order].begin(), const_self_vec[order].end());
        const_self_vec[order].erase(std::unique(const_self_vec[order].begin(),
                                                const_self_vec[order].end()),
                                    const_self_vec[order].end());
        std::sort(const_cross_vec[order].begin(), const_cross_vec[order].end());
        const_cross_vec[order].erase(std::unique(const_cross_vec[order].begin(),
                                                 const_cross_vec[order].end()),
                                     const_cross_vec[order].end());

        // Copy to the return variable
        for (const auto &it: const_self_vec[order]) {
            const_copy.clear();
            counter = 0;
            for (const auto &it2: it) {
                if (counter == 0) {
                    division_factor = 1.0 / it2.val;
                }
                const_copy[it2.col] = it2.val * division_factor;
                ++counter;
            }
            const_rotation_self[order].emplace_back(const_copy);
        }
        const_self_vec[order].clear();

        for (const auto &it: const_cross_vec[order]) {
            const_copy.clear();
            counter = 0;
            for (const auto &it2: it) {
                if (counter == 0) {
                    division_factor = 1.0 / it2.val;
                }
                const_copy[it2.col] = it2.val * division_factor;
                ++counter;
            }
            const_rotation_cross[order].emplace_back(const_copy);
        }
        const_cross_vec[order].clear();

        //  Perform rref
        rref_sparse(nparams[order],
                    const_rotation_self[order],
                    eps6);

        if (order > 0) {
            rref_sparse(nparams[order - 1] + nparams[order],
                        const_rotation_cross[order],
                        eps6);
        }
    }

    if (verbosity > 0) std::cout << "  Finished !\n\n" << std::flush;

    if (status_constraint_subset["rotation"] == 0) status_constraint_subset["rotation"] = 1;
    if (status_constraint_subset["rotation_extra"] == 0) status_constraint_subset["rotation_extra"] = 1;

    deallocate(const_self_vec);
    deallocate(const_cross_vec);
}

void Constraint::set_rotation_constraints(const std::unique_ptr<System> &system,
                                          const std::unique_ptr<Symmetry> &symmetry,
                                          const std::unique_ptr<Cluster> &cluster,
                                          const std::unique_ptr<Fcs> &fcs,
                                          const int order,
                                          const bool valid_rotation_axis[3][3],
                                          const std::unordered_set<FcProperty> &list_found,
                                          const std::unordered_set<FcProperty> &list_found_last,
                                          const double tolerance,
                                          std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
                                          std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec)
{
    const auto natmin = symmetry->get_nat_trueprim();
    const auto maxorder = cluster->get_maxorder();

    int iat, jat;
    int icrd;

    CombinationWithRepetition<int> g;
    Eigen::Vector3d vec_for_rot;

    int ixyz, nxyz{0};
    int loc_nonzero;

    int mu_lambda, lambda;
    int levi_factor;

    std::vector<double> arr_constraint;
    std::vector<double> arr_constraint_self;
    std::vector<double> arr_constraint_lower;

    std::vector<int> atom_tmp;

    std::vector<int> interaction_list;

    size_t nparam_sub;
    std::vector<size_t> nparams;

    std::vector<int> interaction_index, interaction_atom, interaction_tmp;
    interaction_index.resize(order + 2);
    interaction_atom.resize(order + 2);
    interaction_tmp.resize(order + 2);

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    ConstEntry const_tmp;

    for (int i = 0; i < maxorder; ++i) {
        nparams.push_back(fcs->get_nequiv()[i].size());
    }

    if (order == 0) {
        nparam_sub = nparams[order];
    } else {
        nparam_sub = nparams[order] + nparams[order - 1];
    }
    arr_constraint.resize(nparam_sub);
    arr_constraint_self.resize(nparams[order]);

    if (order > 0) arr_constraint_lower.resize(nparams[order - 1]);


    int **xyzcomponent = nullptr;

    if (order > 0) {
        nxyz = static_cast<int>(pow(static_cast<double>(3), order));
        allocate(xyzcomponent, nxyz, order);
        Fcs::get_xyzcomponent(order, xyzcomponent);
    }

    for (int i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        interaction_atom[0] = iat;

        if (order == 0) {
            auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
            std::sort(interaction_list_now.begin(), interaction_list_now.end());

            // Special treatment for harmonic force constants

            for (icrd = 0; icrd < 3; ++icrd) {
                interaction_index[0] = 3 * iat + icrd;

                for (int mu = 0; mu < 3; ++mu) {
                    for (int nu = 0; nu < 3; ++nu) {
                        if (!valid_rotation_axis[mu][nu]) continue;

                        // Clear history

                        for (int j = 0; j < nparam_sub; ++j) arr_constraint[j] = 0.0;

                        for (auto &iter_list: interaction_list_now) {
                            jat = iter_list;


                            atom_tmp.clear();
                            atom_tmp.push_back(jat);
                            const auto iter_cluster = cluster->get_interaction_cluster(order, i).find(
                                InteractionCluster(atom_tmp));

                            // Compute vec_for_rot
                            vec_for_rot.setZero();
                            if (iter_cluster != cluster->get_interaction_cluster(order, i).end()) {
                                const auto nsize_equiv = iter_cluster->cell.size();
                                for (int j = 0; j < nsize_equiv; ++j) {
                                    for (auto k = 0; k < 3; ++k) {
                                        vec_for_rot[k]
                                            += system->get_x_image()[iter_cluster->cell[j][0]](jat, k);
                                    }
                                }
                                // Take average
                                vec_for_rot /= static_cast<double>(nsize_equiv);
                            }

                            interaction_index[1] = 3 * jat + mu;
                            auto iter_found = list_found.find(FcProperty(order + 2,
                                                                         1.0,
                                                                         &interaction_index[0],
                                                                         1));

                            if (iter_found != list_found.end()) {
                                arr_constraint[iter_found->mother]
                                    += iter_found->sign * vec_for_rot[nu];
                            }

                            // Exchange mu <--> nu and repeat.
                            // Note that the sign is inverted (+ --> -) in the summation

                            interaction_index[1] = 3 * jat + nu;
                            iter_found = list_found.find(FcProperty(order + 2,
                                                                    1.0,
                                                                    &interaction_index[0],
                                                                    1));
                            if (iter_found != list_found.end()) {
                                arr_constraint[iter_found->mother]
                                    -= iter_found->sign * vec_for_rot[mu];
                            }
                        }

                        if (!is_allzero(arr_constraint, tolerance, loc_nonzero)) {
                            // Add to constraint list
                            if (arr_constraint[loc_nonzero] < 0.0) {
                                for (int j = 0; j < nparam_sub; ++j) arr_constraint[j] *= -1.0;
                            }
                            const_tmp.clear();
                            for (int j = 0; j < nparam_sub; ++j) {
                                if (std::abs(arr_constraint[j]) >= tolerance) {
                                    const_tmp.emplace_back(j, arr_constraint[j]);
                                }
                            }
                            const_self_vec[order].emplace_back(const_tmp);
                        }
                    } // nu
                }     // mu
            }
        } else {
            // Constraint between different orders

            auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
            auto interaction_list_old(cluster->get_atoms_in_cutoff(order - 1, i));
            std::sort(interaction_list_now.begin(), interaction_list_now.end());
            std::sort(interaction_list_old.begin(), interaction_list_old.end());

            for (icrd = 0; icrd < 3; ++icrd) {
                interaction_index[0] = 3 * iat + icrd;

                const CombinationWithRepetition<int> g_now(interaction_list_now.begin(),
                                                           interaction_list_now.end(),
                                                           order);
                const CombinationWithRepetition<int> g_old(interaction_list_old.begin(),
                                                           interaction_list_old.end(),
                                                           order);

                // m    -th order --> (m-1)-th order
                // (m-1)-th order -->     m-th order
                // 2-different directions to find all constraints

                for (unsigned int direction = 0; direction < 2; ++direction) {
                    if (direction == 0) {
                        g = g_now;
                        interaction_list = interaction_list_now;
                    } else {
                        g = g_old;
                        interaction_list = interaction_list_old;
                    }

                    // Loop for the interacting pairs

                    do {
                        auto data = g.now();

                        for (size_t idata = 0; idata < data.size(); ++idata) {
                            interaction_atom[idata + 1] = data[idata];
                        }

                        for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                            for (int j = 0; j < order; ++j)
                                interaction_index[j + 1] = 3 * interaction_atom[j + 1] + xyzcomponent[ixyz][j];

                            for (int mu = 0; mu < 3; ++mu) {
                                for (int nu = 0; nu < 3; ++nu) {
                                    if (!valid_rotation_axis[mu][nu]) continue;

                                    // Search for a new constraint below

                                    for (int j = 0; j < nparam_sub; ++j) arr_constraint[j] = 0.0;

                                    // Loop for m_{N+1}, a_{N+1}
                                    for (auto &iter_list: interaction_list) {
                                        jat = iter_list;

                                        interaction_atom[order + 1] = jat;
                                        if (!cluster->is_incutoff(order + 2,
                                                                  &interaction_atom[0],
                                                                  order,
                                                                  system->get_supercell().kind)) {
                                            continue;
                                        }

                                        atom_tmp.clear();

                                        for (int j = 1; j < order + 2; ++j) {
                                            atom_tmp.push_back(interaction_atom[j]);
                                        }
                                        std::sort(atom_tmp.begin(), atom_tmp.end());

                                        const auto iter_cluster = cluster->get_interaction_cluster(order, i).find(
                                            InteractionCluster(atom_tmp));
                                        if (iter_cluster != cluster->get_interaction_cluster(order, i).end()) {
                                            int iloc = -1;

                                            for (int j = 0; j < atom_tmp.size(); ++j) {
                                                if (atom_tmp[j] == jat) {
                                                    iloc = j;
                                                    break;
                                                }
                                            }

                                            if (iloc == -1) {
                                                exit("generate_rotational_constraint", "This cannot happen.");
                                            }

                                            for (int j = 0; j < 3; ++j) vec_for_rot[j] = 0.0;

                                            const auto nsize_equiv = (*iter_cluster).cell.size();

                                            for (int j = 0; j < nsize_equiv; ++j) {
                                                for (auto k = 0; k < 3; ++k) {
                                                    vec_for_rot[k] += system->get_x_image()[(*iter_cluster).cell[j][
                                                        iloc]](jat, k);
                                                }
                                            }

                                            for (int j = 0; j < 3; ++j) {
                                                vec_for_rot[j] /= static_cast<double>(nsize_equiv);
                                            }
                                        }


                                        // mu, nu

                                        interaction_index[order + 1] = 3 * jat + mu;
                                        for (int j = 0; j < order + 2; ++j) interaction_tmp[j] = interaction_index[j];

                                        sort_tail(order + 2, interaction_tmp.data());

                                        auto iter_found = list_found.
                                            find(FcProperty(order + 2, 1.0, interaction_tmp.data(), 1));
                                        if (iter_found != list_found.end()) {
                                            arr_constraint[nparams[order - 1] + iter_found->mother]
                                                += iter_found->sign * vec_for_rot[nu];
                                        }

                                        // Exchange mu <--> nu and repeat again.

                                        interaction_index[order + 1] = 3 * jat + nu;
                                        for (int j = 0; j < order + 2; ++j) interaction_tmp[j] = interaction_index[j];

                                        sort_tail(order + 2, &interaction_tmp[0]);

                                        iter_found = list_found.
                                            find(FcProperty(order + 2, 1.0, &interaction_tmp[0], 1));
                                        if (iter_found != list_found.end()) {
                                            arr_constraint[nparams[order - 1] + iter_found->mother]
                                                -= iter_found->sign * vec_for_rot[mu];
                                        }
                                    }

                                    for (lambda = 0; lambda < order + 1; ++lambda) {
                                        mu_lambda = interaction_index[lambda] % 3;

                                        for (int jcrd = 0; jcrd < 3; ++jcrd) {
                                            for (int j = 0; j < order + 1; ++j) {
                                                interaction_tmp[j] = interaction_index[j];
                                            }

                                            interaction_tmp[lambda] = 3 * interaction_atom[lambda] + jcrd;

                                            levi_factor = 0;

                                            for (int j = 0; j < 3; ++j) {
                                                levi_factor += levi_civita(j, mu, nu)
                                                    * levi_civita(j, mu_lambda, jcrd);
                                            }

                                            if (levi_factor == 0) continue;

                                            sort_tail(order + 1, &interaction_tmp[0]);

                                            auto iter_found = list_found_last.find(FcProperty(order + 1,
                                                1.0,
                                                &interaction_tmp[0],
                                                1));
                                            if (iter_found != list_found_last.end()) {
                                                arr_constraint[iter_found->mother]
                                                    += iter_found->sign * static_cast<double>(levi_factor);
                                            }
                                        }
                                    }

                                    if (!is_allzero(arr_constraint, tolerance, loc_nonzero)) {
                                        // A Candidate for another constraint found !
                                        // Add to the appropriate set

                                        if (arr_constraint[loc_nonzero] < 0.0) {
                                            for (int j = 0; j < nparam_sub; ++j) arr_constraint[j] *= -1.0;
                                        }
                                        for (int j = 0; j < nparams[order]; ++j) {
                                            arr_constraint_self[j] = arr_constraint[j + nparams[order - 1]];
                                        }
                                        for (int j = 0; j < nparams[order - 1]; ++j) {
                                            arr_constraint_lower[j] = arr_constraint[j];
                                        }

                                        const_tmp.clear();

                                        if (is_allzero(arr_constraint_self, tolerance, loc_nonzero)) {
                                            // If all elements of the "order"th order is zero,
                                            // the constraint is intraorder of the "order-1"th order.
                                            for (int j = 0; j < nparams[order - 1]; ++j) {
                                                if (std::abs(arr_constraint_lower[j]) >= tolerance) {
                                                    const_tmp.emplace_back(j, arr_constraint_lower[j]);
                                                }
                                            }
                                            const_self_vec[order - 1].emplace_back(const_tmp);
                                        } else if (is_allzero(arr_constraint_lower, tolerance, loc_nonzero)) {
                                            // If all elements of the "order-1"th order is zero,
                                            // the constraint is intraorder of the "order"th order.
                                            for (int j = 0; j < nparams[order]; ++j) {
                                                if (std::abs(arr_constraint_self[j]) >= tolerance) {
                                                    const_tmp.emplace_back(j, arr_constraint_self[j]);
                                                }
                                            }
                                            const_self_vec[order].emplace_back(const_tmp);
                                        } else {
                                            // If nonzero elements exist in both of the "order-1" and "order",
                                            // the constraint is intrerorder.

                                            for (int j = 0; j < nparam_sub; ++j) {
                                                if (std::abs(arr_constraint[j]) >= tolerance) {
                                                    const_tmp.emplace_back(j, arr_constraint[j]);
                                                }
                                            }
                                            const_cross_vec[order].emplace_back(const_tmp);
                                        }
                                    }
                                } // nu
                            }     // mu
                        }         // ixyz
                    } while (g.next());
                } // direction
            }     // icrd
        }
    } // iat

    if (xyzcomponent) deallocate(xyzcomponent);

    status_constraint_subset["rotation"] = 1;
}

void Constraint::set_rotation_constraints_extra(const std::unique_ptr<System> &system,
                                                const std::unique_ptr<Symmetry> &symmetry,
                                                const std::unique_ptr<Cluster> &cluster,
                                                const std::unique_ptr<Fcs> &fcs,
                                                const int order,
                                                const bool valid_rotation_axis[3][3],
                                                const std::unordered_set<FcProperty> &list_found,
                                                const double tolerance,
                                                std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
                                                std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec)
{
    const auto maxorder = cluster->get_maxorder();

    // Additional constraint for the last order.
    // All IFCs over maxorder-th order are neglected.
    if (order != (maxorder - 1) or status_constraint_subset["rotation_extra"] == -1) return;

    int j;
    const auto natmin = symmetry->get_nat_trueprim();

    int iat;
    int icrd;
    int mu, nu;

    CombinationWithRepetition<int> g;

    int ixyz, nxyz{0};
    int loc_nonzero;

    int mu_lambda, lambda;
    int levi_factor;

    std::vector<double> arr_constraint;
    std::vector<double> arr_constraint_self;

    std::vector<int> atom_tmp;
    std::set<InteractionCluster>::iterator iter_cluster;

    std::vector<size_t> nparams;
    std::vector<int> interaction_index, interaction_atom, interaction_tmp;

    interaction_index.resize(order + 2);
    interaction_atom.resize(order + 2);
    interaction_tmp.resize(order + 2);

    typedef std::vector<ConstraintDoubleElement> ConstEntry;
    ConstEntry const_tmp;

    for (int i = 0; i < maxorder; ++i) {
        nparams.push_back(fcs->get_nequiv()[i].size());
    }

    arr_constraint_self.resize(nparams[order]);

    int **xyzcomponent = nullptr;

    nxyz = static_cast<int>(pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    fcs->get_xyzcomponent(order + 1, xyzcomponent);


    for (int i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        interaction_atom[0] = iat;

        auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
        std::sort(interaction_list_now.begin(), interaction_list_now.end());

        for (icrd = 0; icrd < 3; ++icrd) {
            interaction_index[0] = 3 * interaction_atom[0] + icrd;

            CombinationWithRepetition<int> g_now(interaction_list_now.begin(),
                                                 interaction_list_now.end(),
                                                 order + 1);
            do {
                auto data = g_now.now();

                for (auto idata = 0; idata < data.size(); ++idata)
                    interaction_atom[idata + 1] = data[idata];

                for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                    for (j = 0; j < order + 1; ++j)
                        interaction_index[j + 1] = 3 * interaction_atom[j + 1] + xyzcomponent[ixyz][j];

                    for (mu = 0; mu < 3; ++mu) {
                        for (nu = 0; nu < 3; ++nu) {
                            if (!valid_rotation_axis[mu][nu]) continue;

                            for (j = 0; j < nparams[order]; ++j) arr_constraint_self[j] = 0.0;

                            for (lambda = 0; lambda < order + 2; ++lambda) {
                                mu_lambda = interaction_index[lambda] % 3;

                                for (int jcrd = 0; jcrd < 3; ++jcrd) {
                                    for (j = 0; j < order + 2; ++j)
                                        interaction_tmp[j] = interaction_index[j];

                                    interaction_tmp[lambda] = 3 * interaction_atom[lambda] + jcrd;

                                    levi_factor = 0;
                                    for (j = 0; j < 3; ++j) {
                                        levi_factor += levi_civita(j, mu, nu) * levi_civita(j, mu_lambda, jcrd);
                                    }

                                    if (levi_factor == 0) continue;

                                    sort_tail(order + 2, &interaction_tmp[0]);

                                    auto iter_found = list_found.find(FcProperty(order + 2,
                                        1.0,
                                        &interaction_tmp[0],
                                        1));
                                    if (iter_found != list_found.end()) {
                                        arr_constraint_self[(*iter_found).mother]
                                            += (*iter_found).sign * static_cast<double>(levi_factor);
                                    }
                                } // jcrd
                            }     // lambda

                            if (!is_allzero(arr_constraint_self, tolerance, loc_nonzero)) {
                                if (arr_constraint_self[loc_nonzero] < 0.0) {
                                    for (j = 0; j < nparams[order]; ++j) arr_constraint_self[j] *= -1.0;
                                }
                                const_tmp.clear();
                                for (j = 0; j < nparams[order]; ++j) {
                                    if (std::abs(arr_constraint_self[j]) >= tolerance) {
                                        const_tmp.emplace_back(j, arr_constraint_self[j]);
                                    }
                                }
                                const_self_vec[order].emplace_back(const_tmp);
                            }
                        } // nu
                    }     // mu
                }         // ixyz
            } while (g_now.next());
        } // icrd
    }     // iat
    deallocate(xyzcomponent);
}


int Constraint::levi_civita(const int i,
                            const int j,
                            const int k)
{
    return (j - i) * (k - i) * (k - j) / 2;
}


void Constraint::setup_rotation_axis(bool flag[3][3])
{
    for (auto mu = 0; mu < 3; ++mu) {
        for (auto nu = 0; nu < 3; ++nu) {
            if (mu == nu) {
                flag[mu][nu] = false;
            } else {
                flag[mu][nu] = true;
            }
        }
    }
    std::sort(rotation_axis.begin(), rotation_axis.end());

    if (rotation_axis == "x") {
        flag[0][1] = false;
        flag[1][0] = false;
        flag[0][2] = false;
        flag[2][0] = false;
    } else if (rotation_axis == "y") {
        flag[0][1] = false;
        flag[1][0] = false;
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "z") {
        flag[0][2] = false;
        flag[2][0] = false;
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "xy") {
        flag[0][1] = false;
        flag[1][0] = false;
    } else if (rotation_axis == "yz") {
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "xz") {
        flag[0][2] = false;
        flag[2][0] = false;
    } else if (rotation_axis == "xyz") {
        // do nothing
    } else {
        warn("setup_rotation_axis",
             "Invalid rotation_axis. Default value(xyz) will be used.");
    }
}


void Constraint::get_forceconstants_from_file(const int order,
                                              const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs,
                                              const std::string &file_to_fix,
                                              std::vector<std::vector<int>> &intpair_fcs,
                                              std::vector<double> &fcs_values)
{
    const auto file_extension = file_to_fix.substr(file_to_fix.find_last_of('.') + 1);

    if (file_extension == "xml" || file_extension == "XML") {
        parse_forceconstants_from_xml(order,
                                      symmetry,
                                      fcs,
                                      file_to_fix,
                                      intpair_fcs,
                                      fcs_values);
    } else if (file_extension == "h5" || file_extension == "hdf5") {
        parse_forceconstants_from_h5(order,
                                     symmetry,
                                     fcs,
                                     file_to_fix,
                                     intpair_fcs,
                                     fcs_values);
    } else {
        exit("get_forceconstants_from_file", "unacceptable extension type");
    }
}

void Constraint::generate_huang_constraint(const Cell &supercell,
                                           const std::unique_ptr<Symmetry> &symmetry,
                                           const std::unique_ptr<Cluster> &cluster,
                                           const std::unique_ptr<Fcs> &fcs,
                                           const std::vector<Eigen::MatrixXd> &x_image,
                                           const int verbosity)
{
    // Create constraint matrix for the Huang constraints.

    if (const_huang.empty()) {
        const_huang.resize(1);
    }

    if (status_constraint_subset["huang"] == -1) return;

    if (verbosity > 0) {
        std::cout << "  Generating constraints for Huang invariance ...";
    }
    const_huang[0].clear();
    if (verbosity > 0) std::cout << " done.\n" << std::flush;
    const auto nparams = fcs->get_nequiv()[0].size();
    if (nparams == 0) {
        if (verbosity > 0) std::cout << "  No parameters! Skipped.\n";
        return;
    }
    std::unordered_set<FcProperty> list_found;

    list_found.clear();

    // Accumulate sets of non-zero force constants.
    for (auto p = fcs->get_fc_table()[0].begin();
         p != fcs->get_fc_table()[0].end(); ++p) {
        list_found.insert(FcProperty(2,
                                     p->sign,
                                     p->elems.data(),
                                     p->mother));
    }


    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;

    int pair1[2], pair2[2];
    std::vector<int> atom_tmp;
    std::vector<std::vector<Eigen::Matrix3d>> relvec_tensor;
    Eigen::Vector3d vec_tmp;
    std::vector<double> const_now;

    relvec_tensor.resize(natmin, std::vector<Eigen::Matrix3d>(nat));
    using ConstEntry = std::vector<ConstraintDoubleElement>;
    ConstEntry const_tmp;
    std::vector<ConstEntry> const_list;

    // Construct relative vector products considering periodic images
    for (size_t iat = 0; iat < natmin; ++iat) {
        const auto iat_s = symmetry->get_map_trueprim_to_super()[iat][0];

        for (size_t jat = 0; jat < nat; ++jat) {
            atom_tmp.clear();
            atom_tmp.push_back(jat);

            relvec_tensor[iat][jat].setZero();

            const auto iter_cluster = cluster->get_interaction_cluster(0, iat).find(
                InteractionCluster(atom_tmp));

            if (iter_cluster != cluster->get_interaction_cluster(0, iat).end()) {
                const auto nsize_equiv = iter_cluster->cell.size();
                // const auto nsize_equiv = 1;
                for (int j = 0; j < nsize_equiv; ++j) {
                    for (auto k = 0; k < 3; ++k) {
                        vec_tmp[k] = x_image[iter_cluster->cell[j][0]](jat, k)
                                     - x_image[0](iat_s, k);
                    }
                    for (auto k = 0; k < 3; ++k) {
                        for (auto m = 0; m < 3; ++m) {
                            relvec_tensor[iat][jat](k, m) += vec_tmp[k] * vec_tmp[m];
                        }
                    }
                }
                relvec_tensor[iat][jat] /= static_cast<double>(nsize_equiv);
            }
        }
    }

    const_now.resize(nparams);
    const_list.clear();

    int loc_nonzero;

    for (size_t mu1 = 0; mu1 < 3; ++mu1) {
        for (size_t nu1 = 0; nu1 < 3; ++nu1) {
            for (size_t mu2 = 0; mu2 < 3; ++mu2) {
                for (size_t nu2 = 0; nu2 < 3; ++nu2) {

                    if (mu1 == mu2 && nu1 == nu2) {
                        // Skip the case where mu1 == mu2 and nu1 == nu2
                        continue;
                    }

                    // Reset the temporary array for the current constraint
                    for (size_t j = 0; j < nparams; ++j) {
                        const_now[j] = 0.0;
                    }

                    for (size_t iat = 0; iat < natmin; ++iat) {
                        pair1[0] = 3 * iat + mu1; // (0k1;alpha)
                        pair2[0] = 3 * iat + mu2; // (0k1;gamma)

                        for (size_t jat = 0; jat < nat; ++jat) {
                            pair1[1] = 3 * jat + nu1; // (l2k2;beta)
                            pair2[1] = 3 * jat + nu2; // (l2k2;delta)

                            auto iter_found = list_found.find(FcProperty(2,
                                                                         1.0,
                                                                         &pair1[0],
                                                                         1));

                            auto iter_found2 = list_found.find(FcProperty(2,
                                                                          1.0,
                                                                          &pair2[0],
                                                                          1));

                            if (iter_found != list_found.end()) {
                                const_now[iter_found->mother] += iter_found->sign
                                    * relvec_tensor[iat][jat](mu2, nu2);
                            }
                            if (iter_found2 != list_found.end()) {
                                const_now[iter_found2->mother] -= iter_found2->sign
                                    * relvec_tensor[iat][jat](mu1, nu1);
                            }
                        }
                    }

                    const_tmp.clear();
                    if (!is_allzero(const_now, eps8, loc_nonzero, 0)) {
                        if (const_now[loc_nonzero] < 0.0) {
                            for (size_t j = 0; j < nparams; ++j) {
                                const_now[j] *= -1.0;
                            }
                        }
                        for (size_t j = 0; j < nparams; ++j) {
                            if (std::abs(const_now[j]) >= eps8) {
                                const_tmp.emplace_back(j, const_now[j]);
                            }
                        }
                        const_list.emplace_back(const_tmp);
                    }
                }
            }
        }
    }
    MapConstraintElement const_copy;

    auto division_factor = 1.0;
    for (const auto &it: const_list) {
        auto counter = 0;
        const_copy.clear();
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_copy[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_huang[0].emplace_back(const_copy);
    }

    rref_sparse(nparams, const_huang[0], eps8);

    status_constraint_subset["huang"] = 1;
}

void Constraint::parse_forceconstants_from_xml(const int order,
                                               const std::unique_ptr<Symmetry> &symmetry,
                                               const std::unique_ptr<Fcs> &fcs,
                                               const std::string &file_to_fix,
                                               std::vector<std::vector<int>> &intpair_fcs,
                                               std::vector<double> &fcs_values)
{
    using namespace boost::property_tree;
    ptree pt;

    try {
        read_xml(file_to_fix, pt);
    } catch (std::exception &e) {
        std::string str_error;
        if (order == 0) {
            str_error = "Cannot open file FC2FIX ( " + file_to_fix + " )";
        } else if (order == 1) {
            str_error = "Cannot open file FC3FIX ( " + file_to_fix + " )";
        }
        exit("fix_forceconstants_to_file", str_error.c_str());
    }

    const auto version_from_file = get_value_from_xml(pt, "Data.ALM_version");

    std::vector<std::string> version_array;
    boost::split(version_array, version_from_file, boost::is_any_of("."));
    std::vector<int> version_array_int;
    for (const auto &it: version_array) {
        version_array_int.emplace_back(std::stoi(it));
    }
    if ((version_array_int[0] <= 1) && (version_array_int[1] <= 4)) {
        exit("fit_forceconstants_to_file",
             "FCSXML files generated by older versions (<=1.4.2) do not have compatibility with\n"
             " a newer version for fixing force constants. Please use a newer version (>=1.5) and"
             " regenerate the FC2FIX or FC3FIX files");
    }

    const auto nat_ref = boost::lexical_cast<size_t>(
        get_value_from_xml(pt, "Data.Structure.NumberOfAtoms"));
    const auto ntran_ref = boost::lexical_cast<size_t>(
        get_value_from_xml(pt, "Data.Symmetry.NumberOfTranslations"));
    const auto natmin_ref = nat_ref / ntran_ref;

    if (natmin_ref != symmetry->get_nat_trueprim()) {
        exit("fix_forceconstants_to_file",
             "The number of atoms in the primitive cell is not consistent.");
    }

    const auto nfcs = fcs->get_nequiv()[order].size();

    if (order == 0) {
        const auto nfcs_ref = boost::lexical_cast<size_t>(
            get_value_from_xml(pt, "Data.ForceConstants.HarmonicUnique.NFC2"));

        if (nfcs_ref != nfcs) {
            exit("fix_forceconstants_to_file",
                 "The number of harmonic force constants is not consistent.");
        }

        auto preferred_basis_ref = boost::lexical_cast<std::string>(
            get_value_from_xml(pt, "Data.ForceConstants.HarmonicUnique.Basis", 0));

        if (preferred_basis_ref.empty()) preferred_basis_ref = "Cartesian";

        if (preferred_basis_ref != fcs->get_forceconstant_basis()) {
            exit("fix_forceconstants_to_file",
                 "The basis of harmonic force constants is not consistent.");
        }
    } else if (order == 1) {
        const auto nfcs_ref = boost::lexical_cast<size_t>(
            get_value_from_xml(pt, "Data.ForceConstants.CubicUnique.NFC3"));

        if (nfcs_ref != nfcs) {
            exit("fix_forceconstants_to_file",
                 "The number of cubic force constants is not consistent.");
        }

        auto preferred_basis_ref = boost::lexical_cast<std::string>(
            get_value_from_xml(pt, "Data.ForceConstants.CubicUnique.Basis", 0));
        if (preferred_basis_ref.empty()) preferred_basis_ref = "Cartesian";

        if (preferred_basis_ref != fcs->get_forceconstant_basis()) {
            exit("fix_forceconstants_to_file",
                 "The basis of cubic force constants is not consistent.");
        }
    }

    intpair_fcs.resize(nfcs, std::vector<int>(order + 2));
    fcs_values.resize(nfcs);
    std::vector<std::vector<int>> intpairs_to_fix;

    intpairs_to_fix.resize(nfcs, std::vector<int>(2));

    int counter = 0;
    if (order == 0) {
        BOOST_FOREACH(const ptree::value_type &child_,
                      pt.get_child("Data.ForceConstants.HarmonicUnique")) {
            if (child_.first == "FC2") {
                const auto &child = child_.second;
                const auto str_intpair = child.get<std::string>("<xmlattr>.pairs");
                const auto str_multiplicity = child.get<std::string>("<xmlattr>.multiplicity");

                std::istringstream is(str_intpair);
                is >> intpair_fcs[counter][0] >> intpair_fcs[counter][1];
                fcs_values[counter] = boost::lexical_cast<double>(child.data());
                ++counter;
            }
        }
    } else if (order == 1) {
        BOOST_FOREACH(const ptree::value_type &child_,
                      pt.get_child("Data.ForceConstants.CubicUnique")) {
            if (child_.first == "FC3") {
                const auto &child = child_.second;
                const auto str_intpair = child.get<std::string>("<xmlattr>.pairs");
                const auto str_multiplicity = child.get<std::string>("<xmlattr>.multiplicity");

                std::istringstream is(str_intpair);
                is >> intpair_fcs[counter][0] >> intpair_fcs[counter][1] >> intpair_fcs[counter][2];
                fcs_values[counter] = boost::lexical_cast<double>(child.data());
                ++counter;
            }
        }
    }
}

void Constraint::parse_forceconstants_from_h5(const int order,
                                              const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs,
                                              const std::string &file_to_fix,
                                              std::vector<std::vector<int>> &intpair_fcs,
                                              std::vector<double> &fcs_values)
{
    H5Easy::File file(file_to_fix, H5Easy::File::ReadOnly);

    const std::string celltype = "SuperCell";

    std::vector<std::vector<int>> mapping_table;

    get_mapping_table_from_h5(file, celltype, mapping_table);

    const auto nat_trueprim_file = mapping_table.size();

    if (nat_trueprim_file != symmetry->get_nat_trueprim()) {
        exit("parse_forceconstants_from_h5",
             "The number of atoms in the true primitive cell is not consistent.");
    }

    Eigen::MatrixXi atom_indices_file, atom_indices_super_file;
    Eigen::MatrixXi coord_indices_file;
    Eigen::MatrixXd shift_vectors_file;
    Eigen::ArrayXd fcs_values_file;

    get_force_constants_from_h5(file,
                                order,
                                atom_indices_file,
                                atom_indices_super_file,
                                coord_indices_file,
                                shift_vectors_file,
                                fcs_values_file);

    // fcs_values_file are force constants in the Cartesian basis.
    // They need to be converted to the lattice basis if the FCSYM_BASIS = Lattice.


    std::vector<ForceConstantTable> fc_cart, fc_cart_unique, fc_cart_copy;
    std::vector<int> flatten_array(order + 2);

    fc_cart.clear();
    for (auto i = 0; i < atom_indices_super_file.rows(); ++i) {
        for (auto j = 0; j < order + 2; ++j) {
            flatten_array[j] = atom_indices_super_file(i, j) * 3 + coord_indices_file(i, j);
        }
        fc_cart.emplace_back(fcs_values_file[i], flatten_array);
    }

    // Merge force constant entries having the same flatten arrays
    std::sort(fc_cart.begin(), fc_cart.end());

    fc_cart_unique.clear();
    fc_cart_copy.clear();
    for (auto i = 0; i < order + 2; ++i) flatten_array[i] = -1;
    double fc_sum = 0.0;
    for (const auto &it: fc_cart) {
        if (flatten_array == it.flattenarray) {
            fc_sum += it.fc_value;
        } else {
            if (std::abs(fc_sum) > 0.0) {
                fc_cart_copy.emplace_back(fc_sum, flatten_array);
            }
            fc_sum = it.fc_value;
            flatten_array = it.flattenarray;
        }
    }
    if (std::abs(fc_sum) > 0.0) {
        fc_cart_copy.emplace_back(fc_sum, flatten_array);
    }
    fc_cart.clear();

    std::copy_if(fc_cart_copy.begin(),
                 fc_cart_copy.end(),
                 std::back_inserter(fc_cart_unique),
                 [](const ForceConstantTable &obj) {
                     return obj.is_ascending_order;
                 });

    std::vector<int> index_tmp(order + 2);

    if (fcs->get_forceconstant_basis() == "Lattice") {
        std::vector<ForceConstantTable> fc_lattice;
        fcs->change_basis_force_constants(fc_cart_unique, fc_lattice, 1);

        for (const auto &it: fc_lattice) {
            if (it.is_ascending_order) {
                for (auto i = 0; i < order + 2; ++i) {
                    index_tmp[i] = it.flattenarray[i];
                }
                intpair_fcs.emplace_back(index_tmp);
                fcs_values.emplace_back(it.fc_value);
            }
        }
    } else {
        for (const auto &it: fc_cart_unique) {
            for (auto i = 0; i < order + 2; ++i) {
                index_tmp[i] = it.flattenarray[i];
            }
            intpair_fcs.emplace_back(index_tmp);
            fcs_values.emplace_back(it.fc_value);
        }
    }
}

void Constraint::set_forceconstants_to_fix(const std::vector<std::vector<int>> &intpair_fix,
                                           const std::vector<double> &values_fix)
{
    const auto nelems = intpair_fix[0].size();
    const auto order = nelems - 2;

    if (order == 0) {
        intpair_fix_fc2 = intpair_fix;
        values_fix_fc2 = values_fix;
        status_constraint_subset["fix2"] = 0;
    } else if (order == 1) {
        intpair_fix_fc3 = intpair_fix;
        values_fix_fc3 = values_fix;
        status_constraint_subset["fix3"] = 0;
    } else {
        exit("fit_forceconstants", "Currently, only harmonic and cubic terms can be fixed.");
    }
}

void Constraint::generate_fix_constraint(const std::unique_ptr<Symmetry> &symmetry,
                                         const std::unique_ptr<Fcs> &fcs)
{
    if (status_constraint_subset["fix2"] == 0) {
        const auto order = 0;
        auto intpair_to_fix = intpair_fix_fc2;

        fcs->translate_forceconstant_index_to_centercell(symmetry,
                                                         intpair_to_fix);
        std::set<ForceConstantTable> fc_fix_table;

        const auto nfcs = intpair_to_fix.size();

        for (auto i = 0; i < nfcs; ++i) {
            fc_fix_table.insert(ForceConstantTable(values_fix_fc2[i],
                                                   intpair_to_fix[i]));
        }

        size_t ihead = 0;

        double sign;
        std::set<ForceConstantTable>::iterator it_found;

        const_fix[order].clear();
        const_fix[order].shrink_to_fit();

        for (unsigned int ui = 0; ui < fcs->get_nequiv()[order].size(); ++ui) {
            size_t mother = fcs->get_fc_table()[order][ihead].mother;
            bool found_element = false;

            for (auto j = 0; j < fcs->get_nequiv()[order][ui]; ++j) {
                std::vector<int> index_tmp = fcs->get_fc_table()[order][ihead + j].elems;

                it_found = fc_fix_table.find(ForceConstantTable(0.0, index_tmp));

                if (it_found != fc_fix_table.end()) {
                    found_element = true;
                    sign = fcs->get_fc_table()[order][ihead + j].sign;
                    break;
                }
            }

            if (found_element) {
                const_fix[order].emplace_back(mother,
                                              sign * it_found->fc_value);
            } else {
                const_fix[order].emplace_back(mother, 0.0);
            }

            ihead += fcs->get_nequiv()[order][ui];
        }
        status_constraint_subset["fix2"] = 1;
    }

    if (status_constraint_subset["fix3"] == 0 and const_fix.size() > 1) {
        const auto order = 1;
        auto intpair_to_fix = intpair_fix_fc3;

        fcs->translate_forceconstant_index_to_centercell(symmetry,
                                                         intpair_to_fix);

        const auto nfcs = intpair_to_fix.size();

        std::set<ForceConstantTable> fc_fix_table;

        for (auto i = 0; i < nfcs; ++i) {
            fc_fix_table.insert(ForceConstantTable(values_fix_fc3[i],
                                                   intpair_to_fix[i]));
        }

        size_t ihead = 0;

        double sign;
        std::set<ForceConstantTable>::iterator it_found;

        const_fix[order].clear();
        const_fix[order].shrink_to_fit();

        for (unsigned int ui = 0; ui < fcs->get_nequiv()[order].size(); ++ui) {
            size_t mother = fcs->get_fc_table()[order][ihead].mother;
            bool found_element = false;

            for (auto j = 0; j < fcs->get_nequiv()[order][ui]; ++j) {
                std::vector<int> index_tmp = fcs->get_fc_table()[order][ihead + j].elems;
                it_found = fc_fix_table.find(ForceConstantTable(0.0, index_tmp));

                if (it_found != fc_fix_table.end()) {
                    found_element = true;
                    sign = fcs->get_fc_table()[order][ihead + j].sign;
                    break;
                }
            }

            if (found_element) {
                const_fix[order].emplace_back(mother,
                                              sign * it_found->fc_value);
            } else {
                const_fix[order].emplace_back(mother, 0.0);
            }
            ihead += fcs->get_nequiv()[order][ui];
        }

        status_constraint_subset["fix3"] = 1;
    }
}


bool Constraint::is_allzero(const int n,
                            const double *arr,
                            const int nshift)
{
    for (auto i = nshift; i < n; ++i) {
        if (std::abs(arr[i]) > eps10) {
            return false;
        }
    }
    return true;
}

bool Constraint::is_allzero(const std::vector<int> &vec,
                            int &loc)
{
    loc = -1;
    for (auto i = 0; i < vec.size(); ++i) {
        if (std::abs(vec[i]) > 0) {
            loc = i;
            return false;
        }
    }
    return true;
}

bool Constraint::is_allzero(const std::vector<double> &vec,
                            const double tol,
                            int &loc,
                            const int nshift)
{
    loc = -1;
    const auto n = vec.size();
    for (auto i = nshift; i < n; ++i) {
        if (std::abs(vec[i]) > tol) {
            loc = i;
            return false;
        }
    }
    return true;
}


void Constraint::remove_redundant_rows(const size_t n,
                                       std::vector<ConstraintClass> &Constraint_vec,
                                       const double tolerance) const
{

    auto nparam = n;
    const auto nconst = Constraint_vec.size();

    if (nconst > 0) {
        double **mat_tmp;
        double *arr_tmp;
        size_t nrank;
        size_t j;
        allocate(mat_tmp, nconst, nparam);

        size_t i = 0;

        for (auto &p: Constraint_vec) {
            for (j = 0; j < nparam; ++j) {
                mat_tmp[i][j] = p.w_const[j];
            }
            ++i;
        }

        rref(nconst, nparam, mat_tmp, nrank, tolerance);

        allocate(arr_tmp, nparam);

        Constraint_vec.clear();

        for (i = 0; i < nrank; ++i) {
            for (j = 0; j < nparam; ++j) arr_tmp[j] = 0.0;
            long iloc = -1;
            for (j = 0; j < nparam; ++j) {
                if (std::abs(mat_tmp[i][j]) < tolerance) {
                    arr_tmp[j] = 0.0;
                } else {
                    arr_tmp[j] = mat_tmp[i][j];
                }
                if (std::abs(arr_tmp[j]) >= tolerance) {
                    iloc = j;
                }
            }

            if (iloc != -1) {
                Constraint_vec.emplace_back(nparam, arr_tmp);
            }
        }

        deallocate(mat_tmp);
        deallocate(arr_tmp);
    }
}

void Constraint::print_constraint(const ConstraintSparseForm &const_in) const
{
    const auto nconst = const_in.size();
    auto counter = 0;
    std::cout << '\n';
    std::cout << "TOTAL CONST SIZE :" << std::setw(6) << nconst << '\n';
    for (const auto &it: const_in) {
        std::cout << "CONST : " << std::setw(5) << counter + 1 << '\n';
        for (const auto &it2: it) {
            std::cout << std::setw(5) << it2.first + 1;
            std::cout << std::setw(15) << it2.second << '\n';
        }
        std::cout << '\n';
        ++counter;
    }
}
