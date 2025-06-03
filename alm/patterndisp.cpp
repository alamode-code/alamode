/*
patterndisp.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include <iostream>
#include <iomanip>
#include "patterndisp.h"
#include "memory.h"
#include "error.h"
#include "system.h"
#include "fcs.h"
#include "cluster.h"
#include "symmetry.h"
#include "constants.h"
#include "mathfunctions.h"
#include "constraint.h"
#include <map>
#include <boost/bimap.hpp>
#include <Eigen/LU>

using namespace ALM_NS;

Displace::Displace()
{
    set_default_variables();
}

Displace::~Displace()
{
    deallocate_variables();
}

void Displace::gen_displacement_pattern(const std::unique_ptr<Cluster> &cluster,
                                        const std::unique_ptr<Symmetry> &symmetry,
                                        const std::unique_ptr<Fcs> &fcs,
                                        const std::unique_ptr<Constraint> &constraint,
                                        const std::unique_ptr<System> &system,
                                        const int verbosity)
{
    int order;
    const auto maxorder = cluster->get_maxorder();
    std::string preferred_basis;
    std::vector<int> group_tmp;
    ConstraintSparseForm *constsym;
    std::set<int> *include_set;
    std::set<DispAtomSet> *dispset;

    std::vector<size_t> *nequiv;
    std::vector<FcProperty> *fc_table, *fc_zeros;

    std::vector<ConstraintTypeFix> *const_fix_tmp;
    std::vector<ConstraintTypeRelate> *const_relate_tmp;
    boost::bimap<size_t, size_t> *index_bimap_tmp;
    const auto do_rref = true;

    if (verbosity > 0) {
        std::cout << " DISPLACEMENT PATTERN\n";
        std::cout << " ====================\n\n";
    }

    // Decide preferred basis (Cartesian or Lattice)
    auto ncompat_cart = 0;
    auto ncompat_latt = 0;
    for (const auto &it: symmetry->get_symmetry_data()) {
        if (it.compatible_with_cartesian) ++ncompat_cart;
        if (it.compatible_with_lattice) ++ncompat_latt;
    }
    if (ncompat_cart >= ncompat_latt) {
        preferred_basis = "Cartesian";
    } else {
        preferred_basis = "Lattice";
    }

    allocate(fc_table, maxorder);
    allocate(fc_zeros, maxorder);
    allocate(constsym, maxorder);
    allocate(nequiv, maxorder);
    allocate(const_fix_tmp, maxorder);
    allocate(const_relate_tmp, maxorder);
    allocate(index_bimap_tmp, maxorder);

    for (order = 0; order < maxorder; ++order) {

        fcs->generate_force_constant_table(order,
                                           system->get_supercell().number_of_atoms,
                                           cluster->get_unique_clusters(order),
                                           symmetry,
                                           preferred_basis,
                                           fc_table[order],
                                           nequiv[order],
                                           fc_zeros[order],
                                           false);

        fcs->get_constraint_symmetry(system->get_supercell().number_of_atoms,
                                     symmetry,
                                     order,
                                     preferred_basis,
                                     fc_table[order],
                                     nequiv[order].size(),
                                     constraint->get_tolerance_constraint(),
                                     constsym[order],
                                     do_rref);

    }

    for (order = 0; order < maxorder; ++order) {
        const_fix_tmp[order].clear();
    }

    constraint->get_mapping_constraint(maxorder,
                                       nequiv,
                                       constsym,
                                       const_fix_tmp,
                                       const_relate_tmp,
                                       index_bimap_tmp);

    if (verbosity > 0) {
        for (order = 0; order < maxorder; ++order) {
            std::cout << "  Number of free" << std::setw(9)
                << cluster->get_ordername(order) << " FCs : "
                << index_bimap_tmp[order].size() << '\n';
        }
        std::cout << '\n';
    }


    deallocate(constsym);
    deallocate(const_fix_tmp);
    deallocate(const_relate_tmp);
    deallocate(fc_zeros);

    allocate(include_set, maxorder);

    for (order = 0; order < maxorder; ++order) {
        include_set[order].clear();

        for (const auto &it: index_bimap_tmp[order]) {
            include_set[order].insert(it.right);
        }
    }

    deallocate(index_bimap_tmp);

    if (verbosity > 0) {
        std::cout << "  Generating displacement patterns in ";
        std::cout << "Cartesian coordinate... ";
    }

    allocate(dispset, maxorder);

    for (order = 0; order < maxorder; ++order) {

        size_t m = 0;

        auto fc_table_copy = fc_table[order];

        auto nbeg = 0;
        auto nend = 0;

        for (size_t i = 0; i < nequiv[order].size(); ++i) {

            nend += nequiv[order][i];
            std::sort(fc_table_copy.begin() + nbeg, fc_table_copy.begin() + nend);
            nbeg = nend;

            if (include_set[order].find(i) != include_set[order].end()) {

                group_tmp.clear();

                // Store first order + 1 indexes as a necessary displacement pattern.
                // Here, duplicate entries will be removed.
                // For example, (iij) will be reduced to (ij).
                for (auto j = 0; j < order + 1; ++j) {
                    group_tmp.push_back(fc_table_copy[m].elems[j]);

                }
                group_tmp.erase(std::unique(group_tmp.begin(), group_tmp.end()),
                                group_tmp.end());

                // Avoid equivalent entries using set.
                dispset[order].insert(DispAtomSet(group_tmp));

            }

            m += nequiv[order][i];
        }
    }
    deallocate(include_set);
    deallocate(nequiv);
    deallocate(fc_table);

    allocate(pattern_all, maxorder);
    // pattern_all is updated.
    generate_pattern_all(maxorder,
                         system->get_supercell().number_of_atoms,
                         system->get_supercell().lattice_vector,
                         symmetry,
                         dispset,
                         preferred_basis);

    deallocate(dispset);

    if (verbosity > 0) {
        std::cout << " done!\n\n";

        for (order = 0; order < maxorder; ++order) {
            std::cout << "  Number of disp. patterns for " << std::setw(9)
                << cluster->get_ordername(order) << " : "
                << pattern_all[order].size() << '\n';
        }
        std::cout << '\n';
    }
}

void Displace::set_trim_dispsign_for_evenfunc(const bool trim_dispsign_for_evenfunc_in)
{
    trim_dispsign_for_evenfunc = trim_dispsign_for_evenfunc_in;
}

std::string Displace::get_disp_basis() const
{
    return disp_basis;
}

void Displace::set_disp_basis(const std::string &disp_basis_in)
{
    disp_basis = disp_basis_in;
}

const std::vector<AtomWithDirection> &Displace::get_pattern_all(const int order) const
{
    return pattern_all[order];
}

void Displace::set_default_variables()
{
    trim_dispsign_for_evenfunc = true;
    disp_basis = "CART";
    pattern_all = nullptr;
}

void Displace::deallocate_variables()
{
    deallocate(pattern_all);
}

void Displace::generate_pattern_all(const int maxorder,
                                    const size_t nat,
                                    const Eigen::Matrix3d &lavec,
                                    const std::unique_ptr<Symmetry> &symmetry,
                                    const std::set<DispAtomSet> *dispset_in,
                                    const std::string &preferred_basis) const
{
    size_t i, j;
    int order;
    Eigen::Vector3d disp_tmp;

    std::vector<int> atoms, vec_tmp, nums;
    std::vector<double> directions, directions_copy;
    std::vector<std::vector<int>> *sign_prod, sign_reduced;


    allocate(sign_prod, maxorder);

    for (order = 0; order < maxorder; ++order) {
        vec_tmp.clear();
        generate_signvecs(order + 1, sign_prod[order], vec_tmp);
    }

    for (order = 0; order < maxorder; ++order) {

        pattern_all[order].clear();

        for (auto it = dispset_in[order].begin(); it != dispset_in[order].end(); ++it) {

            atoms.clear();
            directions.clear();
            nums.clear();

            for (i = 0; i < (*it).atomset.size(); ++i) {

                auto atom_tmp = (*it).atomset[i] / 3;

                nums.push_back((*it).atomset[i]);
                atoms.push_back(atom_tmp);

                for (j = 0; j < 3; ++j) {
                    disp_tmp[j] = 0.0;
                }
                disp_tmp[(*it).atomset[i] % 3] = 1.0;

                for (j = 0; j < 3; ++j) directions.push_back(disp_tmp[j]);
            }

            const auto natom_disp = atoms.size();

            if (trim_dispsign_for_evenfunc) {
                find_unique_sign_pairs(natom_disp,
                                       nat,
                                       symmetry,
                                       sign_prod[natom_disp - 1],
                                       nums,
                                       sign_reduced,
                                       preferred_basis);
            } else {
                sign_reduced.clear();
                std::copy(sign_prod[natom_disp - 1].begin(),
                          sign_prod[natom_disp - 1].end(),
                          std::back_inserter(sign_reduced));
            }

            directions_copy.clear();
            std::copy(directions.begin(),
                      directions.end(),
                      std::back_inserter(directions_copy));

            for (const auto &it2: sign_reduced) {
                directions.clear();

                for (i = 0; i < it2.size(); ++i) {
                    const auto sign_double = static_cast<double>(it2[i]);

                    for (j = 0; j < 3; ++j) {
                        disp_tmp(j) = directions_copy[3 * i + j] * sign_double;
                    }
                    if (preferred_basis == "Lattice") {
                        disp_tmp = lavec.inverse() * disp_tmp;
                        disp_tmp /= disp_tmp.norm(); // Do we really need this?
                    }

                    for (j = 0; j < 3; ++j) {
                        directions.push_back(disp_tmp(j));
                    }
                }
                pattern_all[order].emplace_back(atoms, directions);
            }
        }
    }

    deallocate(sign_prod);
}

void Displace::generate_signvecs(const int N,
                                 std::vector<std::vector<int>> &sign,
                                 std::vector<int> vec) const
{
    // returns the product of signs ('+','-')

    if (N == 0) {
        sign.push_back(vec);
        vec.clear();
    } else {
        std::vector<int> vec_tmp;

        vec_tmp.clear();
        std::copy(vec.begin(), vec.end(), std::back_inserter(vec_tmp));

        vec_tmp.push_back(1);
        generate_signvecs(N - 1, sign, vec_tmp);

        vec_tmp.clear();
        std::copy(vec.begin(), vec.end(), std::back_inserter(vec_tmp));
        vec_tmp.push_back(-1);
        generate_signvecs(N - 1, sign, vec_tmp);
    }
}

void Displace::find_unique_sign_pairs(const int natom_disp_in,
                                      const size_t nat,
                                      const std::unique_ptr<Symmetry> &symmetry,
                                      const std::vector<std::vector<int>> &sign_in,
                                      const std::vector<int> &pair_in,
                                      std::vector<std::vector<int>> &sign_out,
                                      const std::string &preferred_basis) const
{
    size_t isym, i;
    int j, k;
    int mapped_atom;
    int mapped_index;

    bool flag_avail;

    double disp_tmp;
    double **disp, **disp_sym;

    std::vector<int> symnum_vec;
    std::vector<int>::const_iterator loc;
    std::vector<int> pair_tmp;
    std::vector<std::vector<int>> sign_found;
    std::vector<int> sign_tmp;
    std::vector<int> list_disp_atom;
    std::vector<IndexWithSign> index_for_sort;

    allocate(disp, nat, 3);
    allocate(disp_sym, nat, 3);

    sign_out.clear();
    symnum_vec.clear();
    list_disp_atom.clear();

    for (i = 0; i < pair_in.size(); ++i) {
        list_disp_atom.push_back(pair_in[i] / 3);
    }
    list_disp_atom.erase(std::unique(list_disp_atom.begin(), list_disp_atom.end()),
                         list_disp_atom.end());

    for (i = 0; i < nat; ++i) {
        for (j = 0; j < 3; ++j) {
            disp[i][j] = 0.0;
        }
    }

    for (i = 0; i < pair_in.size(); ++i) {
        disp[pair_in[i] / 3][pair_in[i] % 3] = 1.0;
    }

    const auto natom_disp = static_cast<size_t>(natom_disp_in);

    // Find symmetry operations which can be used to
    // reduce the number of sign patterns (+, -) of displacements

    for (isym = 0; isym < symmetry->get_nsym(); ++isym) {

        flag_avail = true;
        pair_tmp.clear();

        for (i = 0; i < natom_disp; ++i) {
            if (!flag_avail) break;

            mapped_atom = symmetry->get_map_sym()[pair_in[i] / 3][isym];
            mapped_index = 3 * mapped_atom + pair_in[i] % 3;

            loc = std::find(pair_in.begin(), pair_in.end(), mapped_index);
            if (loc == pair_in.end()) {
                flag_avail = false;
            }

            pair_tmp.push_back(mapped_index);
        }

        if (!flag_avail) continue;

        // The symmetry may be useful only when the pair doesn't change
        // by the symmetry operation.
        std::sort(pair_tmp.begin(), pair_tmp.end());

        if (pair_tmp == pair_in) {

            pair_tmp.clear();

            for (i = 0; i < list_disp_atom.size(); ++i) {
                mapped_atom = symmetry->get_map_sym()[list_disp_atom[i]][isym];

                for (j = 0; j < 3; ++j) {
                    disp_sym[mapped_atom][j] = 0.0;

                    if (preferred_basis == "Cartesian") {
                        for (k = 0; k < 3; ++k) {
                            disp_sym[mapped_atom][j]
                                +=
                                symmetry->get_symmetry_data()[isym].rotation_cart(j, k) *
                                disp[list_disp_atom[i]][k];
                        }
                    } else if (preferred_basis == "Lattice") {
                        for (k = 0; k < 3; ++k) {
                            disp_sym[mapped_atom][j]
                                += static_cast<double>(symmetry->get_symmetry_data()[isym].rotation(j, k))
                                * disp[list_disp_atom[i]][k];
                        }
                    } else {
                        exit("find_unique_sign_pairs",
                             "Invalid basis. This cannot happen.");
                    }

                    disp_tmp = disp_sym[mapped_atom][j];
                    if (std::abs(disp_tmp) > eps) {
                        pair_tmp.push_back(3 * mapped_atom + j);
                    }
                }
            }

            std::sort(pair_tmp.begin(), pair_tmp.end());

            if (pair_tmp == pair_in) {
                symnum_vec.push_back(isym);
            }
        }
    }

    // Now find unique pairs of displacement directions

    sign_found.clear();

    for (const auto &it: sign_in) {

        // if the sign has already been found before, cycle the loop.
        // else, add the current sign pairs to the return variable.
        if (std::find(sign_found.begin(), sign_found.end(), it) != sign_found.end()) {
            continue;
        }
        sign_out.push_back(it);

        for (i = 0; i < nat; ++i) {
            for (j = 0; j < 3; ++j) {
                disp[i][j] = 0.0;
            }
        }

        for (i = 0; i < natom_disp; ++i) {
            disp[pair_in[i] / 3][pair_in[i] % 3] = static_cast<double>(it[i]);
        }

        for (isym = 0; isym < symnum_vec.size(); ++isym) {

            index_for_sort.clear();

            for (i = 0; i < list_disp_atom.size(); ++i) {
                mapped_atom = symmetry->get_map_sym()[list_disp_atom[i]][symnum_vec[isym]];

                for (j = 0; j < 3; ++j) {
                    disp_sym[mapped_atom][j] = 0.0;
                    if (preferred_basis == "Cartesian") {
                        for (k = 0; k < 3; ++k) {
                            disp_sym[mapped_atom][j]
                                += symmetry->get_symmetry_data()[symnum_vec[isym]].rotation_cart(j, k)
                                * disp[list_disp_atom[i]][k];
                        }
                    } else if (preferred_basis == "Lattice") {
                        for (k = 0; k < 3; ++k) {
                            disp_sym[mapped_atom][j]
                                +=
                                static_cast<double>(symmetry->get_symmetry_data()[symnum_vec[isym]].rotation(j, k))
                                * disp[list_disp_atom[i]][k];
                        }
                    } else {
                        exit("find_unique_sign_pairs",
                             "Invalid basis. This cannot happen.");
                    }

                    disp_tmp = disp_sym[mapped_atom][j];

                    if (std::abs(disp_tmp) > eps) {

                        if (disp_tmp < 0.0) {
                            index_for_sort.emplace_back(3 * mapped_atom + j, -1);
                        } else {
                            index_for_sort.emplace_back(3 * mapped_atom + j, 1);
                        }
                    }
                }

            }
            std::sort(index_for_sort.begin(), index_for_sort.end());

            sign_tmp.clear();

            for (i = 0; i < index_for_sort.size(); ++i) {
                sign_tmp.push_back(index_for_sort[i].sign);
            }

            if ((sign_tmp.size() == natom_disp) &&
                (std::find(sign_found.begin(), sign_found.end(), sign_tmp) == sign_found.end())) {
                sign_found.push_back(sign_tmp);
                std::sort(sign_found.begin(), sign_found.end());
            }
        }
    }

    deallocate(disp);
    deallocate(disp_sym);
}
