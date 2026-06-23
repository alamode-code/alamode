/*
 fcs.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "fcs.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include "cluster.h"
#include "constraint.h"
#include "error.h"
#include "memory.h"
#include "rref.h"
#include "symmetry.h"
#include "timer.h"

#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <unordered_set>

#include "../external/combination.hpp"
#include "least_squares.h"

#if defined(_WIN32) || defined(_WIN64)
#undef min
#undef max
#endif

using namespace ALM_NS;

Fcs::Fcs()
{
    set_default_variables();
};

Fcs::~Fcs()
{
    deallocate_variables();
};

auto Fcs::init(const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Symmetry> &symmetry,
               const Cell &supercell, const int verbosity, std::unique_ptr<Timer> &timer) -> void
{
    int i;
    const auto maxorder = cluster->get_maxorder();

    timer->start_clock("fcs");

    if (verbosity > 0) {
        std::cout << " ================\n";
        std::cout << "  FORCE CONSTANT \n";
        std::cout << " ================\n\n";
        std::cout << "  Symmetry is handled in ";
        if (preferred_basis == "Lattice") {
            std::cout << "crystallographic (fractional) coordinates.";
        } else {
            std::cout << "Cartesian coordinates.";
        }
        std::cout << '\n';
    }

    if (fc_table) {
        deallocate(fc_table);
    }
    allocate(fc_table, maxorder);

    if (nequiv) {
        deallocate(nequiv);
    }
    allocate(nequiv, maxorder);

    if (fc_zeros) {
        deallocate(fc_zeros);
    }
    allocate(fc_zeros, maxorder);

    // Generate force constants using the information of interacting atom pairs
    for (i = 0; i < maxorder; ++i) {
        generate_force_constant_table(i,
                                      supercell.number_of_atoms,
                                      cluster->get_unique_clusters(i),
                                      symmetry,
                                      preferred_basis,
                                      fc_table[i],
                                      nequiv[i],
                                      fc_zeros[i],
                                      store_zeros);
    }

    set_basis_conversion_matrix(supercell);

    if (verbosity > 0) {
        std::cout << '\n';
        for (i = 0; i < maxorder; ++i) {
            std::cout << "  Number of " << std::setw(9) << cluster->get_ordername(i) << " FCs : " << nequiv[i].size();
            std::cout << '\n';
        }
        std::cout << '\n';


        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------\n\n";
    }

    timer->stop_clock("fcs");
}

auto Fcs::set_default_variables() -> void
{
    nequiv = nullptr;
    fc_table = nullptr;
    fc_zeros = nullptr;
    fc_cart = nullptr;
    store_zeros = true;

    preferred_basis = "Lattice";
    fc_zero_threshold = eps12;
}

auto Fcs::deallocate_variables() -> void
{
    if (nequiv) {
        deallocate(nequiv);
    }
    if (fc_table) {
        deallocate(fc_table);
    }
    if (fc_zeros) {
        deallocate(fc_zeros);
    }
    if (fc_cart) {
        deallocate(fc_cart);
    }
}


auto Fcs::generate_force_constant_table(const int order, const size_t nat, const std::set<IntList> &pairs,
                                        const std::unique_ptr<Symmetry> &symm_in, const std::string &basis,
                                        std::vector<FcProperty> &fc_vec, std::vector<size_t> &ndup,
                                        std::vector<FcProperty> &fc_zeros_out, const bool store_zeros_in) -> void
{
    size_t i;
    int *atmn, *atmn_mapped;
    int *ind, *ind_mapped;
    int *ind_tmp, *ind_mapped_tmp;

    int **xyzcomponent;

    bool *is_searched;
    std::vector<std::vector<int>> map_sym;
    std::vector<Eigen::Matrix3d> rotation;
    const bool use_compatible = true;

    if (order < 0) return;

    get_available_symmop(nat, symm_in, basis, map_sym, rotation, use_compatible);

    const auto nsym_in_use = rotation.size();

    allocate(atmn, order + 2);
    allocate(atmn_mapped, order + 2);
    allocate(ind, order + 2);
    allocate(ind_mapped, order + 2);
    allocate(ind_tmp, order);
    allocate(ind_mapped_tmp, order + 2);
    allocate(is_searched, 3 * nat);

    fc_vec.clear();
    ndup.clear();
    fc_zeros_out.clear();
    size_t nmother = 0;

    const int nxyz = static_cast<int>(std::pow(3.0, order + 2));

    allocate(xyzcomponent, nxyz, order + 2);
    get_xyzcomponent(order + 2, xyzcomponent);

    std::unordered_set<IntList> list_found;

    for (const auto &pair: pairs) {

        for (i = 0; i < order + 2; ++i) {
            atmn[i] = pair.iarray[i];
        }

        for (int i1 = 0; i1 < nxyz; ++i1) {
            for (i = 0; i < order + 2; ++i)
                ind[i] = 3 * atmn[i] + xyzcomponent[i1][i];

            if (!is_ascending(order + 2, ind)) continue;

            int i_prim = get_minimum_index_in_primitive(order + 2,
                                                        ind,
                                                        nat,
                                                        symm_in->get_nat_trueprim(),
                                                        symm_in->get_map_trueprim_to_super());
            std::swap(ind[0], ind[i_prim]);
            sort_tail(order + 2, ind);

            bool is_zero = false;

            if (list_found.find(IntList(order + 2, ind)) != list_found.end()) continue; // Already exits!

            // Search symmetrically dependent parameter set

            size_t ndeps = 0;

            for (unsigned int isym = 0; isym < nsym_in_use; ++isym) {

                for (i = 0; i < order + 2; ++i)
                    atmn_mapped[i] = map_sym[atmn[i]][isym];

                if (!is_inprim(order + 2,
                               atmn_mapped,
                               symm_in->get_nat_trueprim(),
                               symm_in->get_map_trueprim_to_super()))
                    continue;

                for (int i2 = 0; i2 < nxyz; ++i2) {

                    double const c_tmp = coef_sym(order + 2, rotation[isym], xyzcomponent[i1], xyzcomponent[i2]);

                    if (std::abs(c_tmp) > eps12) {
                        for (i = 0; i < order + 2; ++i)
                            ind_mapped[i] = 3 * atmn_mapped[i] + xyzcomponent[i2][i];

                        i_prim = get_minimum_index_in_primitive(order + 2,
                                                                ind_mapped,
                                                                nat,
                                                                symm_in->get_nat_trueprim(),
                                                                symm_in->get_map_trueprim_to_super());
                        std::swap(ind_mapped[0], ind_mapped[i_prim]);
                        sort_tail(order + 2, ind_mapped);

                        if (!is_zero) {
                            bool zeroflag = true;
                            for (i = 0; i < order + 2; ++i) {
                                zeroflag = zeroflag & (ind[i] == ind_mapped[i]);
                            }
                            zeroflag = zeroflag & (std::abs(c_tmp + 1.0) < eps8);
                            is_zero = zeroflag;
                        }

                        // Add to the found list (set) and fcset (vector) if the created is new one.

                        if (list_found.find(IntList(order + 2, ind_mapped)) == list_found.end()) {
                            list_found.insert(IntList(order + 2, ind_mapped));

                            fc_vec.emplace_back(order + 2, c_tmp, ind_mapped, nmother);
                            ++ndeps;

                            // Add equivalent interaction list (permutation) if there are two or more indices,
                            // which belong to the primitive cell.
                            // This procedure is necessary for constructing a sensing matrix.

                            for (i = 0; i < 3 * nat; ++i)
                                is_searched[i] = false;
                            is_searched[ind_mapped[0]] = true;
                            for (i = 1; i < order + 2; ++i) {
                                if ((!is_searched[ind_mapped[i]]) && is_inprim(ind_mapped[i],
                                                                               symm_in->get_nat_trueprim(),
                                                                               symm_in->get_map_trueprim_to_super()))
                                {

                                    for (size_t j = 0; j < order + 2; ++j)
                                        ind_mapped_tmp[j] = ind_mapped[j];
                                    std::swap(ind_mapped_tmp[0], ind_mapped_tmp[i]);
                                    sort_tail(order + 2, ind_mapped_tmp);
                                    fc_vec.emplace_back(order + 2, c_tmp, ind_mapped_tmp, nmother);

                                    ++ndeps;

                                    is_searched[ind_mapped[i]] = true;
                                }
                            }
                        }
                    }
                }
            } // close symmetry loop

            if (is_zero) {
                if (store_zeros_in) {
                    for (auto it = fc_vec.rbegin(); it != fc_vec.rbegin() + ndeps; ++it) {
                        it->mother = std::numeric_limits<size_t>::max();
                        fc_zeros_out.push_back(*it);
                    }
                }
                for (i = 0; i < ndeps; ++i)
                    fc_vec.pop_back();
            } else {
                ndup.push_back(ndeps);
                ++nmother;
            }

        } // close xyz component loop
    } // close atom number loop (iterator)

    deallocate(xyzcomponent);
    list_found.clear();
    deallocate(atmn);
    deallocate(atmn_mapped);
    deallocate(ind);
    deallocate(ind_mapped);
    deallocate(ind_tmp);
    deallocate(ind_mapped_tmp);
    deallocate(is_searched);
}

auto Fcs::get_constraint_symmetry(const size_t nat, const std::unique_ptr<Symmetry> &symmetry, const int order,
                                  const std::string &basis, const std::vector<FcProperty> &fc_table_in,
                                  const size_t nparams, const double tolerance, ConstraintSparseForm &const_out,
                                  const ReductionAlgo algo_in) -> void
{
    // Create constraint matrices arising from the crystal symmetry.
    // Necessary for hexagonal systems.

    int i;
    unsigned int isym;
    int ixyz;
    int *index_tmp;
    int **xyzcomponent;
    std::unordered_set<FcProperty> list_found;

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    std::vector<ConstEntry> constraint_all;
    // ConstEntry const_tmp;

    std::vector<std::vector<int>> map_sym;
    std::vector<Eigen::Matrix3d> rotation;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nfcs = fc_table_in.size();
    const auto use_compatible = false;

    if (order < 0 || nparams == 0) return;

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 2));

    get_available_symmop(nat, symmetry, basis, map_sym, rotation, use_compatible);

    const auto nsym_in_use = rotation.size();
    if (nsym_in_use == 0) return;

    const_out.clear();

    allocate(index_tmp, order + 2);
    allocate(xyzcomponent, nxyz, order + 2);
    get_xyzcomponent(order + 2, xyzcomponent);

    // Generate temporary list of parameters
    list_found.clear();
    for (const auto &p: fc_table_in) {
        for (i = 0; i < order + 2; ++i)
            index_tmp[i] = p.elems[i];
        list_found.insert(FcProperty(order + 2, p.sign, index_tmp, p.mother));
    }


#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int j;
        int i_prim;
        int loc_nonzero;
        int *ind;
        int *atm_index, *atm_index_symm;
        int *xyz_index;
        double c_tmp;
        // double maxabs;

        std::unordered_set<FcProperty>::iterator iter_found;
        std::vector<double> const_now_omp;
        std::vector<std::vector<double>> const_omp;

        ConstEntry const_tmp_omp;
        std::vector<ConstEntry> constraint_list_omp;

        allocate(ind, order + 2);
        allocate(atm_index, order + 2);
        allocate(atm_index_symm, order + 2);
        allocate(xyz_index, order + 2);

        const_omp.clear();
        const_now_omp.resize(nparams);

#ifdef _OPENMP
#pragma omp for private(i, isym, ixyz), schedule(static)
#endif
        for (long ii = 0; ii < nfcs; ++ii) {

            for (i = 0; i < order + 2; ++i) {
                atm_index[i] = fc_table_in[ii].elems[i] / 3;
                xyz_index[i] = fc_table_in[ii].elems[i] % 3;
            }

            for (isym = 0; isym < nsym_in_use; ++isym) {

                for (i = 0; i < order + 2; ++i)
                    atm_index_symm[i] = map_sym[atm_index[i]][isym];
                if (!is_inprim(order + 2, atm_index_symm, natmin, symmetry->get_map_trueprim_to_super())) continue;

                for (i = 0; i < nparams; ++i)
                    const_now_omp[i] = 0.0;

                const_now_omp[fc_table_in[ii].mother] = -fc_table_in[ii].sign;

                for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                    for (i = 0; i < order + 2; ++i)
                        ind[i] = 3 * atm_index_symm[i] + xyzcomponent[ixyz][i];

                    i_prim = get_minimum_index_in_primitive(order + 2,
                                                            ind,
                                                            nat,
                                                            natmin,
                                                            symmetry->get_map_trueprim_to_super());
                    std::swap(ind[0], ind[i_prim]);
                    sort_tail(order + 2, ind);

                    iter_found = list_found.find(FcProperty(order + 2, 1.0, ind, 1));
                    if (iter_found != list_found.end()) {
                        c_tmp = coef_sym(order + 2, rotation[isym], xyz_index, xyzcomponent[ixyz]);
                        const_now_omp[(*iter_found).mother] += (*iter_found).sign * c_tmp;
                    }
                }

                if (!is_allzero(const_now_omp, eps8, loc_nonzero)) {
                    if (const_now_omp[loc_nonzero] < 0.0) {
                        for (j = 0; j < nparams; ++j)
                            const_now_omp[j] *= -1.0;
                    }

                    const_tmp_omp.clear();
                    for (j = 0; j < nparams; ++j) {
                        if (std::abs(const_now_omp[j]) >= eps8) {
                            const_tmp_omp.emplace_back(j, const_now_omp[j]);
                        }
                    }
                    constraint_list_omp.emplace_back(const_tmp_omp);
                }

            } // close isym loop
        } // close ii loop

        deallocate(ind);
        deallocate(atm_index);
        deallocate(atm_index_symm);
        deallocate(xyz_index);

#pragma omp critical
        {
            for (const auto &it: constraint_list_omp) {
                constraint_all.emplace_back(it);
            }
        }
        constraint_list_omp.clear();
    } // close openmp region

    deallocate(xyzcomponent);
    deallocate(index_tmp);

    std::sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(), constraint_all.end()), constraint_all.end());

    MapConstraintElement const_tmp2;
    const_out.clear();

    for (const auto &it: constraint_all) {
        const_tmp2.clear();
        for (const auto &p: it) {
            const_tmp2[p.col] = p.val;
        }
        const_out.emplace_back(const_tmp2);
    }

    if (algo_in == ReductionAlgo::rref) {
        rref_sparse(nparams, const_out, eps12);
    } else if (algo_in == ReductionAlgo::qrd) {
        // verbosity 0: no verbosity in scope here, and the QR diagnostics are debug noise.
        int rank;
        auto info = get_independent_rows_lapack_sparse(nparams, const_out, 0, tolerance, rank);
    }
}

auto Fcs::get_constraint_symmetry_in_integer(const size_t nat, const std::unique_ptr<Symmetry> &symmetry,
                                             const int order, const std::string &basis,
                                             const std::vector<FcProperty> &fc_table_in, const size_t nparams,
                                             const double tolerance, ConstraintSparseForm &const_out,
                                             const ReductionAlgo algo_in) -> void
{
    // Create constraint matrices arising from the crystal symmetry.
    // Necessary for hexagonal systems.
    // This does exactly the same thing as get_constraint_symmetry but assumes
    // all elements of the constraint matrix are integer.
    // This version is expected to be more stable (and fast?).

    int i;
    unsigned int isym;
    int ixyz;
    int *index_tmp;
    int **xyzcomponent;
    int nsym_in_use;
    std::unordered_set<FcProperty> list_found;

    using ConstEntry = std::vector<ConstraintIntegerElement>;
    std::vector<ConstEntry> constraint_all;
    ConstEntry const_tmp;

    std::vector<std::vector<int>> map_sym;
    std::vector<Eigen::Matrix3d> rotation;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nfcs = fc_table_in.size();
    const auto use_compatible = false;

    if (order < 0 || nparams == 0) return;

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 2));

    get_available_symmop(nat, symmetry, basis, map_sym, rotation, use_compatible);

    nsym_in_use = rotation.size();
    if (nsym_in_use == 0) {
        return;
    }

    const_out.clear();

    allocate(index_tmp, order + 2);
    allocate(xyzcomponent, nxyz, order + 2);
    get_xyzcomponent(order + 2, xyzcomponent);

    // Generate temporary lists of parameters
    list_found.clear();
    for (const auto &p: fc_table_in) {
        for (i = 0; i < order + 2; ++i)
            index_tmp[i] = p.elems[i];
        list_found.insert(FcProperty(order + 2, p.sign, index_tmp, p.mother));
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int j;
        int i_prim;
        int loc_nonzero;
        int *ind;
        int *atm_index, *atm_index_symm;
        int *xyz_index;
        int c_tmp;

        std::unordered_set<FcProperty>::iterator iter_found;
        std::vector<int> const_now_omp;
        std::vector<std::vector<int>> const_omp;

        ConstEntry const_tmp_omp;
        std::vector<ConstEntry> constraint_list_omp;

        allocate(ind, order + 2);
        allocate(atm_index, order + 2);
        allocate(atm_index_symm, order + 2);
        allocate(xyz_index, order + 2);

        const_omp.clear();
        const_now_omp.resize(nparams);

#ifdef _OPENMP
#pragma omp for private(i, isym, ixyz), schedule(static)
#endif
        for (long ii = 0; ii < nfcs; ++ii) {

            for (i = 0; i < order + 2; ++i) {
                atm_index[i] = fc_table_in[ii].elems[i] / 3;
                xyz_index[i] = fc_table_in[ii].elems[i] % 3;
            }

            for (isym = 0; isym < nsym_in_use; ++isym) {

                for (i = 0; i < order + 2; ++i)
                    atm_index_symm[i] = map_sym[atm_index[i]][isym];
                if (!is_inprim(order + 2, atm_index_symm, natmin, symmetry->get_map_trueprim_to_super())) continue;

                for (i = 0; i < nparams; ++i)
                    const_now_omp[i] = 0;

                const_now_omp[fc_table_in[ii].mother] = -nint(fc_table_in[ii].sign);

                for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                    for (i = 0; i < order + 2; ++i)
                        ind[i] = 3 * atm_index_symm[i] + xyzcomponent[ixyz][i];

                    i_prim = get_minimum_index_in_primitive(order + 2,
                                                            ind,
                                                            nat,
                                                            natmin,
                                                            symmetry->get_map_trueprim_to_super());
                    std::swap(ind[0], ind[i_prim]);
                    sort_tail(order + 2, ind);

                    iter_found = list_found.find(FcProperty(order + 2, 1.0, ind, 1));
                    if (iter_found != list_found.end()) {
                        c_tmp = nint(coef_sym(order + 2, rotation[isym], xyz_index, xyzcomponent[ixyz]));
                        const_now_omp[iter_found->mother] += nint(iter_found->sign) * c_tmp;
                    }
                }

                if (!is_allzero(const_now_omp, loc_nonzero)) {
                    if (const_now_omp[loc_nonzero] < 0) {
                        for (j = 0; j < nparams; ++j)
                            const_now_omp[j] *= -1;
                    }

                    const_tmp_omp.clear();
                    for (j = 0; j < nparams; ++j) {
                        if (std::abs(const_now_omp[j]) > 0) {
                            const_tmp_omp.emplace_back(j, const_now_omp[j]);
                        }
                    }
                    constraint_list_omp.emplace_back(const_tmp_omp);
                }

            } // close isym loop
        } // close ii loop

        deallocate(ind);
        deallocate(atm_index);
        deallocate(atm_index_symm);
        deallocate(xyz_index);

#pragma omp critical
        {
            for (const auto &it: constraint_list_omp) {
                constraint_all.emplace_back(it);
            }
        }
        constraint_list_omp.clear();
    } // close openmp region

    deallocate(xyzcomponent);
    deallocate(index_tmp);
    boost::sort::block_indirect_sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(), constraint_all.end()), constraint_all.end());

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

    if (algo_in == ReductionAlgo::rref) {
        rref_sparse(nparams, const_out, tolerance);
    } else if (algo_in == ReductionAlgo::qrd) {
        // verbosity 0: no verbosity in scope here, and the QR diagnostics are debug noise.
        int rank;
        auto info = get_independent_rows_lapack_sparse(nparams, const_out, 0, tolerance, rank);
    }
}

auto Fcs::get_nequiv() const -> std::vector<size_t> *
{
    return nequiv;
}

auto Fcs::get_fc_table() const -> std::vector<FcProperty> *
{
    return fc_table;
}

auto Fcs::get_fc_cart() const -> std::vector<ForceConstantTable> *
{
    return fc_cart;
}

auto Fcs::set_forceconstant_basis(const std::string &preferred_basis_in) -> void
{
    if (preferred_basis_in[0] == 'c') {
        preferred_basis = "Cartesian";
    } else if (preferred_basis_in[0] == 'l') {
        preferred_basis = "Lattice";
    } else {
        preferred_basis = preferred_basis_in;
    }
}

auto Fcs::get_forceconstant_basis() const -> std::string
{
    return preferred_basis;
}

auto Fcs::set_forceconstant_cartesian(const int maxorder, double *param_in) -> void
{
    auto ishift = 0;
    int j;

    std::vector<int> atoms_old, atoms_now;
    std::vector<int> coords_now;
    std::vector<std::vector<int>> coord_list;
    std::vector<double> fc_list;
    std::vector<std::vector<std::vector<int>>> coord_list_grp;
    std::vector<std::vector<double>> fc_list_grp;
    std::vector<std::vector<int>> atoms_grp;
    std::vector<FcProperty> fc_table_copy;

    if (fc_cart) {
        deallocate(fc_cart);
    }
    allocate(fc_cart, maxorder);

    nfc_cart_permu.resize(maxorder);
    nfc_cart_nopermu.resize(maxorder);

    if (preferred_basis == "Lattice") {

        std::vector<ForceConstantTable> fc_cart_tmp, fc_latt;

        for (int i = 0; i < maxorder; ++i) {

            fc_latt.clear();
            fc_cart_tmp.clear();

            for (const auto &it: fc_table[i]) {
                fc_latt.emplace_back(param_in[it.mother + ishift] * it.sign, it.elems);
            }

            change_basis_force_constants(fc_latt, fc_cart_tmp, 0);
            for (const auto &it: fc_cart_tmp) {
                if (std::abs(it.fc_value) >= fc_zero_threshold) {
                    fc_cart[i].emplace_back(it);
                }
            }

            ishift += nequiv[i].size();

            nfc_cart_permu[i] = fc_cart[i].size();
            nfc_cart_nopermu[i] = std::count_if(fc_cart[i].begin(),
                                                fc_cart[i].end(),
                                                [](const ForceConstantTable &obj) { return obj.is_ascending_order; });
        }

    } else if (preferred_basis == "Cartesian") {

        for (int i = 0; i < maxorder; ++i) {

            auto nelems = i + 2;
            atoms_now.resize(nelems);
            coords_now.resize(nelems);

            for (const auto &it: fc_table[i]) {
                if (std::abs(param_in[it.mother + ishift]) >= fc_zero_threshold) {
                    std::vector<int> elems_permutation = it.elems;
                    do {
                        for (j = 0; j < nelems; ++j) {
                            atoms_now[j] = elems_permutation[j] / 3;
                            coords_now[j] = elems_permutation[j] % 3;
                        }
                        fc_cart[i].emplace_back(nelems,
                                                param_in[it.mother + ishift] * it.sign,
                                                &atoms_now[0],
                                                &coords_now[0]);
                    } while (std::next_permutation(elems_permutation.begin() + 1, elems_permutation.end()));
                }
            }

            ishift += nequiv[i].size();

            nfc_cart_permu[i] = fc_cart[i].size();
            nfc_cart_nopermu[i] = std::count_if(fc_cart[i].begin(),
                                                fc_cart[i].end(),
                                                [](const ForceConstantTable &obj) { return obj.is_ascending_order; });
        }
    }
}

auto Fcs::change_basis_force_constants(const std::vector<ForceConstantTable> &fc_in,
                                       std::vector<ForceConstantTable> &fc_out,
                                       const int conversion_direction) const -> void
{
    // Convert the basis of force constants between Cartesian basis and Lattice basis.
    // When conversion_direction == 0: fc_in is in Lattice and fc_out is in Cartesian
    // when conversion_direction == 1: fc_in is in Cartesian and fc_out is in Lattice
    //
    // Only the permutationally unique pairs of 2nd to n-th elements (flattened array)
    // should be included in fc_in because the permutation operation is performed
    // inside this function. For example, (i, j, k) and (i, k, j) must not be included
    // simultaneously (j != k) for the 3rd-order IFCs.
    // This limitation applies only to the 2nd to n-th indices (not to the 1st index).

    // This operation takes a while for higher-order anharmonic terms.
    // TODO: Improve performance

    if (fc_in.empty()) return;

    const auto nelems = fc_in[0].flattenarray.size();

    std::vector<int> atoms_old, atoms_now;
    std::vector<int> coords_now;
    std::vector<std::vector<int>> coord_list;
    std::vector<double> fc_list;
    std::vector<std::vector<std::vector<int>>> coord_list_grp;
    std::vector<std::vector<double>> fc_list_grp;
    std::vector<std::vector<int>> atoms_grp;
    std::vector<ForceConstantTable> fc_table_copy;

    int **xyzcomponent;

    atoms_old.resize(nelems);
    atoms_now.resize(nelems);
    coords_now.resize(nelems);
    coord_list.clear();
    fc_list.clear();

    const auto nxyz = static_cast<int>(std::pow(3.0, nelems));
    allocate(xyzcomponent, nxyz, nelems);
    get_xyzcomponent(nelems, xyzcomponent);

    for (auto j = 0; j < nelems; ++j)
        atoms_old[j] = -1;
    auto icount = 0;

    fc_table_copy.clear();

    // Recover all permutationally equivalent terms
    for (const auto &it: fc_in) {
        std::vector<int> elems_permutation = it.flattenarray;
        do {
            fc_table_copy.emplace_back(it.fc_value, elems_permutation);
        } while (std::next_permutation(elems_permutation.begin() + 1, elems_permutation.end()));
    }

    // Sort fc_table_copy in ascending order of atomic indices.
    std::sort(fc_table_copy.begin(), fc_table_copy.end(), ForceConstantTable::compare_atom_index);

    coord_list_grp.clear();
    fc_list_grp.clear();
    atoms_grp.clear();

    Eigen::Matrix3d convmat;

    if (conversion_direction == 0) {
        // From Lattice basis to Cartesian
        convmat = basis_conversion_matrix;
    } else {
        // From Cartesian basis to Lattice
        convmat = basis_conversion_matrix.inverse();
    }

    for (const auto &it: fc_table_copy) {

        atoms_now = it.atoms;
        coords_now = it.coords;

        if (atoms_now == atoms_old) {

            coord_list.emplace_back(coords_now);
            fc_list.emplace_back(it.fc_value);
            ++icount;

        } else {

            if (icount > 0) {
                coord_list_grp.emplace_back(coord_list);
                fc_list_grp.emplace_back(fc_list);
                atoms_grp.emplace_back(atoms_old);
            }

            atoms_old = atoms_now;
            coord_list.clear();
            fc_list.clear();
            coord_list.push_back(coords_now);
            fc_list.push_back(it.fc_value);
            icount = 1;
        }
    }

    if (icount > 0) {
        coord_list_grp.emplace_back(coord_list);
        fc_list_grp.emplace_back(fc_list);
        atoms_grp.emplace_back(atoms_old);
    }

    const auto ngrp = coord_list_grp.size();

#pragma omp parallel
    {
        std::vector<ForceConstantTable> fc_omp;

#pragma omp for nowait
        for (int igrp = 0; igrp < ngrp; ++igrp) {
            for (auto ixyz = 0; ixyz < nxyz; ++ixyz) {

                auto fcs_tmp = 0.0;
                const auto nentry = coord_list_grp[igrp].size();
                for (auto j = 0; j < nentry; ++j) {
                    double prod_matrix = 1.0;
                    for (auto k = 0; k < nelems; ++k) {
                        prod_matrix *= convmat(coord_list_grp[igrp][j][k], xyzcomponent[ixyz][k]);
                    }
                    fcs_tmp += prod_matrix * fc_list_grp[igrp][j];
                }

                fc_omp.emplace_back(nelems, fcs_tmp, &atoms_grp[igrp][0], xyzcomponent[ixyz]);
            }
        }

#pragma omp critical
        {
            for (const auto &it: fc_omp) {
                fc_out.emplace_back(it);
            }
        }
        fc_omp.clear();
    }

    deallocate(xyzcomponent);
}

auto Fcs::set_fc_zero_threshold(const double threshold_in) -> void
{
    fc_zero_threshold = threshold_in;
}

auto Fcs::get_fc_zero_threshold() const -> double
{
    return fc_zero_threshold;
}

auto Fcs::get_nfc_cart(const int permutation) const -> std::vector<size_t>
{
    if (permutation) {
        return nfc_cart_permu;
    } else {
        return nfc_cart_nopermu;
    }
}

auto Fcs::get_available_symmop(const size_t nat, const std::unique_ptr<Symmetry> &symmetry, const std::string &basis,
                               std::vector<std::vector<int>> &mapping_symm, std::vector<Eigen::Matrix3d> &rotation,
                               const bool use_compatible) -> void
{
    // Return mapping information of atoms and the rotation matrices of symmetry operations
    // that are (compatible, incompatible) with the given lattice basis (Cartesian or Lattice).

    // use_compatible == true returns the compatible space group (for creating fc_table)
    // use_compatible == false returnes the incompatible supace group (for creating constraint)

    int counter = 0;

    mapping_symm.clear();
    rotation.clear();

    std::vector<int> isym_to_add;

    if (basis == "Cartesian") {

        for (const auto &it: symmetry->get_symmetry_data()) {
            if (it.compatible_with_cartesian == use_compatible) {
                rotation.emplace_back(it.rotation_cart);
                isym_to_add.emplace_back(counter);
            }
            ++counter;
        }

    } else if (basis == "Lattice") {

        for (const auto &it: symmetry->get_symmetry_data()) {
            if (it.compatible_with_lattice == use_compatible) {
                rotation.emplace_back(it.rotation.unaryExpr([](const int x) { return static_cast<double>(x); }));
                isym_to_add.emplace_back(counter);
            }
            ++counter;
        }

    } else {
        exit("get_available_symmop", "Invalid basis input");
    }

    const auto nsym_to_use = isym_to_add.size();

    mapping_symm.resize(nat, std::vector<int>(nsym_to_use));

    for (auto i = 0; i < nsym_to_use; ++i) {
        for (auto j = 0; j < nat; ++j) {
            mapping_symm[j][i] = symmetry->get_map_sym()[j][isym_to_add[i]];
        }
    }
}

auto Fcs::coef_sym(const int n, const Eigen::Matrix3d &rot, const int *arr1, const int *arr2) -> double
{
    auto tmp = 1.0;

    for (auto i = 0; i < n; ++i) {
        tmp *= rot(arr2[i], arr1[i]);
    }
    return tmp;
}

auto Fcs::is_ascending(const int n, const int *arr) -> bool
{
    for (auto i = 0; i < n - 1; ++i) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

auto Fcs::get_minimum_index_in_primitive(const int n, const int *arr, const size_t nat, const size_t natmin,
                                         const std::vector<std::vector<int>> &map_p2s) -> int
{
    int i;

    std::vector ind(n, 3 * nat);

    for (i = 0; i < n; ++i) {

        int atmnum = arr[i] / 3;

        for (size_t j = 0; j < natmin; ++j) {
            if (map_p2s[j][0] == atmnum) {
                ind[i] = arr[i];
            }
        }
    }

    auto minval = ind[0];
    auto minloc = 0;

    for (i = 0; i < n; ++i) {
        if (ind[i] < minval) {
            minval = ind[i];
            minloc = i;
        }
    }

    return minloc;
}

auto Fcs::is_inprim(const int n, const int *arr, const size_t natmin,
                    const std::vector<std::vector<int>> &map_p2s) -> bool
{
    for (auto i = 0; i < n; ++i) {
        for (size_t j = 0; j < natmin; ++j) {
            if (map_p2s[j][0] == arr[i]) return true;
        }
    }
    return false;
}

auto Fcs::is_inprim(const int n, const size_t natmin, const std::vector<std::vector<int>> &map_p2s) -> bool
{
    const auto atmn = n / 3;

    for (size_t i = 0; i < natmin; ++i) {
        if (map_p2s[i][0] == atmn) return true;
    }

    return false;
}

auto Fcs::get_xyzcomponent(const int n, int **xyz) -> void
{
    // Return xyz component for the given order using boost algorithm library

    int i;

    std::vector<int> v(3 * n);

    for (i = 0; i < n; ++i)
        v[i] = 0;
    for (i = n; i < 2 * n; ++i)
        v[i] = 1;
    for (i = 2 * n; i < 3 * n; ++i)
        v[i] = 2;

    auto m = 0;

    do {
        xyz[m][0] = v[0];
        for (i = 1; i < n; ++i)
            xyz[m][i] = v[i];
        ++m;
    } while (boost::next_partial_permutation(v.begin(), v.begin() + n, v.end()));
}

auto Fcs::is_allzero(const std::vector<double> &vec, const double tol, int &loc) -> bool
{
    loc = -1;
    const auto n = vec.size();
    for (auto i = 0; i < n; ++i) {
        if (std::abs(vec[i]) > tol) {
            loc = i;
            return false;
        }
    }
    return true;
}

auto Fcs::is_allzero(const std::vector<int> &vec, int &loc) -> bool
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

auto Fcs::get_basis_conversion_matrix() const -> Eigen::Matrix3d
{
    return basis_conversion_matrix;
}

auto Fcs::set_basis_conversion_matrix(const Cell &supercell) -> void
{
    if (preferred_basis == "Lattice") {
        // multiply the scale factor for making the determinant of the basis_conversion_matrix
        // as one.
        const auto scale_factor = std::pow(supercell.volume, 1.0 / 3.0) / (2.0 * pi);
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                basis_conversion_matrix(i, j) = supercell.reciprocal_lattice_vector(i, j) * scale_factor;
            }
        }
    } else if (preferred_basis == "Cartesian") {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                if (i == j) {
                    basis_conversion_matrix(i, j) = 1.0;
                } else {
                    basis_conversion_matrix(i, j) = 0.0;
                }
            }
        }
    }
}

auto Fcs::translate_forceconstant_index_to_centercell(const std::unique_ptr<Symmetry> &symmetry,
                                                      std::vector<std::vector<int>> &index_inout) -> void
{
    const auto map_sym = symmetry->get_map_sym();
    const auto map_p2s = symmetry->get_map_trueprim_to_super();
    const auto symnum_tran = symmetry->get_symnum_tran();
    const auto nfcs = index_inout.size();
    const auto nelems = index_inout[0].size();
    const auto natmin = symmetry->get_nat_trueprim();
    std::vector<int> atoms, coords;
    std::vector<int> atoms_tran;
    std::vector<int> index_tran;

    atoms.resize(nelems);
    coords.resize(nelems);
    atoms_tran.resize(nelems);
    index_tran.resize(nelems);

    for (size_t i = 0; i < nfcs; ++i) {
        for (size_t j = 0; j < nelems; ++j) {
            atoms[j] = index_inout[i][j] / 3;
            coords[j] = index_inout[i][j] % 3;
        }
        int isym = -1;
        if (!is_inprim(index_inout[i][0], natmin, map_p2s)) {
            for (auto jsym = 0; jsym < symnum_tran.size(); ++jsym) {
                index_tran[0] = 3 * map_sym[atoms[0]][symnum_tran[jsym]] + coords[0];
                if (is_inprim(index_tran[0], natmin, map_p2s)) {
                    isym = jsym;
                    break;
                }
            }
            if (isym == -1) {
                exit("translate_forceconstant_index_to_centercell", "translational mapping to the center cell failed.");
            }

            for (size_t j = 0; j < nelems; ++j) {
                index_inout[i][j] = 3 * map_sym[atoms[j]][symnum_tran[isym]] + coords[j];
            }
        }
    }
}
