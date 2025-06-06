/*
cluster.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "cluster.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
#include "combination.h"
#include "error.h"
#include "memory.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"

using namespace ALM_NS;

Cluster::Cluster()
{
    set_default_variables();
}

Cluster::~Cluster()
{
    deallocate_variables();
}

void Cluster::init(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                   const int periodic_image_conv, const int verbosity, std::unique_ptr<Timer> &timer)
{
    timer->start_clock("cluster");

    int i;
    size_t j, k;
    const auto nat = system->get_supercell().number_of_atoms;
    const auto nkd = system->get_supercell().number_of_elems;

    if (verbosity > 0) {
        std::cout << " =============\n";
        std::cout << "  INTERACTION \n";
        std::cout << " =============\n\n";
    }

    // Default values of cutoof_radii and nbody_include
    if (!cutoff_radii) {
        allocate(cutoff_radii, maxorder, nkd, nkd);
        for (i = 0; i < maxorder; ++i) {
            for (j = 0; j < nkd; ++j) {
                for (k = 0; k < nkd; ++k) {
                    cutoff_radii[i][j][k] = -1.0;
                }
            }
        }
    }
    if (!nbody_include) {
        allocate(nbody_include, maxorder);
        for (i = 0; i < maxorder; ++i) {
            nbody_include[i] = i + 2;
        }
    }

    get_pairs_of_minimum_distance(nat, system->get_x_image(), system->get_exist_image(), distance_table);

    set_interaction_by_cutoff(system->get_supercell().number_of_atoms,
                              system->get_supercell().kind,
                              symmetry->get_nat_trueprim(),
                              symmetry->get_map_trueprim_to_super(),
                              atoms_in_cutoff);

    interaction_cluster.clear();
    interaction_cluster.resize(maxorder, std::vector<std::set<InteractionCluster>>(symmetry->get_nat_trueprim()));

    calc_interaction_clusters(symmetry->get_nat_trueprim(),
                              system->get_supercell().kind,
                              symmetry->get_map_trueprim_to_super(),
                              system->get_x_image(),
                              system->get_exist_image(),
                              periodic_image_conv);

    unique_clusters.clear();
    unique_clusters.resize(maxorder);

    generate_unique_clusters(symmetry->get_nat_trueprim(), symmetry->get_map_trueprim_to_super(), unique_clusters);

    // check permutation symmetry of anharmonic IFC
    if (periodic_image_conv > 0) {
        std::cout << "check permutation symmetry of the clusters.\n";
        std::cout << "(if no message is printed out, the permutation symmetry is satisfied.)\n";
        for (auto order = 1; order < maxorder; ++order) {
            if (order == 1) {
                std::cout << "  CUBIC ...\n";
            } else if (order == 2) {
                std::cout << "  QUARTIC ...\n";
            } else {
                std::cout << "" << order + 2 << "-th ORDER ...\n";
            }
            check_permutation_symmetry(system, symmetry, order);
        }
        std::cout << "done.\n\n";
    }


    if (verbosity > 0) {
        std::cout << "  +++ Cutoff Radii Matrix (NKD x NKD matrix) +++\n";

        for (i = 0; i < maxorder; ++i) {
            std::cout << "  " << std::setw(9) << get_ordername(i) << '\n';
            for (j = 0; j < nkd; ++j) {
                for (k = 0; k < nkd; ++k) {
                    if (cutoff_radii[i][j][k] < 0.0) {
                        std::cout << std::setw(9) << "None";
                    } else {
                        std::cout << std::setw(9) << cutoff_radii[i][j][k];
                    }
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        print_neighborlist(system->get_supercell().number_of_atoms,
                           symmetry->get_nat_trueprim(),
                           symmetry->get_map_trueprim_to_super(),
                           system->get_supercell().kind,
                           system->get_kdname());
    }

    if (verbosity > 1) {
        print_interaction_information(symmetry->get_nat_trueprim(),
                                      symmetry->get_map_trueprim_to_super(),
                                      system->get_supercell().kind,
                                      system->get_kdname(),
                                      atoms_in_cutoff);
    }

    if (verbosity > 0) {
        for (i = 0; i < maxorder; ++i) {
            if (i + 2 > nbody_include[i]) {
                std::cout << "  For " << std::setw(8) << get_ordername(i) << ", ";
                std::cout << "interactions related to more than" << std::setw(2) << nbody_include[i];
                std::cout << " atoms will be neglected.\n";
            }
        }
        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------\n\n";
    }

    timer->stop_clock("cluster");
}

void Cluster::generate_unique_clusters(const size_t natmin, const std::vector<std::vector<int>> &map_p2s,
                                       std::vector<std::set<IntList>> &cluster_out) const
{
    int *pair_tmp;

    for (auto order = 0; order < maxorder; ++order) {

        cluster_out[order].clear();

        allocate(pair_tmp, order + 2);

        for (size_t i = 0; i < natmin; ++i) {

            const auto iat = map_p2s[i][0];

            for (const auto &it: interaction_cluster[order][i]) {

                pair_tmp[0] = iat;
                for (auto j = 0; j < order + 1; ++j) {
                    pair_tmp[j + 1] = it.atom[j];
                }
                insort(order + 2, pair_tmp);

                // Ignore many-body case
                // if (!satisfy_nbody_rule(order + 2, pair_tmp, order)) continue;
                cluster_out[order].insert(IntList(order + 2, pair_tmp));
            }
        }
        deallocate(pair_tmp);
    }
}

void Cluster::check_permutation_symmetry(const std::unique_ptr<System> &system,
                                         const std::unique_ptr<Symmetry> &symmetry, int order)
{
    const auto nat = system->get_supercell().number_of_atoms;
    int natmin = symmetry->get_nat_trueprim();

    int isym_tran, isym;
    int j, j2;
    int iat, jat, j2at;
    int iat_translated, j2at_translated;
    int iat_prim;
    int jat_prim;
    int i_periodic;
    std::vector<int> data_now;

    std::vector<class RelativeVectors> relvecs1(order), relvecs2(order);

    int is_found;
    std::vector<int> is_checked;

    int itmp, jtmp, xyztmp, itmp2;
    double dtmp1;
    std::vector<double> relvec_tmp;

    // find translation symmetry to bring the atom to the primitive cell
    std::vector<int> symnum_tran_to_prim(nat);
    make_symnum_tran_to_prim(system, symmetry, symnum_tran_to_prim);

    // check permutation symmetry
    int i_ifc = 0;
    for (iat_prim = 0; iat_prim < natmin; iat_prim++) {
        for (auto &cluster_tmp: interaction_cluster[order][iat_prim]) {

            i_ifc++;

            for (j = 0; j < order + 1; j++) {

                // bring j-th atom of the cluster to the primitive cell
                jat = cluster_tmp.atom[j];
                isym_tran = symnum_tran_to_prim[jat];
                isym = symmetry->get_symnum_tran()[isym_tran];

                jat_prim = symmetry->get_map_super_to_trueprim()[jat].atom_num;

                data_now.clear();
                // original center
                iat = symmetry->get_map_trueprim_to_super()[iat_prim][0];
                iat_translated = symmetry->get_map_sym()[iat][isym];
                data_now.push_back(iat_translated);

                // other atoms
                for (j2 = 0; j2 < order + 1; j2++) {
                    j2at = cluster_tmp.atom[j2];
                    j2at_translated = symmetry->get_map_sym()[j2at][isym];

                    if (j2 == j) {
                        continue;
                    }
                    data_now.push_back(j2at_translated);
                }
                std::sort(data_now.begin(), data_now.end());

                // search for the corresponding cluster
                auto cluster_tmp2 = interaction_cluster[order][jat_prim].find(InteractionCluster(data_now));
                if (cluster_tmp2 == interaction_cluster[order][jat_prim].end()) {
                    std::cout << "permutation symmetry is NOT satisfied: ";
                    std::cout << "corresponding cluster does not exist\n";

                    std::cout << "information on current cluster: \n";
                    std::cout << "center atom(in primitive cell) = " << iat_prim;
                    std::cout << "other atoms = ";
                    for (itmp2 = 0; itmp2 < order + 1; itmp2++) {
                        std::cout << cluster_tmp.atom[itmp2];
                        if (itmp2 < order) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "permutation of center atom and atom " << jat << " is considered.\n";
                    continue;
                }

                // prepare relative vector for comparison
                relvecs1.clear();
                for (i_periodic = 0; i_periodic < cluster_tmp.cell.size(); i_periodic++) {
                    relvecs1.emplace_back(order);
                    // first relative vector
                    relvec_tmp.clear();
                    for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                        dtmp1 = system->get_x_image()[0](iat, xyztmp) -
                                system->get_x_image()[cluster_tmp.cell[i_periodic][j]](jat, xyztmp);
                        relvec_tmp.push_back(dtmp1);
                    }
                    relvecs1[i_periodic].relvecs_cartesian.push_back(relvec_tmp);

                    // other relative vectors
                    for (j2 = 0; j2 < order + 1; j2++) {
                        if (j2 == j) {
                            continue;
                        }
                        relvec_tmp.clear();
                        for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                            j2at = cluster_tmp.atom[j2];
                            dtmp1 = system->get_x_image()[cluster_tmp.cell[i_periodic][j2]](j2at, xyztmp) -
                                    system->get_x_image()[cluster_tmp.cell[i_periodic][j]](jat, xyztmp);
                            relvec_tmp.push_back(dtmp1);
                        }
                        relvecs1[i_periodic].relvecs_cartesian.push_back(relvec_tmp);
                    }

                    relvecs1[i_periodic].make_fractional_from_cartesian(
                        system->get_supercell().reciprocal_lattice_vector);
                }

                relvecs2.clear();
                for (i_periodic = 0; i_periodic < (*cluster_tmp2).cell.size(); i_periodic++) {
                    relvecs2.emplace_back(order);

                    for (j2 = 0; j2 < order + 1; j2++) {
                        relvec_tmp.clear();
                        for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                            j2at = (*cluster_tmp2).atom[j2];
                            dtmp1 =
                                system->get_x_image()[(*cluster_tmp2).cell[i_periodic][j2]](j2at, xyztmp) -
                                system->get_x_image()[0](symmetry->get_map_trueprim_to_super()[jat_prim][0], xyztmp);
                            relvec_tmp.push_back(dtmp1);
                        }
                        relvecs2[i_periodic].relvecs_cartesian.push_back(relvec_tmp);
                    }

                    relvecs2[i_periodic].make_fractional_from_cartesian(
                        system->get_supercell().reciprocal_lattice_vector);
                }

                if (relvecs1.size() != relvecs2.size()) {
                    std::cout << "permutation symmetry is NOT satisfied: ";
                    std::cout << "multiplicity of periodic image is different\n";


                    std::cout << "information on current cluster: \n";
                    std::cout << "center atom(in primitive cell) = " << iat_prim;
                    std::cout << "other atoms = ";
                    for (itmp2 = 0; itmp2 < order + 1; itmp2++) {
                        std::cout << cluster_tmp.atom[itmp2];
                        if (itmp2 < order) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << "permutation of center atom and atom " << jat << " is considered.\n";

                    std::cout << "multiplicity in original cluster = " << cluster_tmp.cell.size() << '\n';
                    std::cout << "multiplicity in permuted cluster = " << (*cluster_tmp2).cell.size() << '\n';

                    // print relative vectors
                    std::cout << "relative vectors in current cluster (fractional, cartesian): \n";
                    for (i_periodic = 0; i_periodic < cluster_tmp.cell.size(); i_periodic++) {
                        std::cout << "periodic image pattern: " << i_periodic << '\n';
                        for (itmp = 0; itmp < order + 1; itmp++) {
                            for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                std::cout << relvecs1[i_periodic].relvecs_fractional[itmp][xyztmp] << " ";
                            }
                            std::cout << ", ";
                            for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                std::cout << relvecs1[i_periodic].relvecs_cartesian[itmp][xyztmp] << " ";
                            }
                            std::cout << '\n';
                        }
                    }

                    std::cout << "relative vectors in corresponding cluster (fractional, cartesian): \n";
                    for (i_periodic = 0; i_periodic < (*cluster_tmp2).cell.size(); i_periodic++) {
                        std::cout << "periodic image pattern: " << i_periodic << '\n';
                        for (itmp = 0; itmp < order + 1; itmp++) {
                            for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                std::cout << relvecs2[i_periodic].relvecs_fractional[itmp][xyztmp] << " ";
                            }
                            std::cout << ", ";
                            for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                std::cout << relvecs2[i_periodic].relvecs_cartesian[itmp][xyztmp] << " ";
                            }
                            std::cout << '\n';
                        }
                    }
                    continue;
                }


                // compare relative vector
                is_checked.clear();
                is_checked.resize(relvecs1.size());

                for (itmp = 0; itmp < relvecs1.size(); itmp++) {
                    is_found = 0;
                    for (jtmp = 0; jtmp < relvecs2.size(); jtmp++) {
                        if (is_checked[jtmp] == 1) {
                            continue;
                        }
                        if (relvecs1[itmp].is_equal(relvecs2[jtmp], eps6)) {
                            is_found = 1;
                            is_checked[jtmp] = 1;
                            break;
                        }
                    }
                    if (is_found == 0) {
                        std::cout << "permutation symmetry is not satisfied: ";
                        std::cout << "periodic image is different.\n";

                        std::cout << "information on current cluster: \n";
                        std::cout << "center atom(in primitive cell) = " << iat_prim;
                        std::cout << "other atoms = ";
                        for (itmp2 = 0; itmp2 < order + 1; itmp2++) {
                            std::cout << cluster_tmp.atom[itmp2];
                            if (itmp2 < order) {
                                std::cout << ", ";
                            }
                        }
                        std::cout << "permutation of center atom and atom " << jat << " is considered.\n";

                        std::cout << "multiplicity in original cluster = " << cluster_tmp.cell.size() << '\n';
                        std::cout << "multiplicity in permuted cluster = " << (*cluster_tmp2).cell.size() << '\n';

                        // print relative vectors
                        std::cout << "relative vectors in current cluster (fractional, cartesian): \n";
                        for (i_periodic = 0; i_periodic < cluster_tmp.cell.size(); i_periodic++) {
                            std::cout << "periodic image pattern: " << i_periodic << '\n';
                            for (itmp = 0; itmp < order + 1; itmp++) {
                                for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                    std::cout << relvecs1[i_periodic].relvecs_fractional[itmp][xyztmp] << " ";
                                }
                                std::cout << ", ";
                                for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                    std::cout << relvecs1[i_periodic].relvecs_cartesian[itmp][xyztmp] << " ";
                                }
                                std::cout << '\n';
                            }
                        }

                        std::cout << "relative vectors in corresponding cluster (fractional, cartesian): \n";
                        for (i_periodic = 0; i_periodic < (*cluster_tmp2).cell.size(); i_periodic++) {
                            std::cout << "periodic image pattern: " << i_periodic << '\n';
                            for (itmp = 0; itmp < order + 1; itmp++) {
                                for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                    std::cout << relvecs2[i_periodic].relvecs_fractional[itmp][xyztmp] << " ";
                                }
                                std::cout << ", ";
                                for (xyztmp = 0; xyztmp < 3; xyztmp++) {
                                    std::cout << relvecs2[i_periodic].relvecs_cartesian[itmp][xyztmp] << " ";
                                }
                                std::cout << '\n';
                            }
                        }
                        std::cout << "corresponding periodic image does not exist for " << itmp
                                  << "-th periodic image in current cluster\n";
                    } else {
                        // std::cout << "corresponding periodic image is found for " << itmp << "-th periodic image in current cluster\n";
                    }
                }
            }
        }
    }
}

void Cluster::make_symnum_tran_to_prim(const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
                                       std::vector<int> &symnum_tran_to_prim)
{

    const auto nat = system->get_supercell().number_of_atoms;
    int natmin = symmetry->get_nat_trueprim();
    int isym_tran, isym;
    int iat, jat;

    for (isym_tran = 0; isym_tran < symmetry->get_symnum_tran().size(); isym_tran++) {

        isym = symmetry->get_symnum_tran()[isym_tran];

        for (iat = 0; iat < nat; iat++) {
            jat = symmetry->get_map_sym()[iat][isym];

            // if jat is in the primitive cell
            if (is_inprim(jat, natmin, symmetry->get_map_trueprim_to_super())) {
                symnum_tran_to_prim[iat] = isym_tran;
            }
        }
    }
}

bool Cluster::is_inprim(const int iat, // atom index in supercell
                        const size_t natmin, const std::vector<std::vector<int>> &map_p2s) const
{
    for (size_t i = 0; i < natmin; ++i) {
        if (map_p2s[i][0] == iat) return true;
    }
    return false;
}

void Cluster::set_default_variables()
{
    maxorder = 0;
    nbody_include = nullptr;
    cutoff_radii = nullptr;
}

void Cluster::deallocate_variables()
{
    if (nbody_include) {
        deallocate(nbody_include);
        nbody_include = nullptr;
    }
    if (cutoff_radii) {
        deallocate(cutoff_radii);
        cutoff_radii = nullptr;
    }
    unique_clusters.clear();
    unique_clusters.shrink_to_fit();
    distance_table.clear();
    distance_table.shrink_to_fit();
    atoms_in_cutoff.clear();
    atoms_in_cutoff.shrink_to_fit();
    interaction_cluster.clear();
    interaction_cluster.shrink_to_fit();
}

double Cluster::distance(const Eigen::MatrixXd &x1, const Eigen::MatrixXd &x2) const
{
    return (x1 - x2).norm();
}

void Cluster::get_pairs_of_minimum_distance(const size_t nat, const std::vector<Eigen::MatrixXd> &xc_in,
                                            const int *exist,
                                            std::vector<std::vector<PairDistances>> &dist_test_out) const
{
    size_t i, j;
    int ij;
    const auto nat2 = nat * nat;

    dist_test_out.resize(nat, std::vector<PairDistances>(nat));

#pragma omp parallel
    {
        std::vector<int> cells;
        std::vector<double> distances;
        std::vector<Eigen::Vector3d> relative_vectors;

#pragma omp for private(i, j)
        for (ij = 0; ij < nat2; ++ij) {
            i = ij / nat;
            j = ij % nat;
            cells.clear();
            distances.clear();
            relative_vectors.clear();

            for (auto icell = 0; icell < 27; ++icell) {
                if (exist[icell]) {
                    cells.emplace_back(icell);
                    distances.emplace_back(distance(xc_in[0].row(i), xc_in[icell].row(j)));
                    relative_vectors.emplace_back(xc_in[icell].row(j) - xc_in[0].row(i));
                }
            }
            dist_test_out[i][j] = PairDistances(cells, distances, relative_vectors);
        }
    }
}

void Cluster::print_neighborlist(const size_t nat, const size_t natmin, const std::vector<std::vector<int>> &map_p2s,
                                 const std::vector<int> &kd, const std::vector<std::string> &kdname) const
{
    //
    // Print the list of neighboring atoms and distances
    //
    size_t i, j, k;
    int iat;
    int icount;

    std::vector<std::vector<double>> distances;
    std::vector<std::vector<int>> atoms;

    std::vector<double> dists_tmp;
    std::vector<int> atoms_tmp;
    std::vector<int> indices(nat);

    distances.resize(natmin, std::vector<double>(nat));
    atoms.resize(natmin, std::vector<int>(nat));

    const double dist_tol = distance_table[0][0].tol_distance;

    for (i = 0; i < natmin; ++i) {
        iat = map_p2s[i][0];
        dists_tmp.clear();
        atoms_tmp.clear();

        std::iota(indices.begin(), indices.end(), 0);

        for (j = 0; j < nat; ++j) {
            atoms_tmp.emplace_back(j);
            dists_tmp.emplace_back(distance_table[iat][j].distances[0]);
        }
        std::sort(indices.begin(), indices.end(), [&dists_tmp, &atoms_tmp, &dist_tol](int left, int right) -> bool {
            if (std::abs(dists_tmp[left] - dists_tmp[right]) > dist_tol) {
                return dists_tmp[left] < dists_tmp[right];
            }
            return atoms_tmp[left] < atoms_tmp[right];
        });
        for (j = 0; j < nat; ++j) {
            distances[i][j] = dists_tmp[indices[j]];
            atoms[i][j] = atoms_tmp[indices[j]];
        }
    }

    std::cout << '\n';
    std::cout << "  List of neighboring atoms below.\n";
    std::cout << "  Format [N th-nearest shell, distance (Number of atoms on the shell)]\n\n";

    std::vector<int> atomlist;

    for (i = 0; i < natmin; ++i) {

        auto nthnearest = 0;
        atomlist.clear();

        iat = map_p2s[i][0];
        std::cout << std::setw(5) << iat + 1 << " (" << std::setw(3) << kdname[kd[iat] - 1] << "): ";

        auto dist_tmp = 0.0;

        for (j = 0; j < nat; ++j) {

            if (distances[i][j] < eps8) continue; // distance is zero

            if (std::abs(distances[i][j] - dist_tmp) > 1.0e-3) {

                if (!atomlist.empty()) {
                    nthnearest += 1;

                    if (nthnearest > 1) std::cout << std::setw(13) << " ";

                    std::cout << std::setw(3) << nthnearest << std::setw(10) << dist_tmp << " (" << std::setw(3)
                              << atomlist.size() << ") -";

                    icount = 0;
                    for (k = 0; k < atomlist.size(); ++k) {

                        if (icount % 4 == 0 && icount > 0) {
                            std::cout << '\n';
                            std::cout << std::setw(34) << " ";
                        }
                        ++icount;

                        std::cout << std::setw(4) << atomlist[k] + 1;
                        std::cout << "(" << std::setw(3) << kdname[kd[atomlist[k]] - 1] << ")";
                    }
                    std::cout << '\n';
                }

                dist_tmp = distances[i][j];
                atomlist.clear();
                atomlist.push_back(atoms[i][j]);
            } else {
                atomlist.push_back(atoms[i][j]);
            }
        }
        if (!atomlist.empty()) {
            nthnearest += 1;

            if (nthnearest > 1) std::cout << std::setw(13) << " ";

            std::cout << std::setw(3) << nthnearest << std::setw(10) << dist_tmp << " (" << std::setw(3)
                      << atomlist.size() << ") -";

            icount = 0;
            for (k = 0; k < atomlist.size(); ++k) {

                if (icount % 4 == 0 && icount > 0) {
                    std::cout << '\n';
                    std::cout << std::setw(34) << " ";
                }
                ++icount;

                std::cout << std::setw(4) << atomlist[k] + 1;
                std::cout << "(" << std::setw(3) << kdname[kd[atomlist[k]] - 1] << ")";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    atomlist.clear();
}


void Cluster::generate_interaction_information_by_cutoff(const size_t nat, const size_t natmin,
                                                         const std::vector<int> &kd,
                                                         const std::vector<std::vector<int>> &map_p2s,
                                                         const double *const *rc,
                                                         std::vector<std::vector<int>> &interaction_list) const
{
    interaction_list.resize(natmin);

    for (size_t i = 0; i < natmin; ++i) {

        interaction_list[i].clear();

        const auto iat = map_p2s[i][0];
        const auto ikd = kd[iat] - 1;

        for (size_t jat = 0; jat < nat; ++jat) {

            const auto jkd = kd[jat] - 1;
            const auto cutoff_tmp = rc[ikd][jkd];

            if (cutoff_tmp < 0.0) {
                // Cutoff 'None'
                interaction_list[i].push_back(jat);
            } else {
                if (distance_table[iat][jat].distances[0] <= cutoff_tmp) {
                    interaction_list[i].push_back(jat);
                }
            }
        }
    }
}

void Cluster::set_interaction_by_cutoff(const size_t nat, const std::vector<int> &kd, const size_t nat_prim,
                                        const std::vector<std::vector<int>> &map_p2s,
                                        std::vector<std::vector<std::vector<int>>> &interaction_pair_out) const
{
    interaction_pair_out.resize(maxorder);
    for (auto order = 0; order < maxorder; ++order) {
        generate_interaction_information_by_cutoff(nat,
                                                   nat_prim,
                                                   kd,
                                                   map_p2s,
                                                   cutoff_radii[order],
                                                   interaction_pair_out[order]);
    }
}

int Cluster::get_maxorder() const
{
    return maxorder;
}

void Cluster::define(const int maxorder_in, const size_t nkd, const int *nbody_include_in,
                     const double *cutoff_radii_in)
{
    maxorder = maxorder_in;
    if (nbody_include) {
        deallocate(nbody_include);
    }
    allocate(nbody_include, maxorder);

    for (auto i = 0; i < maxorder; ++i) {
        nbody_include[i] = nbody_include_in[i];
    }

    // nkd == 0: Use current values, which are probably default values
    // defined at Cluster::init().
    if (nkd > 0) {
        if (cutoff_radii) {
            deallocate(cutoff_radii);
        }
        allocate(cutoff_radii, maxorder, nkd, nkd);
    }

    // if cutoff_radii_in = nullptr, use current value
    if (cutoff_radii_in) {
        auto counter = 0;
        for (auto i = 0; i < maxorder; ++i) {
            for (size_t j = 0; j < nkd; ++j) {
                for (size_t k = 0; k < nkd; ++k) {
                    cutoff_radii[i][j][k] = cutoff_radii_in[counter++];
                }
            }
        }
    }
}

int *Cluster::get_nbody_include() const
{
    return nbody_include;
}

std::string Cluster::get_ordername(const unsigned int order) const
{
    if (order == 0) {
        return "HARMONIC";
    } else {
        return "ANHARM" + std::to_string(order + 2);
    }
}

const std::set<IntList> &Cluster::get_unique_clusters(const unsigned int order) const
{
    return unique_clusters[order];
}

const std::vector<int> &Cluster::get_atoms_in_cutoff(const unsigned int order, const size_t atom_index) const
{
    if (order >= atoms_in_cutoff.size()) {
        exit("get_atoms_in_cutoff", "The first index is out of range.");
    }
    if (atom_index >= atoms_in_cutoff[0].size()) {
        exit("get_atoms_in_cutoff", "The second index is out of range.");
    }
    return atoms_in_cutoff[order][atom_index];
}

const std::set<InteractionCluster> &Cluster::get_interaction_cluster(const unsigned int order,
                                                                     const size_t atom_index) const
{
    return interaction_cluster[order][atom_index];
}

void Cluster::print_interaction_information(const size_t natmin, const std::vector<std::vector<int>> &map_p2s,
                                            const std::vector<int> &kd, const std::vector<std::string> &kdname,
                                            const std::vector<std::vector<std::vector<int>>> &interaction_list) const
{
    std::vector<int> intlist;

    std::cout << '\n';
    std::cout << "  List of interacting atom pairs considered for each order:\n";

    for (auto order = 0; order < maxorder; ++order) {

        std::cout << '\n' << "   ***" << get_ordername(order) << "***\n";

        for (size_t i = 0; i < natmin; ++i) {

            if (interaction_list[order][i].empty()) {
                std::cout << "   No interacting atoms! Skipped.\n";
                continue; // no interaction
            }

            const auto iat = map_p2s[i][0];

            intlist.clear();
            for (auto &it: interaction_list[order][i]) {
                intlist.push_back(it);
            }
            std::sort(intlist.begin(), intlist.end());

            // write atoms inside the cutoff radius
            std::cout << "    Atom " << std::setw(5) << iat + 1 << "(" << std::setw(3) << kdname[kd[iat] - 1] << ")"
                      << " interacts with atoms ... \n";

            for (size_t id = 0; id < intlist.size(); ++id) {
                if (id % 6 == 0) {
                    if (id == 0) {
                        std::cout << "   ";
                    } else {
                        std::cout << '\n';
                        std::cout << "   ";
                    }
                }
                std::cout << std::setw(5) << intlist[id] + 1 << "(" << std::setw(3) << kdname[kd[intlist[id]] - 1]
                          << ")";
            }

            std::cout << "\n\n";
            std::cout << "    Number of total interaction pairs = " << interaction_list[order][i].size() << "\n\n";
        }
    }

    std::cout << '\n';
}


bool Cluster::is_incutoff(const int n, const int *atomnumlist, const size_t order, const std::vector<int> &kd) const
{
    for (auto i = 0; i < n; ++i) {
        const auto iat = atomnumlist[i];
        const auto ikd = kd[iat] - 1;

        for (auto j = i + 1; j < n; ++j) {
            const auto jat = atomnumlist[j];
            const auto jkd = kd[jat] - 1;

            const auto cutoff_tmp = cutoff_radii[order][ikd][jkd];

            if (cutoff_tmp >= 0.0 && (distance_table[iat][jat].distances[0] > cutoff_tmp)) {
                return false;
            }
        }
    }
    return true;
}

int Cluster::nbody(const int n, const int *arr)
{
    std::vector<int> v(n);

    for (auto i = 0; i < n; ++i) {
        v[i] = arr[i];
    }
    std::stable_sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    const auto ret = v.size();
    v.clear();

    return static_cast<int>(ret);
}

bool Cluster::satisfy_nbody_rule(const int nelem, const int *arr, const int order) const
{
    return nbody(nelem, arr) <= nbody_include[order];
}


void Cluster::calc_interaction_clusters(const size_t natmin, const std::vector<int> &kd,
                                        const std::vector<std::vector<int>> &map_p2s,
                                        const std::vector<Eigen::MatrixXd> &x_image, const int *exist,
                                        const int periodic_image_conv)
{
    //
    // Calculate the complete set of clusters for all orders.
    //
    for (auto order = 0; order < maxorder; ++order) {
        set_interaction_cluster(order,
                                natmin,
                                kd,
                                map_p2s,
                                atoms_in_cutoff[order],
                                x_image,
                                exist,
                                periodic_image_conv,
                                interaction_cluster[order]);
    }
}


void Cluster::set_interaction_cluster(const int order, const size_t natmin, const std::vector<int> &kd,
                                      const std::vector<std::vector<int>> &map_p2s,
                                      const std::vector<std::vector<int>> &interaction_pair_in,
                                      const std::vector<Eigen::MatrixXd> &x_image, const int *exist,
                                      const int periodic_image_conv,
                                      std::vector<std::set<InteractionCluster>> &interaction_cluster_out) const
{
    //
    // Calculate a set of clusters for the given order
    //

    int *list_now;
    allocate(list_now, order + 2);

    for (size_t i = 0; i < natmin; ++i) {

        interaction_cluster_out[i].clear();

        const auto iat = map_p2s[i][0];
        const auto ikd = kd[iat] - 1;
        list_now[0] = iat;

        // List of 2-body interaction pairs
        std::vector<int> intlist(interaction_pair_in[i]);
        std::sort(intlist.begin(), intlist.end()); // Need to sort here

        if (order == 0) {

            // Harmonic term

            std::vector<int> atom_tmp, cell_tmp;
            std::vector<std::vector<int>> comb_cell_min;

            for (auto ielem: intlist) {

                const auto jat = ielem;
                list_now[1] = jat;

                if (!satisfy_nbody_rule(2, list_now, 0)) continue;

                comb_cell_min.clear();
                atom_tmp.clear();
                atom_tmp.push_back(jat);

                for (auto j = 0; j < distance_table[iat][jat].ncells_minimum_distance; ++j) {
                    cell_tmp.clear();
                    cell_tmp.push_back(distance_table[iat][jat].cells[j]);
                    comb_cell_min.push_back(cell_tmp);
                }
                interaction_cluster_out[i].insert(
                    InteractionCluster(atom_tmp, comb_cell_min, distance_table[iat][jat].distances[0]));
            }

        } else if (order > 0) {

            // Anharmonic terms
            std::vector<std::vector<int>> data_vec, data_vec_sub;

            data_vec.clear();
            data_vec_sub.clear();
            std::vector<int> vv(intlist.size());

            // First, we generate all possible combinations of atom indices from intlist.
            // The number of different atoms in the list is nbody_include[order].
            std::fill(vv.begin(), vv.begin() + nbody_include[order], 1);
            std::vector<int> vec_tmp(nbody_include[order], 0);
            do {
                auto icount = 0;
                for (auto ii = 0; ii < intlist.size(); ++ii) {
                    if (vv[ii] == 1) {
                        vec_tmp[icount] = intlist[ii];
                        ++icount;
                    }
                }

                // For each combination of atoms, we generate all possible combinations with repetition.
                CombinationWithRepetition<int> g_sub2(vec_tmp.begin(), vec_tmp.end(), order + 1);
                do {
                    data_vec_sub.emplace_back(g_sub2.now());
                } while (g_sub2.next());
            } while (std::prev_permutation(vv.begin(), vv.end()));

            // The generated combinations are not necessarily unique. So, let's make them unique.
            std::sort(data_vec_sub.begin(), data_vec_sub.end());
            data_vec_sub.erase(std::unique(data_vec_sub.begin(), data_vec_sub.end()), data_vec_sub.end());

            // Finally, we check if the generated combinations satisfy the NBODY-rule.
            for (const auto &it: data_vec_sub) {
                list_now[0] = iat;
                for (auto m = 0; m < order + 1; ++m)
                    list_now[m + 1] = it[m];
                if (satisfy_nbody_rule(order + 2, list_now, order)) {
                    data_vec.emplace_back(it);
                }
            }

            intlist.clear();
            const auto ndata = data_vec.size();


#pragma omp parallel
            {
                std::vector<int> data_now;
                std::vector<int> intpair_uniq, cellpair;
                std::vector<std::vector<int>> comb_cell_atom_center;
                std::vector<int> cell_vector;
                std::vector<double> dist_vector;
                std::vector<int> group_atom;
                std::vector<int> accum_tmp;
                std::vector<MinDistList> distance_list;
                std::vector<std::vector<int>> pairs_icell, comb_cell, comb_cell_min;
                std::set<InteractionCluster> interaction_cluster_omp;

                int icount;

                interaction_cluster_omp.clear();

#pragma omp for
                for (size_t idata = 0; idata < ndata; ++idata) {

                    data_now = data_vec[idata];

                    // Uniq the list of atoms in data like as follows:
                    // cubic   term : (i, i) --> (i) x 2
                    // quartic term : (i, i, j) --> (i, j) x (2, 1)
                    intpair_uniq.clear();
                    group_atom.clear();
                    icount = 1;

                    for (auto m = 0; m < order; ++m) {
                        if (data_now[m] == data_now[m + 1]) {
                            ++icount;
                        } else {
                            group_atom.push_back(icount);
                            intpair_uniq.push_back(data_now[m]);
                            icount = 1;
                        }
                    }
                    group_atom.push_back(icount);
                    intpair_uniq.push_back(data_now[order]);

                    pairs_icell.clear();
                    for (const int jat: intpair_uniq) {
                        const auto jkd = kd[jat] - 1;

                        const auto rc_tmp = cutoff_radii[order][ikd][jkd];
                        cell_vector.clear();

                        // Loop over the cell images of atom 'jat' and add to the list
                        // as a candidate for the cluster.
                        // The periodic images whose distance is larger than the minimum value
                        // of the distance(iat, jat) can be added to the cell_vector list.
                        for (auto k = 0; k < distance_table[iat][jat].distances.size(); ++k) {
                            if (rc_tmp < 0.0 || distance_table[iat][jat].distances[k] <= rc_tmp) {
                                cell_vector.emplace_back(distance_table[iat][jat].cells[k]);
                            }
                        }
                        pairs_icell.push_back(cell_vector);
                    }

                    accum_tmp.clear();
                    comb_cell.clear();
                    cell_combination(pairs_icell, 0, accum_tmp, comb_cell);

                    distance_list.clear();
                    for (const auto &it_comb_cell: comb_cell) {

                        cellpair.clear();

                        for (auto k = 0; k < group_atom.size(); ++k) {
                            for (auto m = 0; m < group_atom[k]; ++m) {
                                cellpair.push_back(it_comb_cell[k]);
                            }
                        }

                        dist_vector.clear();

                        for (auto k = 0; k < cellpair.size(); ++k) {
                            dist_vector.push_back(distance(x_image[cellpair[k]].row(data_now[k]), x_image[0].row(iat)));
                        }

                        // Flag to check if the distance is smaller than the cutoff radius
                        auto isok = true;

                        for (auto k = 0; k < cellpair.size(); ++k) {
                            for (auto ii = k + 1; ii < cellpair.size(); ++ii) {
                                const auto dist_tmp = distance(x_image[cellpair[k]].row(data_now[k]),
                                                               x_image[cellpair[ii]].row(data_now[ii]));
                                const auto rc_tmp = cutoff_radii[order][kd[data_now[k]] - 1][kd[data_now[ii]] - 1];
                                if (rc_tmp >= 0.0 && dist_tmp > rc_tmp) {
                                    isok = false;
                                    break;
                                }
                                dist_vector.emplace_back(dist_tmp);
                            }
                            if (!isok) break;
                        }
                        if (isok) {
                            // This combination is a candidate of the minimum distance cluster
                            distance_list.emplace_back(cellpair, dist_vector);
                        }
                    } // close loop over the periodic image

                    if (!distance_list.empty()) {
                        // If the distance_list is not empty, there is a set of periodic images
                        // that satisfies the condition of the cluster.

                        if (periodic_image_conv == 0) {
                            // assign IFCs to periodic images in which the center atom and each of the other atoms
                            // are nearest.
                            // The distance between non-center atoms are not considered.
                            // The IFCs in this convention automatically satisfies the ASR without additional constraint,
                            // but does not satisfy the permutation symmetry.

                            pairs_icell.clear();
                            for (const int jat: intpair_uniq) {
                                cell_vector.clear();

                                for (auto ii = 0; ii < distance_table[iat][jat].ncells_minimum_distance; ++ii) {
                                    cell_vector.emplace_back(distance_table[iat][jat].cells[ii]);
                                }
                                pairs_icell.emplace_back(cell_vector);
                            }

                            accum_tmp.clear();
                            comb_cell.clear();
                            comb_cell_atom_center.clear();
                            cell_combination(pairs_icell, 0, accum_tmp, comb_cell);

                            for (const auto &it_comb_cell: comb_cell) {
                                cellpair.clear();
                                for (auto k = 0; k < group_atom.size(); ++k) {
                                    for (auto m = 0; m < group_atom[k]; ++m) {
                                        cellpair.emplace_back(it_comb_cell[k]);
                                    }
                                }
                                comb_cell_atom_center.emplace_back(cellpair);
                            }

                            std::sort(distance_list.begin(), distance_list.end(), MinDistList::compare_max_distance);

                            const auto distmax =
                                *std::max_element(distance_list[0].dist.begin(), distance_list[0].dist.end());

                            interaction_cluster_omp.insert(
                                InteractionCluster(data_now, comb_cell_atom_center, distmax));

                        } else /* if(mirror_image_conv == 1)*/ {

                            // assign IFCs to periodic images in which the sum of the distances between the atom pairs
                            // is the smallest.
                            // The IFCs made in this convention satisfies the permutation symmetry.
                            // Additional constraints are imposed in constraint.cpp to make the IFCs satisfy ASR
                            // after assigning IFCs to the periodic images.

                            std::sort(distance_list.begin(), distance_list.end(), MinDistList::compare_sum_distance);
                            comb_cell_min.clear();

                            double sum_dist_min = 0.0;
                            for (auto j = 0; j < distance_list[0].dist.size(); ++j) {
                                sum_dist_min += distance_list[0].dist[j];
                            }
                            // std::cout << "sum_dist_min = " << sum_dist_min << '\n';
                            double sum_dist;
                            for (const auto &it_dist: distance_list) {
                                sum_dist = 0.0;

                                for (const double dist: it_dist.dist) {
                                    sum_dist += dist;
                                }

                                // In the following, only pairs having minimum sum of distances
                                // are stored.
                                if (std::abs(sum_dist - sum_dist_min) < eps6) {
                                    // if (sum_dist < sum_dist_min*1.2+eps6) { // This version is not used.
                                    comb_cell_min.emplace_back(it_dist.cell);
                                } else {
                                    // break;
                                }
                            }

                            interaction_cluster_omp.insert(InteractionCluster(data_now, comb_cell_min, sum_dist_min));
                        }
                    }
                }

#pragma omp critical
                {
                    for (const auto &it: interaction_cluster_omp) {
                        interaction_cluster_out[i].insert(it);
                    }
                }
                interaction_cluster_omp.clear();
            }
        }
    }
    deallocate(list_now);
}


void Cluster::cell_combination(const std::vector<std::vector<int>> &array, const size_t i,
                               const std::vector<int> &accum, std::vector<std::vector<int>> &comb) const
{
    if (i == array.size()) {
        comb.push_back(accum);
    } else {
        auto row = array[i];
        for (auto j: row) {
            auto tmp(accum);
            tmp.push_back(j);
            cell_combination(array, i + 1, tmp, comb);
        }
    }
}
