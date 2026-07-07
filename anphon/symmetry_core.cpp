/*
 symmetry_core.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "symmetry_core.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "constants.h"
#include "error.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "scph.h"
#include "system.h"

extern "C"
{
#include "spglib.h"
}

using namespace PHON_NS;

Symmetry::Symmetry(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Symmetry::~Symmetry()
{}

void Symmetry::set_default_variables()
{
    file_sym = "SYMM_INFO_PRIM";
    time_reversal_sym = true;
    nsym = 0;
    nsym_ref = 0;
    printsymmetry = false;
    tolerance = 1.0e-3;
}

void Symmetry::setup_symmetry()
{
    time_reversal_sym = system->get_spin_prim().time_reversal_symm;
    SymmList.clear();
    SymmList_ref.clear();

    if ((phon->mode == "SCPH" && relaxation->relax_str != 0) || (phon->mode == "QHA" && relaxation->relax_str != 0)) {

        if (mympi->my_rank == 0) {
            std::cout << " ==========\n";
            std::cout << "  Symmetry \n";
            std::cout << " ==========\n\n";

            const auto cell_tmp = system->get_primcell(true);
            const auto cell_tmp_ref = system->get_primcell();


            std::cout << "  Primitive cell ";
            setup_symmetry_operation(cell_tmp_ref, system->get_spin_prim(), system->get_atomtype_group(), SymmList_ref);

            std::cout << "  Distorted cell ";
            setup_symmetry_operation(cell_tmp, system->get_spin_prim(), system->get_atomtype_group(true), SymmList);

            nsym = SymmList.size();
            nsym_ref = SymmList_ref.size();
        }
    } else {
        if (mympi->my_rank == 0) {
            std::cout << " ==========\n";
            std::cout << "  Symmetry \n";
            std::cout << " ==========\n\n";

            const auto cell_tmp = system->get_primcell();
            setup_symmetry_operation(cell_tmp, system->get_spin_prim(), system->get_atomtype_group(), SymmList);

            nsym = SymmList.size();
        }
    }

    MPI_Bcast(&nsym, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsym_ref, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    broadcast_symmlist(SymmList);
    broadcast_symmlist(SymmList_ref);

    if (mympi->my_rank == 0) {

        bool use_distorted_structure = false;

        std::cout << '\n';
        std::cout << "  Number of symmetry operations : " << nsym << '\n';
        if ((phon->mode == "SCPH" && relaxation->relax_str != 0) || (phon->mode == "QHA" && relaxation->relax_str != 0))
        {
            std::cout << "  Number of symmetry operations in reference structure : " << nsym_ref << "\n\n";
            use_distorted_structure = true;
        }

        const auto cell_tmp = system->get_primcell(use_distorted_structure);

        gensym_withmap(cell_tmp.lattice_vector, cell_tmp.x_fractional, cell_tmp.kind, SymmList, SymmListWithMap);

        if ((phon->mode == "SCPH" && relaxation->relax_str != 0) || (phon->mode == "QHA" && relaxation->relax_str != 0))
        {
            gensym_withmap(system->get_primcell().lattice_vector,
                           system->get_primcell().x_fractional,
                           system->get_primcell().kind,
                           SymmList_ref,
                           SymmListWithMap_ref);
        }
    }
}

void Symmetry::setup_symmetry_operation(const Cell &cell_in, const Spin &spin_in,
                                        const std::vector<std::vector<unsigned int>> &atomtype_in,
                                        std::vector<SymmetryOperation> &symlist, const int verbosity) const
{
    // input cell into a true primitive cell.
    if (spin_in.lspin && spin_in.noncollinear) {
        if (verbosity > 0) {
            std::cout << "  Switch to the internal symmetry finder from spglib when NONCOLLINEAR = 1.\n";
        }
        findsym_alm(cell_in, atomtype_in, spin_in, symlist);
    } else {
        std::string spgsymbol;
        const auto spgnum = findsym_spglib(cell_in, atomtype_in, spin_in, spgsymbol, symlist);
        if (verbosity > 0) {
            std::cout << "  Space group: " << spgsymbol << " (" << std::setw(3) << spgnum << ")\n";
        }
    }

    // The order in symmetry_data_prim changes for each run because it was generated
    // with OpenMP. Therefore, we sort the list here to have the same result at all time.
    std::sort(symlist.begin() + 1, symlist.end());
}


void Symmetry::findsym_alm(const Cell &cell, const std::vector<std::vector<unsigned int>> &atomtype_group,
                           const Spin &spin, std::vector<SymmetryOperation> &symm_out) const
{
    std::vector<RotationMatrix> LatticeSymmList;

    // Generate rotational matrices that don't change the metric tensor
    LatticeSymmList.clear();
    find_lattice_symmetry(cell.lattice_vector, LatticeSymmList);

    // Generate all the space group operations with translational vectors
    // The data is stored in symmetry_data_super.
    symm_out.clear();
    find_crystal_symmetry(cell, atomtype_group, spin, LatticeSymmList, symm_out);

    LatticeSymmList.clear();
}

void Symmetry::find_lattice_symmetry(const Eigen::Matrix3d &aa, std::vector<RotationMatrix> &LatticeSymmList) const
{
    /*
    Find the rotational matrices that leave the metric tensor invariant.

    Metric tensor G = (g)_{ij} = a_{i} * a_{j} is invariant under crystal symmetry operations T,
    i.e. T^{t}GT = G. Since G can be written as G = A^{t}A, the invariance condition is given by
    (AT)^{t}(AT) = G0 (original).
    */

    int i, j, k;

    auto nsym_tmp = 0;
    int mat_tmp[3][3];
    Eigen::Matrix3d rot_tmp;
    Eigen::Matrix3d aa_rot;

    double metric_tensor[3][3];
    double metric_tensor_rot[3][3];


    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            metric_tensor[i][j] = 0.0;
            for (k = 0; k < 3; ++k) {
                metric_tensor[i][j] += aa(k, i) * aa(k, j);
            }
        }
    }

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            if (i == j) {
                mat_tmp[i][i] = 1;
            } else {
                mat_tmp[i][j] = 0;
            }
        }
    }

    // Identity matrix should be the first entry.
    LatticeSymmList.emplace_back(mat_tmp);

    for (int m11 = -1; m11 <= 1; ++m11) {
        for (int m12 = -1; m12 <= 1; ++m12) {
            for (int m13 = -1; m13 <= 1; ++m13) {
                for (int m21 = -1; m21 <= 1; ++m21) {
                    for (int m22 = -1; m22 <= 1; ++m22) {
                        for (int m23 = -1; m23 <= 1; ++m23) {
                            for (int m31 = -1; m31 <= 1; ++m31) {
                                for (int m32 = -1; m32 <= 1; ++m32) {
                                    for (int m33 = -1; m33 <= 1; ++m33) {

                                        if (m11 == 1 && m12 == 0 && m13 == 0 && m21 == 0 && m22 == 1 && m23 == 0 &&
                                            m31 == 0 && m32 == 0 && m33 == 1)
                                            continue;

                                        const double det = m11 * (m22 * m33 - m32 * m23) -
                                                           m21 * (m12 * m33 - m32 * m13) +
                                                           m31 * (m12 * m23 - m22 * m13);

                                        if (det != 1 && det != -1) continue;

                                        rot_tmp(0, 0) = m11;
                                        rot_tmp(0, 1) = m12;
                                        rot_tmp(0, 2) = m13;
                                        rot_tmp(1, 0) = m21;
                                        rot_tmp(1, 1) = m22;
                                        rot_tmp(1, 2) = m23;
                                        rot_tmp(2, 0) = m31;
                                        rot_tmp(2, 1) = m32;
                                        rot_tmp(2, 2) = m33;

                                        // Here, aa_rot = aa * rot_tmp is correct.
                                        aa_rot = aa * rot_tmp;

                                        for (i = 0; i < 3; ++i) {
                                            for (j = 0; j < 3; ++j) {
                                                metric_tensor_rot[i][j] = 0.0;
                                                for (k = 0; k < 3; ++k) {
                                                    metric_tensor_rot[i][j] += aa_rot(k, i) * aa_rot(k, j);
                                                }
                                            }
                                        }

                                        double res = 0.0;
                                        for (i = 0; i < 3; ++i) {
                                            for (j = 0; j < 3; ++j) {
                                                res += pow2(metric_tensor[i][j] - metric_tensor_rot[i][j]);
                                            }
                                        }

                                        // Metric tensor is invariant under symmetry operations.

                                        if (res < tolerance * tolerance) {
                                            ++nsym_tmp;
                                            for (i = 0; i < 3; ++i) {
                                                for (j = 0; j < 3; ++j) {
                                                    mat_tmp[i][j] = static_cast<int>(rot_tmp(i, j));
                                                }
                                            }
                                            LatticeSymmList.emplace_back(mat_tmp);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (LatticeSymmList.size() > 48) {
        exit("find_lattice_symmetry", "Number of lattice symmetry is larger than 48.");
    }
}

void Symmetry::find_crystal_symmetry(const Cell &cell, const std::vector<std::vector<unsigned int>> &atomtype_group,
                                     const Spin &spin, const std::vector<RotationMatrix> &LatticeSymmList,
                                     std::vector<SymmetryOperation> &symm_out) const
{
    unsigned int i, j;
    unsigned int iat, jat, kat, lat;
    Eigen::Vector3d x_rot, x_tmp;
    Eigen::Matrix3d rot;
    Eigen::Matrix3i rot_int;
    Eigen::Matrix3d rot_tmp, rot_cart;
    Eigen::Vector3d mag, mag_rot;
    Eigen::Vector3d tran;
    Eigen::Vector3d x_rot_tmp;
    Eigen::Vector3d xdiff;
    Eigen::Vector3d tmp;
    double diff;
    const auto nclass = atomtype_group.size();

    int ii;
    size_t jj, kk;
    unsigned int itype;

    bool is_found;
    bool isok;
    bool mag_sym1, mag_sym2;
    bool is_identity_matrix;

    const Eigen::Matrix3d mat_identity = Eigen::Matrix3d::Identity();

    // Add identity matrix first.
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            if (i == j) {
                rot_int(i, j) = 1;
                rot_cart(i, j) = 1.0;
            } else {
                rot_int(i, j) = 0;
                rot_cart(i, j) = 0.0;
            }
        }
        tran[i] = 0.0;
    }

    symm_out.emplace_back(rot_int, tran, rot_cart, is_translation(rot_int));

    for (auto &it_latsym: LatticeSymmList) {

        iat = atomtype_group[0][0];
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                rot(i, j) = static_cast<double>(it_latsym.mat(i, j));
            }
        }

        for (i = 0; i < 3; ++i) x_tmp[i] = cell.x_fractional(iat, i);
        //rotvec(x_rot, x_tmp, rot);
        x_rot = rot * x_tmp;

#ifdef _OPENMP
#pragma omp parallel for private(jat,                                                                                  \
                                     tran,                                                                             \
                                     isok,                                                                             \
                                     kat,                                                                              \
                                     x_tmp,                                                                            \
                                     x_rot_tmp,                                                                        \
                                     is_found,                                                                         \
                                     lat,                                                                              \
                                     tmp,                                                                              \
                                     diff,                                                                             \
                                     i,                                                                                \
                                     j,                                                                                \
                                     itype,                                                                            \
                                     jj,                                                                               \
                                     kk,                                                                               \
                                     is_identity_matrix,                                                               \
                                     mag,                                                                              \
                                     mag_rot,                                                                          \
                                     rot_tmp,                                                                          \
                                     rot_cart,                                                                         \
                                     mag_sym1,                                                                         \
                                     mag_sym2,                                                                         \
                                     xdiff)
#endif
        for (ii = 0; ii < atomtype_group[0].size(); ++ii) {
            jat = atomtype_group[0][ii];

            for (i = 0; i < 3; ++i) {
                tran[i] = cell.x_fractional(jat, i) - x_rot[i];
                tran[i] = tran[i] - nint(tran[i]);
            }

            is_identity_matrix = (rot - mat_identity).norm() < eps6;
            if (is_identity_matrix) continue;

            isok = true;

            for (itype = 0; itype < nclass; ++itype) {

                for (jj = 0; jj < atomtype_group[itype].size(); ++jj) {

                    kat = atomtype_group[itype][jj];

                    for (i = 0; i < 3; ++i) x_tmp[i] = cell.x_fractional(kat, i);
                    x_rot_tmp = rot * x_tmp;

                    for (i = 0; i < 3; ++i) {
                        x_rot_tmp[i] += tran[i];
                    }

                    is_found = false;

                    for (kk = 0; kk < atomtype_group[itype].size(); ++kk) {

                        lat = atomtype_group[itype][kk];

                        for (i = 0; i < 3; ++i) {
                            tmp[i] = std::fmod(std::abs(cell.x_fractional(lat, i) - x_rot_tmp[i]), 1.0);
                            tmp[i] = std::min<double>(tmp[i], 1.0 - tmp[i]);
                        }
                        xdiff = cell.lattice_vector * tmp;
                        diff = xdiff.norm(); // distance in Cartesian coordinate
                        if (diff < tolerance) {
                            is_found = true;
                            break;
                        }
                    }

                    if (!is_found) isok = false;
                }
            }

            if (isok) {
                rot_tmp = rot * cell.reciprocal_lattice_vector;
                rot_cart = cell.lattice_vector * rot_tmp * inv_tpi;

                if (spin.lspin && spin.noncollinear) {

                    for (i = 0; i < 3; ++i) {
                        mag[i] = spin.magmom[jat][i];
                        mag_rot[i] = spin.magmom[iat][i];
                    }

                    mag_rot = rot_cart * mag_rot;

                    // In the case of improper rotation, the factor -1 should be multiplied
                    // because the inversion operation doesn't flip the spin.
                    if (!is_proper(rot_cart)) {
                        for (i = 0; i < 3; ++i) {
                            mag_rot[i] = -mag_rot[i];
                        }
                    }

                    mag_sym1 =
                        (pow2(mag[0] - mag_rot[0]) + pow2(mag[1] - mag_rot[1]) + pow2(mag[2] - mag_rot[2])) < eps6;

                    mag_sym2 =
                        (pow2(mag[0] + mag_rot[0]) + pow2(mag[1] + mag_rot[1]) + pow2(mag[2] + mag_rot[2])) < eps6;

                    if (!mag_sym1 && !mag_sym2) {
                        isok = false;
                    } else if (!mag_sym1 && mag_sym2 && !spin.time_reversal_symm) {
                        isok = false;
                    }
                }
            }

            if (isok) {
#ifdef _OPENMP
#pragma omp critical
#endif
                symm_out.emplace_back(it_latsym.mat, tran, rot_cart, is_translation(it_latsym.mat));
            }
        }
    }
}

int Symmetry::findsym_spglib(const Cell &cell, const std::vector<std::vector<unsigned int>> &atomtype_group,
                             const Spin &spin, std::string &spgsymbol, std::vector<SymmetryOperation> &symm_out) const
{
    int i, j;
    // spg_get_dataset takes double(*)[3] / int(*)[3][3] (spglib C ABI boundary),
    // so these three stay on allocate()/deallocate().
    double(*position)[3];
    double(*translation)[3];
    int(*rotation)[3][3];
    char symbol[11];
    double aa_tmp[3][3];
    NDArray<int, 1> types_tmp;

    const auto nat = cell.number_of_atoms;

    allocate(position, nat);
    types_tmp.resize(nat);

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            aa_tmp[i][j] = cell.lattice_vector(i, j);
        }
    }
    for (i = 0; i < nat; ++i) {
        for (j = 0; j < 3; ++j) {
            position[i][j] = cell.x_fractional(i, j);
        }
    }

    if (spin.lspin) {
        for (i = 0; i < atomtype_group.size(); ++i) {
            for (j = 0; j < atomtype_group[i].size(); ++j) {
                types_tmp[atomtype_group[i][j]] = i;
            }
        }
    } else {
        for (i = 0; i < nat; ++i) {
            types_tmp[i] = cell.kind[i];
        }
    }

    const auto spgdataset = spg_get_dataset(aa_tmp, position, types_tmp, nat, tolerance);
    const auto nsym_out = spgdataset->n_operations;
    if (nsym_out == 0) exit("findsym_spglib", "Error occurred in spg_get_dataset");

    allocate(translation, nsym_out);
    allocate(rotation, nsym_out);
    for (auto isym = 0; isym < nsym_out; ++isym) {
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                rotation[isym][i][j] = spgdataset->rotations[isym][i][j];
            }
            translation[isym][i] = spgdataset->translations[isym][i];
        }
    }

    const auto spgnum = spgdataset->spacegroup_number;
    spgsymbol = spgdataset->international_symbol;
    spg_free_dataset(spgdataset);

    // Copy symmetry information
    symm_out.clear();
    Eigen::Matrix3d rot_cartesian;
    Eigen::Matrix3i rot_int;
    Eigen::Vector3d trans_tmp;

    for (i = 0; i < nsym_out; ++i) {

        for (j = 0; j < 3; ++j) {
            for (auto k = 0; k < 3; ++k) {
                rot_int(j, k) = rotation[i][j][k];
            }
            trans_tmp(j) = translation[i][j];
        }

        symop_in_cart(rot_cartesian, rot_int, cell.lattice_vector, cell.reciprocal_lattice_vector);

        symm_out.emplace_back(rot_int, trans_tmp, rot_cartesian, is_translation(rot_int));
    }

    deallocate(rotation);
    deallocate(translation);
    types_tmp.clear();
    deallocate(position);

    return spgnum;
}


void Symmetry::symop_in_cart(Eigen::Matrix3d &rot_cart, const Eigen::Matrix3i &rot_lattice,
                             const Eigen::Matrix3d &lavec, const Eigen::Matrix3d &rlavec)
{
    Eigen::Matrix3d sym_tmp;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sym_tmp(i, j) = static_cast<double>(rot_lattice(i, j));
        }
    }
    rot_cart = lavec * sym_tmp * rlavec * inv_tpi;
}

void Symmetry::gensym_withmap(const Eigen::Matrix3d &aa, const Eigen::MatrixXd &x, const std::vector<int> &kd,
                              const std::vector<SymmetryOperation> &symmlist_in,
                              std::vector<SymmetryOperationWithMapping> &symmlist_withmap_out) const
{
    // Generate symmetry operations in Cartesian coordinate with the atom-mapping information.

    Eigen::Matrix3d T;
    Eigen::Vector3d shift, tmp;
    NDArray<unsigned int, 1> map_tmp;
    int i, j;
    unsigned int natmin = x.rows();

    symmlist_withmap_out.clear();

    map_tmp.resize(natmin);

    for (const auto &isym: symmlist_in) {

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                T(i, j) = static_cast<double>(isym.rotation(i, j));
            }
        }

        for (i = 0; i < 3; ++i) {
            shift[i] = isym.tran[i];
        }

        const Eigen::Matrix3d S_recip = T.inverse().transpose();

        // Convert to Cartesian coordinate
        const Eigen::Matrix3d mat_tmp = T * aa.inverse();
        const Eigen::Matrix3d S = aa * mat_tmp;

        // Generate mapping information

        for (i = 0; i < natmin; ++i) {

            Eigen::Vector3d x_mod = T * x.row(i).transpose();

            for (j = 0; j < 3; ++j) {
                x_mod[j] += shift[j];
            }

            int num_mapped = -1;

            for (j = 0; j < natmin; ++j) {

                if (kd[j] == kd[i]) {

                    for (int k = 0; k < 3; ++k) {
                        tmp[k] = std::fmod(std::abs(x_mod[k] - x(j, k)), 1.0);
                        tmp[k] = std::min<double>(tmp[k], 1.0 - tmp[k]);
                    }
                    const auto diff = tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2];
                    if (diff < tolerance * tolerance) {
                        num_mapped = j;
                        break;
                    }
                }
            }

            if (num_mapped == -1) {
                exit("gensym_withmap", "cannot find a equivalent atom");
            }
            map_tmp[i] = num_mapped;
        }

        // Add to vector

        symmlist_withmap_out.emplace_back(S, T, S_recip, map_tmp, natmin, shift);
    }
}

void Symmetry::broadcast_symmlist(std::vector<SymmetryOperation> &sym) const
{
    int i, j, k;
    int n;
    NDArray<int, 3> rot_tmp;
    NDArray<double, 3> rot_tmp2;
    NDArray<double, 2> tran_tmp;

    if (mympi->my_rank == 0) n = sym.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Nothing to broadcast for an empty symmetry list (e.g. SymmList_ref is only
    // populated for SCPH/QHA structural relaxation; it is empty in other modes).
    // Returning here also avoids calling allocate() with a leading dimension of
    // zero, which would write past a zero-size allocation.
    if (n == 0) return;

    rot_tmp.resize(n, 3, 3);
    rot_tmp2.resize(n, 3, 3);
    tran_tmp.resize(n, 3);

    if (mympi->my_rank == 0) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    rot_tmp[i][j][k] = sym[i].rotation(j, k);
                    rot_tmp2[i][j][k] = sym[i].rotation_cart(j, k);
                }
                tran_tmp[i][j] = sym[i].tran[j];
            }
        }
    }
    MPI_Bcast(&rot_tmp[0][0][0], 9 * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rot_tmp2[0][0][0], 9 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tran_tmp[0][0], 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mympi->my_rank > 0) {
        Eigen::Matrix3i rotation;
        Eigen::Matrix3d rotation_cart;
        Eigen::Vector3d tran;
        for (i = 0; i < n; ++i) {
            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    rotation(j, k) = rot_tmp[i][j][k];
                    rotation_cart(j, k) = rot_tmp2[i][j][k];
                }
                tran[j] = tran_tmp[i][j];
            }
            sym.emplace_back(rotation, tran, rotation_cart, is_translation(rotation));
        }
    }

    rot_tmp.clear();
    rot_tmp2.clear();
    tran_tmp.clear();
}

bool Symmetry::is_proper(const Eigen::Matrix3d &rot)
{
    const auto det = rot.determinant();
    return std::abs(det - 1.0) < eps12;
}

bool Symmetry::is_translation(const Eigen::Matrix3i &rot)
{
    const Eigen::Matrix3i identity = Eigen::Matrix3i::Identity();
    return (rot == identity);
}
