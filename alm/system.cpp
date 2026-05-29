/*
 system.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "system.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
#include <set>
#include <utility>
#include "constants.h"
#include "error.h"
#include "mathfunctions.h"
#include "memory.h"
#include "smith.h"
#include "timer.h"

extern "C"
{
#include "spglib.h"
}

using namespace ALM_NS;

System::System()
{
    set_default_variables();
}

System::~System()
{
    deallocate_variables();
}

auto System::init(const int verbosity, std::unique_ptr<Timer> &timer) -> void
{
    timer->start_clock("system");

    // Compute primitive cell and supercell first.
    build_cells();

    // Set atomic types (kind + magmom)
    set_atomtype_group(supercell, spin_super, atomtype_group_super);
    set_atomtype_group(primcell, spin_prim, atomtype_group_prim);

    const auto nneib = 27;
    x_image.clear();
    x_image.shrink_to_fit();

    if (exist_image) {
        deallocate(exist_image);
    }
    allocate(exist_image, nneib);
    generate_coordinate_of_periodic_images();

    if (verbosity > 0) {
        print_structure_stdout(verbosity);
        if (spin_super.lspin) print_magmom_stdout();
        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------\n\n";
    }

    timer->stop_clock("system");
}

auto System::set_basecell(const double lavec_in[3][3], const size_t nat_in, const int *kind_in,
                          const double xf_in[][3]) -> void
{
    size_t i, j;
    std::vector<int> unique_nums(nat_in);
    auto wrong_number = false;
    bool in_unique_nums;

    for (i = 0; i < nat_in; i++) {
        unique_nums[i] = 0;
    }

    size_t nkd = 0;
    for (i = 0; i < nat_in; i++) {
        in_unique_nums = false;
        for (j = 0; j < nkd; j++) {
            if (unique_nums[j] == kind_in[i]) {
                in_unique_nums = true;
                break;
            }
        }
        if (!in_unique_nums) {
            unique_nums[nkd] = kind_in[i];
            nkd++;
        }
    }

    for (i = 0; i < nkd; i++) {
        if (static_cast<size_t>(unique_nums[i]) > nkd) {
            std::cout << " WARNING : integers assigned to atoms are wrong. "
                      << " The numbers will be resorted.\n";
            wrong_number = true;
            break;
        }
    }

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            inputcell.lattice_vector(i, j) = lavec_in[i][j];
        }
    }
    set_reciprocal_latt(inputcell.lattice_vector, inputcell.reciprocal_lattice_vector);

    inputcell.volume = volume(inputcell.lattice_vector, Direct);
    inputcell.number_of_atoms = nat_in;
    inputcell.number_of_elems = nkd;
    inputcell.kind.clear();
    inputcell.kind.shrink_to_fit();
    inputcell.x_fractional.resize(nat_in, 3);

    std::vector<double> xtmp;

    if (!wrong_number) {
        for (i = 0; i < nat_in; ++i) {
            inputcell.kind.push_back(kind_in[i]);
        }
    } else {
        for (i = 0; i < nat_in; ++i) {
            for (j = 0; j < nkd; j++) {
                if (kind_in[i] == unique_nums[j]) {
                    inputcell.kind.push_back(static_cast<int>(j + 1));
                }
            }
        }
    }

    xtmp.resize(3);
    for (i = 0; i < nat_in; ++i) {
        for (j = 0; j < 3; ++j) {
            xtmp[j] = xf_in[i][j];
        }
        // The fractional coordinate should be in the range of 0<=xf<1.
        for (j = 0; j < 3; ++j) {
            while (xtmp[j] >= 1.0) {
                xtmp[j] -= 1.0;
            }
            while (xtmp[j] < 0.0) {
                xtmp[j] += 1.0;
            }
            inputcell.x_fractional(i, j) = xtmp[j];
        }
    }

    inputcell.x_cartesian = inputcell.x_fractional * inputcell.lattice_vector.transpose();

    // This is needed to avoid segmentation fault.
    spin_input.magmom.clear();
    std::vector<double> vec(3);
    for (i = 0; i < nat_in; ++i) {
        for (j = 0; j < 3; ++j) {
            vec[j] = 0;
        }
        spin_input.magmom.push_back(vec);
    }
}

auto System::build_cells() -> void
{
    build_primcell();
    build_supercell();
}

auto System::build_primcell() -> void
{
    // Generate primitive cell information from the inputcell and PRIMCELL value
    // The symmetry detection is not performed here as it will be done
    // when the Symmetry::init() is called.

    transmat_to_prim_spglib = compute_transmat_to_prim_using_spglib(inputcell, symmetry_tolerance);

    if (autoset_primcell) transmat_to_prim = transmat_to_prim_spglib;

    const auto ndiv = nint(1.0 / transmat_to_prim.determinant());
    if (inputcell.number_of_atoms % ndiv != 0) {
        exit("build_primcell",
             "The determinant of PRIMCELL is not a divisor of NAT of the input.\n"
             " If you want to use the primitive cell detected by spglib, please set PRIMCELL = Auto.\n");
    }

    primcell.number_of_atoms = inputcell.number_of_atoms / ndiv;
    primcell.number_of_elems = inputcell.number_of_elems;

    spin_prim.lspin = spin_input.lspin;
    spin_prim.noncollinear = spin_input.noncollinear;
    spin_prim.time_reversal_symm = spin_input.time_reversal_symm;
    spin_prim.magmom.clear();
    spin_prim.magmom.shrink_to_fit();

    // (a_p, b_p, c_p) = (a_in, b_in, c_in) * Mat(inp->p)
    primcell.lattice_vector = inputcell.lattice_vector * transmat_to_prim;
    set_reciprocal_latt(primcell.lattice_vector, primcell.reciprocal_lattice_vector);
    primcell.volume = volume(primcell.lattice_vector, Direct);

    // Convert the basis of coordinates from inputcell to primitive fractional
    // (a_in, b_in, c_in) * xf_in = xc_in
    // (a_p, b_p, c_p) * xf_p = xc_in
    // xf_p = (a_p, b_p, c_p)^{-1} * (a_in, b_in, c_in) * xf_in
    //      = [Mat(inp->p)]^{-1} * xf_in
    Eigen::Matrix3d conversion_mat = transmat_to_prim.inverse().transpose();
    Eigen::MatrixXd xf_prim_all = inputcell.x_fractional * conversion_mat;

    std::vector<std::vector<double>> xf_unique, magmom_unique;
    std::vector<double> xf_tmp_vec(3), magmom_tmp_vec(3);
    std::vector<int> kind_unique;
    Eigen::VectorXd xf_tmp(3), xf_tmp2(3), xf_diff(3), xf_diff_cart(3);

    for (auto i = 0; i < inputcell.number_of_atoms; ++i) {
        xf_tmp = xf_prim_all.row(i);
        xf_tmp = xf_tmp.unaryExpr([](const double x) { return std::fmod(x, 1.0); });
        for (auto j = 0; j < 3; ++j) {
            if (xf_tmp[j] < -eps6) xf_tmp[j] += 1.0;
        }

        bool is_duplicate = false;
        // Just linear search for simplicity. Should be OK for relatively small number of inputs.
        for (auto k = 0; k < xf_unique.size(); ++k) {
            for (auto kk = 0; kk < 3; ++kk) {
                xf_tmp2[kk] = xf_unique[k][kk];
            }
            xf_diff = (xf_tmp - xf_tmp2).unaryExpr([](const double x) { return std::fmod(x, 1.0); });
            for (auto j = 0; j < 3; ++j) {
                if (xf_diff[j] < -0.5) xf_diff[j] += 1.0;
                if (xf_diff[j] >= 0.5) xf_diff[j] -= 1.0;
            }

            xf_diff_cart = primcell.lattice_vector * xf_diff;
            if (xf_diff_cart.norm() < symmetry_tolerance) {
                is_duplicate = true;

                if (kind_unique[k] != inputcell.kind[i]) {
                    exit("build_primcell",
                         "Different atoms with different element types occupy the same atomic site.\n"
                         " Please check the PRIMCELL and input structure carefully.\n"
                         " If you want to use the primitive cell detected by spglib, please set PRIMCELL = Auto.\n");
                }

                if (spin_input.lspin) {
                    double norm_magmom = 0.0;
                    for (auto kk = 0; kk < 3; ++kk) {
                        norm_magmom += pow2(spin_input.magmom[i][kk] - magmom_unique[k][kk]);
                    }
                    if (std::sqrt(norm_magmom) > eps6) {
                        exit("build_primcell",
                             "Different atoms with different MAGMOM entries occupy the same atomic site.\n"
                             "This is strange. Please check the PRIMCELL, MAGMOM, and input structure carefully.");
                    }
                }
            }
        }

        if (!is_duplicate) {
            for (auto j = 0; j < 3; ++j)
                xf_tmp_vec[j] = xf_tmp[j];
            xf_unique.emplace_back(xf_tmp_vec);
            kind_unique.emplace_back(inputcell.kind[i]);
            if (spin_input.lspin) {
                for (auto j = 0; j < 3; ++j)
                    magmom_tmp_vec[j] = spin_input.magmom[i][j];
                magmom_unique.emplace_back(magmom_tmp_vec);
            }
        }
    }

    if (xf_unique.size() != primcell.number_of_atoms) {
        std::cout << "primcell.number_of_atoms = " << primcell.number_of_atoms << '\n';
        std::cout << "xf_unique.size() = " << xf_unique.size() << '\n';
        exit("build_primcell",
             "Mapping to the primitive cell failed. Please check inputs (PRIMCELL and input structure).");
    }

    primcell.x_fractional.resize(primcell.number_of_atoms, 3);
    primcell.kind.resize(primcell.number_of_atoms);

    for (auto i = 0; i < primcell.number_of_atoms; ++i) {
        for (auto j = 0; j < 3; ++j) {
            primcell.x_fractional(i, j) = xf_unique[i][j];
        }
        primcell.kind[i] = kind_unique[i];
    }

    primcell.x_cartesian = primcell.x_fractional * primcell.lattice_vector.transpose();

    if (spin_prim.lspin) {
        std::copy(magmom_unique.begin(), magmom_unique.end(), std::back_inserter(spin_prim.magmom));
    }
}


auto System::build_supercell() -> void
{
    Eigen::MatrixXi transmat_int(3, 3);
    Eigen::MatrixXi D, R, L;

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            transmat_int(i, j) = nint(transmat_to_super(i, j));
        }
    }
    smith_decomposition(transmat_int, D, L, R);

    Eigen::MatrixXd D_double = D.cast<double>();
    Eigen::MatrixXd R_double = R.cast<double>();
    Eigen::MatrixXd L_double = L.cast<double>();

    //    std::cout << "transmat_to_super:" << transmat_to_super << '\n';

    // (a_s, b_s, c_s) = (a_in, b_in, c_in) * Mat(inp->s)
    supercell.lattice_vector = inputcell.lattice_vector * transmat_to_super;
    set_reciprocal_latt(supercell.lattice_vector, supercell.reciprocal_lattice_vector);
    supercell.volume = volume(supercell.lattice_vector, Direct);

    supercell.number_of_atoms = inputcell.number_of_atoms * nint(transmat_to_super.determinant());
    supercell.number_of_elems = inputcell.number_of_elems;
    supercell.x_fractional.resize(supercell.number_of_atoms, 3);

    Eigen::Matrix3d inv_transmat_to_super = transmat_to_super.inverse();

    Eigen::VectorXd DD(3), DD_mod(3);
    Eigen::MatrixXd DD_inverse = D_double.inverse();
    Eigen::MatrixXd transmat_newprim_to_origsuper = R_double * DD_inverse;

    supercell.kind.clear();

    std::vector<std::vector<double>> magmom_tmp;

    int counter = 0;
    for (auto iat = 0; iat < inputcell.number_of_atoms; ++iat) {
        Eigen::VectorXd xp_f = inputcell.x_fractional.row(iat);
        Eigen::VectorXd xs_f = inv_transmat_to_super * xp_f;
        int kind_now = inputcell.kind[iat];
        for (auto i = 0; i < D(0, 0); ++i) {
            for (auto j = 0; j < D(1, 1); ++j) {
                for (auto k = 0; k < D(2, 2); ++k) {
                    DD << static_cast<double>(i), static_cast<double>(j), static_cast<double>(k);
                    DD = transmat_newprim_to_origsuper * DD + xs_f;
                    DD_mod = DD.unaryExpr([](const double x) { return std::fmod(x, 1.0); });
                    // fmod allows negative values, so correct them.
                    for (auto m = 0; m < 3; ++m) {
                        if (DD_mod[m] < 0.0) DD_mod[m] += 1.0;
                    }
                    supercell.x_fractional.row(counter) = DD_mod;
                    supercell.kind.emplace_back(kind_now);
                    if (spin_input.lspin) magmom_tmp.emplace_back(spin_input.magmom[iat]);
                    ++counter;
                }
            }
        }
    }

    supercell.x_cartesian = supercell.x_fractional * supercell.lattice_vector.transpose();

    spin_super.lspin = spin_input.lspin;
    spin_super.noncollinear = spin_input.noncollinear;
    spin_super.time_reversal_symm = spin_input.time_reversal_symm;
    spin_super.magmom.clear();
    spin_super.magmom.shrink_to_fit();
    if (spin_super.lspin) {
        std::copy(magmom_tmp.begin(), magmom_tmp.end(), std::back_inserter(spin_super.magmom));
    }
}

auto System::get_supercell() const -> const Cell &
{
    return supercell;
}

auto System::get_primcell() const -> const Cell &
{
    return primcell;
}

auto System::get_inputcell() const -> const Cell &
{
    return inputcell;
}

auto System::get_x_image() const -> const std::vector<Eigen::MatrixXd> &
{
    return x_image;
}

auto System::get_exist_image() const -> int *
{
    return exist_image;
}

auto System::set_tolerance(const double tolerance) -> void
{
    symmetry_tolerance = tolerance;
}

auto System::set_periodicity(const int is_periodic_in[3]) -> void
{
    if (!is_periodic) {
        // This should be already allocated though.
        allocate(is_periodic, 3);
    }
    for (unsigned int i = 0; i < 3; i++) {
        is_periodic[i] = is_periodic_in[i];
    }
}

auto System::get_periodicity() const -> int *
{
    return is_periodic;
}

auto System::set_kdname(const std::vector<std::string> &kdname_in) -> void
{
    if (inputcell.number_of_elems != kdname_in.size()) {
        exit("set_kdname",
             "The size of kdname_in is different from the number of elements inferred from the "
             "kind variable of the set_cell method.");
    }
    kdname.clear();
    std::copy(kdname_in.begin(), kdname_in.end(), std::back_inserter(kdname));
}

auto System::get_kdname() const -> const std::vector<std::string> &
{
    return kdname;
}

auto System::set_reciprocal_latt(const Eigen::Matrix3d &lavec_in, Eigen::Matrix3d &rlavec_out) const -> void
{
    // Compute reciprocal lattice vectors
    const auto det = lavec_in.determinant();

    if (std::abs(det) < eps12) {
        exit("set_reciprocal_latt", "Lattice Vector is singular");
    }
    rlavec_out = tpi * lavec_in.inverse();
}

auto System::volume(const Eigen::Matrix3d &mat_in, const LatticeType latttype_in) const -> double
{
    Eigen::Matrix3d mat;

    if (latttype_in == Direct) {
        mat = mat_in.transpose();
    } else if (latttype_in == Reciprocal) {
        mat = mat_in;
    } else {
        exit("volume", "Invalid LatticeType is given");
    }

    Eigen::Vector3d v1 = mat.row(0);
    Eigen::Vector3d v2 = mat.row(1);
    Eigen::Vector3d v3 = mat.row(2);

    const auto vol = std::abs(v1.dot(v2.cross(v3)));
    return vol;
}


auto System::set_default_variables() -> void
{
    supercell.number_of_atoms = 0;
    supercell.number_of_elems = 0;

    allocate(is_periodic, 3);
    is_periodic[0] = 1;
    is_periodic[1] = 1;
    is_periodic[2] = 1;

    exist_image = nullptr;
    str_magmom = "";

    spin_input.lspin = false;
    spin_input.noncollinear = 0;
    spin_input.time_reversal_symm = 1;
    transmat_to_super = Eigen::Matrix3d::Identity();
    transmat_to_prim = Eigen::Matrix3d::Identity();
}

auto System::deallocate_variables() -> void
{
    if (is_periodic) {
        deallocate(is_periodic);
    }
    if (exist_image) {
        deallocate(exist_image);
    }
}

auto System::set_spin_variables(const size_t nat_in, const bool lspin_in, const int noncol_in, const int trev_sym_in,
                                const double (*magmom_in)[3]) -> void
{
    spin_input.lspin = lspin_in;
    spin_input.noncollinear = noncol_in;
    spin_input.time_reversal_symm = trev_sym_in;
    spin_input.magmom.clear();

    std::vector<double> vec(3);
    for (size_t i = 0; i < nat_in; ++i) {
        for (auto j = 0; j < 3; ++j) {
            vec[j] = magmom_in[i][j];
        }
        spin_input.magmom.push_back(vec);
    }
}

auto System::get_spin(const std::string &cell) const -> const Spin &
{
    if (cell == "input") return spin_input;
    if (cell == "prim" || cell == "primitive") return spin_prim;
    return spin_super;
}

auto System::set_str_magmom(std::string str_magmom_in) -> void
{
    str_magmom = std::move(str_magmom_in);
}

auto System::get_str_magmom() const -> const std::string &
{
    return str_magmom;
}

auto System::get_atomtype_group(const std::string &cell) const -> const std::vector<std::vector<unsigned int>> &
{
    if (cell == "prim" || cell == "primitive") return atomtype_group_prim;
    return atomtype_group_super;
}


auto System::set_atomtype_group(const Cell &cell_in, const Spin &spin_in,
                                std::vector<std::vector<unsigned int>> &atomtype_group_out) -> void
{
    // In the case of collinear calculation, spin moments are considered as scalar
    // variables. Therefore, the same elements with different magnetic moments are
    // considered as different types. In noncollinear calculations,
    // magnetic moments are not considered in this stage. They will be treated
    // separately in symmetry.cpp where spin moments will be rotated and flipped
    // using time-reversal symmetry.

    unsigned int i;
    AtomType type_tmp{};
    std::set<AtomType> set_type;
    set_type.clear();

    for (i = 0; i < cell_in.number_of_atoms; ++i) {
        type_tmp.element = cell_in.kind[i];

        if (spin_in.lspin == 1 && spin_in.noncollinear == 0) {
            type_tmp.magmom = spin_in.magmom[i][2];
        } else {
            type_tmp.magmom = 0.0;
        }
        set_type.insert(type_tmp);
    }

    const auto natomtypes = set_type.size();
    atomtype_group_out.resize(natomtypes);

    for (i = 0; i < cell_in.number_of_atoms; ++i) {
        int count = 0;
        for (auto it: set_type) {
            if (spin_in.noncollinear || spin_in.lspin == 0) {
                if (cell_in.kind[i] == it.element) {
                    atomtype_group_out[count].push_back(i);
                }
            } else {
                if ((cell_in.kind[i] == it.element) && (std::abs(spin_in.magmom[i][2] - it.magmom) < eps6)) {
                    atomtype_group_out[count].push_back(i);
                }
            }
            ++count;
        }
    }
    set_type.clear();
}

auto System::set_transformation_matrices(const double transmat_to_super_in[3][3],
                                         const double transmat_to_prim_in[3][3], const int autoset_primcell_in) -> void
{
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            transmat_to_super(i, j) = transmat_to_super_in[i][j];
            transmat_to_prim(i, j) = transmat_to_prim_in[i][j];
        }
    }
    autoset_primcell = autoset_primcell_in;
}

auto System::find_primitive_cell(const Cell &cell_input, const Spin &spin_input, Cell &cell_out, Spin &spin_out,
                                 const double tolerance) const -> void
{
    // const auto spg_major_version = spg_get_major_version();
    // TODO: identify the primitive lattice with the information of magnetic ordering

    size_t i, j;
    int *types_tmp;
    double(*position)[3];
    double aa_tmp[3][3];

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            aa_tmp[i][j] = cell_input.lattice_vector(i, j);
        }
    }
    const auto nat = cell_input.number_of_atoms;

    allocate(position, nat);
    allocate(types_tmp, nat);

    for (i = 0; i < nat; ++i) {
        for (j = 0; j < 3; ++j) {
            position[i][j] = cell_input.x_fractional(i, j);
        }
        types_tmp[i] = cell_input.kind[i];
    }

    // Do not idealize as we want to avoid the rigid rotation of the input cell.
    const auto nat_prim_out = spg_standardize_cell(aa_tmp, position, types_tmp, nat, 1, 1, tolerance);
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            cell_out.lattice_vector(i, j) = aa_tmp[i][j];
        }
    }
    cell_out.number_of_atoms = nat_prim_out;
    cell_out.x_fractional.resize(nat_prim_out, 3);
    cell_out.kind.resize(nat_prim_out);
    for (i = 0; i < nat_prim_out; ++i) {
        for (j = 0; j < 3; ++j) {
            cell_out.x_fractional(i, j) = position[i][j];
        }
        cell_out.kind[i] = types_tmp[i];
    }
    cell_out.x_cartesian = cell_out.x_fractional * cell_out.lattice_vector.transpose();
    cell_out.number_of_elems = cell_input.number_of_elems;
    set_reciprocal_latt(cell_out.lattice_vector, cell_out.reciprocal_lattice_vector);
    cell_out.volume = volume(cell_out.lattice_vector, Direct);

    deallocate(position);
    deallocate(types_tmp);
}

auto System::compute_transmat_to_prim_using_spglib(const Cell &cell_input,
                                                   const double symprec) const -> Eigen::Matrix3d
{
    Cell cell_out;
    Spin spin_out, spin_in;
    find_primitive_cell(cell_input, spin_in, cell_out, spin_out, symprec);

    Eigen::Matrix3d transmat_out = cell_input.lattice_vector.inverse() * cell_out.lattice_vector;

    return transmat_out;
}


auto System::generate_coordinate_of_periodic_images() -> void
{
    // Generate Cartesian coordinates of atoms in the neighboring 27 supercells

    unsigned int i;
    int ia, ja, ka;

    const auto nat = supercell.number_of_atoms;
    const Eigen::MatrixXd xf_in = supercell.x_fractional;

    Eigen::MatrixXd x_tmp(nat, 3);

    auto icell = 0;

    x_tmp = xf_in * supercell.lattice_vector.transpose();

    x_image.emplace_back(x_tmp);

    // Convert to Cartesian coordinate
    for (ia = -1; ia <= 1; ++ia) {
        for (ja = -1; ja <= 1; ++ja) {
            for (ka = -1; ka <= 1; ++ka) {
                if (ia == 0 && ja == 0 && ka == 0) continue;
                ++icell;
                x_tmp = supercell.x_fractional;
                for (i = 0; i < nat; ++i) {
                    x_tmp(i, 0) += static_cast<double>(ia);
                    x_tmp(i, 1) += static_cast<double>(ja);
                    x_tmp(i, 2) += static_cast<double>(ka);
                }
                x_tmp = x_tmp * supercell.lattice_vector.transpose();
                x_image.emplace_back(x_tmp);
            }
        }
    }

    icell = 0;
    exist_image[0] = 1;

    for (ia = -1; ia <= 1; ++ia) {
        for (ja = -1; ja <= 1; ++ja) {
            for (ka = -1; ka <= 1; ++ka) {
                if (ia == 0 && ja == 0 && ka == 0) continue;
                ++icell;
                // When periodic flag is zero along an axis,
                // periodic images along that axis cannot be considered.
                if (((std::abs(ia) == 1) && (is_periodic[0] == 0)) || ((std::abs(ja) == 1) && (is_periodic[1] == 0)) ||
                    ((std::abs(ka) == 1) && (is_periodic[2] == 0)))
                {
                    exist_image[icell] = 0;
                } else {
                    exist_image[icell] = 1;
                }
            }
        }
    }
}


auto System::print_structure_stdout(const int verbosity) const -> void
{
    using namespace std;
    size_t i;

    cout << " ===================\n";
    cout << "  CRYSTAL STRUCTURE \n";
    cout << " ===================\n\n";

    cout.setf(ios::scientific);

    auto cell = get_inputcell();

    cout << "  ++++++++++++\n";
    cout << "   Input Cell \n";
    cout << "  ++++++++++++\n\n";

    cout << "   Lattice Vector (bohr)\n";
    cout << setw(16) << cell.lattice_vector(0, 0);
    cout << setw(15) << cell.lattice_vector(1, 0);
    cout << setw(15) << cell.lattice_vector(2, 0);
    cout << " : a1\n";
    cout << setw(16) << cell.lattice_vector(0, 1);
    cout << setw(15) << cell.lattice_vector(1, 1);
    cout << setw(15) << cell.lattice_vector(2, 1);
    cout << " : a2\n";
    cout << setw(16) << cell.lattice_vector(0, 2);
    cout << setw(15) << cell.lattice_vector(1, 2);
    cout << setw(15) << cell.lattice_vector(2, 2);
    cout << " : a3\n\n";
    cout << "   Number of atoms : " << cell.number_of_atoms << "\n\n";

    if (verbosity > 1) {
        cout << "   Cell volume = " << cell.volume << " (bohr^3)\n\n";
        cout << "   Reciprocal Lattice Vector (1/bohr)\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(0, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 2);
        cout << " : b1\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(1, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 2);
        cout << " : b2\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(2, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 2);
        cout << " : b3\n\n";
    }

    cout << "   Atomic species:\n";
    for (i = 0; i < cell.number_of_elems; ++i) {
        cout << setw(6) << i + 1 << setw(5) << kdname[i] << '\n';
    }

    if (verbosity > 1) {
        cout << '\n';
        cout << "   Atomic positions in fractional basis and atomic species\n";
        for (i = 0; i < cell.number_of_atoms; ++i) {
            cout << setw(6) << i + 1;
            cout << setw(15) << cell.x_fractional(i, 0);
            cout << setw(15) << cell.x_fractional(i, 1);
            cout << setw(15) << cell.x_fractional(i, 2);
            cout << setw(5) << kdname[cell.kind[i] - 1] << '\n';
        }
    }
    cout << "\n\n";

    cell = get_primcell();

    cout << "  ++++++++++++++++\n";
    cout << "   Primitive Cell \n";
    cout << "  ++++++++++++++++\n\n";

    cout << "   Lattice Vector (bohr)\n";
    cout << setw(16) << cell.lattice_vector(0, 0);
    cout << setw(15) << cell.lattice_vector(1, 0);
    cout << setw(15) << cell.lattice_vector(2, 0);
    cout << " : a1\n";
    cout << setw(16) << cell.lattice_vector(0, 1);
    cout << setw(15) << cell.lattice_vector(1, 1);
    cout << setw(15) << cell.lattice_vector(2, 1);
    cout << " : a2\n";
    cout << setw(16) << cell.lattice_vector(0, 2);
    cout << setw(15) << cell.lattice_vector(1, 2);
    cout << setw(15) << cell.lattice_vector(2, 2);
    cout << " : a3\n\n";

    const auto nat_prim = cell.number_of_atoms;

    cout << "   Number of atoms : " << cell.number_of_atoms << "\n\n";

    if (verbosity > 1) {
        cout << "   Cell volume = " << cell.volume << " (bohr^3)\n\n";
        cout << "   Reciprocal Lattice Vector (1/bohr)\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(0, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 2);
        cout << " : b1\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(1, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 2);
        cout << " : b2\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(2, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 2);
        cout << " : b3\n\n";
    }

    cout << "   Atomic positions in fractional basis and atomic species\n";
    for (i = 0; i < cell.number_of_atoms; ++i) {
        cout << setw(6) << i + 1;
        cout << setw(15) << cell.x_fractional(i, 0);
        cout << setw(15) << cell.x_fractional(i, 1);
        cout << setw(15) << cell.x_fractional(i, 2);
        cout << setw(5) << kdname[cell.kind[i] - 1] << '\n';
    }
    cout << "\n\n";

    cell = get_supercell();
    cout << "  ++++++++++++\n";
    cout << "   Super Cell \n";
    cout << "  ++++++++++++\n\n";

    cout << "   Lattice Vector (bohr)\n";
    cout << setw(16) << cell.lattice_vector(0, 0);
    cout << setw(15) << cell.lattice_vector(1, 0);
    cout << setw(15) << cell.lattice_vector(2, 0);
    cout << " : a1\n";
    cout << setw(16) << cell.lattice_vector(0, 1);
    cout << setw(15) << cell.lattice_vector(1, 1);
    cout << setw(15) << cell.lattice_vector(2, 1);
    cout << " : a2\n";
    cout << setw(16) << cell.lattice_vector(0, 2);
    cout << setw(15) << cell.lattice_vector(1, 2);
    cout << setw(15) << cell.lattice_vector(2, 2);
    cout << " : a3\n\n";

    cout << "   Number of atoms : " << cell.number_of_atoms << "\n\n";
    cout << "   Supercell contains " << std::setw(5) << cell.number_of_atoms / nat_prim << " primitive cells\n\n";

    if (verbosity > 1) {
        cout << "   Cell volume = " << cell.volume << " (bohr^3)\n\n";
        cout << "   Reciprocal Lattice Vector (1/bohr)\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(0, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(0, 2);
        cout << " : b1\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(1, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(1, 2);
        cout << " : b2\n";
        cout << setw(16) << cell.reciprocal_lattice_vector(2, 0);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 1);
        cout << setw(15) << cell.reciprocal_lattice_vector(2, 2);
        cout << " : b3\n\n";

        cout << "   Atomic positions in fractional basis and atomic species\n";
        for (i = 0; i < cell.number_of_atoms; ++i) {
            cout << setw(6) << i + 1;
            cout << setw(15) << cell.x_fractional(i, 0);
            cout << setw(15) << cell.x_fractional(i, 1);
            cout << setw(15) << cell.x_fractional(i, 2);
            cout << setw(5) << kdname[cell.kind[i] - 1] << '\n';
        }
    }
    cout << "\n\n";

    cout.unsetf(ios::scientific);
}


auto System::print_magmom_stdout() const -> void
{
    using namespace std;

    cout << " ====================\n";
    cout << "  MAGNETIC STRUCTURE \n";
    cout << " ====================\n\n";

    cout << "  MAGMOM is given.\n"
            "  The magnetic moments of each atom in the primitive cell are as follows:\n";
    for (size_t i = 0; i < primcell.number_of_atoms; ++i) {
        cout << setw(6) << i + 1;
        cout << setw(5) << spin_prim.magmom[i][0];
        cout << setw(5) << spin_prim.magmom[i][1];
        cout << setw(5) << spin_prim.magmom[i][2];
        cout << '\n';
    }
    cout << '\n';
    if (spin_prim.noncollinear == 0) {
        cout << "  NONCOLLINEAR = 0: magnetic moments are considered as scalar variables.\n";
    } else if (spin_prim.noncollinear == 1) {
        cout << "  NONCOLLINEAR = 1: magnetic moments are considered as vector variables.\n";
        if (spin_prim.time_reversal_symm) {
            cout << "  TREVSYM = 1: Time-reversal symmetry will be considered for generating magnetic space group\n";
        } else {
            cout
                << "  TREVSYM = 0: Time-reversal symmetry will NOT be considered for generating magnetic space group\n";
        }
    }
    cout << "\n\n";
}
