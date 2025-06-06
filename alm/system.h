/*
 system.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>
#include "timer.h"

namespace ALM_NS
{
class AtomType
{
public:
    int element;
    double magmom;

    bool operator<(const AtomType &a) const
    {
        if (this->element < a.element) {
            return true;
        }
        if (this->element == a.element) {
            return this->magmom < a.magmom;
        }
        return false;
    }
};

class Cell
{
public:
    Eigen::Matrix3d lattice_vector;
    Eigen::Matrix3d reciprocal_lattice_vector;
    double volume;
    size_t number_of_atoms;
    size_t number_of_elems;
    std::vector<int> kind;
    Eigen::MatrixXd x_fractional;
    Eigen::MatrixXd x_cartesian;
};

class Spin
{
public:
    bool lspin;
    int time_reversal_symm;
    int noncollinear;
    std::vector<std::vector<double>> magmom;
};

class System
{
public:
    System();

    ~System();

    void init(const int, std::unique_ptr<Timer> &timer);

    void set_basecell(const double lavec_in[3][3], const size_t nat_in, const int *kind_in, const double xf_in[][3]);

    void set_kdname(const std::vector<std::string> &kdname_in);

    void set_periodicity(const int[3]);

    void set_spin_variables(const size_t nat, const bool, const int, const int, const double (*)[3]);

    void set_str_magmom(std::string);

    void set_transformation_matrices(const double transmat_to_super_in[3][3], const double transmat_to_prim_in[3][3],
                                     const int autoset_primcell_in = 0);

    void set_tolerance(const double tolerance);

    [[nodiscard]] const Cell &get_supercell() const;

    [[nodiscard]] const Cell &get_primcell() const;

    [[nodiscard]] const Cell &get_inputcell() const;

    [[nodiscard]] const std::vector<Eigen::MatrixXd> &get_x_image() const;

    [[nodiscard]] int *get_exist_image() const;

    [[nodiscard]] const std::vector<std::string> &get_kdname() const;

    [[nodiscard]] int *get_periodicity() const;

    [[nodiscard]] const Spin &get_spin(const std::string &cell = "super") const;

    [[nodiscard]] const std::string &get_str_magmom() const;

    [[nodiscard]] const std::vector<std::vector<unsigned int>> &
    get_atomtype_group(const std::string &cell = "super") const;

    [[nodiscard]] Eigen::Matrix3d compute_transmat_to_prim_using_spglib(const Cell &cell_input,
                                                                        const double symprec) const;

private:
    // Variables for geometric structure
    Cell supercell, primcell, inputcell;

    // Transformation matrices
    Eigen::Matrix3d transmat_to_super, transmat_to_prim;
    Eigen::Matrix3d transmat_to_prim_spglib;

    int autoset_primcell;
    double symmetry_tolerance;

    std::vector<std::string> kdname;
    int *is_periodic; // is_periodic[3];
    std::vector<Eigen::MatrixXd> x_image;
    int *exist_image;

    // Variables for spins
    Spin spin_input, spin_super, spin_prim;
    std::string str_magmom;

    // concatenate atomic kind and magmom (only for collinear case)
    std::vector<std::vector<unsigned int>> atomtype_group_super, atomtype_group_prim;

    enum LatticeType
    {
        Direct,
        Reciprocal
    };

    void build_cells();

    void build_supercell();

    void build_primcell();

    void set_reciprocal_latt(const Eigen::Matrix3d &lavec_in, Eigen::Matrix3d &rlavec_out) const;

    void set_default_variables();

    void deallocate_variables();

    [[nodiscard]] double volume(const Eigen::Matrix3d &mat_in, const LatticeType latttype_in) const;

    static void set_atomtype_group(const Cell &cell_in, const Spin &spin_in,
                                   std::vector<std::vector<unsigned int>> &atomtype_group_out);

    void find_primitive_cell(const Cell &cell_input, const Spin &spin_input, Cell &cell_out, Spin &spin_out,
                             const double tolerance) const;

    void generate_coordinate_of_periodic_images();

    void print_structure_stdout(const int verbosity) const;

    void print_magmom_stdout() const;
};
} // namespace ALM_NS
