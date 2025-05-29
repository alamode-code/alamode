/*
 system.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <Eigen/Core>

namespace PHON_NS
{

class Cell
{
public:
    // lattice_vector(i, j) : i(=x,y,z) component of j-th lattice vector in Bohr
    Eigen::Matrix3d lattice_vector;
    // reciprocal_lattice_vector(i, j) :
    // [j(=x,y,z) component of i-th reciprocal lattice vector in Bohr^-1] x 2pi
    Eigen::Matrix3d reciprocal_lattice_vector;
    double volume;
    size_t number_of_atoms;
    size_t number_of_elems;
    std::vector<int> kind;
    // x_fractional(i, j) : j-th component of i-th atom in Fractional coordinates
    Eigen::MatrixXd x_fractional;
    // x_cartesian(i, j) : j(=x,y,z) component of i-th atom in Cartesian coordinates in Bohr
    Eigen::MatrixXd x_cartesian;
    int has_entry{0};
};

class Spin
{
public:
    int lspin;
    int time_reversal_symm;
    int noncollinear;
    std::vector<std::vector<double>> magmom;
};

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

class Maps
{
public:
    unsigned int atom_num;
    unsigned int tran_num;
};

class MappingTable
{
public:
    std::vector<Maps> to_true_primitive;
    std::vector<std::vector<unsigned int>> from_true_primitive;
};

struct ShiftCell
{
public:
    int sx, sy, sz;
} __attribute__((aligned(16)));

struct MinimumDistList
{
public:
    double dist;
    std::vector<ShiftCell> shift;
} __attribute__((aligned(32)));

class System: protected Pointers
{
public:
    System(class PHON *);

    ~System();

    void setup();

    const Cell &get_supercell(const int index) const;

    const Cell &get_primcell(const bool distorted = false) const;

    const Spin &get_spin_super() const;

    const Spin &get_spin_prim() const;

    const MappingTable &get_mapping_super_alm(const int index) const;

    const MappingTable &get_mapping_prim_alm(const int index) const;

    const std::vector<Maps> &get_map_s2p(const int index = 0) const;

    const std::vector<std::vector<unsigned int>> &get_map_p2s(const int index = 0) const;


    Eigen::Matrix3d lavec_p_input;

    int load_primitive_from_file;

    std::vector<std::string> symbol_kd;
    std::vector<double> mass_kd;

    double Tmin, Tmax, dT;

    int get_atomic_number_by_name(const std::string &);

    const std::vector<double> &get_invsqrt_mass() const;

    const std::vector<double> &get_mass_prim() const;

    const std::vector<double> &get_mass_super() const;

    const std::vector<std::vector<unsigned int>> &get_atomtype_group(const bool distort = false) const;

    void get_minimum_distances(const unsigned int nsize[3],
                               MinimumDistList ***&mindist_list_out);

private:
    enum LatticeType
    {
        Direct, Reciprocal
    };

    std::vector<Cell> supercell;
    Cell primcell, primcell_distort;
    Spin spin_super, spin_prim;
    std::vector<MappingTable> map_super_alm, map_prim_alm;

    std::vector<std::vector<Maps>> map_s2p_new;
    std::vector<std::vector<std::vector<unsigned int>>> map_p2s_new;

    // concatenate atomic kind and magmom (only for collinear case)
    std::vector<std::vector<unsigned int>> atomtype_group_prim, atomtype_group_prim_distort;

    std::vector<double> mass_super, mass_prim;
    std::vector<double> invsqrt_mass_p;

    double tolerance_for_coordinates;

    void set_default_variables();

    void deallocate_variables();

    void set_mass_elem_from_database(const unsigned int,
                                     const std::vector<std::string> &,
                                     std::vector<double> &);

    static void set_atomtype_group(const Cell &cell_in,
                                   const Spin &spin_in,
                                   std::vector<std::vector<unsigned int>> &atomtype_group_out);

    void print_structure_information_stdout() const;

    void load_system_info_from_file();

    void get_structure_and_mapping_table_xml(const std::string &filename,
                                             Cell &scell_out,
                                             Cell &pcell_out,
                                             Spin &spin_super_out,
                                             Spin &spin_prim_out,
                                             MappingTable &map_super_out,
                                             MappingTable &map_prim_out,
                                             std::vector<std::string> &elements) const;

    void get_structure_and_mapping_table_h5(const std::string &filename,
                                            Cell &scell_out,
                                            Cell &pcell_out,
                                            Spin &spin_super_out,
                                            Spin &spin_prim_out,
                                            MappingTable &map_super_out,
                                            MappingTable &map_prim_out,
                                            std::vector<std::string> &elements) const;

    void update_primitive_lattice();

    void generate_mapping_tables();

    void initialize_distorted_primitive_cell();

    void generate_mapping_primitive_super(const Cell &pcell,
                                          const Cell &scell,
                                          std::vector<std::vector<unsigned int>> &map_p2s_out,
                                          std::vector<Maps> &map_s2p_out) const;

    static void recips(const Eigen::Matrix3d &mat_in,
                       Eigen::Matrix3d &rmat_out);

    //    void check_consistency_primitive_lattice() const;


    static double volume(const Eigen::Matrix3d &mat_in,
                         const LatticeType latttype_in);

    std::vector<std::string> element_names{
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu"
    };
    std::vector<double> atomic_masses{
        1.00794,
        4.002602,
        6.941,
        9.0121831,
        10.811,
        12.0107,
        14.0067,
        15.9994,
        18.998403163,
        20.1797,
        22.98976928,
        24.305,
        26.9815384,
        28.0855,
        30.973761998,
        32.065,
        35.453,
        39.948,
        39.0983,
        40.078,
        44.955908,
        47.867,
        50.9415,
        51.9961,
        54.938043,
        55.845,
        58.933194,
        58.6934,
        63.546,
        65.38,
        69.723,
        72.63,
        74.921595,
        78.971,
        79.904,
        83.798,
        85.4678,
        87.62,
        88.90584,
        91.224,
        92.90637,
        95.95,
        -1,
        101.07,
        102.90549,
        106.42,
        107.8682,
        112.414,
        114.818,
        118.71,
        121.76,
        127.6,
        126.90447,
        131.293,
        132.90545196,
        137.327,
        138.90547,
        140.116,
        140.90766,
        144.242,
        -1,
        150.36,
        151.964,
        157.25,
        158.925354,
        162.5,
        164.930328,
        167.259,
        168.934218,
        173.045,
        174.9668,
        178.486,
        180.94788,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.96657,
        200.592,
        204.3833,
        207.2,
        208.9804,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        232.0377,
        231.03588,
        238.02891,
        -1,
        -1
    }; // They are standard atomic weight recommended by CIAAW.
    // Some recent changes are not considered because of the presence of uncertainty interval.
    // For unstable elements, the atomic mass is set to -1.
};
}
