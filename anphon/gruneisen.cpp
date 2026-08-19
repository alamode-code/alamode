/*
gruneisen.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "gruneisen.h"
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <boost/version.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "anharmonic_core.h"
#include "cell_shift_table.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "ifc_derivative.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "pointers.h"
#include "system.h"
#include "version.h"
#include "write_phonons.h"

using namespace PHON_NS;

Gruneisen::Gruneisen(PHON *phon) : Pointers(phon)
{
    set_default_variables();
};

Gruneisen::~Gruneisen()
{
    deallocate_variables();
};

void Gruneisen::set_default_variables()
{
    delta_a = 0.01;
    gruneisen_mode = 0;
    print_newfcs = false;
}

void Gruneisen::deallocate_variables()
{
    gruneisen_bs.clear();
    gruneisen_dos.clear();
    gruneisen_tensor_bs.clear();
    gruneisen_tensor_dos.clear();
    xshift_s.clear();
    delta_fc2.clear();
    delta_fc3.clear();
    delta_fc2_strain.clear();
}

void Gruneisen::setup()
{
    MPI_Bcast(&delta_a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_newfcs, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    build_27cell_shift_table(xshift_s);

    if (gruneisen_mode == 1 || print_newfcs) {
        prepare_delta_fcs(fcs_phonon->force_constant_with_cell[1], delta_fc2);
    }
    if (gruneisen_mode >= 2) {
        prepare_delta_fcs_strain(fcs_phonon->force_constant_with_cell[1], delta_fc2_strain);
    }

    if (print_newfcs && anharmonic_core->quartic_mode > 0) {
        prepare_delta_fcs(fcs_phonon->force_constant_with_cell[2], delta_fc3);
    }
    if (gruneisen_mode == 1) {
        if (kpoint->kpoint_bs.get()) {
            gruneisen_bs.resize(kpoint->kpoint_bs->nk, dynamical->neval);
        }
        if (dos->kmesh_dos.get()) {
            gruneisen_dos.resize(dos->kmesh_dos->nk, dynamical->neval);
        }
    } else if (gruneisen_mode >= 2) {
        const auto ncomp = number_of_strain_components();
        if (kpoint->kpoint_bs.get()) {
            gruneisen_tensor_bs.resize(kpoint->kpoint_bs->nk, dynamical->neval, ncomp);
        }
        if (dos->kmesh_dos.get()) {
            gruneisen_tensor_dos.resize(dos->kmesh_dos->nk, dynamical->neval, ncomp);
        }
    }

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        if (print_newfcs) {
            std::cout << '\n';
            if (anharmonic_core->quartic_mode > 0) {
                std::cout << " NEWFCS = 1 : Harmonic and cubic force constants of \n";
            } else {
                std::cout << " NEWFCS = 1 : Harmonic force constants of \n";
            }
            std::cout << "              expanded/compressed systems will be estimated\n";
            std::cout << "              with DELTA_A = " << std::setw(5) << delta_a << '\n';
        }
    }
    //   print_stress_energy();
}

void Gruneisen::calc_gruneisen()
{
    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << '\n';
        if (gruneisen_mode == 1) {
            std::cout << " GRUNEISEN = 1 : Calculating volumetric Gruneisen parameters ... ";
        } else {
            std::cout << " GRUNEISEN = " << gruneisen_mode
                      << " : Calculating generalized Gruneisen parameters ... ";
        }
    }

    if (kpoint->kpoint_bs.get()) {
        calc_gruneisen_at_kpoints(kpoint->kpoint_bs->nk, kpoint->kpoint_bs->xk,
                                  dynamical->dymat_band->get_eigenvalues(),
                                  dynamical->dymat_band->get_eigenvectors(), gruneisen_bs, gruneisen_tensor_bs);
    }

    if (dos->kmesh_dos.get()) {
        calc_gruneisen_at_kpoints(dos->kmesh_dos->nk, dos->kmesh_dos->xk, dos->dymat_dos->get_eigenvalues(),
                                  dos->dymat_dos->get_eigenvectors(), gruneisen_dos, gruneisen_tensor_dos);
    }

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "done!" << '\n';
    }
}

void Gruneisen::calc_gruneisen_at_kpoints(const unsigned int nk, const NDArray<double, 2> &xk,
                                          const double *const *eval,
                                          const std::complex<double> *const *const *evec,
                                          NDArray<std::complex<double>, 2> &gamma_iso,
                                          NDArray<std::complex<double>, 3> &gamma_tensor) const
{
    // Mode-resolved strain derivative of the dynamical matrix projected onto the
    // harmonic eigenvectors. gruneisen_mode == 1: isotropic parameters
    // gamma = -<e|dD/deps_iso|e> / (6 omega^2) into gamma_iso. gruneisen_mode >= 2:
    // generalized parameters gamma_{mu nu} = -<e|dD/deps_{mu nu}|e> / (2 omega^2)
    // per strain component into gamma_tensor.
    const auto ns = dynamical->neval;

    NDArray<std::complex<double>, 2> dfc2_reciprocal(ns, ns);

    auto project = [&](const unsigned int ik, const unsigned int is) {
        auto gamma = std::complex<double>(0.0, 0.0);
        for (unsigned int i = 0; i < ns; ++i) {
            for (unsigned int j = 0; j < ns; ++j) {
                gamma += std::conj(evec[ik][is][i]) * dfc2_reciprocal[i][j] * evec[ik][is][j];
            }
        }
        if (std::abs(gamma.imag()) > eps10) {
            warn("calc_gruneisen", "Gruneisen parameter is not real");
        }
        return gamma;
    };

    if (gruneisen_mode == 1) {
        for (unsigned int ik = 0; ik < nk; ++ik) {
            dynamical->calc_analytic_k(xk[ik], delta_fc2, dfc2_reciprocal);

            for (unsigned int is = 0; is < ns; ++is) {
                const auto gamma = project(ik, is);
                if (std::abs(eval[ik][is]) < eps8) {
                    gamma_iso[ik][is] = 0.0;
                } else {
                    gamma_iso[ik][is] = gamma / (-6.0 * pow2(eval[ik][is]));
                }
            }
        }
    } else {
        const auto ncomp = delta_fc2_strain.size();

        for (std::size_t icomp = 0; icomp < ncomp; ++icomp) {
            for (unsigned int ik = 0; ik < nk; ++ik) {
                dynamical->calc_analytic_k(xk[ik], delta_fc2_strain[icomp], dfc2_reciprocal);

                for (unsigned int is = 0; is < ns; ++is) {
                    const auto gamma = project(ik, is);
                    if (std::abs(eval[ik][is]) < eps8) {
                        gamma_tensor[ik][is][icomp] = 0.0;
                    } else {
                        gamma_tensor[ik][is][icomp] = gamma / (-2.0 * pow2(eval[ik][is]));
                    }
                }
            }
        }
    }
}

void Gruneisen::prepare_delta_fcs(const std::vector<FcsArrayWithCell> &fcs_in,
                                  std::vector<FcsArrayWithCell> &delta_fcs) const
{
    // Contract the last leg of the order-n IFCs with its own position vector,
    // i.e., take the derivative along an isotropic strain (identity strain tensor).
    delta_fcs.clear();

    if (fcs_in.empty()) return;

    auto fcs_aligned = fcs_in;
    sort_by_heading_indices const operator1(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator1);

    DerivativeIFC::compute_dV_dstrain_real_space(fcs_aligned, delta_fcs, {Eigen::Matrix3d::Identity()},
                                                 system->get_primcell().lattice_vector, eps15);
}

void Gruneisen::prepare_delta_fcs_strain(const std::vector<FcsArrayWithCell> &fcs_in,
                                         std::vector<std::vector<FcsArrayWithCell>> &delta_fcs_strain) const
{
    // Strain-derivative IFCs per component for the generalized Gruneisen
    // parameters: diagonal components for gruneisen_mode = 2, the 6 Voigt
    // components (off-diagonals symmetrized over (mu,nu)) for gruneisen_mode = 3.
    // Component indices refer to the base-9 flattening mu*3+nu of
    // compute_dV_dumn_all_real_space.
    delta_fcs_strain.clear();

    if (fcs_in.empty()) return;

    auto fcs_aligned = fcs_in;
    sort_by_heading_indices const operator1(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator1);

    std::vector<DeltaFcsStrainComponents> strain_groups;
    DerivativeIFC::compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 1,
                                                  system->get_primcell().lattice_vector);

    std::vector<std::vector<std::pair<std::size_t, double>>> components = {{{0, 1.0}}, {{4, 1.0}}, {{8, 1.0}}};
    if (gruneisen_mode == 3) {
        components.push_back({{5, 0.5}, {7, 0.5}}); // yz
        components.push_back({{2, 0.5}, {6, 0.5}}); // xz
        components.push_back({{1, 0.5}, {3, 0.5}}); // xy
    }

    delta_fcs_strain.resize(components.size());
    for (std::size_t icomp = 0; icomp < components.size(); ++icomp) {
        DerivativeIFC::extract_strain_combination(strain_groups, components[icomp], 1, eps15,
                                                  delta_fcs_strain[icomp]);
    }
}


void Gruneisen::write_new_fcsxml_all() const
{
    if (writes->getVerbosity() > 0) std::cout << '\n';

    if (fcs_phonon->update_fc2) {
        warn("write_new_fcsxml_all", "NEWFCS = 1 cannot be combined with the FC2FILE.");
    } else {
        if (writes->getVerbosity() > 0) std::cout << " NEWFCS = 1 : Following XML files are created. \n";

        auto file_xml = phon->job_title + "_+.xml";
        write_new_fcsxml(file_xml, delta_a);

        if (writes->getVerbosity() > 0) {
            std::cout << "  " << std::setw(phon->job_title.length() + 12) << std::left << file_xml;
            std::cout << " : Force constants of the system expanded by " << std::fixed << std::setprecision(3)
                      << delta_a * 100 << " %\n";
        }

        file_xml = phon->job_title + "_-.xml";
        write_new_fcsxml(file_xml, -delta_a);

        if (writes->getVerbosity() > 0) {
            std::cout << "  " << std::setw(phon->job_title.length() + 12) << std::left << file_xml;
            std::cout << " : Force constants of the system compressed by " << std::fixed << std::setprecision(3)
                      << delta_a * 100 << " %\n";
        }
    }
}

void Gruneisen::write_new_fcsxml(const std::string &filename_xml, const double change_ratio_of_a) const
{
    int i, j;
    double lattice_vector[3][3];

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            lattice_vector[i][j] = (1.0 + change_ratio_of_a) * system->get_supercell(0).lattice_vector(i, j);
        }
    }

    using boost::property_tree::ptree;

    ptree pt;
    std::string str_pos[3];

    pt.put("Data.ANPHON_version", ALAMODE_VERSION);
    pt.put("Data.Description.OriginalXML", fcs_phonon->file_fcs);
    pt.put("Data.Description.Delta_A", double2string(change_ratio_of_a));

    pt.put("Data.Structure.NumberOfAtoms", system->get_supercell(0).number_of_atoms);
    pt.put("Data.Structure.NumberOfElements", system->get_primcell().number_of_elems);

    for (i = 0; i < system->get_primcell().number_of_elems; ++i) {
        ptree &child = pt.add("Data.Structure.AtomicElements.element", system->symbol_kd[i]);
        child.put("<xmlattr>.number", i + 1);
    }

    for (i = 0; i < 3; ++i) {
        str_pos[i].clear();
        for (j = 0; j < 3; ++j) {
            str_pos[i] += " " + double2string(lattice_vector[j][i]);
        }
    }
    pt.put("Data.Structure.LatticeVector", "");
    pt.put("Data.Structure.LatticeVector.a1", str_pos[0]);
    pt.put("Data.Structure.LatticeVector.a2", str_pos[1]);
    pt.put("Data.Structure.LatticeVector.a3", str_pos[2]);

    pt.put("Data.Structure.Position", "");
    std::string str_tmp;

    const auto cell_tmp = system->get_supercell(0);
    const auto map_tmp = system->get_map_p2s(0);
    const auto nat_prim_tmp = system->get_primcell().number_of_atoms;

    for (i = 0; i < cell_tmp.number_of_atoms; ++i) {
        str_tmp.clear();
        for (j = 0; j < 3; ++j) str_tmp += " " + double2string(cell_tmp.x_fractional(i, j));
        auto &child = pt.add("Data.Structure.Position.pos", str_tmp);
        child.put("<xmlattr>.index", i + 1);
        child.put("<xmlattr>.element", system->symbol_kd[cell_tmp.kind[i]]);
    }

    pt.put("Data.Symmetry.NumberOfTranslations", map_tmp[0].size());
    for (i = 0; i < map_tmp[0].size(); ++i) {
        for (j = 0; j < nat_prim_tmp; ++j) {
            auto &child = pt.add("Data.Symmetry.Translations.map", map_tmp[j][i] + 1);
            child.put("<xmlattr>.tran", i + 1);
            child.put("<xmlattr>.atom", j + 1);
        }
    }

    pt.put("Data.ForceConstants", "");
    str_tmp.clear();

    for (const auto &it: fcs_phonon->force_constant_with_cell[0]) {

        auto &child = pt.add("Data.ForceConstants.HARMONIC.FC2", double2string(it.fcs_val));

        child.put("<xmlattr>.pair1",
                  std::to_string(it.pairs[0].index / 3 + 1) + " " + std::to_string(it.pairs[0].index % 3 + 1));
        child.put("<xmlattr>.pair2",
                  std::to_string(map_tmp[it.pairs[1].index / 3][it.pairs[1].tran] + 1) + " " +
                      std::to_string(it.pairs[1].index % 3 + 1) + " " + std::to_string(it.pairs[1].cell_s + 1));
    }

    for (const auto &it: delta_fc2) {

        if (std::abs(it.fcs_val) < eps12) continue;

        auto &child = pt.add("Data.ForceConstants.HARMONIC.FC2", double2string(change_ratio_of_a * it.fcs_val));

        child.put("<xmlattr>.pair1",
                  std::to_string(it.pairs[0].index / 3 + 1) + " " + std::to_string(it.pairs[0].index % 3 + 1));
        child.put("<xmlattr>.pair2",
                  std::to_string(map_tmp[it.pairs[1].index / 3][it.pairs[1].tran] + 1) + " " +
                      std::to_string(it.pairs[1].index % 3 + 1) + " " + std::to_string(it.pairs[1].cell_s + 1));
    }

    if (anharmonic_core->quartic_mode) {
        for (const auto &it: fcs_phonon->force_constant_with_cell[1]) {

            if (it.pairs[1].index > it.pairs[2].index) continue;

            auto &child = pt.add("Data.ForceConstants.ANHARM3.FC3", double2string(it.fcs_val));

            child.put("<xmlattr>.pair1",
                      std::to_string(it.pairs[0].index / 3 + 1) + " " + std::to_string(it.pairs[0].index % 3 + 1));
            child.put("<xmlattr>.pair2",
                      std::to_string(map_tmp[it.pairs[1].index / 3][it.pairs[1].tran] + 1) + " " +
                          std::to_string(it.pairs[1].index % 3 + 1) + " " + std::to_string(it.pairs[1].cell_s + 1));
            child.put("<xmlattr>.pair3",
                      std::to_string(map_tmp[it.pairs[2].index / 3][it.pairs[2].tran] + 1) + " " +
                          std::to_string(it.pairs[2].index % 3 + 1) + " " + std::to_string(it.pairs[2].cell_s + 1));
        }

        for (const auto &it: delta_fc3) {

            if (std::abs(it.fcs_val) < eps12) continue;

            if (it.pairs[1].index > it.pairs[2].index) continue;

            auto &child = pt.add("Data.ForceConstants.ANHARM3.FC3", double2string(change_ratio_of_a * it.fcs_val));

            child.put("<xmlattr>.pair1",
                      std::to_string(it.pairs[0].index / 3 + 1) + " " + std::to_string(it.pairs[0].index % 3 + 1));
            child.put("<xmlattr>.pair2",
                      std::to_string(map_tmp[it.pairs[1].index / 3][it.pairs[1].tran] + 1) + " " +
                          std::to_string(it.pairs[1].index % 3 + 1) + " " + std::to_string(it.pairs[1].cell_s + 1));
            child.put("<xmlattr>.pair3",
                      std::to_string(map_tmp[it.pairs[2].index / 3][it.pairs[2].tran] + 1) + " " +
                          std::to_string(it.pairs[2].index % 3 + 1) + " " + std::to_string(it.pairs[2].cell_s + 1));
        }
    }

    using namespace boost::property_tree::xml_parser;
    constexpr auto indent = 2;

#if BOOST_VERSION >= 105600
    write_xml(filename_xml,
              pt,
              std::locale(),
              xml_writer_make_settings<ptree::key_type>(' ', indent, widen<std::string>("utf-8")));
#else
    write_xml(filename_xml, pt, std::locale(), xml_writer_make_settings(' ', indent, widen<char>("utf-8")));
#endif
}

auto Gruneisen::double2string(const double d) -> std::string
{
    std::string rt;
    std::stringstream ss;

    ss << std::scientific << std::setprecision(15) << d;
    ss >> rt;
    return rt;
}


// double Gruneisen::calc_stress_energy2(const std::vector<FcsArrayWithCell> fcs_in)
// {
//     unsigned int i, j;
//     double ret = 0.0;
//     double **vec, **pos;
//     double tmp, tmp2;
//     double xshift[3];
//     unsigned int itran;
//     unsigned int norder = fcs_in[0].pairs.size();
//
//     allocate(vec, norder, 3);
//     allocate(pos, norder, 3);
//
//     for (std::vector<FcsArrayWithCell>::const_iterator it = fcs_in.begin(); it != fcs_in.end(); ++it) {
//
//         for (i = 0; i < norder; ++i) {
//             for (j = 0; j < 3; ++j) {
//                 vec[i][j] = system->get_supercell(0).x_fractional[system->map_trueprim_to_super[(*it).pairs[i].index / 3][(*it).pairs[i].tran]][j]
//                 + xshift_s[(*it).pairs[i].cell_s][j];
//
//                 pos[i][j] = system->get_supercell(0).x_fractional[system->map_trueprim_to_super[(*it).pairs[i].index / 3][0]][j];
//             //    vec[i][j] = system->get_supercell(0).x_fractional[system->map_trueprim_to_super[0][(*it).pairs[i].tran]][j] + xshift_s[(*it).pairs[i].cell_s][j];
//             }
//             rotvec(vec[i], vec[i], system->lavec_s);
//             rotvec(pos[i], pos[i], system->lavec_s);
//         }
//
//
//         ret += (*it).fcs_val
//             * (vec[1][(*it).pairs[0].index % 3] - pos[0][(*it).pairs[0].index % 3])
//             * (vec[1][(*it).pairs[1].index % 3] - pos[0][(*it).pairs[1].index % 3]);
//     }
//
//     deallocate(vec);
//     deallocate(pos);
//     return ret;
// }
//
// void Gruneisen::calc_stress_energy3(const std::vector<FcsArrayWithCell> fcs_in, double ****ret)
// {
//     unsigned int i, j, k, l;
//     double **vec, **pos;
//     double tmp, tmp2;
//     double xshift[3];
//     unsigned int itran;
//     unsigned int norder = fcs_in[0].pairs.size();
//     unsigned int crd[4];
//
//     allocate(vec, norder, 3);
//     allocate(pos, norder, 3);
//
//     for (i = 0; i < 3; ++i) {
//         for (j = 0; j < 3; ++j) {
//             for (k = 0; k < 3; ++k) {
//                 for (l = 0; l < 3; ++l) {
//                     ret[i][j][k][l] = 0.0;
//                 }
//             }
//         }
//     }
//
//     for (std::vector<FcsArrayWithCell>::const_iterator it = fcs_in.begin(); it != fcs_in.end(); ++it) {
//
//         for (i = 0; i < norder; ++i) {
//             for (j = 0; j < 3; ++j) {
//                 vec[i][j] = system->get_supercell(0).x_fractional[system->map_trueprim_to_super[(*it).pairs[i].index / 3][(*it).pairs[i].tran]][j]
//                 + xshift_s[(*it).pairs[i].cell_s][j];
//
//                 pos[i][j] = system->get_supercell(0).x_fractional[system->map_trueprim_to_super[(*it).pairs[i].index / 3][0]][j];
//             }
//             rotvec(vec[i], vec[i], system->lavec_s);
//             rotvec(pos[i], pos[i], system->lavec_s);
//         }
//
//         crd[0] = (*it).pairs[0].index % 3;
//         crd[1] = (*it).pairs[1].index % 3;
//
//         for (k = 0; k < 3; ++k) {
//
//             crd[2] = k;
//
//             for (l = 0; l < 3; ++l) {
//
//                 crd[3] = l;
//
//                 ret[crd[0]][crd[1]][k][l] += (*it).fcs_val * (vec[1][k] - pos[0][k]) * (vec[1][l] - pos[0][l]);
//             }
//         }
//     }
//
//     deallocate(vec);
//     deallocate(pos);
//
//     for (i = 0; i < 3; ++i) {
//         for (j = 0; j < 3; ++j) {
//             for (k = 0; k < 3; ++k) {
//                 for (l = 0; l < 3; ++l) {
//                     ret[i][j][k][l] *= -0.5;
//                 }
//             }
//         }
//     }
// }
//
//
// void Gruneisen::print_stress_energy()
// {
//
//     double volume = system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30;
//
//
//     double ****A, ****C;
//
//     allocate(A, 3, 3, 3, 3);
//     allocate(C, 3, 3, 3, 3);
//
//     calc_stress_energy3(fcs_phonon->force_constant_with_cell[0], A);
//
//     unsigned int i, j, k, l;
//
//     std::cout << "# A [Ryd]" << '\n';
//
//     for (i = 0; i < 3; ++i) {
//         for (j = 0; j < 3; ++j) {
//             for (k = 0; k < 3; ++k) {
//                 for (l = 0; l < 3; ++l) {
//                     std::cout << std::setw(3) << i + 1;
//                     std::cout << std::setw(3) << j + 1;
//                     std::cout << std::setw(3) << k + 1;
//                     std::cout << std::setw(3) << l + 1;
//                     std::cout << std::setw(15) << std::fixed << A[i][j][k][l];
//                     std::cout << '\n';
//                 }
//             }
//         }
//     }
//
//     std::cout << '\n';
//     std::cout << "# C [GPa]" << '\n';
//
//     for (i = 0; i < 3; ++i) {
//         for (j = 0; j < 3; ++j) {
//             for (k = 0; k < 3; ++k) {
//                 for (l = 0; l < 3; ++l) {
//                     C[i][j][k][l] = A[i][k][j][l] + A[j][k][i][l] - A[i][j][k][l];
//                     C[i][j][k][l] *= 1.0e-9 * Ryd / volume;
//                     std::cout << std::setw(3) << i + 1;
//                     std::cout << std::setw(3) << j + 1;
//                     std::cout << std::setw(3) << k + 1;
//                     std::cout << std::setw(3) << l + 1;
//                     std::cout << std::setw(15) << std::fixed << C[i][j][k][l];
//                     std::cout << '\n';
//
//                 }
//             }
//         }
//     }
//
//     std::cout << "Bulk Modulus [GPa] = " << (C[0][0][0][0] + 2.0 * C[0][0][1][1]) / 3.0 << '\n';
//
//     deallocate(A);
//     deallocate(C);
// }
