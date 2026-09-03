/*
gruneisen.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "gruneisen.h"
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "elastic_tensor.h"
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
    strain_newfcs_given = false;
    strain_newfcs.setZero();
    sublattice_relax = 0;
}

void Gruneisen::deallocate_variables()
{
    gruneisen_bs.clear();
    gruneisen_dos.clear();
    gruneisen_tensor_bs.clear();
    gruneisen_tensor_dos.clear();
    delta_fc2.clear();
    delta_fc2_newfcs.clear();
    delta_fc3_newfcs.clear();
    delta_fc2_strain.clear();
}

void Gruneisen::setup()
{
    MPI_Bcast(&delta_a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_newfcs, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&strain_newfcs_given, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(strain_newfcs.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sublattice_relax, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (sublattice_relax && (gruneisen_mode > 0 || print_newfcs)) {
        const ElasticTensor elastic_tensor(*system);
        elastic_tensor.calc_sublattice_response(fcs_phonon->force_constant_with_cell[0], sublattice_response);
    }

    if (gruneisen_mode == 1) {
        prepare_delta_fcs(fcs_phonon->force_constant_with_cell[1], delta_fc2, Eigen::Matrix3d::Identity());
    }
    if (gruneisen_mode >= 2) {
        prepare_delta_fcs_strain(fcs_phonon->force_constant_with_cell[1], delta_fc2_strain);
    }

    if (print_newfcs) {
        const Eigen::Matrix3d strain_dir = strain_newfcs_given ? strain_newfcs : Eigen::Matrix3d::Identity();

        prepare_delta_fcs(fcs_phonon->force_constant_with_cell[1], delta_fc2_newfcs, strain_dir);
        if (anharmonic_core->quartic_mode > 0) {
            prepare_delta_fcs(fcs_phonon->force_constant_with_cell[2], delta_fc3_newfcs, strain_dir);
        }
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
            if (strain_newfcs_given) {
                std::cout << "              strained systems will be estimated\n";
                std::cout << "              with the strain tensor given in the &strain field:\n";
                for (auto i = 0; i < 3; ++i) {
                    std::cout << "              ";
                    for (auto j = 0; j < 3; ++j) {
                        std::cout << std::setw(12) << std::defaultfloat << strain_newfcs(i, j);
                    }
                    std::cout << '\n';
                }
            } else {
                std::cout << "              expanded/compressed systems will be estimated\n";
                std::cout << "              with the isotropic strain +/- " << std::defaultfloat << delta_a << ".\n";
                std::cout << "              An anisotropic strain tensor can be given in the &strain field.\n";
            }
            if (sublattice_relax) {
                std::cout << "              SUBLATTICE_RELAX = 1 : the strain-induced internal displacements\n";
                std::cout << "              are relaxed (relaxed-ion path).\n";
            }
        }
    }
}

void Gruneisen::calc_gruneisen()
{
    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << '\n';
        const std::string ion_path = sublattice_relax ? "relaxed-ion " : "";
        if (gruneisen_mode == 1) {
            std::cout << " GRUNEISEN = 1 : Calculating " << ion_path << "volumetric Gruneisen parameters ... ";
        } else {
            std::cout << " GRUNEISEN = " << gruneisen_mode << " : Calculating " << ion_path
                      << "generalized Gruneisen parameters ... ";
        }
    }

    if (kpoint->kpoint_bs.get()) {
        calc_gruneisen_at_kpoints(kpoint->kpoint_bs->nk,
                                  kpoint->kpoint_bs->xk,
                                  dynamical->dymat_band->get_eigenvalues(),
                                  dynamical->dymat_band->get_eigenvectors(),
                                  gruneisen_bs,
                                  gruneisen_tensor_bs);
    }

    if (dos->kmesh_dos.get()) {
        calc_gruneisen_at_kpoints(dos->kmesh_dos->nk,
                                  dos->kmesh_dos->xk,
                                  dos->dymat_dos->get_eigenvalues(),
                                  dos->dymat_dos->get_eigenvectors(),
                                  gruneisen_dos,
                                  gruneisen_tensor_dos);
    }

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "done!" << '\n';
    }
}

void Gruneisen::calc_gruneisen_at_kpoints(const unsigned int nk, const NDArray<double, 2> &xk,
                                          const double *const *eval, const std::complex<double> *const *const *evec,
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

void Gruneisen::prepare_delta_fcs(const std::vector<FcsArrayWithCell> &fcs_in, std::vector<FcsArrayWithCell> &delta_fcs,
                                  const Eigen::Matrix3d &strain_dir) const
{
    // Directional derivative of the order-n IFCs along the strain tensor
    // strain_dir: the identity gives the isotropic (volumetric) derivative.
    delta_fcs.clear();

    if (fcs_in.empty()) return;

    auto fcs_aligned = fcs_in;
    sort_by_heading_indices const operator1(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator1);

    DerivativeIFC::compute_dV_dstrain_real_space(fcs_aligned,
                                                 delta_fcs,
                                                 {strain_dir},
                                                 system->get_primcell().lattice_vector,
                                                 eps15);

    // With SUBLATTICE_RELAX = 1, append the internal-strain leg (contraction
    // of the tail with the strain-induced sublattice displacement), giving
    // the relaxed-ion derivative.
    if (sublattice_relax) {
        std::vector<FcsArrayWithCell> delta_sub;
        DerivativeIFC::compute_dV_dsublattice_real_space(fcs_aligned,
                                                         delta_sub,
                                                         build_sublattice_field(strain_dir),
                                                         eps15);
        delta_fcs.insert(delta_fcs.end(), delta_sub.begin(), delta_sub.end());
    }
}

Eigen::VectorXd Gruneisen::build_sublattice_field(const Eigen::Matrix3d &strain_dir) const
{
    // S_I = sum_{mu nu} X(I, mu nu) eta_{mu nu}
    Eigen::VectorXd S_field = Eigen::VectorXd::Zero(sublattice_response.rows());
    for (auto mu = 0; mu < 3; ++mu) {
        for (auto nu = 0; nu < 3; ++nu) {
            S_field += sublattice_response.col(mu * 3 + nu) * strain_dir(mu, nu);
        }
    }
    return S_field;
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
    DerivativeIFC::compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 1, system->get_primcell().lattice_vector);

    std::vector<std::vector<std::pair<std::size_t, double>>> components = {{{0, 1.0}}, {{4, 1.0}}, {{8, 1.0}}};
    if (gruneisen_mode == 3) {
        components.push_back({{5, 0.5}, {7, 0.5}}); // yz
        components.push_back({{2, 0.5}, {6, 0.5}}); // xz
        components.push_back({{1, 0.5}, {3, 0.5}}); // xy
    }

    delta_fcs_strain.resize(components.size());
    for (std::size_t icomp = 0; icomp < components.size(); ++icomp) {
        DerivativeIFC::extract_strain_combination(strain_groups, components[icomp], 1, eps15, delta_fcs_strain[icomp]);
    }

    // With SUBLATTICE_RELAX = 1, append the internal-strain leg per component.
    if (sublattice_relax) {
        for (std::size_t icomp = 0; icomp < components.size(); ++icomp) {
            Eigen::Matrix3d eta = Eigen::Matrix3d::Zero();
            for (const auto &term: components[icomp]) {
                eta(term.first / 3, term.first % 3) += term.second;
            }

            std::vector<FcsArrayWithCell> delta_sub;
            DerivativeIFC::compute_dV_dsublattice_real_space(fcs_aligned,
                                                             delta_sub,
                                                             build_sublattice_field(eta),
                                                             eps15);
            delta_fcs_strain[icomp].insert(delta_fcs_strain[icomp].end(), delta_sub.begin(), delta_sub.end());
        }
    }
}


void Gruneisen::write_new_fcsxml_all() const
{
    if (writes->getVerbosity() > 0) std::cout << '\n';

    if (fcs_phonon->update_fc2 || !fcs_phonon->file_fc3.empty()) {
        // The new force-constant files carry a single supercell structure, so
        // the harmonic and cubic terms must come from the same FCSFILE.
        warn("write_new_fcsxml_all", "NEWFCS = 1 cannot be combined with FC2FILE or FC3FILE.");
    } else {
        // FILE_FORMAT rule: h5 (default) writes the schema-stamped HDF5 pair,
        // text the legacy XML pair. Builds without HDF5 always fall back to XML.
#ifdef _HDF5
        const bool write_h5 = writes->use_h5_io;
#else
        const bool write_h5 = false;
#endif

        if (writes->getVerbosity() > 0) {
            if (write_h5) {
                std::cout << " NEWFCS = 1 : Following HDF5 files are created. \n";
            } else {
                std::cout << " NEWFCS = 1 : Following XML files are created. \n";
            }
        }

        // Without the &strain field: isotropic strain of +-delta_a (default 0.001).
        // With &strain: the user strain tensor is applied as +u and -u.
        const Eigen::Matrix3d strain_dir = strain_newfcs_given ? strain_newfcs : Eigen::Matrix3d::Identity();
        const auto scale = strain_newfcs_given ? 1.0 : delta_a;
        const std::string extension = write_h5 ? ".h5" : ".xml";

        // Relaxed-ion path: internal displacement per primitive atom under a
        // unit application of strain_dir (Cartesian bohr); zero when clamped.
        const auto natmin = system->get_primcell().number_of_atoms;
        Eigen::MatrixXd sub_disp = Eigen::MatrixXd::Zero(natmin, 3);
        if (sublattice_relax) {
            const Eigen::VectorXd S_field = build_sublattice_field(strain_dir);
            for (std::size_t kappa = 0; kappa < natmin; ++kappa) {
                for (auto i = 0; i < 3; ++i) {
                    sub_disp(kappa, i) = S_field(3 * kappa + i);
                }
            }
        }

        auto write_one = [&](const std::string &filename, const double scale_signed) {
#ifdef _HDF5
            if (write_h5) {
                writes->writeNewFcsH5(filename, delta_fc2_newfcs, delta_fc3_newfcs, strain_dir, scale_signed, sub_disp);
                return;
            }
#endif
            writes->writeNewFcsXml(filename, delta_fc2_newfcs, delta_fc3_newfcs, strain_dir, scale_signed, sub_disp);
        };

        auto file_out = phon->job_title + "_+" + extension;
        write_one(file_out, scale);

        if (writes->getVerbosity() > 0) {
            std::cout << "  " << std::setw(phon->job_title.length() + 12) << std::left << file_out;
            if (strain_newfcs_given) {
                std::cout << " : Force constants of the system with the strain +u applied\n";
            } else {
                std::cout << " : Force constants of the system expanded by " << std::fixed << std::setprecision(3)
                          << delta_a * 100 << " %\n";
            }
        }

        file_out = phon->job_title + "_-" + extension;
        write_one(file_out, -scale);

        if (writes->getVerbosity() > 0) {
            std::cout << "  " << std::setw(phon->job_title.length() + 12) << std::left << file_out;
            if (strain_newfcs_given) {
                std::cout << " : Force constants of the system with the strain -u applied\n";
            } else {
                std::cout << " : Force constants of the system compressed by " << std::fixed << std::setprecision(3)
                          << delta_a * 100 << " %\n";
            }
        }
    }
}
