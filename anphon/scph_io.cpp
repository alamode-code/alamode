/*
 scph_io.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Functions for writing SCPH outputs.
 This file currently contains SCPH-specific force-constant output routines.

 Functions included:
 - write_anharmonic_correction_fc2: Write anharmonic force constant corrections (legacy text)
 - build_scph_settings_h5 / build_scph_cells_h5 / build_fc2_rows_h5: assemble the
   plain-data payload of the unified PREFIX.scph.h5 / PREFIX.qha.h5 state file
 - write_scph_state_h5 / load_scph_state_h5: store/restore the unified state
*/

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "constants.h"
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "parsephon.h"
#include "scph_qha_common.h"
#include "scph_result_io.h"
#include "system.h"

using namespace PHON_NS;

void ScphQhaCommon::write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat, const unsigned int NT,
                                                    const KpointMeshUniform *kmesh_coarse_in,
                                                    MinimumDistList ***mindist_list_in, const bool is_qha,
                                                    const int type)
{
    // Output anharmonically-renormalized IFC to file

    unsigned int i, j;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    double ***delta_fc2;
    const auto ns = dynamical->neval;
    unsigned int is, js, icell;
    unsigned int iat, jat;

    std::string file_fc2;
    std::ofstream ofs_fc2;

    if (is_qha) {
        file_fc2 = input->job_title + ".qha_dfc2";
    } else {
        if (type == 0) {
            file_fc2 = input->job_title + ".scph_dfc2";
        } else if (type == 1) {
            file_fc2 = input->job_title + ".scph+bubble(0)_dfc2";
        } else if (type == 2) {
            file_fc2 = input->job_title + ".scph+bubble(w)_dfc2";
        } else if (type == 3) {
            file_fc2 = input->job_title + ".scph+bubble(wQP)_dfc2";
        }
    }

    ofs_fc2.open(file_fc2.c_str(), std::ios::out);
    if (!ofs_fc2) exit("write_anharmonic_correction_fc2", "Cannot open file_fc2");

    const auto ncell = kmesh_coarse_in->nk_i[0] * kmesh_coarse_in->nk_i[1] * kmesh_coarse_in->nk_i[2];

    allocate(delta_fc2, ns, ns, ncell);

    ofs_fc2.precision(10);
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << system->get_primcell().lattice_vector(j, i);
        }
        ofs_fc2 << '\n';
    }
    ofs_fc2 << std::setw(5) << system->get_primcell().number_of_atoms << std::setw(5)
            << system->get_primcell().number_of_elems << '\n';
    for (i = 0; i < system->get_primcell().number_of_elems; ++i) {
        ofs_fc2 << std::setw(5) << system->symbol_kd[i];
    }
    ofs_fc2 << '\n';

    for (i = 0; i < system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << system->get_primcell().x_fractional(i, j);
        }
        ofs_fc2 << std::setw(5) << system->get_primcell().kind[i] + 1 << '\n';
    }

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + dT * static_cast<double>(iT);

        ofs_fc2 << "# Temp = " << temp << '\n';

        for (is = 0; is < ns; ++is) {
            iat = is / 3;

            for (js = 0; js < ns; ++js) {
                jat = js / 3;

                for (icell = 0; icell < ncell; ++icell) {
                    delta_fc2[is][js][icell] = delta_dymat[iT][is][js][icell].real() *
                                               std::sqrt(system->get_mass_prim()[iat] * system->get_mass_prim()[jat]);
                }
            }
        }

        for (icell = 0; icell < ncell; ++icell) {

            for (is = 0; is < ns; ++is) {
                iat = is / 3;
                const auto icrd = is % 3;

                for (js = 0; js < ns; ++js) {
                    jat = js / 3;
                    const auto jcrd = js % 3;

                    const auto nmulti = mindist_list_in[iat][jat][icell].shift.size();

                    for (auto it = mindist_list_in[iat][jat][icell].shift.cbegin();
                         it != mindist_list_in[iat][jat][icell].shift.cend();
                         ++it)
                    {

                        ofs_fc2 << std::setw(4) << (*it).sx;
                        ofs_fc2 << std::setw(4) << (*it).sy;
                        ofs_fc2 << std::setw(4) << (*it).sz;
                        ofs_fc2 << std::setw(5) << iat << std::setw(3) << icrd;
                        ofs_fc2 << std::setw(4) << jat << std::setw(3) << jcrd;
                        ofs_fc2 << std::setprecision(15) << std::setw(25)
                                << delta_fc2[is][js][icell] / static_cast<double>(nmulti) << '\n';
                    }
                }
            }
        }

        ofs_fc2 << '\n';
    }

    deallocate(delta_fc2);

    ofs_fc2.close();
    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_fc2;

    if (is_qha) {
        std::cout << " : Anharmonic corrections to the second-order IFCs (QHA)\n";
    } else {
        if (type == 0) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH)\n";
        } else if (type == 1) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(0))\n";
        } else if (type == 2) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(w))\n";
        } else if (type == 3) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(wQP))\n";
        }
    }
}


ScphSettingsH5 ScphQhaCommon::build_scph_settings_h5(const std::string &mode_name, const unsigned int NT,
                                                     const unsigned int nonanalytic_in,
                                                     const bool selfenergy_offdiagonal_in,
                                                     const int relax_str_in) const
{
    ScphSettingsH5 settings;
    settings.mode = mode_name;
    for (auto i = 0; i < 3; ++i) {
        settings.kmesh_interpolate[i] = kmesh_coarse->nk_i[i];
        settings.kmesh_dense[i] = kmesh_dense->nk_i[i];
    }
    settings.temperatures.resize(NT);
    for (unsigned int i = 0; i < NT; ++i) {
        settings.temperatures[i] = system->Tmin + static_cast<double>(i) * system->dT;
    }
    settings.nonanalytic = static_cast<int>(nonanalytic_in);
    settings.selfenergy_offdiag = selfenergy_offdiagonal_in ? 1 : 0;
    settings.relax_str = relax_str_in;
    return settings;
}


ScphCellsH5 ScphQhaCommon::build_scph_cells_h5() const
{
    const auto &primcell = system->get_primcell();
    const auto &spin = system->get_spin_prim();

    ScphCellsH5 cells;
    cells.lavec_prim = primcell.lattice_vector;
    cells.xf_prim = primcell.x_fractional;
    cells.kinds = primcell.kind;
    cells.elements = system->symbol_kd;
    cells.masses_amu.resize(primcell.number_of_atoms);
    for (size_t i = 0; i < primcell.number_of_atoms; ++i) {
        cells.masses_amu[i] = system->get_mass_prim()[i] / amu_ry;
    }
    cells.spin_polarized = spin.lspin ? 1 : 0;
    cells.magmom = spin.magmom;
    cells.noncollinear = spin.noncollinear;
    cells.time_reversal_symmetry = spin.time_reversal_symm;
    for (auto i = 0; i < 3; ++i) {
        cells.ncell_grid[i] = kmesh_coarse->nk_i[i];
    }
    return cells;
}


ScphFc2RowsH5 ScphQhaCommon::build_fc2_rows_h5(const std::complex<double> *const *const *const *delta_dymat,
                                               const unsigned int NT, const KpointMeshUniform *kmesh_coarse_in,
                                               MinimumDistList ***mindist_list_in,
                                               const std::string &variant) const
{
    // Assemble the renormalized FC2 on the virtual supercell in the
    // alamode force-constant schema. The row enumeration is identical to
    // write_anharmonic_correction_fc2 above (one row per minimum-distance
    // image, multiplicity-split); the base harmonic values come from the
    // harmonic dynamical matrix sampled on the coarse mesh and transformed
    // to real space through the same pathway as the anharmonic correction,
    // so base and correction live on identical rows by construction.
    // The folding is exact when KMESH_INTERPOLATE matches the supercell
    // dimensions of the original FC2 (the standard setup); the imaginary
    // part is dropped, as in the legacy .scph_dfc2 file.
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto nk1 = kmesh_coarse_in->nk_i[0];
    const auto nk2 = kmesh_coarse_in->nk_i[1];
    const auto nk3 = kmesh_coarse_in->nk_i[2];
    const auto ncell = nk1 * nk2 * nk3;

    // Harmonic (short-range) dynamical matrix on the coarse mesh -> real space
    std::complex<double> **dymat_tmp = nullptr;
    std::complex<double> ***dymat_harm_q = nullptr;
    std::complex<double> ***dymat_harm_r = nullptr;
    allocate(dymat_tmp, ns, ns);
    allocate(dymat_harm_q, ns, ns, ncell);
    allocate(dymat_harm_r, ns, ns, ncell);

    for (unsigned int ik = 0; ik < ncell; ++ik) {
        dynamical->calc_analytic_k(kmesh_coarse_in->xk[ik], fcs_phonon->force_constant_with_cell[0], dymat_tmp);
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                dymat_harm_q[is][js][ik] = dymat_tmp[is][js];
            }
        }
    }
    Dynamical::fourier_dymat_k_to_r(nk1, nk2, nk3, ns, dymat_harm_q, dymat_harm_r);

    // Count rows first, then fill.
    size_t nrows = 0;
    for (unsigned int icell = 0; icell < ncell; ++icell) {
        for (unsigned int iat = 0; iat < natmin; ++iat) {
            for (unsigned int jat = 0; jat < natmin; ++jat) {
                nrows += 9 * mindist_list_in[iat][jat][icell].shift.size();
            }
        }
    }

    ScphFc2RowsH5 fc2;
    fc2.variant = variant;
    fc2.atom_indices.resize(nrows, 2);
    fc2.atom_indices_super.resize(nrows, 2);
    fc2.coord_indices.resize(nrows, 2);
    fc2.shift_vectors.resize(nrows, 3);
    fc2.base_values.resize(nrows);
    fc2.values_per_temperature.assign(static_cast<size_t>(NT) * nrows, 0.0);

    const auto &lavec = system->get_primcell().lattice_vector;
    const auto &xf = system->get_primcell().x_fractional;
    const Eigen::MatrixXd xc = (lavec * xf.transpose()).transpose(); // Cartesian positions

    size_t irow = 0;
    for (unsigned int icell = 0; icell < ncell; ++icell) {
        for (unsigned int is = 0; is < ns; ++is) {
            const auto iat = is / 3;
            const auto icrd = is % 3;
            for (unsigned int js = 0; js < ns; ++js) {
                const auto jat = js / 3;
                const auto jcrd = js % 3;

                const auto mass_factor =
                    std::sqrt(system->get_mass_prim()[iat] * system->get_mass_prim()[jat]);
                const auto &shifts = mindist_list_in[iat][jat][icell].shift;
                const auto nmulti = static_cast<double>(shifts.size());

                const auto base_val = dymat_harm_r[is][js][icell].real() * mass_factor / nmulti;

                for (const auto &shift: shifts) {
                    const auto cx = ((shift.sx % static_cast<int>(nk1)) + nk1) % nk1;
                    const auto cy = ((shift.sy % static_cast<int>(nk2)) + nk2) % nk2;
                    const auto cz = ((shift.sz % static_cast<int>(nk3)) + nk3) % nk3;
                    const auto icell2 = (cx * nk2 + cy) * nk3 + cz;

                    fc2.atom_indices(irow, 0) = static_cast<int>(iat);
                    fc2.atom_indices(irow, 1) = static_cast<int>(jat);
                    fc2.atom_indices_super(irow, 0) = static_cast<int>(iat);
                    fc2.atom_indices_super(irow, 1) = static_cast<int>(icell2 * natmin + jat);
                    fc2.coord_indices(irow, 0) = static_cast<int>(icrd);
                    fc2.coord_indices(irow, 1) = static_cast<int>(jcrd);

                    const Eigen::Vector3d tvec(static_cast<double>(shift.sx), static_cast<double>(shift.sy),
                                               static_cast<double>(shift.sz));
                    const Eigen::Vector3d relvec = lavec * tvec + xc.row(jat).transpose() - xc.row(iat).transpose();
                    for (auto k = 0; k < 3; ++k) fc2.shift_vectors(irow, k) = relvec[k];

                    fc2.base_values(irow) = base_val;
                    for (unsigned int iT = 0; iT < NT; ++iT) {
                        fc2.values_per_temperature[static_cast<size_t>(iT) * nrows + irow] =
                            base_val + delta_dymat[iT][is][js][icell].real() * mass_factor / nmulti;
                    }
                    ++irow;
                }
            }
        }
    }

    deallocate(dymat_tmp);
    deallocate(dymat_harm_q);
    deallocate(dymat_harm_r);

    return fc2;
}


void ScphQhaCommon::write_scph_state_h5(const std::string &filename, const std::string &mode_name,
                                        const unsigned int NT, const unsigned int nonanalytic_in,
                                        const bool selfenergy_offdiagonal_in, const int relax_str_in,
                                        const std::string &variant,
                                        const std::complex<double> *const *const *const *delta_main,
                                        const std::complex<double> *const *const *const *delta_harm_renorm,
                                        const std::vector<double> *v0,
                                        const KpointMeshUniform *kmesh_coarse_in,
                                        MinimumDistList ***mindist_list_in) const
{
    const auto settings =
        build_scph_settings_h5(mode_name, NT, nonanalytic_in, selfenergy_offdiagonal_in, relax_str_in);
    const auto cells = build_scph_cells_h5();
    const auto fc2 = build_fc2_rows_h5(delta_main, NT, kmesh_coarse_in, mindist_list_in, variant);

    // Empty convergence vectors mean "unknown" (legacy import) and are not
    // written; otherwise warn right away when something did not converge.
    const auto *conv_scph = converged_scph_temp.size() == NT ? &converged_scph_temp : nullptr;
    const auto *conv_str =
        (relax_str_in != 0 && converged_str_temp.size() == NT) ? &converged_str_temp : nullptr;

    const auto count_bad = [NT](const std::vector<unsigned char> *v) {
        if (!v) return 0U;
        unsigned int n = 0;
        for (unsigned int i = 0; i < NT; ++i) {
            if (!(*v)[i]) ++n;
        }
        return n;
    };
    const auto nbad = count_bad(conv_scph) + count_bad(conv_str);
    if (nbad > 0) {
        warn("write_scph_state_h5",
             "Some temperatures did not converge; they are flagged in /convergence of the state file\n"
             " and will be refused by later calculations unless ALLOW_UNCONVERGED = 1 is set.");
    }

    const ScphResultIOH5 io(filename);
    io.write_state(settings, cells, delta_main, delta_harm_renorm, v0, &fc2, conv_scph, conv_str);

    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << filename;
    std::cout << " : Unified " << mode_name
              << " state (restart file + temperature-dependent FC2)\n";
}


bool ScphQhaCommon::load_scph_state_h5(const std::string &filename, const std::string &mode_name,
                                       const unsigned int NT, const unsigned int nonanalytic_in,
                                       const bool selfenergy_offdiagonal_in, const int relax_str_in,
                                       std::complex<double> ****delta_main,
                                       std::complex<double> ****delta_harm_renorm, std::vector<double> *v0)
{
    const auto ns = dynamical->neval;

    int usable = 0;
    if (mympi->my_rank == 0) {
        usable = ScphResultIOH5(filename).is_restartable() ? 1 : 0;
    }
    MPI_Bcast(&usable, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!usable) return false;

    if (mympi->my_rank == 0) {
        const ScphResultIOH5 io(filename);
        const auto settings =
            build_scph_settings_h5(mode_name, NT, nonanalytic_in, selfenergy_offdiagonal_in, relax_str_in);
        io.validate_settings(settings);
        // The restarted data feed the postprocess (DOS, bands, thermo, ...):
        // refuse unconverged temperatures unless the user opted in.
        io.check_convergence(settings.temperatures, input->allow_unconverged);
        io.load_dymat("delta", settings.temperatures, ns, kmesh_coarse->nk, delta_main);
        if (delta_harm_renorm) {
            io.load_dymat("delta_harm_renorm", settings.temperatures, ns, kmesh_coarse->nk,
                          delta_harm_renorm);
        }
        if (v0) {
            io.load_v0(settings.temperatures, *v0);
        }
        std::cout << " done.\n";
    }

    mpi_bcast_complex(delta_main, NT, kmesh_coarse->nk, ns);
    if (delta_harm_renorm) {
        mpi_bcast_complex(delta_harm_renorm, NT, kmesh_coarse->nk, ns);
    }
    if (v0) {
        MPI_Bcast(v0->data(), static_cast<int>(NT), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return true;
}
