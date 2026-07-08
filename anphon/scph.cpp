/*
 scph.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dielec.h"
#include "diis.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "integration.h"
#include "interpolation.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "relaxation.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

Scph::Scph(PHON *phon) : ScphQhaCommon(phon)
{
    set_default_variables();
}

Scph::~Scph()
{
    deallocate_variables();
}

namespace PHON_NS
{
class ScphRelaxationModel final: public IRelaxationModel
{
public:
    ScphRelaxationModel(Scph &scph, StructuralOptWorkspace &ws, std::complex<double> ****dymat_anharm,
                        std::complex<double> ****delta_harmonic_dymat_renormalize, std::complex<double> ***cmat_convert,
                        double ***omega2_anharm, std::complex<double> ***evec_anharm_tmp, double ***omega2_harm_renorm,
                        std::complex<double> ***evec_harm_renorm_tmp, std::complex<double> *v1_SCP,
                        std::complex<double> *del_v0_del_umn_SCP, bool &converged_prev, int &str_diverged,
                        std::ofstream &fout_step_q0, std::ofstream &fout_step_u0, std::ofstream &fout_step_u_tensor) :
        scph_(scph), ws_(ws), dymat_anharm_(dymat_anharm),
        delta_harmonic_dymat_renormalize_(delta_harmonic_dymat_renormalize), cmat_convert_(cmat_convert),
        omega2_anharm_(omega2_anharm), evec_anharm_tmp_(evec_anharm_tmp), omega2_harm_renorm_(omega2_harm_renorm),
        evec_harm_renorm_tmp_(evec_harm_renorm_tmp), v1_SCP_(v1_SCP), del_v0_del_umn_SCP_(del_v0_del_umn_SCP),
        converged_prev_(converged_prev), str_diverged_(str_diverged), fout_step_q0_(fout_step_q0),
        fout_step_u0_(fout_step_u0), fout_step_u_tensor_(fout_step_u_tensor)
    {}

    void before_init_structure(const unsigned int iT, unsigned int, double, const bool converged_prev) override
    {
        const auto nk = scph_.kmesh_dense->nk;
        const auto ns = scph_.dynamical->neval;

        // Initialize phonon eigenvectors with harmonic values

        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                for (auto js = 0; js < ns; ++js) {
                    evec_anharm_tmp_[ik][is][js] = scph_.evec_harmonic[ik][is][js];
                }
            }
        }
        if (converged_prev) {
            if (scph_.lower_temp) {
                for (auto ik = 0; ik < nk; ++ik) {
                    for (auto is = 0; is < ns; ++is) {
                        omega2_anharm_[iT][ik][is] = omega2_anharm_[iT + 1][ik][is];
                    }
                }
            } else {
                for (auto ik = 0; ik < nk; ++ik) {
                    for (auto is = 0; is < ns; ++is) {
                        omega2_anharm_[iT][ik][is] = omega2_anharm_[iT - 1][ik][is];
                    }
                }
            }
        }
    }

    void after_init_structure(unsigned int, double) override
    {
        auto &structure_state = ws_.structure_state;

        // Forget the step history of the previous temperature so that the
        // backtracking rescue for unconverged SCP steps never undoes a step
        // taken at a different temperature.
        std::fill(structure_state.delta_q0.begin(), structure_state.delta_q0.end(), 0.0);
        structure_state.delta_umn.fill(0.0);
        initial_structure_state_this_temp_ = structure_state;
        converged_this_temp_ = false;
        n_scp_failures_ = 0;
    }

    StructOptStepStatus do_structure_step(const unsigned int iT, const double temp, const int i_str_loop,
                                          std::vector<StructOptStepRecord> &step_history) override
    {
        auto &structure_state = ws_.structure_state;
        auto &harm_optical_modes = ws_.harm_optical_modes;

        // recompute the strain- and q0-renormalized IFCs at the
        // current structure
        scph_.renormalize_ifcs_at_structure(ws_);

        // solve the SCP equation at this structure and compute the
        // SCP forces and stress from the converged solution
        bool scp_converged_step = false;
        scph_.solve_scp_and_compute_forces(ws_,
                                           iT,
                                           temp,
                                           cmat_convert_,
                                           converged_prev_,
                                           scp_converged_step,
                                           dymat_anharm_,
                                           omega2_anharm_,
                                           evec_anharm_tmp_,
                                           v1_SCP_,
                                           del_v0_del_umn_SCP_);

        // Print the structure the SCP equation was solved at, together with the
        // SCP stress (cell relaxation only) and the space group detected by spglib.
        std::cout << "\n Structure at this step";
        if (!scp_converged_step) std::cout << " (SCP NOT converged)";
        std::cout << " :\n";
        const auto spg_label = scph_.relaxation->print_structure_and_symmetry(
            structure_state,
            (ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell && scp_converged_step) ? del_v0_del_umn_SCP_
                                                                                            : nullptr);
        std::cout << '\n';

        if (!scp_converged_step) {
            // The forces and stress from an unconverged SCP solution are unreliable.
            // They are not passed to the optimizer; instead the structure is moved by
            // a rescue step built from the accepted (converged) data only.
            ++n_scp_failures_;
            std::cout << " Warning: the SCP equation did not converge at this structure.\n"
                      << " The SCP forces and stress are unreliable and are not passed to the optimizer.\n";

            if (n_scp_failures_ >= max_consecutive_scp_failures_) {
                std::cout << " The SCP equation failed " << n_scp_failures_
                          << " times in a row. Give up the structural optimization at this temperature.\n";
                step_history.push_back({false, 0.0, 0.0, -1.0, -1.0, spg_label});
                return StructOptStepStatus::Aborted;
            }

            scph_.relaxation->rescue_step_after_scp_failure(structure_state,
                                                            v1_SCP_,
                                                            harm_optical_modes,
                                                            scph_.omega2_harmonic,
                                                            scph_.evec_harmonic);
            const auto du0 = structure_state.du0;
            const auto du_tensor = structure_state.du_tensor;

            step_history.push_back({false, du0, du_tensor, -1.0, -1.0, spg_label});

            scph_.relaxation->write_stepresfile(structure_state,
                                                i_str_loop + 1,
                                                fout_step_q0_,
                                                fout_step_u0_,
                                                fout_step_u_tensor_);

            scph_.relaxation->check_str_divergence(str_diverged_, structure_state);

            if (str_diverged_) {
                converged_prev_ = false;
                std::cout << " The crystal structure diverged.";
                std::cout << " Break from the structure loop.\n";
                return StructOptStepStatus::Diverged;
            }

            std::cout << " du0 =" << std::scientific << std::setw(15) << std::setprecision(6) << du0 << " [Bohr]";
            std::cout << " du_tensor =" << std::scientific << std::setw(15) << std::setprecision(6) << du_tensor
                      << '\n';

            // Do not test convergence on this step: du0/du_tensor describe the rescue
            // step and the gradients are unreliable.
            return StructOptStepStatus::SolverFailedRetry;
        }
        n_scp_failures_ = 0;

        scph_.relaxation->update_cell_coordinate(structure_state,
                                                 v1_SCP_,
                                                 omega2_anharm_[iT],
                                                 del_v0_del_umn_SCP_,
                                                 ws_.C2_array,
                                                 cmat_convert_,
                                                 harm_optical_modes,
                                                 scph_.omega2_harmonic,
                                                 scph_.evec_harmonic);
        const auto du0 = structure_state.du0;
        const auto du_tensor = structure_state.du_tensor;

        scph_.relaxation->write_stepresfile(structure_state,
                                            i_str_loop + 1,
                                            fout_step_q0_,
                                            fout_step_u0_,
                                            fout_step_u_tensor_);

        scph_.relaxation->check_str_divergence(str_diverged_, structure_state);

        if (str_diverged_) {
            converged_prev_ = false;
            std::cout << " The crystal structure diverged.";
            std::cout << " Break from the structure loop.\n";
            step_history.push_back({true, du0, du_tensor, -1.0, -1.0, spg_label});
            return StructOptStepStatus::Diverged;
        }

        double grad_norm, cell_grad_norm;
        scph_.compute_and_print_step_gradients(ws_,
                                               v1_SCP_,
                                               del_v0_del_umn_SCP_,
                                               du0,
                                               du_tensor,
                                               spg_label,
                                               step_history,
                                               grad_norm,
                                               cell_grad_norm);

        const bool step_converged =
            (du0 < scph_.relaxation->coord_conv_tol && du_tensor < scph_.relaxation->cell_conv_tol);
        const bool force_converged =
            (scph_.relaxation->gradient_conv_tol <= 0.0) || (grad_norm < scph_.relaxation->gradient_conv_tol);
        const bool cell_force_converged = (ws_.relax_mode != RelaxationStrMode::CoordinatesAndCell) ||
                                          (scph_.relaxation->cell_gradient_conv_tol <= 0.0) ||
                                          (cell_grad_norm < scph_.relaxation->cell_gradient_conv_tol);

        if (step_converged && force_converged && cell_force_converged) {
            std::cout << "\n\n du0 is smaller than COORD_CONV_TOL = " << std::scientific << std::setw(15)
                      << std::setprecision(6) << scph_.relaxation->coord_conv_tol << '\n';
            if (ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                std::cout << " du_tensor is smaller than CELL_CONV_TOL = " << std::scientific << std::setw(15)
                          << std::setprecision(6) << scph_.relaxation->cell_conv_tol << '\n';
            }
            if (scph_.relaxation->gradient_conv_tol > 0.0) {
                std::cout << " |residual force| is smaller than GRADIENT_CONV_TOL = " << std::scientific
                          << std::setw(15) << std::setprecision(6) << scph_.relaxation->gradient_conv_tol << '\n';
            }
            if (ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell &&
                scph_.relaxation->cell_gradient_conv_tol > 0.0)
            {
                std::cout << " |residual stress| is smaller than CELL_GRADIENT_CONV_TOL = " << std::scientific
                          << std::setw(15) << std::setprecision(6) << scph_.relaxation->cell_gradient_conv_tol << '\n';
            }
            std::cout << " Structural optimization converged in " << i_str_loop + 1 << "-th loop.\n";
            std::cout << " break structural loop.\n\n";
            converged_this_temp_ = true;
            return StructOptStepStatus::Converged;
        }

        return StructOptStepStatus::Continue;
    }

    void after_structure_loop(unsigned int iT, const double temp, const int i_str_loop_exit,
                              bool &converged_this_temp) override
    {
        converged_this_temp = converged_this_temp_;

        bench_temp_.push_back(temp);
        // i_str_loop == max_str_iter only when the loop ran to completion without a break;
        // a converged or diverged break leaves i_str_loop at the 0-based loop index.
        bench_steps_.push_back(i_str_loop_exit >= scph_.relaxation->max_str_iter ? scph_.relaxation->max_str_iter
                                                                                 : i_str_loop_exit + 1);

        const auto final_structure_is_finite = structure_state_is_finite(ws_.structure_state);
        const auto accepted_this_temp = converged_this_temp && final_structure_is_finite;
        bench_converged_.push_back(accepted_this_temp);

        if (accepted_this_temp) {
            last_converged_structure_state_ = ws_.structure_state;
            last_converged_iT_ = iT;
            has_last_converged_structure_ = true;
        } else {
            converged_this_temp = false;
            converged_prev_ = false;
            if (!final_structure_is_finite) str_diverged_ = 1;

            std::cout << "\n Structural optimization at " << temp << " K did not converge";
            if (!final_structure_is_finite) std::cout << " and produced non-finite structural parameters";
            std::cout << ".\n";

            if (has_last_converged_structure_) {
                ws_.structure_state = last_converged_structure_state_;
                copy_temperature_result(iT, last_converged_iT_);
                std::cout << " The failed structure and SCP data are discarded; the last converged"
                          << " temperature point is kept as the restart state for the next temperature.\n";
            } else {
                ws_.structure_state = initial_structure_state_this_temp_;
                set_harmonic_temperature_result(iT);
                std::cout << " No converged temperature point is available yet; the initial structure and"
                          << " harmonic dynamical matrix are kept as the restart state.\n";
            }

            str_diverged_ = 0;
        }

        converged_this_temp_ = converged_this_temp;
    }

    void record_v0(const unsigned int iT) override
    {
        // record zero-th order term of PES
        if (converged_this_temp_) scph_.V0[iT] = ws_.v0_renorm;
    }

    void finalize_temperature(const unsigned int iT, double, const bool converged_this_temp,
                              bool &converged_prev) override
    {
        if (converged_this_temp) {
            // get renormalization of harmonic dymat
            scph_.dynamical->compute_renormalized_harmonic_frequency(omega2_harm_renorm_[iT],
                                                                     evec_harm_renorm_tmp_,
                                                                     ws_.delta_v2_renorm,
                                                                     scph_.omega2_harmonic,
                                                                     scph_.evec_harmonic,
                                                                     scph_.kmesh_coarse.get(),
                                                                     scph_.kmesh_dense.get(),
                                                                     scph_.kmap_coarse_to_dense,
                                                                     scph_.mat_transform_sym,
                                                                     scph_.mindist_list,
                                                                     scph_.writes->getVerbosity());

            scph_.dynamical->calc_new_dymat_with_evec(delta_harmonic_dymat_renormalize_[iT],
                                                      omega2_harm_renorm_[iT],
                                                      evec_harm_renorm_tmp_,
                                                      scph_.kmesh_coarse.get(),
                                                      scph_.kmap_coarse_to_dense);
        }

        scph_.converged_str_temp[iT] = converged_this_temp ? 1 : 0;
        converged_prev = scph_.warmstart_scph && converged_this_temp;
    }

    void print_run_summary() override
    {
        // ---- structural-optimization benchmark summary (step counts to convergence) ----
        std::cout << " ============ Structural-optimization benchmark summary ============\n";
        std::cout << "  RELAX_ALGO = " << scph_.relaxation->relax_algo;
        if (scph_.relaxation->relax_algo == 3) {
            std::cout << "   GDIIS_PLAIN = " << (scph_.relaxation->gdiis_control ? 0 : 1);
        }
        std::cout << '\n';
        std::cout << "  " << std::setw(15) << "Temp [K]" << std::setw(12) << "steps" << std::setw(12) << "converged"
                  << '\n';
        int bench_total = 0;
        for (std::size_t ib = 0; ib < bench_temp_.size(); ++ib) {
            std::cout << "  " << std::setw(15) << std::fixed << std::setprecision(2) << bench_temp_[ib] << std::setw(12)
                      << bench_steps_[ib] << std::setw(12) << (bench_converged_[ib] ? "yes" : "no") << '\n';
            bench_total += bench_steps_[ib];
        }
        std::cout << "  " << std::setw(15) << "Total" << std::setw(12) << bench_total << '\n';
        std::cout << " ===================================================================\n\n";
        std::cout.unsetf(std::ios::fixed);
    }

    bool history_has_scp_column() const override
    {
        return true;
    }

private:
    bool structure_state_is_finite(const RelaxationStructureState &state) const
    {
        for (const auto value: state.q0) {
            if (!std::isfinite(value)) return false;
        }
        for (const auto value: state.u0) {
            if (!std::isfinite(value)) return false;
        }
        for (const auto &row: state.u_tensor) {
            for (const auto value: row) {
                if (!std::isfinite(value)) return false;
            }
        }
        return true;
    }

    void copy_temperature_result(const unsigned int dst, const unsigned int src) const
    {
        const auto nk = scph_.kmesh_dense->nk;
        const auto nk_interpolate = scph_.kmesh_coarse->nk;
        const auto ns = scph_.dynamical->neval;

        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                omega2_anharm_[dst][ik][is] = omega2_anharm_[src][ik][is];
                omega2_harm_renorm_[dst][ik][is] = omega2_harm_renorm_[src][ik][is];
            }
        }
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                for (auto ik = 0; ik < nk_interpolate; ++ik) {
                    dymat_anharm_[dst][is][js][ik] = dymat_anharm_[src][is][js][ik];
                    delta_harmonic_dymat_renormalize_[dst][is][js][ik] =
                        delta_harmonic_dymat_renormalize_[src][is][js][ik];
                }
            }
        }
        scph_.V0[dst] = scph_.V0[src];
    }

    void set_harmonic_temperature_result(const unsigned int dst) const
    {
        const auto nk = scph_.kmesh_dense->nk;
        const auto nk_interpolate = scph_.kmesh_coarse->nk;
        const auto ns = scph_.dynamical->neval;

        for (auto ik = 0; ik < nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                omega2_anharm_[dst][ik][is] = scph_.omega2_harmonic[ik][is];
                omega2_harm_renorm_[dst][ik][is] = scph_.omega2_harmonic[ik][is];
            }
        }
        scph_.dynamical->calc_new_dymat_with_evec(dymat_anharm_[dst],
                                                  omega2_anharm_[dst],
                                                  scph_.evec_harmonic,
                                                  scph_.kmesh_coarse.get(),
                                                  scph_.kmap_coarse_to_dense);
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                for (auto ik = 0; ik < nk_interpolate; ++ik) {
                    delta_harmonic_dymat_renormalize_[dst][is][js][ik] = 0.0;
                }
            }
        }
        scph_.V0[dst] = ws_.v0_ref;
    }

    Scph &scph_;
    StructuralOptWorkspace &ws_;
    std::complex<double> ****dymat_anharm_;
    std::complex<double> ****delta_harmonic_dymat_renormalize_;
    std::complex<double> ***cmat_convert_;
    double ***omega2_anharm_;
    std::complex<double> ***evec_anharm_tmp_;
    double ***omega2_harm_renorm_;
    std::complex<double> ***evec_harm_renorm_tmp_;
    std::complex<double> *v1_SCP_;
    std::complex<double> *del_v0_del_umn_SCP_;
    bool &converged_prev_;
    int &str_diverged_;
    std::ofstream &fout_step_q0_;
    std::ofstream &fout_step_u0_;
    std::ofstream &fout_step_u_tensor_;
    int n_scp_failures_ = 0;
    const int max_consecutive_scp_failures_ = 10;
    std::vector<double> bench_temp_;
    std::vector<int> bench_steps_;
    std::vector<bool> bench_converged_;
    RelaxationStructureState initial_structure_state_this_temp_;
    RelaxationStructureState last_converged_structure_state_;
    bool converged_this_temp_ = false;
    bool has_last_converged_structure_ = false;
    unsigned int last_converged_iT_ = 0;
};
} // namespace PHON_NS

void Scph::set_default_variables()
{
    restart_scph = false;
    warmstart_scph = false;
    lower_temp = true;
    tolerance_scph = 1.0e-10;
    mixalpha = 0.1;
    maxiter = 100;
    print_self_consistent_fc2 = false;
    selfenergy_offdiagonal = true;
    mix_anderson_ratio = 1.0;
    imix_scph = 1;

    bubble = 0;
    compute_Cv_anharmonic = 0;

    initialize_variables();
}


void Scph::setup_scph()
{
    MPI_Bcast(&bubble, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Prepare coarse/dense q-point meshes used in SCPH iterations and postprocess.
    setup_kmesh(kmesh_scph, kmesh_interpolate, "SCPH", "KMESH_INTERPOLATE should be a integral multiple of KMESH_SCPH");
    // Allocate and cache harmonic eigenvectors/eigenvalues on the coarse mesh.
    setup_eigvecs();

    // Precompute reciprocal-space anharmonic interactions (phi3/phi4).
    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);
    setup_pp_interaction(relax_mode != RelaxationStrMode::None || bubble > 0);
    // Build structural/symmetry data used for IFC reconstruction and matrix symmetrization.
    setup_structural_data();
}

void Scph::exec_scph()
{
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    NDArray<std::complex<double>, 4> delta_dymat_scph;
    NDArray<std::complex<double>, 4> delta_dymat_scph_plus_bubble;
    // change of harmonic dymat by IFC renormalization
    NDArray<std::complex<double>, 4> delta_harmonic_dymat_renormalize;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    MPI_Bcast(&restart_scph, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_h5_io, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&selfenergy_offdiagonal, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ialgo, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imix_scph, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    delta_dymat_scph.resize(NT, ns, ns, kmesh_coarse->nk);
    delta_harmonic_dymat_renormalize.resize(NT, ns, ns, kmesh_coarse->nk);

    zerofill_harmonic_dymat_renormalize(delta_harmonic_dymat_renormalize, NT);

    // Per-temperature convergence records, updated by the main loops below
    // and stored in the state file.
    converged_scph_temp.assign(NT, 1);
    converged_str_temp.assign(NT, 1);

    // Sized on every rank before the restart branch below, which broadcasts
    // loaded values into it.
    V0.assign(NT, 0.0);

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    if (relax_mode != RelaxationStrMode::None && thermodynamics->calc_FE_bubble) {
        exit("exec_scph", "Sorry, RELAX_STR!=0 can't be used with bubble correction of the free energy.");
    }
    if (relax_mode != RelaxationStrMode::None && bubble > 0) {
        exit("exec_scph", "Sorry, RELAX_STR!=0 can't be used with bubble self-energy on top of the SCPH calculation.");
    }

    if (restart_scph) {

        if (mympi->my_rank == 0) {
            std::cout << " RESTART_SCPH is true.\n";
            std::cout << " Dynamical matrix is read from file ...";
        }

        const auto with_relax = relax_mode != RelaxationStrMode::None;

        // Preferred restart source: the unified state file.
        auto loaded_h5 = false;
        if (use_h5_io) {
            loaded_h5 = load_scph_state_h5(phon->job_title + ".scph.h5",
                                           "SCPH",
                                           NT,
                                           dynamical->nonanalytic,
                                           selfenergy_offdiagonal,
                                           relaxation->relax_str,
                                           delta_dymat_scph,
                                           with_relax ? delta_harmonic_dymat_renormalize.ptr() : nullptr,
                                           with_relax ? &V0 : nullptr);
        }

        if (loaded_h5) {
            // Regenerate the human-readable V0-vs-T output, which may be
            // absent when restarting from the unified file alone.
            if (with_relax && mympi->my_rank == 0) store_V0_to_file();
        } else {
            // Read anharmonic correction to the dynamical matrix from the legacy text files.
            // Resume SCPH by loading previously saved anharmonic dynamical-matrix corrections.
            load_scph_dymat_from_file(delta_dymat_scph,
                                      phon->job_title + ".scph_dymat",
                                      kmesh_dense.get(),
                                      kmesh_coarse.get(),
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);

            if (with_relax) {
                // Resume harmonic-dynamical-matrix renormalization used in structural relaxation.
                load_scph_dymat_from_file(delta_harmonic_dymat_renormalize,
                                          phon->job_title + ".renorm_harm_dymat",
                                          kmesh_dense.get(),
                                          kmesh_coarse.get(),
                                          dynamical->nonanalytic,
                                          selfenergy_offdiagonal);
                // Load previously optimized static potential offset V0.
                load_V0_from_file();
            }

            // One-way migration of the legacy state into the unified file;
            // the text files themselves are left untouched.
            if (use_h5_io && mympi->my_rank == 0) {
                write_scph_state_h5(phon->job_title + ".scph.h5",
                                    "SCPH",
                                    NT,
                                    dynamical->nonanalytic,
                                    selfenergy_offdiagonal,
                                    relaxation->relax_str,
                                    "scph",
                                    delta_dymat_scph,
                                    with_relax ? delta_harmonic_dymat_renormalize.ptr() : nullptr,
                                    with_relax ? &V0 : nullptr,
                                    kmesh_coarse.get(),
                                    mindist_list);
            }
        }

    } else {
        if (relax_mode == RelaxationStrMode::None) {
            // Run standard SCPH fixed-point iteration.
            exec_scph_main(delta_dymat_scph);
        }
        // SCPH + structural optimization
        else
        {
            // Run coupled SCPH + cell/coordinate relaxation loop.
            exec_scph_relax_cell_coordinate_main(delta_dymat_scph, delta_harmonic_dymat_renormalize);
        }

        if (mympi->my_rank == 0) {
            const auto with_relax = relax_mode != RelaxationStrMode::None;
            if (use_h5_io) {
                // Persist the complete SCPH state (restart data + renormalized
                // FC2 per temperature) in one atomically-published file.
                write_scph_state_h5(phon->job_title + ".scph.h5",
                                    "SCPH",
                                    NT,
                                    dynamical->nonanalytic,
                                    selfenergy_offdiagonal,
                                    relaxation->relax_str,
                                    "scph",
                                    delta_dymat_scph,
                                    with_relax ? delta_harmonic_dymat_renormalize.ptr() : nullptr,
                                    with_relax ? &V0 : nullptr,
                                    kmesh_coarse.get(),
                                    mindist_list);
                // .V0 doubles as a human-readable physical output (V0 vs T),
                // so the text file is kept; restart reads only the h5.
                if (with_relax) store_V0_to_file();
            } else {
                // write dymat to file
                // write scph dynamical matrix when scph calculation is performed
                // Persist converged SCPH dynamical-matrix corrections for restart/reuse.
                store_renormalized_dymat_to_file(delta_dymat_scph,
                                                 phon->job_title + ".scph_dymat",
                                                 kmesh_dense.get(),
                                                 kmesh_coarse.get(),
                                                 dynamical->nonanalytic,
                                                 selfenergy_offdiagonal);
                // write renormalized harmonic dynamical matrix when the crystal structure is optimized
                if (with_relax) {
                    // Persist renormalized harmonic dynamical matrix and relaxation offset.
                    store_renormalized_dymat_to_file(delta_harmonic_dymat_renormalize,
                                                     phon->job_title + ".renorm_harm_dymat",
                                                     kmesh_dense.get(),
                                                     kmesh_coarse.get(),
                                                     dynamical->nonanalytic,
                                                     selfenergy_offdiagonal);
                    store_V0_to_file();
                }
            }
            // Convert dynamical-matrix correction back to real-space FC2 and
            // write the text .scph_dfc2 (kept in both modes during the
            // transition; consumed by dfc2.py / dfc2 / scph_to_qefc.py).
            write_anharmonic_correction_fc2(delta_dymat_scph, NT, kmesh_coarse.get(), mindist_list, false, 0);
        }
    }

    if (kpoint->kpoint_mode == 2) {
        if (thermodynamics->calc_FE_bubble) {
            // Evaluate bubble correction to free energy on the interpolation mesh.
            compute_free_energy_bubble_SCPH(kmesh_interpolate, delta_dymat_scph);
        }
    }

    if (bubble) {
        delta_dymat_scph_plus_bubble.resize(NT, ns, ns, kmesh_coarse->nk);
        // Add bubble self-energy to SCPH dynamical-matrix correction.
        bubble_correction(delta_dymat_scph, delta_dymat_scph_plus_bubble);
        if (mympi->my_rank == 0) {
            // Output FC2 after including bubble self-energy contribution.
            write_anharmonic_correction_fc2(delta_dymat_scph_plus_bubble,
                                            NT,
                                            kmesh_coarse.get(),
                                            mindist_list,
                                            false,
                                            bubble);
        }
    }

    postprocess(delta_dymat_scph,
                delta_harmonic_dymat_renormalize,
                delta_dymat_scph_plus_bubble,
                kmesh_coarse.get(),
                mindist_list,
                false,
                bubble);

    delta_dymat_scph.clear();
    delta_harmonic_dymat_renormalize.clear();
    delta_dymat_scph_plus_bubble.clear();
}


void Scph::exec_scph_main(std::complex<double> ****dymat_anharm)
{
    int ik, is;
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    // Compute matrix element of 4-phonon interaction

    NDArray<double, 3> omega2_anharm(NT, nk, ns);
    NDArray<std::complex<double>, 3> evec_anharm_tmp(nk, ns, ns);
    NDArray<std::complex<double>, 3> v3_array_all;
    NDArray<std::complex<double>, 3> v4_array_all(nk_irred_interpolate * nk,
                                                  static_cast<std::size_t>(ns) * ns,
                                                  static_cast<std::size_t>(ns) * ns);
    NDArray<std::complex<double>, 2> delta_v2_renorm;

    // delta_v2_renorm is zero when structural optimization is not performed
    delta_v2_renorm.resize(nk_interpolate, ns * ns);
    for (ik = 0; ik < nk_interpolate; ik++) {
        for (is = 0; is < ns * ns; is++) {
            delta_v2_renorm[ik][is] = 0.0;
        }
    }

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    // Calculate v4 array.
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal & (ialgo == 1)) {
        compute_V4_elements_mpi_over_band(v4_array_all,
                                          omega2_harmonic,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse.get(),
                                          kmesh_dense.get(),
                                          kmap_coarse_to_dense,
                                          phase_factor.get(),
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(v4_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            relax_mode != RelaxationStrMode::None,
                                            kmesh_coarse.get(),
                                            kmesh_dense.get(),
                                            kmap_coarse_to_dense,
                                            phase_factor.get(),
                                            phi4_reciprocal);
    }

    if (relax_mode != RelaxationStrMode::None) {
        v3_array_all.resize(nk, ns, ns * ns);

        compute_V3_elements_mpi_over_kpoint(v3_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            kmesh_coarse.get(),
                                            kmesh_dense.get(),
                                            phase_factor.get(),
                                            phi3_reciprocal);
    }

    if (mympi->my_rank == 0) {
        std::vector<double> vec_temp;
        NDArray<std::complex<double>, 3> cmat_convert;
        cmat_convert.resize(nk, ns, ns);

        vec_temp.clear();

        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);

        if (lower_temp) {
            for (int i = NT - 1; i >= 0; --i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        } else {
            for (int i = 0; i < NT; ++i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        }

        auto converged_prev = false;

        for (const double temp: vec_temp) {
            const auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            // Initialize phonon eigenvectors with harmonic values

            for (ik = 0; ik < nk; ++ik) {
                for (is = 0; is < ns; ++is) {
                    for (int js = 0; js < ns; ++js) {
                        evec_anharm_tmp[ik][is][js] = evec_harmonic[ik][is][js];
                    }
                }
            }

            if (converged_prev) {
                if (lower_temp) {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT + 1][ik][is];
                        }
                    }
                } else {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT - 1][ik][is];
                        }
                    }
                }
            }

            if (imix_scph == 1) {
                compute_anharmonic_frequency_diis(v4_array_all,
                                                  omega2_anharm[iT],
                                                  evec_anharm_tmp,
                                                  temp,
                                                  converged_prev,
                                                  cmat_convert,
                                                  selfenergy_offdiagonal,
                                                  delta_v2_renorm,
                                                  writes->getVerbosity());
            } else {
                compute_anharmonic_frequency(v4_array_all,
                                             omega2_anharm[iT],
                                             evec_anharm_tmp,
                                             temp,
                                             converged_prev,
                                             cmat_convert,
                                             selfenergy_offdiagonal,
                                             delta_v2_renorm,
                                             writes->getVerbosity());
            }

            converged_scph_temp[iT] = converged_prev ? 1 : 0;

            dynamical->calc_new_dymat_with_evec(dymat_anharm[iT],
                                                omega2_anharm[iT],
                                                evec_anharm_tmp,
                                                kmesh_coarse.get(),
                                                kmap_coarse_to_dense);

            if (!warmstart_scph) converged_prev = false;
        }

        cmat_convert.clear();
    }

    mpi_bcast_complex(dymat_anharm, NT, kmesh_coarse->nk, ns);

    omega2_anharm.clear();
    v4_array_all.clear();
    evec_anharm_tmp.clear();
    delta_v2_renorm.clear();
    if (relax_mode != RelaxationStrMode::None) {
        v3_array_all.clear();
    }
}


// relax internal coordinate and lattice
void Scph::exec_scph_relax_cell_coordinate_main(std::complex<double> ****dymat_anharm,
                                                std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    NDArray<double, 3> omega2_anharm;
    NDArray<std::complex<double>, 3> evec_anharm_tmp;

    // Scratch IFC/structure buffers grouped in the workspace consumed by the
    // loop-stage helpers; the names below are aliases into it.
    StructuralOptWorkspace ws;
    // renormalization of harmonic dynamical matrix
    auto &delta_v2_renorm = ws.delta_v2_renorm;
    auto &delta_v2_with_umn = ws.delta_v2_with_umn;
    NDArray<double, 3> omega2_harm_renorm;
    NDArray<std::complex<double>, 3> evec_harm_renorm_tmp;
    // k-space IFCs at the reference and updated structures
    auto &v1_ref = ws.v1_ref;
    auto &v1_renorm = ws.v1_renorm;
    auto &v3_ref = ws.v3_ref;
    auto &v3_renorm = ws.v3_renorm;
    auto &v3_with_umn = ws.v3_with_umn;
    auto &v4_ref = ws.v4_ref;
    auto &v1_with_umn = ws.v1_with_umn;
    auto &v0_ref = ws.v0_ref;
    v0_ref = 0.0; // set original ground state energy as zero

    // elastic constants
    auto &C1_array = ws.C1_array;
    auto &C2_array = ws.C2_array;
    auto &C3_array = ws.C3_array;

    // strain-derivative of k-space IFCs
    DelVStrainData del_v_strain;
    ws.del_v_strain = &del_v_strain;

    auto &del_v0_del_umn_renorm = ws.del_v0_del_umn_renorm;

    // atomic forces and stress tensor at finite temperatures
    NDArray<std::complex<double>, 1> v1_SCP;
    NDArray<std::complex<double>, 1> del_v0_del_umn_SCP;

    // cell optimization
    double pvcell = 0.0; // pressure * v_{cell,reference} [Ry]
    pvcell = relaxation->stat_pressure * system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) *
             1.0e-30;      // in 10^9 J = GJ
    pvcell *= 1.0e9 / Ryd; // in Ry
    ws.pvcell = pvcell;

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);
    ws.relax_mode = relax_mode;

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;


    omega2_anharm.resize(NT, nk, ns);
    evec_anharm_tmp.resize(nk, ns, ns);

    omega2_harm_renorm.resize(NT, nk, ns);
    evec_harm_renorm_tmp.resize(nk, ns, ns);

    // Common buffers, reference V3/V4 elements, strain derivatives of the
    // IFCs, optimizer, and Gamma-point optical modes (collective).
    setup_structural_opt_buffers(ws, eps10);

    auto &structure_state = ws.structure_state;

    v1_SCP.resize(ns);
    del_v0_del_umn_SCP.resize(9);

    // SCPH feeds the reference v4 directly into the q0 renormalization (its
    // strain renormalization is not available; see renormalize_ifcs_at_structure).
    ws.v4_for_renorm = v4_ref;

    if (mympi->my_rank == 0) {
        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);


        NDArray<std::complex<double>, 3> cmat_convert;
        cmat_convert.resize(nk, ns, ns);

        vec_temp.clear();

        if (lower_temp) {
            for (int i = NT - 1; i >= 0; --i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        } else {
            for (int i = 0; i < NT; ++i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        }

        auto converged_prev = false;
        auto str_diverged = 0;

        C1_array.resize(9);
        C2_array.resize(9, 9);
        C3_array.resize(9, 9, 9);

        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_step_q0, fout_step_u0;
        std::ofstream fout_q0, fout_u0;

        // cell optimization
        std::ofstream fout_step_u_tensor, fout_u_tensor;

        fout_step_q0.open("step_q0.txt");
        fout_step_u0.open("step_u0.txt");
        fout_q0.open(phon->job_title + ".normal_disp");
        fout_u0.open(phon->job_title + ".atom_disp");

        // if the unit cell is relaxed
        if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            fout_step_u_tensor.open("step_u_tensor.txt");
            fout_u_tensor.open(phon->job_title + ".umn_tensor");
        }

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        std::cout << " Start structural optimization.\n";

        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
            std::cout << "  Internal coordinates are relaxed.\n";
            std::cout << "  Shape of the unit cell is fixed.\n\n";
        } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            std::cout << "  Internal coordinates and shape of the unit cell are relaxed.\n\n";
        }

        ScphRelaxationModel model(*this,
                                  ws,
                                  dymat_anharm,
                                  delta_harmonic_dymat_renormalize,
                                  cmat_convert,
                                  omega2_anharm,
                                  evec_anharm_tmp,
                                  omega2_harm_renorm,
                                  evec_harm_renorm_tmp,
                                  v1_SCP,
                                  del_v0_del_umn_SCP,
                                  converged_prev,
                                  str_diverged,
                                  fout_step_q0,
                                  fout_step_u0,
                                  fout_step_u_tensor);
        StructuralOptLoopContext loop_ctx{structure_state,
                                          ws,
                                          fout_step_q0,
                                          fout_step_u0,
                                          fout_step_u_tensor,
                                          fout_q0,
                                          fout_u0,
                                          fout_u_tensor,
                                          vec_temp,
                                          Tmin,
                                          dT,
                                          NT,
                                          relax_mode,
                                          converged_prev,
                                          str_diverged};
        run_structural_optimization_loop(model, loop_ctx);

        // output files of structural optimization
        fout_step_q0.close();
        fout_step_u0.close();
        fout_q0.close();
        fout_u0.close();

        if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            fout_step_u_tensor.close();
            fout_u_tensor.close();
        }

        cmat_convert.clear();

        C1_array.clear();
        C2_array.clear();
        C3_array.clear();
    }

    mpi_bcast_complex(dymat_anharm, NT, kmesh_coarse->nk, ns);
    mpi_bcast_complex(delta_harmonic_dymat_renormalize, NT, kmesh_coarse->nk, ns);

    omega2_anharm.clear();
    evec_anharm_tmp.clear();
    delta_v2_renorm.clear();
    delta_v2_with_umn.clear();

    omega2_harm_renorm.clear();
    evec_harm_renorm_tmp.clear();

    v1_ref.clear();
    v1_with_umn.clear();
    v1_renorm.clear();
    v3_ref.clear();
    v3_renorm.clear();
    v3_with_umn.clear();
    v4_ref.clear();
    del_v0_del_umn_renorm.clear();
    v1_SCP.clear();
    del_v0_del_umn_SCP.clear();
}

void Scph::solve_scp_and_compute_forces(StructuralOptWorkspace &ws, const unsigned int iT, const double temp,
                                        std::complex<double> ***cmat_convert, bool &converged_prev,
                                        bool &scp_converged_step, std::complex<double> ****dymat_anharm,
                                        double ***omega2_anharm, std::complex<double> ***evec_anharm_tmp,
                                        std::complex<double> *v1_SCP, std::complex<double> *del_v0_del_umn_SCP)
{
    // solve SCP equation
    if (imix_scph == 1) {
        compute_anharmonic_frequency_diis(ws.v4_ref,
                                          omega2_anharm[iT],
                                          evec_anharm_tmp,
                                          temp,
                                          converged_prev,
                                          cmat_convert,
                                          selfenergy_offdiagonal,
                                          ws.delta_v2_renorm,
                                          writes->getVerbosity(),
                                          true);
    } else {
        compute_anharmonic_frequency(ws.v4_ref,
                                     omega2_anharm[iT],
                                     evec_anharm_tmp,
                                     temp,
                                     converged_prev,
                                     cmat_convert,
                                     selfenergy_offdiagonal,
                                     ws.delta_v2_renorm,
                                     writes->getVerbosity(),
                                     true);
    }

    // SCP convergence of this structure step. converged_prev is reused as the
    // warm-start flag of the next SCP solve, so keep a snapshot here.
    scp_converged_step = converged_prev;
    converged_scph_temp[iT] = scp_converged_step ? 1 : 0;

    dynamical->calc_new_dymat_with_evec(dymat_anharm[iT],
                                        omega2_anharm[iT],
                                        evec_anharm_tmp,
                                        kmesh_coarse.get(),
                                        kmap_coarse_to_dense);

    // calculate SCP force
    compute_anharmonic_v1_array(v1_SCP,
                                ws.v1_renorm,
                                ws.v3_renorm,
                                cmat_convert,
                                omega2_anharm[iT],
                                temp,
                                kmesh_dense.get());

    // calculate SCP stress tensor
    if (ws.relax_mode == RelaxationStrMode::CoordinatesOnly) {
        for (auto i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_SCP[i1] = 0.0;
        }
    } else if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        compute_anharmonic_del_v0_del_umn(del_v0_del_umn_SCP,
                                          ws.del_v0_del_umn_renorm,
                                          *ws.del_v_strain,
                                          ws.structure_state.u_tensor,
                                          ws.structure_state.q0,
                                          cmat_convert,
                                          omega2_anharm[iT],
                                          temp,
                                          kmesh_dense.get());
    }
}


//void Scph::calculate_force_in_real_space(const std::complex<double> *const v1_renorm,
//                                         double *force_array)
//{
//    int natmin = system->natmin;
//    auto ns = dynamical->neval;
//    int is, iatm, ixyz;
//    double force[3] = {0.0, 0.0, 0.0};
//
//    for (iatm = 0; iatm < natmin; iatm++) {
//        for (ixyz = 0; ixyz < 3; ixyz++) {
//            force[ixyz] = 0.0;
//            for (is = 0; is < ns; is++) {
//                force[ixyz] -= evec_harmonic[0][is][iatm * 3 + ixyz].real() *
//                               std::sqrt(system->mass[system->map_p2s[iatm][0]]) * v1_renorm[is].real();
//            }
//            force_array[iatm * 3 + ixyz] = force[ixyz];
//        }
//    }
//}


void Scph::initialize_scph_iteration(const double temp, const bool flag_converged, double **omega2_prev,
                                     const unsigned int verbosity, Eigen::MatrixXd &omega_now,
                                     Eigen::MatrixXd &omega2_HA, std::vector<Eigen::MatrixXcd> &evec_initial,
                                     std::vector<Eigen::MatrixXcd> &evec_initial_adjoint,
                                     std::complex<double> ***cmat_convert) const
{
    using namespace Eigen;
    constexpr auto complex_one = std::complex<double>(1.0, 0.0);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    // Set initial values
    for (unsigned int ik = 0; ik < nk; ++ik) {
        for (unsigned int is = 0; is < ns; ++is) {

            if (flag_converged) {
                if (omega2_prev[ik][is] < 0.0 && std::abs(omega2_prev[ik][is]) > 1.0e-16 && verbosity > 0) {
                    std::cout << "Warning : Large negative frequency detected\n";
                }
                omega_now(ik, is) = std::sqrt(std::abs(omega2_prev[ik][is]));
            } else {
                omega_now(ik, is) = std::sqrt(std::abs(omega2_harmonic[ik][is]));
            }

            omega2_HA(ik, is) = omega2_harmonic[ik][is];

            for (unsigned int js = 0; js < ns; ++js) {
                // transpose evec so that evec_initial can be used as is.
                evec_initial[ik](js, is) = evec_harmonic[ik][is][js];

                if (!flag_converged) {
                    // Initialize Cmat with identity matrix
                    if (is == js) {
                        cmat_convert[ik][is][js] = complex_one;
                    } else {
                        cmat_convert[ik][is][js] = complex_zero;
                    }
                }
            }
        }
        // Compute adjoint once after the entire matrix is filled
        evec_initial_adjoint[ik] = evec_initial[ik].adjoint();
    }
}

void Scph::setup_harmonic_dynamical_matrices(const Eigen::MatrixXd &omega2_HA,
                                             const std::vector<Eigen::MatrixXcd> &evec_initial,
                                             std::complex<double> **delta_v2_renorm,
                                             std::vector<Eigen::MatrixXcd> &Fmat0, std::complex<double> ***dymat_q_HA)
{
    using namespace Eigen;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    MatrixXcd Dymat(ns, ns);

    // Set initial harmonic dymat and eigenvalues
    for (unsigned int ik = 0; ik < nk_irred_interpolate; ++ik) {
        const auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        const auto knum = kmap_coarse_to_dense[knum_interpolate];

        Dymat = omega2_HA.row(knum).asDiagonal();
        auto evec_tmp = evec_initial[knum];

        // Harmonic dynamical matrix
        Dymat = evec_tmp * Dymat * evec_tmp.adjoint();
        symmetrize_dynamical_matrix(ik, kmesh_coarse.get(), ns, mat_transform_sym, Dymat);

        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                dymat_q_HA[is][js][knum_interpolate] = Dymat(is, js);
            }
        }

        // Harmonic Fmat
        Fmat0[ik] = omega2_HA.row(knum).asDiagonal();
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                Fmat0[ik](is, js) += delta_v2_renorm[knum_interpolate][is * ns + js];
            }
        }
    }

    replicate_dymat_for_all_kpoints(kmesh_coarse.get(), ns, mat_transform_sym, dymat_q_HA);
}

void Scph::compute_qmat_and_dmat(const Eigen::MatrixXd &omega_now, const double temp,
                                 std::complex<double> ***cmat_convert,
                                 std::vector<Eigen::MatrixXcd> &dmat_convert) const
{
    using namespace Eigen;

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

#pragma omp parallel
    {
        MatrixXcd Qmat(ns, ns);
        MatrixXcd Cmat(ns, ns);

#pragma omp for
        for (int ik = 0; ik < static_cast<int>(nk); ++ik) {
            Qmat.setZero();
            for (unsigned int is = 0; is < ns; ++is) {
                const auto omega1 = omega_now(ik, is);
                if (std::abs(omega1) > eps8) {
                    // Note that the missing factor 2 in the denominator of Qmat is
                    // already considered in the v4_array_all.
                    const auto factor = thermodynamics->disp_corr_factor(omega1, temp);
                    Qmat(is, is) = std::complex<double>(factor, 0.0);
                }
            }

            for (unsigned int is = 0; is < ns; ++is) {
                for (unsigned int js = 0; js < ns; ++js) {
                    Cmat(is, js) = cmat_convert[ik][is][js];
                }
            }

            dmat_convert[ik] = Cmat * Qmat * Cmat.adjoint();
        }
    }
}

void Scph::update_fmat_with_v4(const std::vector<Eigen::MatrixXcd> &Fmat0,
                               std::complex<double> *const *const *v4_array_all,
                               const std::vector<Eigen::MatrixXcd> &dmat_convert, const bool offdiag,
                               const unsigned int ik_irred, Eigen::MatrixXcd &Fmat) const
{
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;

    // Fmat harmonic
    Fmat = Fmat0[ik_irred];

    // Anharmonic correction to Fmat
    if (!offdiag) {
#pragma omp parallel for
        for (int is = 0; is < ns; ++is) {
            for (unsigned int jk = 0; jk < nk; ++jk) {
                for (unsigned int ks = 0; ks < ns; ++ks) {
                    Fmat(is, is) +=
                        v4_array_all[nk * ik_irred + jk][(ns + 1) * is][(ns + 1) * ks] * dmat_convert[jk](ks, ks);
                }
            }
        }
    } else {
        // Fmat is Hermitian and is consumed by SelfAdjointEigenSolver, which
        // references only the lower triangle, so only the elements with is >= js
        // are computed (half the work); the upper triangle is filled by conjugation.
        // The contraction over (jk, ks, ls) is evaluated as vectorized dot products
        // between the contiguous rows of v4_array_all and the D matrices flattened
        // in the matching (row-major) order.

        // dvec[jk * ns2 + ns * ks + ls] = dmat_convert[jk](ks, ls)
        VectorXcd dvec(static_cast<Index>(nk) * ns2);
#pragma omp parallel for
        for (int jk = 0; jk < static_cast<int>(nk); ++jk) {
            Map<MatrixXcd>(dvec.data() + static_cast<Index>(jk) * ns2, ns, ns) = dmat_convert[jk].transpose();
        }

        std::vector<int> ijs_lower;
        ijs_lower.reserve(ns * (ns + 1) / 2);
        for (int is = 0; is < ns; ++is) {
            for (int js = 0; js <= is; ++js) {
                ijs_lower.push_back(is * ns + js);
            }
        }

#pragma omp parallel for
        for (int ip = 0; ip < static_cast<int>(ijs_lower.size()); ++ip) {
            const auto ijs = ijs_lower[ip];
            std::complex<double> sum(0.0, 0.0);
            for (unsigned int jk = 0; jk < nk; ++jk) {
                sum += Map<const VectorXcd>(v4_array_all[nk * ik_irred + jk][ijs], ns2)
                           .cwiseProduct(dvec.segment(static_cast<Index>(jk) * ns2, ns2))
                           .sum();
            }
            Fmat(ijs / ns, ijs % ns) += sum;
        }

        // Hermitian completion of the upper triangle
        for (int is = 0; is < ns; ++is) {
            for (int js = is + 1; js < ns; ++js) {
                Fmat(is, js) = std::conj(Fmat(js, is));
            }
        }
    }
}

void Scph::diagonalize_and_symmetrize(const Eigen::MatrixXcd &Fmat, const std::vector<Eigen::MatrixXcd> &evec_initial,
                                      std::complex<double> ***v4_array_all, const unsigned int ik_irred,
                                      const unsigned int knum, const unsigned int knum_interpolate,
                                      const bool flag_converged, double **omega2_out, const unsigned int verbosity,
                                      int &icount, Eigen::VectorXd &eval_tmp, std::complex<double> ***dymat_q,
                                      bool *eval_repaired)
{
    using namespace Eigen;
    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense->nk;

    SelfAdjointEigenSolver<MatrixXcd> saes;
    saes.compute(Fmat);
    eval_tmp = saes.eigenvalues();

    for (unsigned int is = 0; is < ns; ++is) {

        double omega2_tmp = eval_tmp(is);

        if (omega2_tmp < 0.0 && std::abs(omega2_tmp) > 1.0e-16) {

            if (verbosity > 1) {
                std::cout << " Detect imaginary : ";
                std::cout << "  knum = " << knum + 1 << " is = " << is + 1 << '\n';
                for (int j = 0; j < 3; ++j) {
                    std::cout << "  xk = " << std::setw(15) << kmesh_dense->xk[knum][j];
                }
                std::cout << '\n';
            }

            if (v4_array_all[nk * ik_irred + knum][(ns + 1) * is][(ns + 1) * is].real() > 0.0) {
                if (verbosity > 1) {
                    std::cout << "  onsite V4 is positive\n\n";
                }

                if (flag_converged) {
                    ++icount;
                    eval_tmp(is) = omega2_out[knum][is] * std::pow(0.99, icount);
                } else {
                    ++icount;
                    eval_tmp(is) = -eval_tmp(is) * std::pow(0.99, icount);
                }
                // This repair depends on the global icount counter, i.e., the
                // effective fixed-point map changes between iterations.
                if (eval_repaired) *eval_repaired = true;
            } else {
                if (verbosity > 1) {
                    std::cout << "  onsite V4 is negative\n\n";
                }
                eval_tmp(is) = std::abs(omega2_tmp);
            }
        }
    }

    // New eigenvector matrix E_{new}= E_{old} * C
    const auto mat_tmp = evec_initial[knum] * saes.eigenvectors();
    MatrixXcd Dymat = mat_tmp * eval_tmp.asDiagonal() * mat_tmp.adjoint();

    symmetrize_dynamical_matrix(ik_irred, kmesh_coarse.get(), ns, mat_transform_sym, Dymat);
    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            dymat_q[is][js][knum_interpolate] = Dymat(is, js);
        }
    }
}

void Scph::interpolate_to_dense_mesh(std::complex<double> ***dymat_q,
                                     const std::complex<double> *const *const *dymat_q_HA,
                                     const std::vector<Eigen::MatrixXcd> &evec_initial,
                                     Eigen::MatrixXd &eval_interpolate, std::vector<Eigen::MatrixXcd> &evec_new,
                                     std::complex<double> ***cmat_convert, Eigen::MatrixXd &omega_now)
{
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto nk_interpolate = kmesh_coarse->nk;

    NDArray<std::complex<double>, 3> dymat_r_new;
    dymat_r_new.resize(ns, ns, nk_interpolate);

    replicate_dymat_for_all_kpoints(kmesh_coarse.get(), ns, mat_transform_sym, dymat_q);

    // Subtract harmonic contribution from the dynamical matrix
    for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                dymat_q[is][js][ik] -= dymat_q_HA[is][js][ik];
            }
        }
    }

    fourier_dymat_k_to_r(kmesh_interpolate[0], kmesh_interpolate[1], kmesh_interpolate[2], ns, dymat_q, dymat_r_new);

    // Create temporary C-style arrays for exec_interpolation
    NDArray<double, 2> eval_temp;
    NDArray<std::complex<double>, 3> evec_temp;
    eval_temp.resize(nk, ns);
    evec_temp.resize(nk, ns, ns);

    dynamical->exec_interpolation(kmesh_interpolate,
                                  dymat_r_new,
                                  nk,
                                  kmesh_dense->xk,
                                  kmesh_dense->kvec_na,
                                  eval_temp,
                                  evec_temp,
                                  dymat_harm_short,
                                  dymat_harm_long,
                                  mindist_list,
                                  true,
                                  true);

    for (unsigned int ik = 0; ik < nk; ++ik) {
        // Copy eigenvalues from temp array to Eigen matrix
        for (unsigned int is = 0; is < ns; ++is) {
            eval_interpolate(ik, is) = eval_temp[ik][is];
        }

        // Copy eigenvectors from temp array to Eigen matrix
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                evec_new[ik](is, js) = evec_temp[ik][is][js];
            }
        }

        build_cmat_at_k(ns, evec_initial[ik], evec_temp[ik], cmat_convert[ik]);

        for (unsigned int is = 0; is < ns; ++is) {
            omega_now(ik, is) = eval_interpolate(ik, is);
        }
    }

    eval_temp.clear();
    evec_temp.clear();
    dymat_r_new.clear();
}


bool Scph::check_convergence(const Eigen::MatrixXd &omega_now, const Eigen::MatrixXd &omega_old, const double conv_tol,
                             const unsigned int verbosity, const int iloop, double &diff) const
{
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;

    if (iloop == 0) {
        if (verbosity > 0) {
            std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " :  DIFF = N/A\n";
        }
        return false;
    }

    diff = 0.0;

    for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
        const auto knum = kmap_coarse_to_dense[ik];
        for (unsigned int is = 0; is < ns; ++is) {
            diff += pow2(omega_now(knum, is) - omega_old(knum, is));
        }
    }
    diff /= static_cast<double>(nk_interpolate * ns);
    if (verbosity > 0) {
        std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " : ";
        std::cout << " DIFF = " << std::scientific << std::setw(15) << std::sqrt(diff) << '\n';
    }

    if (std::sqrt(diff) < conv_tol) {
        auto has_negative = false;

        for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
            const auto knum = kmap_coarse_to_dense[ik];
            for (unsigned int is = 0; is < ns; ++is) {
                if (omega_now(knum, is) < 0.0 && std::abs(omega_now(knum, is)) > eps8) {
                    has_negative = true;
                    break;
                }
            }
        }
        if (!has_negative) {
            if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL : break SCPH loop\n";
            return true;
        }
        if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL but a negative frequency is detected.\n";
    }

    return false;
}

void Scph::compute_anharmonic_frequency(std::complex<double> ***v4_array_all, double **omega2_out,
                                        std::complex<double> ***evec_anharm_scph, const double temp,
                                        bool &flag_converged, std::complex<double> ***cmat_convert, const bool offdiag,
                                        std::complex<double> **delta_v2_renorm, const unsigned int verbosity,
                                        const bool compact_progress)
{
    // This is the main function of the SCPH equation.
    // The detailed algorithm can be found in PRB 92, 054301 (2015).
    // Eigen3 library is used for the compact notation of matrix-matrix products.

    using namespace Eigen;

    int ik;
    unsigned int is, js;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int knum;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    int iloop;

    MatrixXd omega_now(nk, ns), omega_old(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    VectorXd eval_tmp(ns);
    MatrixXcd Fmat(ns, ns);

    double diff, diff_prev = 1.0e10;
    const double conv_tol = tolerance_scph;
    double alpha = mixalpha;

    // Use Eigen types for better memory management and performance
    MatrixXd eval_interpolate(nk, ns);
    std::vector<MatrixXcd> evec_new;

    NDArray<std::complex<double>, 3> dymat_q;
    NDArray<std::complex<double>, 3> dymat_q_HA;

    std::vector<MatrixXcd> dmat_convert, dmat_convert_old;
    std::vector<MatrixXcd> evec_initial, evec_initial_adjoint;
    std::vector<MatrixXcd> Fmat0;

    // Reserve and construct Eigen matrices efficiently
    dmat_convert.reserve(nk);
    dmat_convert_old.reserve(nk);
    evec_initial.reserve(nk);
    evec_initial_adjoint.reserve(nk);
    evec_new.reserve(nk);

    for (ik = 0; ik < nk; ++ik) {
        dmat_convert.emplace_back(ns, ns);
        dmat_convert_old.emplace_back(ns, ns);
        evec_initial.emplace_back(ns, ns);
        evec_initial_adjoint.emplace_back(ns, ns);
        evec_new.emplace_back(ns, ns);
    }

    Fmat0.reserve(nk_irred_interpolate);
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0.emplace_back(ns, ns);
    }

    dymat_q.resize(ns, ns, nk_interpolate);
    dymat_q_HA.resize(ns, ns, nk_interpolate);

    const auto T_in = temp;

    // In the structural-optimization loop (compact_progress), the caller prints its
    // own step header including the temperature, and the per-iteration DIFF lines
    // are suppressed at default verbosity (VERBOSITY >= 2 restores them).
    if (!compact_progress) std::cout << " Temperature = " << T_in << " K\n";
    const unsigned int verbosity_iter = (compact_progress && verbosity <= 1) ? 0 : verbosity;

    // Initialize iteration
    initialize_scph_iteration(T_in,
                              flag_converged,
                              omega2_out,
                              verbosity,
                              omega_now,
                              omega2_HA,
                              evec_initial,
                              evec_initial_adjoint,
                              cmat_convert);

    // Setup harmonic dynamical matrices
    setup_harmonic_dynamical_matrices(omega2_HA, evec_initial, delta_v2_renorm, Fmat0, dymat_q_HA);

    // This can be done outside the function
    dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                     kmesh_dense->xk,
                                     kmesh_dense->kvec_na,
                                     dymat_harm_short,
                                     dymat_harm_long);

    int icount = 0;

    // Main loop
    for (iloop = 0; iloop < maxiter; ++iloop) {

        // Compute Qmat and Dmat from current frequencies
        compute_qmat_and_dmat(omega_now, T_in, cmat_convert, dmat_convert);

        // Mixing dmat
        if (iloop > 0) {
            // if (diff > diff_prev) {
            //     alpha = std::max(alpha * 0.7, 0.02);
            // } else {
            //     alpha = std::min(alpha * 1.1, 1.0);
            // }
            for (ik = 0; ik < nk; ++ik) {
                dmat_convert[ik] = alpha * dmat_convert[ik] + (1.0 - alpha) * dmat_convert_old[ik];
            }
        }

        // TODO: This loop may be parallelized by MPI in the future.
        for (ik = 0; ik < nk_irred_interpolate; ++ik) {

            const unsigned int knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            knum = kmap_coarse_to_dense[knum_interpolate];

            // Update Fmat with V4 contribution
            update_fmat_with_v4(Fmat0, v4_array_all, dmat_convert, offdiag, ik, Fmat);

            // Diagonalize and symmetrize
            diagonalize_and_symmetrize(Fmat,
                                       evec_initial,
                                       v4_array_all,
                                       ik,
                                       knum,
                                       knum_interpolate,
                                       flag_converged,
                                       omega2_out,
                                       verbosity,
                                       icount,
                                       eval_tmp,
                                       dymat_q);

        } // close loop ik

        // Interpolate to dense mesh using FFT-based method (correct Fourier interpolation)
        interpolate_to_dense_mesh(dymat_q,
                                  dymat_q_HA,
                                  evec_initial,
                                  eval_interpolate,
                                  evec_new,
                                  cmat_convert,
                                  omega_now);

        // Check convergence on the coarse k points
        if (check_convergence(omega_now, omega_old, conv_tol, verbosity_iter, iloop, diff)) {
            break;
        }

        // Save current omega for next iteration's comparison
        omega_old = omega_now;

        for (ik = 0; ik < nk; ++ik) {
            dmat_convert_old[ik] = dmat_convert[ik];
        }
        diff_prev = diff;
    } // end loop iteration

    if (std::sqrt(diff) < conv_tol) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in;
            std::cout << " : convergence achieved in " << std::setw(5) << iloop + 1 << " iterations.\n";
        }
        flag_converged = true;
    } else {
        if (verbosity > 0) {
            std::cout << "Temp = " << T_in;
            std::cout << " : not converged.\n";
        }
        flag_converged = false;
    }

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (eval_interpolate(ik, is) < 0.0) {
                if (std::abs(eval_interpolate(ik, is)) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -pow2(eval_interpolate(ik, is));
                }
            } else {
                omega2_out[ik][is] = pow2(eval_interpolate(ik, is));
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik](is, js);
            }
        }
    }

    if (verbosity > 1) {
        std::cout << "New eigenvalues\n";
        for (ik = 0; ik < nk_interpolate; ++ik) {
            knum = kmap_coarse_to_dense[ik];
            for (is = 0; is < ns; ++is) {
                std::cout << " ik_interpolate = " << std::setw(5) << ik + 1;
                std::cout << " is = " << std::setw(5) << is + 1;
                std::cout << " omega2 = " << std::setw(15) << omega2_out[knum][is] << '\n';
            }
            std::cout << '\n';
        }
    }

    // Eigen matrices are automatically deallocated
    dymat_q.clear();
    dymat_q_HA.clear();
}


void Scph::compute_anharmonic_frequency_diis(std::complex<double> ***v4_array_all, double **omega2_out,
                                             std::complex<double> ***evec_anharm_scph, const double temp,
                                             bool &flag_converged, std::complex<double> ***cmat_convert,
                                             const bool offdiag, std::complex<double> **delta_v2_renorm,
                                             const unsigned int verbosity, const bool compact_progress)
{
    // SCPH iteration accelerated by Pulay/Anderson (DIIS) mixing.
    //
    // The fixed-point variable is the full set of D matrices on the dense k mesh,
    //   x = {D_k},   g(x) = K(omega(x), C(x)),
    // and the residual handed to DIIS is r_n = g(x_n) - x_n of the same iterate.
    // The update x_{n+1} = sum_m c_m (x_m + beta * r_m) with beta = mixalpha
    // reduces exactly to the simple mixing of compute_anharmonic_frequency when
    // the history holds a single pair, so the early iterations are identical to
    // the reference implementation.
    //
    // A single DIIS history is kept for the concatenated state of all k points
    // because the SCP equation couples the k points through the inner sum
    // over q1; mixing D (rather than the eigenvalues) makes the residual
    // basis-free, so no mode tracking across iterations is required.

    using namespace Eigen;

    int ik;
    unsigned int is, js;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int knum;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    int iloop;

    MatrixXd omega_now(nk, ns), omega_old(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    VectorXd eval_tmp(ns);
    MatrixXcd Fmat(ns, ns);

    double diff = 1.0;
    const double conv_tol = tolerance_scph;

    MatrixXd eval_interpolate(nk, ns);
    std::vector<MatrixXcd> evec_new;

    NDArray<std::complex<double>, 3> dymat_q;
    NDArray<std::complex<double>, 3> dymat_q_HA;

    std::vector<MatrixXcd> dmat_convert, dmat_new, dmat_trial;
    std::vector<MatrixXcd> evec_initial, evec_initial_adjoint;
    std::vector<MatrixXcd> Fmat0;

    dmat_convert.reserve(nk);
    dmat_new.reserve(nk);
    dmat_trial.reserve(nk);
    evec_initial.reserve(nk);
    evec_initial_adjoint.reserve(nk);
    evec_new.reserve(nk);

    for (ik = 0; ik < nk; ++ik) {
        dmat_convert.emplace_back(ns, ns);
        dmat_new.emplace_back(ns, ns);
        dmat_trial.emplace_back(ns, ns);
        evec_initial.emplace_back(ns, ns);
        evec_initial_adjoint.emplace_back(ns, ns);
        evec_new.emplace_back(ns, ns);
    }

    Fmat0.reserve(nk_irred_interpolate);
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0.emplace_back(ns, ns);
    }

    dymat_q.resize(ns, ns, nk_interpolate);
    dymat_q_HA.resize(ns, ns, nk_interpolate);

    const auto T_in = temp;

    // See compute_anharmonic_frequency for the meaning of compact_progress.
    if (!compact_progress) std::cout << " Temperature = " << T_in << " K\n";
    const unsigned int verbosity_iter = (compact_progress && verbosity <= 1) ? 0 : verbosity;

    initialize_scph_iteration(T_in,
                              flag_converged,
                              omega2_out,
                              verbosity,
                              omega_now,
                              omega2_HA,
                              evec_initial,
                              evec_initial_adjoint,
                              cmat_convert);

    setup_harmonic_dynamical_matrices(omega2_HA, evec_initial, delta_v2_renorm, Fmat0, dymat_q_HA);

    dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                     kmesh_dense->xk,
                                     kmesh_dense->kvec_na,
                                     dymat_harm_short,
                                     dymat_harm_long);

    // ---- DIIS controls ----
    const int diis_history = 6;          // depth of the DIIS subspace
    const int diis_start = 3;            // number of plain simple-mixing iterations before DIIS
    const double growth_tol = 2.0;       // residual-growth factor that triggers a history reset
    const double psd_tol = 1.0e-6;       // relative tolerance of the PSD safeguard for extrapolated D
    const double resid_rel_tol = 1.0e-6; // relative D-space residual required on top of the frequency criterion

    GDIIS gdiis(diis_history, mixalpha, static_cast<int>(verbosity));

    const Index blocksize = 2 * static_cast<Index>(ns) * static_cast<Index>(ns);
    const Index ndim = static_cast<Index>(nk) * blocksize;
    VectorXd xvec(ndim), gvec(ndim), rvec(ndim), xvec_new(ndim);
    double rnorm_prev = -1.0;
    int ndiis_accepted = 0;
    int ndiis_fallback = 0;

    // Each D_k is column-major contiguous; a complex entry is two consecutive doubles.
    auto flatten_dmat = [&](const std::vector<MatrixXcd> &dmat, VectorXd &vec) {
#pragma omp parallel for
        for (int k = 0; k < static_cast<int>(nk); ++k) {
            vec.segment(static_cast<Index>(k) * blocksize, blocksize) =
                Map<const VectorXd>(reinterpret_cast<const double *>(dmat[k].data()), blocksize);
        }
    };
    auto unflatten_dmat = [&](const VectorXd &vec, std::vector<MatrixXcd> &dmat) {
#pragma omp parallel for
        for (int k = 0; k < static_cast<int>(nk); ++k) {
            Map<VectorXd>(reinterpret_cast<double *>(dmat[k].data()), blocksize) =
                vec.segment(static_cast<Index>(k) * blocksize, blocksize);
            // Remove the Hermiticity-breaking roundoff of the linear combination.
            dmat[k] = (0.5 * (dmat[k] + dmat[k].adjoint())).eval();
        }
    };

    auto is_positive_semidefinite = [&](const std::vector<MatrixXcd> &dmat) {
        bool ok = true;
#pragma omp parallel reduction(&& : ok)
        {
            SelfAdjointEigenSolver<MatrixXcd> saes_check;
#pragma omp for
            for (int k = 0; k < static_cast<int>(nk); ++k) {
                saes_check.compute(dmat[k], EigenvaluesOnly);
                if (saes_check.info() != Success) {
                    ok = false;
                    continue;
                }
                const double emax = saes_check.eigenvalues().cwiseAbs().maxCoeff();
                if (saes_check.eigenvalues().minCoeff() < -psd_tol * std::max(emax, 1.0e-12)) {
                    ok = false;
                }
            }
        }
        return ok;
    };

    int icount = 0;
    bool scp_converged = false;

    // x_0: D matrices built from the initial frequencies. This matches the
    // first iterate of compute_anharmonic_frequency.
    compute_qmat_and_dmat(omega_now, T_in, cmat_convert, dmat_convert);

    // Main loop
    for (iloop = 0; iloop < maxiter; ++iloop) {

        // Evaluate g(x_n): build F from the current D, diagonalize on the
        // coarse mesh, and interpolate to the dense mesh.
        bool eval_repaired = false;

        for (ik = 0; ik < nk_irred_interpolate; ++ik) {

            const unsigned int knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            knum = kmap_coarse_to_dense[knum_interpolate];

            update_fmat_with_v4(Fmat0, v4_array_all, dmat_convert, offdiag, ik, Fmat);

            diagonalize_and_symmetrize(Fmat,
                                       evec_initial,
                                       v4_array_all,
                                       ik,
                                       knum,
                                       knum_interpolate,
                                       flag_converged,
                                       omega2_out,
                                       verbosity,
                                       icount,
                                       eval_tmp,
                                       dymat_q,
                                       &eval_repaired);
        }

        interpolate_to_dense_mesh(dymat_q,
                                  dymat_q_HA,
                                  evec_initial,
                                  eval_interpolate,
                                  evec_new,
                                  cmat_convert,
                                  omega_now);

        // g(x_n) expressed in D space and the residual r_n = g(x_n) - x_n.
        compute_qmat_and_dmat(omega_now, T_in, cmat_convert, dmat_new);

        flatten_dmat(dmat_convert, xvec);
        flatten_dmat(dmat_new, gvec);
        rvec = gvec - xvec;

        const double rnorm = rvec.norm();
        const double rnorm_rel = rnorm / std::max(xvec.norm(), 1.0e-100);

        // The icount-dependent imaginary-mode repair in diagonalize_and_symmetrize
        // makes g(x) history-dependent, which formally violates the DIIS
        // assumptions. The pairs are pushed anyway: in soft-mode tests
        // (SrTiO3, cold start at T = 0) the repair stays active for most of
        // the run and DIIS still converges where simple mixing fails, with
        // the residual-growth reset below absorbing the resulting noise.
        // Excluding repaired iterations from the history would leave DIIS
        // disabled exactly where it is needed most.
        if (verbosity > 1) {
            std::cout << "  DIIS: |r|/|x| = " << std::scientific << rnorm_rel;
            if (eval_repaired) std::cout << "  (imaginary-mode repair active)";
            std::cout << '\n';
        }

        // The frequency criterion compares two successive outputs and can in
        // principle be fooled by extrapolated inputs that happen to yield
        // nearly identical frequencies away from self-consistency, so the
        // fixed-point residual in D space is additionally required to be small.
        if (check_convergence(omega_now, omega_old, conv_tol, verbosity_iter, iloop, diff)) {
            if (rnorm_rel < resid_rel_tol) {
                scp_converged = true;
                break;
            }
            if (verbosity > 1) {
                std::cout << "  DIIS: frequency criterion met but |r|/|x| = " << std::scientific << rnorm_rel
                          << " is still large; continuing.\n";
            }
        }
        omega_old = omega_now;
        if (rnorm_prev >= 0.0 && rnorm > growth_tol * rnorm_prev && gdiis.size() > 1) {
            // The iteration left the region where the stored subspace is
            // approximately linear; discard it and rebuild from here.
            gdiis.clear();
            if (verbosity > 1) {
                std::cout << "  DIIS: residual norm grew by more than a factor of " << growth_tol
                          << ", resetting the DIIS history.\n";
            }
        }
        gdiis.push(xvec, rvec);
        rnorm_prev = rnorm;

        // Update x_{n+1}.
        bool diis_accepted = false;
        const bool diis_active = (iloop + 1 >= diis_start) && gdiis.is_ready();

        if (diis_active && gdiis.extrapolate(xvec_new)) {
            unflatten_dmat(xvec_new, dmat_trial);
            if (is_positive_semidefinite(dmat_trial)) {
                diis_accepted = true;
            } else if (verbosity > 1) {
                std::cout << "  DIIS: extrapolated D matrix is not positive semidefinite, "
                             "falling back to simple mixing.\n";
            }
        }

        if (diis_accepted) {
            ++ndiis_accepted;
            for (ik = 0; ik < nk; ++ik) {
                dmat_convert[ik] = dmat_trial[ik];
            }
        } else {
            if (diis_active) ++ndiis_fallback;
            // Simple mixing, identical to compute_anharmonic_frequency.
            for (ik = 0; ik < nk; ++ik) {
                dmat_convert[ik] = mixalpha * dmat_new[ik] + (1.0 - mixalpha) * dmat_convert[ik];
            }
        }
    } // end loop iteration

    if (scp_converged) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in;
            std::cout << " : convergence achieved in " << std::setw(5) << iloop + 1 << " iterations.\n";
            std::cout << "  DIIS steps accepted: " << ndiis_accepted
                      << ", fallbacks to simple mixing: " << ndiis_fallback << '\n';
        }
        flag_converged = true;
    } else {
        if (verbosity > 0) {
            std::cout << "Temp = " << T_in;
            std::cout << " : not converged.\n";
        }
        flag_converged = false;
    }

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (eval_interpolate(ik, is) < 0.0) {
                if (std::abs(eval_interpolate(ik, is)) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -pow2(eval_interpolate(ik, is));
                }
            } else {
                omega2_out[ik][is] = pow2(eval_interpolate(ik, is));
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik](is, js);
            }
        }
    }

    if (verbosity > 1) {
        std::cout << "New eigenvalues\n";
        for (ik = 0; ik < nk_interpolate; ++ik) {
            knum = kmap_coarse_to_dense[ik];
            for (is = 0; is < ns; ++is) {
                std::cout << " ik_interpolate = " << std::setw(5) << ik + 1;
                std::cout << " is = " << std::setw(5) << is + 1;
                std::cout << " omega2 = " << std::setw(15) << omega2_out[knum][is] << '\n';
            }
            std::cout << '\n';
        }
    }

    dymat_q.clear();
    dymat_q_HA.clear();
}
