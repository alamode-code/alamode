/*
qha.cpp

Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "qha.h"
#include <Eigen/Core>
#include <array>
#include <iomanip>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "mathfunctions.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

Qha::Qha(PHON *phon) : ScphQhaCommon(phon)
{
    set_default_variables();
}

Qha::~Qha()
{
    deallocate_variables();
}

namespace PHON_NS
{
class QhaRelaxationModel final: public IRelaxationModel
{
public:
    QhaRelaxationModel(Qha &qha, StructuralOptWorkspace &ws, std::complex<double> ****dymat_anharm,
                       std::complex<double> ****delta_harmonic_dymat_renormalize, std::complex<double> ***cmat_convert,
                       double ***omega2_harm_renorm, std::complex<double> ***evec_harm_renorm_tmp,
                       std::complex<double> ***v4_renorm, std::complex<double> ***v4_with_umn,
                       std::complex<double> *v1_QHA, std::complex<double> *del_v0_del_umn_QHA,
                       std::complex<double> *del_v0_del_umn_ZSISA, std::complex<double> *del_v0_del_umn_vZSISA,
                       std::complex<double> **del_v1_del_umn_renorm, double **delq_delu_ZSISA, double **C2_array_renorm,
                       double **C2_array_ZSISA, DelVStrainData &del_v_strain, bool &converged_prev, int &str_diverged,
                       std::ofstream &fout_step_q0, std::ofstream &fout_step_u0, std::ofstream &fout_step_u_tensor) :
        qha_(qha), ws_(ws), dymat_anharm_(dymat_anharm),
        delta_harmonic_dymat_renormalize_(delta_harmonic_dymat_renormalize), cmat_convert_(cmat_convert),
        omega2_harm_renorm_(omega2_harm_renorm), evec_harm_renorm_tmp_(evec_harm_renorm_tmp), v4_renorm_(v4_renorm),
        v4_with_umn_(v4_with_umn), v1_QHA_(v1_QHA), del_v0_del_umn_QHA_(del_v0_del_umn_QHA),
        del_v0_del_umn_ZSISA_(del_v0_del_umn_ZSISA), del_v0_del_umn_vZSISA_(del_v0_del_umn_vZSISA),
        del_v1_del_umn_renorm_(del_v1_del_umn_renorm), delq_delu_ZSISA_(delq_delu_ZSISA),
        C2_array_renorm_(C2_array_renorm), C2_array_ZSISA_(C2_array_ZSISA), del_v_strain_(del_v_strain),
        converged_prev_(converged_prev), str_diverged_(str_diverged), fout_step_q0_(fout_step_q0),
        fout_step_u0_(fout_step_u0), fout_step_u_tensor_(fout_step_u_tensor)
    {}

    void before_init_structure(unsigned int, unsigned int, double, bool) override
    {}

    void after_init_structure(const unsigned int iT, double) override
    {
        qha_.converged_str_temp[iT] = 0; // set to 1 only when the loop converges
    }

    StructOptStepStatus do_structure_step(const unsigned int iT, const double temp, const int i_str_loop,
                                          std::vector<StructOptStepRecord> &step_history) override
    {
        const auto nk = qha_.kmesh_dense->nk;
        const auto ns = qha_.dynamical->neval;
        const auto nk_irred_interpolate = qha_.kmesh_coarse->nk_irred;
        const auto complex_zero = std::complex<double>(0.0, 0.0);
        auto &structure_state = ws_.structure_state;
        auto &q0 = structure_state.q0;
        auto &u_tensor = structure_state.u_tensor;
        auto &v4_ref = ws_.v4_ref;
        auto &C2_array = ws_.C2_array;
        auto &harm_optical_modes = ws_.harm_optical_modes;

        // Strain renormalization of v4 is not available; the identity
        // copy keeps the historical bookkeeping (ws.v4_for_renorm
        // points at v4_with_umn).
        for (auto ik = 0; ik < nk_irred_interpolate * nk; ik++) {
            for (auto is = 0; is < ns * ns; is++) {
                for (auto is1 = 0; is1 < ns * ns; is1++) {
                    v4_with_umn_[ik][is][is1] = v4_ref[ik][is][is1];
                }
            }
        }

        // recompute the strain- and q0-renormalized IFCs at the
        // current structure
        qha_.renormalize_ifcs_at_structure(ws_);

        // copy v4_ref to v4_renorm
        for (auto ik = 0; ik < nk_irred_interpolate * qha_.kmesh_dense->nk; ik++) {
            for (auto is1 = 0; is1 < ns * ns; is1++) {
                for (auto is2 = 0; is2 < ns * ns; is2++) {
                    v4_renorm_[ik][is1][is2] = v4_ref[ik][is1][is2];
                }
            }
        }

        // QHA-only: strain-force coupling entering the ZSISA and
        // v-ZSISA stress corrections.
        if (ws_.relax_mode == RelaxationStrMode::CoordinatesOnly) {
            for (auto i1 = 0; i1 < 9; i1++) {
                for (auto is1 = 0; is1 < ns; is1++) {
                    del_v1_del_umn_renorm_[i1][is1] = complex_zero;
                }
            }
        } else if (ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            qha_.calculate_del_v1_del_umn_renorm(del_v1_del_umn_renorm_, u_tensor, del_v_strain_, q0);
        }

        // solve the renormalized-harmonic lattice dynamics at this
        // structure and compute the QHA forces and stress (with the
        // ZSISA/v-ZSISA overwrites when requested)
        qha_.solve_qha_and_compute_forces(ws_,
                                          iT,
                                          temp,
                                          cmat_convert_,
                                          delta_harmonic_dymat_renormalize_,
                                          omega2_harm_renorm_,
                                          evec_harm_renorm_tmp_,
                                          v1_QHA_,
                                          del_v0_del_umn_QHA_,
                                          del_v0_del_umn_ZSISA_,
                                          del_v0_del_umn_vZSISA_,
                                          del_v1_del_umn_renorm_,
                                          delq_delu_ZSISA_,
                                          C2_array_renorm_,
                                          C2_array_ZSISA_);

        // Print the structure the QHA forces were evaluated at, together with the
        // stress used for the update (cell relaxation only) and the space group
        // detected by spglib.
        std::cout << "\n Structure at this step :\n";
        const auto spg_label = qha_.relaxation->print_structure_and_symmetry(
            structure_state,
            ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell ? del_v0_del_umn_QHA_ : nullptr);
        std::cout << '\n';

        qha_.relaxation->update_cell_coordinate(structure_state,
                                                v1_QHA_,
                                                omega2_harm_renorm_[iT],
                                                del_v0_del_umn_QHA_,
                                                C2_array,
                                                cmat_convert_,
                                                harm_optical_modes,
                                                qha_.omega2_harmonic,
                                                qha_.evec_harmonic);
        const auto du0 = structure_state.du0;
        const auto du_tensor = structure_state.du_tensor;

        qha_.relaxation->write_stepresfile(structure_state,
                                           i_str_loop + 1,
                                           fout_step_q0_,
                                           fout_step_u0_,
                                           fout_step_u_tensor_);
        qha_.relaxation->check_str_divergence(str_diverged_, structure_state);

        if (str_diverged_) {
            converged_prev_ = false;
            std::cout << " The crystal structure diverged.";
            std::cout << " Break from the structure loop.\n";
            step_history.push_back({true, du0, du_tensor, -1.0, -1.0, spg_label});
            return StructOptStepStatus::Diverged;
        }

        double grad_norm, cell_grad_norm;
        qha_.compute_and_print_step_gradients(ws_,
                                              v1_QHA_,
                                              del_v0_del_umn_QHA_,
                                              du0,
                                              du_tensor,
                                              spg_label,
                                              step_history,
                                              grad_norm,
                                              cell_grad_norm);

        if (du0 < qha_.relaxation->coord_conv_tol && du_tensor < qha_.relaxation->cell_conv_tol) {
            std::cout << "\n\n du0 is smaller than COORD_CONV_TOL = " << std::scientific << std::setw(15)
                      << std::setprecision(6) << qha_.relaxation->coord_conv_tol << '\n';
            if (ws_.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                std::cout << " du_tensor is smaller than CELL_CONV_TOL = " << std::scientific << std::setw(15)
                          << std::setprecision(6) << qha_.relaxation->cell_conv_tol << '\n';
            }
            std::cout << " Structural optimization converged in " << i_str_loop + 1 << "-th loop.\n\n";
            std::cout << " break structural loop.\n\n";
            qha_.converged_str_temp[iT] = 1;
            return StructOptStepStatus::Converged;
        }

        return StructOptStepStatus::Continue;
    }

    void after_structure_loop(const unsigned int iT, double, int, bool &converged_this_temp) override
    {
        converged_this_temp = qha_.converged_str_temp[iT] != 0;
    }

    void record_v0(const unsigned int iT) override
    {
        const auto ns = qha_.dynamical->neval;

        // record zero-th order term of PES
        qha_.V0[iT] = ws_.v0_renorm;

        // copy delta_harmonic_dymat_renormalize to dymat_anharm
        // This process is required for postprocess.
        for (auto is1 = 0; is1 < ns; is1++) {
            for (auto is2 = 0; is2 < ns; is2++) {
                for (auto ik = 0; ik < qha_.kmesh_coarse->nk; ik++) {
                    dymat_anharm_[iT][is1][is2][ik] = delta_harmonic_dymat_renormalize_[iT][is1][is2][ik];
                }
            }
        }
    }

    void finalize_temperature(unsigned int, double, bool, bool &) override
    {}

    void print_run_summary() override
    {}

    bool history_has_scp_column() const override
    {
        return false;
    }

private:
    Qha &qha_;
    StructuralOptWorkspace &ws_;
    std::complex<double> ****dymat_anharm_;
    std::complex<double> ****delta_harmonic_dymat_renormalize_;
    std::complex<double> ***cmat_convert_;
    double ***omega2_harm_renorm_;
    std::complex<double> ***evec_harm_renorm_tmp_;
    std::complex<double> ***v4_renorm_;
    std::complex<double> ***v4_with_umn_;
    std::complex<double> *v1_QHA_;
    std::complex<double> *del_v0_del_umn_QHA_;
    std::complex<double> *del_v0_del_umn_ZSISA_;
    std::complex<double> *del_v0_del_umn_vZSISA_;
    std::complex<double> **del_v1_del_umn_renorm_;
    double **delq_delu_ZSISA_;
    double **C2_array_renorm_;
    double **C2_array_ZSISA_;
    DelVStrainData &del_v_strain_;
    bool &converged_prev_;
    int &str_diverged_;
    std::ofstream &fout_step_q0_;
    std::ofstream &fout_step_u0_;
    std::ofstream &fout_step_u_tensor_;
};
} // namespace PHON_NS

void Qha::set_default_variables()
{
    restart_qha = false;
    qha_scheme = QhaScheme::Standard;

    initialize_variables();
}


void Qha::setup_qha()
{
    setup_kmesh(kmesh_qha, kmesh_interpolate, "QHA", "KMESH_QHA should be a integral multiple of KMESH_INTERPOLATE.");
    setup_eigvecs();
    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);
    setup_pp_interaction(relax_mode != RelaxationStrMode::None);

    setup_structural_data();
}


void Qha::exec_qha_optimization()
{
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    NDArray<std::complex<double>, 4> delta_dymat_qha;
    NDArray<std::complex<double>, 4> delta_harmonic_dymat_renormalize;
    delta_dymat_qha.resize(NT, ns, ns, kmesh_coarse->nk);
    delta_harmonic_dymat_renormalize.resize(NT, ns, ns, kmesh_coarse->nk);

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    zerofill_harmonic_dymat_renormalize(delta_harmonic_dymat_renormalize, NT);

    // Per-temperature convergence records, updated by the relaxation loop
    // below and stored in the state file (QHA has no SCP inner loop).
    converged_scph_temp.assign(NT, 1);
    converged_str_temp.assign(NT, 1);

    // Sized on every rank before the restart branch below, which broadcasts
    // loaded values into it.
    V0.assign(NT, 0.0);

    MPI_Bcast(&restart_qha, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_h5_io, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    // These are parsed on rank 0 only and read on every rank below:
    // ialgo selects between V4 kernels with different MPI collectives, and
    // selfenergy_offdiagonal enters the collective V3/V4 element computations.
    MPI_Bcast(&selfenergy_offdiagonal, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ialgo, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (restart_qha) {

        if (mympi->my_rank == 0) {
            std::cout << " RESTART_QHA is true.\n";
            std::cout << " Dynamical matrix is read from file ...";
        }

        const auto with_relax = relax_mode != RelaxationStrMode::None;

        // Preferred restart source: the unified state file. It stores the
        // QHA dynamical-matrix correction and the renormalized-harmonic one
        // separately (the legacy text restart approximated both by the
        // contents of .renorm_harm_dymat).
        auto loaded_h5 = false;
        if (use_h5_io) {
            loaded_h5 = load_scph_state_h5(phon->job_title + ".qha.h5",
                                           "QHA",
                                           NT,
                                           dynamical->nonanalytic,
                                           true,
                                           relaxation->relax_str,
                                           delta_dymat_qha,
                                           delta_harmonic_dymat_renormalize,
                                           with_relax ? &V0 : nullptr);
        }

        if (loaded_h5) {
            // Regenerate the human-readable V0-vs-T output, which may be
            // absent when restarting from the unified file alone.
            if (with_relax && mympi->my_rank == 0) store_V0_to_file();
        } else {
            load_scph_dymat_from_file(delta_dymat_qha,
                                      phon->job_title + ".renorm_harm_dymat",
                                      kmesh_dense.get(),
                                      kmesh_coarse.get(),
                                      dynamical->nonanalytic,
                                      true);
            load_scph_dymat_from_file(delta_harmonic_dymat_renormalize,
                                      phon->job_title + ".renorm_harm_dymat",
                                      kmesh_dense.get(),
                                      kmesh_coarse.get(),
                                      dynamical->nonanalytic,
                                      true);

            // structural optimization
            if (with_relax) {
                load_V0_from_file();
            }

            // One-way migration of the legacy state into the unified file.
            if (use_h5_io && mympi->my_rank == 0) {
                write_scph_state_h5(phon->job_title + ".qha.h5",
                                    "QHA",
                                    NT,
                                    dynamical->nonanalytic,
                                    true,
                                    relaxation->relax_str,
                                    "qha",
                                    delta_dymat_qha,
                                    delta_harmonic_dymat_renormalize,
                                    with_relax ? &V0 : nullptr,
                                    kmesh_coarse.get(),
                                    mindist_list);
            }
        }
    } else {

        // QHA + structural optimization
        if (relax_mode == RelaxationStrMode::CoordinatesOnly || relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            exec_QHA_relax_main(delta_dymat_qha, delta_harmonic_dymat_renormalize);
        }
        // lowest-order QHA
        else if (relax_mode == RelaxationStrMode::PerturbativeQha)
        {
            exec_perturbative_QHA(delta_dymat_qha, delta_harmonic_dymat_renormalize);
        }

        if (mympi->my_rank == 0) {
            const auto with_relax = relax_mode != RelaxationStrMode::None;
            if (use_h5_io) {
                write_scph_state_h5(phon->job_title + ".qha.h5",
                                    "QHA",
                                    NT,
                                    dynamical->nonanalytic,
                                    true,
                                    relaxation->relax_str,
                                    "qha",
                                    delta_dymat_qha,
                                    delta_harmonic_dymat_renormalize,
                                    with_relax ? &V0 : nullptr,
                                    kmesh_coarse.get(),
                                    mindist_list);
                // .V0 doubles as a human-readable physical output (V0 vs T),
                // so the text file is kept; restart reads only the h5.
                if (with_relax) store_V0_to_file();
            } else if (with_relax) {
                // write dymat to file
                // write renormalized harmonic dynamical matrix when the crystal structure is optimized
                store_renormalized_dymat_to_file(delta_harmonic_dymat_renormalize,
                                                 phon->job_title + ".renorm_harm_dymat",
                                                 kmesh_dense.get(),
                                                 kmesh_coarse.get(),
                                                 dynamical->nonanalytic,
                                                 true);
                store_V0_to_file();
            }
            // Text .qha_dfc2 is kept in both modes during the transition.
            write_anharmonic_correction_fc2(delta_dymat_qha, NT, kmesh_coarse.get(), mindist_list, true);
        }
    }

    postprocess(delta_dymat_qha,
                delta_harmonic_dymat_renormalize,
                delta_dymat_qha,
                kmesh_coarse.get(),
                mindist_list,
                true,
                0);

    delta_dymat_qha.clear();
    delta_harmonic_dymat_renormalize.clear();
}

void Qha::exec_QHA_relax_main(std::complex<double> ****dymat_anharm,
                              std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    // Scratch IFC/structure buffers grouped in the workspace consumed by the
    // loop-stage helpers; the names below are aliases into it.
    StructuralOptWorkspace ws;
    ws.relax_mode = relax_mode;
    // renormalization of harmonic dynamical matrix
    auto &delta_v2_renorm = ws.delta_v2_renorm;
    auto &delta_v2_with_umn = ws.delta_v2_with_umn;
    NDArray<double, 3> omega2_harm_renorm;
    NDArray<std::complex<double>, 3> evec_harm_renorm_tmp;
    // k-space IFCs at the reference and updated structures
    auto &v1_ref = ws.v1_ref;
    auto &v1_renorm = ws.v1_renorm;
    auto &v1_with_umn = ws.v1_with_umn;
    auto &v3_ref = ws.v3_ref;
    auto &v3_renorm = ws.v3_renorm;
    auto &v3_with_umn = ws.v3_with_umn;
    auto &v4_ref = ws.v4_ref;
    NDArray<std::complex<double>, 3> v4_renorm;
    NDArray<std::complex<double>, 3> v4_with_umn;
    auto &v0_ref = ws.v0_ref;
    v0_ref = 0.0; // set original ground state energy as zero

    // elastic constants
    auto &C1_array = ws.C1_array;
    auto &C2_array = ws.C2_array;
    auto &C3_array = ws.C3_array;
    NDArray<double, 2> C2_array_ZSISA;

    // strain-derivative of k-space IFCs
    DelVStrainData del_v_strain;
    ws.del_v_strain = &del_v_strain;

    auto &del_v0_del_umn_renorm = ws.del_v0_del_umn_renorm;
    NDArray<std::complex<double>, 2> del_v1_del_umn_renorm;
    NDArray<double, 2> C2_array_renorm;

    // atomic forces and stress tensor at finite temperatures
    NDArray<std::complex<double>, 1> v1_QHA;
    NDArray<std::complex<double>, 1> del_v0_del_umn_QHA;
    NDArray<std::complex<double>, 1> del_v0_del_umn_ZSISA;
    NDArray<std::complex<double>, 1> del_v0_del_umn_vZSISA;

    NDArray<double, 2> delq_delu_ZSISA;

    // cell optimization
    auto pvcell = relaxation->stat_pressure * system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) *
                  1.0e-30; // in 10^9 J = GJ
    pvcell *= 1.0e9 / Ryd; // in Ry
    ws.pvcell = pvcell;

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    omega2_harm_renorm.resize(NT, nk, ns);
    evec_harm_renorm_tmp.resize(nk, ns, ns);

    // Common buffers, reference V3/V4 elements, strain derivatives of the
    // IFCs, optimizer, and Gamma-point optical modes (collective).
    setup_structural_opt_buffers(ws);

    auto &structure_state = ws.structure_state;

    v1_QHA.resize(ns);
    del_v0_del_umn_QHA.resize(9);
    del_v0_del_umn_ZSISA.resize(9);
    del_v0_del_umn_vZSISA.resize(9);
    del_v1_del_umn_renorm.resize(9, ns);

    delq_delu_ZSISA.resize(ns, 9);
    C2_array_renorm.resize(9, 9);

    v4_renorm.resize(nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);
    v4_with_umn.resize(nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);
    // QHA feeds the (numerically identical) v4_with_umn copy into the q0
    // renormalization; see renormalize_ifcs_at_structure.
    ws.v4_for_renorm = v4_with_umn;

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
        C2_array_ZSISA.resize(9, 9);

        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_step_q0, fout_step_u0;
        std::ofstream fout_q0, fout_u0;
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


        QhaRelaxationModel model(*this,
                                 ws,
                                 dymat_anharm,
                                 delta_harmonic_dymat_renormalize,
                                 cmat_convert,
                                 omega2_harm_renorm,
                                 evec_harm_renorm_tmp,
                                 v4_renorm,
                                 v4_with_umn,
                                 v1_QHA,
                                 del_v0_del_umn_QHA,
                                 del_v0_del_umn_ZSISA,
                                 del_v0_del_umn_vZSISA,
                                 del_v1_del_umn_renorm,
                                 delq_delu_ZSISA,
                                 C2_array_renorm,
                                 C2_array_ZSISA,
                                 del_v_strain,
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

        // Output files of structural optimization
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
        C2_array_ZSISA.clear();
    }

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
    v4_renorm.clear();
    v4_with_umn.clear();

    v1_QHA.clear();
    del_v1_del_umn_renorm.clear();
    del_v0_del_umn_QHA.clear();
    del_v0_del_umn_ZSISA.clear();
    del_v0_del_umn_vZSISA.clear();
    del_v0_del_umn_renorm.clear();

    delq_delu_ZSISA.clear();

    C2_array_renorm.clear();
}

void Qha::solve_qha_and_compute_forces(StructuralOptWorkspace &ws, const unsigned int iT, const double temp,
                                       std::complex<double> ***cmat_convert,
                                       std::complex<double> ****delta_harmonic_dymat_renormalize,
                                       double ***omega2_harm_renorm, std::complex<double> ***evec_harm_renorm_tmp,
                                       std::complex<double> *v1_QHA, std::complex<double> *del_v0_del_umn_QHA,
                                       std::complex<double> *del_v0_del_umn_ZSISA,
                                       std::complex<double> *del_v0_del_umn_vZSISA,
                                       std::complex<double> **del_v1_del_umn_renorm, double **delq_delu_ZSISA,
                                       double **C2_array_renorm, double **C2_array_ZSISA)
{
    const auto ns = dynamical->neval;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    auto &q0 = ws.structure_state.q0;
    auto &u_tensor = ws.structure_state.u_tensor;
    auto &eta_tensor = ws.structure_state.eta_tensor;

    dynamical->compute_renormalized_harmonic_frequency(omega2_harm_renorm[iT],
                                                       evec_harm_renorm_tmp,
                                                       ws.delta_v2_renorm,
                                                       omega2_harmonic,
                                                       evec_harmonic,
                                                       kmesh_coarse.get(),
                                                       kmesh_dense.get(),
                                                       kmap_coarse_to_dense,
                                                       mat_transform_sym,
                                                       mindist_list,
                                                       writes->getVerbosity());

    dynamical->calc_new_dymat_with_evec(delta_harmonic_dymat_renormalize[iT],
                                        omega2_harm_renorm[iT],
                                        evec_harm_renorm_tmp,
                                        kmesh_coarse.get(),
                                        kmap_coarse_to_dense);
    // delta_harmonic_dymat_renormalize is copied to dymat_anharm after structure convergence,
    // which is required for postprocess.

    compute_cmat(cmat_convert, evec_harm_renorm_tmp);

    // The same functions (compute_anharmonic_v1_array, compute_anharmonic_del_v0_del_umn) as
    // in Scph::exec_scph_relax_cell_coordinate_main can be used for
    // calculating the finite-temperature forces and stress tensor.
    // This is because we truncate the Taylor expansion of PES at the fourth order (?)
    compute_anharmonic_v1_array(v1_QHA,
                                ws.v1_renorm,
                                ws.v3_renorm,
                                cmat_convert,
                                omega2_harm_renorm[iT],
                                temp,
                                kmesh_dense.get());

    if (ws.relax_mode == RelaxationStrMode::CoordinatesOnly) {
        for (auto i1 = 0; i1 < 9; i1++) {
            del_v0_del_umn_QHA[i1] = complex_zero;
        }
    } else if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        compute_anharmonic_del_v0_del_umn(del_v0_del_umn_QHA,
                                          ws.del_v0_del_umn_renorm,
                                          *ws.del_v_strain,
                                          u_tensor,
                                          q0,
                                          cmat_convert,
                                          omega2_harm_renorm[iT],
                                          temp,
                                          kmesh_dense.get());

        compute_ZSISA_stress(delq_delu_ZSISA,
                             del_v0_del_umn_ZSISA,
                             cmat_convert,
                             omega2_harm_renorm[iT],
                             del_v0_del_umn_QHA,
                             del_v1_del_umn_renorm,
                             v1_QHA,
                             ws.harm_optical_modes);

        // qha_scheme == 1 : ZSISA
        if (qha_scheme == QhaScheme::ZSISA) {
            // overwrite v1_QHA by zero-temperature first-order IFCs.
            for (auto is = 0; is < ns; is++) {
                v1_QHA[is] = ws.v1_renorm[is];
            }
            // overwrite finite-temperature stress tensor
            for (auto i1 = 0; i1 < 9; i1++) {
                del_v0_del_umn_QHA[i1] = del_v0_del_umn_ZSISA[i1];
            }
        }

        // calculate renormalized second-order elastic constants
        calculate_C2_array_renorm(C2_array_renorm,
                                  u_tensor,
                                  eta_tensor,
                                  ws.C2_array,
                                  ws.C3_array,
                                  *ws.del_v_strain,
                                  q0);

        calculate_C2_array_ZSISA(C2_array_ZSISA, C2_array_renorm, del_v1_del_umn_renorm, delq_delu_ZSISA);

        compute_vZSISA_stress(del_v0_del_umn_vZSISA,
                              C2_array_ZSISA,
                              ws.del_v0_del_umn_renorm,
                              del_v0_del_umn_ZSISA,
                              u_tensor);

        // qha_scheme == 2 : v-ZSISA
        // overwrite finite-temperature force and stress tensor
        if (qha_scheme == QhaScheme::VZSISA) {
            for (auto is = 0; is < ns; is++) {
                v1_QHA[is] = ws.v1_renorm[is];
            }

            for (auto i1 = 0; i1 < 9; i1++) {
                del_v0_del_umn_QHA[i1] = del_v0_del_umn_vZSISA[i1];
            }
        }
    }
}


void Qha::exec_perturbative_QHA(std::complex<double> ****dymat_anharm,
                                std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    int ik, is, js, ik1, is1, is2;
    int ixyz1, ixyz2, ixyz3;
    int itmp1, itmp2, itmp3, itmp4;
    static auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    // renormalization of harmonic dynamical matrix
    NDArray<std::complex<double>, 2> delta_v2_renorm;
    NDArray<std::complex<double>, 2> delta_v2_with_umn;
    NDArray<double, 3> omega2_harm_renorm;
    NDArray<std::complex<double>, 3> evec_harm_renorm_tmp;
    // original and renormalized IFCs
    NDArray<std::complex<double>, 1> v1_ref, v1_renorm, v1_with_umn;
    NDArray<std::complex<double>, 3> v3_ref;         // We fix cubic IFCs in perturbative QHA.
    NDArray<std::complex<double>, 3> v4_array_dummy; // We set quartic IFCs as zero.

    // elastic constants
    NDArray<double, 1> C1_array;
    NDArray<double, 2> C2_array;
    NDArray<double, 3> C3_array;

    // force and stress tensor from F_vib
    NDArray<std::complex<double>, 1> v1_vib;
    NDArray<std::complex<double>, 1> del_v0_del_umn_vib;

    // strain-derivative of k-space IFCs
    // (calculated by real-space IFC renormalization or finite-difference method)
    DelVStrainData del_v_strain;

    // IFC renormalization
    double v0_with_umn, v0_renorm;

    // structural optimization
    int i_temp_loop;
    RelaxationStructureState structure_state;
    std::vector<int> harm_optical_modes(ns - 3);

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    MatrixXcd elastic_mat_tmp(ns - 3 + 6, ns - 3 + 6); // optical phonons + independent strain
    VectorXcd q0_umn(ns - 3 + 6), del_Fvib_q0_umn(ns - 3 + 6);

    omega2_harm_renorm.resize(NT, nk, ns);
    evec_harm_renorm_tmp.resize(nk, ns, ns);
    delta_v2_renorm.resize(nk_interpolate, ns * ns);
    delta_v2_with_umn.resize(nk_interpolate, ns * ns);

    v1_ref.resize(ns);
    v1_with_umn.resize(ns);
    v1_renorm.resize(ns);

    v1_vib.resize(ns);
    del_v0_del_umn_vib.resize(9);

    structure_state.resize(ns);
    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &eta_tensor = structure_state.eta_tensor;

    v4_array_dummy.resize(nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);

    for (ik1 = 0; ik1 < nk_irred_interpolate * kmesh_dense->nk; ik1++) {
        for (is1 = 0; is1 < ns * ns; is1++) {
            for (is2 = 0; is2 < ns * ns; is2++) {
                v4_array_dummy[ik1][is1][is2] = complex_zero;
            }
        }
    }

    v3_ref.resize(nk, ns, ns * ns);

    compute_V3_elements_mpi_over_kpoint(v3_ref,
                                        evec_harmonic,
                                        selfenergy_offdiagonal,
                                        kmesh_coarse.get(),
                                        kmesh_dense.get(),
                                        phase_factor.get(),
                                        phi3_reciprocal);

    // assume that the atomic forces are zero at initial structure
    for (is = 0; is < ns; is++) {
        v1_ref[is] = 0.0;
    }

    del_v_strain.resize(nk, ns);

    relaxation->compute_del_v_strain(kmesh_coarse.get(),
                                     kmesh_dense.get(),
                                     del_v_strain,
                                     omega2_harmonic,
                                     evec_harmonic,
                                     RelaxationStrMode::PerturbativeQha,
                                     mindist_list,
                                     phase_factor.get());

    // set dummy variables as zero for perturbative-QHA paths
    del_v_strain.del3_v1.setZero();
    for (auto &mat: del_v_strain.del2_v2) {
        mat.setZero();
    }
    for (auto &per_strain: del_v_strain.del_v3) {
        for (auto &mat: per_strain) {
            mat.setZero();
        }
    }

    C1_array.resize(9);
    C2_array.resize(9, 9);
    C3_array.resize(9, 9, 9);

    // get indices of optical modes at Gamma point
    js = 0;
    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < eps8) {
            continue;
        }
        harm_optical_modes[js] = is;
        js++;
    }
    if (js != ns - 3) {
        exit("exec_scph_relax_cell_coordinate_main", "The number of detected optical modes is not ns-3.");
    }

    if (mympi->my_rank == 0) {

        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);


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

        // set elastic constants
        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_q0, fout_u0, fout_u_tensor;

        fout_q0.open(phon->job_title + ".normal_disp");
        fout_u0.open(phon->job_title + ".atom_disp");
        fout_u_tensor.open(phon->job_title + ".umn_tensor");

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        i_temp_loop = -1;

        std::cout << " Start QHA calculation.\n";
        std::cout
            << " Internal coordinates and shape of the unit cell are calculated by lowest-order perturbation theory ...\n\n";

        for (double temp: vec_temp) {
            i_temp_loop++;
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            calc_v1_vib(v1_vib, v3_ref, temp);
            calc_del_v0_del_umn_vib(del_v0_del_umn_vib, del_v_strain, temp);

            // calculate matrix
            // elastic_mat_tmp
            for (itmp1 = 0; itmp1 < ns + 3; itmp1++) {
                for (itmp2 = 0; itmp2 < ns + 3; itmp2++) {
                    elastic_mat_tmp(itmp1, itmp2) = complex_zero;
                }
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                elastic_mat_tmp(is1, is1) = omega2_harmonic[0][harm_optical_modes[is1]];
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    elastic_mat_tmp(is1, ns - 3 + ixyz1) = del_v_strain.del_v1(ixyz1 * 3 + ixyz1, is2);
                    elastic_mat_tmp(ns - 3 + ixyz1, is1) = elastic_mat_tmp(is1, ns - 3 + ixyz1);
                }

                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    ixyz2 = (ixyz1 + 1) % 3;
                    ixyz3 = (ixyz1 + 2) % 3;

                    elastic_mat_tmp(is1, ns + ixyz1) =
                        del_v_strain.del_v1(ixyz2 * 3 + ixyz3, is2) + del_v_strain.del_v1(ixyz3 * 3 + ixyz2, is2);
                    elastic_mat_tmp(ns + ixyz1, is1) = elastic_mat_tmp(is1, ns + ixyz1);
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    itmp1 = (ixyz1 + 1) % 3;
                    itmp2 = (ixyz1 + 2) % 3;
                    itmp3 = (ixyz2 + 1) % 3;
                    itmp4 = (ixyz2 + 2) % 3;

                    elastic_mat_tmp(ns - 3 + ixyz1, ns - 3 + ixyz2) = C2_array[ixyz1 * 4][ixyz2 * 4];

                    elastic_mat_tmp(ns - 3 + ixyz1, ns + ixyz2) =
                        C2_array[ixyz1 * 4][itmp3 * 3 + itmp4] + C2_array[ixyz1 * 4][itmp4 * 3 + itmp3];
                    elastic_mat_tmp(ns + ixyz1, ns - 3 + ixyz2) =
                        C2_array[itmp1 * 3 + itmp2][ixyz2 * 4] + C2_array[itmp2 * 3 + itmp1][ixyz2 * 4];

                    elastic_mat_tmp(ns + ixyz1, ns + ixyz2) = C2_array[itmp1 * 3 + itmp2][itmp3 * 3 + itmp4] +
                                                              C2_array[itmp2 * 3 + itmp1][itmp3 * 3 + itmp4] +
                                                              C2_array[itmp1 * 3 + itmp2][itmp4 * 3 + itmp3] +
                                                              C2_array[itmp2 * 3 + itmp1][itmp4 * 3 + itmp3];
                }
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                del_Fvib_q0_umn(is1) = -v1_vib[is2];
            }
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                ixyz2 = (ixyz1 + 1) % 3;
                ixyz3 = (ixyz1 + 2) % 3;

                del_Fvib_q0_umn(ns - 3 + ixyz1) = -del_v0_del_umn_vib[ixyz1 * 3 + ixyz1];
                del_Fvib_q0_umn(ns + ixyz1) =
                    -del_v0_del_umn_vib[ixyz2 * 3 + ixyz3] + del_v0_del_umn_vib[ixyz3 * 3 + ixyz2];
            }
            q0_umn = elastic_mat_tmp.colPivHouseholderQr().solve(del_Fvib_q0_umn);

            for (is1 = 0; is1 < ns; is1++) {
                q0[is1] = 0.0;
            }
            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                q0[is2] = q0_umn(is1).real();
            }
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                ixyz2 = (ixyz1 + 1) % 3;
                ixyz3 = (ixyz1 + 2) % 3;

                u_tensor[ixyz1][ixyz1] = q0_umn(ns - 3 + ixyz1).real();
                u_tensor[ixyz2][ixyz3] = q0_umn(ns + ixyz1).real();
                u_tensor[ixyz3][ixyz2] = q0_umn(ns + ixyz1).real();
            }

            // print obtained structure
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            relaxation->write_resfile_atT(structure_state, temp, fout_q0, fout_u0, fout_u_tensor);

            // calculate renormalized IFCs for postprocess
            // Note that the cubic IFCs are fixed at the reference values in perturbative QHA.

            // renormalization by strain
            relaxation->calculate_eta_tensor(eta_tensor, u_tensor);
            relaxation->renormalize_v0_from_umn(v0_with_umn,
                                                0.0,
                                                eta_tensor,
                                                C1_array,
                                                C2_array,
                                                C3_array,
                                                u_tensor,
                                                0.0); // pressure is limited to zero

            relaxation->renormalize_v1_from_umn(v1_with_umn, v1_ref, del_v_strain, u_tensor);

            relaxation->renormalize_v2_from_umn(kmesh_coarse.get(),
                                                kmap_coarse_to_dense,
                                                delta_v2_with_umn,
                                                del_v_strain,
                                                u_tensor);

            // renormalization by displacements
            relaxation->renormalize_v1_from_q0(omega2_harmonic,
                                               kmesh_coarse.get(),
                                               kmesh_dense.get(),
                                               v1_renorm,
                                               v1_with_umn,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            relaxation->renormalize_v2_from_q0(evec_harmonic,
                                               kmesh_coarse.get(),
                                               kmesh_dense.get(),
                                               kmap_coarse_to_dense,
                                               mat_transform_sym,
                                               delta_v2_renorm,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            relaxation->renormalize_v0_from_q0(omega2_harmonic,
                                               kmesh_dense.get(),
                                               v0_renorm,
                                               v0_with_umn,
                                               v1_with_umn,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            V0[iT] = v0_renorm;

            // calculate renormalizations of harmonic IFCs, which is stored in delta_harmonic_dymat_renormalize
            dynamical->compute_renormalized_harmonic_frequency(omega2_harm_renorm[iT],
                                                               evec_harm_renorm_tmp,
                                                               delta_v2_renorm,
                                                               omega2_harmonic,
                                                               evec_harmonic,
                                                               kmesh_coarse.get(),
                                                               kmesh_dense.get(),
                                                               kmap_coarse_to_dense,
                                                               mat_transform_sym,
                                                               mindist_list,
                                                               writes->getVerbosity());

            dynamical->calc_new_dymat_with_evec(delta_harmonic_dymat_renormalize[iT],
                                                omega2_harm_renorm[iT],
                                                evec_harm_renorm_tmp,
                                                kmesh_coarse.get(),
                                                kmap_coarse_to_dense);

            // copy delta_harmonic_dymat_renormalize to dymat_anharm
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    for (ik = 0; ik < kmesh_coarse->nk; ik++) {
                        dymat_anharm[iT][is1][is2][ik] = delta_harmonic_dymat_renormalize[iT][is1][is2][ik];
                    }
                }
            }
        }

        fout_q0.close();
        fout_u0.close();
        fout_u_tensor.close();
    }

    del_v0_del_umn_vib.clear();

    v1_vib.clear();
    v1_ref.clear();
    v1_with_umn.clear();
    v1_renorm.clear();

    omega2_harm_renorm.clear();
    evec_harm_renorm_tmp.clear();
    delta_v2_renorm.clear();
    delta_v2_with_umn.clear();

    v4_array_dummy.clear();
    v3_ref.clear();

    C1_array.clear();
    C2_array.clear();
    C3_array.clear();
}

void Qha::calc_del_v0_del_umn_vib(std::complex<double> *del_v0_del_umn_vib, const DelVStrainData &del_v_strain,
                                  double T_in)
{
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    static auto complex_zero = std::complex<double>(0.0, 0.0);

    int ixyz1, ixyz2, is, ik;
    std::complex<double> Qtmp;
    double omega1_tmp;

    double factor = 0.25 / static_cast<double>(nk);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            del_v0_del_umn_vib[ixyz1 * 3 + ixyz2] = complex_zero;

            for (is = 0; is < ns; is++) {
                for (ik = 0; ik < nk; ik++) {
                    omega1_tmp = std::sqrt(std::fabs(omega2_harmonic[ik][is]));

                    if (omega2_harmonic[ik][is] < 0 && omega1_tmp > eps8) {
                        std::cout << "Warning : Negative frequency is detected in perturbative QHA.\n";
                    }

                    if (std::abs(omega1_tmp) < eps8) {
                        Qtmp = 0.0;
                    } else {
                        const auto factor = thermodynamics->disp_corr_factor(omega1_tmp, T_in);
                        Qtmp = std::complex<double>(factor, 0.0);
                    }

                    del_v0_del_umn_vib[ixyz1 * 3 + ixyz2] +=
                        factor * del_v_strain.del_v2[ixyz1 * 3 + ixyz2](ik, is * ns + is) * Qtmp;
                }
            }
        }
    }
}

void Qha::calculate_del_v1_del_umn_renorm(std::complex<double> **del_v1_del_umn_renorm,
                                          const std::array<std::array<double, 3>, 3> &u_tensor,
                                          const DelVStrainData &del_v_strain, const std::vector<double> &q0)
{
    int ns = dynamical->neval;
    int nk = kmesh_dense->nk;
    std::vector<Eigen::MatrixXcd> del_v2_strain_tmp(9, Eigen::MatrixXcd::Zero(ns, ns));

    double factor = 0.5 * 4.0 * nk;
    int i1, i2, i3, ixyz1, ixyz2;
    int is1, is2, is3;

    const auto complex_zero = std::complex<double>(0.0, 0.0);

    // initialize
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_renorm[i1][is1] = complex_zero;
        }
    }

    // calculate renormalization from strain
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_renorm[i1][is1] = del_v_strain.del_v1(i1, is1);

            for (i2 = 0; i2 < 9; i2++) {
                del_v1_del_umn_renorm[i1][is1] += del_v_strain.del2_v1(i1 * 9 + i2, is1) * u_tensor[i2 / 3][i2 % 3];
                for (i3 = 0; i3 < 9; i3++) {
                    del_v1_del_umn_renorm[i1][is1] += 0.5 * del_v_strain.del3_v1(i1 * 81 + i2 * 9 + i3, is1) *
                                                      u_tensor[i2 / 3][i2 % 3] * u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    // first order in internal coordinate
    // preparation
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v2_strain_tmp[i1](is1, is2) = del_v_strain.del_v2[i1](0, is1 * ns + is2);

                for (i2 = 0; i2 < 9; i2++) {
                    del_v2_strain_tmp[i1](is1, is2) +=
                        del_v_strain.del2_v2[i1 * 9 + i2](0, is1 * ns + is2) * u_tensor[i2 / 3][i2 % 3];
                }
            }
        }
    }

    // add to del_v1_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v1_del_umn_renorm[i1][is1] += del_v2_strain_tmp[i1](is1, is2) * q0[is2];
            }
        }
    }

    // second order in internal coordinate
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {

            for (is2 = 0; is2 < ns; is2++) {
                for (is3 = 0; is3 < ns; is3++) {
                    del_v1_del_umn_renorm[i1][is1] +=
                        factor * del_v_strain.del_v3[i1][0](is1, is2 * ns + is3) * q0[is2] * q0[is3];
                }
            }
        }
    }


    // symmetrize with respect to the interchange of indices of strain tensor
    for (is1 = 0; is1 < ns; is1++) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = ixyz1 + 1; ixyz2 < 3; ixyz2++) {
                del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1] = 0.5 * (del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1] +
                                                                       del_v1_del_umn_renorm[ixyz2 * 3 + ixyz1][is1]);

                del_v1_del_umn_renorm[ixyz2 * 3 + ixyz1][is1] = del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1];
            }
        }
    }
}


void Qha::calculate_C2_array_renorm(double **C2_array_renorm, const std::array<std::array<double, 3>, 3> &u_tensor,
                                    std::array<std::array<double, 3>, 3> &eta_tensor, double **C2_array,
                                    double ***C3_array, const DelVStrainData &del_v_strain,
                                    const std::vector<double> &q0)
{
    int ns = dynamical->neval;
    NDArray<double, 2> del_eta_del_u;
    del_eta_del_u.resize(9, 9);

    int i1, i2, i3, i4, ixyz1, ixyz2, ixyz3, ixyz4;
    int is1, is2;

    NDArray<double, 2> C2_array_with_strain_eta;
    C2_array_with_strain_eta.resize(9, 9);


    // calculate the derivative of eta_tensor by u_tensor
    for (i1 = 0; i1 < 9; i1++) {
        ixyz1 = i1 / 3;
        ixyz2 = i1 % 3;
        for (i2 = 0; i2 < 9; i2++) {
            ixyz3 = i2 / 3;
            ixyz4 = i2 % 3;

            del_eta_del_u[i1][i2] = 0.0;

            if (ixyz1 == ixyz3 && ixyz2 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz2 == ixyz3 && ixyz1 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz1 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz2][ixyz4];
            }
            if (ixyz2 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz1][ixyz4];
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_with_strain_eta[i1][i2] = C2_array[i1][i2];
            for (i3 = 0; i3 < 9; i3++) {
                C2_array_with_strain_eta[i1][i2] += C3_array[i1][i2][i3] * eta_tensor[i3 / 3][i3 % 3];
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_renorm[i1][i2] = 0.0;

            for (i3 = 0; i3 < 9; i3++) {
                for (i4 = 0; i4 < 9; i4++) {
                    C2_array_renorm[i1][i2] +=
                        C2_array_with_strain_eta[i3][i4] * del_eta_del_u[i3][i1] * del_eta_del_u[i4][i2];
                }
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            for (is1 = 0; is1 < ns; is1++) {
                C2_array_renorm[i1][i2] += del_v_strain.del2_v1(i1 * 9 + i2, is1).real() * q0[is1];
            }

            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    C2_array_renorm[i1][i2] +=
                        0.5 * del_v_strain.del2_v2[i1 * 9 + i2](0, is1 * ns + is2).real() * q0[is1] * q0[is2];
                }
            }

            for (is1 = 0; is1 < ns; is1++) {
                for (i3 = 0; i3 < 9; i3++) {
                    C2_array_renorm[i1][i2] +=
                        del_v_strain.del3_v1(i1 * 81 + i2 * 9 + i3, is1).real() * q0[is1] * u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    C2_array_with_strain_eta.clear();
    del_eta_del_u.clear();
}

void Qha::calculate_C2_array_ZSISA(double **C2_array_ZSISA, double **C2_array_renorm,
                                   std::complex<double> **del_v1_del_umn_renorm, double **delq_delu_ZSISA)
{
    // calculate ZSISA second-order elastic constants,
    // which is the elastic constants with ZSISA internal coordinates.

    int i1, i2, is;
    int ns = dynamical->neval;

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_ZSISA[i1][i2] = C2_array_renorm[i1][i2];
            for (is = 0; is < ns; is++) {
                C2_array_ZSISA[i1][i2] += del_v1_del_umn_renorm[i1][is].real() * delq_delu_ZSISA[is][i2];
            }
        }
    }
}


void Qha::compute_ZSISA_stress(double **delq_delu_ZSISA_out, std::complex<double> *del_v0_del_umn_ZSISA_out,
                               std::complex<double> ***cmat_convert, double **omega2_harm_renorm_in,
                               std::complex<double> *del_v0_del_umn_QHA, std::complex<double> **del_v1_del_umn_renorm,
                               std::complex<double> *v1_QHA, std::vector<int> &harm_optical_modes)
{
    using namespace Eigen;
    int i1;
    int is, js;

    const auto ns = dynamical->neval;


    MatrixXcd Cmat(ns, ns), v2_mat_full(ns, ns);
    MatrixXcd v2_mat_optical(ns - 3, ns - 3);
    VectorXcd vec_del_V1_strain(ns - 3);
    VectorXcd vec_delq_delu_ZSISA(ns - 3);


    // calculate delq_delu_ZSISA_out
    for (i1 = 0; i1 < 9; i1++) {

        for (is = 0; is < ns; is++) {
            for (js = 0; js < ns; js++) {
                Cmat(js, is) = cmat_convert[0][is][js]; // transpose
                v2_mat_full(is, js) = 0.0;
            }
            v2_mat_full(is, is) = omega2_harm_renorm_in[0][is];
        }
        v2_mat_full = Cmat.adjoint() * v2_mat_full * Cmat;

        for (is = 0; is < ns - 3; is++) {
            for (js = 0; js < ns - 3; js++) {
                v2_mat_optical(is, js) = v2_mat_full(harm_optical_modes[is], harm_optical_modes[js]);
            }
        }

        for (is = 0; is < ns - 3; is++) {
            vec_del_V1_strain(is) = del_v1_del_umn_renorm[i1][harm_optical_modes[is]];
        }

        vec_delq_delu_ZSISA = -1.0 * v2_mat_optical.colPivHouseholderQr().solve(vec_del_V1_strain);

        for (is = 0; is < ns; is++) {
            delq_delu_ZSISA_out[is][i1] = 0.0;
        }
        for (is = 0; is < ns - 3; is++) {
            delq_delu_ZSISA_out[harm_optical_modes[is]][i1] = vec_delq_delu_ZSISA(is).real();
        }
    }

    // calculate ZSISA stress tensor
    for (i1 = 0; i1 < 9; i1++) {

        del_v0_del_umn_ZSISA_out[i1] = del_v0_del_umn_QHA[i1];

        // add correction to QHA stress tensor
        for (is = 0; is < ns - 3; is++) {
            del_v0_del_umn_ZSISA_out[i1] +=
                v1_QHA[harm_optical_modes[is]] * delq_delu_ZSISA_out[harm_optical_modes[is]][i1];
        }
    }
}

void Qha::compute_vZSISA_stress(std::complex<double> *del_v0_del_umn_vZSISA, double **C2_array_ZSISA,
                                std::complex<double> *del_v0_del_umn_renorm, std::complex<double> *del_v0_del_umn_ZSISA,
                                const std::array<std::array<double, 3>, 3> &u_tensor)
{
    using namespace Eigen;

    int i1, i2;
    int is1, is2, ixyz1, ixyz2, ixyz3, ixyz4;
    int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;

    double F_tensor[3][3]; // F_{mu nu} = delta_{mu nu} + u_{mu nu}
    double ddetF_dumn[9], u_tilde[9], delta_umn_vZSISA[9];
    VectorXcd ddetF_dumn_vec(6), delta_umn_vZSISA_vec(6);
    double factor_tmp;
    double deltaF_vZSISA, deltaU_vZSISA;

    MatrixXcd C2_mat_tmp(6, 6);

    // calculate ddetF_dumn = (d det(I+u))/(du)
    for (i1 = 0; i1 < 3; i1++) {
        for (i2 = 0; i2 < 3; i2++) {
            F_tensor[i1][i2] = u_tensor[i1][i2];
        }
        F_tensor[i1][i1] += 1.0;
    }
    for (i1 = 0; i1 < 9; i1++) {
        is1 = i1 / 3;
        is2 = i1 % 3;
        ixyz1 = (is1 + 1) % 3;
        ixyz2 = (is1 + 2) % 3;
        ixyz3 = (is2 + 1) % 3;
        ixyz4 = (is2 + 2) % 3;

        ddetF_dumn[i1] =
            F_tensor[ixyz1][ixyz3] * F_tensor[ixyz2][ixyz4] - F_tensor[ixyz1][ixyz4] * F_tensor[ixyz2][ixyz3];
    }


    factor_tmp = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        factor_tmp += ddetF_dumn[i1] * ddetF_dumn[i1];
    }
    factor_tmp = 1.0 / std::sqrt(factor_tmp);

    for (i1 = 0; i1 < 9; i1++) {
        u_tilde[i1] = factor_tmp * ddetF_dumn[i1];
    }

    // calcualte delta_umn_vZSISA
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        ddetF_dumn_vec(itmp1) = ddetF_dumn[itmp1 * 3 + itmp1];

        itmp2 = (itmp1 + 1) % 3;
        itmp3 = (itmp1 + 2) % 3;
        ddetF_dumn_vec(itmp1 + 3) = ddetF_dumn[itmp2 * 3 + itmp3];
    }

    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            C2_mat_tmp(itmp1, itmp2) = C2_array_ZSISA[itmp1 * 3 + itmp1][itmp2 * 3 + itmp2];
        }
    }
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            itmp3 = (itmp2 + 1) % 3;
            itmp4 = (itmp2 + 2) % 3;
            C2_mat_tmp(itmp1, itmp2 + 3) = 2.0 * C2_array_ZSISA[itmp1 * 3 + itmp1][itmp3 * 3 + itmp4];
            C2_mat_tmp(itmp2 + 3, itmp1) = C2_array_ZSISA[itmp3 * 3 + itmp4][itmp1 * 3 + itmp1];
        }
    }
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            itmp3 = (itmp1 + 1) % 3;
            itmp4 = (itmp1 + 2) % 3;
            itmp5 = (itmp2 + 1) % 3;
            itmp6 = (itmp2 + 2) % 3;
            C2_mat_tmp(itmp1 + 3, itmp2 + 3) = 2.0 * C2_array_ZSISA[itmp3 * 3 + itmp4][itmp5 * 3 + itmp6];
        }
    }

    // solve linear equation for delta_umn_vZSISA
    delta_umn_vZSISA_vec = C2_mat_tmp.colPivHouseholderQr().solve(ddetF_dumn_vec);
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        itmp3 = (itmp1 + 1) % 3;
        itmp4 = (itmp1 + 2) % 3;

        delta_umn_vZSISA[itmp1 * 3 + itmp1] = delta_umn_vZSISA_vec(itmp1).real();
        delta_umn_vZSISA[itmp3 * 3 + itmp4] = delta_umn_vZSISA_vec(itmp1 + 3).real();
        delta_umn_vZSISA[itmp3 + itmp4 * 3] = delta_umn_vZSISA_vec(itmp1 + 3).real();
    }

    // normalize delta_umn_vZSISA
    factor_tmp = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        factor_tmp += u_tilde[i1] * delta_umn_vZSISA[i1];
    }
    factor_tmp = 1.0 / factor_tmp;

    for (i1 = 0; i1 < 9; i1++) {
        delta_umn_vZSISA[i1] *= factor_tmp;
    }

    // calculate delta_q0_vZSISA

    deltaF_vZSISA = 0.0;
    deltaU_vZSISA = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        deltaF_vZSISA += delta_umn_vZSISA[i1] * del_v0_del_umn_ZSISA[i1].real();
        deltaU_vZSISA += u_tilde[i1] * del_v0_del_umn_renorm[i1].real();
    }

    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_vZSISA[i1] = del_v0_del_umn_renorm[i1] + u_tilde[i1] * (deltaF_vZSISA - deltaU_vZSISA);
    }
}


void Qha::calc_v1_vib(std::complex<double> *v1_vib, std::complex<double> ***v3_ref, const double T_in)
{
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    static auto complex_zero = std::complex<double>(0.0, 0.0);

    int is1, is2, ik;
    std::complex<double> Qtmp;
    double omega1_tmp;


    for (is1 = 0; is1 < ns; is1++) {
        v1_vib[is1] = complex_zero;

        for (is2 = 0; is2 < ns; is2++) {
            for (ik = 0; ik < nk; ik++) {
                omega1_tmp = std::sqrt(std::fabs(omega2_harmonic[ik][is2]));

                if (omega2_harmonic[ik][is2] < 0 && omega1_tmp > eps8) {
                    std::cout << "Warning : Negative frequency is detected in perturbative QHA.\n";
                }

                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                } else {
                    const auto factor = thermodynamics->disp_corr_factor(omega1_tmp, T_in);
                    Qtmp = std::complex<double>(factor, 0.0);
                }

                v1_vib[is1] += v3_ref[ik][is1][is2 * ns + is2] * Qtmp;
            }
        }
    }
}

void Qha::compute_cmat(std::complex<double> ***cmat_convert, const std::complex<double> *const *const *const evec_new)
{

    using namespace Eigen;

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    int ik, is, js;
    MatrixXcd evec_mat_original(ns, ns);

    for (ik = 0; ik < nk; ++ik) {

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                evec_mat_original(is, js) = evec_harmonic[ik][js][is];
            }
        }

        build_cmat_at_k(ns, evec_mat_original, evec_new[ik], cmat_convert[ik]);
    }
}
