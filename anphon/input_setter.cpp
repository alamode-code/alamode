/*
 input_setter.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "input_setter.h"
#include "anharmonic_core.h"
#include "conductivity.h"
#include "constants.h"
#include "dielec.h"
#include "dynamical.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "gruneisen.h"
#include "integration.h"
#include "isotope.h"
#include "iterativebte.h"
#include "kpoint.h"
#include "mode_analysis.h"
#include "mode_symmetry.h"
#include "phonon.h"
#include "phonon_dos.h"
#include "phonon_velocity.h"
#include "qha.h"
#include "relaxation.h"
#include "relaxation_types.h"
#include "scph.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

InputSetter::InputSetter() = default;

InputSetter::~InputSetter() = default;

void InputSetter::set_general_vars(PHON *phon, const GeneralInputVars &vars) const
{
    phon->job_title = vars.prefix;
    phon->mode = vars.mode;
    phon->use_hdf5_io = vars.use_hdf5_io;
    phon->allow_unconverged = vars.allow_unconverged;

    const auto file_result = vars.prefix + ".result";
    const auto file_result4 = vars.prefix + ".4ph.result";
    const auto file_kappa_h5 = vars.prefix + ".kappa.h5";

    phon->conductivity->set_conductivity_params(file_result,
                                                file_result4,
                                                file_kappa_h5,
                                                vars.restart,
                                                vars.restart_4ph,
                                                vars.use_hdf5_io);

    phon->symmetry->tolerance = vars.tolerance;
    phon->symmetry->printsymmetry = vars.printsymmetry;

    phon->system->Tmin = vars.Tmin;
    phon->system->Tmax = vars.Tmax;
    phon->system->dT = vars.dT;

    if (!vars.kdname.empty()) {
        phon->system->symbol_kd = vars.kdname;
    }
    if (!vars.masskd.empty()) {
        phon->system->mass_kd = vars.masskd;
    }

    if (vars.emin.has_value()) {
        phon->dos->emin = vars.emin.value();
        phon->dos->auto_set_emin = false;
    }
    if (vars.emax.has_value()) {
        phon->dos->emax = vars.emax.value();
        phon->dos->auto_set_emax = false;
    }
    phon->dos->delta_e = vars.delta_e;

    phon->dynamical->nonanalytic = vars.nonanalytic;
    phon->dynamical->na_sigma = vars.na_sigma;
    phon->dynamical->band_connection = vars.band_connection;
    phon->dielec->symmetrize_borncharge = vars.bornsym;
    phon->dielec->file_born = vars.borninfo;

    if (vars.nonanalytic == 3) {
        phon->ewald->is_longrange = true;
        phon->ewald->file_longrange = vars.borninfo;
        phon->ewald->prec_ewald = vars.prec_ewald;
        phon->ewald->rate_ab = 6.0 / pi;
    } else {
        phon->ewald->is_longrange = false;
    }

    phon->writes->nbands = vars.nbands;
    phon->set_verbosity(vars.verbosity);
    phon->writes->use_h5_io = vars.use_hdf5_io;

    phon->integration->epsilon = vars.epsilon;
    phon->integration->ismear = vars.ismear;

    phon->fcs_phonon->file_fcs = vars.fcsfile;
    phon->fcs_phonon->file_fc2 = vars.fc2file;
    phon->fcs_phonon->fc2_temperature = vars.fc2_temperature;
    phon->fcs_phonon->file_dfc2 = vars.dfc2file;
    phon->fcs_phonon->update_fc2 = !vars.fc2file.empty();

    phon->thermodynamics->classical = vars.classical;
    phon->anharmonic_core->use_triplet_symmetry = vars.use_triplet_symmetry;
}

void InputSetter::set_analysis_vars(PHON *phon, const AnalysisInputVars &vars) const
{
    phon->phonon_velocity->print_velocity = vars.print_vel;
    phon->dynamical->print_eigenvectors = vars.print_evec;
    phon->dynamical->participation_ratio = vars.participation_ratio;
    phon->mode_symmetry->print_irreps = vars.print_irreps;

    if (vars.projection_directions.has_value()) {
        phon->dynamical->set_projection_directions(vars.projection_directions.value());
    }

    phon->writes->setWriteOptions(vars.print_msd,
                                  vars.print_xsf,
                                  vars.print_anime,
                                  vars.anime_format,
                                  vars.anime_frames,
                                  vars.anime_cellsize,
                                  vars.anime_kpoint,
                                  vars.print_ucorr,
                                  vars.shift_ucorr,
                                  vars.print_zmode,
                                  vars.print_eval);

    phon->dos->compute_dos = vars.compute_dos;
    phon->dos->projected_dos = vars.projected_dos;
    phon->dos->two_phonon_dos = vars.two_phonon_dos;
    phon->dos->scattering_phase_space = vars.scattering_phase_space;
    phon->dos->longitudinal_projected_dos = vars.longitudinal_dos;

    phon->anharmonic_core->quartic_mode = vars.quartic_mode;
    phon->dielec->calc_dielectric_constant = vars.calc_dielectric_constant;

    phon->mode_analysis->ks_input = vars.ks_input;
    phon->mode_analysis->calc_realpart = vars.calc_realpart;
    phon->mode_analysis->calc_fstate_omega = vars.fstate_omega;
    phon->mode_analysis->print_V3 = vars.print_V3;
    phon->mode_analysis->print_V4 = vars.print_V4;
    phon->mode_analysis->spectral_func = vars.bubble_omega;
    phon->mode_analysis->calc_selfenergy = vars.calc_selfenergy;

    phon->gruneisen->print_gruneisen = vars.print_gruneisen;
    phon->gruneisen->print_newfcs = vars.print_newfcs;
    phon->gruneisen->delta_a = vars.delta_a;
    phon->thermodynamics->calc_FE_bubble = vars.calc_FE_bubble;

    phon->ewald->print_fc2_ewald = vars.print_fc2_ewald;

    if (phon->mode == "SCPH") {
        phon->scph->print_self_consistent_fc2 = vars.print_self_consistent_fc2;
    }
}

void InputSetter::set_kappa_vars(PHON *phon, const KappaInputVars &vars) const
{
    phon->anharmonic_core->quartic_mode = vars.quartic_mode;
    phon->conductivity->fph_rta = vars.include_4ph;

    if (vars.restart.has_value()) {
        phon->conductivity->set_restart_flag(3, vars.restart.value());
    }
    if (vars.restart_4ph.has_value()) {
        phon->conductivity->set_restart_flag(4, vars.restart_4ph.value());
    }

    phon->integration->epsilon_4ph = vars.epsilon_4ph;
    phon->integration->ismear_4ph = vars.ismear_4ph;
    phon->integration->adaptive_factor = vars.adaptive_factor;

    phon->conductivity->len_boundary = vars.len_boundary; // m
    phon->conductivity->calc_kappa_spec = vars.calc_kappa_spec;
    phon->conductivity->calc_coherent = vars.calc_coherent;
    phon->conductivity->write_interpolation = vars.write_interpol;

    phon->conductivity->solver_ibte = vars.solver == "IBTE" || vars.solver == "VBTE" || vars.solver == "DBTE";
    phon->iterativebte->use_variational = vars.solver == "VBTE";
    phon->iterativebte->use_direct = vars.solver == "DBTE";
    phon->iterativebte->isotope_inscattering = vars.isotope_inscattering != 0;
    phon->iterativebte->max_cycle = vars.max_cycle;
    phon->iterativebte->min_cycle = vars.min_cycle;
    phon->iterativebte->mixing_factor = vars.iterative_mixing;
    phon->iterativebte->convergence_criteria = vars.iter_threshold;

    phon->conductivity->set_interpolator(vars.interpolator);
    phon->conductivity->set_kmesh_coarse(vars.kmesh_coarse);

    if (vars.include_isotope && !vars.isotope_factor.empty()) {
        phon->isotope->isotope_factor = vars.isotope_factor;
    }
    phon->isotope->include_isotope = vars.include_isotope;
}

void InputSetter::set_scph_vars(PHON *phon, const ScphInputVars &vars) const
{
    for (auto i = 0; i < 3; ++i) {
        phon->scph->kmesh_scph[i] = vars.kmesh_scph[i];
        phon->scph->kmesh_interpolate[i] = vars.kmesh_interpolate[i];
    }
    phon->scph->mixalpha = vars.mixalpha;
    phon->scph->imix_scph = vars.imix_scph;
    phon->scph->maxiter = vars.maxiter;
    phon->scph->restart_scph = vars.restart_scph;
    phon->scph->use_h5_io = phon->use_hdf5_io;
    phon->scph->selfenergy_offdiagonal = vars.selfenergy_offdiagonal;
    phon->scph->ialgo = vars.ialgo;
    phon->scph->tolerance_scph = vars.tolerance_scph;
    phon->scph->lower_temp = vars.lower_temp;
    phon->scph->warmstart_scph = vars.warmstart;
    phon->scph->bubble = vars.bubble;
    phon->relaxation->relax_str = vars.relax_str;
}

void InputSetter::set_qha_vars(PHON *phon, const QhaInputVars &vars) const
{
    for (auto i = 0; i < 3; ++i) {
        phon->qha->kmesh_qha[i] = vars.kmesh_qha[i];
        phon->qha->kmesh_interpolate[i] = vars.kmesh_interpolate[i];
    }
    phon->qha->lower_temp = vars.lower_temp;
    phon->qha->qha_scheme = to_qha_scheme(vars.qha_scheme);
    phon->qha->selfenergy_offdiagonal = vars.selfenergy_offdiagonal;
    phon->qha->ialgo = vars.ialgo;
    phon->qha->restart_qha = vars.restart_qha;
    phon->qha->use_h5_io = phon->use_hdf5_io;
    phon->relaxation->relax_str = vars.relax_str;
}

void InputSetter::set_relax_vars(PHON *phon, const RelaxInputVars &vars) const
{
    phon->relaxation->relax_algo = vars.relax_algo;
    phon->relaxation->max_str_iter = vars.max_str_iter;

    phon->relaxation->coord_conv_tol = vars.coord_conv_tol;
    phon->relaxation->gradient_conv_tol = vars.gradient_conv_tol;
    phon->relaxation->cell_gradient_conv_tol = vars.cell_gradient_conv_tol;
    phon->relaxation->gdiis_control = vars.gdiis_control;
    phon->relaxation->mixbeta_coord = vars.mixbeta_coord;
    phon->relaxation->alpha_steepest_decent = vars.alpha_steepest_decent;

    phon->relaxation->cell_conv_tol = vars.cell_conv_tol;
    phon->relaxation->mixbeta_cell = vars.mixbeta_cell;

    phon->relaxation->set_init_str = vars.set_init_str;
    phon->relaxation->cooling_u0_index = vars.cooling_u0_index;
    phon->relaxation->cooling_u0_thr = vars.cooling_u0_thr;

    phon->relaxation->add_hess_diag = vars.add_hess_diag;
    phon->relaxation->stat_pressure = vars.stat_pressure;

    phon->relaxation->renorm_3to2nd = vars.renorm_3to2nd;
    phon->relaxation->renorm_2to1st = vars.renorm_2to1st;
    phon->relaxation->renorm_34to1st = vars.renorm_34to1st;

    phon->relaxation->strain_IFC_dir = vars.strain_IFC_dir;
}

void InputSetter::set_cell_parameter(PHON *phon, const double lavec_in[3][3]) const
{
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            phon->system->lavec_p_input(i, j) = lavec_in[i][j];
        }
    }
    phon->system->load_primitive_from_file = 1;
}

void InputSetter::set_kpoint_vars(PHON *phon, const int kpmode,
                                  const std::vector<std::vector<std::string>> &kpdata) const
{
    phon->kpoint->kpInp.clear();
    for (const auto &kpelem: kpdata) {
        phon->kpoint->kpInp.emplace_back(kpelem);
    }
    phon->kpoint->kpoint_mode = kpmode;
}

void InputSetter::set_initial_strain(PHON *phon, const double u_tensor_in[3][3]) const
{
    phon->relaxation->setInitialDistortion(u_tensor_in);
}

void InputSetter::set_initial_displacements(PHON *phon, const std::vector<std::vector<double>> &u_xyz) const
{
    phon->relaxation->init_u0.clear();
    for (const auto &u_atom: u_xyz) {
        for (auto ixyz = 0; ixyz < 3; ++ixyz) {
            phon->relaxation->init_u0.push_back(u_atom[ixyz]);
        }
    }
}
