/*
write_phonons.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "write_phonons.h"
#include "phonon_velocity.h"

namespace
{
// PRINTVEL follows the transport velocity formulation: matrix diagonal by default,
// finite differences under the legacy opt-out.
bool use_velmat_velocities()
{
    return !PHON_NS::PhononVelocity::legacy_velocity();
}
} // namespace
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "constants.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "fcs_xml_schema.h"
#include "gruneisen.h"
#include "integration.h"
#include "isotope.h"
#include "iterativebte.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mode_analysis.h"
#include "mode_symmetry.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "phonon_velocity.h"
#include "qha.h"
#include "relaxation.h"
#include "scph.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "version.h"

#ifdef _HDF5

#include "H5Cpp.h"
#include "fcs_hdf5_schema.h"
#include "hdf5_parser.h"

#endif

using namespace PHON_NS;

Writes::Writes(PHON *phon_in) : phon(phon_in), run(phon_in->run_info)
{
    print_ucorr = false;
    print_xsf = false;
    print_anime = false;
    print_msd = false;
    print_zmode = false;
    print_eval = false;
    anime_cellsize[0] = 0;
    anime_cellsize[1] = 0;
    anime_cellsize[2] = 0;
    shift_ucorr[0] = 0;
    shift_ucorr[1] = 0;
    shift_ucorr[2] = 0;
    anime_kpoint[0] = 0.0;
    anime_kpoint[1] = 0.0;
    anime_kpoint[2] = 0.0;
    anime_frames = 20;
    anime_format = "xyz";
};

Writes::~Writes() {};

void Writes::writeInputVars()
{
    // Share the echo between the log and HDF5 metadata, even when quiet.
    std::ostringstream os;
    unsigned int i;

    // Every tag accepted by the input parser is echoed here, grouped by
    // input field, so that a log file documents the run completely.
    const auto print_mesh = [&os](const char *tag, const unsigned int mesh[3]) {
        os << "  " << tag << " = ";
        for (auto k = 0; k < 3; ++k) os << std::setw(5) << mesh[k];
        os << '\n';
    };

    os << '\n';
    os << " Input variables:\n";
    os << " -----------------------------------------------------------------\n";
    os << " General:\n";
    os << "  PREFIX = " << run.job_title << '\n';
    os << "  MODE = " << run.mode;
    if (phon->mode_analysis->selfenergy_mode) os << " (selfenergy)";
    os << '\n';
    os << "  FCSFILE = " << phon->fcs_phonon->file_fcs << '\n';
    if (phon->fcs_phonon->update_fc2) {
        os << "  FC2FILE = " << phon->fcs_phonon->file_fc2 << '\n';
    }
    if (!phon->fcs_phonon->file_fc3.empty()) {
        os << "  FC3FILE = " << phon->fcs_phonon->file_fc3 << '\n';
    }
    if (!phon->fcs_phonon->file_fc4.empty()) {
        os << "  FC4FILE = " << phon->fcs_phonon->file_fc4 << '\n';
    }
    if (!phon->fcs_phonon->file_dfc2.empty()) {
        os << "  DFC2FILE = " << phon->fcs_phonon->file_dfc2 << '\n';
    }
    if (phon->fcs_phonon->fc2_temperature >= 0.0) {
        os << "  FC2_TEMPERATURE = " << phon->fcs_phonon->fc2_temperature << '\n';
    }
    os << "  FILE_FORMAT = " << (run.use_hdf5_io ? "h5" : "text") << "; VERBOSITY = " << run.verbosity << '\n';
    os << '\n';

    // KD and MASS are echoed only when given in the input; otherwise they
    // are taken from the force constant file, which is read later.
    if (!phon->system->symbol_kd.empty()) {
        os << "  KD = ";
        for (i = 0; i < phon->system->symbol_kd.size(); ++i) os << std::setw(5) << phon->system->symbol_kd[i];
        os << '\n';
    }
    if (!phon->system->mass_kd.empty()) {
        os << "  MASS = ";
        for (i = 0; i < phon->system->mass_kd.size(); ++i) os << std::setw(10) << phon->system->mass_kd[i];
        os << '\n';
    }
    os << "  TREVSYM = " << phon->symmetry->use_time_reversal << '\n';
    os << "  TOLERANCE = " << phon->symmetry->tolerance << "; PRINTSYM = " << phon->symmetry->printsymmetry << '\n';
    os << '\n';

    os << "  NONANALYTIC = " << phon->dynamical->nonanalytic << '\n';
    if (phon->dynamical->nonanalytic) {
        os << "  BORNINFO = " << phon->dielec->file_born << "; NA_SIGMA = " << phon->dynamical->na_sigma
           << "; BORNSYM = " << phon->dielec->symmetrize_borncharge << '\n';
        if (phon->dynamical->nonanalytic == 3) {
            os << "  PREC_EWALD = " << phon->ewald->prec_ewald << '\n';
        }
    }
    os << '\n';
    if (nbands >= 0) {
        os << "  NBANDS = " << nbands << '\n';
    }

    os << "  TMIN = " << phon->system->Tmin << "; TMAX = " << phon->system->Tmax << "; DT = " << phon->system->dT << '\n';
    os << "  EMIN = " << phon->dos->emin << "; EMAX = " << phon->dos->emax << "; DELTA_E = " << phon->dos->delta_e << '\n';
    os << '\n';

    os << "  ISMEAR = " << phon->integration->ismear << "; EPSILON = " << phon->integration->epsilon << '\n';
    os << '\n';
    os << "  CLASSICAL = " << phon->thermodynamics->classical << '\n';
    os << "  BCONNECT = " << phon->dynamical->band_connection << '\n';
    if (run.mode == "SCPH" || run.mode == "QHA" || phon->fcs_phonon->fc2_temperature >= 0.0) {
        os << "  ALLOW_UNCONVERGED = " << run.allow_unconverged << '\n';
    }
    os << '\n';

    if (run.mode == "KAPPA") {
        os << "  RESTART = " << phon->conductivity->get_restart_conductivity(3) << '\n';
        os << "  TRISYM = " << phon->anharmonic_core->use_triplet_symmetry << "\n\n";
    } else if (run.mode == "SCPH") {
        os << " Scph:" << '\n';
        print_mesh("KMESH_INTERPOLATE", phon->scph->kmesh_interpolate);
        print_mesh("KMESH_SCPH       ", phon->scph->kmesh_scph);
        os << "  SELF_OFFDIAG = " << phon->scph->selfenergy_offdiagonal << '\n';
        os << "  IALGO = " << phon->scph->ialgo << '\n';
        os << "  BUBBLE = " << phon->scph->bubble << '\n';
        if (phon->scph->bubble > 0) print_mesh("KMESH_BUBBLE     ", phon->scph->kmesh_bubble);
        os << '\n';
        os << "  RESTART_SCPH = " << phon->scph->restart_scph << '\n';
        os << "  LOWER_TEMP = " << phon->scph->lower_temp << '\n';
        os << "  WARMSTART = " << phon->scph->warmstart_scph << '\n' << '\n';
        os << "  TOL_SCPH = " << phon->scph->tolerance_scph << '\n';
        os << "  MAXITER = " << phon->scph->maxiter << '\n';
        os << "  MIXALPHA = " << phon->scph->mixalpha << '\n';
        os << "  IMIX = " << phon->scph->imix_scph << '\n';

        // variables related to structural optimization
        os << '\n';
        os << "  RELAX_STR = " << phon->relaxation->relax_str << '\n';
    } else if (run.mode == "QHA") {
        os << " QHA:" << '\n';
        print_mesh("KMESH_INTERPOLATE", phon->qha->kmesh_interpolate);
        print_mesh("KMESH_QHA        ", phon->qha->kmesh_qha);
        os << "  SELF_OFFDIAG = " << phon->qha->selfenergy_offdiagonal << '\n';
        os << "  IALGO = " << phon->qha->ialgo << '\n';
        os << "  RESTART_QHA = " << phon->qha->restart_qha << '\n';
        os << "  LOWER_TEMP = " << phon->qha->lower_temp << '\n';
        // variables related to structural optimization
        os << "  RELAX_STR = " << phon->relaxation->relax_str << '\n';
    }
    os << '\n';

    if ((run.mode == "SCPH" || run.mode == "QHA") && phon->relaxation->relax_str != 0) {
        os << " Structure_opt:" << '\n';

        os << "  RELAX_ALGO = " << phon->relaxation->relax_algo << '\n';
        os << "  MAX_STR_ITER = " << phon->relaxation->max_str_iter << '\n';
        os << "  COORD_CONV_TOL = " << phon->relaxation->coord_conv_tol << '\n';
        if (phon->relaxation->gradient_conv_tol > 0.0) {
            os << "  GRADIENT_CONV_TOL = " << phon->relaxation->gradient_conv_tol << '\n';
        }
        if (phon->relaxation->relax_str == 2) {
            os << "  CELL_CONV_TOL = " << phon->relaxation->cell_conv_tol << '\n';
            if (phon->relaxation->cell_gradient_conv_tol > 0.0) {
                os << "  CELL_GRADIENT_CONV_TOL = " << phon->relaxation->cell_gradient_conv_tol << '\n';
            }
        }
        if (phon->relaxation->relax_algo == 1) {
            os << "  ALPHA_STDECENT = " << phon->relaxation->alpha_steepest_decent << '\n';
        } else if (phon->relaxation->relax_algo == 2) {
            os << "  MIXBETA_COORD = " << phon->relaxation->mixbeta_coord << '\n';
            if (phon->relaxation->relax_str == 2) {
                os << "  MIXBETA_CELL = " << phon->relaxation->mixbeta_cell << '\n';
            }
        } else if (phon->relaxation->relax_algo == 3) {
            os << "  GDIIS_PLAIN = " << (phon->relaxation->gdiis_control ? 0 : 1) << '\n';
        }

        os << "  SET_INIT_STR = " << phon->relaxation->set_init_str << '\n';

        os << "  ADD_HESS_DIAG = " << phon->relaxation->add_hess_diag << '\n';
        os << "  STAT_PRESSURE = " << phon->relaxation->stat_pressure << '\n';

        if (run.mode == "QHA" && phon->relaxation->relax_str == 2) {
            os << "  QHA_SCHEME = " << to_int(phon->qha->qha_scheme) << '\n';
        }
        if (uses_strain_coupling(to_relaxation_str_mode(phon->relaxation->relax_str))) {
            if (phon->relaxation->strain_coupling >= 0) {
                os << "  STRAIN_COUPLING = " << phon->relaxation->strain_coupling << '\n';
            } else {
                os << "  STRAIN_COUPLING = (set by the deprecated RENORM_*/ELASTIC_CONST tags)\n";
            }
            os << "    elastic constants C2, C3        : "
               << (phon->relaxation->elastic_const == 2 ? "file" : "harmonic and cubic IFCs") << '\n';
            os << "    strain-force coupling dV1/du    : "
               << (phon->relaxation->renorm_2to1st == 2   ? "file"
                   : phon->relaxation->renorm_2to1st == 1 ? "harmonic IFCs (needs rotational invariance)"
                                                    : "zero")
               << '\n';
            os << "    d2V1/du2, d3V1/du3              : "
               << (phon->relaxation->renorm_34to1st == 1 ? "cubic and quartic IFCs (needs rotational invariance)" : "zero")
               << '\n';
            os << "    strain-harmonic coupling dV2/du : "
               << (phon->relaxation->renorm_3to2nd == 1   ? "cubic IFCs"
                   : phon->relaxation->renorm_3to2nd == 4 ? "k-space file (B_array_kspace.txt)"
                                                    : "file")
               << '\n';
            if (!phon->relaxation->strain_file.empty()) {
                os << "  STRAINFILE = " << phon->relaxation->strain_file << '\n';
            } else {
                os << "  STRAIN_IFC_DIR = " << phon->relaxation->strain_IFC_dir << '\n';
            }
        }
        os << '\n';
    }


    os << " Kpoint:" << '\n';
    if (phon->mode_analysis->selfenergy_mode) {
        os << "  KPMODE (1st entry for &kpoint) = " << phon->kpoint->target_mode << '\n';
    } else {
        os << "  KPMODE (1st entry for &kpoint) = " << phon->kpoint->kpoint_mode << '\n';
    }
    os << '\n';
    os << '\n';

    if (phon->mode_analysis->selfenergy_mode) {
        const auto &ma = *phon->mode_analysis;
        os << " Selfenergy:" << '\n';
        os << "  KMESH = ";
        if (!phon->kpoint->kpInp.empty()) {
            for (const auto &str: phon->kpoint->kpInp[0].kpelem) os << std::setw(5) << str;
        }
        os << '\n';
        os << "  BRANCHES = " << ma.branches_spec << '\n';
        os << "  LINEWIDTH = " << ma.linewidth_requested << "; SHIFT = " << ma.calc_realpart
           << "; SELF_W = " << ma.spectral_func << "; FSTATE_W = " << ma.calc_fstate_omega << '\n';
        os << "  PRINTV3 = " << ma.print_V3 << "; PRINTV4 = " << ma.print_V4 << '\n';
        os << "  INTERPOLATE = " << ma.interpolate << '\n';
        if (ma.interpolate) {
            print_mesh("KMESH_COARSE", ma.kmesh_coarse);
            os << "  OMEGA_RANGE = ";
            for (i = 0; i < 3; ++i) os << std::setw(10) << ma.omega_range[i];
            os << '\n';
        }
        os << '\n';
    }

    if (run.mode == "KAPPA" && !phon->mode_analysis->selfenergy_mode) {
        std::string solver = "RTA";
        if (phon->conductivity->solver_ibte) {
            solver = phon->iterativebte->use_direct ? "DBTE" : phon->iterativebte->use_variational ? "VBTE" : "IBTE";
        }
        os << " Kappa:" << '\n';
        os << "  SOLVER = " << solver << '\n';
        if (phon->conductivity->solver_ibte) {
            os << "  MAX_CYCLE = " << phon->iterativebte->max_cycle << "; MIN_CYCLE = " << phon->iterativebte->min_cycle
               << "; ITER_THRESHOLD = " << phon->iterativebte->convergence_criteria
               << "; IBTE_MIXING = " << phon->iterativebte->mixing_factor << '\n';
        }
        os << '\n';
        os << "  ISOTOPE = " << phon->isotope->include_isotope << '\n';
        if (phon->isotope->include_isotope) {
            // Without ISOFACT the natural-abundance factors are set up later.
            if (!phon->isotope->isotope_factor.empty()) {
                os << "  ISOFACT = ";
                for (i = 0; i < phon->isotope->isotope_factor.size(); ++i) {
                    os << std::scientific << std::setw(13) << phon->isotope->isotope_factor[i];
                }
                os << std::defaultfloat << '\n';
            }
            if (phon->conductivity->solver_ibte) {
                os << "  ISOTOPE_INSCATTERING = " << phon->iterativebte->isotope_inscattering << '\n';
            }
        }
        os << "  LEN_BOUNDARY = " << phon->conductivity->len_boundary << '\n';
        os << "  KAPPA_SPEC = " << phon->conductivity->calc_kappa_spec << "; KAPPA_COHERENT = " << phon->conductivity->calc_coherent
           << '\n';
        if (phon->integration->ismear == 2 || (phon->conductivity->fph_rta > 0 && phon->integration->ismear_4ph == 2)) {
            os << "  ADAPTIVE_FACTOR = " << phon->integration->adaptive_factor << '\n';
        }
        os << '\n';
        os << "  INCLUDE_4PH = " << phon->conductivity->fph_rta << '\n';
        if (phon->conductivity->fph_rta > 0) {
            print_mesh("KMESH_COARSE", phon->conductivity->get_nk_coarse());
            os << "  ISMEAR_4PH = " << phon->integration->ismear_4ph << "; EPSILON_4PH = " << phon->integration->epsilon_4ph
               << '\n';
            os << "  INTERPOLATOR = " << phon->conductivity->get_interpolator()
               << "; WRITE_INTERPOL = " << phon->conductivity->write_interpolation << '\n';
            os << "  RESTART_4PH = " << phon->conductivity->get_restart_conductivity(4) << '\n';
        }
        os << '\n';
    }

    if (run.mode == "PHONONS" || (run.mode == "KAPPA" && !phon->mode_analysis->ks_input.empty())) {
        os << " Analysis:" << '\n';
    }
    if (run.mode == "PHONONS") {
        os << "  PRINTEVAL = " << print_eval << "; PRINTEVEC = " << phon->dynamical->print_eigenvectors
           << "; PRINTVEL = " << phon->phonon_velocity->print_velocity << '\n';
        os << "  PRINTPR = " << phon->dynamical->participation_ratio << "; PRINTXSF = " << print_xsf
           << "; ZMODE = " << print_zmode << '\n';
        os << "  IRREPS = " << phon->mode_symmetry->print_irreps << "; DIELEC = " << phon->dielec->calc_dielectric_constant
           << "; FC2_EWALD = " << phon->ewald->print_fc2_ewald << '\n';
        const auto &proj = phon->dynamical->get_projection_directions();
        if (!proj.empty()) {
            os << "  PROJECTION_AXES = ";
            for (const auto &axis: proj) {
                os << " [";
                for (const auto &x: axis) os << std::setw(8) << x;
                os << " ]";
            }
            os << '\n';
        }
        os << '\n';

        if (print_anime) {
            os << "  ANIME = ";
            for (i = 0; i < 3; ++i) os << std::setw(5) << anime_kpoint[i];
            os << '\n';
            print_mesh("ANIME_CELLSIZE", anime_cellsize);
            os << "  ANIME_FORMAT = " << anime_format << "; ANIME_FRAMES = " << anime_frames << '\n';
            os << '\n';
        }

        if (phon->kpoint->kpoint_mode == 2) {
            os << "  DOS = " << phon->dos->compute_dos << "; PDOS = " << phon->dos->projected_dos
               << "; TDOS = " << phon->dos->two_phonon_dos << "; LONGITUDINAL_DOS = " << phon->dos->longitudinal_projected_dos
               << '\n';
            os << "  SPS = " << phon->dos->scattering_phase_space << "; FE_BUBBLE = " << phon->thermodynamics->calc_FE_bubble
               << '\n';
            os << "  PRINTMSD = " << print_msd << "; UCORR = " << print_ucorr;
            if (print_ucorr) {
                os << "; SHIFT_UCORR =";
                for (i = 0; i < 3; ++i) os << std::setw(4) << shift_ucorr[i];
            }
            os << '\n';
            os << '\n';
        }
        os << "  GRUNEISEN = " << phon->gruneisen->gruneisen_mode << "; NEWFCS = " << phon->gruneisen->print_newfcs << '\n';
        if (phon->gruneisen->gruneisen_mode > 0 || phon->gruneisen->print_newfcs) {
            os << "  SUBLATTICE_RELAX = " << phon->gruneisen->sublattice_relax << "; DELTA_A = " << phon->gruneisen->delta_a
               << '\n';
        }
        if (phon->gruneisen->print_newfcs) {
            os << "  QUARTIC = " << phon->anharmonic_core->quartic_mode << '\n';
        }

    } else if (run.mode == "KAPPA") {
        // Legacy mode analysis driven by KS_INPUT in the &analysis field
        // (MODE = selfenergy lists its own tags above).
        if (!phon->mode_analysis->ks_input.empty()) {
            const auto &ma = *phon->mode_analysis;
            os << "  KS_INPUT = " << ma.ks_input << '\n';
            os << "  QUARTIC = " << phon->anharmonic_core->quartic_mode << "; REALPART = " << ma.calc_realpart
               << "; SELF_W = " << ma.spectral_func << "; FSTATE_W = " << ma.calc_fstate_omega << '\n';
            os << "  PRINTV3 = " << ma.print_V3 << "; PRINTV4 = " << ma.print_V4 << '\n';
        }
    } else if (run.mode == "SCPH") {
        // Do nothing
    } else if (run.mode == "QHA") {
        // Do nothing
    } else {
        exit("writeInputVars", "This cannot happen");
    }

    os << "\n\n";
    os << " -----------------------------------------------------------------\n\n";

#ifdef _HDF5
    phon->run_info.input_variables = parse_input_echo(os.str());
#endif
    if (run.verbosity > 0) std::cout << os.str();
}


void Writes::setWriteOptions(const bool print_msd_, const bool print_xsf_, const bool print_anime_,
                             const std::string &anime_format_, const int anime_frames_,
                             const unsigned int anime_cellsize_[3], const double anime_kpoint_[3],
                             const bool print_ucorr_, const int shift_ucorr_[3], const bool print_zmode_,
                             const bool print_eval_)
{
    print_msd = print_msd_;
    print_xsf = print_xsf_;
    print_anime = print_anime_;
    anime_format = anime_format_;
    anime_frames = anime_frames_;
    print_ucorr = print_ucorr_;
    print_zmode = print_zmode_;
    print_eval = print_eval_;

    for (auto i = 0; i < 3; ++i) {
        anime_cellsize[i] = anime_cellsize_[i];
        anime_kpoint[i] = anime_kpoint_[i];
        shift_ucorr[i] = shift_ucorr_[i];
    }
}

bool Writes::getPrintMSD() const
{
    return print_msd;
}

bool Writes::getPrintUcorr() const
{
    return print_ucorr;
}

std::array<int, 3> Writes::getShiftUcorr() const
{
    return {shift_ucorr[0], shift_ucorr[1], shift_ucorr[2]};
}

void Writes::printPhononEnergies() const
{
    if (run.verbosity == 0) return;

    unsigned int i;
    unsigned int ik, is;
    const auto ns = phon->dynamical->neval;

    const auto kayser_to_THz = 0.0299792458;

    std::cout << '\n';
    std::cout << " -----------------------------------------------------------------\n\n";
    std::cout << " Phonon frequencies below:\n\n";

    if (phon->kpoint->kpoint_mode == 0) {

        auto nk_now = phon->kpoint->kpoint_general->nk;
        auto &xk_now = phon->kpoint->kpoint_general->xk;
        auto eval_now = phon->dynamical->dymat_general->get_eigenvalues();

        for (ik = 0; ik < nk_now; ++ik) {
            std::cout << " # k point " << std::setw(5) << ik + 1;
            std::cout << " : (";

            for (i = 0; i < 3; ++i) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << xk_now[ik][i];
                if (i < 2) std::cout << ",";
            }
            std::cout << ")\n";

            std::cout << "   Mode, Frequency \n";

            for (is = 0; is < ns; ++is) {
                std::cout << std::setw(7) << is + 1;
                std::cout << std::fixed << std::setprecision(4) << std::setw(12) << in_kayser(eval_now[ik][is]);
                std::cout << " cm^-1  (";
                std::cout << std::fixed << std::setprecision(4) << std::setw(12)
                          << kayser_to_THz * in_kayser(eval_now[ik][is]);
                std::cout << " THz )\n";
            }
            std::cout << '\n';
        }

    } else if (phon->kpoint->kpoint_bs.get()) {

        auto nk = phon->kpoint->kpoint_bs->nk;

        for (ik = 0; ik < nk; ++ik) {
            std::cout << " # k point " << std::setw(5) << ik + 1;
            std::cout << " : (";

            for (i = 0; i < 3; ++i) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << phon->kpoint->kpoint_bs->xk[ik][i];
                if (i < 2) std::cout << ",";
            }
            std::cout << ")\n";

            std::cout << "   Mode, Frequency \n";

            for (is = 0; is < ns; ++is) {
                std::cout << std::setw(7) << is + 1;
                std::cout << std::fixed << std::setprecision(4) << std::setw(12)
                          << in_kayser(phon->dynamical->dymat_band->get_eigenvalues()[ik][is]);
                std::cout << " cm^-1  (";
                std::cout << std::fixed << std::setprecision(4) << std::setw(12)
                          << kayser_to_THz * in_kayser(phon->dynamical->dymat_band->get_eigenvalues()[ik][is]);
                std::cout << " THz )\n";
            }
            std::cout << '\n';
        }

    } else if (phon->kpoint->kpoint_mode == 2) {

        for (ik = 0; ik < phon->dos->kmesh_dos->kpoint_irred_all.size(); ++ik) {

            std::cout << " # Irred. k point" << std::setw(5) << ik + 1;
            std::cout << " : (";

            for (i = 0; i < 3; ++i) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(8)
                          << phon->dos->kmesh_dos->kpoint_irred_all[ik][0].kval[i];
                if (i < 2) std::cout << ",";
            }
            std::cout << ")\n";

            std::cout << "   Mode, Frequency \n";

            const auto knum = phon->dos->kmesh_dos->kpoint_irred_all[ik][0].knum;

            for (is = 0; is < ns; ++is) {
                std::cout << std::setw(7) << is + 1;
                std::cout << std::fixed << std::setprecision(4) << std::setw(12)
                          << in_kayser(phon->dos->dymat_dos->get_eigenvalues()[knum][is]);
                std::cout << " cm^-1  (";
                std::cout << std::fixed << std::setprecision(4) << std::setw(12)
                          << kayser_to_THz * in_kayser(phon->dos->dymat_dos->get_eigenvalues()[knum][is]);
                std::cout << " THz )\n";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

void Writes::writePhononInfo()
{
    if (nbands < 0) {
        nbands = 3 * phon->system->get_primcell().number_of_atoms;
    }

    if (print_anime) {
        writeNormalModeAnimation(anime_kpoint, anime_cellsize);
    }

    if (run.verbosity > 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n\n";
        std::cout << " The following files are created: \n";
    }

    if (phon->kpoint->kpoint_mode == 1) {
        writePhononBands();
    }

    if (phon->phonon_velocity->print_velocity) {
        if (phon->kpoint->kpoint_bs.get()) {
            writePhononVel();
        }
        if (phon->dos->kmesh_dos.get()) {
            writePhononVelAll();
        }
    }

    if (phon->dos->flag_dos) {

        if (phon->dos->compute_dos || phon->dos->projected_dos) {
            writePhononDos();
        }

        if (phon->dos->two_phonon_dos) {
            writeTwoPhononDos();
        }

        if (phon->dos->longitudinal_projected_dos) {
            writeLongitudinalProjDos();
        }

        if (phon->dos->scattering_phase_space == 1) {
            writeScatteringPhaseSpace();
        } else if (phon->dos->scattering_phase_space == 2) {
            writeScatteringAmplitude();
        }

        writeThermodynamicFunc();
        if (print_msd) writeMSD();
        if (print_ucorr) writeDispCorrelation();
    }

    if (print_xsf) {
        writeNormalModeDirection();
    }

    // FILE_FORMAT rule: h5 (default) writes the schema-stamped HDF5
    // variants, text writes the plain-text files. Builds without HDF5
    // always fall back to text.
    if (phon->dynamical->print_eigenvectors) {
#ifdef _HDF5
        if (run.use_hdf5_io) {
            writeEigenvectorsHdf5();
        } else {
            writeEigenvectors();
        }
#else
        writeEigenvectors();
#endif
    }

    if (print_eval) {
#ifdef _HDF5
        if (run.use_hdf5_io) {
            writeEigenvaluesHdf5();
        } else {
            writeEigenvalues();
        }
#else
        writeEigenvalues();
#endif
    }

    if (phon->dynamical->participation_ratio) {
        writeParticipationRatio();
    }

    if (phon->gruneisen->gruneisen_mode > 0) {
        writeGruneisen();
    }

    if (phon->dielec->calc_dielectric_constant) {
        writeDielectricFunction();
    }

    if (print_anime && run.verbosity > 0) {
        if (anime_format == "XSF" || anime_format == "AXSF") {
            std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left
                      << run.job_title + ".anime*.axsf";
            std::cout << " : AXSF files for animate phonon modes\n";
        } else if (anime_format == "XYZ") {
            std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left
                      << run.job_title + ".anime*.xyz";
            std::cout << " : XYZ files for animate phonon modes\n";
        }
    }

    if (print_zmode) {
        printNormalmodeBorncharge();
    }

    if (phon->mode_symmetry->print_irreps) {
        writeModeIrreps();
    }
}

void Writes::writePhononBands() const
{
    std::ofstream ofs_bands;
    auto file_bands = run.job_title + ".bands";

    ofs_bands.open(file_bands.c_str(), std::ios::out);
    if (!ofs_bands) exit("writePhononBands", "cannot open file_bands");

    unsigned int i, j;
    const auto nk = phon->kpoint->kpoint_bs->nk;
    const auto &kaxis = phon->kpoint->kpoint_bs->kaxis;
    const auto eval = phon->dynamical->dymat_band->get_eigenvalues();

    auto kcount = 0;

    std::string str_tmp = "NONE";
    std::string str_kpath;
    std::string str_kval;

    for (i = 0; i < phon->kpoint->kpInp.size(); ++i) {
        if (str_tmp != phon->kpoint->kpInp[i].kpelem[0]) {
            str_tmp = phon->kpoint->kpInp[i].kpelem[0];
            str_kpath += " " + str_tmp;

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << kaxis[kcount];
            str_kval += " " + ss.str();
        }
        kcount += std::atoi(phon->kpoint->kpInp[i].kpelem[8].c_str());

        if (str_tmp != phon->kpoint->kpInp[i].kpelem[4]) {
            str_tmp = phon->kpoint->kpInp[i].kpelem[4];
            str_kpath += " " + str_tmp;

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << kaxis[kcount - 1];
            str_kval += " " + ss.str();
        }
    }

    ofs_bands << "# " << str_kpath << '\n';
    ofs_bands << "#" << str_kval << '\n';
    ofs_bands << "# k-axis, Eigenvalues [cm^-1]\n";

    if (phon->dynamical->band_connection == 0) {
        for (i = 0; i < nk; ++i) {
            ofs_bands << std::setw(8) << std::fixed << kaxis[i];
            for (j = 0; j < nbands; ++j) {
                ofs_bands << std::setw(15) << std::scientific << in_kayser(eval[i][j]);
            }
            ofs_bands << '\n';
        }
    } else {
        for (i = 0; i < nk; ++i) {
            ofs_bands << std::setw(8) << std::fixed << kaxis[i];
            for (j = 0; j < nbands; ++j) {
                ofs_bands << std::setw(15) << std::scientific << in_kayser(eval[i][phon->dynamical->index_bconnect[i][j]]);
            }
            ofs_bands << '\n';
        }
    }

    ofs_bands.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_bands;
        std::cout << " : Phonon band structure\n";
    }

    if (phon->dynamical->band_connection == 2) {
        std::ofstream ofs_connect;
        auto file_connect = run.job_title + ".connection";

        ofs_connect.open(file_connect.c_str(), std::ios::out);
        if (!ofs_connect) exit("writePhononBands", "cannot open file_connect");

        ofs_connect << "# " << str_kpath << '\n';
        ofs_connect << "#" << str_kval << '\n';
        ofs_connect << "# k-axis, mapping\n";

        for (i = 0; i < nk; ++i) {
            ofs_connect << std::setw(8) << std::fixed << kaxis[i];
            for (j = 0; j < nbands; ++j) {
                ofs_connect << std::setw(5) << phon->dynamical->index_bconnect[i][j] + 1;
            }
            ofs_connect << '\n';
        }
        ofs_connect.close();
        if (run.verbosity > 0) {
            std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_connect;
            std::cout << " : Connectivity map information of band dispersion\n";
        }
    }
}

void Writes::writePhononVel() const
{
    std::ofstream ofs_vel;
    auto file_vel = run.job_title + ".phvel";

    ofs_vel.open(file_vel.c_str(), std::ios::out);
    if (!ofs_vel) exit("writePhononVel", "cannot open file_vel");

    const auto nk = phon->kpoint->kpoint_bs->nk;
    const auto &kaxis = phon->kpoint->kpoint_bs->kaxis;
    const auto Ry_to_SI_vel = Bohr_in_Angstrom * 1.0e-10 / time_ry;

    NDArray<double, 2> phvel_bs;
    phvel_bs.resize(nk, phon->dynamical->neval);

    // Same velocity machinery as the transport terms. This makes
    // the printed velocities come from the same source; it does NOT make them reproduce
    // the conductivity, which treats degenerate multiplets as blocks having no per-mode
    // velocity. Printed values at a degeneracy remain one admissible basis choice.
    if (use_velmat_velocities()) {
        phon->phonon_velocity->get_phonon_group_velocity_bandstructure_velmat(phon->kpoint->kpoint_bs.get(),
                                                                        phon->system->get_primcell().lattice_vector,
                                                                        phon->fcs_phonon->force_constant_with_cell[0],
                                                                        phvel_bs);
    } else {
        phon->phonon_velocity->get_phonon_group_velocity_bandstructure(phon->kpoint->kpoint_bs.get(),
                                                                 phon->system->get_primcell().lattice_vector,
                                                                 phon->system->get_primcell().reciprocal_lattice_vector,
                                                                 phon->fcs_phonon->force_constant_with_cell[0],
                                                                 phon->ewald->fc2_without_dipole,
                                                                 phvel_bs);
    }

    ofs_vel << "# k-axis, |Velocity| [m / sec]\n";
    ofs_vel.setf(std::ios::fixed);

    if (phon->dynamical->band_connection == 0) {
        for (auto i = 0; i < nk; ++i) {
            ofs_vel << std::setw(8) << kaxis[i];
            for (auto j = 0; j < nbands; ++j) {
                ofs_vel << std::setw(15) << std::abs(phvel_bs[i][j] * Ry_to_SI_vel);
            }
            ofs_vel << '\n';
        }
    } else {
        for (auto i = 0; i < nk; ++i) {
            ofs_vel << std::setw(8) << kaxis[i];
            for (auto j = 0; j < nbands; ++j) {
                ofs_vel << std::setw(15) << std::abs(phvel_bs[i][phon->dynamical->index_bconnect[i][j]] * Ry_to_SI_vel);
            }
            ofs_vel << '\n';
        }
    }

    ofs_vel.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_vel;
        std::cout << " : Phonon velocity along given k path\n";
    }

    phvel_bs.clear();
}

void Writes::writePhononVelAll() const
{
    std::ofstream ofs_vel;
    auto file_vel = run.job_title + ".phvel_all";

    ofs_vel.open(file_vel.c_str(), std::ios::out);
    if (!ofs_vel) exit("writePhononVelAll", "cannot open file_vel_all");

    const auto nk = phon->dos->kmesh_dos->nk;
    const auto nk_irred = phon->dos->kmesh_dos->nk_irred;
    const auto ns = phon->dynamical->neval;
    const auto Ry_to_SI_vel = Bohr_in_Angstrom * 1.0e-10 / time_ry;
    const auto eval = phon->dos->dymat_dos->get_eigenvalues();

    NDArray<double, 3> phvel_xyz;
    NDArray<double, 2> phvel;

    phvel.resize(nk, ns);
    phvel_xyz.resize(nk, ns, 3);

    if (use_velmat_velocities()) {
        phon->phonon_velocity->get_phonon_group_velocity_mesh_velmat(*phon->dos->kmesh_dos.get(),
                                                               phon->system->get_primcell().lattice_vector,
                                                               phvel_xyz);
    } else {
        phon->phonon_velocity->get_phonon_group_velocity_mesh(*phon->dos->kmesh_dos.get(),
                                                        phon->system->get_primcell().lattice_vector,
                                                        false,
                                                        phvel_xyz);
    }
    unsigned int ik, is;
#ifdef _OPENMP
#pragma omp parallel for private(is)
#endif
    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            phvel[ik][is] =
                std::sqrt(pow2(phvel_xyz[ik][is][0]) + pow2(phvel_xyz[ik][is][1]) + pow2(phvel_xyz[ik][is][2]));
        }
    }

    ofs_vel << "# Phonon group velocity at all reducible k points.\n";
    ofs_vel << "# irred. knum, knum, mode num, frequency [cm^-1], "
               "|velocity| [m/sec], velocity_(x,y,z) [m/sec]\n\n";
    ofs_vel.setf(std::ios::fixed);

    for (unsigned int i = 0; i < nk_irred; ++i) {
        ofs_vel << "# Irreducible k point  : " << std::setw(8) << i + 1;
        ofs_vel << " (" << std::setw(4) << phon->dos->kmesh_dos->kpoint_irred_all[i].size() << ")\n";

        for (unsigned int j = 0; j < phon->dos->kmesh_dos->kpoint_irred_all[i].size(); ++j) {
            const auto knum = phon->dos->kmesh_dos->kpoint_irred_all[i][j].knum;

            ofs_vel << "## xk =    ";
            for (auto k = 0; k < 3; ++k)
                ofs_vel << std::setw(15) << std::fixed << std::setprecision(10) << phon->dos->kmesh_dos->xk[knum][k];
            ofs_vel << '\n';

            for (auto k = 0; k < ns; ++k) {
                ofs_vel << std::setw(7) << i + 1;
                ofs_vel << std::setw(8) << knum + 1;
                ofs_vel << std::setw(5) << k + 1;
                ofs_vel << std::setw(10) << std::fixed << std::setprecision(2) << in_kayser(eval[knum][k]);
                ofs_vel << std::setw(10) << std::fixed << std::setprecision(2) << phvel[knum][k] * Ry_to_SI_vel;
                for (auto ii = 0; ii < 3; ++ii) {
                    ofs_vel << std::setw(10) << std::fixed << std::setprecision(2)
                            << phvel_xyz[knum][k][ii] * Ry_to_SI_vel;
                }
                ofs_vel << '\n';
            }
            ofs_vel << '\n';
        }

        ofs_vel << '\n';
    }

    ofs_vel.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_vel;
        std::cout << " : Phonon velocity at all k points\n";
    }

    phvel.clear();
    phvel_xyz.clear();
}


void Writes::writePhononDos() const
{
    int i;
    std::ofstream ofs_dos;
    auto file_dos = run.job_title + ".dos";

    ofs_dos.open(file_dos.c_str(), std::ios::out);
    if (!ofs_dos) exit("writePhononDos", "cannot open file_dos");

    ofs_dos << "#";
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        ofs_dos << std::setw(5) << phon->system->symbol_kd[i];
    }
    ofs_dos << '\n';
    ofs_dos << "#";

    NDArray<unsigned int, 1> nat_each_kd;
    nat_each_kd.resize(phon->system->get_primcell().number_of_elems);
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) nat_each_kd[i] = 0;
    for (i = 0; i < phon->system->get_primcell().number_of_atoms; ++i) {
        //        ++nat_each_kd[phon->system->get_supercell(0).kind[phon->system->get_map_p2s(0)[i][0]]];
        ++nat_each_kd[phon->system->get_primcell().kind[i]];
    }
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        ofs_dos << std::setw(5) << nat_each_kd[i];
    }
    ofs_dos << '\n';
    nat_each_kd.clear();

    if (phon->dos->compute_dos) {
        ofs_dos << "# Energy [cm^-1], TOTAL-DOS";
    } else {
        ofs_dos << "# Energy [cm^-1]";
    }
    if (phon->dos->projected_dos) {
        ofs_dos << ", Atom Projected-DOS";
    }
    ofs_dos << '\n';
    ofs_dos.setf(std::ios::scientific);

    for (i = 0; i < phon->dos->n_energy; ++i) {
        ofs_dos << std::setw(15) << phon->dos->energy_dos[i];
        if (phon->dos->compute_dos) {
            ofs_dos << std::setw(15) << phon->dos->dos_phonon[i];
        }
        if (phon->dos->projected_dos) {
            for (auto iat = 0; iat < phon->system->get_primcell().number_of_atoms; ++iat) {
                ofs_dos << std::setw(15) << phon->dos->pdos_phonon[iat][i];
            }
        }
        ofs_dos << '\n';
    }
    ofs_dos.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_dos;

        if (phon->dos->projected_dos & phon->dos->compute_dos) {
            std::cout << " : Phonon DOS and atom projected DOS\n";
        } else if (phon->dos->projected_dos) {
            std::cout << " : Atom projected phonon DOS\n";
        } else {
            std::cout << " : Phonon DOS\n";
        }
    }
}

void Writes::writeTwoPhononDos() const
{
    std::ofstream ofs_tdos;
    auto file_tdos = run.job_title + ".tdos";
    ofs_tdos.open(file_tdos.c_str(), std::ios::out);

    ofs_tdos << "# Two-phonon DOS (TDOS) for all irreducible k points. \n";
    ofs_tdos << "# Energy [cm^-1], emission delta(e-e1-e2), absorption delta (e-e1+e2)\n";

    const auto n = phon->dos->n_energy;

    for (auto ik = 0; ik < phon->dos->kmesh_dos->nk_irred; ++ik) {

        ofs_tdos << "# Irred. kpoint : " << std::setw(5) << ik + 1 << '\n';
        for (auto i = 0; i < n; ++i) {
            ofs_tdos << std::setw(15) << phon->dos->emin + phon->dos->delta_e * static_cast<double>(i);

            for (auto j = 0; j < 2; ++j) ofs_tdos << std::setw(15) << phon->dos->dos2_phonon[ik][i][j];
            ofs_tdos << '\n';
        }
        ofs_tdos << '\n';
    }

    ofs_tdos.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_tdos;
        std::cout << " : Two-phonon DOS\n";
    }
}

void Writes::writeScatteringPhaseSpace() const
{
    std::ofstream ofs_sps;

    auto file_sps = run.job_title + ".sps";
    ofs_sps.open(file_sps.c_str(), std::ios::out);

    ofs_sps << "# Total scattering phase space (cm): " << std::scientific << phon->dos->total_sps3 << '\n';
    ofs_sps << "# Mode decomposed scattering phase space are printed below.\n";
    ofs_sps << "# Irred. k, mode, omega (cm^-1), P+ (absorption) (cm), P- (emission) (cm)\n";

    for (auto ik = 0; ik < phon->dos->kmesh_dos->nk_irred; ++ik) {
        const auto knum = phon->dos->kmesh_dos->kpoint_irred_all[ik][0].knum;

        for (auto is = 0; is < phon->dynamical->neval; ++is) {
            ofs_sps << std::setw(5) << ik + 1;
            ofs_sps << std::setw(5) << is + 1;
            ofs_sps << std::setw(15) << in_kayser(phon->dos->dymat_dos->get_eigenvalues()[knum][is]);
            ofs_sps << std::setw(15) << std::scientific << phon->dos->sps3_mode[ik][is][1];
            ofs_sps << std::setw(15) << std::scientific << phon->dos->sps3_mode[ik][is][0];
            ofs_sps << '\n';
        }
        ofs_sps << '\n';
    }

    ofs_sps.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_sps;
        std::cout << " : Three-phonon scattering phase space\n";
    }
}

void Writes::writeLongitudinalProjDos() const
{
    int i;
    std::ofstream ofs_dos;
    auto file_dos = run.job_title + ".longitudinal_dos";

    ofs_dos.open(file_dos.c_str(), std::ios::out);
    if (!ofs_dos) exit("writeLongitudinalProjDos", "cannot open file_dos");

    ofs_dos << "# Energy [cm^-1], LONGITUDINAL-PROJECTED DOS\n";
    ofs_dos.setf(std::ios::scientific);

    for (i = 0; i < phon->dos->n_energy; ++i) {
        ofs_dos << std::setw(15) << phon->dos->energy_dos[i];
        ofs_dos << std::setw(15) << phon->dos->longitude_dos[i];
        ofs_dos << '\n';
    }
    ofs_dos.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_dos;
        std::cout << " : Longitudinal projected DOS" << '\n';
    }
}

void Writes::writeScatteringAmplitude() const
{
    int i, j;
    unsigned int knum;
    const auto ns = phon->dynamical->neval;

    auto file_w = run.job_title + ".sps_Bose";
    std::ofstream ofs_w;

    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    ofs_w.open(file_w.c_str(), std::ios::out);

    ofs_w << "# Scattering phase space with the Bose-Einstein distribution function\n";
    ofs_w << "# Irreducible kpoints \n";
    for (i = 0; i < phon->dos->kmesh_dos->kpoint_irred_all.size(); ++i) {
        ofs_w << "#" << std::setw(5) << i + 1;

        knum = phon->dos->kmesh_dos->kpoint_irred_all[i][0].knum;
        for (j = 0; j < 3; ++j) ofs_w << std::setw(15) << phon->dos->kmesh_dos->xk[knum][j];
        ofs_w << '\n';
    }
    ofs_w << '\n';
    ofs_w << "# k, mode, frequency (cm^-1), temperature, W+ (absorption) (cm), W- (emission) (cm)\n\n";

    for (i = 0; i < phon->dos->kmesh_dos->kpoint_irred_all.size(); ++i) {

        knum = phon->dos->kmesh_dos->kpoint_irred_all[i][0].knum;

        for (unsigned int is = 0; is < ns; ++is) {

            const auto omega = in_kayser(phon->dos->dymat_dos->get_eigenvalues()[knum][is]);

            for (j = 0; j < NT; ++j) {
                ofs_w << std::setw(5) << i + 1 << std::setw(5) << is + 1 << std::setw(15) << omega;
                ofs_w << std::setw(8) << Tmin + static_cast<double>(j) * dT;
                ofs_w << std::setw(15) << phon->dos->sps3_with_bose[i][is][j][1];
                ofs_w << std::setw(15) << phon->dos->sps3_with_bose[i][is][j][0];
                ofs_w << '\n';
            }
            ofs_w << '\n';
        }
        ofs_w << '\n';
    }

    ofs_w.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_w;
        std::cout << " : Three-phonon scattering phase space \n";
        std::cout << " " << std::setw(run.job_title.length() + 16) << " "
                  << "with the Bose distribution function\n";
    }
}

void Writes::writeNormalModeDirection() const
{
    std::string fname_axsf;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_axsf = run.job_title + ".axsf";
        writeNormalModeDirectionEach(fname_axsf,
                                     phon->kpoint->kpoint_general->nk,
                                     phon->dynamical->dymat_general->get_eigenvectors());
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_axsf = run.job_title + ".band.axsf";
        writeNormalModeDirectionEach(fname_axsf, phon->kpoint->kpoint_bs->nk, phon->dynamical->dymat_band->get_eigenvectors());
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_axsf = run.job_title + ".mesh.axsf";
        writeNormalModeDirectionEach(fname_axsf, phon->dos->kmesh_dos->nk, phon->dos->dymat_dos->get_eigenvectors());
    }
}

void Writes::writeNormalModeDirectionEach(const std::string &fname_axsf, const unsigned int nk_in,
                                          const std::complex<double> *const *const *evec_in) const
{
    std::ofstream ofs_anime;

    ofs_anime.open(fname_axsf.c_str(), std::ios::out);
    if (!ofs_anime) exit("writeNormalModeDirectionEach", "cannot open fname_axsf");

    ofs_anime.setf(std::ios::scientific);

    unsigned int i, j, k;
    const auto natmin = phon->system->get_primcell().number_of_atoms;
    const auto force_factor = 100.0;

    NDArray<double, 2> xmod;
    NDArray<std::string, 1> kd_tmp;

    xmod.resize(natmin, 3);
    kd_tmp.resize(natmin);

    ofs_anime << "ANIMSTEPS " << nbands * nk_in << '\n';
    ofs_anime << "CRYSTAL\n";
    ofs_anime << "PRIMVEC\n";

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_anime << std::setw(15) << phon->system->get_primcell().lattice_vector(j, i) * Bohr_in_Angstrom;
        }
        ofs_anime << '\n';
    }

    for (i = 0; i < natmin; ++i) {
        k = phon->system->get_map_p2s(0)[i][0];
        for (j = 0; j < 3; ++j) {
            xmod[i][j] = phon->system->get_supercell(0).x_cartesian(k, j);
        }

        for (j = 0; j < 3; ++j) {
            xmod[i][j] *= Bohr_in_Angstrom;
        }
        kd_tmp[i] = phon->system->symbol_kd[phon->system->get_primcell().kind[i]];
    }

    i = 0;

    for (unsigned int ik = 0; ik < nk_in; ++ik) {
        for (unsigned int imode = 0; imode < nbands; ++imode) {
            ofs_anime << "PRIMCOORD " << std::setw(10) << i + 1 << '\n';
            ofs_anime << std::setw(10) << natmin << std::setw(10) << 1 << '\n';
            auto norm = 0.0;

            for (j = 0; j < 3 * natmin; ++j) {
                auto evec_tmp = evec_in[ik][imode][j];
                norm += pow2(evec_tmp.real()) + pow2(evec_tmp.imag());
            }

            norm *= force_factor / static_cast<double>(natmin);

            for (j = 0; j < natmin; ++j) {

                const auto m = phon->system->get_map_p2s(0)[j][0];

                ofs_anime << std::setw(10) << kd_tmp[j];

                for (k = 0; k < 3; ++k) {
                    ofs_anime << std::setw(15) << xmod[j][k];
                }
                for (k = 0; k < 3; ++k) {
                    ofs_anime << std::setw(15)
                              << evec_in[ik][imode][3 * j + k].real() / (std::sqrt(phon->system->get_mass_super()[m]) * norm);
                }
                ofs_anime << '\n';
            }

            ++i;
        }
    }

    xmod.clear();
    kd_tmp.clear();

    ofs_anime.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_axsf;
        std::cout << " : XcrysDen AXSF file to visualize phonon mode directions\n";
    }
}

void Writes::writeEigenvalues() const
{
    std::string fname_eval;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_eval = run.job_title + ".eval";
        writeEigenvaluesEach(fname_eval,
                             phon->kpoint->kpoint_general->nk,
                             phon->kpoint->kpoint_general->xk,
                             phon->dynamical->dymat_general->get_eigenvalues());
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_eval = run.job_title + ".band.eval";
        writeEigenvaluesEach(fname_eval,
                             phon->kpoint->kpoint_bs->nk,
                             phon->kpoint->kpoint_bs->xk,
                             phon->dynamical->dymat_band->get_eigenvalues());
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_eval = run.job_title + ".mesh.eval";
        writeEigenvaluesEach(fname_eval, phon->dos->kmesh_dos->nk, phon->dos->kmesh_dos->xk, phon->dos->dymat_dos->get_eigenvalues());
    }
}

void Writes::writeEigenvaluesEach(const std::string &fname_eval, const unsigned int nk_in, const double *const *xk_in,
                                  const double *const *eval_in) const
{
    unsigned int i, j, k;
    std::ofstream ofs_eval;

    ofs_eval.open(fname_eval.c_str(), std::ios::out);
    if (!ofs_eval) exit("writeEigenvaluesEach", "cannot open file_eval");
    ofs_eval.setf(std::ios::scientific);

    ofs_eval << "# Lattice vectors of the primitive cell\n";

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_eval << std::setw(15) << phon->system->get_primcell().lattice_vector(j, i);
        }
        ofs_eval << '\n';
    }

    ofs_eval << '\n';
    ofs_eval << "# Reciprocal lattice vectors of the primitive cell\n";

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_eval << std::setw(15) << phon->system->get_primcell().reciprocal_lattice_vector(i, j);
        }
        ofs_eval << '\n';
    }

    ofs_eval << '\n';
    ofs_eval << "# Number of phonon modes: " << std::setw(10) << nbands << '\n';
    ofs_eval << "# Number of k points : " << std::setw(10) << nk_in << '\n';
    ofs_eval << "# Number of atomic kinds : " << std::setw(4) << phon->system->get_primcell().number_of_elems << '\n';
    ofs_eval << "# Atomic masses :";
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        ofs_eval << std::setw(15) << phon->system->mass_kd[i];
    }
    ofs_eval << "\n\n";
    ofs_eval << "# Eigenvalues (omega^2) for each phonon modes below:\n\n";

    NDArray<unsigned int, 2> index_bconnect_tmp;
    index_bconnect_tmp.resize(nk_in, nbands);

    if (phon->dynamical->index_bconnect) {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = phon->dynamical->index_bconnect[i][j];
            }
        }
    } else {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = j;
            }
        }
    }

    for (i = 0; i < nk_in; ++i) {
        ofs_eval << "## kpoint " << std::setw(7) << i + 1 << " : ";
        for (j = 0; j < 3; ++j) {
            ofs_eval << std::setw(15) << xk_in[i][j];
        }
        ofs_eval << '\n';
        for (j = 0; j < nbands; ++j) {

            k = index_bconnect_tmp[i][j];

            auto omega2 = eval_in[i][k];
            if (omega2 >= 0.0) {
                omega2 = omega2 * omega2;
            } else {
                omega2 = -omega2 * omega2;
            }

            ofs_eval << std::setw(8) << j + 1 << " : ";
            ofs_eval << std::setw(15) << omega2 << '\n';
        }
        ofs_eval << '\n';
    }
    ofs_eval.close();

    index_bconnect_tmp.clear();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_eval;
        std::cout << " : Eigenvalues of all k points\n";
    }
}

#ifdef _HDF5

// Chunked + shuffle + deflate property list for the large per-k arrays.
static auto compressed_plist(const std::vector<size_t> &dims) -> H5::DSetCreatPropList
{
    const auto chunk = h5_chunk_dims(dims, sizeof(double));
    H5::DSetCreatPropList plist;
    plist.setChunk(static_cast<int>(chunk.size()), chunk.data());
    plist.setShuffle();
    plist.setDeflate(1);
    return plist;
}

void Writes::writeEigenvaluesHdf5() const
{
    std::string fname_eval;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_eval = run.job_title + ".eval.hdf5";
        writeEigenvaluesEachHdf5(fname_eval,
                                 phon->kpoint->kpoint_general->nk,
                                 phon->kpoint->kpoint_general->xk,
                                 phon->dynamical->dymat_general->get_eigenvalues(),
                                 0);
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_eval = run.job_title + ".band.eval.hdf5";
        writeEigenvaluesEachHdf5(fname_eval,
                                 phon->kpoint->kpoint_bs->nk,
                                 phon->kpoint->kpoint_bs->xk,
                                 phon->dynamical->dymat_band->get_eigenvalues(),
                                 1);
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_eval = run.job_title + ".mesh.eval.hdf5";
        writeEigenvaluesEachHdf5(fname_eval,
                                 phon->dos->kmesh_dos->nk,
                                 phon->dos->kmesh_dos->xk,
                                 phon->dos->dymat_dos->get_eigenvalues(),
                                 2);
    }
}

void Writes::writeEigenvaluesEachHdf5(const std::string &fname_eval, const unsigned int nk_in,
                                      const double *const *xk_in, const double *const *eval_in,
                                      const unsigned int kpmode_in) const
{
    using namespace H5;

    unsigned int i, j, k;

    H5File file(fname_eval, H5F_ACC_TRUNC);
    Group group_cell(file.createGroup("/PrimitiveCell"));
    Group group_band(file.createGroup("/Eigenvalues"));
    Group group_kpoint(file.createGroup("/Kpoints"));

    // Write setting information
    hid_t str_datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_datatype, H5T_VARIABLE);
    std::vector<const char *> arr_c_str;
    for (unsigned int ii = 0; ii < phon->system->get_primcell().number_of_elems; ++ii) {
        arr_c_str.push_back(phon->system->symbol_kd[ii].c_str());
    }
    hsize_t str_dim[1]{arr_c_str.size()};
    DataSpace dataspace(1, str_dim);
    DataSet dataset(group_cell.createDataSet("elements", str_datatype, dataspace));
    dataset.write(&arr_c_str[0], str_datatype);
    dataset.close();
    dataspace.close();

    std::vector<double> mass_tmp;
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        mass_tmp.push_back(phon->system->mass_kd[i]);
    }
    dataspace = DataSpace(1, str_dim);
    dataset = DataSet(group_cell.createDataSet("masses", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(&mass_tmp[0], PredType::NATIVE_DOUBLE);
    dataset.close();
    dataspace.close();


    // Write primitive cell information
    hsize_t dims[2];
    dims[0] = 3;
    dims[1] = 3;
    double lavec_tmp[3][3];
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            lavec_tmp[i][j] = phon->system->get_primcell().lattice_vector(j, i);
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_cell.createDataSet("lattice_vector", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(lavec_tmp, PredType::NATIVE_DOUBLE);
    DataSpace attr_dataspace_str(H5S_SCALAR);
    Attribute myatt_in = dataset.createAttribute("unit", str_datatype, attr_dataspace_str);
    myatt_in.write(str_datatype, std::string("bohr"));
    myatt_in.close();
    dataset.close();
    dataspace.close();

    dims[0] = phon->system->get_primcell().number_of_atoms;
    dims[1] = 3;
    std::vector<double> xfrac_1D(dims[0] * dims[1]);
    hsize_t counter = 0;

    double xtmp[3];
    for (i = 0; i < phon->system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) xtmp[j] = phon->system->get_supercell(0).x_fractional(phon->system->get_map_p2s(0)[i][0], j);
        rotvec(xtmp, xtmp, phon->system->get_supercell(0).lattice_vector);
        rotvec(xtmp, xtmp, phon->system->get_primcell().reciprocal_lattice_vector);
        for (j = 0; j < 3; ++j) xtmp[j] /= 2.0 * pi;
        for (j = 0; j < 3; ++j) {
            while (xtmp[j] >= 1.0) {
                xtmp[j] -= 1.0;
            }
            while (xtmp[j] < 0.0) {
                xtmp[j] += 1.0;
            }
        }
        for (j = 0; j < 3; ++j) {
            xfrac_1D[counter++] = xtmp[j];
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_cell.createDataSet("fractional_coordinate", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(&xfrac_1D[0], PredType::NATIVE_DOUBLE);
    dataset.close();
    dataspace.close();

    hsize_t dims2[1];
    dims2[0] = phon->system->get_primcell().number_of_atoms;
    dataspace = DataSpace(1, dims2);
    dataset = DataSet(group_cell.createDataSet("atomic_kinds", PredType::NATIVE_INT, dataspace));
    std::vector<int> kdtmp(dims[0]);
    for (i = 0; i < phon->system->get_primcell().number_of_atoms; ++i) {
        kdtmp[i] = phon->system->get_primcell().kind[i];
    }

    dataset.write(&kdtmp[0], PredType::NATIVE_INT);
    dataset.close();

    // write eigenvalues

    NDArray<unsigned int, 2> index_bconnect_tmp;
    int band_index_reordered = 0;
    index_bconnect_tmp.resize(nk_in, nbands);

    if (phon->dynamical->index_bconnect) {
        band_index_reordered = 1;
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = phon->dynamical->index_bconnect[i][j];
            }
        }
    } else {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = j;
            }
        }
    }

    // Write band structure information
    dims[0] = nk_in;
    dims[1] = nbands;

    NDArray<double, 2> freq_kayser;
    freq_kayser.resize(nk_in, nbands);

    for (i = 0; i < nk_in; ++i) {
        for (j = 0; j < nbands; ++j) {
            k = index_bconnect_tmp[i][j];
            freq_kayser[i][j] = in_kayser(eval_in[i][k]);
        }
    }

    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_band.createDataSet("frequencies",
                                               PredType::NATIVE_DOUBLE,
                                               dataspace,
                                               compressed_plist({nk_in, static_cast<size_t>(nbands)})));
    IntType int_type(PredType::NATIVE_INT);
    DataSpace attr_dataspace_int(H5S_SCALAR);
    myatt_in = dataset.createAttribute("band_index_reordered", int_type, attr_dataspace_int);
    myatt_in.write(int_type, &band_index_reordered);
    myatt_in = dataset.createAttribute("unit", str_datatype, attr_dataspace_str);
    myatt_in.write(str_datatype, std::string("kayser (cm^-1)"));

    dataset.write(&freq_kayser[0][0], PredType::NATIVE_DOUBLE);
    myatt_in.close();
    dataset.close();
    dataspace.close();
    freq_kayser.clear();

    index_bconnect_tmp.clear();

    group_cell.close();
    group_band.close();

    dims[0] = nk_in;
    dims[1] = 3;
    std::vector<double> xk_1D(dims[0] * dims[1]);
    counter = 0;

    for (i = 0; i < nk_in; ++i) {
        for (j = 0; j < 3; ++j) {
            xk_1D[counter++] = xk_in[i][j];
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_kpoint.createDataSet("kpoint_coordinates", PredType::NATIVE_DOUBLE, dataspace));

    myatt_in = dataset.createAttribute("kpoint_mode", int_type, attr_dataspace_int);
    myatt_in.write(int_type, &kpmode_in);
    dataset.write(&xk_1D[0], PredType::NATIVE_DOUBLE);
    myatt_in.close();
    dataset.close();
    dataspace.close();

    if (kpmode_in == 1 && phon->kpoint->kpoint_bs.get()) {
        const auto &kaxis = phon->kpoint->kpoint_bs->kaxis;
        dims2[0] = nk_in;
        dataspace = DataSpace(1, dims2);
        dataset = DataSet(group_kpoint.createDataSet("bandstructure_xaxis", PredType::NATIVE_DOUBLE, dataspace));
        dataset.write(&kaxis[0], PredType::NATIVE_DOUBLE);
        dataset.close();
        dataspace.close();
    }

    group_kpoint.close();
    file.close();

    // Versioned schema stamp (same convention as the kappa/scph state files).
    {
        HighFive::File fh(fname_eval, HighFive::File::ReadWrite);
        stamp_h5_schema(fh, h5_schema_eigenvalues, h5_version_eigen);
        write_input_variables_h5(fh, run.input_variables);
    }

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_eval;
        std::cout << " : Eigenvalues of all k points (HDF5)\n";
    }
}

#endif

void Writes::writeEigenvectors() const
{
    std::string fname_evec;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_evec = run.job_title + ".evec";
        writeEigenvectorsEach(fname_evec,
                              phon->kpoint->kpoint_general->nk,
                              phon->kpoint->kpoint_general->xk,
                              phon->dynamical->dymat_general->get_eigenvalues(),
                              phon->dynamical->dymat_general->get_eigenvectors());
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_evec = run.job_title + ".band.evec";
        writeEigenvectorsEach(fname_evec,
                              phon->kpoint->kpoint_bs->nk,
                              phon->kpoint->kpoint_bs->xk,
                              phon->dynamical->dymat_band->get_eigenvalues(),
                              phon->dynamical->dymat_band->get_eigenvectors());
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_evec = run.job_title + ".mesh.evec";
        writeEigenvectorsEach(fname_evec,
                              phon->dos->kmesh_dos->nk,
                              phon->dos->kmesh_dos->xk,
                              phon->dos->dymat_dos->get_eigenvalues(),
                              phon->dos->dymat_dos->get_eigenvectors());
    }
}

void Writes::writeEigenvectorsEach(const std::string &fname_evec, const unsigned int nk_in, const double *const *xk_in,
                                   const double *const *eval_in,
                                   const std::complex<double> *const *const *evec_in) const
{
    unsigned int i, j, k;
    const auto neval = phon->dynamical->neval;
    std::ofstream ofs_evec;

    ofs_evec.open(fname_evec.c_str(), std::ios::out);
    if (!ofs_evec) exit("writeEigenvectorsEach", "cannot open file_evec");
    ofs_evec.setf(std::ios::scientific);

    ofs_evec << "# Lattice vectors of the primitive cell\n";

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_evec << std::setw(15) << phon->system->get_primcell().lattice_vector(j, i);
        }
        ofs_evec << '\n';
    }

    ofs_evec << '\n';
    ofs_evec << "# Reciprocal lattice vectors of the primitive cell\n";

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_evec << std::setw(15) << phon->system->get_primcell().reciprocal_lattice_vector(i, j);
        }
        ofs_evec << '\n';
    }

    ofs_evec << '\n';
    ofs_evec << "# Number of phonon modes: " << std::setw(10) << nbands << '\n';
    ofs_evec << "# Number of k points : " << std::setw(10) << nk_in << '\n';
    ofs_evec << "# Number of atomic kinds : " << std::setw(4) << phon->system->get_primcell().number_of_elems << '\n';
    ofs_evec << "# Atomic masses :";
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        ofs_evec << std::setw(15) << phon->system->mass_kd[i];
    }
    ofs_evec << "\n\n";
    ofs_evec << "# Eigenvalues and eigenvectors for each phonon modes below:\n\n";

    NDArray<unsigned int, 2> index_bconnect_tmp;
    index_bconnect_tmp.resize(nk_in, nbands);

    if (phon->dynamical->index_bconnect) {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = phon->dynamical->index_bconnect[i][j];
            }
        }
    } else {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = j;
            }
        }
    }

    for (i = 0; i < nk_in; ++i) {
        ofs_evec << "## kpoint " << std::setw(7) << i + 1 << " : ";
        for (j = 0; j < 3; ++j) {
            ofs_evec << std::setw(15) << xk_in[i][j];
        }
        ofs_evec << '\n';
        for (j = 0; j < nbands; ++j) {

            k = index_bconnect_tmp[i][j];

            auto omega2 = eval_in[i][k];
            if (omega2 >= 0.0) {
                omega2 = omega2 * omega2;
            } else {
                omega2 = -omega2 * omega2;
            }

            ofs_evec << "### mode " << std::setw(8) << j + 1 << " : ";
            ofs_evec << std::setw(15) << omega2 << '\n';

            for (unsigned int m = 0; m < neval; ++m) {
                ofs_evec << std::setw(15) << real(evec_in[i][k][m]);
                ofs_evec << std::setw(15) << imag(evec_in[i][k][m]) << '\n';
            }
            ofs_evec << '\n';
        }
        ofs_evec << '\n';
    }
    ofs_evec.close();

    index_bconnect_tmp.clear();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_evec;
        std::cout << " : Eigenvector of all k points\n";
    }
}

#ifdef _HDF5

void Writes::writeEigenvectorsHdf5() const
{
    std::string fname_evec;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_evec = run.job_title + ".evec.hdf5";
        writeEigenvectorsEachHdf5(fname_evec,
                                  phon->kpoint->kpoint_general->nk,
                                  phon->kpoint->kpoint_general->xk,
                                  phon->dynamical->dymat_general->get_eigenvalues(),
                                  phon->dynamical->dymat_general->get_eigenvectors(),
                                  0);
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_evec = run.job_title + ".band.evec.hdf5";
        writeEigenvectorsEachHdf5(fname_evec,
                                  phon->kpoint->kpoint_bs->nk,
                                  phon->kpoint->kpoint_bs->xk,
                                  phon->dynamical->dymat_band->get_eigenvalues(),
                                  phon->dynamical->dymat_band->get_eigenvectors(),
                                  1);
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_evec = run.job_title + ".mesh.evec.hdf5";
        writeEigenvectorsEachHdf5(fname_evec,
                                  phon->dos->kmesh_dos->nk,
                                  phon->dos->kmesh_dos->xk,
                                  phon->dos->dymat_dos->get_eigenvalues(),
                                  phon->dos->dymat_dos->get_eigenvectors(),
                                  2);
    }
}

void Writes::writeEigenvectorsEachHdf5(const std::string &fname_evec, const unsigned int nk_in,
                                       const double *const *xk_in, const double *const *eval_in,
                                       const std::complex<double> *const *const *evec_in,
                                       const unsigned int kpmode_in) const
{
    using namespace H5;

    unsigned int i, j, k;
    const auto neval = phon->dynamical->neval;
    std::ofstream ofs_evec;

    H5File file(fname_evec, H5F_ACC_TRUNC);
    Group group_cell(file.createGroup("/PrimitiveCell"));
    Group group_band(file.createGroup("/Eigenvalues"));
    Group group_kpoint(file.createGroup("/Kpoints"));

    // Write setting information
    hid_t str_datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_datatype, H5T_VARIABLE);
    std::vector<const char *> arr_c_str;
    for (unsigned int ii = 0; ii < phon->system->get_primcell().number_of_elems; ++ii) {
        arr_c_str.push_back(phon->system->symbol_kd[ii].c_str());
    }
    hsize_t str_dim[1]{arr_c_str.size()};
    DataSpace dataspace(1, str_dim);
    DataSet dataset(group_cell.createDataSet("elements", str_datatype, dataspace));
    dataset.write(&arr_c_str[0], str_datatype);
    dataset.close();
    dataspace.close();

    std::vector<double> mass_tmp;
    for (i = 0; i < phon->system->get_primcell().number_of_elems; ++i) {
        mass_tmp.push_back(phon->system->mass_kd[i]);
    }
    dataspace = DataSpace(1, str_dim);
    dataset = DataSet(group_cell.createDataSet("masses", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(&mass_tmp[0], PredType::NATIVE_DOUBLE);
    dataset.close();
    dataspace.close();


    // Write primitive cell information
    hsize_t dims[2];
    dims[0] = 3;
    dims[1] = 3;
    double lavec_tmp[3][3];
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            lavec_tmp[i][j] = phon->system->get_primcell().lattice_vector(j, i);
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_cell.createDataSet("lattice_vector", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(lavec_tmp, PredType::NATIVE_DOUBLE);
    DataSpace attr_dataspace_str(H5S_SCALAR);
    Attribute myatt_in = dataset.createAttribute("unit", str_datatype, attr_dataspace_str);
    myatt_in.write(str_datatype, std::string("bohr"));
    myatt_in.close();
    dataset.close();
    dataspace.close();

    dims[0] = phon->system->get_primcell().number_of_atoms;
    dims[1] = 3;
    std::vector<double> xfrac_1D(dims[0] * dims[1]);
    hsize_t counter = 0;

    double xtmp[3];
    for (i = 0; i < phon->system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) xtmp[j] = phon->system->get_supercell(0).x_fractional(phon->system->get_map_p2s(0)[i][0], j);
        rotvec(xtmp, xtmp, phon->system->get_supercell(0).lattice_vector);
        rotvec(xtmp, xtmp, phon->system->get_primcell().reciprocal_lattice_vector);
        for (j = 0; j < 3; ++j) xtmp[j] /= 2.0 * pi;
        for (j = 0; j < 3; ++j) {
            while (xtmp[j] >= 1.0) {
                xtmp[j] -= 1.0;
            }
            while (xtmp[j] < 0.0) {
                xtmp[j] += 1.0;
            }
        }
        for (j = 0; j < 3; ++j) {
            xfrac_1D[counter++] = xtmp[j];
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_cell.createDataSet("fractional_coordinate", PredType::NATIVE_DOUBLE, dataspace));
    dataset.write(&xfrac_1D[0], PredType::NATIVE_DOUBLE);
    dataset.close();
    dataspace.close();

    hsize_t dims2[1];
    dims2[0] = phon->system->get_primcell().number_of_atoms;
    dataspace = DataSpace(1, dims2);
    dataset = DataSet(group_cell.createDataSet("atomic_kinds", PredType::NATIVE_INT, dataspace));
    std::vector<int> kdtmp(dims[0]);
    for (i = 0; i < phon->system->get_primcell().number_of_atoms; ++i) {
        kdtmp[i] = phon->system->get_primcell().kind[i];
    }

    dataset.write(&kdtmp[0], PredType::NATIVE_INT);
    dataset.close();

    // write eigenvalues

    NDArray<unsigned int, 2> index_bconnect_tmp;
    int band_index_reordered = 0;
    index_bconnect_tmp.resize(nk_in, nbands);

    if (phon->dynamical->index_bconnect) {
        band_index_reordered = 1;
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = phon->dynamical->index_bconnect[i][j];
            }
        }
    } else {
        for (i = 0; i < nk_in; ++i) {
            for (j = 0; j < nbands; ++j) {
                index_bconnect_tmp[i][j] = j;
            }
        }
    }

    // Write band structure information
    dims[0] = nk_in;
    dims[1] = nbands;

    hsize_t dims_evec[4];
    dims_evec[0] = nk_in;
    dims_evec[1] = nbands;
    dims_evec[2] = neval;
    dims_evec[3] = 2;

    NDArray<double, 2> freq_kayser;
    NDArray<double, 4> evec_tmp;
    freq_kayser.resize(nk_in, nbands);
    evec_tmp.resize(nk_in, nbands, neval, 2);

    for (i = 0; i < nk_in; ++i) {
        for (j = 0; j < nbands; ++j) {
            k = index_bconnect_tmp[i][j];
            freq_kayser[i][j] = in_kayser(eval_in[i][k]);

            for (unsigned int m = 0; m < neval; ++m) {
                evec_tmp[i][j][m][0] = evec_in[i][k][m].real();
                evec_tmp[i][j][m][1] = evec_in[i][k][m].imag();
            }
        }
    }

    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_band.createDataSet("frequencies",
                                               PredType::NATIVE_DOUBLE,
                                               dataspace,
                                               compressed_plist({nk_in, static_cast<size_t>(nbands)})));
    IntType int_type(PredType::NATIVE_INT);
    DataSpace attr_dataspace_int(H5S_SCALAR);
    myatt_in = dataset.createAttribute("band_index_reordered", int_type, attr_dataspace_int);
    myatt_in.write(int_type, &band_index_reordered);
    myatt_in = dataset.createAttribute("unit", str_datatype, attr_dataspace_str);
    myatt_in.write(str_datatype, std::string("kayser (cm^-1)"));

    dataset.write(&freq_kayser[0][0], PredType::NATIVE_DOUBLE);
    myatt_in.close();
    dataset.close();
    dataspace.close();
    freq_kayser.clear();

    dataspace = DataSpace(4, dims_evec);
    dataset = DataSet(group_band.createDataSet("polarization_vectors",
                                               PredType::NATIVE_DOUBLE,
                                               dataspace,
                                               compressed_plist({nk_in, static_cast<size_t>(nbands), neval, 2})));
    dataset.write(&evec_tmp[0][0][0][0], PredType::NATIVE_DOUBLE);
    dataset.close();
    dataspace.close();

    evec_tmp.clear();

    group_cell.close();
    group_band.close();

    dims[0] = nk_in;
    dims[1] = 3;
    std::vector<double> xk_1D(dims[0] * dims[1]);
    counter = 0;

    for (i = 0; i < nk_in; ++i) {
        for (j = 0; j < 3; ++j) {
            xk_1D[counter++] = xk_in[i][j];
        }
    }
    dataspace = DataSpace(2, dims);
    dataset = DataSet(group_kpoint.createDataSet("kpoint_coordinates", PredType::NATIVE_DOUBLE, dataspace));

    myatt_in = dataset.createAttribute("kpoint_mode", int_type, attr_dataspace_int);
    myatt_in.write(int_type, &kpmode_in);
    dataset.write(&xk_1D[0], PredType::NATIVE_DOUBLE);
    myatt_in.close();
    dataset.close();
    dataspace.close();

    if (kpmode_in == 1 && phon->kpoint->kpoint_bs.get()) {
        const auto &kaxis = phon->kpoint->kpoint_bs->kaxis;
        dims2[0] = nk_in;
        dataspace = DataSpace(1, dims2);
        dataset = DataSet(group_kpoint.createDataSet("bandstructure_xaxis", PredType::NATIVE_DOUBLE, dataspace));
        dataset.write(&kaxis[0], PredType::NATIVE_DOUBLE);
        dataset.close();
        dataspace.close();
    }

    group_kpoint.close();
    file.close();

    // Versioned schema stamp (same convention as the kappa/scph state files).
    {
        HighFive::File fh(fname_evec, HighFive::File::ReadWrite);
        stamp_h5_schema(fh, h5_schema_eigenvectors, h5_version_eigen);
        write_input_variables_h5(fh, run.input_variables);
    }
}

#endif

void Writes::writeThermodynamicFunc() const
{
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::ofstream ofs_thermo;
    auto file_thermo = run.job_title + ".thermo";
    ofs_thermo.open(file_thermo.c_str(), std::ios::out);
    if (!ofs_thermo) exit("writeThermodynamicFunc", "cannot open file_thermo");
    if (phon->thermodynamics->calc_FE_bubble) {
        ofs_thermo << "# The bubble free-energy is also shown.\n";
        ofs_thermo
            << "# Temperature [K], Heat capacity / kB, Entropy / kB, Internal energy [Ry], Free energy (QHA) [Ry], Free energy (Bubble) [Ry]\n";
    } else {
        ofs_thermo
            << "# Temperature [K], Heat capacity / kB, Entropy / kB, Internal energy [Ry], Free energy (QHA) [Ry]\n";
    }

    if (phon->thermodynamics->classical) {
        ofs_thermo << "# CLASSICAL = 1: use classical statistics\n";
    }

    for (unsigned int i = 0; i < NT; ++i) {
        const auto T = Tmin + dT * static_cast<double>(i);

        const auto heat_capacity = phon->thermodynamics->Cv_tot(T,
                                                          phon->dos->kmesh_dos->nk_irred,
                                                          phon->dynamical->neval,
                                                          phon->dos->kmesh_dos->kpoint_irred_all,
                                                          &phon->dos->kmesh_dos->weight_k[0],
                                                          phon->dos->dymat_dos->get_eigenvalues());

        const auto Svib = phon->thermodynamics->vibrational_entropy(T,
                                                              phon->dos->kmesh_dos->nk_irred,
                                                              phon->dynamical->neval,
                                                              phon->dos->kmesh_dos->kpoint_irred_all,
                                                              &phon->dos->kmesh_dos->weight_k[0],
                                                              phon->dos->dymat_dos->get_eigenvalues());

        const auto Uvib = phon->thermodynamics->internal_energy(T,
                                                          phon->dos->kmesh_dos->nk_irred,
                                                          phon->dynamical->neval,
                                                          phon->dos->kmesh_dos->kpoint_irred_all,
                                                          &phon->dos->kmesh_dos->weight_k[0],
                                                          phon->dos->dymat_dos->get_eigenvalues());

        const auto FE_QHA = phon->thermodynamics->free_energy_QHA(T,
                                                            phon->dos->kmesh_dos->nk_irred,
                                                            phon->dynamical->neval,
                                                            phon->dos->kmesh_dos->kpoint_irred_all,
                                                            &phon->dos->kmesh_dos->weight_k[0],
                                                            phon->dos->dymat_dos->get_eigenvalues());

        ofs_thermo << std::setw(16) << std::fixed << T;
        ofs_thermo << std::setw(18) << std::scientific << heat_capacity / k_Boltzmann;
        ofs_thermo << std::setw(18) << Svib / k_Boltzmann;
        ofs_thermo << std::setw(18) << Uvib;
        ofs_thermo << std::setw(18) << FE_QHA;

        if (phon->thermodynamics->calc_FE_bubble) {
            ofs_thermo << std::setw(18) << phon->thermodynamics->FE_bubble[i];
        }
        ofs_thermo << '\n';
    }

    ofs_thermo.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_thermo;
        std::cout << " : Thermodynamic quantities\n";
    }
}

void Writes::writeGruneisen()
{
    const auto ncomp = phon->gruneisen->number_of_strain_components();
    const std::string components_header =
        ncomp == 3 ? "gamma_xx, gamma_yy, gamma_zz" : "gamma_xx, gamma_yy, gamma_zz, gamma_yz, gamma_xz, gamma_xy";

    if (phon->kpoint->kpoint_bs.get() && (phon->gruneisen->gruneisen_bs || phon->gruneisen->gruneisen_tensor_bs)) {
        if (nbands < 0 || nbands > 3 * phon->system->get_primcell().number_of_atoms) {
            nbands = 3 * phon->system->get_primcell().number_of_atoms;
        }

        std::ofstream ofs_gruneisen;

        auto file_gru = run.job_title + ".gruneisen";
        ofs_gruneisen.open(file_gru.c_str(), std::ios::out);
        if (!ofs_gruneisen) exit("writeGruneisen", "cannot open file_vel");

        const auto nk = phon->kpoint->kpoint_bs->nk;
        const auto &kaxis = phon->kpoint->kpoint_bs->kaxis;

        if (phon->gruneisen->gruneisen_mode == 1) {
            ofs_gruneisen << "# Volumetric Gruneisen parameter: gamma = -dln(omega)/dln(V)\n";
            ofs_gruneisen << "# k-axis, gamma\n";
            ofs_gruneisen.setf(std::ios::fixed);

            if (phon->dynamical->band_connection == 0) {
                for (unsigned int i = 0; i < nk; ++i) {
                    ofs_gruneisen << std::setw(8) << kaxis[i];
                    for (unsigned int j = 0; j < nbands; ++j) {
                        ofs_gruneisen << std::setw(15) << phon->gruneisen->gruneisen_bs[i][j].real();
                    }
                    ofs_gruneisen << '\n';
                }
            } else {
                for (unsigned int i = 0; i < nk; ++i) {
                    ofs_gruneisen << std::setw(8) << kaxis[i];
                    for (unsigned int j = 0; j < nbands; ++j) {
                        ofs_gruneisen << std::setw(15)
                                      << phon->gruneisen->gruneisen_bs[i][phon->dynamical->index_bconnect[i][j]].real();
                    }
                    ofs_gruneisen << '\n';
                }
            }
        } else {
            const auto eval = phon->dynamical->dymat_band->get_eigenvalues();

            ofs_gruneisen << "# Generalized Gruneisen parameters: gamma_munu = -dln(omega)/d(eps_munu)\n";
            ofs_gruneisen << "# k-axis, band, omega [cm^-1], " << components_header << '\n';
            ofs_gruneisen.setf(std::ios::fixed);

            for (unsigned int i = 0; i < nk; ++i) {
                for (unsigned int j = 0; j < nbands; ++j) {
                    const auto js = phon->dynamical->band_connection == 0 ? j : phon->dynamical->index_bconnect[i][j];
                    ofs_gruneisen << std::setw(8) << kaxis[i];
                    ofs_gruneisen << std::setw(5) << j;
                    ofs_gruneisen << std::setw(15) << in_kayser(eval[i][js]);
                    for (auto ic = 0; ic < ncomp; ++ic) {
                        ofs_gruneisen << std::setw(15) << phon->gruneisen->gruneisen_tensor_bs[i][js][ic].real();
                    }
                    ofs_gruneisen << '\n';
                }
            }
        }

        ofs_gruneisen.close();

        if (run.verbosity > 0) {
            std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_gru;
            if (phon->gruneisen->gruneisen_mode == 1) {
                std::cout << " : Volumetric Gruneisen parameters along given k-path\n";
            } else {
                std::cout << " : Generalized Gruneisen parameters along given k-path\n";
            }
        }
    }

    if (phon->dos->kmesh_dos.get() && (phon->gruneisen->gruneisen_dos || phon->gruneisen->gruneisen_tensor_dos)) {

        std::ofstream ofs_gruall;
        auto file_gruall = run.job_title + ".gru_all";
        ofs_gruall.open(file_gruall.c_str(), std::ios::out);
        if (!ofs_gruall) exit("writeGruneisen", "cannot open file_gruall");

        const auto nk = phon->dos->kmesh_dos->nk;
        const auto ns = phon->dynamical->neval;
        const auto &xk = phon->dos->kmesh_dos->xk;
        const auto eval = phon->dos->dymat_dos->get_eigenvalues();

        if (phon->gruneisen->gruneisen_mode == 1) {
            ofs_gruall << "# Volumetric Gruneisen parameter: gamma = -dln(omega)/dln(V)\n";
            ofs_gruall << "# knum, snum, omega [cm^-1], gruneisen parameter\n";
        } else {
            ofs_gruall << "# Generalized Gruneisen parameters: gamma_munu = -dln(omega)/d(eps_munu)\n";
            ofs_gruall << "# knum, snum, omega [cm^-1], " << components_header << '\n';
        }

        for (unsigned int i = 0; i < nk; ++i) {
            ofs_gruall << "# knum = " << i;
            for (unsigned int k = 0; k < 3; ++k) {
                ofs_gruall << std::setw(15) << xk[i][k];
            }
            ofs_gruall << '\n';

            for (unsigned int j = 0; j < ns; ++j) {
                ofs_gruall << std::setw(5) << i;
                ofs_gruall << std::setw(5) << j;
                ofs_gruall << std::setw(15) << in_kayser(eval[i][j]);
                if (phon->gruneisen->gruneisen_mode == 1) {
                    ofs_gruall << std::setw(15) << phon->gruneisen->gruneisen_dos[i][j].real();
                } else {
                    for (auto ic = 0; ic < ncomp; ++ic) {
                        ofs_gruall << std::setw(15) << phon->gruneisen->gruneisen_tensor_dos[i][j][ic].real();
                    }
                }
                ofs_gruall << '\n';
            }
        }
        ofs_gruall.close();

        if (run.verbosity > 0) {
            std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_gruall;
            if (phon->gruneisen->gruneisen_mode == 1) {
                std::cout << " : Volumetric Gruneisen parameters at all k points" << '\n';
            } else {
                std::cout << " : Generalized Gruneisen parameters at all k points" << '\n';
            }
        }
    }
}

void Writes::writeNewFcsXml(const std::string &filename_xml, const std::vector<FcsArrayWithCell> &delta_fc2,
                            const std::vector<FcsArrayWithCell> &delta_fc3, const Eigen::Matrix3d &strain_dir,
                            const double fc_scale, const Eigen::MatrixXd &sublattice_disp) const
{
    int i, j;

    const Eigen::Matrix3d u_applied = fc_scale * strain_dir;
    const Eigen::Matrix3d lattice_vector =
        (Eigen::Matrix3d::Identity() + u_applied) * phon->system->get_supercell(0).lattice_vector;

    using boost::property_tree::ptree;

    ptree pt;

    pt.put("Data.ANPHON_version", ALAMODE_VERSION);
    pt.put("Data.Description.OriginalFCS", phon->fcs_phonon->file_fcs);
    for (i = 0; i < 3; ++i) {
        std::string str_strain;
        for (j = 0; j < 3; ++j) {
            str_strain += " " + fcsxml::double2string(u_applied(i, j));
        }
        pt.add("Data.Description.Strain.u" + std::to_string(i + 1), str_strain);
    }

    const auto &cell_tmp = phon->system->get_supercell(0);
    const auto &map_tmp = phon->system->get_map_p2s(0);

    std::vector<std::string> element_names(phon->system->symbol_kd.begin(),
                                           phon->system->symbol_kd.begin() + phon->system->get_primcell().number_of_elems);
    const std::vector<int> atomic_kinds(cell_tmp.kind.begin(), cell_tmp.kind.end());

    // Atomic positions: affine deformation keeps the fractional coordinates;
    // the relaxed-ion path adds the strain-induced sublattice displacement.
    Eigen::MatrixXd x_fractional = cell_tmp.x_fractional;
    if (sublattice_disp.size() != 0) {
        const Eigen::Matrix3d lattice_inv = lattice_vector.inverse();
        const auto &map_s2p = phon->system->get_map_s2p(0);
        for (i = 0; i < cell_tmp.number_of_atoms; ++i) {
            const auto kappa = map_s2p[i].atom_num;
            x_fractional.row(i) += (lattice_inv * (fc_scale * sublattice_disp.row(kappa).transpose())).transpose();
        }
    }

    fcsxml::add_structure_group_xml(pt, lattice_vector, x_fractional, atomic_kinds, element_names);
    fcsxml::add_symmetry_group_xml(pt, map_tmp);

    pt.put("Data.ForceConstants", "");

    // Store base IFCs plus fc_scale times strain corrections; the loader sums
    // identical entries and regenerates trailing-leg permutations. Store only
    // entries sorted by the trailing-leg key 3*atom_super + coord.
    auto legs_ascending = [&](const FcsArrayWithCell &it, const int norder) {
        for (auto k = 1; k < norder - 1; ++k) {
            if (3 * it.atoms_s[k] + it.pairs[k].index % 3 > 3 * it.atoms_s[k + 1] + it.pairs[k + 1].index % 3) {
                return false;
            }
        }
        return true;
    };

    auto build_rows = [&](const std::vector<FcsArrayWithCell> &fcs_base,
                          const std::vector<FcsArrayWithCell> &fcs_delta,
                          const int norder) {
        std::vector<fcsxml::FcCartesianRowXml> rows;
        auto append = [&](const FcsArrayWithCell &it, const double value) {
            fcsxml::FcCartesianRowXml row;
            row.value = value;
            row.atom1_prim = it.pairs[0].index / 3;
            row.coords.push_back(it.pairs[0].index % 3);
            for (auto k = 1; k < norder; ++k) {
                row.atoms_super.push_back(static_cast<int>(map_tmp[it.pairs[k].index / 3][it.pairs[k].tran]));
                row.coords.push_back(it.pairs[k].index % 3);
                row.cells.push_back(static_cast<int>(it.pairs[k].cell_s));
            }
            rows.emplace_back(std::move(row));
        };

        for (const auto &it: fcs_base) {
            if (!legs_ascending(it, norder)) continue;
            append(it, it.fcs_val);
        }
        for (const auto &it: fcs_delta) {
            if (std::abs(it.fcs_val) < eps12) continue;
            if (!legs_ascending(it, norder)) continue;
            append(it, fc_scale * it.fcs_val);
        }
        return rows;
    };

    fcsxml::add_fc_cartesian_group_xml(pt,
                                       "HARMONIC",
                                       2,
                                       build_rows(phon->fcs_phonon->force_constant_with_cell[0], delta_fc2, 2));

    if (phon->anharmonic_core->quartic_mode) {
        fcsxml::add_fc_cartesian_group_xml(pt,
                                           "ANHARM3",
                                           3,
                                           build_rows(phon->fcs_phonon->force_constant_with_cell[1], delta_fc3, 3));
    }

    fcsxml::write_fcs_xml_file(filename_xml, pt);
}

#ifdef _HDF5

void Writes::writeNewFcsH5(const std::string &filename_h5, const std::vector<FcsArrayWithCell> &delta_fc2,
                           const std::vector<FcsArrayWithCell> &delta_fc3, const Eigen::Matrix3d &strain_dir,
                           const double fc_scale, const Eigen::MatrixXd &sublattice_disp) const
{
    using namespace H5Easy;

    const Eigen::Matrix3d u_applied = fc_scale * strain_dir;
    const Eigen::Matrix3d deform = Eigen::Matrix3d::Identity() + u_applied;

    File file(filename_h5, File::ReadWrite | File::Create | File::Truncate);

    const auto &supercell = phon->system->get_supercell(0);
    const auto &primcell = phon->system->get_primcell();
    const auto &map_p2s = phon->system->get_map_p2s(0);

    const std::vector<std::string> element_names(phon->system->symbol_kd.begin(),
                                                 phon->system->symbol_kd.begin() + primcell.number_of_elems);
    const std::vector<std::vector<double>> no_magmom;

    // Atomic positions: affine deformation keeps the fractional coordinates;
    // the relaxed-ion path adds the strain-induced sublattice displacement.
    const bool with_sublattice = sublattice_disp.size() != 0;

    Eigen::MatrixXd xf_super = supercell.x_fractional;
    Eigen::MatrixXd xf_prim = primcell.x_fractional;
    if (with_sublattice) {
        const Eigen::Matrix3d lavec_super_inv = (deform * supercell.lattice_vector).inverse();
        const Eigen::Matrix3d lavec_prim_inv = (deform * primcell.lattice_vector).inverse();
        const auto &map_s2p = phon->system->get_map_s2p(0);
        for (auto i = 0; i < supercell.number_of_atoms; ++i) {
            const auto kappa = map_s2p[i].atom_num;
            xf_super.row(i) += (lavec_super_inv * (fc_scale * sublattice_disp.row(kappa).transpose())).transpose();
        }
        for (std::size_t kappa = 0; kappa < primcell.number_of_atoms; ++kappa) {
            xf_prim.row(kappa) += (lavec_prim_inv * (fc_scale * sublattice_disp.row(kappa).transpose())).transpose();
        }
    }

    {
        std::vector<std::vector<int>> mapping(map_p2s.size());
        for (std::size_t i = 0; i < map_p2s.size(); ++i) {
            mapping[i].assign(map_p2s[i].begin(), map_p2s[i].end());
        }
        write_cell_group_h5(file,
                            "SuperCell",
                            Eigen::Matrix3d(deform * supercell.lattice_vector),
                            xf_super,
                            supercell.kind,
                            element_names,
                            0,
                            no_magmom,
                            0,
                            1,
                            map_p2s[0].size(),
                            mapping,
                            units::FcUnitSystem::ry_bohr);
    }
    {
        std::vector<std::vector<int>> mapping(primcell.number_of_atoms, std::vector<int>(1));
        for (std::size_t i = 0; i < primcell.number_of_atoms; ++i) {
            mapping[i][0] = static_cast<int>(i);
        }
        write_cell_group_h5(file,
                            "PrimitiveCell",
                            Eigen::Matrix3d(deform * primcell.lattice_vector),
                            xf_prim,
                            primcell.kind,
                            element_names,
                            0,
                            no_magmom,
                            0,
                            1,
                            1,
                            mapping,
                            units::FcUnitSystem::ry_bohr);
    }

    // Shift vectors of the deformed geometry in Cartesian bohr:
    // relvecs_velocity is stored in the primitive lattice basis.
    const Eigen::Matrix3d lavec_prim_deformed = deform * primcell.lattice_vector;

    auto dump_order = [&](const int order,
                          const std::vector<FcsArrayWithCell> &fcs_base,
                          const std::vector<FcsArrayWithCell> &fcs_delta) {
        const auto norder = order + 2;

        // The h5 loader stores one canonical row per permutation multiset of the
        // trailing legs and regenerates the permutations on read (compared by
        // 3*atom_super + coord), so keep only rows whose trailing legs are in
        // ascending order of that key.
        auto legs_ascending = [&](const FcsArrayWithCell &it) {
            for (auto k = 1; k < norder - 1; ++k) {
                if (3 * it.atoms_s[k] + it.pairs[k].index % 3 > 3 * it.atoms_s[k + 1] + it.pairs[k + 1].index % 3) {
                    return false;
                }
            }
            return true;
        };

        std::vector<std::pair<const FcsArrayWithCell *, double>> selected;
        for (const auto &it: fcs_base) {
            if (!legs_ascending(it)) continue;
            selected.emplace_back(&it, it.fcs_val);
        }
        for (const auto &it: fcs_delta) {
            if (std::abs(it.fcs_val) < eps12) continue;
            if (!legs_ascending(it)) continue;
            selected.emplace_back(&it, fc_scale * it.fcs_val);
        }

        const auto nrows = static_cast<Eigen::Index>(selected.size());
        Eigen::MatrixXi atom_indices(nrows, norder), atom_indices_super(nrows, norder), coord_indices(nrows, norder);
        Eigen::MatrixXd shift_vectors(nrows, 3 * (norder - 1));
        Eigen::ArrayXd fcs_values(nrows);

        for (Eigen::Index i = 0; i < nrows; ++i) {
            const auto &it = *selected[i].first;
            for (auto k = 0; k < norder; ++k) {
                atom_indices(i, k) = static_cast<int>(it.pairs[k].index / 3);
                atom_indices_super(i, k) = static_cast<int>(it.atoms_s[k]);
                coord_indices(i, k) = static_cast<int>(it.pairs[k].index % 3);
            }
            for (auto k = 0; k < norder - 1; ++k) {
                Eigen::Vector3d shift_cart = lavec_prim_deformed * it.relvecs_velocity[k];
                if (with_sublattice) {
                    const auto kappa_leg = it.pairs[k + 1].index / 3;
                    const auto kappa_first = it.pairs[0].index / 3;
                    shift_cart +=
                        fc_scale * (sublattice_disp.row(kappa_leg) - sublattice_disp.row(kappa_first)).transpose();
                }
                for (auto j = 0; j < 3; ++j) {
                    shift_vectors(i, 3 * k + j) = shift_cart[j];
                }
            }
            fcs_values[i] = selected[i].second;
        }

        write_fc_order_group_h5(file,
                                order,
                                atom_indices,
                                atom_indices_super,
                                coord_indices,
                                std::move(shift_vectors),
                                std::move(fcs_values),
                                units::FcUnitSystem::ry_bohr,
                                9);
    };

    dump_order(0, phon->fcs_phonon->force_constant_with_cell[0], delta_fc2);
    if (phon->anharmonic_core->quartic_mode) {
        dump_order(1, phon->fcs_phonon->force_constant_with_cell[1], delta_fc3);
    }

    stamp_h5_schema(file, h5_schema_force_constants, h5_version_force_constants);
    write_input_variables_h5(file, run.input_variables);
    dump(file, "/version", ALAMODE_VERSION);
    dump(file, "/original_fcsfile", phon->fcs_phonon->file_fcs);
    dump(file, "/applied_strain", Eigen::Matrix3d(u_applied));

    const std::time_t result = std::time(nullptr);
    std::string time_str;
    time_str.resize(100);
    std::strftime(&time_str[0], time_str.size(), "%Y-%b-%d %T", std::localtime(&result));
    dump(file, "/created date", time_str);
}

#endif

void Writes::writeMSD() const
{
    // Write room mean square displacement of atoms

    auto file_rmsd = run.job_title + ".msd";
    std::ofstream ofs_rmsd;

    const auto ns = phon->dynamical->neval;

    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto nk = phon->dos->kmesh_dos->nk;
    const auto &xk = phon->dos->kmesh_dos->xk;
    const auto eval = phon->dos->dymat_dos->get_eigenvalues();
    const auto evec = phon->dos->dymat_dos->get_eigenvectors();

    ofs_rmsd.open(file_rmsd.c_str(), std::ios::out);
    if (!ofs_rmsd) exit("writeMSD", "Could not open file_rmsd");

    ofs_rmsd << "# Mean Square Displacements at a function of temperature.\n";
    ofs_rmsd << "# Temperature [K], <(u_{1}^{x})^{2}>, <(u_{1}^{y})^{2}>, <(u_{1}^{z})^{2}>, .... [Angstrom^2]\n";

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    for (unsigned int i = 0; i < NT; ++i) {

        const auto T = Tmin + static_cast<double>(i) * dT;
        ofs_rmsd << std::setw(15) << T;

        for (unsigned int j = 0; j < ns; ++j) {
            const auto d2_tmp = phon->thermodynamics->disp2_avg(T, j, j, nk, ns, xk, eval, evec, *phon->system);
            ofs_rmsd << std::setw(15) << d2_tmp * pow2(Bohr_in_Angstrom);
        }
        ofs_rmsd << '\n';
    }
    ofs_rmsd.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_rmsd;
        std::cout << " : Mean-square-displacement (MSD)\n";
    }
}

void Writes::writeMSD(double **msd_in, const bool is_qha, const int bubble) const
{
    const auto ns = phon->dynamical->neval;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::ofstream ofs_msd;
    std::string file_msd;
    if (is_qha) {
        file_msd = run.job_title + ".qha_msd";
    } else {
        if (bubble == 0) {
            file_msd = run.job_title + ".scph_msd";
        } else if (bubble == 1) {
            file_msd = run.job_title + ".scph+bubble(0)_msd";
        } else if (bubble == 2) {
            file_msd = run.job_title + ".scph+bubble(w)_msd";
        } else if (bubble == 3) {
            file_msd = run.job_title + ".scph+bubble(wQP)_msd";
        }
    }

    ofs_msd.open(file_msd.c_str(), std::ios::out);
    if (!ofs_msd) exit("writeMSD", "cannot open file_thermo");
    ofs_msd << "# Mean Square Displacements at a function of temperature.\n";
    ofs_msd << "# Temperature [K], <(u_{1}^{x})^{2}>, <(u_{1}^{y})^{2}>, <(u_{1}^{z})^{2}>, .... [Angstrom^2]\n";

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + static_cast<double>(iT) * dT;

        ofs_msd << std::setw(15) << temp;
        for (unsigned int i = 0; i < ns; ++i) {
            ofs_msd << std::setw(15) << msd_in[iT][i] * pow2(Bohr_in_Angstrom);
        }
        ofs_msd << '\n';
    }

    ofs_msd.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_msd;
        if (is_qha) {
            std::cout << " : Mean-square-displacement (QHA level)\n";
        } else {
            if (bubble == 0) {
                std::cout << " : Mean-square-displacement (SCPH level)\n";
            } else if (bubble == 1) {
                std::cout << " : Mean-square-displacement (SCPH+Bubble(0) level)\n";
            } else if (bubble == 2) {
                std::cout << " : Mean-square-displacement (SCPH+Bubble(w) level)\n";
            } else if (bubble == 3) {
                std::cout << " : Mean-square-displacement (SCPH+Bubble(wQP) level)\n";
            }
        }
    }
}

void Writes::writeDispCorrelation() const
{
    if (!phon->dos->kmesh_dos.get()) return;

    auto file_ucorr = run.job_title + ".ucorr";
    std::ofstream ofs;

    const auto ns = phon->dynamical->neval;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    ofs.open(file_ucorr.c_str(), std::ios::out);
    if (!ofs) exit("writeDispCorrelation", "Could not open file_rmsd");

    ofs << "# Displacement-displacement correlation function at various temperatures.\n";
    if (phon->thermodynamics->classical) ofs << "# CLASSICAL = 1: classical statistics is used.\n";

    double shift[3];

    for (auto i = 0; i < 3; ++i) {
        shift[i] = static_cast<double>(shift_ucorr[i]);
    }

    ofs << "# Temperature [K], (atom1,crd1), (atom2,crd2), SHIFT_UCORR, <u_{0,atom1}^{crd1} * u_{L, atom2}^{crd2}> [Angstrom^2]\n";

    for (unsigned int i = 0; i < NT; ++i) {

        const auto T = Tmin + static_cast<double>(i) * dT;

        for (unsigned int j = 0; j < ns; ++j) {
            for (unsigned int k = 0; k < ns; ++k) {

                const auto ucorr = phon->thermodynamics->disp_corrfunc(T,
                                                                 j,
                                                                 k,
                                                                 shift,
                                                                 phon->dos->kmesh_dos->nk,
                                                                 ns,
                                                                 phon->dos->kmesh_dos->xk,
                                                                 phon->dos->dymat_dos->get_eigenvalues(),
                                                                 phon->dos->dymat_dos->get_eigenvectors(),
                                                                 *phon->system);

                ofs << std::setw(17) << T;
                ofs << std::setw(11) << j / 3 + 1;
                ofs << std::setw(3) << j % 3 + 1;
                ofs << std::setw(11) << k / 3 + 1;
                ofs << std::setw(3) << k % 3 + 1;
                ofs << std::setw(4) << shift_ucorr[0];
                ofs << std::setw(4) << shift_ucorr[1];
                ofs << std::setw(4) << shift_ucorr[2];
                ofs << std::setw(15) << ucorr * pow2(Bohr_in_Angstrom);
                ofs << '\n';
            }
        }
        ofs << '\n';
    }
    ofs << std::flush;
    ofs.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_ucorr;
        std::cout << " : displacement correlation functions\n";
    }
}

void Writes::writeDispCorrelation(double ***ucorr_in, const bool is_qha, const int bubble) const
{
    std::string file_ucorr;
    std::ofstream ofs;

    const auto ns = phon->dynamical->neval;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    if (is_qha) {
        file_ucorr = run.job_title + ".qha_ucorr";
    } else {
        if (bubble == 0) {
            file_ucorr = run.job_title + ".scph_ucorr";
        } else if (bubble == 1) {
            file_ucorr = run.job_title + ".scph+bubble(0)_ucorr";
        } else if (bubble == 2) {
            file_ucorr = run.job_title + ".scph+bubble(w)_ucorr";
        } else if (bubble == 3) {
            file_ucorr = run.job_title + ".scph+bubble(wQP)_ucorr";
        }
    }


    ofs.open(file_ucorr.c_str(), std::ios::out);
    if (!ofs) exit("writeDispCorrelation", "Could not open file_rmsd");

    ofs << "# Displacement-displacement correlation function at various temperatures.\n";
    ofs << "# Self-consistent phonon frequencies and eigenvectors are used.\n";
    if (phon->thermodynamics->classical) ofs << "# CLASSICAL = 1: classical statistics is used.\n";

    double shift[3];

    for (auto i = 0; i < 3; ++i) {
        shift[i] = static_cast<double>(shift_ucorr[i]);
    }

    ofs << "# Temperature [K], (atom1,crd1), (atom2,crd2), SHIFT_UCORR, <u_{0,atom1}^{crd1} * u_{L, atom2}^{crd2}> [Angstrom^2]\n";

    for (unsigned int i = 0; i < NT; ++i) {

        const auto T = Tmin + static_cast<double>(i) * dT;

        for (unsigned int j = 0; j < ns; ++j) {
            for (unsigned int k = 0; k < ns; ++k) {

                ofs << std::setw(17) << T;
                ofs << std::setw(11) << j / 3 + 1;
                ofs << std::setw(3) << j % 3 + 1;
                ofs << std::setw(11) << k / 3 + 1;
                ofs << std::setw(3) << k % 3 + 1;
                ofs << std::setw(4) << shift_ucorr[0];
                ofs << std::setw(4) << shift_ucorr[1];
                ofs << std::setw(4) << shift_ucorr[2];
                ofs << std::setw(15) << ucorr_in[i][j][k] * pow2(Bohr_in_Angstrom);
                ofs << '\n';
            }
        }
        ofs << '\n';
    }
    ofs << std::flush;
    ofs.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_ucorr;

        if (is_qha) {
            std::cout << " : displacement correlation functions (QHA level)\n";
        } else {
            if (bubble == 0) {
                std::cout << " : displacement correlation functions (SCPH level)\n";
            } else if (bubble == 1) {
                std::cout << " : displacement correlation functions (SCPH+Bubble(0) level)\n";
            } else if (bubble == 2) {
                std::cout << " : displacement correlation functions (SCPH+Bubble(w) level)\n";
            } else if (bubble == 3) {
                std::cout << " : displacement correlation functions (SCPH+Bubble(wQP) level)\n";
            }
        }
    }
}

void PHON_NS::print_output_file(const RunInfo &run, const std::string &file, const std::string &description)
{
    if (run.my_rank != 0 || run.verbosity == 0) return;
    std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file << " : " << description << '\n';
}

void Writes::writeKappaIterative(const unsigned int ntemp_in, const double *temperature_in,
                                 const double *const *const *kappa_in,
                                 const std::vector<unsigned char> &converged_in) const
{
    if (run.my_rank != 0) return;

    const auto file_kappa = run.job_title + ".kl_iter";

    std::ofstream ofs_kl;

    ofs_kl.open(file_kappa.c_str(), std::ios::out);
    if (!ofs_kl) exit("writeKappaIterative", "Could not open file_kappa");

    ofs_kl << "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]" << '\n';
    ofs_kl << "# Iterative result." << '\n';

    std::vector<double> t_unconverged;
    for (unsigned int i = 0; i < ntemp_in && i < converged_in.size(); ++i) {
        if (!converged_in[i]) t_unconverged.push_back(temperature_in[i]);
    }
    if (!t_unconverged.empty()) {
        ofs_kl << "# WARNING: the iteration did NOT converge at the following temperatures:" << '\n';
        ofs_kl << "#";
        for (const auto t: t_unconverged) {
            ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2) << t;
        }
        ofs_kl << " [K]" << '\n';
    }

    if (phon->isotope->include_isotope) ofs_kl << "# Isotope effects are included." << '\n';
    if (phon->conductivity->fph_rta > 0) ofs_kl << "# 4ph is included non-iteratively." << '\n';
    if (phon->conductivity->len_boundary > eps) {
        ofs_kl << "# Size of boundary " << std::scientific << std::setprecision(2) << phon->conductivity->len_boundary * 1e9
               << " [nm]" << '\n';
    }

    for (unsigned int itemp = 0; itemp < ntemp_in; ++itemp) {
        ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2) << temperature_in[itemp];
        for (auto ix = 0; ix < 3; ++ix) {
            for (auto iy = 0; iy < 3; ++iy) {
                ofs_kl << std::setw(15) << std::scientific << std::setprecision(4) << kappa_in[itemp][ix][iy];
            }
        }
        ofs_kl << '\n';
    }
    ofs_kl.close();
    if (run.verbosity > 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n\n";
        std::cout << " The following files are created: \n";
    }
    print_output_file(run, file_kappa, "Lattice thermal conductivity (iterative BTE)");
    if (phon->conductivity->get_use_h5_io()) {
        print_output_file(run, run.job_title + ".kappa.h5", "Self-energies and thermal conductivity (restart file)");
    }
}

void Writes::writeKappa() const
{
    // Write lattice thermal conductivity

    if (run.my_rank == 0) {
        int i, j, k;

        std::string file_kappa;
        std::string file_kappa_3only;

        if (phon->conductivity->fph_rta > 0) {
            file_kappa_3only = run.job_title + ".kl3";
            file_kappa = run.job_title + ".kl4";
        } else {
            file_kappa = run.job_title + ".kl";
        }

        auto file_kappa2 = run.job_title + ".kl_spec";
        auto file_kappa_coherent = run.job_title + ".kl_coherent";

        std::ofstream ofs_kl;

        if (phon->conductivity->fph_rta > 0) {
            ofs_kl.open(file_kappa_3only.c_str(), std::ios::out);
            if (!ofs_kl) exit("write_kappa", "Could not open file_kappa");

            ofs_kl << "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]"
                   << std::endl;
            ofs_kl << "# three phonon part\n";

            if (phon->isotope->include_isotope) {
                ofs_kl << "# Isotope effects are included." << std::endl;
            }

            if (phon->conductivity->len_boundary > eps) {
                ofs_kl << "# Size of boundary " << std::scientific << std::setprecision(2)
                       << phon->conductivity->len_boundary * 1e9 << " [nm]" << std::endl;
            }

            for (i = 0; i < phon->conductivity->ntemp; ++i) {
                ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                       << phon->conductivity->temperature[i];
                for (j = 0; j < 3; ++j) {
                    for (k = 0; k < 3; ++k) {
                        ofs_kl << std::setw(15) << std::fixed << std::setprecision(4)
                               << phon->conductivity->kappa_3only[i][j][k];
                    }
                }
                ofs_kl << std::endl;
            }
            ofs_kl.close();
        }

        ofs_kl.open(file_kappa.c_str(), std::ios::out);
        if (!ofs_kl) exit("writeKappa", "Could not open file_kappa");

        ofs_kl << "# Temperature [K], Thermal Conductivity (xx, xy, xz, yx, yy, yz, zx, zy, zz) [W/mK]\n";

        if (phon->isotope->include_isotope) {
            ofs_kl << "# Isotope effects are included.\n";
        }

        if (phon->conductivity->len_boundary > eps) {
            ofs_kl << "# Size of boundary " << std::scientific << std::setprecision(2)
                   << phon->conductivity->len_boundary * 1e9 << " [nm]" << std::endl;
        }

        for (i = 0; i < phon->conductivity->ntemp; ++i) {
            ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2) << phon->conductivity->temperature[i];
            for (j = 0; j < 3; ++j) {
                for (k = 0; k < 3; ++k) {
                    ofs_kl << std::setw(15) << std::fixed << std::setprecision(4) << phon->conductivity->kappa[i][j][k];
                }
            }
            ofs_kl << '\n';
        }
        ofs_kl.close();

        if (phon->conductivity->calc_kappa_spec) {

            ofs_kl.open(file_kappa2.c_str(), std::ios::out);
            if (!ofs_kl) exit("writeKappa", "Could not open file_kappa2");

            ofs_kl << "# Temperature [K], Frequency [cm^-1], Thermal Conductivity Spectra (xx, yy, zz) [W/mK * cm]\n";

            if (phon->isotope->include_isotope) {
                ofs_kl << "# Isotope effects are included.\n";
            }

            for (i = 0; i < phon->conductivity->ntemp; ++i) {
                for (j = 0; j < phon->dos->n_energy; ++j) {
                    ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                           << phon->conductivity->temperature[i];
                    ofs_kl << std::setw(10) << phon->dos->energy_dos[j];
                    for (k = 0; k < 3; ++k) {
                        ofs_kl << std::setw(15) << std::fixed << std::setprecision(6)
                               << phon->conductivity->kappa_spec[j][i][k];
                    }
                    ofs_kl << '\n';
                }
                ofs_kl << '\n';
            }
            ofs_kl.close();
        }

        if (phon->conductivity->calc_coherent) {
            ofs_kl.open(file_kappa_coherent.c_str(), std::ios::out);
            if (!ofs_kl) exit("writeKappa", "Could not open file_kappa_coherent");

            ofs_kl << "# Temperature [K], Coherent part of the lattice thermal Conductivity "
                      "(xx, yy, zz, xy, xz, yx, yz, zx, zy) [W/mK]\n";

            if (phon->isotope->include_isotope) {
                ofs_kl << "# Isotope effects are included.\n";
            }

            for (i = 0; i < phon->conductivity->ntemp; ++i) {
                ofs_kl << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                       << phon->conductivity->temperature[i];
                // Diagonal elements keep columns 2-4 of the original format; off-diagonal ones are appended.
                for (j = 0; j < 3; ++j) {
                    ofs_kl << std::setw(15) << std::fixed << std::setprecision(4)
                           << phon->conductivity->kappa_coherent[i][j][j];
                }
                for (j = 0; j < 3; ++j) {
                    for (k = 0; k < 3; ++k) {
                        if (j == k) continue;
                        ofs_kl << std::setw(15) << std::fixed << std::setprecision(4)
                               << phon->conductivity->kappa_coherent[i][j][k];
                    }
                }
                ofs_kl << '\n';
            }
            ofs_kl.close();
        }


        if (run.verbosity > 0) {
            std::cout << '\n';
            std::cout << " -----------------------------------------------------------------\n\n";
            std::cout << " The following files are created: \n";
        }
        if (phon->conductivity->fph_rta > 0) {
            print_output_file(run, file_kappa_3only, "Lattice thermal conductivity (3-phonon only)");
            print_output_file(run, file_kappa, "Lattice thermal conductivity (3-phonon + 4-phonon)");
            if (phon->conductivity->write_interpolation > 0) {
                print_output_file(run, run.job_title + ".interpolated_gamma",
                                "Four-phonon linewidths interpolated onto the 3-phonon mesh");
            }
        } else {
            print_output_file(run, file_kappa, "Lattice thermal conductivity");
        }
        if (phon->conductivity->calc_kappa_spec) print_output_file(run, file_kappa2, "Spectral thermal conductivity");
        if (phon->conductivity->calc_coherent) print_output_file(run, file_kappa_coherent, "Coherent (interband) part of kappa");
        if (phon->conductivity->get_use_h5_io()) {
            print_output_file(run, run.job_title + ".kappa.h5", "Self-energies and thermal conductivity (restart file)");
        }
    }
}

void Writes::writeSelfenergyIsotope() const
{
    unsigned int k;
    const auto ns = phon->dynamical->neval;
    const auto eval = phon->dos->dymat_dos->get_eigenvalues();
    const auto &gamma_iso = phon->isotope->gamma_isotope;

    if (run.my_rank == 0) {
        if (phon->isotope->include_isotope == 2) {

            auto file_iso = run.job_title + ".self_isotope";
            std::ofstream ofs_iso;

            ofs_iso.open(file_iso.c_str(), std::ios::out);
            if (!ofs_iso) exit("writeSelfenergyIsotope", "Could not open file_iso");

            ofs_iso << "# Phonon selfenergy due to phonon-isotope scatterings for the irreducible k points."
                    << std::endl;
            ofs_iso << "# Irred. knum, mode num, frequency [cm^-1], Gamma_iso [cm^-1]\n\n";

            for (unsigned int i = 0; i < phon->dos->kmesh_dos->nk_irred; ++i) {
                ofs_iso << "# Irreducible k point  : " << std::setw(8) << i + 1;
                ofs_iso << " (" << std::setw(4) << phon->dos->kmesh_dos->kpoint_irred_all[i].size() << ")\n";

                const auto knum = phon->dos->kmesh_dos->kpoint_irred_all[i][0].knum;

                ofs_iso << "## xk = " << std::setw(3);
                for (k = 0; k < 3; ++k) ofs_iso << std::setw(15) << phon->dos->kmesh_dos->xk[knum][k];
                ofs_iso << '\n';

                for (k = 0; k < ns; ++k) {
                    ofs_iso << std::setw(7) << i + 1;
                    ofs_iso << std::setw(5) << k + 1;
                    ofs_iso << std::setw(15) << in_kayser(eval[knum][k]);
                    ofs_iso << std::setw(15) << in_kayser(gamma_iso[i][k]);
                    ofs_iso << '\n';
                }
                ofs_iso << '\n';
            }

            ofs_iso.close();
            print_output_file(run, file_iso, "Phonon self-energy due to phonon-isotope scattering (ISOTOPE = 2)");
        }
    }
}

void Writes::writeNormalModeAnimation(const double xk_in[3], const unsigned int ncell[3]) const
{
    unsigned int i, j, k;
    unsigned int iband, istep;
    const auto ns = phon->dynamical->neval;
    const auto natmin = phon->system->get_primcell().number_of_atoms;
    const auto nsuper = ncell[0] * ncell[1] * ncell[2];
    unsigned int ntmp = nbands;
    unsigned int ndigits = 0;

    double phase_time;
    const auto max_disp_factor = 0.1;
    double lavec_super[3][3];
    double dmod[3];
    double xk[3], kvec[3];

    NDArray<double, 1> eval;
    NDArray<double, 2> evec_mag;
    NDArray<double, 2> evec_theta;
    NDArray<double, 2> disp_mag;
    NDArray<double, 1> mass;
    NDArray<double, 1> phase_cell;
    NDArray<double, 3> xmod;
    Eigen::MatrixXd xtmp;

    NDArray<std::complex<double>, 2> evec;

    std::ofstream ofs_anime;
    std::ostringstream ss;
    std::string file_anime;
    NDArray<std::string, 1> kd_tmp;

    for (i = 0; i < 3; ++i) {
        xk[i] = xk_in[i];
    }
    if (run.verbosity > 0) {
        std::cout << " -----------------------------------------------------------------\n\n";
        std::cout << " ANIME-tag is given: Making animation files for the given\n";
        std::cout << "                     k point ( ";
        std::cout << std::setw(5) << xk[0] << ", " << std::setw(5) << xk[1] << ", " << std::setw(5) << xk[2] << ").\n";
        std::cout << " ANIME_CELLSIZE = ";
        std::cout << std::setw(3) << ncell[0] << std::setw(3) << ncell[1] << std::setw(3) << ncell[2] << '\n';
        std::cout << " ANIME_FORMAT = " << anime_format << '\n';
    }

    for (i = 0; i < 3; ++i) dmod[i] = std::fmod(xk[i] * static_cast<double>(ncell[i]), 1.0);

    if (std::sqrt(dmod[0] * dmod[0] + dmod[1] * dmod[1] + dmod[2] * dmod[2]) > eps12) {
        warn("writeNormalModeAnimation", "The supercell size is not commensurate with given k point.");
    }

    rotvec(kvec, xk, phon->system->get_primcell().reciprocal_lattice_vector, 'T');
    const auto norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);
    if (norm > eps) {
        for (i = 0; i < 3; ++i) kvec[i] /= norm;
    }

    // Allocation

    eval.resize(ns);
    evec.resize(ns, ns);
    evec_mag.resize(ns, ns);
    evec_theta.resize(ns, ns);
    disp_mag.resize(ns, ns);
    xmod.resize(nsuper, natmin, 3);
    kd_tmp.resize(natmin);
    mass.resize(natmin);
    phase_cell.resize(nsuper);

    // Get eigenvalues and eigenvectors at xk

    phon->dynamical->eval_k(xk, kvec, phon->fcs_phonon->force_constant_with_cell[0], eval, evec, true);

    for (i = 0; i < ns; ++i) {
        for (j = 0; j < ns; ++j) {
            evec_mag[i][j] = std::abs(evec[i][j]);
            evec_theta[i][j] = std::arg(evec[i][j]);
        }
    }

    // Get fractional coordinates of atoms in a primitive cell

    xtmp.resize(natmin, 3);

    for (i = 0; i < natmin; ++i) {
        for (j = 0; j < 3; ++j) {
            xtmp(i, j) = phon->system->get_supercell(0).x_fractional(phon->system->get_map_p2s(0)[i][0], j);
        }
    }
    xtmp = xtmp * phon->system->get_supercell(0).lattice_vector.transpose();
    xtmp = xtmp * phon->system->get_primcell().lattice_vector.inverse().transpose();

    // Prepare fractional coordinates of atoms in the supercell
    unsigned int icell = 0;

    for (unsigned int ix = 0; ix < ncell[0]; ++ix) {
        for (unsigned int iy = 0; iy < ncell[1]; ++iy) {
            for (unsigned int iz = 0; iz < ncell[2]; ++iz) {

                phase_cell[icell] = 2.0 * pi *
                                    (xk_in[0] * static_cast<double>(ix) + xk_in[1] * static_cast<double>(iy) +
                                     xk_in[2] * static_cast<double>(iz));

                for (i = 0; i < natmin; ++i) {
                    xmod[icell][i][0] = (xtmp(i, 0) + static_cast<double>(ix)) / static_cast<double>(ncell[0]);
                    xmod[icell][i][1] = (xtmp(i, 1) + static_cast<double>(iy)) / static_cast<double>(ncell[1]);
                    xmod[icell][i][2] = (xtmp(i, 2) + static_cast<double>(iz)) / static_cast<double>(ncell[2]);
                }
                ++icell;
            }
        }
    }

    // Prepare atomic symbols and masses

    for (i = 0; i < natmin; ++i) {
        k = phon->system->get_map_p2s(0)[i][0];
        kd_tmp[i] = phon->system->symbol_kd[phon->system->get_primcell().kind[i]];
        mass[i] = phon->system->get_mass_super()[k];
    }

    // Prepare lattice vectors of the supercell

    for (i = 0; i < 3; ++i) {
        lavec_super[i][0] = phon->system->get_primcell().lattice_vector(i, 0) * ncell[0] * Bohr_in_Angstrom;
        lavec_super[i][1] = phon->system->get_primcell().lattice_vector(i, 1) * ncell[1] * Bohr_in_Angstrom;
        lavec_super[i][2] = phon->system->get_primcell().lattice_vector(i, 2) * ncell[2] * Bohr_in_Angstrom;
    }

    // Normalize the magnitude of displacements

    auto mass_min = mass[0];
    for (i = 0; i < natmin; ++i) {
        if (mass[i] < mass_min) mass_min = mass[i];
    }

    for (iband = 0; iband < nbands; ++iband) {
        auto max_disp_mag = 0.0;

        for (j = 0; j < ns; ++j) {
            disp_mag[iband][j] = std::sqrt(mass_min / mass[j / 3]) * evec_mag[iband][j];
        }

        for (j = 0; j < natmin; ++j) {
            auto disp_mag_tmp = 0.0;
            for (k = 0; k < 3; ++k) disp_mag_tmp += pow2(disp_mag[iband][3 * j + k]);
            disp_mag_tmp = std::sqrt(disp_mag_tmp);
            max_disp_mag = std::max(max_disp_mag, disp_mag_tmp);
        }

        for (j = 0; j < ns; ++j) disp_mag[iband][j] *= max_disp_factor / max_disp_mag;
    }

    // Convert atomic positions to Cartesian coordinate
    for (i = 0; i < nsuper; ++i) {
        for (j = 0; j < natmin; ++j) {
            rotvec(xmod[i][j], xmod[i][j], lavec_super);
        }
    }

    while (ntmp > 0) {
        ++ndigits;
        ntmp /= 10;
    }

    if (anime_format == "XSF" || anime_format == "AXSF") {

        // Save animation to AXSF (XcrysDen) files

        for (iband = 0; iband < nbands; ++iband) {

            eval[iband] = phon->dynamical->freq(eval[iband]);
            ss.str("");
            ss.clear();
            ss << std::setw(ndigits) << std::setfill('0') << iband + 1;
            const auto result = ss.str();

            file_anime = run.job_title + ".anime" + result + ".axsf";

            ofs_anime.open(file_anime.c_str(), std::ios::out);
            if (!ofs_anime) exit("writeNormalModeAnimation", "cannot open file_anime");

            ofs_anime.setf(std::ios::scientific);

            ofs_anime << "ANIMSTEPS " << anime_frames << '\n';
            ofs_anime << "CRYSTAL\n";
            ofs_anime << "PRIMVEC\n";

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    ofs_anime << std::setw(15) << lavec_super[j][i];
                }
                ofs_anime << '\n';
            }

            for (istep = 0; istep < anime_frames; ++istep) {

                phase_time = 2.0 * pi / static_cast<double>(anime_frames) * static_cast<double>(istep);

                ofs_anime << "PRIMCOORD " << std::setw(10) << istep + 1 << '\n';
                ofs_anime << std::setw(10) << natmin * nsuper << std::setw(10) << 1 << '\n';

                for (i = 0; i < nsuper; ++i) {
                    for (j = 0; j < natmin; ++j) {

                        ofs_anime << std::setw(10) << kd_tmp[j];

                        for (k = 0; k < 3; ++k) {
                            ofs_anime << std::setw(15)
                                      << xmod[i][j][k] +
                                             disp_mag[iband][3 * j + k] *
                                                 std::sin(phase_cell[i] + evec_theta[iband][3 * j + k] + phase_time);
                        }
                        ofs_anime << '\n';
                    }
                }
            }

            ofs_anime.close();
        }

    } else if (anime_format == "XYZ") {

        // Save animation to XYZ files

        for (iband = 0; iband < nbands; ++iband) {

            eval[iband] = phon->dynamical->freq(eval[iband]);
            ss.str("");
            ss.clear();
            ss << std::setw(ndigits) << std::setfill('0') << iband + 1;
            const auto result = ss.str();

            file_anime = run.job_title + ".anime" + result + ".xyz";

            ofs_anime.open(file_anime.c_str(), std::ios::out);
            if (!ofs_anime) exit("writeNormalModeAnimation", "cannot open file_anime");

            ofs_anime.setf(std::ios::scientific);

            for (istep = 0; istep < anime_frames; ++istep) {

                phase_time = 2.0 * pi / static_cast<double>(anime_frames) * static_cast<double>(istep);

                ofs_anime.unsetf(std::ios::scientific);

                ofs_anime << natmin * nsuper << '\n';
                ofs_anime << "Mode " << std::setw(4) << iband + 1 << " at (";
                for (i = 0; i < 3; ++i) ofs_anime << std::setw(8) << xk_in[i];
                ofs_anime << "), Frequency (cm^-1) = " << in_kayser(eval[iband]) << ", Time step = " << std::setw(4)
                          << istep + 1 << '\n';

                ofs_anime.setf(std::ios::scientific);

                for (i = 0; i < nsuper; ++i) {
                    for (j = 0; j < natmin; ++j) {

                        ofs_anime << std::setw(4) << kd_tmp[j];

                        for (k = 0; k < 3; ++k) {
                            ofs_anime << std::setw(15)
                                      << xmod[i][j][k] +
                                             disp_mag[iband][3 * j + k] *
                                                 std::sin(phase_cell[i] + evec_theta[iband][3 * j + k] + phase_time);
                        }
                        ofs_anime << '\n';
                    }
                }
            }
            ofs_anime.close();
        }
    }

    xmod.clear();
    kd_tmp.clear();
    eval.clear();
    evec.clear();
    phase_cell.clear();
    evec_mag.clear();
    evec_theta.clear();
    disp_mag.clear();
    mass.clear();
}

void Writes::printNormalmodeBorncharge() const
{

    if (run.my_rank == 0) {

        if (!phon->dielec->has_borncharge()) {
            warn("printNormalmodeBorncharge", "ZMODE = 1 requires BORNINFO; the .zmode file is not created.");
            return;
        }

        auto zstar_born = phon->dielec->get_zstar_mode(*phon->dynamical);

        const auto ns = phon->dynamical->neval;

        std::string file_zstar = run.job_title + ".zmode";
        std::ofstream ofs_zstar;
        ofs_zstar.open(file_zstar.c_str(), std::ios::out);
        if (!ofs_zstar) exit("printNormalmodeBorncharge", "Cannot open file file_zstar");

        ofs_zstar << "# Born effective charges of each phonon mode at q = (0, 0, 0). Unit is (amu)^{-1/2}\n";
        for (auto is = 0; is < ns; ++is) {
            ofs_zstar << "# Mode " << std::setw(5) << is + 1 << '\n';
            ofs_zstar << "#";
            ofs_zstar << std::setw(14) << 'x';
            ofs_zstar << std::setw(15) << 'y';
            ofs_zstar << std::setw(15) << 'z';
            ofs_zstar << '\n';
            for (auto i = 0; i < 3; ++i) {
                ofs_zstar << std::setw(15) << std::fixed << zstar_born[is][i];
            }
            ofs_zstar << "\n\n";
        }
        ofs_zstar.close();
    }
}

namespace
{

std::string irrep_activity_string(const GammaModeGroup &grp)
{
    std::string str;
    if (grp.is_acoustic) {
        str = "acoustic";
    } else if (grp.ir_active && grp.raman_active) {
        str = "IR+Raman";
    } else if (grp.ir_active) {
        str = "IR";
    } else if (grp.raman_active) {
        str = "Raman";
    } else {
        str = "silent";
    }
    if (!grp.activity_known) {
        str += " (?)";
    }
    return str;
}

} // namespace

void Writes::printModeIrrepsSummary() const
{
    if (run.my_rank != 0 || run.verbosity == 0) {
        return;
    }

    const auto &result = phon->mode_symmetry->get_result();

    std::cout << '\n';
    std::cout << " -----------------------------------------------------------------\n\n";
    std::cout << " Irreducible representations of phonon modes at Gamma (IRREPS = 1)\n\n";

    for (const auto &warning: result.warnings) {
        std::cout << "  WARNING: " << warning << '\n';
    }
    if (!result.warnings.empty()) {
        std::cout << '\n';
    }

    if (result.available) {
        std::cout << "  Point group : " << result.pg_schoenflies << " (" << result.pg_international << ")";
        if (!result.spg_symbol.empty()) {
            std::cout << "   [space group: " << result.spg_symbol << "]";
        }
        std::cout << "\n";
        if (!result.axis_convention_note.empty()) {
            std::cout << "  Axis convention: " << result.axis_convention_note << '\n';
        }
        std::cout << '\n';
        std::cout << "  Gamma_total    = " << result.decomp_total << '\n';
        std::cout << "  Gamma_acoustic = " << result.decomp_acoustic << '\n';
        std::cout << "  Gamma_optic    = " << result.decomp_optic << "\n\n";
    } else {
        std::cout << "  Mulliken labels could not be assigned for this run (see warnings);\n";
        std::cout << "  frequencies and projection-based activities are listed below.\n\n";
    }

    std::cout << "  " << std::setw(9) << "multiplet" << std::setw(12) << "branches" << std::setw(15) << "freq (cm^-1)"
              << std::setw(12) << "irrep" << std::setw(5) << "deg" << std::setw(11) << "activity";
    if (result.has_borncharge) {
        std::cout << std::setw(22) << "IR strength (e^2/amu)";
    }
    std::cout << '\n';

    auto ig = 0;
    for (const auto &grp: result.groups) {
        ++ig;
        const auto branch_first = grp.mode_indices.front() + 1;
        const auto branch_last = grp.mode_indices.back() + 1;
        std::cout << "  " << std::setw(9) << ig << std::setw(5) << branch_first << " -" << std::setw(5) << branch_last
                  << std::setw(15) << std::fixed << std::setprecision(4) << in_kayser(grp.omega) << std::setw(12)
                  << (grp.irrep_label.empty() ? "-" : grp.irrep_label) << std::setw(5) << grp.mode_indices.size()
                  << std::setw(11) << irrep_activity_string(grp);
        if (grp.has_ir_strength) {
            std::cout << std::setw(22) << std::scientific << std::setprecision(4) << grp.ir_strength.trace()
                      << std::fixed;
        } else if (result.has_borncharge) {
            std::cout << std::setw(22) << "-";
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    if (phon->dynamical->nonanalytic > 0) {
        std::cout << "  Note: frequencies, labels, and strengths refer to the analytic (TO)\n";
        std::cout << "        Gamma limit; the direction-dependent LO-TO splitting is not\n";
        std::cout << "        reflected in this table.\n\n";
    }
}

void Writes::writeModeIrreps() const
{
    if (run.my_rank != 0) {
        return;
    }

    const auto &result = phon->mode_symmetry->get_result();

    const auto file_irreps = run.job_title + ".irreps";
    std::ofstream ofs_irreps;
    ofs_irreps.open(file_irreps.c_str(), std::ios::out);
    if (!ofs_irreps) {
        exit("writeModeIrreps", "Cannot open file file_irreps");
    }

    ofs_irreps << "# Irreducible representations of phonon modes at q = (0, 0, 0)\n";

    for (const auto &warning: result.warnings) {
        ofs_irreps << "# WARNING: " << warning << '\n';
    }

    if (result.available) {
        ofs_irreps << "# Point group: " << result.pg_schoenflies << " (" << result.pg_international << ")";
        if (!result.spg_symbol.empty()) {
            ofs_irreps << "; space group: " << result.spg_symbol;
        }
        ofs_irreps << '\n';
        if (!result.axis_convention_note.empty()) {
            ofs_irreps << "# Axis convention: " << result.axis_convention_note << '\n';
        }
        ofs_irreps << "# Gamma_total    = " << result.decomp_total << '\n';
        ofs_irreps << "# Gamma_acoustic = " << result.decomp_acoustic << '\n';
        ofs_irreps << "# Gamma_optic    = " << result.decomp_optic << '\n';
    } else {
        ofs_irreps << "# Mulliken labels could not be assigned for this run (see warnings).\n";
    }

    if (!result.classes.empty()) {
        ofs_irreps << "# Classes (label, #elements, axes or mirror normals in Cartesian):\n";
        auto icl = 0;
        for (const auto &cl: result.classes) {
            ++icl;
            ofs_irreps << "#  " << std::setw(3) << icl << ": " << std::setw(12) << std::left << cl.label << std::right
                       << std::setw(4) << cl.nelem;
            if (!cl.axes.empty()) {
                ofs_irreps << "   axes:";
                for (const auto &ax: cl.axes) {
                    ofs_irreps << " [" << std::fixed << std::setprecision(3) << std::setw(7) << ax.x() << std::setw(7)
                               << ax.y() << std::setw(7) << ax.z() << "]";
                }
            }
            ofs_irreps << '\n';
        }
    }

    if (phon->dynamical->nonanalytic > 0) {
        ofs_irreps << "# Note: frequencies, labels, and strengths refer to the analytic (TO) "
                      "Gamma limit.\n";
    }

    ofs_irreps << "#\n";
    ofs_irreps << "# multiplet, first & last branch, frequency [cm^-1], irrep, degeneracy, "
                  "activity, acoustic content tr(P_T P)";
    if (result.has_borncharge) {
        ofs_irreps << ", IR strength I_tot [e^2/amu]";
    }
    ofs_irreps << '\n';
    auto ig = 0;
    for (const auto &grp: result.groups) {
        ++ig;
        ofs_irreps << std::setw(5) << ig << std::setw(6) << grp.mode_indices.front() + 1 << std::setw(6)
                   << grp.mode_indices.back() + 1 << std::setw(16) << std::fixed << std::setprecision(6)
                   << in_kayser(grp.omega) << "  " << std::setw(12) << std::left
                   << (grp.irrep_label.empty() ? "-" : grp.irrep_label) << std::right << std::setw(4)
                   << grp.mode_indices.size() << "  " << std::setw(10) << std::left << irrep_activity_string(grp)
                   << std::right << std::setw(8) << std::fixed << std::setprecision(3) << grp.acoustic_content;
        if (grp.has_ir_strength) {
            ofs_irreps << std::setw(15) << std::scientific << std::setprecision(6) << grp.ir_strength.trace()
                       << std::fixed;
        } else if (result.has_borncharge) {
            ofs_irreps << std::setw(15) << "-";
        }
        ofs_irreps << '\n';
    }

    // Approximate projection weights for multiplets whose exact activity
    // could not be certified.
    ig = 0;
    for (const auto &grp: result.groups) {
        ++ig;
        if (grp.activity_known) {
            continue;
        }
        ofs_irreps << "# multiplet " << ig << " approximate projections: n_IR = " << std::fixed << std::setprecision(3)
                   << grp.n_ir_proj << ", n_Raman = " << grp.n_raman_proj << '\n';
    }

    if (result.has_borncharge) {
        ofs_irreps << "#\n# IR oscillator-strength tensors "
                      "S_ab = sum_{nu in multiplet} Z*_mode[nu][a] Z*_mode[nu][b] [e^2/amu]:\n";
        ofs_irreps << "# multiplet" << std::setw(15) << "S_xx" << std::setw(15) << "S_yy" << std::setw(15) << "S_zz"
                   << std::setw(15) << "S_xy" << std::setw(15) << "S_yz" << std::setw(15) << "S_zx" << '\n';
        ig = 0;
        for (const auto &grp: result.groups) {
            ++ig;
            if (!grp.has_ir_strength) {
                continue;
            }
            const auto &s = grp.ir_strength;
            ofs_irreps << std::setw(10) << ig << std::scientific << std::setprecision(6) << std::setw(15) << s(0, 0)
                       << std::setw(15) << s(1, 1) << std::setw(15) << s(2, 2) << std::setw(15) << s(0, 1)
                       << std::setw(15) << s(1, 2) << std::setw(15) << s(2, 0) << std::fixed << '\n';
        }
    }

    // Raw numerical characters: everything needed to re-derive labels under a
    // different axis convention.
    if (!result.classes.empty()) {
        ofs_irreps << "#\n# Characters chi(class) per multiplet (real part, class-averaged):\n";
        ofs_irreps << "# multiplet";
        for (const auto &cl: result.classes) {
            ofs_irreps << std::setw(10) << cl.label;
        }
        ofs_irreps << '\n';
        ig = 0;
        for (const auto &grp: result.groups) {
            ++ig;
            ofs_irreps << std::setw(10) << ig;
            for (const auto chi: grp.characters) {
                ofs_irreps << std::setw(10) << std::fixed << std::setprecision(3) << chi;
            }
            ofs_irreps << '\n';
        }
    }

    ofs_irreps.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_irreps;
        std::cout << " : Irreducible representations and IR/Raman activity at Gamma\n";
    }
}

void Writes::writeParticipationRatio() const
{
    std::string fname_pr, fname_apr;

    if (phon->kpoint->kpoint_general.get() && phon->dynamical->dymat_general) {
        fname_pr = run.job_title + ".pr";
        fname_apr = run.job_title + ".apr";
        writeParticipationRatioEach(fname_pr,
                                    fname_apr,
                                    phon->kpoint->kpoint_general->nk,
                                    phon->kpoint->kpoint_general->xk,
                                    phon->dynamical->dymat_general->get_eigenvalues(),
                                    phon->dynamical->dymat_general->get_eigenvectors());
    }

    if (phon->kpoint->kpoint_bs.get() && phon->dynamical->dymat_band) {
        fname_pr = run.job_title + ".band.pr";
        fname_apr = run.job_title + ".band.apr";
        writeParticipationRatioEach(fname_pr,
                                    fname_apr,
                                    phon->kpoint->kpoint_bs->nk,
                                    phon->kpoint->kpoint_bs->xk,
                                    phon->dynamical->dymat_band->get_eigenvalues(),
                                    phon->dynamical->dymat_band->get_eigenvectors());
    }

    if (phon->dos->kmesh_dos.get() && phon->dos->dymat_dos.get()) {
        fname_pr = run.job_title + ".mesh.pr";
        fname_apr = run.job_title + ".mesh.apr";
        writeParticipationRatioMesh(fname_pr,
                                    fname_apr,
                                    phon->dos->kmesh_dos.get(),
                                    phon->dos->dymat_dos->get_eigenvalues(),
                                    phon->dos->dymat_dos->get_eigenvectors());
    }
}

void Writes::writeParticipationRatioEach(const std::string &fname_pr, const std::string &fname_apr,
                                         const unsigned int nk_in, const double *const *xk_in,
                                         const double *const *eval_in,
                                         const std::complex<double> *const *const *evec_in) const
{
    unsigned int i, j, k;
    const auto neval = phon->dynamical->neval;
    const auto natmin = phon->system->get_primcell().number_of_atoms;

    NDArray<double, 2> participation_ratio;
    NDArray<double, 3> atomic_participation_ratio;

    std::ofstream ofs_pr, ofs_apr;

    ofs_pr.open(fname_pr.c_str(), std::ios::out);
    if (!ofs_pr) exit("writeParticipationRatioEach", "cannot open file_pr");
    ofs_pr.setf(std::ios::scientific);

    ofs_apr.open(fname_apr.c_str(), std::ios::out);
    if (!ofs_apr) exit("writeParticipationRatio", "cannot open file_apr");
    ofs_apr.setf(std::ios::scientific);

    participation_ratio.resize(nk_in, neval);
    atomic_participation_ratio.resize(nk_in, neval, natmin);

    phon->dynamical->calc_participation_ratio_all(nk_in, evec_in, participation_ratio, atomic_participation_ratio);

    ofs_pr << "# Participation ratio of each phonon modes at k points\n";
    ofs_pr << "# kpoint, mode, PR[kpoint][mode]\n";

    for (i = 0; i < nk_in; ++i) {
        ofs_pr << "#" << std::setw(8) << i + 1;
        ofs_pr << " xk = ";
        for (j = 0; j < 3; ++j) {
            ofs_pr << std::setw(15) << xk_in[i][j];
        }
        ofs_pr << '\n';
        for (j = 0; j < nbands; ++j) {
            ofs_pr << std::setw(8) << i + 1;
            ofs_pr << std::setw(5) << j + 1;
            ofs_pr << std::setw(15) << participation_ratio[i][j];
            ofs_pr << '\n';
        }
        ofs_pr << '\n';
    }
    ofs_pr.close();

    ofs_apr << "# Atomic participation ratio of each phonon modes at k points\n";
    ofs_apr << "# kpoint, mode, atom, APR[kpoint][mode][atom]" << '\n';

    for (i = 0; i < nk_in; ++i) {
        ofs_apr << "#" << std::setw(8) << i + 1;
        ofs_apr << " xk = ";
        for (j = 0; j < 3; ++j) {
            ofs_apr << std::setw(15) << xk_in[i][j];
        }
        ofs_apr << '\n';
        for (j = 0; j < nbands; ++j) {
            for (k = 0; k < natmin; ++k) {
                ofs_apr << std::setw(8) << i + 1;
                ofs_apr << std::setw(5) << j + 1;
                ofs_apr << std::setw(5) << k + 1;
                ofs_apr << std::setw(15) << atomic_participation_ratio[i][j][k];
                ofs_apr << '\n';
            }
        }
        ofs_apr << '\n';
    }
    ofs_apr.close();

    participation_ratio.clear();
    atomic_participation_ratio.clear();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_pr;
        std::cout << " : Participation ratio for all k points\n";
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_apr;
        std::cout << " : Atomic participation ratio for all k points\n";
    }
}

void Writes::writeParticipationRatioMesh(const std::string &fname_pr, const std::string &fname_apr,
                                         const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                         const std::complex<double> *const *const *evec_in) const
{
    unsigned int i, j, k;
    unsigned int knum;
    const auto neval = phon->dynamical->neval;
    const auto natmin = phon->system->get_primcell().number_of_atoms;
    const auto nk = kmesh_in->nk;

    NDArray<double, 2> participation_ratio;
    NDArray<double, 3> atomic_participation_ratio;

    std::ofstream ofs_pr, ofs_apr;

    ofs_pr.open(fname_pr.c_str(), std::ios::out);
    if (!ofs_pr) exit("writeParticipationRatioMesh", "cannot open file_pr");
    ofs_pr.setf(std::ios::scientific);

    ofs_apr.open(fname_apr.c_str(), std::ios::out);
    if (!ofs_apr) exit("writeParticipationRatio", "cannot open file_apr");
    ofs_apr.setf(std::ios::scientific);

    participation_ratio.resize(nk, neval);
    atomic_participation_ratio.resize(nk, neval, natmin);

    phon->dynamical->calc_participation_ratio_all(nk, evec_in, participation_ratio, atomic_participation_ratio);

    ofs_pr << "# Participation ratio of each phonon modes at k points\n";
    ofs_pr << "# irred. kpoint, mode, frequency[kpoint][mode] (cm^-1), PR[kpoint][mode]\n";

    for (i = 0; i < kmesh_in->nk_irred; ++i) {
        knum = kmesh_in->kpoint_irred_all[i][0].knum;
        ofs_pr << "#" << std::setw(8) << i + 1;
        ofs_pr << " xk = ";
        for (j = 0; j < 3; ++j) {
            ofs_pr << std::setw(15) << kmesh_in->xk[knum][j];
        }
        ofs_pr << '\n';
        for (j = 0; j < nbands; ++j) {
            ofs_pr << std::setw(8) << i + 1;
            ofs_pr << std::setw(5) << j + 1;
            ofs_pr << std::setw(15) << in_kayser(eval_in[knum][j]);
            ofs_pr << std::setw(15) << participation_ratio[knum][j];
            ofs_pr << '\n';
        }
        ofs_pr << '\n';
    }
    ofs_pr.close();

    ofs_apr << "# Atomic participation ratio of each phonon modes at k points\n";
    ofs_apr << "# irred. kpoint, mode, atom, frequency[kpoint][mode] (cm^-1), APR[kpoint][mode][atom]\n";

    for (i = 0; i < kmesh_in->nk_irred; ++i) {
        knum = kmesh_in->kpoint_irred_all[i][0].knum;

        ofs_apr << "#" << std::setw(8) << i + 1;
        ofs_apr << " xk = ";
        for (j = 0; j < 3; ++j) {
            ofs_apr << std::setw(15) << kmesh_in->xk[knum][j];
        }
        ofs_apr << '\n';
        for (j = 0; j < nbands; ++j) {
            for (k = 0; k < natmin; ++k) {
                ofs_apr << std::setw(8) << i + 1;
                ofs_apr << std::setw(5) << j + 1;
                ofs_apr << std::setw(5) << k + 1;
                ofs_apr << std::setw(15) << in_kayser(eval_in[knum][j]);
                ofs_apr << std::setw(15) << atomic_participation_ratio[knum][j][k];
                ofs_apr << '\n';
            }
        }
        ofs_apr << '\n';
    }
    ofs_apr.close();

    participation_ratio.clear();
    atomic_participation_ratio.clear();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_pr;
        std::cout << " : Participation ratio for all k points\n";
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << fname_apr;
        std::cout << " : Atomic participation ratio for all k points\n";
    }
}

void Writes::writeDielectricFunction() const
{
    std::ofstream ofs_dielec;
    auto file_dielec = run.job_title + ".dielec";

    ofs_dielec.open(file_dielec.c_str(), std::ios::out);
    if (!ofs_dielec) exit("writePhononVel", "cannot open file_vel");

    unsigned int nomega;
    auto omega_grid = phon->dielec->get_omega_grid(nomega);
    auto dielecfunc = phon->dielec->get_dielectric_func();

    ofs_dielec << "# Real part of dielectric function (phonon part only)\n";
    ofs_dielec << "# Frequency (cm^-1), xx, yy, zz,   xy, xz, yx, yz, zx, zy\n";
    for (auto iomega = 0; iomega < nomega; ++iomega) {
        ofs_dielec << std::setw(10) << omega_grid[iomega];
        for (auto i = 0; i < 3; ++i) {
            ofs_dielec << std::setw(15) << dielecfunc[iomega][i][i];
        }
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                if (i == j) continue;
                ofs_dielec << std::setw(15) << dielecfunc[iomega][i][j];
            }
        }
        ofs_dielec << '\n';
    }
    ofs_dielec << '\n';
    ofs_dielec.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_dielec;
        std::cout << " : Frequency-dependent dielectric function\n";
    }
}

void Writes::writePhononEnergies(const unsigned int nk_in, const double *const *const *eval_in, const bool is_qha,
                                 const int bubble) const
{
    const auto ns = phon->dynamical->neval;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::ofstream ofs_energy;
    std::string file_energy;

    if (is_qha) {
        file_energy = run.job_title + ".qha_eval";
    } else {
        if (bubble == 0) {
            file_energy = run.job_title + ".scph_eval";
        } else if (bubble == 1) {
            file_energy = run.job_title + ".scph+bubble(0)_eval";
        } else if (bubble == 2) {
            file_energy = run.job_title + ".scph+bubble(w)_eval";
        } else if (bubble == 3) {
            file_energy = run.job_title + ".scph+bubble(wQP)_eval";
        }
    }

    ofs_energy.open(file_energy.c_str(), std::ios::out);
    if (!ofs_energy) exit("writePhononEnergies", "cannot open file_energy");

    ofs_energy << "# K point, mode, Temperature [K], Eigenvalues [cm^-1]\n";

    for (unsigned int ik = 0; ik < nk_in; ++ik) {
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int iT = 0; iT < NT; ++iT) {
                const auto temp = Tmin + static_cast<double>(iT) * dT;

                ofs_energy << std::setw(5) << ik + 1;
                ofs_energy << std::setw(5) << is + 1;
                ofs_energy << std::setw(8) << temp;
                ofs_energy << std::setw(15) << in_kayser(eval_in[iT][ik][is]);
                ofs_energy << '\n';
            }
            ofs_energy << '\n';
        }
        ofs_energy << '\n';
    }

    ofs_energy.close();
}

void Writes::writePhononBands(const unsigned int nk_in, const double *kaxis_in, const double *const *const *eval,
                              const bool is_qha, const int bubble) const
{
    std::ofstream ofs_bands;
    std::string file_bands;

    if (is_qha) {
        file_bands = run.job_title + ".qha_bands";
    } else {
        if (bubble == 0) {
            file_bands = run.job_title + ".scph_bands";
        } else if (bubble == 1) {
            file_bands = run.job_title + ".scph+bubble(0)_bands";
        } else if (bubble == 2) {
            file_bands = run.job_title + ".scph+bubble(w)_bands";
        } else if (bubble == 3) {
            file_bands = run.job_title + ".scph+bubble(wQP)_bands";
        }
    }

    ofs_bands.open(file_bands.c_str(), std::ios::out);
    if (!ofs_bands) exit("writePhononBands", "cannot open file_bands");

    unsigned int i;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;
    const auto ns = phon->dynamical->neval;
    auto kcount = 0;

    std::string str_tmp = "NONE";
    std::string str_kpath;
    std::string str_kval;

    for (i = 0; i < phon->kpoint->kpInp.size(); ++i) {
        if (str_tmp != phon->kpoint->kpInp[i].kpelem[0]) {
            str_tmp = phon->kpoint->kpInp[i].kpelem[0];
            str_kpath += " " + str_tmp;

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << kaxis_in[kcount];
            str_kval += " " + ss.str();
        }
        kcount += std::atoi(phon->kpoint->kpInp[i].kpelem[8].c_str());

        if (str_tmp != phon->kpoint->kpInp[i].kpelem[4]) {
            str_tmp = phon->kpoint->kpInp[i].kpelem[4];
            str_kpath += " " + str_tmp;

            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << kaxis_in[kcount - 1];
            str_kval += " " + ss.str();
        }
    }

    ofs_bands << "# " << str_kpath << '\n';
    ofs_bands << "#" << str_kval << '\n';
    ofs_bands << "# Temperature [K], k-axis, Eigenvalues [cm^-1]\n";

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + static_cast<double>(iT) * dT;

        for (i = 0; i < nk_in; ++i) {
            ofs_bands << std::setw(15) << std::fixed << temp;
            ofs_bands << std::setw(15) << std::fixed << kaxis_in[i];
            for (unsigned int j = 0; j < ns; ++j) {
                ofs_bands << std::setw(15) << std::scientific << in_kayser(eval[iT][i][j]);
            }
            ofs_bands << '\n';
        }
        ofs_bands << '\n';
    }

    ofs_bands.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_bands;
        if (is_qha) {
            std::cout << " : QHA band structure\n";
        } else {
            if (bubble == 0) {
                std::cout << " : SCPH band structure\n";
            } else if (bubble == 1) {
                std::cout << " : SCPH+Bubble(0) band structure\n";
            } else if (bubble == 2) {
                std::cout << " : SCPH+Bubble(w) band structure\n";
            } else if (bubble == 3) {
                std::cout << " : SCPH+Bubble(wQP) band structure\n";
            }
        }
    }
}

void Writes::writePhononDos(double **dos_in, const bool is_qha, const int bubble) const
{
    unsigned int iT;
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::ofstream ofs_dos;
    std::string file_dos;

    if (is_qha) {
        file_dos = run.job_title + ".qha_dos";
    } else {
        if (bubble == 0) {
            file_dos = run.job_title + ".scph_dos";
        } else if (bubble == 1) {
            file_dos = run.job_title + ".scph+bubble(0)_dos";
        } else if (bubble == 2) {
            file_dos = run.job_title + ".scph+bubble(w)_dos";
        } else if (bubble == 3) {
            file_dos = run.job_title + ".scph+bubble(wQP)_dos";
        }
    }

    ofs_dos.open(file_dos.c_str(), std::ios::out);
    if (!ofs_dos) exit("writePhononDos", "cannot open file_dos");

    ofs_dos << "# ";

    for (iT = 0; iT < NT; ++iT) {
        ofs_dos << std::setw(15) << Tmin + static_cast<double>(iT) * dT;
    }
    ofs_dos << '\n';

    for (unsigned int j = 0; j < phon->dos->n_energy; ++j) {
        ofs_dos << std::setw(15) << phon->dos->energy_dos[j];

        for (iT = 0; iT < NT; ++iT) {
            ofs_dos << std::setw(15) << dos_in[iT][j];
        }
        ofs_dos << '\n';
    }

    ofs_dos << '\n';
    ofs_dos.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_dos;
        if (is_qha) {
            std::cout << " : QHA DOS\n";
        } else {
            if (bubble == 0) {
                std::cout << " : SCPH DOS\n";
            } else if (bubble == 1) {
                std::cout << " : SCPH+Bubble(0) DOS\n";
            } else if (bubble == 2) {
                std::cout << " : SCPH+Bubble(w) DOS\n";
            } else if (bubble == 3) {
                std::cout << " : SCPH+Bubble(wQP) DOS\n";
            }
        }
    }
}

void Writes::writeThermodynamicFunc(double *heat_capacity, double *heat_capacity_correction, double *FE_QHA,
                                    double *dFE_scph, double *FE_total, double *entropy, const double *v0_renorm,
                                    const bool is_qha) const
{
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    bool print_anharmonic_correction_Cv = false;

    if (heat_capacity_correction) {
        print_anharmonic_correction_Cv = true;
    }

    std::ofstream ofs_thermo;
    std::string file_thermo;

    if (is_qha) {
        file_thermo = run.job_title + ".qha_thermo";
    } else {
        file_thermo = run.job_title + ".scph_thermo";
    }
    ofs_thermo.open(file_thermo.c_str(), std::ios::out);
    if (!ofs_thermo) exit("writeThermodynamicFunc", "cannot open file_thermo");

    // write header
    if (v0_renorm) {
        ofs_thermo << "# The renormalized static potential Phi_0 is also shown.\n";
    }
    if (phon->thermodynamics->calc_FE_bubble) {
        ofs_thermo << "# The bubble free-energy calculated on top of the SCPH wavefunction is also shown.\n";
        ofs_thermo << "# However, the bubble contributions to the heat capacity and entropy are not included.\n";
        ofs_thermo << "# If these are needed, please fit the free energy data including the bubble term \n "
                      "# by polynomial function and then estimate S and Cv by numerical derivatives.\n";
    }
    if (!is_qha) {
        ofs_thermo << "# The Cv data accounts for the QHA-like term only.\n";
    }

    ofs_thermo << "# Temperature [K], Cv [in kB unit]";
    if (print_anharmonic_correction_Cv) {
        ofs_thermo << ", Cv (anharm correction) [in kB unit]";
    }
    ofs_thermo << ", F_{vib} (QHA term) [Ry]";
    // do not write scph correction in QHA + structural optimization
    if (run.mode == "SCPH") {
        ofs_thermo << ", F_{vib} (SCPH correction) [Ry]";
    }
    if (phon->thermodynamics->calc_FE_bubble) {
        ofs_thermo << ", F_{vib} (Bubble correction) [Ry]";
    }
    // write renormalized zero-th order IFC
    if (v0_renorm) {
        ofs_thermo << ", Phi0 [Ry]";
    }
    ofs_thermo << ", F_{total} [Ry], S_{vib} [in kB unit]\n";

    if (phon->thermodynamics->classical) {
        ofs_thermo << "# CLASSICAL = 1: Use classical limit.\n";
    }

    for (unsigned int iT = 0; iT < NT; ++iT) {

        const auto temp = Tmin + static_cast<double>(iT) * dT;

        ofs_thermo << std::setw(16) << std::fixed << temp;
        ofs_thermo << std::setw(18) << std::scientific << heat_capacity[iT] / k_Boltzmann;
        if (print_anharmonic_correction_Cv) {
            ofs_thermo << std::setw(18) << std::scientific << heat_capacity_correction[iT] / k_Boltzmann;
        }
        ofs_thermo << std::setw(18) << FE_QHA[iT];
        // skip scph correction for QHA + structural optimization
        if (run.mode == "SCPH") {
            ofs_thermo << std::setw(18) << dFE_scph[iT];
        }
        if (phon->thermodynamics->calc_FE_bubble) {
            ofs_thermo << std::setw(18) << phon->thermodynamics->FE_bubble[iT];
        }

        if (v0_renorm) {
            ofs_thermo << std::setw(18) << v0_renorm[iT];
        }
        ofs_thermo << std::setw(18) << FE_total[iT];
        ofs_thermo << std::setw(18) << entropy[iT] << '\n';
    }

    ofs_thermo.close();
    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_thermo;
        if (is_qha) {
            std::cout << " : QHA heat capacity, free energy, entropy\n";
        } else {
            std::cout << " : SCPH heat capacity, free energy, entropy\n";
        }
    }
}

void Writes::writeDielecFunc(double ****dielec_in, const bool is_qha) const
{
    const auto Tmin = phon->system->Tmin;
    const auto Tmax = phon->system->Tmax;
    const auto dT = phon->system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::ofstream ofs_dielec;
    std::string file_dielec;
    if (is_qha) {
        file_dielec = run.job_title + ".qha_dielec";
    } else {
        file_dielec = run.job_title + ".scph_dielec";
    }

    ofs_dielec.open(file_dielec.c_str(), std::ios::out);
    if (!ofs_dielec) exit("writeDielecFunc", "cannot open PREFIX.scph_dielec");

    unsigned int nomega;
    auto omega_grid = phon->dielec->get_omega_grid(nomega);

    ofs_dielec << "# Real part of dielectric function (phonon part only)\n";
    ofs_dielec << "# Temperature (K), Frequency (cm^-1), xx, yy, zz\n";

    for (unsigned int iT = 0; iT < NT; ++iT) {

        const auto temp = Tmin + static_cast<double>(iT) * dT;

        for (auto iomega = 0; iomega < nomega; ++iomega) {
            ofs_dielec << std::setw(16) << std::fixed << temp;
            ofs_dielec << std::setw(15) << std::scientific << omega_grid[iomega];
            for (auto i = 0; i < 3; ++i) {
                ofs_dielec << std::setw(15) << dielec_in[iT][iomega][i][i];
            }
            ofs_dielec << '\n';
        }
        ofs_dielec << '\n';
    }

    ofs_dielec.close();

    if (run.verbosity > 0) {
        std::cout << "  " << std::setw(run.job_title.length() + 12) << std::left << file_dielec;
        if (is_qha) {
            std::cout << " : QHA frequency-dependent dielectric function\n";
        } else {
            std::cout << " : SCPH frequency-dependent dielectric function\n";
        }
    }
}
