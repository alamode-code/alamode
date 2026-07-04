/*
 parsephon.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "parsephon.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "anharmonic_core.h"
#include "conductivity.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "gruneisen.h"
#include "integration.h"
#include "isotope.h"
#include "iterativebte.h"
#include "kpoint.h"
#include "memory.h"
#include "mode_analysis.h"
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

Input::Input(PHON *phon) : Pointers(phon), job_title(""), from_stdin(false)
{}

Input::~Input()
{
    if (ifs_input.is_open()) ifs_input.close();
}

void Input::parce_input(int narg, char **arg)
{
    if (narg == 1) {

        from_stdin = true;

    } else {

        from_stdin = false;

        ifs_input.open(arg[1], std::ios::in);
        if (!ifs_input) {
            std::cout << "No such file or directory: " << arg[1] << '\n';
            exit("parse_input", "could not open the file");
        }
    }

    if (!locate_tag("&general")) exit("parse_input", "&general entry not found in the input file");
    parse_general_vars();

    if (locate_tag("&cell")) parse_cell_parameter();

    const auto use_defaults_for_analysis = !locate_tag("&analysis");
    parse_analysis_vars(use_defaults_for_analysis);

    if (!locate_tag("&kpoint")) exit("parse_input", "&kpoint entry not found in the input file");
    parse_kpoints();

    if (phon->mode == "RTA") {
        const auto use_defaults_for_kappa = !locate_tag("&kappa");
        parse_kappa_vars(use_defaults_for_kappa);
    }

    if (phon->mode == "SCPH") {
        if (!locate_tag("&scph")) exit("parse_input", "&scph entry not found in the input file");
        parse_scph_vars();
    }
    if (phon->mode == "QHA") {
        if (!locate_tag("&qha")) exit("parse_input", "&qha entry not found in the input file");
        parse_qha_vars();
    }
    if ((phon->mode == "SCPH" || phon->mode == "QHA") && relaxation->relax_str != 0) {
        if (!locate_tag("&relax")) exit("parse_input", "&relax entry not found in the input file");
        parse_relax_vars();

        check_relax_vars();

        if (relaxation->relax_str != 1) {
            if (!locate_tag("&strain")) exit("parse_input", "&strain entry not found in the input file");
            parse_initial_strain();
        }
        if (!locate_tag("&displace")) exit("parse_input", "&displace entry not found in the input file");

        parse_initial_displace();
    }
}

void Input::parse_general_vars()
{
    // Read input parameters in the &general-field.

    int i;
    struct stat st
    {};
    std::string str_tmp;
    const std::vector<std::string> input_list{"PREFIX",
                                              "MODE",
                                              "NSYM",
                                              "TOLERANCE",
                                              "PRINTSYM",
                                              "TMIN",
                                              "TMAX",
                                              "DT",
                                              "NBANDS",
                                              "NONANALYTIC",
                                              "BORNINFO",
                                              "NA_SIGMA",
                                              "ISMEAR",
                                              "EPSILON",
                                              "EMIN",
                                              "EMAX",
                                              "DELTA_E",
                                              "RESTART",
                                              // "TREVSYM",
                                              "KD",
                                              "MASS",
                                              "TRISYM",
                                              "PREC_EWALD",
                                              "CLASSICAL",
                                              "BCONNECT",
                                              "BORNSYM",
                                              "VERBOSITY",
                                              "FC2FILE",
                                              "FC3FILE",
                                              "FC4FILE",
                                              "FCSFILE",
                                              "RESTART_4PH",
                                              "FILE_FORMAT",
                                              "FC2_TEMPERATURE",
                                              "ALLOW_UNCONVERGED",
                                              "DFC2FILE"};

    std::vector<std::string> no_defaults{"PREFIX", "MODE"};
    std::vector<std::string> kdname_input, masskd_v;
    std::map<std::string, std::string> general_var_dict;

    std::vector<double> masskd_input;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, general_var_dict);

    for (auto &no_default: no_defaults) {
        if (general_var_dict.find(no_default) == general_var_dict.end()) {
            exit("parse_general_vars",
                 "The following variable is not found in &general input region: ",
                 no_default.c_str());
        }
    }

    const auto prefix = general_var_dict["PREFIX"];
    auto mode = general_var_dict["MODE"];
    auto file_result = prefix + ".result";
    auto file_result4 = prefix + ".4ph.result";
    auto file_kappa_h5 = prefix + ".kappa.h5";

    // FILE_FORMAT decides the restart-file format (h5 is the default;
    // "text" forces the legacy .result files for one transition release).
    // It must be resolved before the restart auto-detection below.
    std::string file_format = "h5";
    assign_val(file_format, "FILE_FORMAT", general_var_dict);
    std::transform(file_format.begin(), file_format.end(), file_format.begin(), tolower);
    if (file_format != "h5" && file_format != "text") {
        exit("parse_general_vars", "FILE_FORMAT must be either h5 or text.");
    }
    const auto use_h5_io = file_format == "h5";
    this->use_hdf5_io = use_h5_io;

    std::transform(mode.begin(), mode.end(), mode.begin(), toupper);

    const auto fcsfile = general_var_dict["FCSFILE"];
    const auto fc2file = general_var_dict["FC2FILE"];
    const auto fc3file = general_var_dict["FC3FILE"];
    const auto fc4file = general_var_dict["FC4FILE"];

    if (fcsfile.empty() && fc2file.empty()) {
        exit("parse_general_vars", "Either FCSFILE or FC2FILE must be given to start a phonon calculation.");
    }

    if (!general_var_dict["KD"].empty()) {
        split_str_by_space(general_var_dict["KD"], kdname_input);
    }

    if (!general_var_dict["MASS"].empty()) {
        split_str_by_space(general_var_dict["MASS"], masskd_v);
        masskd_input.resize(masskd_v.size());
        for (i = 0; i < masskd_v.size(); ++i) {
            masskd_input[i] = my_cast<double>(masskd_v[i]);
        }
    }


    // Default values

    auto Tmin = 0.0;
    auto Tmax = 1000.0;
    auto dT = 10.0;

    auto delta_e = 10.0;

    unsigned int nonanalytic = 0;
    auto nsym = 0;
    auto tolerance = 1.0e-3;
    auto printsymmetry = false;
    //auto sym_time_reversal = false;
    auto use_triplet_symmetry = true;
    auto classical = false;
    unsigned int band_connection = 0;
    unsigned int bornsym = 0;
    unsigned int verbosity = 1;

    auto prec_ewald = 1.0e-12;

    // if a restart file exists in the current directory,
    // restart mode will be automatically turned on. With the default h5
    // format, an existing PREFIX.kappa.h5 or a legacy text file enables the
    // restart attempt for each channel independently; the actually finished
    // modes are read from the per-channel completion flags at setup.
    auto restart = stat(file_result.c_str(), &st) == 0;
    auto restart_4ph = stat(file_result4.c_str(), &st) == 0;
    if (use_h5_io && stat(file_kappa_h5.c_str(), &st) == 0) {
        restart = true;
        restart_4ph = true;
    }

    auto nbands = -1;
    std::string borninfo;

    auto ismear = -1;
    //auto ismear_4ph = 1; // default for guassian smearing
    auto epsilon = 10.0;
    //auto epsilon_4ph = 10.0;
    auto na_sigma = 0.1;

    //std::string interpolator = "linear";

    //double len_boundary = 0.0;

    // Assign given values

    assign_val(Tmin, "TMIN", general_var_dict);
    assign_val(Tmax, "TMAX", general_var_dict);
    assign_val(dT, "DT", general_var_dict);

    if (!general_var_dict["EMIN"].empty()) {
        auto emin = 0.0;
        assign_val(emin, "EMIN", general_var_dict);
        dos->emin = emin;
        dos->auto_set_emin = false;
    }
    if (!general_var_dict["EMAX"].empty()) {
        auto emax = 1000.0;
        assign_val(emax, "EMAX", general_var_dict);
        dos->emax = emax;
        dos->auto_set_emax = false;
    }

    assign_val(delta_e, "DELTA_E", general_var_dict);

    assign_val(nsym, "NSYM", general_var_dict);
    //assign_val(sym_time_reversal, "TREVSYM", general_var_dict);
    assign_val(tolerance, "TOLERANCE", general_var_dict);
    assign_val(printsymmetry, "PRINTSYM", general_var_dict);

    assign_val(nonanalytic, "NONANALYTIC", general_var_dict);
    // RESTART and RESTART_4PH now belong to the &kappa field; they are
    // still honored here for backward compatibility (the &kappa values win).
    if (!general_var_dict["RESTART"].empty() || !general_var_dict["RESTART_4PH"].empty()) {
        warn("parse_general_vars",
             "RESTART and RESTART_4PH in the &general field are deprecated;\n"
             " please move them to the &kappa field.");
    }
    assign_val(restart, "RESTART", general_var_dict);
    assign_val(restart_4ph, "RESTART_4PH", general_var_dict);

    assign_val(nbands, "NBANDS", general_var_dict);
    assign_val(borninfo, "BORNINFO", general_var_dict);

    assign_val(ismear, "ISMEAR", general_var_dict);
    assign_val(epsilon, "EPSILON", general_var_dict);
    assign_val(na_sigma, "NA_SIGMA", general_var_dict);

    // FC2_TEMPERATURE selects one temperature row of the temperature-
    // dependent FC2 stored in a PREFIX.scph.h5 / PREFIX.qha.h5 file given
    // as FCSFILE or FC2FILE (negative = disabled).
    double fc2_temperature = -1.0;
    assign_val(fc2_temperature, "FC2_TEMPERATURE", general_var_dict);

    auto allow_unconverged_in = false;
    assign_val(allow_unconverged_in, "ALLOW_UNCONVERGED", general_var_dict);
    this->allow_unconverged = allow_unconverged_in;
    assign_val(classical, "CLASSICAL", general_var_dict);
    assign_val(band_connection, "BCONNECT", general_var_dict);
    assign_val(use_triplet_symmetry, "TRISYM", general_var_dict);
    assign_val(bornsym, "BORNSYM", general_var_dict);
    assign_val(verbosity, "VERBOSITY", general_var_dict);


    if (band_connection > 2) {
        exit("parse_general_vars", "BCONNECT-tag can take 0, 1, or 2.");
    }

    if (nonanalytic == 3) {
        assign_val(prec_ewald, "PREC_EWALD", general_var_dict);
        if (prec_ewald <= 0.0 || prec_ewald >= 1.0) {
            exit("parse_general_vars", "PREC_EWALD should be a small positive value.");
        }
        ewald->is_longrange = true;
        ewald->file_longrange = boost::lexical_cast<std::string>(general_var_dict["BORNINFO"]);
        ewald->prec_ewald = prec_ewald;
        ewald->rate_ab = 6.0 / pi;

    } else {
        ewald->is_longrange = false;
    }

    if (nonanalytic > 3) {
        exit("parse_general_vars", "NONANALYTIC-tag can take 0, 1, 2, or 3.");
    }
    if (nonanalytic && borninfo == "") {
        exit("parse_general_vars", "BORNINFO must be specified when NONANALYTIC > 0.");
    }

    // Copy the values to appropriate classes.

    job_title = prefix;
    phon->mode = mode;

    conductivity->set_conductivity_params(file_result, file_result4, file_kappa_h5, restart, restart_4ph, use_h5_io);

    symmetry->nsym = nsym;
    symmetry->tolerance = tolerance;
    symmetry->printsymmetry = printsymmetry;
    //symmetry->time_reversal_sym = sym_time_reversal;

    system->Tmin = Tmin;
    system->Tmax = Tmax;
    system->dT = dT;

    if (!kdname_input.empty()) {
        system->symbol_kd = kdname_input;
    }
    if (!masskd_input.empty()) {
        system->mass_kd = masskd_input;
    }

    dos->delta_e = delta_e;

    dynamical->nonanalytic = nonanalytic;
    dynamical->na_sigma = na_sigma;
    dielec->symmetrize_borncharge = bornsym;
    writes->nbands = nbands;
    writes->setVerbosity(verbosity);
    dielec->file_born = borninfo;
    dynamical->band_connection = band_connection;
    integration->epsilon = epsilon;
    fcs_phonon->file_fcs = fcsfile;
    fcs_phonon->file_fc2 = fc2file;
    fcs_phonon->fc2_temperature = fc2_temperature;
    const auto dfc2file = general_var_dict["DFC2FILE"];
    if (!dfc2file.empty() && fc2_temperature < 0.0) {
        exit("parse_general_vars", "DFC2FILE requires FC2_TEMPERATURE to select the temperature.");
    }
    fcs_phonon->file_dfc2 = dfc2file;
    fcs_phonon->update_fc2 = !fc2file.empty();
    thermodynamics->classical = classical;
    integration->ismear = ismear;
    anharmonic_core->use_triplet_symmetry = use_triplet_symmetry;
    general_var_dict.clear();
}

void Input::parse_kappa_vars(const bool use_default_values)
{
    std::string str_tmp;
    const std::vector<std::string> input_list{"KMESH_COARSE",
                                              "EPSILON_4PH",
                                              "ISMEAR_4PH",
                                              "INCLUDE_4PH",
                                              "RESTART",
                                              "RESTART_4PH",
                                              "INTERPOLATOR",
                                              "LEN_BOUNDARY",
                                              "ISOTOPE",
                                              "ISOFACT",
                                              "KAPPA_COHERENT",
                                              "KAPPA_COHERENT_BLOCK",
                                              "KAPPA_SPEC",
                                              "WRITE_INTERPOL",
                                              "ADAPTIVE_FACTOR",
                                              "ITERATIVE",
                                              "MAX_CYCLE",
                                              "MIN_CYCLE",
                                              "ITER_THRESHOLD",
                                              "IBTE_MIXING"};

    std::vector<double> isotope_factor;
    std::map<std::string, std::string> kappa_var_dict;
    std::vector<std::string> isofact_v;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, kappa_var_dict);

    // Default values
    auto ismear_4ph = 1;
    auto epsilon_4ph = 10.0;
    std::string interpolator = "log-linear";
    double len_boundary = 0.0;
    // Isotope scattering is cheap to evaluate and physically present in
    // natural samples, so it is on by default; set ISOTOPE = 0 for
    // isotopically pure crystals or elements absent from the internal
    // natural-abundance database.
    auto include_isotope = 1;
    auto calc_coherent = 0;
    auto calc_coherent_block = 0;
    auto calculate_kappa_spec = 0;
    auto write_interpol = 0;

    double adaptive_factor = 1.0;

    auto iterative = false;
    int max_cycle = 20;
    int min_cycle = 5;
    auto iter_threshold = 0.02;
    auto iterative_mixing = 0.9;

    // -1 = tag not given; resolved below (legacy QUARTIC fallback).
    int include_4ph = -1;

    // Assign given values
    if (!use_default_values) {
        assign_val(include_4ph, "INCLUDE_4PH", kappa_var_dict);
        assign_val(ismear_4ph, "ISMEAR_4PH", kappa_var_dict);
        assign_val(epsilon_4ph, "EPSILON_4PH", kappa_var_dict);
        assign_val(interpolator, "INTERPOLATOR", kappa_var_dict);
        assign_val(write_interpol, "WRITE_INTERPOL", kappa_var_dict);
        assign_val(len_boundary, "LEN_BOUNDARY", kappa_var_dict);
        assign_val(calc_coherent, "KAPPA_COHERENT", kappa_var_dict);
        assign_val(calc_coherent_block, "KAPPA_COHERENT_BLOCK", kappa_var_dict);
        assign_val(include_isotope, "ISOTOPE", kappa_var_dict);
        assign_val(calculate_kappa_spec, "KAPPA_SPEC", kappa_var_dict);
        str_tmp = kappa_var_dict["KMESH_COARSE"];
        assign_val(adaptive_factor, "ADAPTIVE_FACTOR", kappa_var_dict);
        assign_val(iterative, "ITERATIVE", kappa_var_dict);
        assign_val(max_cycle, "MAX_CYCLE", kappa_var_dict);
        assign_val(min_cycle, "MIN_CYCLE", kappa_var_dict);
        assign_val(iterative_mixing, "IBTE_MIXING", kappa_var_dict);
        assign_val(iter_threshold, "ITER_THRESHOLD", kappa_var_dict);
    }

    // INCLUDE_4PH is the explicit switch for the four-phonon channel.
    // Legacy inputs enabled it implicitly via QUARTIC > 0 (an &analysis
    // tag), which is still honored with a deprecation warning.
    if (include_4ph < 0) {
        include_4ph = anharmonic_core->quartic_mode > 0 ? 1 : 0;
        if (include_4ph == 1) {
            warn("parse_kappa_vars",
                 "Enabling four-phonon scattering because QUARTIC > 0 is deprecated;\n"
                 " please set INCLUDE_4PH = 1 in the &kappa field instead.");
        }
    } else if (include_4ph > 0 && anharmonic_core->quartic_mode == 0) {
        // The quartic (FC4) machinery is required for the 4ph self-energies.
        anharmonic_core->quartic_mode = 1;
    }
    conductivity->fph_rta = include_4ph;

    // RESTART / RESTART_4PH given here override the (deprecated) &general
    // values and the file-existence auto-detection.
    if (!use_default_values) {
        if (!kappa_var_dict["RESTART"].empty()) {
            auto restart_in = false;
            assign_val(restart_in, "RESTART", kappa_var_dict);
            conductivity->set_restart_flag(3, restart_in);
        }
        if (!kappa_var_dict["RESTART_4PH"].empty()) {
            auto restart_in = false;
            assign_val(restart_in, "RESTART_4PH", kappa_var_dict);
            conductivity->set_restart_flag(4, restart_in);
        }
    }

    integration->epsilon_4ph = epsilon_4ph;
    integration->ismear_4ph = ismear_4ph;
    integration->adaptive_factor = adaptive_factor;
    conductivity->len_boundary = len_boundary; // m
    conductivity->calc_kappa_spec = calculate_kappa_spec;
    conductivity->calc_coherent = calc_coherent;
    conductivity->calc_coherent_block = calc_coherent_block;
    conductivity->write_interpolation = write_interpol;

    iterativebte->do_iterative = iterative;
    iterativebte->max_cycle = max_cycle;
    iterativebte->min_cycle = min_cycle;
    iterativebte->mixing_factor = iterative_mixing;
    iterativebte->convergence_criteria = iter_threshold; // wh

    // set 4ph mesh
    std::vector<unsigned int> kmesh_v;
    kmesh_v.clear();
    if (!str_tmp.empty()) {

        std::istringstream is(str_tmp);

        while (true) {
            str_tmp.clear();
            is >> str_tmp;
            if (str_tmp.empty()) {
                break;
            }
            kmesh_v.push_back(my_cast<unsigned int>(str_tmp));
        }

        if (kmesh_v.size() != 3) {
            exit("parse_kappa_vars", "The number of entries for KMESH_COARSE has to be 3.");
        }
    } else {
        kmesh_v.resize(3);
        for (auto i = 0; i < 3; ++i)
            kmesh_v[i] = 0;
    }

    boost::to_lower(interpolator);
    std::vector<std::string> supported_interpolator{"linear", "log-linear", "modified-log-linear"};
    if (std::find(std::begin(supported_interpolator), std::end(supported_interpolator), interpolator) ==
        std::end(supported_interpolator))
    {
        exit("parse_kappa_vars", "INTERPOLATOR is not supported.");
    }

    conductivity->set_interpolator(interpolator);
    conductivity->set_kmesh_coarse(&kmesh_v[0]);

    // set isotope
    if (include_isotope) {
        if (!kappa_var_dict["ISOFACT"].empty()) {
            split_str_by_space(kappa_var_dict["ISOFACT"], isofact_v);
            isotope_factor.resize(isofact_v.size());
            for (auto i = 0; i < isofact_v.size(); ++i) {
                isotope_factor[i] = my_cast<double>(isofact_v[i]);
            }
        }
    }

    if (include_isotope) {
        if (!isotope_factor.empty()) {
            isotope->isotope_factor = isotope_factor;
        }
    }

    isotope->include_isotope = include_isotope;

    kappa_var_dict.clear();
}


void Input::parse_scph_vars()
{
    // Read input parameters in the &scph-field.

    struct stat st
    {};
    const std::vector<std::string> input_list{"KMESH_SCPH",
                                              "KMESH_INTERPOLATE",
                                              "MIXALPHA",
                                              "MAXITER",
                                              "RESTART_SCPH",
                                              "IALGO",
                                              "IMIX",
                                              "SELF_OFFDIAG",
                                              "TOL_SCPH",
                                              "LOWER_TEMP",
                                              "WARMSTART",
                                              "BUBBLE",
                                              "RELAX_STR"};
    std::vector<std::string> no_defaults{"KMESH_SCPH", "KMESH_INTERPOLATE"};
    std::vector<int> kmesh_v, kmesh_interpolate_v;
    std::map<std::string, std::string> scph_var_dict;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, scph_var_dict);

    for (auto &no_default: no_defaults) {
        if (scph_var_dict.find(no_default) == scph_var_dict.end()) {
            exit("parse_scph_vars", "The following variable is not found in &scph input region: ", no_default.c_str());
        }
    }

    // restart mode will be automatically turned on for SCPH calculations.
    auto file_dymat = this->job_title + ".scph_dymat";
    bool restart_scph = false;
    restart_scph = stat(file_dymat.c_str(), &st) == 0;

    // Default values
    auto tolerance_scph = 1.0e-10;
    unsigned int maxiter = 1000;
    auto mixalpha = 0.1;
    auto selfenergy_offdiagonal = true;
    unsigned int ialgo_scph = 0;
    unsigned int imix_scph = 1; // DIIS mixing by default; IMIX = 0 restores simple mixing
    auto lower_temp = true;
    auto warm_start = true;
    unsigned int bubble = 0;
    int relax_str = 0;

    // Assign given values

    assign_val(maxiter, "MAXITER", scph_var_dict);
    assign_val(mixalpha, "MIXALPHA", scph_var_dict);
    assign_val(selfenergy_offdiagonal, "SELF_OFFDIAG", scph_var_dict);
    assign_val(ialgo_scph, "IALGO", scph_var_dict);
    assign_val(imix_scph, "IMIX", scph_var_dict);
    if (imix_scph > 1) {
        exit("parse_scph_vars", "IMIX must be 0 (simple mixing) or 1 (DIIS mixing).");
    }
    if (imix_scph == 1 && (mixalpha <= 0.0 || mixalpha > 1.0)) {
        exit("parse_scph_vars", "MIXALPHA must be in (0, 1] when IMIX = 1.");
    }
    assign_val(tolerance_scph, "TOL_SCPH", scph_var_dict);
    assign_val(lower_temp, "LOWER_TEMP", scph_var_dict);
    assign_val(warm_start, "WARMSTART", scph_var_dict);
    assign_val(bubble, "BUBBLE", scph_var_dict);
    assign_val(relax_str, "RELAX_STR", scph_var_dict);
    if (!is_valid_relaxation_str_mode(relax_str)) {
        exit("parse_scph_vars", "RELAX_STR must be 0, 1, 2, or 3.");
    }
    if (relax_str != to_int(RelaxationStrMode::None) && !selfenergy_offdiagonal) {
        exit("parse_scph_vars", "SELF_OFFDIAG = 0 cannot be used when RELAX_STR != 0.");
    }

    if (relax_str != to_int(RelaxationStrMode::None)) {
        auto file_harm_dymat = this->job_title + ".renorm_harm_dymat";
        auto file_v0 = this->job_title + ".V0";
        restart_scph = restart_scph && (stat(file_harm_dymat.c_str(), &st) == 0) && (stat(file_v0.c_str(), &st) == 0);
    }

    // The unified state file replaces the legacy text trio; either enables
    // the restart attempt (completeness is validated at load time).
    if (use_hdf5_io) {
        auto file_scph_h5 = this->job_title + ".scph.h5";
        if (stat(file_scph_h5.c_str(), &st) == 0) restart_scph = true;
    }

    assign_val(restart_scph, "RESTART_SCPH", scph_var_dict);

    auto str_tmp = scph_var_dict["KMESH_SCPH"];

    kmesh_v.clear();
    if (!str_tmp.empty()) {

        std::istringstream is(str_tmp);

        while (true) {
            str_tmp.clear();
            is >> str_tmp;
            if (str_tmp.empty()) {
                break;
            }
            kmesh_v.push_back(my_cast<unsigned int>(str_tmp));
        }

        if (kmesh_v.size() != 3) {
            exit("parse_scph_vars", "The number of entries for KMESH_SCPH has to be 3.");
        }
    } else {
        exit("parse_scph_vars", "Please specify KMESH_SCPH for mode = SCPH");
    }

    str_tmp = scph_var_dict["KMESH_INTERPOLATE"];
    if (!str_tmp.empty()) {

        std::istringstream is(str_tmp);

        while (true) {
            str_tmp.clear();
            is >> str_tmp;
            if (str_tmp.empty()) {
                break;
            }
            kmesh_interpolate_v.push_back(my_cast<unsigned int>(str_tmp));
        }

        if (kmesh_interpolate_v.size() != 3) {
            exit("parse_scph_vars", "The number of entries for KMESH_INTERPOLATE has to be 3.");
        }
    } else {
        exit("parse_scph_vars", "Please specify KMESH_INTERPOLATE for mode = SCPH");
    }

    // Copy the values to appropriate classes.

    for (auto i = 0; i < 3; ++i) {
        scph->kmesh_scph[i] = kmesh_v[i];
        scph->kmesh_interpolate[i] = kmesh_interpolate_v[i];
    }
    scph->mixalpha = mixalpha;
    scph->imix_scph = imix_scph;
    scph->maxiter = maxiter;
    scph->restart_scph = restart_scph;
    scph->use_h5_io = use_hdf5_io;
    scph->selfenergy_offdiagonal = selfenergy_offdiagonal;
    scph->ialgo = ialgo_scph;
    scph->tolerance_scph = tolerance_scph;
    scph->lower_temp = lower_temp;
    scph->warmstart_scph = warm_start;
    scph->bubble = bubble;
    relaxation->relax_str = relax_str;

    kmesh_v.clear();
    kmesh_interpolate_v.clear();

    scph_var_dict.clear();
}

void Input::parse_qha_vars()
{
    // Read input parameters in the &qha-field.

    struct stat st
    {};
    const std::vector<std::string> input_list{"KMESH_QHA",
                                              "KMESH_INTERPOLATE",
                                              "LOWER_TEMP",
                                              "RELAX_STR",
                                              "QHA_SCHEME",
                                              "IALGO",
                                              "SELF_OFFDIAG",
                                              "RESTART_QHA"};
    std::vector<std::string> no_defaults{"KMESH_QHA", "KMESH_INTERPOLATE"};
    std::vector<int> kmesh_v, kmesh_interpolate_v;

    std::map<std::string, std::string> qha_var_dict;

    get_var_dict(input_list, qha_var_dict);

    for (auto &no_default: no_defaults) {
        if (qha_var_dict.find(no_default) == qha_var_dict.end()) {
            exit("parse_qha_vars", "The following variable is not found in &qha input region: ", no_default.c_str());
        }
    }

    auto lower_temp = true;
    int relax_str = 1;
    int qha_scheme = to_int(QhaScheme::Standard);
    auto selfenergy_offdiagonal = true;
    unsigned int ialgo_scph = 0;

    assign_val(lower_temp, "LOWER_TEMP", qha_var_dict);
    assign_val(relax_str, "RELAX_STR", qha_var_dict);
    assign_val(qha_scheme, "QHA_SCHEME", qha_var_dict);
    assign_val(selfenergy_offdiagonal, "SELF_OFFDIAG", qha_var_dict);
    assign_val(ialgo_scph, "IALGO", qha_var_dict);

    if (!is_valid_relaxation_str_mode(relax_str)) {
        exit("parse_qha_vars", "RELAX_STR must be 1, 2, or 3 when mode = QHA.");
    }
    if (relax_str == to_int(RelaxationStrMode::None)) {
        exit("parse_qha_vars", "RELAX_STR = 0 is not supported when mode = QHA.");
    }
    if (!is_valid_qha_scheme(qha_scheme)) {
        exit("parse_qha_vars", "QHA_SCHEME must be 0, 1, or 2.");
    }
    if (relax_str != to_int(RelaxationStrMode::None) && !selfenergy_offdiagonal) {
        exit("parse_qha_vars", "SELF_OFFDIAG = 0 cannot be used when RELAX_STR != 0.");
    }

    auto str_tmp = qha_var_dict["KMESH_QHA"];

    if (!str_tmp.empty()) {

        std::istringstream is(str_tmp);

        while (true) {
            str_tmp.clear();
            is >> str_tmp;
            if (str_tmp.empty()) {
                break;
            }
            kmesh_v.push_back(my_cast<unsigned int>(str_tmp));
        }

        if (kmesh_v.size() != 3) {
            exit("parse_qha_vars", "The number of entries for KMESH_QHA has to be 3.");
        }
    } else {
        exit("parse_qha_vars", "Please specify KMESH_QHA for mode = QHA");
    }

    str_tmp = qha_var_dict["KMESH_INTERPOLATE"];
    if (!str_tmp.empty()) {

        std::istringstream is(str_tmp);

        while (true) {
            str_tmp.clear();
            is >> str_tmp;
            if (str_tmp.empty()) {
                break;
            }
            kmesh_interpolate_v.push_back(my_cast<unsigned int>(str_tmp));
        }

        if (kmesh_interpolate_v.size() != 3) {
            exit("parse_qha_vars", "The number of entries for KMESH_INTERPOLATE has to be 3.");
        }
    } else {
        exit("parse_qha_vars", "Please specify KMESH_INTERPOLATE for mode = QHA");
    }

    // Copy the values to appropriate classes.

    for (auto i = 0; i < 3; ++i) {
        qha->kmesh_qha[i] = kmesh_v[i];
        qha->kmesh_interpolate[i] = kmesh_interpolate_v[i];
    }
    qha->lower_temp = lower_temp;
    relaxation->relax_str = relax_str;
    qha->qha_scheme = to_qha_scheme(qha_scheme);
    qha->selfenergy_offdiagonal = selfenergy_offdiagonal;
    qha->ialgo = ialgo_scph;

    // Set other values

    auto file_renorm_harm_dymat = this->job_title + ".renorm_harm_dymat";
    auto file_v0 = this->job_title + ".V0";
    auto restart_qha = (stat(file_renorm_harm_dymat.c_str(), &st) == 0) && (stat(file_v0.c_str(), &st) == 0);

    if (use_hdf5_io) {
        auto file_qha_h5 = this->job_title + ".qha.h5";
        if (stat(file_qha_h5.c_str(), &st) == 0) restart_qha = true;
    }

    assign_val(restart_qha, "RESTART_QHA", qha_var_dict);

    qha->restart_qha = restart_qha;
    qha->use_h5_io = use_hdf5_io;

    kmesh_v.clear();
    kmesh_interpolate_v.clear();

    qha_var_dict.clear();
}

void Input::parse_relax_vars()
{
    // Read input parameters in the &relax-field.

    const std::vector<std::string> input_list{"RELAX_ALGO",
                                              "MAX_STR_ITER",
                                              "COORD_CONV_TOL",
                                              "GRADIENT_CONV_TOL",
                                              "CELL_GRADIENT_CONV_TOL",
                                              "GDIIS_CONTROL",
                                              "GDIIS_PLAIN",
                                              "MIXBETA_COORD",
                                              "ALPHA_STDECENT",
                                              "CELL_CONV_TOL",
                                              "MIXBETA_CELL",
                                              "SET_INIT_STR",
                                              "COOLING_U0_INDEX",
                                              "COOLING_U0_THR",
                                              "ADD_HESS_DIAG",
                                              "STAT_PRESSURE",
                                              "RENORM_3TO2ND",
                                              "RENORM_2TO1ST",
                                              "RENORM_34TO1ST",
                                              "STRAIN_IFC_DIR"};

    std::map<std::string, std::string> stropt_var_dict;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, stropt_var_dict);

    // used to determine restart options
    auto file_dymat = this->job_title + ".scph_dymat";
    auto file_harm_dymat = this->job_title + ".renorm_harm_dymat";
    auto file_v0 = this->job_title + ".V0";

    int relax_algo = 2;
    int max_str_iter = 100;
    double coord_conv_tol = 1.0e-5;
    double mixbeta_coord = 0.5;
    double alpha_steepest_decent = 1.0e4;
    double gradient_conv_tol = 0.0;
    double cell_gradient_conv_tol = 0.0;
    // The Farkas-Schlegel controlled GDIIS is the default for RELAX_ALGO = 3.
    // GDIIS_PLAIN = 1 switches it off (regular GDIIS without the step-acceptance
    // tests). GDIIS_CONTROL is kept as a deprecated alias of the old enabling
    // switch; an explicit GDIIS_PLAIN takes precedence over it.
    int gdiis_control = 1;
    int gdiis_plain = 0;

    double cell_conv_tol = 1.0e-5;
    double mixbeta_cell = 0.5;

    int set_init_str = 1;
    int cooling_u0_index = 0;
    double cooling_u0_thr = 0.001;

    double add_hess_diag = 100.0; // [cm^{-1}]
    double stat_pressure = 0.0;   // [GPa]

    int renorm_3to2nd = 2;
    int renorm_2to1st = 2;
    int renorm_34to1st = 0;

    std::string strain_IFC_dir;

    assign_val(relax_algo, "RELAX_ALGO", stropt_var_dict);
    assign_val(max_str_iter, "MAX_STR_ITER", stropt_var_dict);
    assign_val(coord_conv_tol, "COORD_CONV_TOL", stropt_var_dict);
    assign_val(gradient_conv_tol, "GRADIENT_CONV_TOL", stropt_var_dict);
    assign_val(cell_gradient_conv_tol, "CELL_GRADIENT_CONV_TOL", stropt_var_dict);
    assign_val(gdiis_control, "GDIIS_CONTROL", stropt_var_dict); // deprecated
    assign_val(gdiis_plain, "GDIIS_PLAIN", stropt_var_dict);
    if (stropt_var_dict.find("GDIIS_CONTROL") != stropt_var_dict.end()) {
        warn("parse_relax_vars",
             "GDIIS_CONTROL is deprecated: controlled GDIIS is now the default. Use GDIIS_PLAIN = 1 to disable it.");
    }
    if (stropt_var_dict.find("GDIIS_PLAIN") != stropt_var_dict.end()) {
        gdiis_control = gdiis_plain ? 0 : 1;
    }

    if (relax_algo < 1 || relax_algo > 3) {
        exit("parse_relax_vars", "RELAX_ALGO must be 1 (steepest decent), 2 (Newton), or 3 (BFGS+GDIIS).");
    }

    if (relax_algo == 1) {
        assign_val(alpha_steepest_decent, "ALPHA_STDECENT", stropt_var_dict);
    } else if (relax_algo == 2) {
        assign_val(mixbeta_coord, "MIXBETA_COORD", stropt_var_dict);
    }

    assign_val(cell_conv_tol, "CELL_CONV_TOL", stropt_var_dict);
    if (relax_algo == 2) {
        assign_val(mixbeta_cell, "MIXBETA_CELL", stropt_var_dict);
    }

    assign_val(set_init_str, "SET_INIT_STR", stropt_var_dict);
    assign_val(cooling_u0_index, "COOLING_U0_INDEX", stropt_var_dict);
    assign_val(cooling_u0_thr, "COOLING_U0_THR", stropt_var_dict);
    assign_val(add_hess_diag, "ADD_HESS_DIAG", stropt_var_dict);
    assign_val(stat_pressure, "STAT_PRESSURE", stropt_var_dict);

    assign_val(renorm_3to2nd, "RENORM_3TO2ND", stropt_var_dict);
    assign_val(renorm_2to1st, "RENORM_2TO1ST", stropt_var_dict);
    assign_val(renorm_34to1st, "RENORM_34TO1ST", stropt_var_dict);

    assign_val(strain_IFC_dir, "STRAIN_IFC_DIR", stropt_var_dict);
    if (!strain_IFC_dir.empty() && strain_IFC_dir.at(strain_IFC_dir.length() - 1) != '/') {
        strain_IFC_dir = strain_IFC_dir + "/";
    }

    //    bool restart_scph = (stat(file_dymat.c_str(), &st) == 0) && (stat(file_harm_dymat.c_str(), &st) == 0);
    //    restart_scph = restart_scph & (stat(file_v0.c_str(), &st) == 0);

    relaxation->relax_algo = relax_algo;
    relaxation->max_str_iter = max_str_iter;

    relaxation->coord_conv_tol = coord_conv_tol;
    relaxation->gradient_conv_tol = gradient_conv_tol;
    relaxation->cell_gradient_conv_tol = cell_gradient_conv_tol;
    relaxation->gdiis_control = gdiis_control;
    relaxation->mixbeta_coord = mixbeta_coord;
    relaxation->alpha_steepest_decent = alpha_steepest_decent;

    relaxation->cell_conv_tol = cell_conv_tol;
    relaxation->mixbeta_cell = mixbeta_cell;

    relaxation->set_init_str = set_init_str;
    relaxation->cooling_u0_index = cooling_u0_index;
    relaxation->cooling_u0_thr = cooling_u0_thr;

    relaxation->add_hess_diag = add_hess_diag;
    relaxation->stat_pressure = stat_pressure;

    relaxation->renorm_3to2nd = renorm_3to2nd;
    relaxation->renorm_2to1st = renorm_2to1st;
    relaxation->renorm_34to1st = renorm_34to1st;

    relaxation->strain_IFC_dir = strain_IFC_dir;

    stropt_var_dict.clear();
}

void Input::check_relax_vars()
{

    std::fstream fin_test;

    // structural optimization
    if (relaxation->relax_str != 0) {

        if (thermodynamics->calc_FE_bubble) {
            exit("check_relax_vars", "Sorry, RELAX_STR!=0 can't be used with bubble correction of the free energy.");
        }
        if (scph->bubble > 0) {
            exit("check_relax_vars",
                 "Sorry, RELAX_STR!=0 can't be used with bubble self-energy on top of the SCPH calculation.");
        }

        // relax the shape of the unit cell
        if (relaxation->relax_str == 2 || relaxation->relax_str == 3) {
            // strain-force coupling
            if (relaxation->renorm_2to1st == 2) {
                fin_test.open(relaxation->strain_IFC_dir + "strain_force.in");

                if (!fin_test) {
                    exit("check_relax_vars", "strain_force.in is required in STRAIN_IFC_DIR when RENORM_2TO1ST = 2.");
                }
                fin_test.close();
            }

            // strain-IFC coupling
            if (relaxation->renorm_3to2nd == 2 || relaxation->renorm_3to2nd == 3) {
                fin_test.open(relaxation->strain_IFC_dir + "strain_harmonic.in");

                if (!fin_test) {
                    exit("check_relax_vars",
                         "strain_harmonic.in is required in STRAIN_IFC_DIR when RENORM_3TO2ND >= 2.");
                }

                fin_test.close();
            }
        }
    }
}


void Input::parse_initial_strain()
{
    int i, j;
    double u_tensor_tmp[3][3];
    std::string line;
    std::string line_wo_comment, line_tmp;
    std::vector<std::string> line_vec, line_split;
    std::string::size_type pos_first_comment_tag;

    line_vec.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }

    } else {
        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }
    }

    if (line_vec.size() != 3) {
        exit("parse_initial_strain", "Too few or too much lines for the &strain field.\n \
                                            The number of valid lines for the &cell field should be 3.");
    }

    for (i = 0; i < 3; ++i) {

        line = line_vec[i];
        split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);


        // u_tensor
        if (line_split.size() == 3) {
            for (j = 0; j < 3; ++j) {
                u_tensor_tmp[i][j] = boost::lexical_cast<double>(line_split[j]);
            }
        } else {
            exit("parse_initial_strain", "Unacceptable format for &strain field.");
        }
    }
    relaxation->setInitialDistortion(u_tensor_tmp);
}

void Input::parse_initial_displace()
{

    int i, j;
    int itmp;
    int ixyz;
    int iat;
    std::string line;
    std::string line_wo_comment, line_tmp;
    std::vector<std::string> line_vec, line_split;
    std::string::size_type pos_first_comment_tag;

    int input_mode{-1};

    double unit;
    double a[3][3];
    std::vector<std::vector<double>> u_fractional, u_xyz;
    std::vector<double> vec_tmp(3);

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }

    } else {
        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }
    }

    if (line_vec.empty()) {
        exit("parse_initial_displace", "Too few lines for the &displace field.");
    }

    line = line_vec[0];
    split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

    if (line_split.size() == 1) {
        input_mode = boost::lexical_cast<int>(line_split[0]);
    } else {
        exit("parse_initial_displace", "Unacceptable format for &displace field.");
    }

    if (input_mode < 0 || input_mode >= 2) {
        exit("parse_initial_displace", "Invalid value of input_mode");
    }

    // read displacements
    u_fractional.clear();
    u_xyz.clear();

    if (input_mode == 0) {
        // check the number of input lines later because
        // system->natmin has not been set at this stage.
        // if (line_vec.size() != 5 + natmin) {
        //     exit("parse_initial_displace",
        //          "Too few or too many lines for the &displace field.");
        // }

        // read cell information
        for (i = 1; i < 5; ++i) {

            line = line_vec[i];
            split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

            if (i == 1) {
                // read unit
                if (line_split.size() == 1) {
                    unit = boost::lexical_cast<double>(line_split[0]);
                } else {
                    exit("parse_initial_displace", "Unacceptable format for &displace field.");
                }

            } else {
                // Lattice vectors a1, a2, a3
                if (line_split.size() == 3) {
                    for (j = 0; j < 3; ++j) {
                        a[i - 2][j] = boost::lexical_cast<double>(line_split[j]);
                    }
                } else {
                    exit("parse_initial_displace", "Unacceptable format for &displace field.");
                }
            }
        }

        for (itmp = 0; itmp < 3; itmp++) {
            for (ixyz = 0; ixyz < 3; ixyz++) {
                a[itmp][ixyz] *= unit;
            }
        }

        // read fractional coordinate
        for (i = 5; i < line_vec.size(); i++) {

            line = line_vec[i];
            split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

            if (line_split.size() == 3) {
                for (j = 0; j < 3; ++j) {
                    vec_tmp[j] = boost::lexical_cast<double>(line_split[j]);
                }
                u_fractional.push_back(vec_tmp);
            } else {
                exit("parse_initial_displace", "Unacceptable format for &displace field.");
            }
        }

        // transform to xyz coordinate
        for (iat = 0; iat < u_fractional.size(); iat++) {
            for (ixyz = 0; ixyz < 3; ixyz++) {
                vec_tmp[ixyz] = 0.0;
                for (itmp = 0; itmp < 3; itmp++) {
                    vec_tmp[ixyz] += a[itmp][ixyz] * u_fractional[iat][itmp];
                }
            }
            u_xyz.push_back(vec_tmp);
        }

    } else if (input_mode == 1) {

        // check the number of input lines later because
        // system->natmin has not been set at this stage.

        // if (line_vec.size() != 1 + natmin) {
        //     exit("parse_initial_displace",
        //          "Too few or too many lines for the &displace field.");
        // }

        for (i = 1; i < line_vec.size(); i++) {

            line = line_vec[i];
            split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

            if (line_split.size() == 3) {
                for (j = 0; j < 3; ++j) {
                    vec_tmp[j] = boost::lexical_cast<double>(line_split[j]);
                }
                u_xyz.push_back(vec_tmp);
            } else {
                exit("parse_cell_parameter", "Unacceptable format for &displace field.");
            }
        }
    }

    // Copy the values to appropriate classes
    relaxation->init_u0.clear();
    for (iat = 0; iat < u_xyz.size(); iat++) {
        for (ixyz = 0; ixyz < 3; ixyz++) {
            relaxation->init_u0.push_back(u_xyz[iat][ixyz]);
        }
    }
}

void Input::parse_analysis_vars(const bool use_default_values)
{
    // Read input parameters in the &analysis field.
    int i;

    std::vector<std::string> input_list{"PRINTEVAL",
                                        "PRINTEVEC",
                                        "PRINTXSF",
                                        "PRINTVEL",
                                        "QUARTIC",
                                        "KS_INPUT",
                                        "REALPART",
                                        "FSTATE_W",
                                        "FSTATE_K",
                                        "PRINTMSD",
                                        "DOS",
                                        "PDOS",
                                        "TDOS",
                                        "GRUNEISEN",
                                        "NEWFCS",
                                        "DELTA_A",
                                        "ANIME",
                                        "ANIME_CELLSIZE",
                                        "ANIME_FORMAT",
                                        "ANIME_FRAMES",
                                        "SPS",
                                        "PRINTV3",
                                        "PRINTPR",
                                        "FC2_EWALD",
                                        "SELF_W",
                                        "UCORR",
                                        "SHIFT_UCORR",
                                        "DIELEC",
                                        "SELF_ENERGY",
                                        "PRINTV4",
                                        "ZMODE",
                                        "PROJECTION_AXES",
                                        "LONGITUDINAL_DOS",
                                        "FE_BUBBLE"};

    unsigned int cellsize[3] = {1, 1, 1};
    int shift_ucorr[3] = {0, 0, 0};
    double anime_kpoint_double[3] = {0., 0., 0.};

    std::vector<double> isotope_factor;
    std::string ks_input, anime_format;
    std::map<std::string, std::string> analysis_var_dict;
    std::vector<std::string> anime_kpoint, anime_cellsize;
    std::vector<std::string> projection_axes;
    std::vector<std::string> isofact_v;

    // Default values

    auto print_xsf = false;
    auto print_anime = false;

    auto print_vel = false;
    auto print_eval = false;
    auto print_evec = false;
    auto print_msd = false;
    auto print_ucorr = false;
    auto anime_frames = 20;

    bool compute_dos = true;
    bool projected_dos = false;
    bool two_phonon_dos = false;
    bool longitudinal_dos = false;
    int scattering_phase_space = 0;
    bool print_gruneisen = false;
    bool print_newfcs = false;
    int print_V3 = 0;
    int print_V4 = 0;
    bool participation_ratio = false;

    auto delta_a = 0.001;

    auto quartic_mode = 0;
    auto calc_realpart = false;
    auto fstate_omega = false;
    auto fstate_k = false;
    auto bubble_omega = false;

    auto print_fc2_ewald = false;
    auto print_self_consistent_fc2 = false;
    auto calc_FE_bubble = false;

    auto calculate_dielectric_constant = 0;
    auto calc_selfenergy = 0;
    auto print_zmode = false;

    auto do_projection = false;

    // Assign values to variables

    if (!use_default_values) {
        get_var_dict(input_list, analysis_var_dict);

        assign_val(print_vel, "PRINTVEL", analysis_var_dict);
        assign_val(print_evec, "PRINTEVEC", analysis_var_dict);
        assign_val(print_eval, "PRINTEVAL", analysis_var_dict);
        assign_val(print_msd, "PRINTMSD", analysis_var_dict);
        assign_val(print_ucorr, "UCORR", analysis_var_dict);

        assign_val(compute_dos, "DOS", analysis_var_dict);
        assign_val(projected_dos, "PDOS", analysis_var_dict);
        assign_val(two_phonon_dos, "TDOS", analysis_var_dict);
        assign_val(longitudinal_dos, "LONGITUDINAL_DOS", analysis_var_dict);

        assign_val(scattering_phase_space, "SPS", analysis_var_dict);
        assign_val(print_gruneisen, "GRUNEISEN", analysis_var_dict);
        assign_val(print_newfcs, "NEWFCS", analysis_var_dict);
        assign_val(delta_a, "DELTA_A", analysis_var_dict);

        assign_val(quartic_mode, "QUARTIC", analysis_var_dict);
        assign_val(calc_realpart, "REALPART", analysis_var_dict);
        assign_val(fstate_omega, "FSTATE_W", analysis_var_dict);
        assign_val(fstate_k, "FSTATE_K", analysis_var_dict);
        assign_val(ks_input, "KS_INPUT", analysis_var_dict);
        assign_val(bubble_omega, "SELF_W", analysis_var_dict);
        assign_val(calc_selfenergy, "SELF_ENERGY", analysis_var_dict);

        assign_val(print_xsf, "PRINTXSF", analysis_var_dict);
        assign_val(print_V3, "PRINTV3", analysis_var_dict);
        assign_val(print_V4, "PRINTV4", analysis_var_dict);
        assign_val(participation_ratio, "PRINTPR", analysis_var_dict);
        assign_val(print_fc2_ewald, "FC2_EWALD", analysis_var_dict);
        assign_val(calculate_dielectric_constant, "DIELEC", analysis_var_dict);
        assign_val(print_zmode, "ZMODE", analysis_var_dict);
        assign_val(calc_FE_bubble, "FE_BUBBLE", analysis_var_dict);

        if (analysis_var_dict.find("ANIME") == analysis_var_dict.end()) {
            print_anime = false;
        } else {
            print_anime = true;
        }

        if (analysis_var_dict.find("PROJECTION_AXES") != analysis_var_dict.end()) {
            do_projection = true;
        }
    }

    if (print_anime) {
        split_str_by_space(analysis_var_dict["ANIME"], anime_kpoint);

        if (anime_kpoint.size() != 3) {
            exit("parse_analysis_vars", "The number of entries for ANIME should be 3.");
        }
        for (i = 0; i < 3; ++i) {
            anime_kpoint_double[i] = my_cast<double>(anime_kpoint[i]);
        }

        split_str_by_space(analysis_var_dict["ANIME_CELLSIZE"], anime_cellsize);

        if (anime_cellsize.size() != 3) {
            exit("parse_analysis_vars", "The number of entries for ANIME_CELLSIZE should be 3.");
        }

        for (i = 0; i < 3; ++i) {
            try {
                cellsize[i] = boost::lexical_cast<unsigned int>(anime_cellsize[i]);
            } catch (std::exception &e) {
                std::cout << e.what() << '\n';
                exit("parse_analysis_vars", "ANIME_CELLSIZE must be a set of positive integers.");
            }
            if (cellsize[i] < 1) {
                exit("parse_analysis_vars", "Please give positive integers for ANIME_CELLSIZE.");
            }
        }

        assign_val(anime_format, "ANIME_FORMAT", analysis_var_dict);
        std::transform(anime_format.begin(), anime_format.end(), anime_format.begin(), toupper);

        if (anime_format.empty()) anime_format = "XYZ";

        if (anime_format != "XSF" && anime_format != "AXSF" && anime_format != "XYZ") {
            exit("parse_analysis_vars", "Invalid ANIME_FORMAT");
        }

        assign_val(anime_frames, "ANIME_FRAMES", analysis_var_dict);
    }

    if (print_ucorr) {
        std::string str_shift_ucorr;
        std::vector<std::string> list_shift_ucorr;
        assign_val(str_shift_ucorr, "SHIFT_UCORR", analysis_var_dict);

        if (!str_shift_ucorr.empty()) {
            split_str_by_space(str_shift_ucorr, list_shift_ucorr);
            if (list_shift_ucorr.size() != 3) {
                exit("parse_analysis_vars", "The number of entries for SHIFT_UCORR must be 3.");
            }

            for (i = 0; i < 3; ++i) {
                try {
                    shift_ucorr[i] = boost::lexical_cast<int>(list_shift_ucorr[i]);
                } catch (std::exception &e) {
                    std::cout << e.what() << '\n';
                    exit("parse_analysis_vars", "SHIFT_UCORR must be an array of integers.");
                }
            }
        }
    }

    if (do_projection) {
        std::string str_projection_axes;
        assign_val(str_projection_axes, "PROJECTION_AXES", analysis_var_dict);
        std::vector<double> direction(3);
        std::vector<std::vector<double>> projection_directions;
        if (!str_projection_axes.empty()) {
            std::vector<std::string> str_projection_each, str_vec;
            boost::split(str_projection_each, str_projection_axes, boost::is_any_of(","));

            if (str_projection_each.size() > 2) {
                warn("parse_analysis_vars", "Too many entries for PROJECTION_AXES. Only the first two will be used.");
            }

            for (i = 0; i < str_projection_each.size(); ++i) {
                split_str_by_space(str_projection_each[i], str_vec);
                if (str_vec.size() != 3) {
                    exit("parse_analysis_vars", "The number of entries for each vector in PROJECTION_AXES must be 3.");
                }
                for (auto j = 0; j < 3; ++j) {
                    try {
                        direction[j] = boost::lexical_cast<double>(str_vec[j]);
                    } catch (std::exception &e) {
                        std::cout << e.what() << '\n';
                        exit("parse_analysis_vars", "subset of PROJECTION_AXES must be an array of doubles.");
                    }
                }
                projection_directions.push_back(direction);
            }
        }

        dynamical->set_projection_directions(projection_directions);
    }

    // Copy the values to appropriate classes

    phonon_velocity->print_velocity = print_vel;
    dynamical->print_eigenvectors = print_evec;
    dynamical->participation_ratio = participation_ratio;

    writes->setWriteOptions(print_msd,
                            print_xsf,
                            print_anime,
                            anime_format,
                            anime_frames,
                            cellsize,
                            anime_kpoint_double,
                            print_ucorr,
                            shift_ucorr,
                            print_zmode,
                            print_eval);


    dos->compute_dos = compute_dos;
    dos->projected_dos = projected_dos;
    dos->two_phonon_dos = two_phonon_dos;
    dos->scattering_phase_space = scattering_phase_space;
    dos->longitudinal_projected_dos = longitudinal_dos;

    anharmonic_core->quartic_mode = quartic_mode;
    dielec->calc_dielectric_constant = calculate_dielectric_constant;

    mode_analysis->ks_input = ks_input;
    mode_analysis->calc_realpart = calc_realpart;
    mode_analysis->calc_fstate_omega = fstate_omega;
    mode_analysis->calc_fstate_k = fstate_k;
    mode_analysis->print_V3 = print_V3;
    mode_analysis->print_V4 = print_V4;
    mode_analysis->spectral_func = bubble_omega;
    mode_analysis->calc_selfenergy = calc_selfenergy;

    gruneisen->print_gruneisen = print_gruneisen;
    gruneisen->print_newfcs = print_newfcs;
    gruneisen->delta_a = delta_a;
    thermodynamics->calc_FE_bubble = calc_FE_bubble;

    ewald->print_fc2_ewald = print_fc2_ewald;

    if (phon->mode == "SCPH") {
        scph->print_self_consistent_fc2 = print_self_consistent_fc2;
    }

    analysis_var_dict.clear();
}

void Input::parse_cell_parameter()
{
    // Read the cell parameter

    int i, j;
    double a;
    double lavec_tmp[3][3];
    std::string line;
    std::string line_wo_comment, line_tmp;
    std::vector<std::string> line_vec, line_split;
    std::string::size_type pos_first_comment_tag;

    line_vec.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }

    } else {
        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }
    }

    if (line_vec.size() != 4) {
        exit("parse_cell_parameter",
             "Too few or too much lines for the &cell field.\n "
             "The number of valid lines for the &cell field should be 4.");
    }

    for (i = 0; i < 4; ++i) {

        line = line_vec[i];
        split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

        if (i == 0) {
            // Lattice factor a
            if (line_split.size() == 1) {
                a = boost::lexical_cast<double>(line_split[0]);
            } else {
                exit("parse_cell_parameter", "Unacceptable format for &cell field.");
            }

        } else {
            // Lattice vectors a1, a2, a3
            if (line_split.size() == 3) {
                for (j = 0; j < 3; ++j) {
                    lavec_tmp[j][i - 1] = boost::lexical_cast<double>(line_split[j]);
                }
            } else {
                exit("parse_cell_parameter", "Unacceptable format for &cell field.");
            }
        }
    }

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            system->lavec_p_input(i, j) = a * lavec_tmp[i][j];
        }
    }
    system->load_primitive_from_file = 1;
}

void Input::parse_kpoints()
{
    // Read the settings in the &kpoint field.

    int kpmode;
    std::string line, line_wo_comment, str_tmp;
    std::vector<std::string> kpelem, line_vec;
    std::string::size_type pos_first_comment_tag;

    line_vec.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }

    } else {
        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }

            trim_if(line_wo_comment, boost::is_any_of("\t\r\n "));

            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

            line_vec.push_back(line_wo_comment);
        }
    }

    for (int i = 0; i < line_vec.size(); ++i) {
        line = line_vec[i];
        split(kpelem, line, boost::is_any_of("\t "), boost::token_compress_on);

        if (i == 0) {
            // kpmode
            if (kpelem.size() == 1) {
                try {
                    kpmode = boost::lexical_cast<int>(kpelem[0]);
                } catch (std::exception &e) {
                    std::cout << e.what() << '\n';
                    exit("parse_kpoints", "KPMODE must be an integer. [0, 1, or 2]");
                }

                if (!(kpmode >= 0 && kpmode <= 3)) {
                    exit("parse_kpoints", "KPMODE must be 0, 1, or 2.");
                }

            } else {
                exit("parse_kpoints", "Unacceptable format for the &kpoint field.");
            }

        } else {
            // Read each entry of kpoint

            if (kpmode == 0 && kpelem.size() != 3) {
                exit("parse_kpoints", "The number of columns must be 3 when KPMODE = 0");
            }
            if (kpmode == 1 && kpelem.size() != 9) {
                exit("parse_kpoints", "The number of columns must be 9 when KPMODE = 1");
            }
            if (kpmode == 2 && kpelem.size() != 3) {
                exit("parse_kpoints", "The number of columns must be 3 when KPMODE = 2");
            }
            if (kpmode == 3 && kpelem.size() != 8) {
                exit("parse_kpoints", "The number of columns must be 8 when KPMODE = 3");
            }

            kpoint->kpInp.emplace_back(kpelem);
        }
    }

    kpoint->kpoint_mode = kpmode;
}

int Input::locate_tag(const std::string &key)
{
    int ret = 0;
    std::string line, line2;

    if (from_stdin) {

        // The following two lines do nothing when MPI version is executed.
        // I don't know why this happens.
        std::cin.clear();
        std::cin.seekg(0, std::ios_base::beg);

        while (std::cin >> line) {
#ifdef _USE_BOOST
            boost::to_lower(line);
            boost::trim(line);
#else
            std::transform(line.begin(), line.end(), line.begin(), tolower);
            line2 = line;
            line = trim(line2);
#endif
            if (line == key) {
                ret = 1;
                break;
            }
        }
        return ret;
    }
    ifs_input.clear();
    ifs_input.seekg(0, std::ios_base::beg);

    while (ifs_input >> line) {
#ifdef _USE_BOOST
        boost::to_lower(line);
        boost::trim(line);
#else
        std::transform(line.begin(), line.end(), line.begin(), tolower);
        line2 = line;
        line = trim(line2);
#endif
        if (line == key) {
            ret = 1;
            break;
        }
    }
    return ret;
}

void Input::get_var_dict(const std::vector<std::string> &input_list, std::map<std::string, std::string> &var_dict)
{
    std::string line, key, val;
    std::string line_wo_comment, line_tmp;
    std::string::size_type pos_first_comment_tag;
    std::vector<std::string> str_entry, str_varval;
    std::set<std::string> keyword_set;

    for (const auto &it: input_list) {
        keyword_set.insert(it);
    }

    var_dict.clear();

    if (from_stdin) {

        while (std::getline(std::cin, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }
#ifdef _USE_BOOST
            boost::trim_left(line_wo_comment);
#else
            line_tmp = line_wo_comment;
            line_wo_comment = ltrim(line_tmp);
#endif
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

                // Split the input line by ';'
#ifdef _USE_BOOST
            boost::split(str_entry, line_wo_comment, boost::is_any_of(";"));
#else
            str_entry = my_split(line_wo_comment, ';');
#endif

            for (auto it = str_entry.begin(); it != str_entry.end(); ++it) {

                // Split the input entry by '='
#ifdef _USE_BOOST
                std::string str_tmp = boost::trim_copy(*it);
#else
                std::string str_tmp = trim(*it);
#endif
                if (!str_tmp.empty()) {
#ifdef _USE_BOOST
                    boost::split(str_varval, str_tmp, boost::is_any_of("="));
#else
                    str_varval = my_split(str_tmp, '=');
#endif
                    if (str_varval.size() != 2) {
                        std::cout << " Failed to parse : ";
                        for (auto &it2: str_varval) {
                            std::cout << it2 << ' ';
                        }
                        std::cout << '\n';
                        exit("get_var_dict", "Unacceptable format");
                    }
#ifdef _USE_BOOST
                    key = boost::to_upper_copy(boost::trim_copy(str_varval[0]));
                    val = boost::trim_copy(str_varval[1]);
#else
                    key = trim(str_varval[0]);
                    std::transform(key.begin(), key.end(), key.begin(), toupper);
                    val = trim(str_varval[1]);
#endif
                    if (keyword_set.find(key) == keyword_set.end()) {
                        std::cout << "Could not recognize the variable " << key << '\n';
                        exit("get_var_dict", "Invalid variable found");
                    }

                    if (var_dict.find(key) != var_dict.end()) {
                        std::cout << "Variable " << key << " appears twice in the input file.\n";
                        exit("get_var_dict", "Redundant input parameter");
                    }

                    // If everything is OK, add the variable and the corresponding value
                    // to the dictionary.

                    var_dict.insert(std::map<std::string, std::string>::value_type(key, val));
                }
            }
        }
    } else {

        while (std::getline(ifs_input, line)) {

            // Ignore comment region
            pos_first_comment_tag = line.find_first_of('#');

            if (pos_first_comment_tag == std::string::npos) {
                line_wo_comment = line;
            } else {
                line_wo_comment = line.substr(0, pos_first_comment_tag);
            }
#ifdef _USE_BOOST
            boost::trim_left(line_wo_comment);
#else
            line_tmp = line_wo_comment;
            line_wo_comment = ltrim(line_tmp);
#endif
            if (line_wo_comment.empty()) continue;
            if (is_endof_entry(line_wo_comment)) break;

                // Split the input line by ';'
#ifdef _USE_BOOST
            boost::split(str_entry, line_wo_comment, boost::is_any_of(";"));
#else
            str_entry = my_split(line_wo_comment, ';');
#endif
            for (auto &it: str_entry) {

                // Split the input entry by '='

#ifdef _USE_BOOST
                std::string str_tmp = boost::trim_copy(*it);
#else
                std::string str_tmp = trim(it);
#endif
                if (!str_tmp.empty()) {
#ifdef _USE_BOOST
                    boost::split(str_varval, str_tmp, boost::is_any_of("="));
#else
                    str_varval = my_split(str_tmp, '=');
#endif

                    if (str_varval.size() != 2) {
                        std::cout << " Failed to parse : ";
                        for (auto &it2: str_varval) {
                            std::cout << it2 << ' ';
                        }
                        std::cout << '\n';
                        exit("get_var_dict", "Unacceptable format");
                    }

#ifdef _USE_BOOST
                    key = boost::to_upper_copy(boost::trim_copy(str_varval[0]));
                    val = boost::trim_copy(str_varval[1]);
#else
                    key = trim(str_varval[0]);
                    std::transform(key.begin(), key.end(), key.begin(), toupper);
                    val = trim(str_varval[1]);
#endif

                    if (keyword_set.find(key) == keyword_set.end()) {
                        std::cout << "Could not recognize the variable " << key << '\n';
                        exit("get_var_dict", "Invalid variable found");
                    }

                    if (var_dict.find(key) != var_dict.end()) {
                        std::cout << "Variable " << key << " appears twice in the input file.\n";
                        exit("get_var_dict", "Redundant input parameter");
                    }

                    // If everything is OK, add the variable and the corresponding value
                    // to the dictionary.

                    var_dict.insert(std::map<std::string, std::string>::value_type(key, val));
                }
            }
        }
    }
    keyword_set.clear();
}

bool Input::is_endof_entry(const std::string &str)
{
    return str[0] == '/';
}

void Input::split_str_by_space(const std::string &str, std::vector<std::string> &str_vec)
{
    std::string str_tmp;
    std::istringstream is(str);

    str_vec.clear();

    while (true) {
        str_tmp.clear();
        is >> str_tmp;
        if (str_tmp.empty()) {
            break;
        }
        str_vec.push_back(str_tmp);
    }
    str_tmp.clear();
}

template <typename T>
void Input::assign_val(T &val, const std::string &key, std::map<std::string, std::string> dict)
{
    // Assign a value to the variable "key" using the boost::lexica_cast.

    if (!dict[key].empty()) {
        try {
            val = boost::lexical_cast<T>(dict[key]);
        } catch (std::exception &e) {
            std::cout << e.what() << '\n';
            std::string str_tmp = "Invalid entry for the " + key + " tag.\n";
            str_tmp += " Please check the input value.";
            exit("assign_val", str_tmp.c_str());
        }
    }
}

template <typename T_to, typename T_from>
T_to Input::my_cast(T_from const &x)
{
    std::stringstream ss;
    T_to ret;

    ss << x;
    ss >> ret;

    return ret;
}
