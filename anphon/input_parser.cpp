/*
 input_parser.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "input_parser.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "error.h"
#include "input_setter.h"
#include "relaxation_types.h"

using namespace PHON_NS;

InputParser::InputParser() : input_setter(std::make_unique<InputSetter>())
{}

InputParser::~InputParser()
{
    if (ifs_input.is_open()) ifs_input.close();
}

std::string InputParser::get_run_mode() const
{
    return run_mode;
}

void InputParser::run(PHON *phon, const int narg, const char *const *arg)
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

    parse_input(phon);
}

void InputParser::parse_input(PHON *phon)
{
    if (!locate_tag("&general")) exit("parse_input", "&general entry not found in the input file");
    parse_general_vars(phon);

    if (locate_tag("&cell")) parse_cell_parameter(phon);

    const auto use_defaults_for_analysis = !locate_tag("&analysis");
    parse_analysis_vars(phon, use_defaults_for_analysis);

    if (!locate_tag("&kpoint")) exit("parse_input", "&kpoint entry not found in the input file");
    parse_kpoints(phon);

    if (run_mode == "PHONONS") {
        // Optional strain tensor for NEWFCS = 1: when given, the force constants
        // of the anisotropically strained system are estimated instead of the
        // isotropic volume change by DELTA_A.
        if (locate_tag("&strain")) {
            double u_tensor_tmp[3][3];
            parse_strain_tensor(u_tensor_tmp);
            input_setter->set_strain_newfcs(phon, u_tensor_tmp);
        }
    }

    if (run_mode == "KAPPA") {
        const auto use_defaults_for_kappa = !locate_tag("&kappa");
        parse_kappa_vars(phon, use_defaults_for_kappa);
    }

    if (run_mode == "SCPH") {
        if (!locate_tag("&scph")) exit("parse_input", "&scph entry not found in the input file");
        parse_scph_vars(phon);
    }
    if (run_mode == "QHA") {
        if (!locate_tag("&qha")) exit("parse_input", "&qha entry not found in the input file");
        parse_qha_vars(phon);
    }
    if ((run_mode == "SCPH" || run_mode == "QHA") && relax_str != 0) {
        if (!locate_tag("&relax")) exit("parse_input", "&relax entry not found in the input file");
        parse_relax_vars(phon);

        check_relax_vars();

        if (relax_str != 1) {
            if (!locate_tag("&strain")) exit("parse_input", "&strain entry not found in the input file");
            parse_initial_strain(phon);
        }
        if (!locate_tag("&displace")) exit("parse_input", "&displace entry not found in the input file");

        parse_initial_displace(phon);
    }
}

void InputParser::parse_general_vars(PHON *phon)
{
    // Read input parameters in the &general-field.

    int i;
    struct stat st
    {};
    const std::vector<std::string> input_list{"PREFIX",
                                              "MODE",
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
    std::vector<std::string> kdname_v, masskd_v;
    std::map<std::string, std::string> general_var_dict;

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

    GeneralInputVars general_vars;

    general_vars.prefix = general_var_dict["PREFIX"];
    auto mode = general_var_dict["MODE"];

    // FILE_FORMAT decides the restart-file format (h5 is the default;
    // "text" forces the legacy .result files for one transition release).
    // It must be resolved before the restart auto-detection below.
    std::string file_format = "h5";
    assign_val(file_format, "FILE_FORMAT", general_var_dict);
    std::transform(file_format.begin(), file_format.end(), file_format.begin(), tolower);
    if (file_format != "h5" && file_format != "text") {
        exit("parse_general_vars", "FILE_FORMAT must be either h5 or text.");
    }
    general_vars.use_hdf5_io = file_format == "h5";

    std::transform(mode.begin(), mode.end(), mode.begin(), toupper);

    // MODE = kappa is the current name of the thermal-conductivity mode;
    // the solver (RTA, IBTE, ...) is chosen by the SOLVER tag in &kappa.
    if (mode == "RTA") {
        warn("parse_general_vars",
             "MODE = RTA is deprecated; please use MODE = kappa and, if needed,\n"
             " choose the solver with the SOLVER tag in the &kappa field.");
        mode = "KAPPA";
    }
    general_vars.mode = mode;

    general_vars.fcsfile = general_var_dict["FCSFILE"];
    general_vars.fc2file = general_var_dict["FC2FILE"];
    general_vars.fc3file = general_var_dict["FC3FILE"];
    general_vars.fc4file = general_var_dict["FC4FILE"];

    if (general_vars.fcsfile.empty() && general_vars.fc2file.empty()) {
        exit("parse_general_vars", "Either FCSFILE or FC2FILE must be given to start a phonon calculation.");
    }

    if (!general_var_dict["KD"].empty()) {
        split_str_by_space(general_var_dict["KD"], kdname_v);
        general_vars.kdname = kdname_v;
    }

    if (!general_var_dict["MASS"].empty()) {
        split_str_by_space(general_var_dict["MASS"], masskd_v);
        general_vars.masskd.resize(masskd_v.size());
        for (i = 0; i < masskd_v.size(); ++i) {
            general_vars.masskd[i] = my_cast<double>(masskd_v[i]);
        }
    }

    // if a restart file exists in the current directory,
    // restart mode will be automatically turned on. With the default h5
    // format, an existing PREFIX.kappa.h5 or a legacy text file enables the
    // restart attempt for each channel independently; the actually finished
    // modes are read from the per-channel completion flags at setup.
    const auto file_result = general_vars.prefix + ".result";
    const auto file_result4 = general_vars.prefix + ".4ph.result";
    const auto file_kappa_h5 = general_vars.prefix + ".kappa.h5";

    general_vars.restart = stat(file_result.c_str(), &st) == 0;
    general_vars.restart_4ph = stat(file_result4.c_str(), &st) == 0;
    if (general_vars.use_hdf5_io && stat(file_kappa_h5.c_str(), &st) == 0) {
        general_vars.restart = true;
        general_vars.restart_4ph = true;
    }

    // Assign given values

    assign_val(general_vars.Tmin, "TMIN", general_var_dict);
    assign_val(general_vars.Tmax, "TMAX", general_var_dict);
    assign_val(general_vars.dT, "DT", general_var_dict);

    if (!general_var_dict["EMIN"].empty()) {
        auto emin = 0.0;
        assign_val(emin, "EMIN", general_var_dict);
        general_vars.emin = emin;
    }
    if (!general_var_dict["EMAX"].empty()) {
        auto emax = 1000.0;
        assign_val(emax, "EMAX", general_var_dict);
        general_vars.emax = emax;
    }

    assign_val(general_vars.delta_e, "DELTA_E", general_var_dict);

    assign_val(general_vars.tolerance, "TOLERANCE", general_var_dict);
    assign_val(general_vars.printsymmetry, "PRINTSYM", general_var_dict);

    assign_val(general_vars.nonanalytic, "NONANALYTIC", general_var_dict);
    // RESTART and RESTART_4PH now belong to the &kappa field; they are
    // still honored here for backward compatibility (the &kappa values win).
    if (!general_var_dict["RESTART"].empty() || !general_var_dict["RESTART_4PH"].empty()) {
        warn("parse_general_vars",
             "RESTART and RESTART_4PH in the &general field are deprecated;\n"
             " please move them to the &kappa field.");
    }
    assign_val(general_vars.restart, "RESTART", general_var_dict);
    assign_val(general_vars.restart_4ph, "RESTART_4PH", general_var_dict);

    assign_val(general_vars.nbands, "NBANDS", general_var_dict);
    assign_val(general_vars.borninfo, "BORNINFO", general_var_dict);

    assign_val(general_vars.ismear, "ISMEAR", general_var_dict);
    assign_val(general_vars.epsilon, "EPSILON", general_var_dict);
    assign_val(general_vars.na_sigma, "NA_SIGMA", general_var_dict);

    assign_val(general_vars.fc2_temperature, "FC2_TEMPERATURE", general_var_dict);
    assign_val(general_vars.allow_unconverged, "ALLOW_UNCONVERGED", general_var_dict);

    assign_val(general_vars.classical, "CLASSICAL", general_var_dict);
    assign_val(general_vars.band_connection, "BCONNECT", general_var_dict);
    assign_val(general_vars.use_triplet_symmetry, "TRISYM", general_var_dict);
    assign_val(general_vars.bornsym, "BORNSYM", general_var_dict);
    assign_val(general_vars.verbosity, "VERBOSITY", general_var_dict);

    if (general_vars.band_connection > 2) {
        exit("parse_general_vars", "BCONNECT-tag can take 0, 1, or 2.");
    }

    if (general_vars.nonanalytic == 3) {
        assign_val(general_vars.prec_ewald, "PREC_EWALD", general_var_dict);
        if (general_vars.prec_ewald <= 0.0 || general_vars.prec_ewald >= 1.0) {
            exit("parse_general_vars", "PREC_EWALD should be a small positive value.");
        }
    }

    if (general_vars.nonanalytic > 3) {
        exit("parse_general_vars", "NONANALYTIC-tag can take 0, 1, 2, or 3.");
    }
    if (general_vars.nonanalytic && general_vars.borninfo.empty()) {
        exit("parse_general_vars", "BORNINFO must be specified when NONANALYTIC > 0.");
    }

    general_vars.dfc2file = general_var_dict["DFC2FILE"];
    if (!general_vars.dfc2file.empty() && general_vars.fc2_temperature < 0.0) {
        exit("parse_general_vars", "DFC2FILE requires FC2_TEMPERATURE to select the temperature.");
    }

    // Keep the values the later blocks depend on.

    job_title = general_vars.prefix;
    run_mode = general_vars.mode;
    use_hdf5_io = general_vars.use_hdf5_io;

    input_setter->set_general_vars(phon, general_vars);

    general_var_dict.clear();
}

void InputParser::parse_analysis_vars(PHON *phon, const bool use_default_values)
{
    // Read input parameters in the &analysis field.
    int i;

    std::vector<std::string> input_list{
        "PRINTEVAL", "PRINTEVEC",   "PRINTXSF",        "PRINTVEL",         "QUARTIC",
        "KS_INPUT",  "REALPART",    "FSTATE_W",        "PRINTMSD",         "DOS",
        "PDOS",      "TDOS",        "GRUNEISEN",       "NEWFCS",           "SUBLATTICE_RELAX",
        "DELTA_A",   "ANIME",       "ANIME_CELLSIZE",  "ANIME_FORMAT",     "ANIME_FRAMES",
        "SPS",       "PRINTV3",     "PRINTPR",         "FC2_EWALD",        "SELF_W",
        "UCORR",     "SHIFT_UCORR", "DIELEC",          "SELF_ENERGY",      "PRINTV4",
        "ZMODE",     "IRREPS",      "PROJECTION_AXES", "LONGITUDINAL_DOS", "FE_BUBBLE"};

    std::map<std::string, std::string> analysis_var_dict;
    std::vector<std::string> anime_kpoint, anime_cellsize;

    AnalysisInputVars analysis_vars;
    auto do_projection = false;

    // Assign values to variables

    if (!use_default_values) {
        get_var_dict(input_list, analysis_var_dict);

        assign_val(analysis_vars.print_vel, "PRINTVEL", analysis_var_dict);
        assign_val(analysis_vars.print_evec, "PRINTEVEC", analysis_var_dict);
        assign_val(analysis_vars.print_eval, "PRINTEVAL", analysis_var_dict);
        assign_val(analysis_vars.print_msd, "PRINTMSD", analysis_var_dict);
        assign_val(analysis_vars.print_ucorr, "UCORR", analysis_var_dict);

        assign_val(analysis_vars.compute_dos, "DOS", analysis_var_dict);
        assign_val(analysis_vars.projected_dos, "PDOS", analysis_var_dict);
        assign_val(analysis_vars.two_phonon_dos, "TDOS", analysis_var_dict);
        assign_val(analysis_vars.longitudinal_dos, "LONGITUDINAL_DOS", analysis_var_dict);

        assign_val(analysis_vars.scattering_phase_space, "SPS", analysis_var_dict);
        assign_val(analysis_vars.gruneisen_mode, "GRUNEISEN", analysis_var_dict);
        if (analysis_vars.gruneisen_mode < 0 || analysis_vars.gruneisen_mode > 3) {
            exit("parse_analysis_vars", "GRUNEISEN must be 0, 1, 2, or 3.");
        }
        assign_val(analysis_vars.sublattice_relax, "SUBLATTICE_RELAX", analysis_var_dict);
        if (analysis_vars.sublattice_relax < 0 || analysis_vars.sublattice_relax > 1) {
            exit("parse_analysis_vars", "SUBLATTICE_RELAX must be 0 or 1.");
        }
        assign_val(analysis_vars.print_newfcs, "NEWFCS", analysis_var_dict);
        assign_val(analysis_vars.delta_a, "DELTA_A", analysis_var_dict);

        assign_val(analysis_vars.quartic_mode, "QUARTIC", analysis_var_dict);
        assign_val(analysis_vars.calc_realpart, "REALPART", analysis_var_dict);
        assign_val(analysis_vars.fstate_omega, "FSTATE_W", analysis_var_dict);
        assign_val(analysis_vars.ks_input, "KS_INPUT", analysis_var_dict);
        assign_val(analysis_vars.bubble_omega, "SELF_W", analysis_var_dict);
        assign_val(analysis_vars.calc_selfenergy, "SELF_ENERGY", analysis_var_dict);

        assign_val(analysis_vars.print_xsf, "PRINTXSF", analysis_var_dict);
        assign_val(analysis_vars.print_V3, "PRINTV3", analysis_var_dict);
        assign_val(analysis_vars.print_V4, "PRINTV4", analysis_var_dict);
        assign_val(analysis_vars.participation_ratio, "PRINTPR", analysis_var_dict);
        assign_val(analysis_vars.print_fc2_ewald, "FC2_EWALD", analysis_var_dict);
        assign_val(analysis_vars.calc_dielectric_constant, "DIELEC", analysis_var_dict);
        assign_val(analysis_vars.print_zmode, "ZMODE", analysis_var_dict);
        assign_val(analysis_vars.print_irreps, "IRREPS", analysis_var_dict);
        assign_val(analysis_vars.calc_FE_bubble, "FE_BUBBLE", analysis_var_dict);

        analysis_vars.print_anime = analysis_var_dict.find("ANIME") != analysis_var_dict.end();

        do_projection = analysis_var_dict.find("PROJECTION_AXES") != analysis_var_dict.end();
    }

    if (analysis_vars.print_anime) {
        split_str_by_space(analysis_var_dict["ANIME"], anime_kpoint);

        if (anime_kpoint.size() != 3) {
            exit("parse_analysis_vars", "The number of entries for ANIME should be 3.");
        }
        for (i = 0; i < 3; ++i) {
            analysis_vars.anime_kpoint[i] = my_cast<double>(anime_kpoint[i]);
        }

        split_str_by_space(analysis_var_dict["ANIME_CELLSIZE"], anime_cellsize);

        if (anime_cellsize.size() != 3) {
            exit("parse_analysis_vars", "The number of entries for ANIME_CELLSIZE should be 3.");
        }

        for (i = 0; i < 3; ++i) {
            try {
                analysis_vars.anime_cellsize[i] = boost::lexical_cast<unsigned int>(anime_cellsize[i]);
            } catch (std::exception &e) {
                std::cout << e.what() << '\n';
                exit("parse_analysis_vars", "ANIME_CELLSIZE must be a set of positive integers.");
            }
            if (analysis_vars.anime_cellsize[i] < 1) {
                exit("parse_analysis_vars", "Please give positive integers for ANIME_CELLSIZE.");
            }
        }

        assign_val(analysis_vars.anime_format, "ANIME_FORMAT", analysis_var_dict);
        std::transform(analysis_vars.anime_format.begin(),
                       analysis_vars.anime_format.end(),
                       analysis_vars.anime_format.begin(),
                       toupper);

        if (analysis_vars.anime_format != "XSF" && analysis_vars.anime_format != "AXSF" &&
            analysis_vars.anime_format != "XYZ")
        {
            exit("parse_analysis_vars", "Invalid ANIME_FORMAT");
        }

        assign_val(analysis_vars.anime_frames, "ANIME_FRAMES", analysis_var_dict);
    }

    if (analysis_vars.print_ucorr) {
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
                    analysis_vars.shift_ucorr[i] = boost::lexical_cast<int>(list_shift_ucorr[i]);
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

        analysis_vars.projection_directions = projection_directions;
    }

    // MODE = SCPH/QHA always requires the quartic machinery. This decision
    // is input policy, so it lives here; Fcs_phonon and the other consumers
    // only read the value. (A QUARTIC = 2 request is overridden for these
    // modes, matching the historical behavior.)
    if (run_mode == "SCPH" || run_mode == "QHA") analysis_vars.quartic_mode = 1;

    // The irrep analysis is defined for the static structure at Gamma only;
    // disabling it here keeps it from triggering Born-charge loading in the
    // other modes.
    if (analysis_vars.print_irreps && run_mode != "PHONONS") {
        warn("parse_analysis_vars", "IRREPS = 1 is effective only when MODE = phonons. The tag is ignored.");
        analysis_vars.print_irreps = false;
    }

    // Keep the values the later blocks depend on.

    quartic_mode = analysis_vars.quartic_mode;
    calc_FE_bubble = analysis_vars.calc_FE_bubble;

    input_setter->set_analysis_vars(phon, analysis_vars);

    analysis_var_dict.clear();
}

void InputParser::parse_kappa_vars(PHON *phon, const bool use_default_values)
{
    std::string str_tmp;
    const std::vector<std::string> input_list{
        "KMESH_COARSE",         "EPSILON_4PH",  "ISMEAR_4PH",     "INCLUDE_4PH",     "RESTART",
        "RESTART_4PH",          "INTERPOLATOR", "LEN_BOUNDARY",   "ISOTOPE",         "ISOFACT",
        "KAPPA_COHERENT",       "KAPPA_SPEC",   "WRITE_INTERPOL", "ADAPTIVE_FACTOR", "SOLVER",
        "ISOTOPE_INSCATTERING", "ITERATIVE",    "MAX_CYCLE",      "MIN_CYCLE",       "ITER_THRESHOLD",
        "IBTE_MIXING"};

    std::map<std::string, std::string> kappa_var_dict;
    std::vector<std::string> isofact_v;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, kappa_var_dict);

    KappaInputVars kappa_vars;

    auto iterative = false;
    auto iterative_given = false;
    std::string solver{};

    // -1 = tag not given; resolved below (legacy QUARTIC fallback).
    int include_4ph = -1;

    // Assign given values
    if (!use_default_values) {
        assign_val(include_4ph, "INCLUDE_4PH", kappa_var_dict);
        assign_val(kappa_vars.ismear_4ph, "ISMEAR_4PH", kappa_var_dict);
        assign_val(kappa_vars.epsilon_4ph, "EPSILON_4PH", kappa_var_dict);
        assign_val(kappa_vars.interpolator, "INTERPOLATOR", kappa_var_dict);
        assign_val(kappa_vars.write_interpol, "WRITE_INTERPOL", kappa_var_dict);
        assign_val(kappa_vars.len_boundary, "LEN_BOUNDARY", kappa_var_dict);
        assign_val(kappa_vars.calc_coherent, "KAPPA_COHERENT", kappa_var_dict);
        assign_val(kappa_vars.include_isotope, "ISOTOPE", kappa_var_dict);
        assign_val(kappa_vars.calc_kappa_spec, "KAPPA_SPEC", kappa_var_dict);
        str_tmp = kappa_var_dict["KMESH_COARSE"];
        assign_val(kappa_vars.adaptive_factor, "ADAPTIVE_FACTOR", kappa_var_dict);
        if (kappa_vars.adaptive_factor <= 0.0) {
            exit("parse_kappa_vars", "ADAPTIVE_FACTOR must be positive.");
        }
        iterative_given = !kappa_var_dict["ITERATIVE"].empty();
        assign_val(iterative, "ITERATIVE", kappa_var_dict);
        assign_val(kappa_vars.isotope_inscattering, "ISOTOPE_INSCATTERING", kappa_var_dict);
        assign_val(solver, "SOLVER", kappa_var_dict);
        assign_val(kappa_vars.max_cycle, "MAX_CYCLE", kappa_var_dict);
        assign_val(kappa_vars.min_cycle, "MIN_CYCLE", kappa_var_dict);
        assign_val(kappa_vars.iterative_mixing, "IBTE_MIXING", kappa_var_dict);
        assign_val(kappa_vars.iter_threshold, "ITER_THRESHOLD", kappa_var_dict);
    }

    // INCLUDE_4PH is the explicit switch for the four-phonon channel.
    // Legacy inputs enabled it implicitly via QUARTIC > 0 (an &analysis
    // tag), which is still honored with a deprecation warning.
    if (include_4ph < 0) {
        include_4ph = quartic_mode > 0 ? 1 : 0;
        if (include_4ph == 1) {
            warn("parse_kappa_vars",
                 "Enabling four-phonon scattering because QUARTIC > 0 is deprecated;\n"
                 " please set INCLUDE_4PH = 1 in the &kappa field instead.");
        }
    } else if (include_4ph > 0 && quartic_mode == 0) {
        // The quartic (FC4) machinery is required for the 4ph self-energies.
        quartic_mode = 1;
    }
    kappa_vars.include_4ph = include_4ph;
    kappa_vars.quartic_mode = quartic_mode;

    // RESTART / RESTART_4PH given here override the (deprecated) &general
    // values and the file-existence auto-detection.
    if (!use_default_values) {
        if (!kappa_var_dict["RESTART"].empty()) {
            auto restart_in = false;
            assign_val(restart_in, "RESTART", kappa_var_dict);
            kappa_vars.restart = restart_in;
        }
        if (!kappa_var_dict["RESTART_4PH"].empty()) {
            auto restart_in = false;
            assign_val(restart_in, "RESTART_4PH", kappa_var_dict);
            kappa_vars.restart_4ph = restart_in;
        }
    }

    // SOLVER selects the BTE solver level; ITERATIVE = 1 is the deprecated
    // spelling of SOLVER = IBTE.
    std::transform(solver.begin(), solver.end(), solver.begin(), toupper);
    if (solver.empty()) {
        if (iterative_given) {
            warn("parse_kappa_vars", "The ITERATIVE tag is deprecated; please use SOLVER = RTA or SOLVER = IBTE.");
        }
        solver = iterative ? "IBTE" : "RTA";
    } else if (iterative_given) {
        warn("parse_kappa_vars", "Both SOLVER and the deprecated ITERATIVE tag are given; SOLVER wins.");
    }
    if (solver == "SERTA") solver = "RTA";
    if (solver != "RTA" && solver != "IBTE" && solver != "VBTE" && solver != "DBTE") {
        exit("parse_kappa_vars", "SOLVER must be one of RTA, IBTE, DBTE, or VBTE.");
    }
    if (solver == "IBTE" || solver == "VBTE") {
        warn("parse_kappa_vars",
             ("SOLVER = " + solver +
              " is a pilot implementation under development;\n"
              " please check the validity of the results carefully.")
                 .c_str());
    }
    if (solver == "DBTE") {
        warn("parse_kappa_vars",
             "SOLVER = DBTE is a diagnostic direct solver (dense eigendecomposition\n"
             " of the collision kernel); it is intended for small k meshes.");
    }
    kappa_vars.solver = solver;

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

        for (auto i = 0; i < 3; ++i) kappa_vars.kmesh_coarse[i] = kmesh_v[i];
    }

    boost::to_lower(kappa_vars.interpolator);
    std::vector<std::string> supported_interpolator{"linear", "log-linear", "modified-log-linear"};
    if (std::find(std::begin(supported_interpolator), std::end(supported_interpolator), kappa_vars.interpolator) ==
        std::end(supported_interpolator))
    {
        exit("parse_kappa_vars", "INTERPOLATOR is not supported.");
    }

    // set isotope
    if (kappa_vars.include_isotope) {
        if (!kappa_var_dict["ISOFACT"].empty()) {
            split_str_by_space(kappa_var_dict["ISOFACT"], isofact_v);
            kappa_vars.isotope_factor.resize(isofact_v.size());
            for (auto i = 0; i < isofact_v.size(); ++i) {
                kappa_vars.isotope_factor[i] = my_cast<double>(isofact_v[i]);
            }
        }
    }

    input_setter->set_kappa_vars(phon, kappa_vars);

    kappa_var_dict.clear();
}

void InputParser::parse_scph_vars(PHON *phon)
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
    std::vector<unsigned int> kmesh_v, kmesh_interpolate_v;
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

    ScphInputVars scph_vars;

    // restart mode will be automatically turned on for SCPH calculations.
    auto file_dymat = job_title + ".scph_dymat";
    scph_vars.restart_scph = stat(file_dymat.c_str(), &st) == 0;

    // Assign given values

    assign_val(scph_vars.maxiter, "MAXITER", scph_var_dict);
    assign_val(scph_vars.mixalpha, "MIXALPHA", scph_var_dict);
    assign_val(scph_vars.selfenergy_offdiagonal, "SELF_OFFDIAG", scph_var_dict);
    assign_val(scph_vars.ialgo, "IALGO", scph_var_dict);
    assign_val(scph_vars.imix_scph, "IMIX", scph_var_dict);
    if (scph_vars.imix_scph > 1) {
        exit("parse_scph_vars", "IMIX must be 0 (simple mixing) or 1 (DIIS mixing).");
    }
    if (scph_vars.imix_scph == 1 && (scph_vars.mixalpha <= 0.0 || scph_vars.mixalpha > 1.0)) {
        exit("parse_scph_vars", "MIXALPHA must be in (0, 1] when IMIX = 1.");
    }
    assign_val(scph_vars.tolerance_scph, "TOL_SCPH", scph_var_dict);
    assign_val(scph_vars.lower_temp, "LOWER_TEMP", scph_var_dict);
    assign_val(scph_vars.warmstart, "WARMSTART", scph_var_dict);
    assign_val(scph_vars.bubble, "BUBBLE", scph_var_dict);
    assign_val(scph_vars.relax_str, "RELAX_STR", scph_var_dict);
    if (!is_valid_relaxation_str_mode(scph_vars.relax_str)) {
        exit("parse_scph_vars", "RELAX_STR must be 0, 1, 2, or 3.");
    }
    if (scph_vars.relax_str != to_int(RelaxationStrMode::None) && !scph_vars.selfenergy_offdiagonal) {
        exit("parse_scph_vars", "SELF_OFFDIAG = 0 cannot be used when RELAX_STR != 0.");
    }

    if (scph_vars.relax_str != to_int(RelaxationStrMode::None)) {
        auto file_harm_dymat = job_title + ".renorm_harm_dymat";
        auto file_v0 = job_title + ".V0";
        scph_vars.restart_scph =
            scph_vars.restart_scph && (stat(file_harm_dymat.c_str(), &st) == 0) && (stat(file_v0.c_str(), &st) == 0);
    }

    // The unified state file replaces the legacy text trio; either enables
    // the restart attempt (completeness is validated at load time).
    if (use_hdf5_io) {
        auto file_scph_h5 = job_title + ".scph.h5";
        if (stat(file_scph_h5.c_str(), &st) == 0) scph_vars.restart_scph = true;
    }

    assign_val(scph_vars.restart_scph, "RESTART_SCPH", scph_var_dict);

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

    for (auto i = 0; i < 3; ++i) {
        scph_vars.kmesh_scph[i] = kmesh_v[i];
        scph_vars.kmesh_interpolate[i] = kmesh_interpolate_v[i];
    }

    // Keep the values the later blocks depend on.

    relax_str = scph_vars.relax_str;
    scph_bubble = scph_vars.bubble;

    input_setter->set_scph_vars(phon, scph_vars);

    scph_var_dict.clear();
}

void InputParser::parse_qha_vars(PHON *phon)
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
    std::vector<unsigned int> kmesh_v, kmesh_interpolate_v;

    std::map<std::string, std::string> qha_var_dict;

    get_var_dict(input_list, qha_var_dict);

    for (auto &no_default: no_defaults) {
        if (qha_var_dict.find(no_default) == qha_var_dict.end()) {
            exit("parse_qha_vars", "The following variable is not found in &qha input region: ", no_default.c_str());
        }
    }

    QhaInputVars qha_vars;

    assign_val(qha_vars.lower_temp, "LOWER_TEMP", qha_var_dict);
    assign_val(qha_vars.relax_str, "RELAX_STR", qha_var_dict);
    assign_val(qha_vars.qha_scheme, "QHA_SCHEME", qha_var_dict);
    assign_val(qha_vars.selfenergy_offdiagonal, "SELF_OFFDIAG", qha_var_dict);
    assign_val(qha_vars.ialgo, "IALGO", qha_var_dict);

    if (!is_valid_relaxation_str_mode(qha_vars.relax_str)) {
        exit("parse_qha_vars", "RELAX_STR must be 1, 2, or 3 when mode = QHA.");
    }
    if (qha_vars.relax_str == to_int(RelaxationStrMode::None)) {
        exit("parse_qha_vars", "RELAX_STR = 0 is not supported when mode = QHA.");
    }
    if (!is_valid_qha_scheme(qha_vars.qha_scheme)) {
        exit("parse_qha_vars", "QHA_SCHEME must be 0, 1, or 2.");
    }
    if (qha_vars.relax_str != to_int(RelaxationStrMode::None) && !qha_vars.selfenergy_offdiagonal) {
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

    for (auto i = 0; i < 3; ++i) {
        qha_vars.kmesh_qha[i] = kmesh_v[i];
        qha_vars.kmesh_interpolate[i] = kmesh_interpolate_v[i];
    }

    // Set other values

    auto file_renorm_harm_dymat = job_title + ".renorm_harm_dymat";
    auto file_v0 = job_title + ".V0";
    qha_vars.restart_qha = (stat(file_renorm_harm_dymat.c_str(), &st) == 0) && (stat(file_v0.c_str(), &st) == 0);

    if (use_hdf5_io) {
        auto file_qha_h5 = job_title + ".qha.h5";
        if (stat(file_qha_h5.c_str(), &st) == 0) qha_vars.restart_qha = true;
    }

    assign_val(qha_vars.restart_qha, "RESTART_QHA", qha_var_dict);

    // Keep the values the later blocks depend on.

    relax_str = qha_vars.relax_str;

    input_setter->set_qha_vars(phon, qha_vars);

    qha_var_dict.clear();
}

void InputParser::parse_relax_vars(PHON *phon)
{
    // Read input parameters in the &relax-field.

    const std::vector<std::string> input_list{
        "RELAX_ALGO",    "MAX_STR_ITER",  "COORD_CONV_TOL",   "GRADIENT_CONV_TOL", "CELL_GRADIENT_CONV_TOL",
        "GDIIS_CONTROL", "GDIIS_PLAIN",   "MIXBETA_COORD",    "ALPHA_STDECENT",    "CELL_CONV_TOL",
        "MIXBETA_CELL",  "SET_INIT_STR",  "COOLING_U0_INDEX", "COOLING_U0_THR",    "ADD_HESS_DIAG",
        "STAT_PRESSURE", "RENORM_3TO2ND", "RENORM_2TO1ST",    "RENORM_34TO1ST",    "STRAIN_IFC_DIR",
        "ELASTIC_CONST"};

    std::map<std::string, std::string> stropt_var_dict;

    if (from_stdin) {
        std::cin.ignore();
    } else {
        ifs_input.ignore();
    }

    get_var_dict(input_list, stropt_var_dict);

    relax_vars = RelaxInputVars();

    // The Farkas-Schlegel controlled GDIIS is the default for RELAX_ALGO = 3.
    // GDIIS_PLAIN = 1 switches it off (regular GDIIS without the step-acceptance
    // tests). GDIIS_CONTROL is kept as a deprecated alias of the old enabling
    // switch; an explicit GDIIS_PLAIN takes precedence over it.
    int gdiis_plain = 0;

    assign_val(relax_vars.relax_algo, "RELAX_ALGO", stropt_var_dict);
    assign_val(relax_vars.max_str_iter, "MAX_STR_ITER", stropt_var_dict);
    assign_val(relax_vars.coord_conv_tol, "COORD_CONV_TOL", stropt_var_dict);
    assign_val(relax_vars.gradient_conv_tol, "GRADIENT_CONV_TOL", stropt_var_dict);
    assign_val(relax_vars.cell_gradient_conv_tol, "CELL_GRADIENT_CONV_TOL", stropt_var_dict);
    assign_val(relax_vars.gdiis_control, "GDIIS_CONTROL", stropt_var_dict); // deprecated
    assign_val(gdiis_plain, "GDIIS_PLAIN", stropt_var_dict);
    if (stropt_var_dict.find("GDIIS_CONTROL") != stropt_var_dict.end()) {
        warn("parse_relax_vars",
             "GDIIS_CONTROL is deprecated: controlled GDIIS is now the default. Use GDIIS_PLAIN = 1 to disable it.");
    }
    if (stropt_var_dict.find("GDIIS_PLAIN") != stropt_var_dict.end()) {
        relax_vars.gdiis_control = gdiis_plain ? 0 : 1;
    }

    if (relax_vars.relax_algo < 1 || relax_vars.relax_algo > 3) {
        exit("parse_relax_vars", "RELAX_ALGO must be 1 (steepest decent), 2 (Newton), or 3 (BFGS+GDIIS).");
    }

    if (relax_vars.relax_algo == 1) {
        assign_val(relax_vars.alpha_steepest_decent, "ALPHA_STDECENT", stropt_var_dict);
    } else if (relax_vars.relax_algo == 2) {
        assign_val(relax_vars.mixbeta_coord, "MIXBETA_COORD", stropt_var_dict);
    }

    assign_val(relax_vars.cell_conv_tol, "CELL_CONV_TOL", stropt_var_dict);
    if (relax_vars.relax_algo == 2) {
        assign_val(relax_vars.mixbeta_cell, "MIXBETA_CELL", stropt_var_dict);
    }

    assign_val(relax_vars.set_init_str, "SET_INIT_STR", stropt_var_dict);
    assign_val(relax_vars.cooling_u0_index, "COOLING_U0_INDEX", stropt_var_dict);
    assign_val(relax_vars.cooling_u0_thr, "COOLING_U0_THR", stropt_var_dict);
    assign_val(relax_vars.add_hess_diag, "ADD_HESS_DIAG", stropt_var_dict);
    assign_val(relax_vars.stat_pressure, "STAT_PRESSURE", stropt_var_dict);

    assign_val(relax_vars.renorm_3to2nd, "RENORM_3TO2ND", stropt_var_dict);
    assign_val(relax_vars.renorm_2to1st, "RENORM_2TO1ST", stropt_var_dict);
    assign_val(relax_vars.renorm_34to1st, "RENORM_34TO1ST", stropt_var_dict);

    assign_val(relax_vars.elastic_const, "ELASTIC_CONST", stropt_var_dict);
    if (relax_vars.elastic_const < 1 || relax_vars.elastic_const > 2) {
        exit("parse_relax_vars",
             "ELASTIC_CONST must be 1 (analytic, computed from the force constants)\n"
             " or 2 (read from elastic_constants.in).");
    }

    assign_val(relax_vars.strain_IFC_dir, "STRAIN_IFC_DIR", stropt_var_dict);
    if (!relax_vars.strain_IFC_dir.empty() &&
        relax_vars.strain_IFC_dir.at(relax_vars.strain_IFC_dir.length() - 1) != '/')
    {
        relax_vars.strain_IFC_dir = relax_vars.strain_IFC_dir + "/";
    }

    input_setter->set_relax_vars(phon, relax_vars);

    stropt_var_dict.clear();
}

void InputParser::check_relax_vars() const
{
    std::fstream fin_test;

    // structural optimization
    if (relax_str != 0) {

        if (calc_FE_bubble) {
            exit("check_relax_vars", "Sorry, RELAX_STR!=0 can't be used with bubble correction of the free energy.");
        }
        if (scph_bubble > 0) {
            exit("check_relax_vars",
                 "Sorry, RELAX_STR!=0 can't be used with bubble self-energy on top of the SCPH calculation.");
        }

        // relax the shape of the unit cell
        if (relax_str == 2 || relax_str == 3) {
            // strain-force coupling
            if (relax_vars.renorm_2to1st == 2) {
                fin_test.open(relax_vars.strain_IFC_dir + "strain_force.in");

                if (!fin_test) {
                    exit("check_relax_vars", "strain_force.in is required in STRAIN_IFC_DIR when RENORM_2TO1ST = 2.");
                }
                fin_test.close();
            }

            // strain-IFC coupling
            if (relax_vars.renorm_3to2nd == 2 || relax_vars.renorm_3to2nd == 3) {
                fin_test.open(relax_vars.strain_IFC_dir + "strain_harmonic.in");

                if (!fin_test) {
                    exit("check_relax_vars",
                         "strain_harmonic.in is required in STRAIN_IFC_DIR when RENORM_3TO2ND >= 2.");
                }

                fin_test.close();
            }
        }
    }
}

void InputParser::parse_strain_tensor(double (&u_tensor)[3][3])
{
    int i, j;
    std::string line;
    std::vector<std::string> line_split;

    const auto line_vec = read_block_lines();

    if (line_vec.size() != 3) {
        exit("parse_strain_tensor", "Too few or too much lines for the &strain field.\n \
                                            The number of valid lines for the &strain field should be 3.");
    }

    for (i = 0; i < 3; ++i) {

        line = line_vec[i];
        split(line_split, line, boost::is_any_of("\t "), boost::token_compress_on);

        // u_tensor
        if (line_split.size() == 3) {
            for (j = 0; j < 3; ++j) {
                u_tensor[i][j] = boost::lexical_cast<double>(line_split[j]);
            }
        } else {
            exit("parse_strain_tensor", "Unacceptable format for &strain field.");
        }
    }
}

void InputParser::parse_initial_strain(PHON *phon)
{
    double u_tensor_tmp[3][3];
    parse_strain_tensor(u_tensor_tmp);
    input_setter->set_initial_strain(phon, u_tensor_tmp);
}

void InputParser::parse_initial_displace(PHON *phon)
{
    int i, j;
    int itmp;
    int ixyz;
    int iat;
    std::string line;
    std::vector<std::string> line_split;

    int input_mode{-1};

    double unit;
    double a[3][3];
    std::vector<std::vector<double>> u_fractional, u_xyz;
    std::vector<double> vec_tmp(3);

    const auto line_vec = read_block_lines();

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

    input_setter->set_initial_displacements(phon, u_xyz);
}

void InputParser::parse_cell_parameter(PHON *phon)
{
    // Read the cell parameter

    int i, j;
    double a;
    double lavec_tmp[3][3];
    double lavec[3][3];
    std::string line;
    std::vector<std::string> line_split;

    const auto line_vec = read_block_lines();

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
            lavec[i][j] = a * lavec_tmp[i][j];
        }
    }

    input_setter->set_cell_parameter(phon, lavec);
}

void InputParser::parse_kpoints(PHON *phon)
{
    // Read the settings in the &kpoint field.

    int kpmode;
    std::string line;
    std::vector<std::string> kpelem;
    std::vector<std::vector<std::string>> kpdata;

    const auto line_vec = read_block_lines();

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

                if (!(kpmode >= 0 && kpmode <= 2)) {
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

            kpdata.push_back(kpelem);
        }
    }

    input_setter->set_kpoint_vars(phon, kpmode, kpdata);
}

std::vector<std::string> InputParser::read_block_lines()
{
    std::istream &is = from_stdin ? static_cast<std::istream &>(std::cin) : ifs_input;

    std::string line;
    std::vector<std::string> line_vec;

    while (std::getline(is, line)) {

        // Ignore comment region
        const auto pos_first_comment_tag = line.find_first_of('#');

        std::string line_wo_comment;
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

    return line_vec;
}

int InputParser::locate_tag(const std::string &key)
{
    std::istream &is = from_stdin ? static_cast<std::istream &>(std::cin) : ifs_input;

    auto ret = 0;
    std::string line, line2;

    // The following two lines do nothing when MPI version is executed.
    // I don't know why this happens.
    is.clear();
    is.seekg(0, std::ios_base::beg);

    while (is >> line) {
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

void InputParser::get_var_dict(const std::vector<std::string> &input_list, std::map<std::string, std::string> &var_dict)
{
    std::istream &is = from_stdin ? static_cast<std::istream &>(std::cin) : ifs_input;

    std::string line, key, val;
    std::string line_wo_comment, line_tmp;
    std::string::size_type pos_first_comment_tag;
    std::vector<std::string> str_entry, str_varval;
    std::set<std::string> keyword_set;

    for (const auto &it: input_list) {
        keyword_set.insert(it);
    }

    var_dict.clear();

    while (std::getline(is, line)) {

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
            std::string str_tmp = boost::trim_copy(it);
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

    keyword_set.clear();
}

bool InputParser::is_endof_entry(const std::string &str)
{
    return str[0] == '/';
}

void InputParser::split_str_by_space(const std::string &str, std::vector<std::string> &str_vec)
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
void InputParser::assign_val(T &val, const std::string &key, std::map<std::string, std::string> dict)
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
T_to InputParser::my_cast(T_from const &x)
{
    std::stringstream ss;
    T_to ret;

    ss << x;
    ss >> ret;

    return ret;
}
