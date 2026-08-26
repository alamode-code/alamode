/*
 input_setter.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace PHON_NS
{
class PHON;

// Values of each input block, filled by InputParser (or programmatically,
// e.g. from Python bindings) and applied to the PHON instance classes by
// InputSetter. The member initializers are the single source of the
// default values; the parser only overwrites what the input file provides.

struct GeneralInputVars
{
    std::string prefix;
    std::string mode; // uppercased run mode: PHONONS, KAPPA, SCPH, or QHA

    bool use_hdf5_io = true; // FILE_FORMAT = h5 (default) or text

    // Resolved restart flags (file auto-detection merged with the
    // deprecated &general RESTART/RESTART_4PH tags).
    bool restart = false;
    bool restart_4ph = false;

    double tolerance = 1.0e-3;  // TOLERANCE
    bool printsymmetry = false; // PRINTSYM

    double Tmin = 0.0;
    double Tmax = 1000.0;
    double dT = 10.0;

    std::vector<std::string> kdname; // KD
    std::vector<double> masskd;      // MASS

    // EMIN/EMAX: when given, the automatic energy-range detection is off.
    std::optional<double> emin;
    std::optional<double> emax;
    double delta_e = 10.0;

    unsigned int nonanalytic = 0;
    double na_sigma = 0.1;            // NA_SIGMA
    unsigned int band_connection = 0; // BCONNECT
    unsigned int bornsym = 0;         // BORNSYM
    std::string borninfo;             // BORNINFO
    double prec_ewald = 1.0e-12;      // PREC_EWALD (used when NONANALYTIC = 3)

    int nbands = -1;
    unsigned int verbosity = 1;

    double epsilon = 10.0;
    int ismear = -1;

    std::string fcsfile;  // FCSFILE
    std::string fc2file;  // FC2FILE
    std::string dfc2file; // DFC2FILE

    // FC2_TEMPERATURE selects one temperature row of the temperature-
    // dependent FC2 stored in a PREFIX.scph.h5 / PREFIX.qha.h5 file given
    // as FCSFILE or FC2FILE (negative = disabled).
    double fc2_temperature = -1.0;

    bool classical = false;
    bool use_triplet_symmetry = true; // TRISYM
    bool allow_unconverged = false;   // ALLOW_UNCONVERGED
};

struct AnalysisInputVars
{
    bool print_eval = false; // PRINTEVAL
    bool print_evec = false; // PRINTEVEC
    bool print_vel = false;  // PRINTVEL
    bool print_xsf = false;  // PRINTXSF
    bool print_msd = false;  // PRINTMSD

    bool print_ucorr = false; // UCORR
    int shift_ucorr[3] = {0, 0, 0};

    bool compute_dos = true;
    bool projected_dos = false;
    bool two_phonon_dos = false;
    bool longitudinal_dos = false;
    int scattering_phase_space = 0; // SPS

    int gruneisen_mode = 0;
    int sublattice_relax = 0;
    bool print_newfcs = false;
    double delta_a = 0.001;

    // QUARTIC, after the input policy that MODE = SCPH/QHA always requires
    // the quartic machinery has been applied.
    int quartic_mode = 0;

    std::string ks_input;
    bool calc_realpart = false; // REALPART
    bool fstate_omega = false;  // FSTATE_W
    bool bubble_omega = false;  // SELF_W
    int calc_selfenergy = 0;    // SELF_ENERGY

    int print_V3 = 0;
    int print_V4 = 0;
    bool participation_ratio = false; // PRINTPR
    bool print_fc2_ewald = false;     // FC2_EWALD
    int calc_dielectric_constant = 0; // DIELEC
    bool print_zmode = false;         // ZMODE
    bool print_irreps = false;        // IRREPS
    bool calc_FE_bubble = false;      // FE_BUBBLE

    bool print_anime = false;
    std::string anime_format = "XYZ";
    int anime_frames = 20;
    unsigned int anime_cellsize[3] = {1, 1, 1};
    double anime_kpoint[3] = {0.0, 0.0, 0.0};

    // Engaged when the PROJECTION_AXES tag is present.
    std::optional<std::vector<std::vector<double>>> projection_directions;

    bool print_self_consistent_fc2 = false;
};

struct KappaInputVars
{
    // INCLUDE_4PH resolved by the parser (legacy QUARTIC fallback applied).
    int include_4ph = 0;
    // The quartic (FC4) machinery is required for the 4ph self-energies, so
    // the parser may promote this above the &analysis QUARTIC value.
    int quartic_mode = 0;

    int ismear_4ph = 1;
    double epsilon_4ph = 10.0;
    double adaptive_factor = 1.0;

    std::string interpolator = "log-linear";
    int write_interpol = 0;
    unsigned int kmesh_coarse[3] = {0, 0, 0}; // KMESH_COARSE

    double len_boundary = 0.0; // [m]
    int calc_kappa_spec = 0;   // KAPPA_SPEC
    int calc_coherent = 0;     // KAPPA_COHERENT

    int include_isotope = 1;
    std::vector<double> isotope_factor; // ISOFACT

    // RESTART/RESTART_4PH given in &kappa override the &general values and
    // the file-existence auto-detection.
    std::optional<bool> restart;
    std::optional<bool> restart_4ph;

    std::string solver = "RTA"; // resolved: RTA, IBTE, VBTE, or DBTE
    int isotope_inscattering = 1;
    int max_cycle = 20;
    int min_cycle = 5;
    double iter_threshold = 0.02;
    double iterative_mixing = 0.9; // IBTE_MIXING
};

struct ScphInputVars
{
    unsigned int kmesh_scph[3] = {0, 0, 0};
    unsigned int kmesh_interpolate[3] = {0, 0, 0};

    double mixalpha = 0.1;
    unsigned int imix_scph = 1; // DIIS mixing by default; IMIX = 0 restores simple mixing
    unsigned int maxiter = 1000;
    double tolerance_scph = 1.0e-10;

    bool restart_scph = false; // resolved (file auto-detection merged with the tag)
    bool selfenergy_offdiagonal = true;
    unsigned int ialgo = 0;
    bool lower_temp = true;
    bool warmstart = true;
    unsigned int bubble = 0;
    int relax_str = 0;
};

struct QhaInputVars
{
    unsigned int kmesh_qha[3] = {0, 0, 0};
    unsigned int kmesh_interpolate[3] = {0, 0, 0};

    bool lower_temp = true;
    int relax_str = 1;
    int qha_scheme = 0;
    bool selfenergy_offdiagonal = true;
    unsigned int ialgo = 0;
    bool restart_qha = false; // resolved (file auto-detection merged with the tag)
};

struct RelaxInputVars
{
    int relax_algo = 2;
    int max_str_iter = 100;
    double coord_conv_tol = 1.0e-5;
    double gradient_conv_tol = 0.0;
    double cell_gradient_conv_tol = 0.0;
    int gdiis_control = 1;
    double mixbeta_coord = 0.5;
    double alpha_steepest_decent = 1.0e4;

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
};

class InputSetter
{
public:
    InputSetter();

    ~InputSetter();

    void set_general_vars(PHON *phon, const GeneralInputVars &vars) const;

    void set_analysis_vars(PHON *phon, const AnalysisInputVars &vars) const;

    void set_kappa_vars(PHON *phon, const KappaInputVars &vars) const;

    void set_scph_vars(PHON *phon, const ScphInputVars &vars) const;

    void set_qha_vars(PHON *phon, const QhaInputVars &vars) const;

    void set_relax_vars(PHON *phon, const RelaxInputVars &vars) const;

    // lavec_in holds the lattice vectors (already scaled by the lattice
    // factor) in the same row/column convention as System::lavec_p_input.
    void set_cell_parameter(PHON *phon, const double lavec_in[3][3]) const;

    void set_kpoint_vars(PHON *phon, int kpmode, const std::vector<std::vector<std::string>> &kpdata) const;

    void set_initial_strain(PHON *phon, const double u_tensor_in[3][3]) const;

    void set_strain_newfcs(PHON *phon, const double u_tensor_in[3][3]) const;

    void set_initial_displacements(PHON *phon, const std::vector<std::vector<double>> &u_xyz) const;
};
} // namespace PHON_NS
