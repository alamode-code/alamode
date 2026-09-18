/*
relaxation.cpp

Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "relaxation.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "dynamical.h"
#include "elastic_tensor.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "hdf5_parser.h"
#include "ifc_derivative.h"
#include "interpolation.h"
#include "memory.h"
#include "optimizers.h"
#include "scph.h"
#include "strain_coupling_io.h"
#include "strain_reference_cell.h"
#include "symmetry_core.h"
#include "system.h"
#include "timer.h"
#include "write_phonons.h"

extern "C"
{
#include "spglib.h"
}
// #include <boost/range/algorithm.hpp> // boost/range may induce compile error

using namespace PHON_NS;

Relaxation::Relaxation(PHON *phon) : Pointers(phon)
{
    set_default_variables();
    derivative_ifc = std::make_unique<DerivativeIFC>(*system,
                                                     *symmetry,
                                                     *fcs_phonon,
                                                     *dynamical,
                                                     *anharmonic_core,
                                                     mympi->my_rank,
                                                     mympi->nprocs);
}

Relaxation::~Relaxation()
{
    deallocate_variables();
}

void Relaxation::set_default_variables()
{
    // variables related to the structural optimization
    relax_str = 0;
    relax_algo = 2;
    max_str_iter = 100;
    coord_conv_tol = 1.0e-5;
    mixbeta_coord = 0.5;
    alpha_steepest_decent = 1.0e4;
    cell_conv_tol = 1.0e-5;
    mixbeta_cell = 0.5;
    gradient_conv_tol = 0.0;      // disabled by default (step-size convergence only)
    cell_gradient_conv_tol = 0.0; // disabled by default (step-size convergence only)
    gdiis_control = 1;            // controlled GDIIS by default; GDIIS_PLAIN = 1 disables it

    set_init_str = 1;

    add_hess_diag = 100.0; // [cm^{-1}]
    stat_pressure = 0.0;   // [GPa]

    // Sources of the strain couplings and elastic constants. The input parser
    // sets these on rank 0 only; the defaults must match RelaxInputVars so
    // that the other ranks hold defined values until setup_relaxation
    // broadcasts the parsed ones.
    strain_coupling = 7;
    renorm_3to2nd = 2;
    renorm_2to1st = 2;
    renorm_34to1st = 0;
    elastic_const = 2;
    strain_IFC_dir.clear();
    strain_file.clear();
}

void Relaxation::deallocate_variables()
{}

void Relaxation::setup_relaxation()
{
    MPI_Bcast(&relax_str, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // The strain-coupling settings are consumed on every rank
    // (compute_del_v_strain is collective), so they must not stay at their
    // defaults on the non-root ranks.
    MPI_Bcast(&renorm_3to2nd, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&renorm_2to1st, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&renorm_34to1st, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&elastic_const, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_string(strain_IFC_dir, 0, MPI_COMM_WORLD);
    mympi->MPI_Bcast_string(strain_file, 0, MPI_COMM_WORLD);

    const auto relax_mode = to_relaxation_str_mode(relax_str);
    if (relax_mode != RelaxationStrMode::None) {
        // Use the per-step symmetry tolerance for the reference cell on every rank.
        const auto &primcell = system->get_primcell();
        std::vector<Eigen::Vector3d> xf(primcell.number_of_atoms);
        for (auto iat = 0; iat < primcell.number_of_atoms; ++iat) xf[iat] = primcell.x_fractional.row(iat);
        std::string label;
        spacegroup_number_ref = detect_spacegroup(primcell.lattice_vector, xf, label);
        if (mympi->my_rank == 0 && writes->getVerbosity() > 0 && set_init_str == 3) {
            std::cout << "  SET_INIT_STR = 3: the high-symmetry phase is " << label << " (spglib, tolerance "
                      << std::scientific << std::setprecision(2) << symmetry->tolerance << std::defaultfloat << ")\n\n";
        }
    }
    if (!uses_strain_coupling(relax_mode)) {
        return;
    }
    if (!strain_file.empty()) {
        validate_strain_file();
    } else if (mympi->my_rank == 0 &&
               (renorm_2to1st == 2 || renorm_3to2nd == 2 || renorm_3to2nd == 3 || elastic_const == 2))
    {
        std::cout << "  NOTE: the strain couplings and elastic constants are read from the text files in\n"
                  << "        STRAIN_IFC_DIR (and C1_array.in in the working directory). This route is deprecated:\n"
                  << "        pack the files into one container with 'tools/strainfile.py pack' and give it as\n"
                  << "        STRAINFILE in the &relax field.\n\n";
    }
}

void Relaxation::validate_strain_file() const
{
    using strain_coupling::StrainCouplingFile;
    try {
        const StrainCouplingFile container(strain_file);
        const auto summary = container.probe();

        const auto need = [this](const bool present, const char *group, const std::string &tag) {
            if (present) return;
            throw std::runtime_error(tag + " needs the " + group + " data, but " + strain_file +
                                     " does not contain it.\n " + StrainCouplingFile::missing_group_hint(group));
        };
        if (elastic_const == 2) {
            need(summary.has_c2c3, "/Elastic", "Reading the elastic constants from file (STRAIN_COUPLING bit 1)");
        }
        if (renorm_2to1st == 2) {
            need(summary.has_strain_force,
                 "/StrainForce",
                 "Reading the strain-force coupling from file (STRAIN_COUPLING bit 2)");
        }
        if (renorm_3to2nd == 2 || renorm_3to2nd == 3) {
            need(summary.has_strain_harmonic,
                 "/StrainHarmonic",
                 "Reading the strain-harmonic-IFC coupling from file (STRAIN_COUPLING bit 4)");
        }

        // The reference structure must describe the crystal of this run; the
        // primitive cell of the run may be a nested super- or sub-cell of it.
        const auto ref = container.read_reference_cell();
        const auto &pcell = system->get_primcell();
        std::vector<std::string> symbols(pcell.number_of_atoms);
        for (std::size_t i = 0; i < pcell.number_of_atoms; ++i) symbols[i] = system->symbol_kd[pcell.kind[i]];
        const auto what = strain_file + ":/ReferenceCell";
        const auto match =
            strain_parsers::match_atoms(ref, pcell.lattice_vector, pcell.x_cartesian, symbols, what.c_str());

        if (mympi->my_rank == 0) {
            const auto flags = std::cout.flags();
            const auto prec = std::cout.precision();
            std::cout << "  STRAINFILE = " << strain_file << "\n"
                      << "    schema " << h5_schema_strain_coupling << " v" << summary.format_version;
            if (!summary.writer.empty()) std::cout << ", written by strainkit " << summary.writer;
            if (!summary.created_date.empty()) std::cout << " on " << summary.created_date;
            std::cout << "\n    Reference cell : " << ref.natom()
                      << " atoms;  V(current) / V(reference) = " << std::fixed << std::setprecision(4) << match.ratio
                      << "\n"
                      << "    Elastic        : "
                      << (summary.has_c2c3      ? "C2, C3"
                          : summary.has_elastic ? "no C2/C3"
                                                : "absent")
                      << (summary.has_stress ? " + reference stress" : ", no reference stress") << "\n"
                      << "    StrainForce    : " << (summary.has_strain_force ? "present" : "absent") << "\n"
                      << "    StrainHarmonic : " << (summary.has_strain_harmonic ? "present" : "absent") << "\n\n";
            std::cout.flags(flags);
            std::cout.precision(prec);
        }
    } catch (const std::runtime_error &e) {
        exit("setup_relaxation", e.what());
    }
}

void Relaxation::load_reference_stress(const ElasticTensor &elastic, double *C1_array) const
{
    if (strain_file.empty()) {
        elastic.read_C1_array(C1_array);
        return;
    }
    std::fill_n(C1_array, 9, 0.0);
    try {
        const strain_coupling::StrainCouplingFile container(strain_file);
        if (container.probe().has_stress) {
            elastic.set_reference_stress_from_set(container.read_elastic(), C1_array);
            return;
        }
    } catch (const std::runtime_error &e) {
        exit("load_reference_stress", e.what());
    }
    std::cout << "  " << strain_file
              << " has no /Elastic/stress dataset: the stress tensor at the reference structure is set to zero.\n";
}

void Relaxation::create_optimizer(const size_t num_modes)
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    if (relax_algo == 1) {
        // Steepest descent is independent of the relaxation mode and dimension.
        optimizer = std::make_unique<SteepestDescent_Optimizer>(alpha_steepest_decent);
    } else if (optimizes_coordinates_only(relax_mode) && relax_algo == 2) {
        optimizer = std::make_unique<Newton_Optimizer>(mixbeta_coord);
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell && relax_algo == 2) {
        optimizer = std::make_unique<CellCoord_Newton_Optimizer>(mixbeta_cell, mixbeta_coord);
    } else if (optimizes_coordinates_only(relax_mode) && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes - 3, num_modes - 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init, gdiis_control != 0);
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell && relax_algo == 3) {
        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(num_modes + 3, num_modes + 3);
        optimizer = std::make_unique<FarkasIII_Optimizer>(6, H_init, gdiis_control != 0);
    }
}

void Relaxation::set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array) const
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    // if the energy is evaluated at a nonzero strain (the cell is relaxed, or held
    // at the strain given in &strain), compute the elastic constants from the IFCs
    // or read them from file
    if (uses_strain_coupling(relax_mode)) {
        if (elastic_const == 1) {
            set_elastic_constants_from_ifcs(C1_array, C2_array, C3_array);
            return;
        }
        const ElasticTensor elastic(*system);
        std::string source_stress = "C1_array.in";
        std::string source_c2c3 = "elastic_constants.in";
        if (!strain_file.empty()) {
            // STRAINFILE: the reference stress and the elastic constants come
            // from its /Elastic group (GPa, multiplied by the volume of the
            // current primitive cell).
            strain_coupling::ElasticSet es;
            try {
                const strain_coupling::StrainCouplingFile container(strain_file);
                es = container.read_elastic();
            } catch (const std::runtime_error &e) {
                exit("set_elastic_constants", e.what());
            }
            if (!es.has_c2c3) {
                const auto msg = "Reading the elastic constants from file (STRAIN_COUPLING bit 1) needs the\n"
                                 " second- and third-order elastic constants, but " +
                                 strain_file + " has no /Elastic/soec and /Elastic/toec datasets.\n " +
                                 strain_coupling::StrainCouplingFile::missing_group_hint("/Elastic");
                exit("set_elastic_constants", msg.c_str());
            }
            if (!es.has_stress) {
                std::cout
                    << "  " << strain_file
                    << " has no /Elastic/stress dataset: the stress tensor at the reference structure is set to zero.\n";
            }
            elastic.set_reference_stress_from_set(es, C1_array);
            elastic.set_elastic_constants_from_set(es, C2_array, C3_array);
            source_stress = strain_file + ":/Elastic/stress";
            source_c2c3 = strain_file + ":/Elastic/soec,toec";
        } else {
            elastic.read_C1_array(C1_array);
            elastic.read_elastic_constants(C2_array, C3_array, strain_IFC_dir);
        }

        // Detect unexpected symmetry breaking in the user-supplied arrays:
        // the stress tensor must be symmetric, and C2/C3 must satisfy the
        // minor and pair-permutation symmetries of the elastic tensors.
        // The arrays are used as given; only a warning is issued.
        auto scale = 0.0;
        for (auto i = 0; i < 9; ++i) scale = std::max(scale, std::abs(C1_array[i]));
        if (const auto dev = ElasticTensor::stress_tensor_asymmetry(C1_array); dev > 1.0e-6 * std::max(scale, 1.0)) {
            const auto msg = "The stress tensor in " + source_stress + " is not symmetric.\n" +
                             " Please check the file for input errors.";
            warn("set_elastic_constants", msg.c_str());
        }
        scale = 0.0;
        for (auto i = 0; i < 9; ++i)
            for (auto j = 0; j < 9; ++j) scale = std::max(scale, std::abs(C2_array[i][j]));
        if (ElasticTensor::elastic_tensor2_asymmetry(C2_array) > 1.0e-6 * std::max(scale, 1.0)) {
            const auto msg = "The second-order elastic constants in " + source_c2c3 + " violate\n" +
                             " the minor or pair-permutation symmetry. Please check the file for input errors.";
            warn("set_elastic_constants", msg.c_str());
        }
        scale = 0.0;
        for (auto i = 0; i < 9; ++i)
            for (auto j = 0; j < 9; ++j)
                for (auto k = 0; k < 9; ++k) scale = std::max(scale, std::abs(C3_array[i][j][k]));
        if (ElasticTensor::elastic_tensor3_asymmetry(C3_array) > 1.0e-6 * std::max(scale, 1.0)) {
            const auto msg = "The third-order elastic constants in " + source_c2c3 + " violate\n" +
                             " the minor or pair-permutation symmetry. Please check the file for input errors.";
            warn("set_elastic_constants", msg.c_str());
        }
        return;
    }
    // if the unit cell is fixed,
    // dummy values are set in the elastic constants
    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
        ElasticTensor::set_dummy_elastic_constants(C1_array, C2_array, C3_array);
    }
}

void Relaxation::set_elastic_constants_from_ifcs(double *C1_array, double **C2_array, double ***C3_array) const
{
    if (fcs_phonon->force_constant_with_cell.size() < 2) {
        exit("set_elastic_constants_from_ifcs",
             "Computing the elastic constants from the IFCs (STRAIN_COUPLING bit 1 clear) requires\n"
             " the harmonic and cubic force constants (FCSFILE, or FC2FILE + FC3FILE).");
    }
    const auto &fc2 = fcs_phonon->force_constant_with_cell[0];
    const auto &fc3 = fcs_phonon->force_constant_with_cell[1];

    const ElasticTensor elastic(*system);

    // The reference stress is not contained in the IFC model; it is taken
    // from STRAINFILE (/Elastic/stress) or C1_array.in when present (zero
    // otherwise).
    load_reference_stress(elastic, C1_array);

    // Clamped-ion tensors: the sublattice relaxation is treated explicitly by
    // the internal coordinates (q0) of the structural optimization, so the
    // internal-strain correction must NOT be folded into C2/C3 here.
    // With NONANALYTIC = 3 the dipole-dipole long-range correction is applied
    // to C2 (short-range brackets from fc2_without_dipole plus the analytic
    // dipole brackets at fixed E field); C3 remains short-range because the
    // strain derivatives of Z* and epsilon are not contained in the IFCs.
    NDArray<double, 4> C2_gpa;
    const bool longrange = ewald->is_longrange;
    if (longrange) {
        elastic.calc_elastic_tensor_longrange(ewald->fc2_without_dipole, *ewald, C2_gpa, true);
    } else {
        elastic.calc_elastic_tensor(fc2, C2_gpa, true);
    }

    Tensor6 C3_gpa;
    elastic.calc_elastic_tensor3(fc2, fc3, false, C3_gpa, true);

    // V0(u) stores V0 * C in Ry per primitive cell.
    const auto gpa2ry = elastic.gpa_to_ry_per_cell();

    for (auto i = 0; i < 9; ++i) {
        for (auto j = 0; j < 9; ++j) {
            C2_array[i][j] = C2_gpa[i / 3][i % 3][j / 3][j % 3] * gpa2ry;
            for (auto k = 0; k < 9; ++k) {
                C3_array[i][j][k] = C3_gpa(i / 3, i % 3, j / 3, j % 3, k / 3, k % 3) * gpa2ry;
            }
        }
    }

    const auto flags = std::cout.flags();
    const auto prec = std::cout.precision();
    std::cout << "  STRAIN_COUPLING bit 1 clear: elastic constants are computed from the force constants.\n";
    std::cout << "  Harmonic force constants from  : "
              << (fcs_phonon->file_fc2.empty() ? fcs_phonon->file_fcs : fcs_phonon->file_fc2) << '\n';
    std::cout << "  Cubic force constants from     : "
              << (fcs_phonon->file_fc3.empty() ? fcs_phonon->file_fcs : fcs_phonon->file_fc3) << '\n';
    if (longrange) {
        std::cout << "  NONANALYTIC = 3: the dipole-dipole long-range correction is applied\n";
        std::cout << "  to the second-order elastic constants (fixed-E, macroscopic term excluded).\n";
    }
    std::cout << "  Clamped-ion second-order elastic constants (GPa, Voigt notation):\n";
    constexpr int voigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {2, 0}, {0, 1}};
    std::cout << std::fixed << std::setprecision(3);
    for (const auto &vi: voigt) {
        std::cout << "   ";
        for (const auto &vj: voigt) {
            std::cout << std::setw(11) << C2_gpa[vi[0]][vi[1]][vj[0]][vj[1]];
        }
        std::cout << '\n';
    }
    std::cout << "  The third-order elastic constants are computed from the cubic force constants.\n\n";
    std::cout.flags(flags);
    std::cout.precision(prec);
}


void Relaxation::set_init_structure_atT(RelaxationStructureState &structure_state, bool &converged_prev,
                                        int &str_diverged, const int i_temp_loop, double **omega2_harmonic,
                                        std::complex<double> ***evec_harmonic) const
{
    const auto relax_mode = to_relaxation_str_mode(relax_str);
    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    int i1, i2;

    // The optimizer is null for relaxation modes that do not use a coordinate
    // optimizer (e.g. PerturbativeQha), so guard every access to it here.
    if (optimizer) optimizer->reset();

    if (str_diverged) {
        if (writes->getVerbosity() > 0) {
            std::cout << " The crystal structure at the previous temperature is divergent.\n";
            std::cout << " read initial structure from input files.\n\n";
        }

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

        // set initial strain
        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }
        } else {
            set_initial_strain(u_tensor);
        }
        converged_prev = false;
        str_diverged = 0;

        // set the flag to initialize the optimizer
        if (optimizer) optimizer->reset();

        return;
    }

    if (writes->getVerbosity() > 0) {
        std::cout << " SET_INIT_STR = " << set_init_str << ":";
    }

    if (set_init_str == 1) {
        if (writes->getVerbosity() > 0) {
            std::cout << " set initial structure from the input file.\n\n";
        }

        set_initial_q0(q0, evec_harmonic);
        calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
            for (i1 = 0; i1 < 3; i1++) {
                for (i2 = 0; i2 < 3; i2++) {
                    u_tensor[i1][i2] = 0.0;
                }
            }
        } else {
            set_initial_strain(u_tensor);
        }
        converged_prev = false;
        if (optimizer) optimizer->reset();

    } else if (set_init_str == 2) {
        if (i_temp_loop == 0) {
            if (writes->getVerbosity() > 0) {
                std::cout << " set initial structure from the input file.\n\n";
            }

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            converged_prev = false;
            if (optimizer) optimizer->reset();
        } else {
            if (writes->getVerbosity() > 0) {
                std::cout << " start from structure from the previous temperature.\n\n";
            }
        }

    } else if (set_init_str == 3) {
        // read initial structure at initial temperature
        if (i_temp_loop == 0) {
            if (writes->getVerbosity() > 0) {
                std::cout << " read initial structure from input files.\n\n";
            }

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            } else {
                set_initial_strain(u_tensor);
            }
            if (optimizer) optimizer->reset();
        }
        // Re-seed symmetry breaking when the reference space group is restored.
        else if (spacegroup_of(structure_state) == spacegroup_number_ref)
        {
            if (writes->getVerbosity() > 0) {
                std::cout << '\n';
                std::cout << " the structure at the previous temperature has the space group of the"
                             " reference cell (#"
                          << spacegroup_number_ref << "):\n";
                std::cout << " set again initial displacement from input file.\n\n";
            }

            set_initial_q0(q0, evec_harmonic);
            calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);
            converged_prev = false;
            if (optimizer) optimizer->reset();
        } else {
            if (writes->getVerbosity() > 0) {
                std::cout << " start from the structure at the previous temperature.\n\n";
            }
        }
    }

    // CoordinatesAtFixedStrain: the cell is an input, not an optimization variable.
    // Re-assert the prescribed strain at every temperature so that it can neither be
    // inherited from the previous temperature (SET_INIT_STR = 2, 3) nor drift.
    if (relax_mode == RelaxationStrMode::CoordinatesAtFixedStrain) {
        set_initial_strain(u_tensor);
    }
}

void Relaxation::set_initial_q0(std::vector<double> &q0, std::complex<double> ***evec_harmonic) const
{
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;
    if (q0.size() != static_cast<std::size_t>(ns)) {
        q0.resize(ns);
    }

    std::fill(q0.begin(), q0.end(), 0.0);
    if (init_u0.empty()) return; // no &displace: start from the undistorted structure

    for (int is = 0; is < ns; is++) {
        for (int i_atm = 0; i_atm < natmin; i_atm++) {
            for (int ixyz = 0; ixyz < 3; ixyz++) {
                q0[is] += evec_harmonic[0][is][i_atm * 3 + ixyz].real() * std::sqrt(system->get_mass_prim()[i_atm]) *
                          init_u0[i_atm * 3 + ixyz];
            }
        }
    }
}

void Relaxation::set_init_u0_from_modes()
{
    // &displace DISPMODE = 2. The eigenvectors of a degenerate subspace S are gauge
    // dependent, so the pattern is defined by the symmetry it keeps instead: the member
    // of S invariant under a proper rotation of the reference cell about `axis`. A polar
    // mode transforms as a vector and an octahedral tilt as an axial vector, so for both
    // the axis component survives while the other components are mixed away. Only the
    // projector E E^T and the restrictions A(g) = E^T T(g) E enter, never a single eigenvector.
    // Runs on rank 0 (init_u0 lives there); needs Fcs_phonon::setup and SymmListWithMap_ref.
    if (mympi->my_rank != 0) return;

    const int natmin = system->get_primcell().number_of_atoms;
    const int ns = 3 * natmin;
    const auto &invsqrt_mass = system->get_invsqrt_mass();
    const double tol_omega = 1.0e-7; // ~0.01 cm^-1, same as degeneracy_utils.h

    Eigen::MatrixXd dymat = Eigen::MatrixXd::Zero(ns, ns);
    for (const auto &it: fcs_phonon->force_constant_with_cell[0]) {
        const int i = it.pairs[0].index, j = it.pairs[1].index;
        dymat(i, j) += it.fcs_val * invsqrt_mass[i / 3] * invsqrt_mass[j / 3];
    }
    dymat = 0.5 * (dymat + dymat.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(dymat);
    const auto &evec = saes.eigenvectors();
    std::vector<double> omega(ns);
    for (int is = 0; is < ns; ++is) omega[is] = dynamical->freq(saes.eigenvalues()[is]);

    // T(g) at Gamma on a set of column vectors: (T v)_{map[j], a} = sum_b R_ab v_{j, b}.
    // The translation part acts through the atom mapping; Bloch phases are unity at Gamma.
    const auto apply_op = [natmin](const SymmetryOperationWithMapping &op, const Eigen::MatrixXd &in) {
        Eigen::Matrix3d rot;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) rot(i, j) = op.rot[3 * i + j];
        Eigen::MatrixXd out(in.rows(), in.cols());
        for (int jat = 0; jat < natmin; ++jat) {
            out.middleRows(3 * op.mapping[jat], 3) = rot * in.middleRows(3 * jat, 3);
        }
        return out;
    };
    // Mass-weighted pattern -> Cartesian, scaled to a largest atomic displacement of one.
    const auto to_cartesian = [&](const Eigen::VectorXd &e) {
        Eigen::VectorXd u(ns);
        double umax = 0.0;
        for (int iat = 0; iat < natmin; ++iat) {
            u.segment<3>(3 * iat) = e.segment<3>(3 * iat) * invsqrt_mass[iat];
            umax = std::max(umax, u.segment<3>(3 * iat).norm());
        }
        return Eigen::VectorXd(u / umax);
    };

    const auto verbosity = writes->getVerbosity();
    if (verbosity > 0) {
        std::cout << " Initial displacements from normal modes (&displace, DISPMODE = 2):\n";
    }

    init_u0.assign(ns, 0.0);

    for (const auto &m: init_disp_modes) {
        if (m.branch >= ns) exit("set_init_u0_from_modes", "Branch index in &displace exceeds 3 * natmin.");

        int lo = m.branch, hi = m.branch;
        while (lo > 0 && std::abs(omega[lo - 1] - omega[lo]) < tol_omega) --lo;
        while (hi < ns - 1 && std::abs(omega[hi + 1] - omega[hi]) < tol_omega) ++hi;
        const int ndeg = hi - lo + 1;
        const Eigen::MatrixXd sub = evec.middleCols(lo, ndeg);

        Eigen::Vector3d axis(m.axis[0], m.axis[1], m.axis[2]);
        Eigen::VectorXd coef;
        std::string axis_info;

        if (ndeg == 1) {
            coef = Eigen::VectorXd::Ones(1); // the axis only matters inside a degenerate set
        } else {
            if (axis.norm() < eps) {
                exit("set_init_u0_from_modes",
                     "The branch given in &displace is degenerate; add the symmetry axis (nx ny nz) kept by the"
                     " distortion.");
            }
            axis.normalize();

            // Candidates: proper rotations g about the axis. In a supercell the same rotation
            // exists with several translation parts (axes through different points), and the ones
            // shifted by a lattice vector of the parent cell flip a folded mode, so invariance is
            // required under one rotation only. Every candidate that pins a unique vector is kept;
            // distinct results are accepted when they are symmetry-equivalent domains.
            std::vector<Eigen::VectorXd> found;
            std::vector<std::string> found_info;
            int nrot = 0;
            bool any_higher = false;
            for (const auto &op: symmetry->SymmListWithMap_ref) {
                Eigen::Matrix3d rot;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) rot(i, j) = op.rot[3 * i + j];
                if (rot.determinant() < 0.0 || (rot * axis - axis).norm() > 1.0e-6 ||
                    (rot - Eigen::Matrix3d::Identity()).norm() < 1.0e-6)
                    continue;
                ++nrot;
                const Eigen::MatrixXd tsub = apply_op(op, sub);
                const Eigen::MatrixXd M = sub.transpose() * tsub; // A(g) restricted to S
                const double leak = (tsub - sub * M).norm() / sub.norm();
                if (leak > 1.0e-5) {
                    exit("set_init_u0_from_modes",
                         "The degenerate set of the branch given in &displace is not closed under a symmetry"
                         " operation of the reference cell: the set is incomplete (check the harmonic"
                         " frequencies at Gamma) or the force constants break the symmetry.");
                }
                const Eigen::MatrixXd D = Eigen::MatrixXd::Identity(ndeg, ndeg) - M;
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> inv(D.transpose() * D);
                int ninv = 0;
                while (ninv < ndeg && inv.eigenvalues()[ninv] < 1.0e-8) ++ninv;
                if (ninv > 1) any_higher = true;
                if (ninv != 1) continue;
                const Eigen::VectorXd c = inv.eigenvectors().col(0);
                bool is_new = true;
                for (const auto &f: found) {
                    if (std::abs(f.dot(c)) > 1.0 - 1.0e-6) is_new = false;
                }
                if (!is_new) continue;
                found.push_back(c);
                const auto angle = std::acos(std::clamp(0.5 * (rot.trace() - 1.0), -1.0, 1.0)) * 180.0 / pi;
                std::ostringstream info;
                info << "rotation by " << std::lround(angle) << " deg, translation " << std::fixed
                     << std::setprecision(4) << op.shift[0] << " " << op.shift[1] << " " << op.shift[2];
                found_info.push_back(info.str());
            }
            if (nrot == 0) {
                exit("set_init_u0_from_modes",
                     "The reference structure has no proper rotation about the axis given in &displace.");
            }
            if (found.empty()) {
                exit("set_init_u0_from_modes",
                     any_higher ? "The rotations about the axis given in &displace leave more than one member of the"
                                  " degenerate set invariant; give a higher-order axis."
                                : "No member of the degenerate set is invariant under the rotations about the axis"
                                  " given in &displace; the axis is incompatible with the mode.");
            }
            // Several distinct patterns: fine when they are domains of one phase (an operation of
            // the reference cell maps the first onto each other one); otherwise the user must choose.
            for (std::size_t k = 1; k < found.size(); ++k) {
                const Eigen::VectorXd e0 = sub * found[0], ek = sub * found[k];
                bool equivalent = false;
                for (const auto &op: symmetry->SymmListWithMap_ref) {
                    if (std::abs(apply_op(op, e0).col(0).dot(ek)) > 1.0 - 1.0e-6) {
                        equivalent = true;
                        break;
                    }
                }
                if (equivalent) continue;
                std::cout << "\n Rotations about the axis given in &displace through different points select"
                             " physically distinct patterns.\n Choose one and give it explicitly with"
                             " DISPMODE = 1 (largest displacement scaled to 1 Bohr):\n";
                for (std::size_t l = 0; l < found.size(); ++l) {
                    const Eigen::VectorXd u = to_cartesian(sub * found[l]);
                    std::cout << "  candidate " << l + 1 << " (" << found_info[l] << ")\n";
                    for (int iat = 0; iat < natmin; ++iat) {
                        std::cout << "   ";
                        for (int kk = 0; kk < 3; ++kk) {
                            std::cout << std::setw(15) << std::scientific << std::setprecision(6) << u[3 * iat + kk];
                        }
                        std::cout << '\n';
                    }
                }
                std::cout << std::defaultfloat << std::flush;
                exit("set_init_u0_from_modes", "Ambiguous axis position; see the candidate patterns above.");
            }
            coef = found[0];
            axis_info = " (" + found_info[0] + ")";
            if (found.size() > 1) {
                axis_info += ", " + std::to_string(found.size()) + " equivalent domains found, the first is used";
            }
        }

        // Largest component positive, largest atomic displacement equal to |amplitude|.
        const Eigen::VectorXd u = to_cartesian(sub * coef);
        int imax = 0;
        for (int i = 1; i < ns; ++i) {
            if (std::abs(u[i]) > std::abs(u[imax])) imax = i;
        }
        const double sign = u[imax] < 0.0 ? -1.0 : 1.0;
        for (int i = 0; i < ns; ++i) init_u0[i] += sign * m.amplitude * u[i];

        if (verbosity > 0) {
            std::cout << "  branch " << std::setw(3) << m.branch + 1 << " (" << std::fixed << std::setprecision(2)
                      << omega[m.branch] * Ry_to_kayser << " cm^-1, degenerate set " << lo + 1 << "-" << hi + 1
                      << "), amplitude " << std::setprecision(6) << m.amplitude << " Bohr";
            if (ndeg > 1) {
                std::cout << ", invariant under a rotation about (" << std::setprecision(4) << axis[0] << ", "
                          << axis[1] << ", " << axis[2] << ")" << axis_info;
            }
            std::cout << '\n';
        }
    }

    if (verbosity > 0) {
        std::cout << "  Resulting Cartesian displacements [Bohr]:\n";
        for (int iat = 0; iat < natmin; ++iat) {
            std::cout << "   " << std::setw(4) << iat + 1;
            for (int k = 0; k < 3; ++k) {
                std::cout << std::setw(15) << std::scientific << std::setprecision(6) << init_u0[3 * iat + k];
            }
            std::cout << '\n';
        }
        std::cout << std::defaultfloat << '\n';
    }
}

void Relaxation::set_initial_strain(std::array<std::array<double, 3>, 3> &u_tensor) const
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            u_tensor[i][j] = init_u_tensor[i][j];
        }
    }
}

void Relaxation::calculate_u0(const double *const q0, double *const u0, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;

    for (int i_atm = 0; i_atm < natmin; i_atm++) {
        for (int ixyz = 0; ixyz < 3; ixyz++) {
            const auto is = i_atm * 3 + ixyz;
            u0[is] = 0.0;
            for (int is2 = 0; is2 < ns; is2++) {
                if (std::fabs(omega2_harmonic[0][is2]) < eps8) {
                    continue;
                }
                u0[is] += evec_harmonic[0][is2][is].real() * q0[is2];
            }
            u0[is] /= std::sqrt(system->get_mass_prim()[i_atm]);
        }
    }
}

void Relaxation::calculate_u0(const std::vector<double> &q0, std::vector<double> &u0, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;

    if (u0.size() != q0.size()) {
        u0.resize(q0.size());
    }

    for (int i_atm = 0; i_atm < natmin; i_atm++) {
        for (int ixyz = 0; ixyz < 3; ixyz++) {
            const auto is = i_atm * 3 + ixyz;
            u0[is] = 0.0;
            for (int is2 = 0; is2 < ns; is2++) {
                if (std::fabs(omega2_harmonic[0][is2]) < eps8) {
                    continue;
                }
                u0[is] += evec_harmonic[0][is2][is].real() * q0[is2];
            }
            u0[is] /= std::sqrt(system->get_mass_prim()[i_atm]);
        }
    }
}

void Relaxation::update_cell_coordinate(
    RelaxationStructureState &structure_state, const std::complex<double> *const v1_array_atT,
    const double *const *const omega2_array, const std::complex<double> *const del_v0_strain_atT,
    const double *const *const C2_array, const std::complex<double> *const *const *const cmat_convert,
    const std::vector<int> &harm_optical_modes, double **omega2_harmonic, std::complex<double> ***evec_harmonic) const
{
    using namespace Eigen;
    const auto relax_mode = to_relaxation_str_mode(relax_str);

    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &delta_q0 = structure_state.delta_q0;
    auto &delta_u0 = structure_state.delta_u0;
    auto &delta_umn = structure_state.delta_umn;
    auto &du0 = structure_state.du0;
    auto &du_tensor = structure_state.du_tensor;

    const auto ns = dynamical->neval;
    int is;

    MatrixXcd Cmat(ns, ns), v2_mat_full(ns, ns);
    std::vector<double> grad_vec;
    std::vector<double> delta_vec;
    std::vector<double> state_vec;
    std::vector<std::vector<double>> hessian_mat;

    MatrixXcd C2_mat_tmp(6, 6);
    VectorXcd del_v0_strain_vec(6);


    const double add_hess_diag_omega2 = pow2(add_hess_diag / Ry_to_kayser);

    if (optimizes_coordinates_only(relax_mode)) {
        delta_vec.assign(ns - 3, 0.0);
        state_vec.assign(ns - 3, 0.0);
        grad_vec.assign(ns - 3, 0.0);
        hessian_mat.assign(ns - 3, std::vector<double>(ns - 3, 0.0));
    } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        delta_vec.assign(ns + 3, 0.0);
        state_vec.assign(ns + 3, 0.0);
        grad_vec.assign(ns + 3, 0.0);
        hessian_mat.assign(ns + 3, std::vector<double>(ns + 3, 0.0));
    }

    for (is = 0; is < ns; is++) {
        delta_q0[is] = 0.0;
    }
    for (is = 0; is < 6; is++) {
        delta_umn[is] = 0.0;
    }

    if (relax_algo >= 1) {

        // Newton (relax_algo == 2) and BFGS+GDIIS (relax_algo == 3) use the harmonic IFC
        // matrix as the (initial) Hessian. Steepest descent (relax_algo == 1) ignores the
        // Hessian, so skip building it.
        if (relax_algo >= 2) {
            int js;
            // prepare harmonic IFC matrix
            for (is = 0; is < ns; is++) {
                for (js = 0; js < ns; js++) {
                    Cmat(js, is) = cmat_convert[0][is][js]; // transpose
                    v2_mat_full(is, js) = 0.0;
                }
                v2_mat_full(is, is) = omega2_array[0][is];
            }
            v2_mat_full = Cmat.adjoint() * v2_mat_full * Cmat;

            // set hessian
            for (is = 0; is < ns - 3; is++) {
                for (js = 0; js < ns - 3; js++) {
                    hessian_mat[is][js] = v2_mat_full(harm_optical_modes[is], harm_optical_modes[js]).real();
                }
                hessian_mat[is][is] += add_hess_diag_omega2;
            }
        }

        // set gradient and current state over the optical modes
        for (is = 0; is < ns - 3; is++) {
            grad_vec[is] = v1_array_atT[harm_optical_modes[is]].real();
            state_vec[is] = q0[harm_optical_modes[is]];
        }

        // Steepest descent (relax_algo == 1) historically froze near-acoustic modes using a
        // coarser cutoff (|omega2_harmonic| < eps8) than the eps10 used to build
        // harm_optical_modes. Zero their gradient so their steepest-descent step stays
        // exactly zero, reproducing the original behavior.
        if (relax_algo == 1) {
            for (is = 0; is < ns - 3; is++) {
                if (std::fabs(omega2_harmonic[0][harm_optical_modes[is]]) < eps8) {
                    grad_vec[is] = 0.0;
                }
            }
        }


        if (optimizes_coordinates_only(relax_mode) || relax_algo == 1) {
            int i1, i2;
            // Relax the internal coordinates only. This covers CoordinatesOnly mode (any
            // algorithm) and steepest descent (relax_algo == 1), which keeps the cell shape
            // fixed even in CoordinatesAndCell mode.
            optimizer->update_state(ns - 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
                q0[harm_optical_modes[is]] += delta_q0[harm_optical_modes[is]];
            }
            // keep the cell fixed: no strain step
            for (i1 = 0; i1 < 6; i1++) {
                delta_umn[i1] = 0.0;
            }
            // reset the strain to zero only in CoordinatesOnly mode (in CoordinatesAndCell
            // mode the current strain is left untouched)
            if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                for (i1 = 0; i1 < 3; i1++) {
                    for (i2 = 0; i2 < 3; i2++) {
                        u_tensor[i1][i2] = 0.0;
                    }
                }
            }

            // std::cout << "grad_vec = ";
            // for (auto &val: grad_vec) {
            //     std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            // }
            // std::cout << '\n';
            // std::cout << "state_vec = ";
            // for (auto &val: state_vec) {
            //     std::cout << std::setw(15) << std::setprecision(6) << val << ' ';
            // }
            // std::cout << '\n';

        } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;

            // prepare matrix of elastic constants and vector of del_v0_strain_atT
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                del_v0_strain_vec(itmp1) = del_v0_strain_atT[itmp1 * 3 + itmp1];

                itmp2 = (itmp1 + 1) % 3;
                itmp3 = (itmp1 + 2) % 3;
                del_v0_strain_vec(itmp1 + 3) = del_v0_strain_atT[itmp2 * 3 + itmp3];
            }

            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    C2_mat_tmp(itmp1, itmp2) = C2_array[itmp1 * 3 + itmp1][itmp2 * 3 + itmp2];
                }
            }
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    itmp3 = (itmp2 + 1) % 3;
                    itmp4 = (itmp2 + 2) % 3;
                    C2_mat_tmp(itmp1, itmp2 + 3) = 2.0 * C2_array[itmp1 * 3 + itmp1][itmp3 * 3 + itmp4];
                    C2_mat_tmp(itmp2 + 3, itmp1) = C2_array[itmp3 * 3 + itmp4][itmp1 * 3 + itmp1];
                }
            }
            for (itmp1 = 0; itmp1 < 3; itmp1++) {
                for (itmp2 = 0; itmp2 < 3; itmp2++) {
                    itmp3 = (itmp1 + 1) % 3;
                    itmp4 = (itmp1 + 2) % 3;
                    itmp5 = (itmp2 + 1) % 3;
                    itmp6 = (itmp2 + 2) % 3;
                    C2_mat_tmp(itmp1 + 3, itmp2 + 3) = 2.0 * C2_array[itmp3 * 3 + itmp4][itmp5 * 3 + itmp6];
                }
            }

            // write C2mat to hessian matrix
            for (itmp1 = 0; itmp1 < 6; itmp1++) {
                for (itmp2 = 0; itmp2 < 6; itmp2++) {
                    hessian_mat[itmp1 + ns - 3][itmp2 + ns - 3] = C2_mat_tmp(itmp1, itmp2).real();
                }
            }
            // write to grad vector and state vector
            for (itmp1 = 0; itmp1 < 6; itmp1++) {
                grad_vec[itmp1 + ns - 3] = del_v0_strain_vec(itmp1).real();

                if (itmp1 < 3) {
                    state_vec[itmp1 + ns - 3] = u_tensor[itmp1][itmp1];
                } else {
                    itmp2 = (itmp1 + 1) % 3;
                    itmp3 = (itmp1 + 2) % 3;
                    state_vec[itmp1 + ns - 3] = u_tensor[itmp2][itmp3];
                }
            }

            // call optimizer
            optimizer->update_state(ns + 3, grad_vec, state_vec, hessian_mat, delta_vec);

            // update q0
            // std::cout << "update state";
            for (is = 0; is < ns - 3; is++) {
                delta_q0[harm_optical_modes[is]] = delta_vec[is];
                q0[harm_optical_modes[is]] += delta_q0[harm_optical_modes[is]];
            }
            // update u tensor
            for (is = 0; is < 6; is++) {
                delta_umn[is] = delta_vec[is + ns - 3];
                if (is < 3) {
                    u_tensor[is][is] += delta_umn[is];
                } else {
                    itmp1 = (is + 1) % 3;
                    itmp2 = (is + 2) % 3;
                    u_tensor[itmp1][itmp2] += delta_umn[is];
                    u_tensor[itmp2][itmp1] += delta_umn[is];
                }
            }
        }
    }

    calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

    du0 = 0.0;
    calculate_u0(delta_q0, delta_u0, omega2_harmonic, evec_harmonic);
    for (is = 0; is < ns; is++) {
        du0 += delta_u0[is] * delta_u0[is];
    }
    du0 = std::sqrt(du0);

    du_tensor = 0.0;
    for (is = 0; is < 6; is++) {
        du_tensor += delta_umn[is] * delta_umn[is];
        if (is >= 3) {
            du_tensor += delta_umn[is] * delta_umn[is];
        }
    }
    du_tensor = std::sqrt(du_tensor);
}

void Relaxation::rescue_step_after_scp_failure(RelaxationStructureState &structure_state,
                                               const std::complex<double> *const v1_array_atT,
                                               const std::vector<int> &harm_optical_modes, double **omega2_harmonic,
                                               std::complex<double> ***evec_harmonic) const
{
    // On SCP failure, keep unreliable forces/stress out of optimizer history
    // and halve the last structure step. With no previous step, take a strongly
    // damped force step at fixed cell to leave the initial structure.

    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &delta_q0 = structure_state.delta_q0;
    auto &delta_u0 = structure_state.delta_u0;
    auto &delta_umn = structure_state.delta_umn;
    auto &du0 = structure_state.du0;
    auto &du_tensor = structure_state.du_tensor;

    const auto ns = dynamical->neval;
    int is, i1, i2;

    double last_step_norm = 0.0;
    for (is = 0; is < ns; is++) {
        last_step_norm += delta_q0[is] * delta_q0[is];
    }
    for (is = 0; is < 6; is++) {
        last_step_norm += delta_umn[is] * delta_umn[is];
    }
    last_step_norm = std::sqrt(last_step_norm);

    if (last_step_norm > eps12) {
        constexpr double backtrack_ratio = 0.5;
        for (is = 0; is < ns; is++) {
            delta_q0[is] *= -backtrack_ratio;
            q0[is] += delta_q0[is];
        }
        for (is = 0; is < 6; is++) {
            delta_umn[is] *= -backtrack_ratio;
            if (is < 3) {
                u_tensor[is][is] += delta_umn[is];
            } else {
                i1 = (is + 1) % 3;
                i2 = (is + 2) % 3;
                u_tensor[i1][i2] += delta_umn[is];
                u_tensor[i2][i1] += delta_umn[is];
            }
        }
        if (writes->getVerbosity() > 0) {
            std::cout << " Moving back halfway along the last step and retrying.\n";
        }
    } else {
        // No accepted step exists yet: use the unreliable force, but with a very
        // small weight.
        const double alpha_rescue = 0.1 * alpha_steepest_decent;
        for (is = 0; is < ns; is++) {
            delta_q0[is] = 0.0;
        }
        for (const auto mode: harm_optical_modes) {
            delta_q0[mode] = -alpha_rescue * v1_array_atT[mode].real();
            q0[mode] += delta_q0[mode];
        }
        for (is = 0; is < 6; is++) {
            delta_umn[is] = 0.0;
        }
        if (writes->getVerbosity() > 0) {
            std::cout << " No accepted step exists yet: taking a strongly damped steepest-descent\n"
                         " step on the unreliable force (the cell shape is kept fixed).\n";
        }
    }

    calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

    du0 = 0.0;
    calculate_u0(delta_q0, delta_u0, omega2_harmonic, evec_harmonic);
    for (is = 0; is < ns; is++) {
        du0 += delta_u0[is] * delta_u0[is];
    }
    du0 = std::sqrt(du0);

    du_tensor = 0.0;
    for (is = 0; is < 6; is++) {
        du_tensor += delta_umn[is] * delta_umn[is];
        if (is >= 3) {
            du_tensor += delta_umn[is] * delta_umn[is];
        }
    }
    du_tensor = std::sqrt(du_tensor);
}

void Relaxation::distorted_cell_of(const RelaxationStructureState &state, Eigen::Matrix3d &lavec,
                                   std::vector<Eigen::Vector3d> &xf) const
{
    // Lattice vectors (columns) and fractional positions of the structure given by
    // the strain u_tensor and the displacements u0 of the state.
    const auto &primcell = system->get_primcell();
    const auto natmin = primcell.number_of_atoms;
    Eigen::Matrix3d Fmat = Eigen::Matrix3d::Identity();
    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j) Fmat(i, j) += state.u_tensor[i][j];
    lavec = Fmat * primcell.lattice_vector;
    const Eigen::Matrix3d lavec_inv = lavec.inverse();
    xf.resize(natmin);
    for (size_t iat = 0; iat < natmin; ++iat) {
        const Eigen::Vector3d x_ref = primcell.x_fractional.row(iat).transpose();
        Eigen::Vector3d r_cart = Fmat * (primcell.lattice_vector * x_ref);
        for (auto i = 0; i < 3; ++i) r_cart(i) += state.u0[3 * iat + i];
        xf[iat] = lavec_inv * r_cart;
    }
}

int Relaxation::spacegroup_of(const RelaxationStructureState &state) const
{
    Eigen::Matrix3d lavec;
    std::vector<Eigen::Vector3d> xf;
    distorted_cell_of(state, lavec, xf);
    std::string label;
    return detect_spacegroup(lavec, xf, label);
}

int Relaxation::detect_spacegroup(const Eigen::Matrix3d &lavec, const std::vector<Eigen::Vector3d> &xf,
                                  std::string &label) const
{
    // Lattice vectors are columns; positions are fractional, species from the phonon cell.
    // Returns 0 on failure; label includes the symbol and number, e.g. "P4mm (#99)".
    const auto &primcell = system->get_primcell();
    const int natmin = primcell.number_of_atoms;
    double aa[3][3];
    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j) aa[i][j] = lavec(i, j);
    // Kept raw: spg_get_dataset takes double(*)[3] (spglib C ABI boundary).
    double(*position)[3];
    NDArray<int, 1> types;
    allocate(position, natmin);
    types.resize(natmin);
    for (auto iat = 0; iat < natmin; ++iat) {
        for (auto i = 0; i < 3; ++i) position[iat][i] = xf[iat](i);
        types[iat] = primcell.kind[iat];
    }
    int number = 0;
    label = "detection failed";
    const auto spgdataset = spg_get_dataset(aa, position, types, natmin, symmetry->tolerance);
    if (spgdataset && spgdataset->spacegroup_number > 0) {
        number = spgdataset->spacegroup_number;
        label = std::string(spgdataset->international_symbol) + " (#" + std::to_string(number) + ")";
    }
    if (spgdataset) spg_free_dataset(spgdataset);
    deallocate(position);
    return number;
}

std::string Relaxation::print_structure_and_symmetry(const RelaxationStructureState &structure_state,
                                                     const std::complex<double> *del_v0_del_umn_atT) const
{
    // Print the crystal structure described by structure_state (strained lattice
    // and displaced atomic positions), the Cauchy stress tensor and pressure when
    // the strain gradient is available (del_v0_del_umn_atT != nullptr), and the
    // space group detected by spglib. Returns the space group label, e.g.
    // "P4mm (#99)", for use in summary tables.

    using namespace Eigen;

    const auto &primcell = system->get_primcell();
    const auto natmin = primcell.number_of_atoms;

    // deformation gradient F = I + u
    Matrix3d Fmat = Matrix3d::Identity();
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            Fmat(i, j) += structure_state.u_tensor[i][j];
        }
    }

    Matrix3d lavec_new;
    std::vector<Vector3d> xf_new;
    distorted_cell_of(structure_state, lavec_new, xf_new);

    if (writes->getVerbosity() > 0) {
        std::cout << "  Lattice vectors [Bohr]:\n";
        for (auto j = 0; j < 3; ++j) {
            std::cout << "   a" << j + 1 << " :";
            for (auto i = 0; i < 3; ++i) {
                std::cout << std::setw(15) << std::setprecision(8) << std::fixed << lavec_new(i, j);
            }
            std::cout << '\n';
        }
    }

    if (writes->getVerbosity() > 0) {
        std::cout << "  Atomic positions (fractional):\n";
    }
    for (size_t iat = 0; iat < natmin; ++iat) {

        if (writes->getVerbosity() > 0) {
            std::cout << "   " << std::setw(4) << std::left << system->symbol_kd[primcell.kind[iat]] << std::right
                      << " :";
            for (auto i = 0; i < 3; ++i) {
                std::cout << std::setw(15) << std::setprecision(8) << std::fixed << xf_new[iat](i);
            }
            std::cout << '\n';
        }
    }
    if (writes->getVerbosity() > 0) {
        std::cout << std::scientific;
    }

    if (del_v0_del_umn_atT) {
        // Cauchy stress sigma = (dE/dF) F^T / (V_ref det F). del_v0_del_umn_atT is the
        // gradient of the optimized potential including the external-pressure term
        // p V, so the p V gradient p V F^{-T} is subtracted to recover the internal
        // stress. At full equilibrium, the pressure printed here equals the applied
        // pressure (STAT_PRESSURE).
        const auto detF = Fmat.determinant();
        const auto vol = primcell.volume * detF; // Bohr^3
        const auto gpa_to_ry_bohr3 = 1.0e9 / Ryd * std::pow(Bohr_in_Angstrom, 3) * 1.0e-30;
        const auto p_ext = stat_pressure * gpa_to_ry_bohr3; // Ry/Bohr^3

        Matrix3d grad_total;
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                grad_total(i, j) = del_v0_del_umn_atT[3 * i + j].real();
            }
        }
        const Matrix3d grad_internal = grad_total - p_ext * vol * Fmat.inverse().transpose();
        const Matrix3d stress = grad_internal * Fmat.transpose() / vol; // Ry/Bohr^3, tension positive

        const auto pressure_gpa = -stress.trace() / 3.0 / gpa_to_ry_bohr3;

        if (writes->getVerbosity() > 0) {
            std::cout << "  Stress tensor [GPa]:\n";
            for (auto i = 0; i < 3; ++i) {
                std::cout << "   ";
                for (auto j = 0; j < 3; ++j) {
                    std::cout << std::setw(15) << std::setprecision(6) << stress(i, j) / gpa_to_ry_bohr3;
                }
                std::cout << '\n';
            }
            std::cout << "  Pressure :" << std::setw(15) << std::setprecision(6) << pressure_gpa << " [GPa]\n";
        }
    }

    // space group detection by spglib
    std::string spg_label;
    detect_spacegroup(lavec_new, xf_new, spg_label);
    if (writes->getVerbosity() > 0) {
        std::cout << "  Space group :  " << spg_label << '\n';
    }
    return spg_label;
}

void Relaxation::print_optimization_history(const std::vector<StructOptStepRecord> &step_history, const double temp,
                                            const bool with_cell, const bool show_scp_column,
                                            const unsigned int verbosity)
{
    // At-a-glance summary table of a structural optimization at one temperature.
    // show_scp_column controls the column marking whether the SCP equation
    // converged at each step (meaningless for QHA, which has no inner
    // self-consistency loop).

    if (step_history.empty()) return;

    if (verbosity > 0) {
        auto print_rule = [&]() {
            std::cout << " ------------------------------------------------------------";
            if (with_cell) std::cout << "-------------";
            std::cout << '\n';
        };

        std::cout << "\n Optimization history at " << temp << " K :\n";
        print_rule();
        std::cout << "   step  ";
        if (show_scp_column) std::cout << " SCP    ";
        std::cout << "  du0 [Bohr]    du_tensor      |force|   ";
        if (with_cell) std::cout << "    |stress|  ";
        std::cout << "  space group\n";
        print_rule();
        std::cout << std::scientific << std::setprecision(3);
        for (std::size_t istep = 0; istep < step_history.size(); ++istep) {
            const auto &rec = step_history[istep];
            std::cout << std::setw(7) << istep + 1;
            if (show_scp_column) std::cout << (rec.scp_ok ? "   conv " : "   FAIL ");
            std::cout << std::setw(14) << rec.du0 << std::setw(13) << rec.du_tensor;
            if (rec.grad_norm >= 0.0) {
                std::cout << std::setw(13) << rec.grad_norm;
            } else {
                std::cout << std::setw(13) << "-";
            }
            if (with_cell) {
                if (rec.cell_grad_norm >= 0.0) {
                    std::cout << std::setw(14) << rec.cell_grad_norm;
                } else {
                    std::cout << std::setw(14) << "-";
                }
            }
            std::cout << "    " << rec.spacegroup << '\n';
        }
        print_rule();
    }
}

void Relaxation::check_str_divergence(int &diverged, const RelaxationStructureState &structure_state) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;

    int i, j;

    int flag_diverged = 0;

    for (i = 0; i < static_cast<int>(q0.size()); i++) {
        if (!std::isfinite(q0[i]) || !std::isfinite(u0[i])) {
            flag_diverged = 1;
            break;
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (!std::isfinite(u_tensor[i][j])) {
                flag_diverged = 1;
                i = 3;
                break;
            }
        }
    }

    diverged = flag_diverged;
}


void Relaxation::compute_del_v_strain(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                      DelVStrainData &del_v_strain, double **omega2_harmonic,
                                      std::complex<double> ***evec_harmonic, const RelaxationStrMode relax_mode,
                                      MinimumDistList ***mindist_list, const PhaseFactorCache *phase_cache_in)
{
    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense->nk;

    derivative_ifc->set_verbosity(writes->getVerbosity());

    // CoordinatesOnly: keep the unit cell fixed and relax internal coordinates
    // set renormalization from strain as zero
    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
        derivative_ifc->set_del_v_fixed_cell(nk, ns, del_v_strain);
        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // CoordinatesAndCell: relax both the cell shape and the internal coordinates.
    // CoordinatesAtFixedStrain: the cell is held at a nonzero strain, so it needs the
    // same strain derivatives even though the cell is not a variable of the optimization.
    if (uses_full_strain_derivatives(relax_mode)) {

        derivative_ifc->set_del_v_relax_cell(kmesh_coarse,
                                             kmesh_dense,
                                             ns,
                                             del_v_strain,
                                             omega2_harmonic,
                                             evec_harmonic,
                                             renorm_2to1st,
                                             renorm_34to1st,
                                             renorm_3to2nd,
                                             strain_source(),
                                             mindist_list,
                                             phase_cache_in);

        if (mympi->my_rank == 0) timer->print_elapsed();

        return;
    }

    // PerturbativeQha: calculate lowest-order linear equation of QHA.
    if (relax_mode == RelaxationStrMode::PerturbativeQha) {

        derivative_ifc->set_del_v_relax_cell_linearQHA(kmesh_coarse,
                                                       kmesh_dense,
                                                       ns,
                                                       del_v_strain,
                                                       omega2_harmonic,
                                                       evec_harmonic,
                                                       renorm_2to1st,
                                                       renorm_34to1st,
                                                       renorm_3to2nd,
                                                       strain_source(),
                                                       mindist_list);

        if (mympi->my_rank == 0) timer->print_elapsed();
    }
}

void Relaxation::renormalize_v0_from_umn(double &v0_with_umn, double v0_ref,
                                         std::array<std::array<double, 3>, 3> &eta_tensor, double *C1_array,
                                         double **C2_array, double ***C3_array,
                                         const std::array<std::array<double, 3>, 3> &u_tensor, const double pvcell)
{
    // This function computes the total enthalpy change induced by the strain.
    // The inputs are the eta_tensor (the Green-Lagrange strain tensor
    // eta = sym(u) + 1/2 u u^T, see calculate_eta_tensor) and u_tensor.
    // The elastic constants C1, C2, and C3 are the Brugger constants (derivatives w.r.t. eta)
    // multiplied by the cell volume; they give the enthalpy change up to the third order in strain.
    int ixyz1;

    constexpr double factor1 = 0.5;
    constexpr double factor2 = 1.0 / 6.0;

    v0_with_umn = v0_ref;

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            v0_with_umn += C1_array[ixyz1 * 3 + ixyz2] * eta_tensor[ixyz1][ixyz2];
            for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                for (int ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                    v0_with_umn += factor1 * C2_array[ixyz1 * 3 + ixyz2][ixyz3 * 3 + ixyz4] * eta_tensor[ixyz1][ixyz2] *
                                   eta_tensor[ixyz3][ixyz4];
                    for (int ixyz5 = 0; ixyz5 < 3; ixyz5++) {
                        for (int ixyz6 = 0; ixyz6 < 3; ixyz6++) {
                            v0_with_umn += factor2 * C3_array[ixyz1 * 3 + ixyz2][ixyz3 * 3 + ixyz4][ixyz5 * 3 + ixyz6] *
                                           eta_tensor[ixyz1][ixyz2] * eta_tensor[ixyz3][ixyz4] *
                                           eta_tensor[ixyz5][ixyz6];
                        }
                    }
                }
            }
        }
    }

    // add pV term
    Eigen::Vector3d vec_tmp1, vec_tmp2, vec_tmp3;
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        vec_tmp1[ixyz1] = u_tensor[0][ixyz1];
        vec_tmp2[ixyz1] = u_tensor[1][ixyz1];
        vec_tmp3[ixyz1] = u_tensor[2][ixyz1];
    }
    vec_tmp1[0] += 1.0;
    vec_tmp2[1] += 1.0;
    vec_tmp3[2] += 1.0;

    const double det_F_tensor = std::abs(vec_tmp1.dot(vec_tmp2.cross(vec_tmp3)));
    v0_with_umn += pvcell * det_F_tensor;
}

void Relaxation::renormalize_v1_from_umn(std::complex<double> *v1_with_umn, const std::complex<double> *const v1_ref,
                                         const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
{
    const auto ns = dynamical->neval;

    constexpr double factor1 = 0.5;
    constexpr double factor2 = 1.0 / 6.0;

    for (int is = 0; is < ns; is++) {
        // original 1st-order IFCs
        v1_with_umn[is] = v1_ref[is];

        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                // renormalization from harmonic IFCs
                v1_with_umn[is] += del_v_strain.del_v1(ixyz1 * 3 + ixyz2, is) * u_tensor[ixyz1][ixyz2];

                for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    for (int ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                        // renormalization from cubic IFCs
                        int ixyz_comb = ixyz1 * 27 + ixyz2 * 9 + ixyz3 * 3 + ixyz4;
                        v1_with_umn[is] += factor1 * del_v_strain.del2_v1(ixyz_comb, is) * u_tensor[ixyz1][ixyz2] *
                                           u_tensor[ixyz3][ixyz4];

                        for (int ixyz5 = 0; ixyz5 < 3; ixyz5++) {
                            for (int ixyz6 = 0; ixyz6 < 3; ixyz6++) {
                                // renormalization from quartic IFCs
                                ixyz_comb = ixyz1 * 243 + ixyz2 * 81 + ixyz3 * 27 + ixyz4 * 9 + ixyz5 * 3 + ixyz6;
                                v1_with_umn[is] += factor2 * del_v_strain.del3_v1(ixyz_comb, is) *
                                                   u_tensor[ixyz1][ixyz2] * u_tensor[ixyz3][ixyz4] *
                                                   u_tensor[ixyz5][ixyz6];
                            }
                        }
                    }
                }
            }
        }
    }
}

void Relaxation::renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse,
                                         const std::vector<int> &kmap_coarse_to_dense,
                                         std::complex<double> **delta_v2_renorm, const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
{
    // This computes the renormalization of the 2nd-order IFCs due to strain, using the 3rd- and 4th-order IFCs.
    // The result is stored in delta_v2_renorm, which is a 2D array of size nk_interpolate x (ns*ns), where nk_interpolate is the number of k-points in the coarse mesh and ns is the number of phonon modes.
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    unsigned int ik, knum;
    unsigned int is1, is2;
    int ixyz1, ixyz2;
    int ixyz, ixyz11, ixyz12, ixyz21, ixyz22, itmp;

    const auto ns2 = ns * ns;
    const auto nkns2 = nk_interpolate * ns2;

#pragma omp parallel for private(ik, is1, is2, knum, ixyz1, ixyz2, ixyz, ixyz11, ixyz12, ixyz21, ixyz22, itmp)
    for (int iks = 0; iks < nkns2; ++iks) {
        ik = iks / ns2;
        is1 = (iks % ns2) / ns;
        is2 = iks % ns;

        knum = kmap_coarse_to_dense[ik];

        // initialize delta_v2_renorm
        delta_v2_renorm[ik][is1 * ns + is2] = 0.0;

        // renormalization from cubic IFCs
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                delta_v2_renorm[ik][is1 * ns + is2] +=
                    del_v_strain.del_v2[ixyz1 * 3 + ixyz2](knum, is1 * ns + is2) * u_tensor[ixyz1][ixyz2];
            }
        }

        // renormalization from quartic IFCs
        for (ixyz = 0; ixyz < 81; ixyz++) {
            itmp = ixyz;
            ixyz22 = itmp % 3;
            itmp /= 3;
            ixyz21 = itmp % 3;
            itmp /= 3;
            ixyz12 = itmp % 3;
            ixyz11 = itmp / 3;

            delta_v2_renorm[ik][is1 * ns + is2] += 0.5 * del_v_strain.del2_v2[ixyz](knum, is1 * ns + is2) *
                                                   u_tensor[ixyz11][ixyz12] * u_tensor[ixyz21][ixyz22];
        }
    }
}

void Relaxation::renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                         std::complex<double> ***v3_with_umn, std::complex<double> ***v3_ref,
                                         const DelVStrainData &del_v_strain,
                                         const std::array<std::array<double, 3>, 3> &u_tensor) const
{
    const auto nk_scph = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int ik;
    unsigned int is1, is2, is3;
    unsigned int ixyz1, ixyz2;

    const auto ns2 = ns * ns;
    const auto ns3 = ns * ns2;
    const auto nkns3 = nk_scph * ns3;

#pragma omp parallel for private(ik, is1, is2, is3, ixyz1, ixyz2)
    for (int iks = 0; iks < nkns3; ++iks) {
        ik = iks / ns3;
        is1 = (iks % ns3) / ns2;
        is2 = (iks % ns2) / ns;
        is3 = iks % ns;

        // initialize v3_with_umn
        v3_with_umn[ik][is1][is2 * ns + is3] = v3_ref[ik][is1][is2 * ns + is3];

        // renormalization from cubic IFCs
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                v3_with_umn[ik][is1][is2 * ns + is3] +=
                    del_v_strain.del_v3[ixyz1 * 3 + ixyz2][ik](is1, is2 * ns + is3) * u_tensor[ixyz1][ixyz2];
            }
        }
    }
}

void Relaxation::renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense,
                                        std::complex<double> *v1_renorm, std::complex<double> *v1_ref,
                                        std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                        const std::complex<double> *const *q4_gamma,
                                        const std::vector<double> &q0) const
{
    int is1;
    const auto ns = dynamical->neval;
    const auto factor = 0.5 * 4.0 * kmesh_dense->nk;
    const auto factor2 = 1.0 / 6.0 * 4.0 * kmesh_dense->nk;

    // w[is1*ns+is2] = q0[is1] q0[is2] (real; Eigen's dot conjugates its left operand, which is
    // harmless for the real w, so w is the left operand and the complex v3 row the right one)
    Eigen::VectorXcd w(ns * ns);
    for (is1 = 0; is1 < ns; is1++) {
        for (int is2 = 0; is2 < ns; is2++) {
            w(is1 * ns + is2) = std::complex<double>(q0[is1] * q0[is2], 0.0);
        }
    }

    // renormalize v1 array
#pragma omp parallel for private(is1)
    for (int is = 0; is < ns; is++) {

        v1_renorm[is] = v1_ref[is];
        v1_renorm[is] += omega2_harmonic[0][is] * q0[is]; // original v2 is assumed to be diagonal

        for (is1 = 0; is1 < ns; is1++) {
            v1_renorm[is] += delta_v2_array_original[0][is * ns + is1] * q0[is1];
        }

        // cubic term sum_{is1,is2} v3[Gamma][is][is1,is2] q0[is1] q0[is2] = (row is of V3) . w
        // (w on the conjugating side of dot(): it is real, the v3 row must not be conjugated)
        v1_renorm[is] += factor * w.dot(Eigen::Map<const Eigen::VectorXcd>(v3_ref[0][is], ns * ns));

        // quartic term sum_{is1,is2,is3} v4[Gamma][is,is1][is2,is3] q0[is1] q0[is2] q0[is3],
        // with the (is2,is3) sum precomputed in q4_gamma (q0_contraction.h).
        // the factor 4.0 appears due to the definition of v4_array = 1.0/(4.0*N_scph) Phi_4
        for (is1 = 0; is1 < ns; is1++) {
            v1_renorm[is] += factor2 * q4_gamma[is][is1] * q0[is1];
        }
    }
}

void Relaxation::renormalize_v2_from_q0(std::complex<double> ***evec_harmonic, const KpointMeshUniform *kmesh_coarse,
                                        const KpointMeshUniform *kmesh_dense,
                                        const std::vector<int> &kmap_coarse_to_dense,
                                        std::complex<double> ****mat_transform_sym,
                                        std::complex<double> **delta_v2_renorm,
                                        std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                        const std::complex<double> *const *const *q4_q0,
                                        const std::vector<double> &q0) const
{
    using namespace Eigen;
    using MatrixXcdRowMajor = Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor>;

    int ik;
    int is1, is2;
    unsigned int knum, knum_interpolate;
    const auto nk_scph = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto factor = 4.0 * nk_scph;
    const auto factor2 = 4.0 * nk_scph * 0.5;

    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    NDArray<std::complex<double>, 3> dymat_q;
    MatrixXcd Dymat(ns, ns);
    MatrixXcd evec_tmp(ns, ns);
    dymat_q.resize(ns, ns, nk_interpolate);

    VectorXcd q0c(ns);
    for (is1 = 0; is1 < ns; is1++) {
        q0c(is1) = std::complex<double>(q0[is1], 0.0);
    }

    for (ik = 0; ik < nk_irred_interpolate; ik++) {

        knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        knum = kmap_coarse_to_dense[knum_interpolate];

        // calculate renormalization
        // cubic: t[is2*ns+is1] = sum_js1 v3[knum][js1][is2,is1] q0[js1]  (one GEMV V3^T q0)
        const VectorXcd t = Map<const MatrixXcdRowMajor>(v3_ref[knum][0], ns, ns * ns).transpose() * q0c;
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                Dymat(is1, is2) = factor * t(is2 * ns + is1);
                // quartic renormalization sum_{js1,js2} v4[ik][is1,is2][js1,js2] q0[js1] q0[js2],
                // precomputed in q4_q0 (q0_contraction.h)
                Dymat(is1, is2) += factor2 * q4_q0[ik][is1][is2];
            }
        }

        // unitary transform Dymat
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                evec_tmp(is1, is2) = evec_harmonic[knum][is2][is1]; // transpose
            }
        }
        Dymat = evec_tmp * Dymat * evec_tmp.adjoint();

        // symmetrize dynamical matrix
        symmetrize_dynamical_matrix(ik, kmesh_coarse, ns, mat_transform_sym, Dymat);

        // store to dymat_q
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                dymat_q[is1][is2][knum_interpolate] = Dymat(is1, is2);
            }
        }
    }

    // replicate dymat_q to all q
    replicate_dymat_for_all_kpoints(kmesh_coarse, ns, mat_transform_sym, dymat_q);

    // copy to delta_v2_renorm
    for (ik = 0; ik < nk_interpolate; ik++) {
        knum = kmap_coarse_to_dense[ik];
        // unitary transform Dymat
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                Dymat(is1, is2) = dymat_q[is1][is2][ik];
                evec_tmp(is1, is2) = evec_harmonic[knum][is2][is1]; // transpose
            }
        }
        Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                delta_v2_renorm[ik][is1 * ns + is2] = Dymat(is1, is2);
            }
        }
    }

    for (ik = 0; ik < nk_interpolate; ik++) {
        for (is1 = 0; is1 < ns * ns; is1++) {
            delta_v2_renorm[ik][is1] += delta_v2_array_original[ik][is1];
        }
    }

    dymat_q.clear();
}

void Relaxation::renormalize_v0_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense,
                                        double &v0_renorm, double v0_ref, std::complex<double> *v1_ref,
                                        std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                        const std::complex<double> *const *q4_gamma,
                                        const std::vector<double> &q0) const
{
    int is1, is2;
    const auto ns = dynamical->neval;
    const auto nk_scph = kmesh_dense->nk;
    constexpr double factor2 = 1.0 / 2.0;
    const double factor3 = 1.0 / 6.0 * 4.0 * nk_scph;
    const double factor4 = 1.0 / 24.0 * 4.0 * nk_scph;

    std::complex<double> v0_renorm_tmp = v0_ref;
    // renormalize from the 1st order, harmonic IFC
    for (is1 = 0; is1 < ns; is1++) {
        v0_renorm_tmp += v1_ref[is1] * q0[is1];
        v0_renorm_tmp += factor2 * omega2_harmonic[0][is1] * q0[is1] * q0[is1]; // original v2 is assumed to be diagonal
    }
    // renormalize from the delta_v2_array
    for (is1 = 0; is1 < ns; is1++) {
        for (is2 = 0; is2 < ns; is2++) {
            v0_renorm_tmp += factor2 * delta_v2_array_original[0][is1 * ns + is2] * q0[is1] * q0[is2];
        }
    }
    // renormalize from the cubic IFC: sum_{is1,is2,is3} v3[Gamma][is1][is2,is3] q0 q0 q0 = q0^T (V3 w)
    {
        using MatrixXcdRowMajor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::VectorXcd q0c(ns), w(ns * ns);
        for (is1 = 0; is1 < ns; is1++) {
            q0c(is1) = std::complex<double>(q0[is1], 0.0);
            for (is2 = 0; is2 < ns; is2++) {
                w(is1 * ns + is2) = std::complex<double>(q0[is1] * q0[is2], 0.0);
            }
        }
        const Eigen::VectorXcd v3w = Eigen::Map<const MatrixXcdRowMajor>(v3_ref[0][0], ns, ns * ns) * w;
        v0_renorm_tmp += factor3 * q0c.dot(v3w);
    }
    // renormalize from the quartic IFC: sum_{is3,is4} v4[Gamma][is2,is1][is3,is4] q0[is3] q0[is4] = q4_gamma[is2][is1]
    // (q0_contraction.h)
    for (is1 = 0; is1 < ns; is1++) {
        for (is2 = 0; is2 < ns; is2++) {
            v0_renorm_tmp += factor4 * q4_gamma[is2][is1] * q0[is1] * q0[is2];
        }
    }

    v0_renorm = v0_renorm_tmp.real();
}

void Relaxation::write_resfile_header(std::ofstream &fout_q0, std::ofstream &fout_u0,
                                      std::ofstream &fout_u_tensor) const
{
    const auto ns = dynamical->neval;

    int is1, iat1, ixyz1, ixyz2;
    std::string str_tmp, str_tmp2;

    // atomic displacement (normal coordinate)
    fout_q0 << "#";
    fout_q0 << std::setw(14) << "temp [K]";
    for (is1 = 0; is1 < ns; is1++) {
        fout_q0 << std::setw(15) << ("q_{" + std::to_string(is1) + "}");
    }
    fout_q0 << '\n';

    // atomic displacement
    fout_u0 << "#";
    fout_u0 << std::setw(14) << "temp [K]";
    for (iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            get_xyz_string(ixyz1, str_tmp);
            fout_u0 << std::setw(15) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
        }
    }
    fout_u0 << '\n';

    // if the cell shape is relaxed
    if (fout_u_tensor) {
        fout_u_tensor << "#";
        fout_u_tensor << std::setw(14) << "temp [K]";
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                get_xyz_string(ixyz1, str_tmp);
                get_xyz_string(ixyz2, str_tmp2);
                fout_u_tensor << std::setw(15) << ("u_{" + str_tmp + str_tmp2 + "}");
            }
        }
        fout_u_tensor << '\n';
    }
}

void Relaxation::write_resfile_atT(const RelaxationStructureState &structure_state, const double temp,
                                   std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;
    int is;

    if (fout_q0) {
        fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < static_cast<int>(q0.size()); is++) {
            fout_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_q0 << '\n';
    }

    if (fout_u0) {
        fout_u0 << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < static_cast<int>(u0.size()); is++) {
            fout_u0 << std::scientific << std::setw(15) << std::setprecision(6) << u0[is];
        }
        fout_u0 << '\n';
    }

    if (fout_u_tensor) {
        fout_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << temp;
        for (is = 0; is < 9; is++) {
            fout_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[is / 3][is % 3];
        }
        fout_u_tensor << '\n';
    }
}


void Relaxation::write_stepresfile_header_atT(std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                              std::ofstream &fout_step_u_tensor, const double temp) const
{
    const auto ns = dynamical->neval;

    int ixyz1;
    std::string str_tmp;

    if (fout_step_q0) {
        fout_step_q0 << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp << " K\n";
        fout_step_q0 << std::setw(6) << "step";
        for (int is1 = 0; is1 < ns; is1++) {
            fout_step_q0 << std::setw(15) << ("q_{" + std::to_string(is1) + "}");
        }
        fout_step_q0 << '\n';
    }

    if (fout_step_u0) {
        fout_step_u0 << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp << " K\n";
        fout_step_u0 << std::setw(6) << "step";
        for (int iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                get_xyz_string(ixyz1, str_tmp);
                fout_step_u0 << std::setw(15) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
            }
        }
        fout_step_u0 << '\n';
    }

    if (fout_step_u_tensor) {
        std::string str_tmp2;
        fout_step_u_tensor << "Temperature :" << std::scientific << std::setw(15) << std::setprecision(6) << temp
                           << " K\n";
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                get_xyz_string(ixyz1, str_tmp);
                get_xyz_string(ixyz2, str_tmp2);
                fout_step_u_tensor << std::setw(15) << ("u_{" + str_tmp + str_tmp2 + "}");
            }
        }
        fout_step_u_tensor << '\n';
    }
}

void Relaxation::write_stepresfile(const RelaxationStructureState &structure_state, const int i_str_loop,
                                   std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                   std::ofstream &fout_step_u_tensor) const
{
    const auto &q0 = structure_state.q0;
    const auto &u0 = structure_state.u0;
    const auto &u_tensor = structure_state.u_tensor;

    int is, i1;

    if (fout_step_q0) {
        fout_step_q0 << std::setw(6) << i_str_loop;
        for (is = 0; is < static_cast<int>(q0.size()); is++) {
            fout_step_q0 << std::scientific << std::setw(15) << std::setprecision(6) << q0[is];
        }
        fout_step_q0 << '\n';
    }

    if (fout_step_u0) {
        fout_step_u0 << std::setw(6) << i_str_loop;
        for (is = 0; is < static_cast<int>(u0.size()); is++) {
            fout_step_u0 << std::scientific << std::setw(15) << std::setprecision(6) << u0[is];
        }
        fout_step_u0 << '\n';
    }

    if (fout_step_u_tensor) {
        fout_step_u_tensor << std::setw(6) << i_str_loop;
        for (i1 = 0; i1 < 9; i1++) {
            fout_step_u_tensor << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[i1 / 3][i1 % 3];
        }
        fout_step_u_tensor << '\n';
    }
}

int Relaxation::get_xyz_string(const int ixyz, std::string &xyz_str)
{
    if (ixyz == 0) {
        xyz_str = "x";
    } else if (ixyz == 1) {
        xyz_str = "y";
    } else {
        xyz_str = "z";
    }
    return 0;
}

void Relaxation::calculate_eta_tensor(std::array<std::array<double, 3>, 3> &eta_tensor,
                                      const std::array<std::array<double, 3>, 3> &u_tensor)
{
    // Green-Lagrange strain for the deformation gradient F = I + u:
    //   eta = 1/2 (F^T F - I) = sym(u) + 1/2 u^T u.
    // The deformation variables used in the relaxation are symmetric, so u u^T = u^T u.
    // The factor 1/2 of the quadratic term is consistent with the chain rule
    // d(eta)/d(u) used in ScphQhaCommon::calculate_del_v0_del_umn_renorm and Qha.
    for (auto i1 = 0; i1 < 3; i1++) {
        for (auto i2 = 0; i2 < 3; i2++) {
            eta_tensor[i1][i2] = 0.5 * (u_tensor[i1][i2] + u_tensor[i2][i1]);
            for (auto j = 0; j < 3; j++) {
                eta_tensor[i1][i2] += 0.5 * u_tensor[i1][j] * u_tensor[i2][j];
            }
        }
    }
}

void Relaxation::setInitialDistortion(const double (*u_tensor_in)[3])
{
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            init_u_tensor[i][j] = u_tensor_in[i][j];
        }
    }
}
