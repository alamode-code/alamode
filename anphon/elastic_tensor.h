/*
 elastic_tensor.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <string>
#include <vector>
#include "fcs_phonon.h"
#include "ndarray.h"

namespace PHON_NS
{

class System;

// Elastic-constant utilities: parsers of the user-provided elastic constants
// used by the SCPH/QHA structural relaxation, and the clamped-ion (Born
// long-wave) stress-energy and elastic tensors computed from the harmonic
// IFCs. All dependencies are explicit constructor arguments (no Pointers
// base); the file parsers are static.
class ElasticTensor
{
public:
    explicit ElasticTensor(const System &system_in);
    ~ElasticTensor() = default;

    // ---- Parsers of the user-provided elastic constants ----

    // Read the first-order coefficients (stress tensor at the reference
    // structure, 9 entries) from "C1_array.in" in the working directory.
    // A missing file is not an error: C1 is set to zero with a warning.
    static void read_C1_array(double *C1_array);

    // Read the second- and third-order elastic constants (9x9 and 9x9x9,
    // row-major in the strain components mu*3+nu) from
    // strain_ifc_dir + "elastic_constants.in".
    static void read_elastic_constants(double *const *C2_array, double *const *const *C3_array,
                                       const std::string &strain_ifc_dir);

    // Positive-definite dummy elastic constants for fixed-cell relaxation
    // (only the coordinates are optimized, so C never enters physically).
    static void set_dummy_elastic_constants(double *C1_array, double *const *C2_array, double *const *const *C3_array);

    // ---- Clamped-ion elastic tensor from harmonic IFCs ----

    // The Born-Huang long-wave brackets [ab, cd]:
    // A(a, b, c, d) = -1/2 sum_{entries} Phi_{ab}(0 kappa; l' kappa') r_c r_d, in Ry,
    // with r = r(l' kappa') - r(0 kappa).
    void calc_longwave_brackets(const std::vector<FcsArrayWithCell> &fcs_in, NDArray<double, 4> &ret) const;

    // Clamped-ion (Born) elastic tensor C_{abcd} = A_{acbd} + A_{bcad} - A_{abcd},
    // converted to GPa with the primitive-cell volume. The inner-displacement
    // (internal-strain) relaxation is NOT included.
    void calc_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic, NDArray<double, 4> &C_gpa) const;

    // Internal-strain (sublattice displacement) response in real space:
    // X(I, mu*3+nu) is the Cartesian displacement (bohr) of primitive
    // atom-coordinate I = 3*kappa+lambda per unit strain eta_{mu nu},
    // X = -K^+ Lambda with K the zone-center harmonic matrix and Lambda the
    // force-strain coupling (symmetrized over the strain indices). The
    // pseudoinverse removes the acoustic translations.
    void calc_sublattice_response(const std::vector<FcsArrayWithCell> &fcs_harmonic, Eigen::MatrixXd &X) const;

    // Relaxed-ion elastic tensor: the clamped-ion tensor plus the
    // internal-strain (sublattice relaxation) correction
    //   C^rel_ab = C^cl_ab - (1/Vcell) Lambda^T K^+ Lambda,
    // evaluated entirely in real space from the harmonic IFCs (the masses of
    // the equivalent mode-basis form cancel; no eigenvectors are needed).
    void calc_elastic_tensor_relaxed(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                                     NDArray<double, 4> &C_gpa) const;

    // Print the brackets A [Ry], the clamped-ion C [GPa], and the Voigt bulk modulus.
    void print_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic) const;

private:
    // Force-strain coupling Lambda(I, mu*3+nu) (symmetrized over the strain
    // indices) and the sublattice response X = -K^+ Lambda, both in real space.
    void calc_force_strain_coupling(const std::vector<FcsArrayWithCell> &fcs_harmonic, Eigen::MatrixXd &Lambda,
                                    Eigen::MatrixXd &X) const;

    const System &system_;
};

} // namespace PHON_NS
