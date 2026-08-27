/*
 elastic_tensor.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <array>
#include <string>
#include <vector>
#include "fcs_phonon.h"
#include "ndarray.h"

namespace PHON_NS
{

class System;

// Fixed-size 3^6 tensor for the third-order elastic quantities.
struct Tensor6
{
    std::array<double, 729> data{};

    double &operator()(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6)
    {
        return data[((((i1 * 3 + i2) * 3 + i3) * 3 + i4) * 3 + i5) * 3 + i6];
    }

    double operator()(const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) const
    {
        return data[((((i1 * 3 + i2) * 3 + i3) * 3 + i4) * 3 + i5) * 3 + i6];
    }

    void setZero()
    {
        data.fill(0.0);
    }
};

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
    // (internal-strain) relaxation is NOT included. symmetrize applies the
    // intrinsic index-symmetry projection (minor + pair exchange); it is a
    // no-op when the IFCs satisfy the rotational invariance.
    void calc_elastic_tensor(const std::vector<FcsArrayWithCell> &fcs_harmonic, NDArray<double, 4> &C_gpa,
                             bool symmetrize = true) const;

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

    // ---- Third-order elastic tensor from cubic IFCs (Wallace, Ch. 8) ----

    // Wallace's restricted surface-free third-order coefficient
    // A_hat(mu1, mu2, nu1, nu2, mu3, nu3) = A^hat_{mu1 mu2, nu1 nu2; mu3 nu3}
    // [Eq. (8.42) plus the XRR/XXR/XXX internal-strain groups of Eq. (8.41)],
    // in Ry per primitive cell. (mu1, mu2) are the force components of the
    // first two IFC legs, (nu1, nu2) their symmetrized position indices, and
    // (mu3, nu3) the untouched third displacement-gradient pair. Pass an
    // empty X for the clamped-ion path.
    void calc_longwave_brackets3(const std::vector<FcsArrayWithCell> &fcs_cubic, const Eigen::MatrixXd &X,
                                 Tensor6 &A_hat) const;

    // Third-order (finite-strain) elastic tensor C3_{ij kl mn} in GPa via
    // Wallace's Eq. (8.14), combining the restricted brackets with the
    // second-order elastic tensor of the same (clamped or relaxed) path.
    // symmetrize selects the final projection onto the exact elastic index
    // symmetries (minor symmetry within each pair and permutations of the
    // three pairs; 48 operations). With rotationally invariant IFCs the
    // projection is a no-op; for fitted IFCs (which generally violate the
    // cubic rotational invariance) it returns the nearest (least-squares)
    // tensor with the exact symmetries. Note that the violated invariance
    // relations are NOT index permutations of A_hat itself (A_hat comes out
    // exactly symmetric in its own index space), so the projection can only
    // be applied at the C3 level.
    void calc_elastic_tensor3(const std::vector<FcsArrayWithCell> &fcs_harmonic,
                              const std::vector<FcsArrayWithCell> &fcs_cubic, bool relax_ions, Tensor6 &C3_gpa,
                              bool symmetrize = true) const;

    // The 48-element index-symmetry projection described above; returns the
    // largest change applied to any component.
    static double symmetrize_elastic_tensor3(Tensor6 &C3);

    // Intrinsic index-symmetry projection of a second-order elastic tensor
    // (minor symmetry within each pair and pair exchange, 8 operations);
    // returns the largest change applied.
    static double symmetrize_elastic_tensor2(NDArray<double, 4> &C2);

    // ---- Symmetry diagnostics for the user-supplied C1/C2/C3 arrays ----
    // Maximum violation of the intrinsic index symmetries, without modifying
    // the arrays (layouts as in read_C1_array / read_elastic_constants).
    static double stress_tensor_asymmetry(const double *C1_array);
    static double elastic_tensor2_asymmetry(const double *const *C2_array);
    static double elastic_tensor3_asymmetry(const double *const *const *C3_array);

private:
    // Force-strain coupling Lambda(I, mu*3+nu) (symmetrized over the strain
    // indices) and the sublattice response X = -K^+ Lambda, both in real space.
    void calc_force_strain_coupling(const std::vector<FcsArrayWithCell> &fcs_harmonic, Eigen::MatrixXd &Lambda,
                                    Eigen::MatrixXd &X) const;

    const System &system_;
};

} // namespace PHON_NS
