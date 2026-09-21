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
#include "strain_file_parsers.h"

namespace PHON_NS
{

class System;
class Ewald;

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

// Read SCPH/QHA elastic constants and compute clamped-ion stress and
// elastic tensors from harmonic IFCs. strain_file_parsers.h handles
// parsing; this class converts units using the primitive-cell volume.
class ElasticTensor
{
public:
    explicit ElasticTensor(const System &system_in);
    ~ElasticTensor() = default;

    // ---- Readers of the user-provided elastic constants ----
    // Both files store either the legacy V*C in Ry for one specific cell
    // (no unit token, or "Ry") or intensive values in GPa ("GPa" after the
    // section label), which are multiplied by the volume of the current
    // primitive cell here and therefore do not depend on the &cell field.

    // Read the first-order coefficients (stress tensor at the reference
    // structure, 9 entries) from "C1_array.in" in the working directory.
    // A missing file is not an error: C1 is set to zero with a warning.
    void read_C1_array(double *C1_array) const;

    // Read the second- and third-order elastic constants (9x9 and 9x9x9,
    // row-major in the strain components mu*3+nu) from
    // strain_ifc_dir + "elastic_constants.in".
    void read_elastic_constants(double *const *C2_array, double *const *const *C3_array,
                                const std::string &strain_ifc_dir) const;

    // The reference stress (C1) and the elastic constants (C2, C3) from the
    // /Elastic group of the STRAINFILE container. The container stores GPa,
    // so the values are multiplied by the volume of the current primitive
    // cell like a GPa text file. An absent stress gives C1 = 0.
    void set_reference_stress_from_set(const strain_coupling::ElasticSet &set, double *C1_array) const;
    void set_elastic_constants_from_set(const strain_coupling::ElasticSet &set, double *const *C2_array,
                                        double *const *const *C3_array) const;

    // GPa -> Ry per current primitive cell (V0(u) stores V0 * C in Ry).
    double gpa_to_ry_per_cell() const;

    // Bring the `n` values of one section of `filename` to Ry per current
    // primitive cell according to the unit declared for that section (GPa
    // values are multiplied by the cell volume; legacy V*X in Ry is used as
    // is). Returns true when the legacy, cell-bound convention was used.
    bool convert_to_ry_per_cell(const char *filename, const char *section, strain_parsers::ElasticUnit unit,
                                double *values, std::size_t n) const;

    // Warn that `filename` holds legacy V*X in Ry (`what` = "V*sigma" or
    // "V*C", `how` describes the declaration) when the &cell field overrides
    // the primitive cell, since the file can then no longer be checked.
    void warn_legacy_unit(const char *filename, const char *what, const char *how) const;

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

    // ---- Long-range (dipole-dipole) correction for polar crystals ----

    // Dipole contribution to the long-wave brackets, computed as the second
    // q-derivative of the Ewald dipole force-constant map with the
    // macroscopic (G = 0) term excluded (fixed-E response; Born & Huang,
    // Sec. 26-27). Requires the Ewald machinery (NONANALYTIC = 3).
    void calc_longwave_brackets_dipole(const Ewald &ewald_in, NDArray<double, 4> &ret) const;

    // Clamped-ion elastic tensor with the dipole long-range correction:
    // brackets of the short-range IFCs (Ewald::fc2_without_dipole, i.e. the
    // fitted IFCs minus the supercell-folded dipole part) plus the analytic
    // dipole brackets of the infinite lattice. Cures the slow supercell-size
    // convergence of the plain bracket sums in polar crystals.
    void calc_elastic_tensor_longrange(const std::vector<FcsArrayWithCell> &fcs_short, const Ewald &ewald_in,
                                       NDArray<double, 4> &C_gpa, bool symmetrize = true) const;

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

    // Compute C3_{ij kl mn} in GPa using Wallace Eq. (8.14) and C2 from the
    // same clamped/relaxed path. symmetrize projects C3 onto minor and pair
    // permutation symmetries (48 operations). Apply this at C3, not A_hat:
    // rotational-invariance violations are not index permutations of A_hat.
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

    // Wallace Eq. (7.30) at zero initial stress: brackets A [Ry] to the
    // elastic tensor [GPa], with the optional intrinsic-symmetry projection.
    void brackets_to_elastic(const NDArray<double, 4> &A, NDArray<double, 4> &C_gpa, bool symmetrize) const;

    const System &system_;
};

} // namespace PHON_NS
