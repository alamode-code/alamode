/*
 relaxation_types.h

 Copyright (c) 2026

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <array>
#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

namespace PHON_NS
{
enum class RelaxationStrMode : int
{
    None = 0,
    CoordinatesOnly = 1,
    CoordinatesAndCell = 2,
    PerturbativeQha = 3
};

enum class QhaScheme : int
{
    Standard = 0,
    ZSISA = 1,
    VZSISA = 2
};

inline constexpr int to_int(const RelaxationStrMode mode)
{
    return static_cast<int>(mode);
}

inline constexpr int to_int(const QhaScheme scheme)
{
    return static_cast<int>(scheme);
}

inline constexpr bool is_valid_relaxation_str_mode(const int mode)
{
    return mode == to_int(RelaxationStrMode::None) || mode == to_int(RelaxationStrMode::CoordinatesOnly) ||
           mode == to_int(RelaxationStrMode::CoordinatesAndCell) || mode == to_int(RelaxationStrMode::PerturbativeQha);
}

inline constexpr bool is_valid_qha_scheme(const int scheme)
{
    return scheme == to_int(QhaScheme::Standard) || scheme == to_int(QhaScheme::ZSISA) ||
           scheme == to_int(QhaScheme::VZSISA);
}

inline constexpr RelaxationStrMode to_relaxation_str_mode(const int mode)
{
    if (mode == to_int(RelaxationStrMode::CoordinatesOnly)) return RelaxationStrMode::CoordinatesOnly;
    if (mode == to_int(RelaxationStrMode::CoordinatesAndCell)) return RelaxationStrMode::CoordinatesAndCell;
    if (mode == to_int(RelaxationStrMode::PerturbativeQha)) return RelaxationStrMode::PerturbativeQha;
    return RelaxationStrMode::None;
}

inline constexpr QhaScheme to_qha_scheme(const int scheme)
{
    if (scheme == to_int(QhaScheme::ZSISA)) return QhaScheme::ZSISA;
    if (scheme == to_int(QhaScheme::VZSISA)) return QhaScheme::VZSISA;
    return QhaScheme::Standard;
}

class KpointMeshUniform;
struct MinimumDistList;
class PhaseFactorStorage;
class DelVStrainData;

struct DelVStrainOutputs
{
    DelVStrainData *del_v_strain{};
};

struct DelVStrainComputeInputs
{
    const KpointMeshUniform *kmesh_coarse{};
    const KpointMeshUniform *kmesh_dense{};
    double **omega2_harmonic{};
    std::complex<double> ***evec_harmonic{};
    RelaxationStrMode relax_str{RelaxationStrMode::None};
    MinimumDistList ***mindist_list{};
    const PhaseFactorStorage *phase_storage{};
};

// One row of the per-temperature structural-optimization history table.
struct StructOptStepRecord
{
    bool scp_ok{true}; // false when the SCP equation did not converge at this step
    double du0{};
    double du_tensor{};
    double grad_norm{-1.0};      // < 0 : not available
    double cell_grad_norm{-1.0}; // < 0 : not available
    std::string spacegroup;
};

struct RelaxationStructureState
{
    std::vector<double> q0;
    std::vector<double> u0;
    std::array<std::array<double, 3>, 3> u_tensor{}, eta_tensor{};
    std::vector<double> delta_q0;
    std::vector<double> delta_u0;
    std::array<double, 6> delta_umn{};
    double du0{};
    double du_tensor{};

    void resize(const std::size_t ns)
    {
        q0.assign(ns, 0.0);
        u0.assign(ns, 0.0);
        delta_q0.assign(ns, 0.0);
        delta_u0.assign(ns, 0.0);
        for (auto &row: u_tensor) {
            row.fill(0.0);
        }
        for (auto &row: eta_tensor) {
            row.fill(0.0);
        }
        delta_umn.fill(0.0);
        du0 = 0.0;
        du_tensor = 0.0;
    }
};

// Scratch buffers shared by the SCPH/QHA structural-optimization drivers.
// Non-owning: the driver allocates and frees the arrays; this struct only
// groups them so the loop stages can be factored into functions without
// dozens of parameters.
struct StructuralOptWorkspace
{
    RelaxationStrMode relax_mode = RelaxationStrMode::None;
    double pvcell = 0.0; // pressure * reference cell volume [Ry]

    // k-space IFCs at the reference structure
    std::complex<double> *v1_ref{};
    std::complex<double> ***v3_ref{};
    std::complex<double> ***v4_ref{};
    double v0_ref = 0.0;

    // IFCs renormalized by the strain u_{mu nu}
    std::complex<double> *v1_with_umn{};
    std::complex<double> **delta_v2_with_umn{};
    std::complex<double> ***v3_with_umn{};
    double v0_with_umn = 0.0;

    // IFCs renormalized by strain and internal displacement q0
    std::complex<double> *v1_renorm{};
    std::complex<double> **delta_v2_renorm{};
    std::complex<double> ***v3_renorm{};
    double v0_renorm = 0.0;

    // v4 entering the q0 renormalization: v4_ref in SCPH, the (numerically
    // identical) v4_with_umn copy in QHA. Strain renormalization of v4 would
    // need d(v4)/du IFC data, which del_v_strain does not provide.
    std::complex<double> ***v4_for_renorm{};

    // strain derivatives of the IFCs and elastic constants
    DelVStrainData *del_v_strain{};
    double *C1_array{};
    double **C2_array{};
    double ***C3_array{};

    // PES gradient w.r.t. strain at the current structure
    std::complex<double> *del_v0_del_umn_renorm{};

    // current structure
    RelaxationStructureState structure_state;

    // indices of the optical modes at the Gamma point
    std::vector<int> harm_optical_modes;
};

struct RelaxationUpdateInput
{
    double *q0{};
    double *u0{};
    double **u_tensor{};
    const std::complex<double> *v1_array_atT{};
    const double *const *omega2_array{};
    const std::complex<double> *del_v0_strain_atT{};
    const double *const *C2_array{};
    const std::complex<double> *const *const *cmat_convert{};
    const std::vector<int> *harm_optical_modes{};
    double **omega2_harmonic{};
    std::complex<double> ***evec_harmonic{};
};

struct RelaxationUpdateBuffers
{
    double *delta_q0{};
    double *delta_u0{};
    double *delta_umn{};
};

struct RelaxationUpdateOutputs
{
    double *du0{};
    double *du_tensor{};
};

struct TemperatureOptimizationLoopInputs
{
    const std::vector<double> *temperature_grid{};
    double Tmin{};
    double dT{};
    unsigned int NT{};
    double *q0{};
    double *u0{};
    double **u_tensor{};
    double **omega2_harmonic{};
    std::complex<double> ***evec_harmonic{};
    std::ofstream *fout_step_q0{};
    std::ofstream *fout_step_u0{};
    std::ofstream *fout_step_u_tensor{};
    std::ofstream *fout_q0{};
    std::ofstream *fout_u0{};
    std::ofstream *fout_u_tensor{};
};

class IRelaxationModel
{
public:
    virtual ~IRelaxationModel() = default;
    virtual RelaxationUpdateInput *update_input() = 0;
    virtual void before_temperature(unsigned int iT, double temp, bool &converged_prev) = 0;
    virtual void evaluate_structure_step(unsigned int iT, double temp, int i_str_loop, bool &converged_prev) = 0;
    virtual void evaluate_temperature(unsigned int iT, double temp, bool &converged_prev) = 0;
    virtual void after_update_step(unsigned int iT, double temp, int i_str_loop) = 0;
    virtual void after_temperature(unsigned int iT, double temp, bool &converged_prev) = 0;
    virtual double current_v0() const = 0;
};
} // namespace PHON_NS
