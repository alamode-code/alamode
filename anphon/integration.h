/*
 integration.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <memory>
#include <vector>
#include "constants.h"
#include "kpoint.h"
#include "memory.h"

namespace PHON_NS
{
class PhononVelocity;

struct tetra_pair
{
    double e;
    double f;
};

inline bool operator<(const tetra_pair &a, const tetra_pair &b)
{
    return a.e < b.e;
}

struct TetraWithKnum
{
    double e;
    int knum;
};

inline bool operator<(const TetraWithKnum &a, const TetraWithKnum &b)
{
    return a.e < b.e;
}

class TetraNodes
{
public:
    TetraNodes()
    {
        nk1 = 0;
        nk2 = 0;
        nk3 = 0;
        ntetra = 0;
    };

    TetraNodes(unsigned int nk1_in, unsigned int nk2_in, unsigned int nk3_in)
    {
        nk1 = nk1_in;
        nk2 = nk2_in;
        nk3 = nk3_in;
        ntetra = 6 * nk1 * nk2 * nk3;
        tetras.resize(ntetra, 4);
    };

    TetraNodes(const TetraNodes &) = delete;
    TetraNodes &operator=(const TetraNodes &) = delete;

    void setup();

    unsigned int get_ntetra() const;

    const unsigned int *const *get_tetras() const;

private:
    unsigned int nk1, nk2, nk3;
    unsigned int ntetra;
    NDArray<unsigned int, 2> tetras;
};

class AdaptiveSmearingSigma
{
public:
    AdaptiveSmearingSigma() {};

    AdaptiveSmearingSigma(const unsigned int nk_in, const unsigned int ns_in, const double factor)
    {

        vel.resize(nk_in, ns_in, 3);
        adaptive_factor = factor;
    };

    AdaptiveSmearingSigma(const AdaptiveSmearingSigma &) = delete;
    AdaptiveSmearingSigma &operator=(const AdaptiveSmearingSigma &) = delete;

    void setup(const PhononVelocity *phvel_class, const KpointMeshUniform *kmesh_in, const Eigen::Matrix3d &lavec_p_in,
               const Eigen::Matrix3d &rlavec_p_in);

    // overload for 3ph or 4ph
    void get_sigma(const unsigned int k1, const unsigned int s1, double &sigma_out);

    void get_sigma(const unsigned int k1, const unsigned int s1, const unsigned int k2, const unsigned int s2,
                   std::array<double, 2> &sigma_out);

    void get_sigma(const unsigned int k1, const unsigned int s1, const unsigned int k2, const unsigned int s2,
                   const unsigned int k3, const unsigned int s3, std::array<double, 4> &sigma_out);

    // Lower bound of the adaptive width (Ry, about 3 cm^-1).
    static constexpr double sigma_min = 2.0e-5;

    // Projections of the group velocity of mode (k, s) on the three mesh
    // spacings; the adaptive widths are quadratic forms of these, which lets
    // the four-phonon kernel tabulate them per band pair instead of calling
    // get_sigma for every band triple.
    void get_projected_velocity(const unsigned int k, const unsigned int s, double proj_out[3]) const
    {
        for (auto u = 0; u < 3; ++u) {
            proj_out[u] = 0.0;
            for (auto a = 0; a < 3; ++a) proj_out[u] += vel[k][s][a] * dq[u][a];
        }
    }

    double get_adaptive_factor() const
    {
        return adaptive_factor;
    }

private:
    double adaptive_factor;
    NDArray<double, 3> vel;
    double dq[3][3];
};

// Brillouin-zone integration kernels (tetrahedron / fixed and adaptive
// smearing). No Pointers base: setup_integration receives its inputs from
// the caller (PHON::setup_base); the parser fills the public settings.
class Integration
{
public:
    Integration();

    ~Integration();

    int ismear; // ismear = -1: tetrahedron, ismear = 0: gaussian
    int ismear_4ph;
    double epsilon;
    double epsilon_4ph;
    double adaptive_factor;

    std::unique_ptr<AdaptiveSmearingSigma> adaptive_sigma;
    std::unique_ptr<AdaptiveSmearingSigma> adaptive_sigma4;

    void setup_integration(const KpointMeshUniform *kmesh_dos_in, const PhononVelocity *phonon_velocity_in,
                           unsigned int ns_in, const Eigen::Matrix3d &lavec_p, const Eigen::Matrix3d &rlavec_p,
                           int quartic_mode_in, int my_rank_in, unsigned int verbosity = 1);

    // Allocate and initialize the adaptive smearing table for the 4ph
    // channel on its (possibly coarser) mesh. Called from
    // Conductivity::setup_kappa_4ph; Integration owns the object and
    // deletes it in the destructor. No-op if already created.
    void create_adaptive_sigma4(unsigned int nk_in, unsigned int ns_in, const KpointMeshUniform *kmesh_in,
                                const PhononVelocity *phonon_velocity_in, const Eigen::Matrix3d &lavec_p,
                                const Eigen::Matrix3d &rlavec_p);

    double do_tetrahedron(const double *energy, const double *f, const unsigned int ntetra,
                          const unsigned int *const *tetras, const double e_ref);

    void calc_weight_tetrahedron(const unsigned int nk_irreducible, const unsigned int *map_to_irreducible_k,
                                 const double *energy, const double e_ref, const unsigned int ntetra,
                                 const unsigned int *const *tetras, double *weight) const;

    void calc_weight_smearing(const unsigned int nk, const unsigned int nk_irreducible,
                              const unsigned int *map_to_irreducible_k, const double *energy, const double e_ref,
                              const int smearing_method, double *weight) const;

private:
    void set_default_variables();

    void deallocate_variables();

    void prepare_adaptivesmearing(const KpointMeshUniform *kmesh_dos_in, const PhononVelocity *phonon_velocity_in,
                                  unsigned int ns_in, const Eigen::Matrix3d &lavec_p, const Eigen::Matrix3d &rlavec_p);

    static inline double fij(double, double, double);

    std::vector<tetra_pair> tetra_data;

    static void insertion_sort(double *, int *, int);
};

inline double delta_lorentz(const double omega, const double epsilon)
{
    return inverse_pi * epsilon / (omega * omega + epsilon * epsilon);
}

inline double delta_gauss(const double omega, const double epsilon)
{
    return std::exp(-omega * omega / (epsilon * epsilon)) / (epsilon * std::sqrt(pi));
}
} // namespace PHON_NS
