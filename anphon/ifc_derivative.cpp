#include "ifc_derivative.h"
#include <algorithm>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "interpolation.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "scph_v3v4_elements.h"
#include "system.h"

using namespace PHON_NS;

namespace
{

// Iterate over fcs_aligned (sorted by the first nelems indices), grouping
// consecutive entries with identical heading indices and relative vectors.
// accept(entry) filters entries; accumulate(entry) adds an accepted entry's
// contribution; flush(index_with_cell, atoms_s, relvecs, relvecs_velocity) is
// called once per closed group (and must reset the accumulator state).
template <class Accept, class Accumulate, class Flush>
void scan_fcs_groups(const std::vector<FcsArrayWithCell> &fcs_aligned, const std::size_t nelems, Accept &&accept,
                     Accumulate &&accumulate, Flush &&flush)
{
    std::vector<int> index_prev(nelems, -1), index_curr;
    std::vector<int> relvecs_int_prev(3 * (nelems - 1), 1000000), relvecs_int_curr;
    std::vector<int> index_with_cell_prev(3 * (nelems - 1) + 1, -1), index_with_cell_curr;
    std::vector<unsigned int> atoms_s_prev, atoms_s_curr;
    std::vector<Eigen::Vector3d> relvecs_prev(nelems - 1), relvecs_curr(nelems - 1);
    std::vector<Eigen::Vector3d> relvecs_vel_prev(nelems - 1), relvecs_vel_curr(nelems - 1);

    bool has_group = false;

    for (const auto &it: fcs_aligned) {

        if (!accept(it)) {
            continue;
        }

        index_curr.clear();
        relvecs_int_curr.clear();
        index_with_cell_curr.clear();
        atoms_s_curr.clear();

        index_curr.push_back(it.pairs[0].index);
        index_with_cell_curr.push_back(it.pairs[0].index);
        atoms_s_curr.emplace_back(it.atoms_s[0]);

        for (std::size_t i = 1; i < nelems; ++i) {
            index_curr.push_back(it.pairs[i].index);

            for (int j = 0; j < 3; ++j) {
                relvecs_int_curr.push_back(nint(it.relvecs[i - 1][j]));
            }

            index_with_cell_curr.push_back(it.pairs[i].index);
            index_with_cell_curr.push_back(it.pairs[i].tran);
            index_with_cell_curr.push_back(it.pairs[i].cell_s);

            atoms_s_curr.emplace_back(it.atoms_s[i]);
            relvecs_curr[i - 1] = it.relvecs[i - 1];
            relvecs_vel_curr[i - 1] = it.relvecs_velocity[i - 1];
        }

        if (!has_group || index_curr != index_prev || relvecs_int_curr != relvecs_int_prev) {
            if (has_group) {
                flush(index_with_cell_prev, atoms_s_prev, relvecs_prev, relvecs_vel_prev);
            }

            index_prev = index_curr;
            relvecs_int_prev = relvecs_int_curr;
            atoms_s_prev = atoms_s_curr;
            relvecs_prev = relvecs_curr;
            relvecs_vel_prev = relvecs_vel_curr;
            index_with_cell_prev = index_with_cell_curr;
            has_group = true;
        }

        accumulate(it);
    }

    if (has_group) {
        flush(index_with_cell_curr, atoms_s_curr, relvecs_curr, relvecs_vel_curr);
    }
}

// Rebuild the emitted pairs array from the packed index_with_cell key: the
// first atom is folded to the primitive cell (tran = cell_s = 0), the rest
// keep their translation and cell indices.
void build_pairs_vec(const std::vector<int> &index_with_cell, const std::size_t nelems,
                     std::vector<AtomCellSuper> &pairs_vec)
{
    AtomCellSuper pairs_tmp{};

    pairs_vec.clear();
    pairs_tmp.index = index_with_cell[0];
    pairs_tmp.tran = 0;
    pairs_tmp.cell_s = 0;
    pairs_vec.push_back(pairs_tmp);

    for (std::size_t i = 1; i < nelems; ++i) {
        pairs_tmp.index = index_with_cell[3 * i - 2];
        pairs_tmp.tran = index_with_cell[3 * i - 1];
        pairs_tmp.cell_s = index_with_cell[3 * i];
        pairs_vec.push_back(pairs_tmp);
    }
}

} // namespace

DerivativeIFC::DerivativeIFC(const System &system_in, const Symmetry &symmetry_in, const Fcs_phonon &fcs_phonon_in,
                             const Dynamical &dynamical_in, AnharmonicCore &anharmonic_core_in, const int my_rank_in,
                             const int nprocs_in) :
    system_(system_in), symmetry_(symmetry_in), fcs_phonon_(fcs_phonon_in), dynamical_(dynamical_in),
    anharmonic_core_(anharmonic_core_in), my_rank_(my_rank_in), nprocs_(nprocs_in)
{}

void DerivativeIFC::compute_dV1_dumn(MatrixXcdRowMajor &del_v1_del_umn,
                                     const std::complex<double> *const *const *const evec_harmonic) const
{
    // Calculate the first-order derivative of IFC1 with respect to strain in real space and transform it to the reciprocal space representation.
    // This term corresponds to Eq. (C1) of 10.1103/PhysRevB.106.224104
    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto ns = dynamical_.neval;
    const auto invsqrt_mass = system_.get_invsqrt_mass();
    Eigen::MatrixXd del_v1_del_umn_in_real_space(9, ns);

    del_v1_del_umn_in_real_space.setZero();

    const auto &force_constants = fcs_phonon_.force_constant_with_cell[0];
    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.reserve(force_constants.size());
    for (const auto &it: force_constants) {
        fcs_aligned.emplace_back(it);
    }
    if (fcs_aligned.size() > 1) {
        const unsigned int m = 1;
        const unsigned int n = static_cast<unsigned int>(fcs_aligned.front().pairs.size());
        const unsigned int number_of_tails = (n > m) ? m : 0;
        const sort_by_heading_indices operator_fcs(number_of_tails);
        boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);
    }

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 1, system_.get_primcell().lattice_vector);

    for (const auto &group: strain_groups) {
        const int ind1 = group.pairs[0].index;
        for (int ixyz = 0; ixyz < 9; ++ixyz) {
            del_v1_del_umn_in_real_space(ixyz, ind1) += group.values[ixyz];
        }
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 9; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            std::complex<double> sum(0.0, 0.0);
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    sum += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                           del_v1_del_umn_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
            del_v1_del_umn(ixyz, is1) = sum;
        }
    }
}

void DerivativeIFC::compute_d2V1_dumn2(MatrixXcdRowMajor &del2_v1_del_umn2,
                                       const std::complex<double> *const *const *const evec_harmonic) const
{
    // Calculates the second-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC3 in the unstrained system.
    // This term corresponds to Eq. (C2) of 10.1103/PhysRevB.106.224104
    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto ns = dynamical_.neval;
    const auto invsqrt_mass = system_.get_invsqrt_mass();
    Eigen::MatrixXd del2_v1_del_umn2_in_real_space(81, ns);

    del2_v1_del_umn2_in_real_space.setZero();

    const auto &force_constants = fcs_phonon_.force_constant_with_cell[1];
    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.reserve(force_constants.size());
    for (const auto &it: force_constants) {
        fcs_aligned.emplace_back(it);
    }
    if (fcs_aligned.size() > 1) {
        const unsigned int m = 2;
        const unsigned int n = static_cast<unsigned int>(fcs_aligned.front().pairs.size());
        const unsigned int number_of_tails = (n > m) ? m : 0;
        const sort_by_heading_indices operator_fcs(number_of_tails);
        boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);
    }

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 2, system_.get_primcell().lattice_vector);

    for (const auto &group: strain_groups) {
        const int ind1 = group.pairs[0].index;
        for (int ixyz_comb = 0; ixyz_comb < 81; ++ixyz_comb) {
            del2_v1_del_umn2_in_real_space(ixyz_comb, ind1) += group.values[ixyz_comb];
        }
    }


#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 81; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            std::complex<double> sum(0.0, 0.0);
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    sum += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                           del2_v1_del_umn2_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
            del2_v1_del_umn2(ixyz, is1) = sum;
        }
    }
}

void DerivativeIFC::compute_d3V1_dumn3(MatrixXcdRowMajor &del3_v1_del_umn3,
                                       const std::complex<double> *const *const *const evec_harmonic) const
{
    // Calculates the third-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC4 in the unstrained system.
    // This term corresponds to Eq. (C3) of 10.1103/PhysRevB.106.224104
    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto ns = dynamical_.neval;
    const auto invsqrt_mass = system_.get_invsqrt_mass();
    Eigen::MatrixXd del3_v1_del_umn3_in_real_space(729, ns);

    del3_v1_del_umn3_in_real_space.setZero();

    const auto &force_constants = fcs_phonon_.force_constant_with_cell[2];
    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.reserve(force_constants.size());
    for (const auto &it: force_constants) {
        fcs_aligned.emplace_back(it);
    }
    if (fcs_aligned.size() > 1) {
        const unsigned int m = 3;
        const unsigned int n = static_cast<unsigned int>(fcs_aligned.front().pairs.size());
        const unsigned int number_of_tails = (n > m) ? m : 0;
        const sort_by_heading_indices operator_fcs(number_of_tails);
        boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);
    }

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 3, system_.get_primcell().lattice_vector);

    for (const auto &group: strain_groups) {
        const int ind1 = group.pairs[0].index;
        for (int ixyz_comb = 0; ixyz_comb < 729; ++ixyz_comb) {
            del3_v1_del_umn3_in_real_space(ixyz_comb, ind1) += group.values[ixyz_comb];
        }
    }


#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 729; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            std::complex<double> sum(0.0, 0.0);
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    sum += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                           del3_v1_del_umn3_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
            del3_v1_del_umn3(ixyz, is1) = sum;
        }
    }
}

void DerivativeIFC::compute_dV2_dumn(std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
                                     const std::complex<double> *const *const *const evec_harmonic,
                                     const unsigned int nk, const double *const *xk_in) const
{
    // Calculate the first-order derivative of IFC2 with respect to strain in real space and transform it to the reciprocal space representation.
    // This term corresponds to Eq. (C4) of 10.1103/ PhysRevB.106.224104.

    using namespace Eigen;

    const auto ns = dynamical_.neval;
    int is1, is2;

    std::vector<FcsArrayWithCell> delta_fcs;

    NDArray<std::complex<double>, 2> mat_tmp;
    mat_tmp.resize(ns, ns);

    MatrixXcd Dymat(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    std::vector<FcsArrayWithCell> fcs_aligned;

    fcs_aligned.clear();

    // Copy 3rd-order force constants to fcs_aligned and sort them by heading indices
    for (const auto &it: fcs_phonon_.force_constant_with_cell[1]) {
        fcs_aligned.emplace_back(it);
    }
    sort_by_heading_indices const operator_fcs(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 1, system_.get_primcell().lattice_vector);

    for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            extract_strain_component(strain_groups, ixyz1 * 3 + ixyz2, 1, eps15, delta_fcs);

            auto &per_strain = del_v2_del_umn[ixyz1 * 3 + ixyz2];
            for (int ik = 0; ik < nk; ik++) {

                dynamical_.calc_analytic_k(xk_in[ik], delta_fcs, mat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        Dymat(is1, is2) = mat_tmp[is1][is2];
                        evec_tmp(is1, is2) = evec_harmonic[ik][is2][is1];
                    }
                }
                Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        per_strain(ik, is1 * ns + is2) = Dymat(is1, is2);
                    }
                }
            }
        }
    }

    mat_tmp.clear();
}

void DerivativeIFC::compute_d2V2_dumn2(std::vector<MatrixXcdRowMajor> &del2_v2_del_umn2,
                                       const std::complex<double> *const *const *const evec_harmonic,
                                       const unsigned int nk, const double *const *xk_in) const
{
    // Calculate the second-order derivative of IFC2 with respect to strain in real space and transform it to the reciprocal space representation.
    // This term corresponds to Eq. (C5) of 10.1103/ PhysRevB.106.224104.
    using namespace Eigen;

    const auto ns = dynamical_.neval;

    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.clear();
    for (const auto &it: fcs_phonon_.force_constant_with_cell[2]) {
        fcs_aligned.emplace_back(it);
    }
    sort_by_heading_indices const operator_fcs(2);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 2, system_.get_primcell().lattice_vector);

#pragma omp parallel
    {
        int ixyz;
        int is1, is2;
        std::vector<FcsArrayWithCell> delta_fcs;

        NDArray<std::complex<double>, 2> mat_tmp;
        mat_tmp.resize(ns, ns);

        MatrixXcd Dymat(ns, ns);
        MatrixXcd evec_tmp(ns, ns);

#pragma omp for
        for (ixyz = 0; ixyz < 81; ixyz++) {
            extract_strain_component(strain_groups, ixyz, 2, eps15, delta_fcs);

            auto &per_strain = del2_v2_del_umn2[ixyz];
            for (int ik = 0; ik < nk; ik++) {
                dynamical_.calc_analytic_k(xk_in[ik], delta_fcs, mat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        Dymat(is1, is2) = mat_tmp[is1][is2];
                        evec_tmp(is1, is2) = evec_harmonic[ik][is2][is1];
                    }
                }
                Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        per_strain(ik, is1 * ns + is2) = Dymat(is1, is2);
                    }
                }
            }
        }

        mat_tmp.clear();
    }
}

void DerivativeIFC::compute_dV3_dumn(std::vector<std::vector<MatrixXcdRowMajor>> &del_v3_del_umn,
                                     const std::complex<double> *const *const *const evec_harmonic,
                                     const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
                                     const PhaseFactorCache *phase_cache_in) const
{
    // Calculate the first-order derivative of IFC3 with respect to strain in real space and transform it to the reciprocal space representation.
    // This term corresponds to Eq. (C6) of 10.1103/PhysRevB.106.224104.
    const auto ns = dynamical_.neval;
    const auto ns2 = static_cast<std::size_t>(ns) * ns;
    const auto nk_dense = static_cast<int>(kmesh_dense_in->nk);

    if (del_v3_del_umn.size() != 9) {
        exit("compute_dV3_dumn", "del_v3_del_umn must have 9 strain components.");
    }
    for (const auto &per_strain: del_v3_del_umn) {
        if (static_cast<int>(per_strain.size()) != nk_dense) {
            exit("compute_dV3_dumn", "del_v3_del_umn has inconsistent k-point dimension.");
        }
        for (const auto &mat: per_strain) {
            if (mat.rows() != ns || static_cast<std::size_t>(mat.cols()) != ns2) {
                exit("compute_dV3_dumn", "del_v3_del_umn entries must be shaped [ns, ns*ns].");
            }
        }
    }

    int ngroup_tmp;
    NDArray<double, 1> invmass_v3_tmp;
    NDArray<int, 2> evec_index_v3_tmp;
    NDArray<std::vector<double>, 1> fcs_group_tmp;
    NDArray<std::vector<RelativeVector>, 1> relvec_tmp;

    int i;
    int ixyz1, ixyz2;

    NDArray<double, 1> invsqrt_mass_p;
    invsqrt_mass_p.resize(system_.get_primcell().number_of_atoms);
    for (i = 0; i < system_.get_primcell().number_of_atoms; ++i) {
        invsqrt_mass_p[i] = std::sqrt(1.0 / system_.get_mass_prim()[i]);
    }

    std::vector<FcsArrayWithCell> delta_fcs;
    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.clear();
    for (const auto &it: fcs_phonon_.force_constant_with_cell[2]) {
        fcs_aligned.emplace_back(it);
    }
    const sort_by_heading_indices operator_fcs(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

    std::vector<DeltaFcsStrainComponents> strain_groups;
    compute_dV_dumn_all_real_space(fcs_aligned, strain_groups, 1, system_.get_primcell().lattice_vector);

    // Scratch pointer views over the row-major Eigen matrices of one strain index,
    // used to bridge with the legacy raw-pointer interface of compute_V3_elements_for_given_IFCs.
    std::vector<std::complex<double> *> row_ptrs(static_cast<std::size_t>(nk_dense) * ns);
    std::vector<std::complex<double> **> kptr_view(nk_dense);

    const auto is_acoustic_gamma = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {

            extract_strain_component(strain_groups, ixyz1 * 3 + ixyz2, 1, eps15, delta_fcs);

            auto &per_strain = del_v3_del_umn[ixyz1 * 3 + ixyz2];

            if (delta_fcs.empty()) {
#pragma omp parallel for schedule(static)
                for (int ik = 0; ik < nk_dense; ++ik) {
                    per_strain[ik].setZero();
                }
                continue;
            }

            boost::sort::block_indirect_sort(delta_fcs.begin(), delta_fcs.end());

            AnharmonicCore::prepare_group_of_force_constants(delta_fcs, ngroup_tmp, fcs_group_tmp);

            if (ngroup_tmp == 0) {
#pragma omp parallel for schedule(static)
                for (int ik = 0; ik < nk_dense; ++ik) {
                    per_strain[ik].setZero();
                }
                continue;
            }

            invmass_v3_tmp.resize(ngroup_tmp);
            evec_index_v3_tmp.resize(ngroup_tmp, 3);
            relvec_tmp.resize(ngroup_tmp);

            AnharmonicCore::prepare_relative_vector(delta_fcs, ngroup_tmp, fcs_group_tmp, relvec_tmp);

            int k = 0;
            for (i = 0; i < ngroup_tmp; ++i) {
                for (int j = 0; j < 3; ++j) {
                    evec_index_v3_tmp[i][j] = delta_fcs[k].pairs[j].index;
                }
                invmass_v3_tmp[i] = invsqrt_mass_p[evec_index_v3_tmp[i][0] / 3] *
                                    invsqrt_mass_p[evec_index_v3_tmp[i][1] / 3] *
                                    invsqrt_mass_p[evec_index_v3_tmp[i][2] / 3];
                k += fcs_group_tmp[i].size();
            }

            for (int ik = 0; ik < nk_dense; ++ik) {
                std::complex<double> *base = per_strain[ik].data();
                for (int is = 0; is < ns; ++is) {
                    row_ptrs[static_cast<std::size_t>(ik) * ns + is] = base + static_cast<std::size_t>(is) * ns2;
                }
                kptr_view[ik] = row_ptrs.data() + static_cast<std::size_t>(ik) * ns;
            }

            compute_V3_elements_for_given_IFCs(kptr_view.data(),
                                               is_acoustic_gamma,
                                               ngroup_tmp,
                                               fcs_group_tmp,
                                               relvec_tmp,
                                               invmass_v3_tmp,
                                               evec_index_v3_tmp,
                                               evec_harmonic,
                                               true,
                                               ns,
                                               kmesh_coarse_in,
                                               kmesh_dense_in,
                                               phase_cache_in,
                                               anharmonic_core_,
                                               my_rank_,
                                               nprocs_);

            fcs_group_tmp.clear();
            invmass_v3_tmp.clear();
            evec_index_v3_tmp.clear();
            relvec_tmp.clear();
        }
    }
    invsqrt_mass_p.clear();
}

void DerivativeIFC::compute_dV_dumn_all_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                   std::vector<DeltaFcsStrainComponents> &groups, const std::size_t m,
                                                   const Eigen::Matrix3d &convmat)
{
    // Computes the m-th derivative of the force constants with respect to strain in real space,
    // for all 9^m strain-tensor components in a single scan over fcs_aligned. Each FC entry has
    // its tail Cartesian indices mu_j fixed, so it contributes fcs_val * prod_j vec_j[nu_j] to
    // the 3^m components sharing its mu-combination (one per nu-combination).
    // The input array fcs_aligned is assumed to be sorted by the first (n-m) indices of the
    // force constant pairs, where n is the order of the force constants.
    groups.clear();

    if (fcs_aligned.empty()) return;

    // touched_mu packs one bit per mu-combination, so 3^m must fit in 32 bits.
    if (m == 0 || m > 3) {
        exit("compute_dV_dumn_all_real_space", "Derivative order m must be 1, 2, or 3.");
    }

    const auto norder = fcs_aligned[0].pairs.size();

    if (m >= norder) {
        exit("compute_dV_dumn_all_real_space", "Derivative order m must be smaller than IFC order n.");
    }

    const auto nelems = norder - m;

    constexpr std::size_t pow3[4] = {1, 3, 9, 27};
    const auto ncomp = pow3[m] * pow3[m];

    std::vector<double> acc(ncomp, 0.0);
    uint32_t touched_mu = 0;
    std::vector<AtomCellSuper> pairs_vec;

    scan_fcs_groups(
        fcs_aligned,
        nelems,
        [](const FcsArrayWithCell &) { return true; },
        [&](const FcsArrayWithCell &it) {
            Eigen::Vector3d vecs[3];
            std::size_t mu[3];
            std::size_t p = 0;
            for (std::size_t j = 0; j < m; ++j) {
                mu[j] = it.coords[nelems + j];
                vecs[j] = convmat * it.relvecs_velocity[(nelems - 1) + j];
                p = p * 3 + mu[j];
            }
            touched_mu |= 1u << p;

            for (std::size_t q = 0; q < pow3[m]; ++q) {
                auto term = it.fcs_val;
                std::size_t c = 0;
                for (std::size_t j = 0; j < m; ++j) {
                    const auto nu = (q / pow3[m - 1 - j]) % 3;
                    term *= vecs[j][nu];
                    c = c * 9 + mu[j] * 3 + nu;
                }
                acc[c] += term;
            }
        },
        [&](const std::vector<int> &index_with_cell,
            const std::vector<unsigned int> &atoms_s,
            const std::vector<Eigen::Vector3d> &relvecs,
            const std::vector<Eigen::Vector3d> &relvecs_vel) {
            build_pairs_vec(index_with_cell, nelems, pairs_vec);

            groups.emplace_back();
            auto &group = groups.back();
            group.pairs = pairs_vec;
            group.atoms_s = atoms_s;
            group.relvecs = relvecs;
            group.relvecs_velocity = relvecs_vel;
            group.values = acc;
            group.touched_mu = touched_mu;

            std::fill(acc.begin(), acc.end(), 0.0);
            touched_mu = 0;
        });
}

void DerivativeIFC::extract_strain_component(const std::vector<DeltaFcsStrainComponents> &groups,
                                             const std::size_t component, const std::size_t m,
                                             const double emit_threshold, std::vector<FcsArrayWithCell> &delta_fcs)
{
    extract_strain_combination(groups, {{component, 1.0}}, m, emit_threshold, delta_fcs);
}

void DerivativeIFC::extract_strain_combination(const std::vector<DeltaFcsStrainComponents> &groups,
                                               const std::vector<std::pair<std::size_t, double>> &terms,
                                               const std::size_t m, const double emit_threshold,
                                               std::vector<FcsArrayWithCell> &delta_fcs)
{
    delta_fcs.clear();

    constexpr std::size_t pow9[4] = {1, 9, 81, 729};

    if (m == 0 || m > 3 || terms.empty()) {
        exit("extract_strain_combination", "Invalid derivative order or empty term list.");
    }

    // The mu-combination bits of the terms, matching the touched_mu convention.
    uint32_t mu_mask = 0;
    for (const auto &term: terms) {
        if (term.first >= pow9[m]) {
            exit("extract_strain_combination", "Invalid strain component index.");
        }
        std::size_t p = 0;
        for (std::size_t j = 0; j < m; ++j) {
            p = p * 3 + ((term.first / pow9[m - 1 - j]) % 9) / 3;
        }
        mu_mask |= 1u << p;
    }

    for (const auto &group: groups) {
        if (!(group.touched_mu & mu_mask)) continue;

        double val = 0.0;
        for (const auto &term: terms) {
            val += term.second * group.values[term.first];
        }
        if (emit_threshold >= 0.0 && std::abs(val) <= emit_threshold) continue;

        delta_fcs.emplace_back(val, group.pairs, group.atoms_s, group.relvecs, group.relvecs_velocity);
    }
}

void DerivativeIFC::compute_dV_dsublattice_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                      std::vector<FcsArrayWithCell> &delta_fcs,
                                                      const Eigen::VectorXd &S_field, const double emit_threshold)
{
    delta_fcs.clear();

    if (fcs_aligned.empty()) return;

    const auto norder = fcs_aligned[0].pairs.size();

    if (norder < 2) {
        exit("compute_dV_dsublattice_real_space", "IFC order must be at least 2.");
    }

    const auto nelems = norder - 1;

    double fcs_tmp = 0.0;
    std::vector<AtomCellSuper> pairs_vec;

    scan_fcs_groups(
        fcs_aligned,
        nelems,
        [](const FcsArrayWithCell &) { return true; },
        [&](const FcsArrayWithCell &it) { fcs_tmp += it.fcs_val * S_field(it.pairs[norder - 1].index); },
        [&](const std::vector<int> &index_with_cell,
            const std::vector<unsigned int> &atoms_s,
            const std::vector<Eigen::Vector3d> &relvecs,
            const std::vector<Eigen::Vector3d> &relvecs_vel) {
            if (!(emit_threshold >= 0.0 && std::abs(fcs_tmp) <= emit_threshold)) {
                build_pairs_vec(index_with_cell, nelems, pairs_vec);
                delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s, relvecs, relvecs_vel);
            }
            fcs_tmp = 0.0;
        });
}

void DerivativeIFC::compute_dV_dstrain_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                  std::vector<FcsArrayWithCell> &delta_fcs,
                                                  const std::vector<Eigen::Matrix3d> &strain_dirs,
                                                  const Eigen::Matrix3d &convmat, const double emit_threshold)
{
    // This is a helper function that computes the directional derivative of the force constants
    // with respect to strain in real space, along the strain tensor strain_dirs[j] for the j-th derivative.
    // The derivative order is determined by the size of the strain_dirs vector.
    // The input array fcs_aligned is assumed to be sorted by the first (n-m) indices of the force constant pairs,
    // where m is the derivative order and n is the order of the force constants (n=2: harmonic, n=3: cubic, etc.).
    if (fcs_aligned.empty()) {
        delta_fcs.clear();
        return;
    }

    const auto m = strain_dirs.size();

    if (m == 0) {
        exit("compute_dV_dstrain_real_space", "Inconsistent derivative-order input.");
    }

    const auto norder = fcs_aligned[0].pairs.size();

    if (m >= norder) {
        exit("compute_dV_dstrain_real_space", "Derivative order m must be smaller than IFC order n.");
    }

    const auto nelems = norder - m;

    delta_fcs.clear();

    auto tail_mu_matches = [&](const FcsArrayWithCell &it) {
        for (std::size_t j = 0; j < m; ++j) {
            const auto mu = it.coords[nelems + j];
            if (strain_dirs[j](mu, 0) == 0.0 && strain_dirs[j](mu, 1) == 0.0 && strain_dirs[j](mu, 2) == 0.0) {
                return false;
            }
        }
        return true;
    };

    auto compute_term = [&](const FcsArrayWithCell &it) {
        double term = it.fcs_val;
        for (std::size_t j = 0; j < m; ++j) {
            const auto mu = it.coords[nelems + j];
            const Eigen::Vector3d vec = convmat * it.relvecs_velocity[(nelems - 1) + j];
            double contracted = 0.0;
            for (int nu = 0; nu < 3; ++nu) {
                contracted += strain_dirs[j](mu, nu) * vec[nu];
            }
            term *= contracted;
        }
        return term;
    };

    double fcs_tmp = 0.0;
    std::vector<AtomCellSuper> pairs_vec;

    scan_fcs_groups(
        fcs_aligned,
        nelems,
        tail_mu_matches,
        [&](const FcsArrayWithCell &it) { fcs_tmp += compute_term(it); },
        [&](const std::vector<int> &index_with_cell,
            const std::vector<unsigned int> &atoms_s,
            const std::vector<Eigen::Vector3d> &relvecs,
            const std::vector<Eigen::Vector3d> &relvecs_vel) {
            if (!(emit_threshold >= 0.0 && std::abs(fcs_tmp) <= emit_threshold)) {
                build_pairs_vec(index_with_cell, nelems, pairs_vec);
                delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s, relvecs, relvecs_vel);
            }
            fcs_tmp = 0.0;
        });
}

void DerivativeIFC::set_del_v_fixed_cell(const size_t nk, const size_t ns, DelVStrainData &del_v_strain) const
{
    if (static_cast<int>(nk) != del_v_strain.nk() || static_cast<int>(ns) != del_v_strain.nmode()) {
        exit("set_del_v_fixed_cell", "inconsistent dimensions between input sizes and DelVStrainData.");
    }

    del_v_strain.del_v1.setZero();
    del_v_strain.del2_v1.setZero();
    del_v_strain.del3_v1.setZero();

    for (auto &mat: del_v_strain.del_v2) {
        mat.setZero();
    }
    for (auto &mat: del_v_strain.del2_v2) {
        mat.setZero();
    }
    for (auto &per_strain: del_v_strain.del_v3) {
        for (auto &mat: per_strain) {
            mat.setZero();
        }
    }
}

void DerivativeIFC::set_del_v_relax_cell(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                         const size_t ns, DelVStrainData &del_v_strain, double **omega2_harmonic,
                                         std::complex<double> ***evec_harmonic, const int renorm_2to1st,
                                         const int renorm_34to1st, const int renorm_3to2nd,
                                         const std::string &strain_ifc_dir, MinimumDistList ***mindist_list,
                                         const PhaseFactorCache *phase_cache_in) const
{
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    if (static_cast<int>(ns) != del_v_strain.nmode() || static_cast<int>(nk) != del_v_strain.nk()) {
        exit("set_del_v_relax_cell", "inconsistent dimensions between input sizes and DelVStrainData.");
    }

    switch (renorm_2to1st) {
    case 0:
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        del_v_strain.del_v1.setZero();
        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    case 1:
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (from harmonic IFCs) ... ";
        compute_dV1_dumn(del_v_strain.del_v1, evec_harmonic);
        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    case 2:
        if (my_rank_ == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (finite difference method) ... ";
        calculate_delv1_delumn_finite_difference(del_v_strain.del_v1, evec_harmonic, strain_ifc_dir);
        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    default:
        break;
    }

    switch (renorm_34to1st) {
    case 0:
        if (my_rank_ == 0) std::cout << "  - second-order derivatives of first-order IFCs (set zero) ... ";
        del_v_strain.del2_v1.setZero();
        if (my_rank_ == 0) {
            std::cout << "  done!\n";
            std::cout << "  - third-order derivatives of first-order IFCs (set zero) ... ";
        }
        del_v_strain.del3_v1.setZero();
        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    case 1:
        if (my_rank_ == 0) std::cout << "  - second-order derivatives of first-order IFCs (from cubic IFCs) ... ";
        compute_d2V1_dumn2(del_v_strain.del2_v1, evec_harmonic);

        if (my_rank_ == 0) {
            std::cout << "  done!\n";
            std::cout << "  - third-order derivatives of first-order IFCs (from quartic IFCs) ... ";
        }
        compute_d3V1_dumn3(del_v_strain.del3_v1, evec_harmonic);

        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    default:
        break;
    }

    switch (renorm_3to2nd) {
    case 1:
        if (my_rank_ == 0)
            std::cout << "  - first-order derivatives of harmonic IFCs (from cubic IFCs) ... " << std::flush;

        compute_dV2_dumn(del_v_strain.del_v2, evec_harmonic, nk_interpolate, kmesh_coarse->xk);
        break;
    case 2:
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    use inputs with all strain patterns ... " << std::flush;
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v_strain.del_v2,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 renorm_3to2nd,
                                                 strain_ifc_dir,
                                                 mindist_list);
        break;

    case 3:
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    use inputs with specified strain patterns ... " << std::flush;
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v_strain.del_v2,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 renorm_3to2nd,
                                                 strain_ifc_dir,
                                                 mindist_list);
        break;

    case 4:
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs\n";
            std::cout << "    (read from file in k-space representation) ... " << std::flush;
        }

        read_del_v2_del_umn_in_kspace(omega2_harmonic, evec_harmonic, del_v_strain.del_v2, nk);
        break;

    default:
        break;
    }
    if (my_rank_ == 0) std::cout << "  done!\n";

    if (my_rank_ == 0)
        std::cout << "  - second-order derivatives of harmonic IFCs (from quartic IFCs) ... " << std::flush;

    compute_d2V2_dumn2(del_v_strain.del2_v2, evec_harmonic, nk, kmesh_dense->xk);

    if (my_rank_ == 0) {
        std::cout << "  done!\n";
        std::cout << "  - first-order derivatives of cubic IFCs (from quartic IFCs) ... " << std::flush;
    }

    compute_dV3_dumn(del_v_strain.del_v3, evec_harmonic, kmesh_coarse, kmesh_dense, phase_cache_in);

    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }
}

void DerivativeIFC::set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse,
                                                   const KpointMeshUniform *kmesh_dense, const size_t ns,
                                                   DelVStrainData &del_v_strain, double **omega2_harmonic,
                                                   std::complex<double> ***evec_harmonic, const int renorm_2to1st,
                                                   const int renorm_34to1st, const int renorm_3to2nd,
                                                   const std::string &strain_ifc_dir,
                                                   MinimumDistList ***mindist_list) const
{
    const auto nk = kmesh_dense->nk;
    if (static_cast<int>(ns) != del_v_strain.nmode() || static_cast<int>(nk) != del_v_strain.nk()) {
        exit("set_del_v_relax_cell_linearQHA", "inconsistent dimensions between input sizes and DelVStrainData.");
    }

    if (renorm_2to1st == 0) {
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        del_v_strain.del_v1.setZero();
    } else if (renorm_2to1st == 1) {
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (from harmonic IFCs) ... ";
        compute_dV1_dumn(del_v_strain.del_v1, evec_harmonic);

    } else if (renorm_2to1st == 2) {
        if (my_rank_ == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (finite difference method) ... ";
        calculate_delv1_delumn_finite_difference(del_v_strain.del_v1, evec_harmonic, strain_ifc_dir);
    }
    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }

    if (renorm_34to1st == 0) {
        if (my_rank_ == 0) std::cout << "  - second-order derivatives of first-order IFCs (set zero) ... ";
        del_v_strain.del2_v1.setZero();

    } else if (renorm_34to1st == 1) {
        if (my_rank_ == 0) std::cout << "  - second-order derivatives of first-order IFCs (from cubic IFCs) ... ";
        compute_d2V1_dumn2(del_v_strain.del2_v1, evec_harmonic);
    }
    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }

    if (renorm_3to2nd == 1) {
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of harmonic IFCs (from cubic IFCs) ... ";

        compute_dV2_dumn(del_v_strain.del_v2, evec_harmonic, nk, kmesh_coarse->xk);
    } else if (renorm_3to2nd == 2 || renorm_3to2nd == 3) {
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            if (renorm_3to2nd == 2) {
                std::cout << "   use inputs with all strain patterns ...\n";
            } else if (renorm_3to2nd == 3) {
                std::cout << "   use inputs with specified strain patterns ...\n";
            }
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v_strain.del_v2,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 renorm_3to2nd,
                                                 strain_ifc_dir,
                                                 mindist_list);
    } else if (renorm_3to2nd == 4) {
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs\n";
            std::cout << "    (read from file in k-space representation) ... ";
        }
        read_del_v2_del_umn_in_kspace(omega2_harmonic, evec_harmonic, del_v_strain.del_v2, nk);
    }
    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }
}

void DerivativeIFC::read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                                  const std::complex<double> *const *const *const evec_harmonic,
                                                  std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
                                                  const unsigned int nk) const
{
    using namespace Eigen;

    const auto ns = dynamical_.neval;

    int ixyz1, ixyz2;
    int ik, is, js;

    double re_tmp, im_tmp;

    MatrixXcd dymat_tmp_mode(ns, ns);
    MatrixXcd dymat_tmp_alphamu(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    std::fstream fin_strain_mode_coupling_kspace;

    NDArray<std::complex<double>, 3> del_v2_del_umn_alphamu;
    del_v2_del_umn_alphamu.resize(9, nk, ns * ns);

    fin_strain_mode_coupling_kspace.open("B_array_kspace.txt");

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            for (ik = 0; ik < static_cast<int>(nk); ik++) {
                for (is = 0; is < ns * ns; is++) {
                    fin_strain_mode_coupling_kspace >> re_tmp >> im_tmp;
                    del_v2_del_umn_alphamu[ixyz1 * 3 + ixyz2][ik][is] = std::complex<double>(re_tmp, im_tmp);
                }
            }
        }
    }

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            auto &per_strain = del_v2_del_umn[ixyz1 * 3 + ixyz2];
            for (ik = 0; ik < static_cast<int>(nk); ik++) {

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        evec_tmp(is, js) = evec_harmonic[ik][js][is];
                        dymat_tmp_alphamu(is, js) = del_v2_del_umn_alphamu[ixyz1 * 3 + ixyz2][ik][is * ns + js];
                    }
                }
                dymat_tmp_mode = evec_tmp.adjoint() * dymat_tmp_alphamu * evec_tmp;

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        per_strain(ik, is * ns + js) = dymat_tmp_mode(is, js);
                    }
                }
            }
        }
    }
    del_v2_del_umn_alphamu.clear();

    const auto is_acoustic_gamma = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        auto &per_strain = del_v2_del_umn[ixyz1];
        for (is = 0; is < ns; is++) {
            if (!is_acoustic_gamma[is]) {
                continue;
            }
            for (js = 0; js < ns; js++) {
                per_strain(0, is * ns + js) = complex_zero;
                per_strain(0, js * ns + is) = complex_zero;
            }
        }
    }
}

void DerivativeIFC::calculate_delv1_delumn_finite_difference(
    MatrixXcdRowMajor &del_v1_del_umn, const std::complex<double> *const *const *const evec_harmonic,
    const std::string &strain_ifc_dir) const
{
    const auto natmin = system_.get_primcell().number_of_atoms;
    auto ns = dynamical_.neval;

    int ixyz1, ixyz2, ixyz3, ixyz12, ixyz22, ixyz32, i1, i2;
    int iat1, iat2, is1, isymm;
    double dtmp;
    std::string mode_tmp;
    double smag, weight;
    Eigen::Matrix3d weight_sum;
    std::fstream fin_strain_force_coupling;

    NDArray<double, 2> del_v1_del_umn_in_real_space;
    NDArray<double, 2> del_v1_del_umn_in_real_space_symm;
    del_v1_del_umn_in_real_space.resize(9, ns);
    del_v1_del_umn_in_real_space_symm.resize(9, ns);

    fin_strain_force_coupling.open(strain_ifc_dir + "strain_force.in");

    if (!fin_strain_force_coupling) {
        exit("calculate_delv1_delumn_finite_difference", "strain_force.in not found");
    }

    weight_sum.setZero();

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        std::fill_n(del_v1_del_umn_in_real_space[ixyz1], ns, 0.0);
    }

    while (true) {
        if (fin_strain_force_coupling >> mode_tmp >> smag >> weight) {
            if (mode_tmp == "xx") {
                ixyz1 = ixyz2 = 0;
            } else if (mode_tmp == "yy") {
                ixyz1 = ixyz2 = 1;
            } else if (mode_tmp == "zz") {
                ixyz1 = ixyz2 = 2;
            } else if (mode_tmp == "xy") {
                ixyz1 = 0;
                ixyz2 = 1;
            } else if (mode_tmp == "yz") {
                ixyz1 = 1;
                ixyz2 = 2;
            } else if (mode_tmp == "zx") {
                ixyz1 = 2;
                ixyz2 = 0;
            }

            for (iat1 = 0; iat1 < natmin; iat1++) {
                for (ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    fin_strain_force_coupling >> dtmp;
                    del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3] += dtmp * -1.0 / smag * weight;

                    if (ixyz1 != ixyz2) {
                        del_v1_del_umn_in_real_space[ixyz2 * 3 + ixyz1][iat1 * 3 + ixyz3] =
                            del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3];
                    }
                }
            }

            if (ixyz1 == ixyz2) {
                weight_sum(ixyz1, ixyz2) += weight;
            } else {
                weight_sum(ixyz1, ixyz2) += weight;
                weight_sum(ixyz2, ixyz1) += weight;
            }
        } else {
            break;
        }
    }

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) > eps6) {
                exit("calculate_delv1_delumn_finite_difference", "Sum of weights must be 1.");
            }
        }
    }

    constexpr double eV_to_Ry = 1.6021766208e-19 / Ryd;
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_in_real_space[ixyz1][is1] *= Bohr_in_Angstrom * eV_to_Ry;
        }
    }

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        std::fill_n(del_v1_del_umn_in_real_space_symm[ixyz1], ns, 0.0);
    }

    for (isymm = 0; isymm < symmetry_.SymmListWithMap_ref.size(); isymm++) {
        for (iat1 = 0; iat1 < natmin; iat1++) {
            iat2 = symmetry_.SymmListWithMap_ref[isymm].mapping[iat1];

            for (i1 = 0; i1 < 27; i1++) {
                ixyz1 = i1 / 9;
                ixyz2 = (i1 % 9) / 3;
                ixyz3 = i1 % 3;
                for (i2 = 0; i2 < 27; i2++) {
                    ixyz12 = i2 / 9;
                    ixyz22 = (i2 % 9) / 3;
                    ixyz32 = i2 % 3;

                    del_v1_del_umn_in_real_space_symm[ixyz12 * 3 + ixyz22][iat2 * 3 + ixyz32] +=
                        del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3] *
                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz12 * 3 + ixyz1] *
                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz22 * 3 + ixyz2] *
                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz32 * 3 + ixyz3];
                }
            }
        }
    }

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_in_real_space_symm[ixyz1][is1] /= static_cast<double>(symmetry_.SymmListWithMap_ref.size());
        }
    }

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            std::complex<double> sum(0.0, 0.0);
            for (iat1 = 0; iat1 < natmin; iat1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    sum += evec_harmonic[0][is1][iat1 * 3 + ixyz2] * system_.get_invsqrt_mass()[iat1] *
                           del_v1_del_umn_in_real_space_symm[ixyz1][iat1 * 3 + ixyz2];
                }
            }
            del_v1_del_umn(ixyz1, is1) = sum;
        }
    }

    del_v1_del_umn_in_real_space.clear();
    del_v1_del_umn_in_real_space_symm.clear();
}

void DerivativeIFC::calculate_delv2_delumn_finite_difference(
    double **omega2_harmonic, const std::complex<double> *const *const *const evec_harmonic,
    std::vector<MatrixXcdRowMajor> &del_v2_del_umn, const KpointMeshUniform *kmesh_coarse,
    const KpointMeshUniform *kmesh_dense, const int renorm_3to2nd, const std::string &strain_ifc_dir,
    MinimumDistList ***mindist_list) const
{
    using namespace Eigen;

    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto nat = system_.get_supercell(0).number_of_atoms;
    const auto natmin3 = natmin * 3;
    const auto nat3 = nat * 3;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical_.neval;

    NDArray<int, 2> symm_mapping_s;
    NDArray<int, 2> inv_translation_mapping;

    std::vector<FcsClassExtent> fc2_tmp;

    int ixyz1, ixyz2, ixyz3, ixyz4;
    int ixyz1_2, ixyz2_2, ixyz3_2, ixyz4_2;
    int ixyz_comb1, ixyz_comb2;
    int i1, i2;
    int iat1, iat2, iat1_2, iat2_2, iat2_2_prim;
    int itran1, itran2;
    int ik, is1, is2, is, js;
    int isymm, imode;
    int index2;

    NDArray<std::complex<double>, 3> dymat_q;
    NDArray<std::complex<double>, 2> dymat_tmp;
    NDArray<std::complex<double>, 3> dymat_new;

    const auto nk1 = kmesh_coarse->nk_i[0];
    const auto nk2 = kmesh_coarse->nk_i[1];
    const auto nk3 = kmesh_coarse->nk_i[2];

    MatrixXcd dymat_tmp_mode(ns, ns);
    MatrixXcd dymat_tmp_alphamu(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    std::fstream fin_strain_mode_coupling;
    int nmode;
    std::vector<std::string> mode_list;
    std::vector<double> smag_list;
    std::vector<double> weight_list;
    std::vector<std::string> filename_list;

    double smag_tmp, weight_tmp;
    std::string mode_tmp, filename_tmp;

    NDArray<double, 4> dphi2_dumn_realspace_in;
    NDArray<double, 4> dphi2_dumn_realspace_symm;
    MatrixXd dphi2_dumn_realspace_tmp(natmin3, nat3);
    Matrix3i exist_in;
    Matrix3d weight_sum;
    NDArray<int, 4> count_tmp;

    dphi2_dumn_realspace_in.resize(3, 3, natmin3, nat3);
    dphi2_dumn_realspace_symm.resize(3, 3, natmin3, nat3);
    count_tmp.resize(3, 3, natmin3, nat3);

    NDArray<std::complex<double>, 3> del_v2_strain_from_cubic_alphamu;
    del_v2_strain_from_cubic_alphamu.resize(9, nk, ns * ns);

    exist_in.setZero();
    weight_sum.setZero();
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            for (i1 = 0; i1 < natmin3; i1++) {
                std::fill_n(dphi2_dumn_realspace_in[ixyz1][ixyz2][i1], nat3, 0.0);
                std::fill_n(dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1], nat3, 0.0);
                std::fill_n(count_tmp[ixyz1][ixyz2][i1], nat3, 0.0);
            }
        }
    }

    fin_strain_mode_coupling.open(strain_ifc_dir + "strain_harmonic.in");

    if (!fin_strain_mode_coupling) {
        exit("calculate_delv2_delumn_finite_difference", "strain_harmonic.in not found");
    }

    mode_list.clear();
    smag_list.clear();
    weight_list.clear();
    filename_list.clear();

    nmode = 0;
    while (true) {
        if (fin_strain_mode_coupling >> mode_tmp >> smag_tmp >> weight_tmp >> filename_tmp) {
            mode_list.push_back(mode_tmp);
            smag_list.push_back(smag_tmp);
            weight_list.push_back(weight_tmp);
            filename_list.push_back(filename_tmp);
            nmode++;
        } else {
            break;
        }
    }

    std::vector<std::vector<FcsArrayWithCell>> fc2_deformed(nmode);

    for (imode = 0; imode < nmode; imode++) {
        fcs_phonon_.get_fcs_from_file(strain_ifc_dir + filename_list[imode], 0, fc2_deformed[imode]);
        fcs_phonon_.replicate_force_constant(&system_, fc2_deformed[imode]);
    }

    weight_sum.setZero();

    for (imode = 0; imode < nmode; imode++) {
        if (mode_list[imode] == "xx") {
            ixyz1 = ixyz2 = 0;
        } else if (mode_list[imode] == "yy") {
            ixyz1 = ixyz2 = 1;
        } else if (mode_list[imode] == "zz") {
            ixyz1 = ixyz2 = 2;
        } else if (mode_list[imode] == "xy") {
            ixyz1 = 0;
            ixyz2 = 1;
        } else if (mode_list[imode] == "yz") {
            ixyz1 = 1;
            ixyz2 = 2;
        } else if (mode_list[imode] == "zx") {
            ixyz1 = 2;
            ixyz2 = 0;
        } else {
            exit("calculate_delv2_delumn_finite_difference", "Invalid name of strain mode in strain_harmonic.in.");
        }

        dphi2_dumn_realspace_tmp.setZero();

        for (const auto &it: fc2_deformed[imode]) {
            index2 = system_.get_map_p2s(0)[it.pairs[1].index / 3][it.pairs[1].tran];
            dphi2_dumn_realspace_tmp(it.pairs[0].index, index2 * 3 + it.pairs[1].index % 3) += it.fcs_val;
        }
        for (const auto &it: fcs_phonon_.force_constant_with_cell[0]) {
            index2 = system_.get_map_p2s(0)[it.pairs[1].index / 3][it.pairs[1].tran];
            dphi2_dumn_realspace_tmp(it.pairs[0].index, index2 * 3 + it.pairs[1].index % 3) -= it.fcs_val;
        }

        if (ixyz1 == ixyz2) {
            for (i1 = 0; i1 < natmin3; i1++) {
                for (i2 = 0; i2 < nat3; i2++) {
                    dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                }
            }
            weight_sum(ixyz1, ixyz2) += weight_list[imode];
        } else {
            for (i1 = 0; i1 < natmin3; i1++) {
                for (i2 = 0; i2 < nat3; i2++) {
                    dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                    dphi2_dumn_realspace_in[ixyz2][ixyz1][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / smag_list[imode] * weight_list[imode];
                }
            }
            weight_sum(ixyz1, ixyz2) += weight_list[imode];
            weight_sum(ixyz2, ixyz1) += weight_list[imode];
        }
    }

    if (renorm_3to2nd == 2) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) < eps6) {
                    exist_in(ixyz1, ixyz2) = 1;
                } else {
                    exit("calculate_delv2_delumn_finite_difference", "Sum of weights must be 1.");
                }
            }
        }
    } else if (renorm_3to2nd == 3) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) < eps6) {
                    exist_in(ixyz1, ixyz2) = 1;
                } else if (std::fabs(weight_sum(ixyz1, ixyz2)) < eps6) {
                    exist_in(ixyz1, ixyz2) = 0;
                } else {
                    exit("calculate_delv2_delumn_finite_difference", "Sum of weights must be 1 or 0 for each mode.");
                }
            }
        }
    }

    symm_mapping_s.resize(symmetry_.SymmListWithMap_ref.size(), nat);
    symmetry_.make_supercell_mapping_by_symmetry_operations(symm_mapping_s);

    const auto ntran = system_.get_map_p2s(0)[0].size();

    inv_translation_mapping.resize(ntran, ntran);
    symmetry_.make_inverse_translation_mapping(inv_translation_mapping);

    if (renorm_3to2nd == 2) {
        for (isymm = 0; isymm < symmetry_.SymmListWithMap_ref.size(); isymm++) {
            for (iat1 = 0; iat1 < natmin; iat1++) {
                for (ixyz_comb1 = 0; ixyz_comb1 < 81; ixyz_comb1++) {
                    ixyz1 = ixyz_comb1 / 27;
                    ixyz2 = (ixyz_comb1 / 9) % 3;
                    ixyz3 = (ixyz_comb1 / 3) % 3;
                    ixyz4 = ixyz_comb1 % 3;
                    for (ixyz_comb2 = 0; ixyz_comb2 < 81; ixyz_comb2++) {
                        ixyz1_2 = ixyz_comb2 / 27;
                        ixyz2_2 = (ixyz_comb2 / 9) % 3;
                        ixyz3_2 = (ixyz_comb2 / 3) % 3;
                        ixyz4_2 = ixyz_comb2 % 3;

                        iat1_2 = symmetry_.SymmListWithMap_ref[isymm].mapping[iat1];

                        for (i1 = 0; i1 < ntran; i1++) {
                            if (system_.get_map_p2s(0)[iat1_2][i1] ==
                                symm_mapping_s[isymm][system_.get_map_p2s(0)[iat1][0]])
                            {
                                itran1 = i1;
                            }
                        }

                        for (iat2 = 0; iat2 < nat; iat2++) {
                            iat2_2 = symm_mapping_s[isymm][iat2];
                            iat2_2_prim = system_.get_map_s2p(0)[iat2_2].atom_num;
                            itran2 = system_.get_map_s2p(0)[iat2_2].tran_num;

                            iat2_2 = system_.get_map_p2s(0)[iat2_2_prim][inv_translation_mapping[itran1][itran2]];

                            dphi2_dumn_realspace_symm[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2][iat2_2 * 3 + ixyz4_2] +=
                                dphi2_dumn_realspace_in[ixyz1][ixyz2][iat1 * 3 + ixyz3][iat2 * 3 + ixyz4] *
                                symmetry_.SymmListWithMap_ref[isymm].rot[ixyz1_2 * 3 + ixyz1] *
                                symmetry_.SymmListWithMap_ref[isymm].rot[ixyz2_2 * 3 + ixyz2] *
                                symmetry_.SymmListWithMap_ref[isymm].rot[ixyz3_2 * 3 + ixyz3] *
                                symmetry_.SymmListWithMap_ref[isymm].rot[ixyz4_2 * 3 + ixyz4];
                        }
                    }
                }
            }
        }

        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                for (i1 = 0; i1 < natmin * 3; i1++) {
                    for (i2 = 0; i2 < nat * 3; i2++) {
                        dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] /= symmetry_.SymmListWithMap_ref.size();
                    }
                }
            }
        }
    } else if (renorm_3to2nd == 3) {
        int mapping_xyz[3];
        for (isymm = 0; isymm < symmetry_.SymmListWithMap_ref.size(); isymm++) {
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                mapping_xyz[ixyz1] = -1;
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    if (std::fabs(std::fabs(symmetry_.SymmListWithMap_ref[isymm].rot[ixyz1 * 3 + ixyz2]) - 1.0) < eps6)
                    {
                        mapping_xyz[ixyz2] = ixyz1;
                    }
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                if (mapping_xyz[ixyz1] == -1) {
                    exit("calculate_delv2_delumn_finite_difference",
                         "RENORM_3TO2ND == 3 cannot be used for this material.");
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    if (exist_in(ixyz1, ixyz2) == 0) {
                        continue;
                    }

                    for (iat1 = 0; iat1 < natmin; iat1++) {

                        iat1_2 = symmetry_.SymmListWithMap_ref[isymm].mapping[iat1];
                        ixyz1_2 = mapping_xyz[ixyz1];
                        ixyz2_2 = mapping_xyz[ixyz2];

                        for (i1 = 0; i1 < ntran; i1++) {
                            if (system_.get_map_p2s(0)[iat1_2][i1] ==
                                symm_mapping_s[isymm][system_.get_map_p2s(0)[iat1][0]])
                            {
                                itran1 = i1;
                            }
                        }

                        for (iat2 = 0; iat2 < nat; iat2++) {
                            iat2_2 = symm_mapping_s[isymm][iat2];
                            iat2_2_prim = system_.get_map_s2p(0)[iat2_2].atom_num;
                            itran2 = system_.get_map_s2p(0)[iat2_2].tran_num;

                            iat2_2 = system_.get_map_p2s(0)[iat2_2_prim][inv_translation_mapping[itran1][itran2]];

                            for (ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                                for (ixyz4 = 0; ixyz4 < 3; ixyz4++) {
                                    ixyz3_2 = mapping_xyz[ixyz3];
                                    ixyz4_2 = mapping_xyz[ixyz4];

                                    dphi2_dumn_realspace_symm[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2]
                                                             [iat2_2 * 3 + ixyz4_2] +=
                                        dphi2_dumn_realspace_in[ixyz1][ixyz2][iat1 * 3 + ixyz3][iat2 * 3 + ixyz4] *
                                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz1_2 * 3 + ixyz1] *
                                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz2_2 * 3 + ixyz2] *
                                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz3_2 * 3 + ixyz3] *
                                        symmetry_.SymmListWithMap_ref[isymm].rot[ixyz4_2 * 3 + ixyz4];

                                    count_tmp[ixyz1_2][ixyz2_2][iat1_2 * 3 + ixyz3_2][iat2_2 * 3 + ixyz4_2]++;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                for (i1 = 0; i1 < natmin * 3; i1++) {
                    for (i2 = 0; i2 < nat * 3; i2++) {
                        if (count_tmp[ixyz1][ixyz2][i1][i2] == 0) {
                            std::cout << "Warning: dphi2_dumn_realspace[" << ixyz1 << "][" << ixyz2 << "][" << i1
                                      << "][" << i2 << "] is not given\n";
                            std::cout << "The corresponding component is set zero.\n";
                            dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] = 0.0;
                        } else {
                            dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2] /= count_tmp[ixyz1][ixyz2][i1][i2];
                        }
                    }
                }
            }
        }
    }

    symm_mapping_s.clear();
    inv_translation_mapping.clear();

    dymat_q.resize(ns, ns, nk_interpolate);
    dymat_new.resize(ns, ns, nk_interpolate);
    dymat_tmp.resize(ns, ns);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            fc2_tmp.clear();
            for (i1 = 0; i1 < natmin * 3; i1++) {
                for (i2 = 0; i2 < nat * 3; i2++) {
                    FcsClassExtent fce_tmp;
                    fce_tmp.atm1 = i1 / 3;
                    fce_tmp.atm2 = i2 / 3;
                    fce_tmp.xyz1 = i1 % 3;
                    fce_tmp.xyz2 = i2 % 3;
                    fce_tmp.cell_s = 0;
                    fce_tmp.fcs_val = dphi2_dumn_realspace_symm[ixyz1][ixyz2][i1][i2];
                    fc2_tmp.push_back(fce_tmp);
                }
            }

            for (ik = 0; ik < nk_interpolate; ik++) {
                dynamical_.calc_analytic_k(kmesh_coarse->xk[ik], fc2_tmp, dymat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        dymat_q[is1][is2][ik] = dymat_tmp[is1][is2];
                    }
                }
            }

            fourier_dymat_k_to_r(nk1, nk2, nk3, ns, dymat_q, dymat_new);

            auto &per_strain = del_v2_del_umn[ixyz1 * 3 + ixyz2];
            for (ik = 0; ik < static_cast<int>(nk); ik++) {
                r2q(kmesh_dense->xk[ik], nk1, nk2, nk3, ns, mindist_list, dymat_new, dymat_tmp);

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        del_v2_strain_from_cubic_alphamu[ixyz1 * 3 + ixyz2][ik][is * ns + js] = dymat_tmp[is][js];
                    }
                }

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        evec_tmp(is, js) = evec_harmonic[ik][js][is];
                        dymat_tmp_alphamu(is, js) = dymat_tmp[is][js];
                    }
                }
                dymat_tmp_mode = evec_tmp.adjoint() * dymat_tmp_alphamu * evec_tmp;

                for (is = 0; is < ns; is++) {
                    for (js = 0; js < ns; js++) {
                        per_strain(ik, is * ns + js) = dymat_tmp_mode(is, js);
                    }
                }
            }
        }
    }

    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    const auto is_acoustic_gamma = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        auto &per_strain = del_v2_del_umn[ixyz1];
        for (is = 0; is < ns; is++) {
            if (!is_acoustic_gamma[is]) {
                continue;
            }
            for (js = 0; js < ns; js++) {
                per_strain(0, is * ns + js) = complex_zero;
                per_strain(0, js * ns + is) = complex_zero;
            }
        }
    }

    dphi2_dumn_realspace_symm.clear();
    dphi2_dumn_realspace_in.clear();
    count_tmp.clear();

    dymat_q.clear();
    dymat_tmp.clear();
    dymat_new.clear();
}
