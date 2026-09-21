#include "ifc_derivative.h"
#include <algorithm>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "interpolation.h"
#include "mpi_common.h"
#include "relaxation.h"
#include "scph_v3v4_elements.h"
#include "stage_timer.h"
#include "strain_coupling_io.h"
#include "strain_file_parsers.h"
#include "strain_reference_cell.h"
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

void DerivativeIFC::print_stage(const std::string &label, const double t_start, const bool newline_first) const
{
    print_stage_line(label, stage_clock() - t_start, my_rank_, verbosity_, newline_first);
}

void DerivativeIFC::compute_dV1_dumn(MatrixXcdRowMajor &del_v1_del_umn,
                                     const std::complex<double> *const *const *const evec_harmonic) const
{
    // Calculate the first-order derivative of IFC1 with respect to strain in real space and transform it to the reciprocal space representation.
    // This term corresponds to Eq. (C1) of 10.1103/PhysRevB.106.224104
    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto ns = system_.get_num_modes();
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

    // Correct rotational-sum-rule violations by symmetrizing strain index pairs
    // and removing Gamma acoustic components with the mass-weighted metric
    // (c_kappa = M_kappa / sum M). This removes rigid-rotation and net-force
    // responses while preserving optical generalized forces.
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        for (auto mu = 0; mu < 3; ++mu) {
            for (auto nu = mu + 1; nu < 3; ++nu) {
                const auto avg = 0.5 * (del_v1_del_umn(mu * 3 + nu, is) + del_v1_del_umn(nu * 3 + mu, is));
                del_v1_del_umn(mu * 3 + nu, is) = avg;
                del_v1_del_umn(nu * 3 + mu, is) = avg;
            }
        }
    }
    const auto is_acoustic_proj = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        if (!is_acoustic_proj[is]) continue;
        for (int ixyz = 0; ixyz < 9; ++ixyz) {
            del_v1_del_umn(ixyz, is) = 0.0;
        }
    }
}

void DerivativeIFC::warn_if_strain_force_coupling_allowed(
    MatrixXcdRowMajor &work, const std::complex<double> *const *const *const evec_harmonic) const
{
    // Setting the strain-force coupling to zero (STRAIN_COUPLING bit 2 clear,
    // one-digit values) is valid only when the site symmetries forbid it. alm
    // imposes the crystal symmetry on the harmonic IFCs, so the fc2 estimate
    // is exactly zero in that case; a nonzero estimate proves that the
    // symmetry allows a coupling. The estimate itself is not used.
    compute_dV1_dumn(work, evec_harmonic);
    const auto norm = work.norm();
    if (my_rank_ == 0 && norm > 1.0e-8) {
        std::ostringstream msg;
        msg << "The strain-force coupling is set to zero, but the site symmetries of this crystal allow a\n"
            << " nonzero one (estimate from the harmonic IFCs: " << std::scientific << std::setprecision(2) << norm
            << "). Read it from file (STRAIN_COUPLING bit 2) unless you know it vanishes.";
        warn("set_del_v_relax_cell", msg.str().c_str());
    }
}

void DerivativeIFC::compute_d2V1_dumn2(MatrixXcdRowMajor &del2_v1_del_umn2,
                                       const std::complex<double> *const *const *const evec_harmonic) const
{
    // Calculates the second-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC3 in the unstrained system.
    // This term corresponds to Eq. (C2) of 10.1103/PhysRevB.106.224104
    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto ns = system_.get_num_modes();
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

    // Correct rotational-sum-rule violations by symmetrizing strain index pairs
    // and removing Gamma acoustic components with the mass-weighted metric
    // (c_kappa = M_kappa / sum M). This removes rigid-rotation and net-force
    // responses while preserving optical generalized forces.
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        std::array<std::complex<double>, 81> sym_tmp{};
        for (auto a1 = 0; a1 < 3; ++a1)
            for (auto b1 = 0; b1 < 3; ++b1)
                for (auto a2 = 0; a2 < 3; ++a2)
                    for (auto b2 = 0; b2 < 3; ++b2)
                        sym_tmp[a1 * 27 + b1 * 9 + a2 * 3 + b2] =
                            0.25 * (del2_v1_del_umn2(a1 * 27 + b1 * 9 + a2 * 3 + b2, is) +
                                    del2_v1_del_umn2(b1 * 27 + a1 * 9 + a2 * 3 + b2, is) +
                                    del2_v1_del_umn2(a1 * 27 + b1 * 9 + b2 * 3 + a2, is) +
                                    del2_v1_del_umn2(b1 * 27 + a1 * 9 + b2 * 3 + a2, is));
        for (auto ixyz = 0; ixyz < 81; ++ixyz) del2_v1_del_umn2(ixyz, is) = sym_tmp[ixyz];
    }
    const auto is_acoustic_proj = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        if (!is_acoustic_proj[is]) continue;
        for (int ixyz = 0; ixyz < 81; ++ixyz) {
            del2_v1_del_umn2(ixyz, is) = 0.0;
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
    const auto ns = system_.get_num_modes();
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

    // Correct rotational-sum-rule violations by symmetrizing strain index pairs
    // and removing Gamma acoustic components with the mass-weighted metric
    // (c_kappa = M_kappa / sum M). This removes rigid-rotation and net-force
    // responses while preserving optical generalized forces.
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        std::array<std::complex<double>, 729> sym_tmp{};
        for (auto a1 = 0; a1 < 3; ++a1)
            for (auto b1 = 0; b1 < 3; ++b1)
                for (auto a2 = 0; a2 < 3; ++a2)
                    for (auto b2 = 0; b2 < 3; ++b2)
                        for (auto a3 = 0; a3 < 3; ++a3)
                            for (auto b3 = 0; b3 < 3; ++b3) {
                                auto acc = std::complex<double>(0.0, 0.0);
                                for (auto f = 0; f < 8; ++f) {
                                    const auto p1 = (f & 1) ? b1 * 243 + a1 * 81 : a1 * 243 + b1 * 81;
                                    const auto p2 = (f & 2) ? b2 * 27 + a2 * 9 : a2 * 27 + b2 * 9;
                                    const auto p3 = (f & 4) ? b3 * 3 + a3 : a3 * 3 + b3;
                                    acc += del3_v1_del_umn3(p1 + p2 + p3, is);
                                }
                                sym_tmp[a1 * 243 + b1 * 81 + a2 * 27 + b2 * 9 + a3 * 3 + b3] = 0.125 * acc;
                            }
        for (auto ixyz = 0; ixyz < 729; ++ixyz) del3_v1_del_umn3(ixyz, is) = sym_tmp[ixyz];
    }
    const auto is_acoustic_proj = dynamical_.detect_acoustic_modes_at_gamma(evec_harmonic[0], 0.9, false);
    for (unsigned int is = 0; is < static_cast<unsigned int>(ns); ++is) {
        if (!is_acoustic_proj[is]) continue;
        for (int ixyz = 0; ixyz < 729; ++ixyz) {
            del3_v1_del_umn3(ixyz, is) = 0.0;
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

    const auto ns = system_.get_num_modes();
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

    const auto ns = system_.get_num_modes();

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
    const auto ns = system_.get_num_modes();
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

    double t_v3_build = 0.0;

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

            const auto t_build = stage_clock();
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
            t_v3_build += stage_clock() - t_build;

            fcs_group_tmp.clear();
            invmass_v3_tmp.clear();
            evec_index_v3_tmp.clear();
            relvec_tmp.clear();
        }
    }
    invsqrt_mass_p.clear();
    print_stage_line("dV3/du: 9 x V3 build", t_v3_build, my_rank_, verbosity_, true);
}

void DerivativeIFC::compute_dV_dumn_all_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                   std::vector<DeltaFcsStrainComponents> &groups, const std::size_t m,
                                                   const Eigen::Matrix3d &convmat)
{
    // Compute all 9^m strain derivatives in one pass. Each IFC contributes
    // fcs_val * prod_j vec_j[nu_j] to the 3^m components with its fixed mu indices.
    // fcs_aligned must be sorted by its first n-m indices (n is the IFC order).
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
                                                      const Eigen::VectorXd &sublattice_displacement,
                                                      const double emit_threshold)
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
        [&](const FcsArrayWithCell &it) {
            fcs_tmp += it.fcs_val * sublattice_displacement(it.pairs[norder - 1].index);
        },
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

void DerivativeIFC::compute_dV_ddeform_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                  std::vector<FcsArrayWithCell> &delta_fcs,
                                                  const std::vector<DeformationField> &fields,
                                                  const Eigen::Matrix3d &convmat, const double emit_threshold)
{
    // Contraction of the last m legs with the displacement fields
    // d_lambda(l kappa) = sum_nu u(lambda, nu) R_nu + S(3*kappa + lambda),
    // one field per contracted leg. The relative-vector form of the affine
    // part is exact under the acoustic sum rule at every order m.
    delta_fcs.clear();

    if (fcs_aligned.empty()) return;

    const auto m = fields.size();

    if (m == 0) {
        exit("compute_dV_ddeform_real_space", "Inconsistent contraction-order input.");
    }

    const auto norder = fcs_aligned[0].pairs.size();

    if (m >= norder) {
        exit("compute_dV_ddeform_real_space", "Contraction order m must be smaller than IFC order n.");
    }

    const auto nelems = norder - m;

    auto compute_term = [&](const FcsArrayWithCell &it) {
        double term = it.fcs_val;
        for (std::size_t j = 0; j < m; ++j) {
            const auto index_tail = it.pairs[nelems + j].index;
            const auto mu = index_tail % 3;
            const Eigen::Vector3d vec = convmat * it.relvecs_velocity[(nelems - 1) + j];
            double disp = 0.0;
            for (auto nu = 0; nu < 3; ++nu) {
                disp += fields[j].displacement_gradient(mu, nu) * vec[nu];
            }
            if (fields[j].sublattice_displacement.size() != 0) {
                disp += fields[j].sublattice_displacement(index_tail);
            }
            term *= disp;
        }
        return term;
    };

    double fcs_tmp = 0.0;
    std::vector<AtomCellSuper> pairs_vec;

    scan_fcs_groups(
        fcs_aligned,
        nelems,
        [](const FcsArrayWithCell &) { return true; },
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

void DerivativeIFC::compute_deformed_ifcs(const std::vector<const std::vector<FcsArrayWithCell> *> &fcs_by_order,
                                          const std::size_t target_order_index,
                                          const Eigen::Matrix3d &displacement_gradient,
                                          const Eigen::VectorXd &sublattice_displacement,
                                          const Eigen::Matrix3d &convmat, std::vector<FcsArrayWithCell> &fcs_deformed)
{
    if (target_order_index >= fcs_by_order.size()) {
        exit("compute_deformed_ifcs", "The requested IFC order is not available.");
    }

    fcs_deformed = *fcs_by_order[target_order_index];

    DeformationField field;
    field.displacement_gradient = displacement_gradient;
    field.sublattice_displacement = sublattice_displacement;

    auto inv_factorial = 1.0;
    for (std::size_t mth = 1; target_order_index + mth < fcs_by_order.size(); ++mth) {

        inv_factorial /= static_cast<double>(mth);

        const auto &fcs_higher = *fcs_by_order[target_order_index + mth];
        if (fcs_higher.empty()) continue;

        auto fcs_aligned = fcs_higher;
        sort_by_heading_indices const sorter(static_cast<unsigned int>(mth));
        boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), sorter);

        std::vector<FcsArrayWithCell> delta;
        const std::vector<DeformationField> fields(mth, field);
        compute_dV_ddeform_real_space(fcs_aligned, delta, fields, convmat, eps15);

        for (auto &it: delta) {
            it.fcs_val *= inv_factorial;
        }
        fcs_deformed.insert(fcs_deformed.end(), delta.begin(), delta.end());
    }
}

void DerivativeIFC::compute_dV_dstrain_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                  std::vector<FcsArrayWithCell> &delta_fcs,
                                                  const std::vector<Eigen::Matrix3d> &strain_dirs,
                                                  const Eigen::Matrix3d &convmat, const double emit_threshold)
{
    // Compute real-space directional strain derivatives along strain_dirs[j].
    // With m = strain_dirs.size() and n the IFC order, fcs_aligned must be
    // sorted by its first n-m indices.
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
                                         const strain_coupling::StrainSource &strain_source,
                                         MinimumDistList ***mindist_list, const PhaseFactorCache *phase_cache_in) const
{
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    if (static_cast<int>(ns) != del_v_strain.nmode() || static_cast<int>(nk) != del_v_strain.nk()) {
        exit("set_del_v_relax_cell", "inconsistent dimensions between input sizes and DelVStrainData.");
    }

    auto t_stage = stage_clock();
    switch (renorm_2to1st) {
    case 0:
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        warn_if_strain_force_coupling_allowed(del_v_strain.del_v1, evec_harmonic);
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
        calculate_delv1_delumn_finite_difference(del_v_strain.del_v1, evec_harmonic, strain_source);
        if (my_rank_ == 0) std::cout << "  done!\n";
        break;
    default:
        break;
    }

    print_stage("dV1/du", t_stage);

    t_stage = stage_clock();
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

    print_stage("d2V1/du2, d3V1/du3", t_stage);

    t_stage = stage_clock();
    switch (renorm_3to2nd) {
    case 1:
        if (my_rank_ == 0)
            std::cout << "  - first-order derivatives of harmonic IFCs (from cubic IFCs) ... " << std::flush;

        compute_dV2_dumn(del_v_strain.del_v2, evec_harmonic, nk, kmesh_dense->xk);
        break;
    case 2:
    case 3:
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    from the harmonic IFCs of the strained supercells ... " << std::flush;
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v_strain.del_v2,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 strain_source,
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
    print_stage("dV2/du (total)", t_stage);

    if (my_rank_ == 0)
        std::cout << "  - second-order derivatives of harmonic IFCs (from quartic IFCs) ... " << std::flush;

    t_stage = stage_clock();
    compute_d2V2_dumn2(del_v_strain.del2_v2, evec_harmonic, nk, kmesh_dense->xk);

    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }
    print_stage("d2V2/du2 (from quartic IFCs)", t_stage);
    if (my_rank_ == 0) {
        std::cout << "  - first-order derivatives of cubic IFCs (from quartic IFCs) ... " << std::flush;
    }

    t_stage = stage_clock();
    compute_dV3_dumn(del_v_strain.del_v3, evec_harmonic, kmesh_coarse, kmesh_dense, phase_cache_in);

    if (my_rank_ == 0) {
        std::cout << "  done!\n";
    }
    print_stage("dV3/du (from quartic IFCs, total)", t_stage);
}

void DerivativeIFC::set_del_v_relax_cell_linearQHA(const KpointMeshUniform *kmesh_coarse,
                                                   const KpointMeshUniform *kmesh_dense, const size_t ns,
                                                   DelVStrainData &del_v_strain, double **omega2_harmonic,
                                                   std::complex<double> ***evec_harmonic, const int renorm_2to1st,
                                                   const int renorm_34to1st, const int renorm_3to2nd,
                                                   const strain_coupling::StrainSource &strain_source,
                                                   MinimumDistList ***mindist_list) const
{
    const auto nk = kmesh_dense->nk;
    if (static_cast<int>(ns) != del_v_strain.nmode() || static_cast<int>(nk) != del_v_strain.nk()) {
        exit("set_del_v_relax_cell_linearQHA", "inconsistent dimensions between input sizes and DelVStrainData.");
    }

    if (renorm_2to1st == 0) {
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (set as zero) ... ";
        warn_if_strain_force_coupling_allowed(del_v_strain.del_v1, evec_harmonic);
        del_v_strain.del_v1.setZero();
    } else if (renorm_2to1st == 1) {
        if (my_rank_ == 0) std::cout << "  - first-order derivatives of first-order IFCs (from harmonic IFCs) ... ";
        compute_dV1_dumn(del_v_strain.del_v1, evec_harmonic);

    } else if (renorm_2to1st == 2) {
        if (my_rank_ == 0)
            std::cout << "  - first-order derivatives of first-order IFCs (finite difference method) ... ";
        calculate_delv1_delumn_finite_difference(del_v_strain.del_v1, evec_harmonic, strain_source);
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

        compute_dV2_dumn(del_v_strain.del_v2, evec_harmonic, nk, kmesh_dense->xk);
    } else if (renorm_3to2nd == 2 || renorm_3to2nd == 3) {
        if (my_rank_ == 0) {
            std::cout << "  - first-order derivatives of harmonic IFCs (finite displacement method)\n";
            std::cout << "    from the harmonic IFCs of the strained supercells ...\n";
        }

        calculate_delv2_delumn_finite_difference(omega2_harmonic,
                                                 evec_harmonic,
                                                 del_v_strain.del_v2,
                                                 kmesh_coarse,
                                                 kmesh_dense,
                                                 strain_source,
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

    const auto ns = system_.get_num_modes();

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
    const strain_coupling::StrainSource &strain_source) const
{
    const auto set = load_strain_force_set(strain_source);
    process_strain_force_set(set, del_v1_del_umn, evec_harmonic);
}

strain_coupling::StrainForceSet
DerivativeIFC::load_strain_force_set(const strain_coupling::StrainSource &strain_source) const
{
    strain_coupling::StrainForceSet set;

    if (strain_source.use_file()) {
        try {
            const strain_coupling::StrainCouplingFile container(strain_source.file);
            set = container.read_strain_force();
        } catch (const std::runtime_error &e) {
            exit("calculate_delv1_delumn_finite_difference", e.what());
        }
        return set;
    }

    const auto path = strain_source.ifc_dir + "strain_force.in";
    std::ifstream fin(path);
    if (!fin) {
        exit("calculate_delv1_delumn_finite_difference", "strain_force.in not found");
    }
    try {
        set = strain_parsers::parse_strain_force(fin, system_.get_primcell().number_of_atoms, "strain_force.in");
    } catch (const std::runtime_error &e) {
        exit("calculate_delv1_delumn_finite_difference", e.what());
    }
    set.origin = path;
    set.cell_description = "&reference_cell header found in " + path;

    // Trailing data after the last block was silently skipped before, so it
    // is reported, on one rank, but not fatal.
    if (my_rank_ == 0 && set.trailing_data) {
        warn("calculate_delv1_delumn_finite_difference",
             "Unexpected extra data at the end of strain_force.in is ignored.");
    }
    return set;
}

void DerivativeIFC::process_strain_force_set(const strain_coupling::StrainForceSet &set,
                                             MatrixXcdRowMajor &del_v1_del_umn,
                                             const std::complex<double> *const *const *const evec_harmonic) const
{
    const auto natmin = system_.get_primcell().number_of_atoms;
    auto ns = system_.get_num_modes();

    int ixyz1, ixyz2, ixyz3, ixyz12, ixyz22, ixyz32, i1, i2;
    int iat1, iat2, is1, isymm;
    double dtmp;
    Eigen::Matrix3d weight_sum;

    NDArray<double, 2> del_v1_del_umn_in_real_space;
    NDArray<double, 2> del_v1_del_umn_in_real_space_symm;
    del_v1_del_umn_in_real_space.resize(9, ns);
    del_v1_del_umn_in_real_space_symm.resize(9, ns);

    // When the set names the cell its rows belong to, the rows of every block
    // are mapped onto the atoms of the current primitive cell (which the &cell
    // field may have enlarged); otherwise the rows are those of the current
    // cell.
    const auto &pcell = system_.get_primcell();
    strain_parsers::AtomMatch atom_match;
    if (set.has_cell) {
        try {
            std::vector<std::string> symbols_cur(natmin);
            for (std::size_t iat = 0; iat < natmin; ++iat) symbols_cur[iat] = system_.symbol_kd[pcell.kind[iat]];
            atom_match = strain_parsers::match_atoms(set.cell,
                                                     pcell.lattice_vector,
                                                     pcell.x_cartesian,
                                                     symbols_cur,
                                                     set.origin.c_str());
        } catch (const std::runtime_error &e) {
            exit("calculate_delv1_delumn_finite_difference", e.what());
        }
        if (my_rank_ == 0) {
            const auto flags = std::cout.flags();
            const auto prec = std::cout.precision();
            std::cout << "  " << set.cell_description << "\n"
                      << "    Reference cell : " << set.natom_rows << " atoms,  V = " << std::scientific
                      << std::setprecision(4) << std::fabs(set.cell.lattice.determinant()) << " (a.u.)^3\n"
                      << "    Current cell   : " << natmin << " atoms,  V = " << pcell.volume
                      << " (a.u.)^3   (V(current) / V(reference) = " << std::fixed << std::setprecision(4)
                      << atom_match.ratio << ")\n"
                      << "    The " << set.natom_rows << " force rows per strain mode are mapped onto the " << natmin
                      << " atoms of the current primitive cell.\n"
                      << "    The file on disk is not modified.\n";
            std::cout.flags(flags);
            std::cout.precision(prec);
        }
    } else if (set.natom_rows != natmin) {
        exit("calculate_delv1_delumn_finite_difference",
             "The number of force rows per block does not match the number of atoms of the primitive cell.");
    }

    std::vector<double> rows_now;
    double max_spread = 0.0;

    weight_sum.setZero();

    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        std::fill_n(del_v1_del_umn_in_real_space[ixyz1], ns, 0.0);
    }

    for (const auto &block: set.blocks) {
        if (block.mode == "xx") {
            ixyz1 = ixyz2 = 0;
        } else if (block.mode == "yy") {
            ixyz1 = ixyz2 = 1;
        } else if (block.mode == "zz") {
            ixyz1 = ixyz2 = 2;
        } else if (block.mode == "xy") {
            ixyz1 = 0;
            ixyz2 = 1;
        } else if (block.mode == "yz") {
            ixyz1 = 1;
            ixyz2 = 2;
        } else if (block.mode == "zx") {
            ixyz1 = 2;
            ixyz2 = 0;
        } else {
            exit("calculate_delv1_delumn_finite_difference", "Invalid name of strain mode in strain_force.in.");
        }

        if (block.forces.size() != set.natom_rows * 3) {
            exit("calculate_delv1_delumn_finite_difference",
                 "Inconsistent number of force rows in a strain-force block.");
        }
        if (set.has_cell) {
            max_spread = std::max(max_spread, strain_parsers::expand_atom_rows(atom_match, block.forces, rows_now));
        } else {
            rows_now = block.forces;
        }

        for (iat1 = 0; iat1 < natmin; iat1++) {
            for (ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                dtmp = rows_now[iat1 * 3 + ixyz3];
                del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3] +=
                    dtmp * -1.0 / block.smag * block.weight;

                if (ixyz1 != ixyz2) {
                    del_v1_del_umn_in_real_space[ixyz2 * 3 + ixyz1][iat1 * 3 + ixyz3] =
                        del_v1_del_umn_in_real_space[ixyz1 * 3 + ixyz2][iat1 * 3 + ixyz3];
                }
            }
        }

        if (ixyz1 == ixyz2) {
            weight_sum(ixyz1, ixyz2) += block.weight;
        } else {
            weight_sum(ixyz1, ixyz2) += block.weight;
            weight_sum(ixyz2, ixyz1) += block.weight;
        }
    }

    if (set.has_cell && max_spread > 1.0e-8 && my_rank_ == 0) {
        std::ostringstream os;
        os << "The reference cell of " << set.origin << " is larger than the primitive cell of this run;\n"
           << " the force rows of translation-equivalent atoms are averaged. Largest spread: " << std::scientific
           << std::setprecision(2) << max_spread << " eV/Angstrom.\n"
           << " A large spread means the reference data does not have the translational symmetry assumed here.";
        warn("calculate_delv1_delumn_finite_difference", os.str().c_str());
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
    const KpointMeshUniform *kmesh_dense, const strain_coupling::StrainSource &strain_source,
    MinimumDistList ***mindist_list) const
{
    std::vector<std::vector<FcsArrayWithCell>> fc2_deformed;
    const auto set = load_strain_harmonic_set(strain_source, fc2_deformed);
    process_strain_harmonic_set(set.entries,
                                fc2_deformed,
                                omega2_harmonic,
                                evec_harmonic,
                                del_v2_del_umn,
                                kmesh_coarse,
                                kmesh_dense,
                                mindist_list);
}

strain_coupling::StrainHarmonicSet
DerivativeIFC::load_strain_harmonic_set(const strain_coupling::StrainSource &strain_source,
                                        std::vector<std::vector<FcsArrayWithCell>> &fc2_deformed) const
{
    strain_coupling::StrainHarmonicSet set;

    if (strain_source.use_file()) {
        // Every embedded entry must be the supercell of the harmonic force
        // constants of this run, deformed by its own strain mode, atom by
        // atom: replicate_force_constant relies on that index correspondence.
        try {
            const strain_coupling::StrainCouplingFile container(strain_source.file);
            set = container.read_strain_harmonic();
            const auto &scell = system_.get_supercell(0);
            std::vector<std::string> symbols(scell.number_of_atoms);
            for (std::size_t i = 0; i < scell.number_of_atoms; ++i) symbols[i] = system_.symbol_kd[scell.kind[i]];
            fc2_deformed.assign(set.entries.size(), {});
            for (std::size_t imode = 0; imode < set.entries.size(); ++imode) {
                const auto &entry = set.entries[imode];
                const auto what = strain_source.file + ":" + entry.label;
                strain_parsers::check_strained_supercell(entry.supercell,
                                                         entry.mode,
                                                         entry.smag,
                                                         scell.lattice_vector,
                                                         scell.x_fractional,
                                                         symbols,
                                                         what.c_str());
                container.load_harmonic_fc2(entry, fcs_phonon_, fc2_deformed[imode]);
                fcs_phonon_.replicate_force_constant(&system_, fc2_deformed[imode]);
            }
        } catch (const std::runtime_error &e) {
            exit("calculate_delv2_delumn_finite_difference", e.what());
        }
        if (my_rank_ == 0) {
            std::cout << "\n    " << set.entries.size() << " strained supercells read from " << strain_source.file
                      << ":/StrainHarmonic;\n    each one matches the supercell of the harmonic force constants.\n";
        }
        return set;
    }

    const auto path = strain_source.ifc_dir + "strain_harmonic.in";
    std::ifstream fin(path);
    if (!fin) {
        exit("calculate_delv2_delumn_finite_difference", "strain_harmonic.in not found");
    }
    try {
        set = strain_parsers::parse_strain_harmonic(fin, "strain_harmonic.in");
    } catch (const std::runtime_error &e) {
        exit("calculate_delv2_delumn_finite_difference", e.what());
    }
    set.origin = path;

    // A trailing line with fewer than four tokens (unless it ends exactly at
    // EOF) was silently skipped before; report it on one rank but keep the
    // previous (non-fatal) behavior.
    if (my_rank_ == 0 && set.trailing_data) {
        warn("calculate_delv2_delumn_finite_difference",
             "Unexpected extra data at the end of strain_harmonic.in is ignored.");
    }

    fc2_deformed.assign(set.entries.size(), {});
    for (std::size_t imode = 0; imode < set.entries.size(); ++imode) {
        fcs_phonon_.get_fcs_from_file(strain_source.ifc_dir + set.entries[imode].label, 0, fc2_deformed[imode]);
        fcs_phonon_.replicate_force_constant(&system_, fc2_deformed[imode]);
    }
    return set;
}

void DerivativeIFC::process_strain_harmonic_set(
    const std::vector<strain_coupling::StrainHarmonicEntry> &entries,
    const std::vector<std::vector<FcsArrayWithCell>> &fc2_deformed, double **omega2_harmonic,
    const std::complex<double> *const *const *const evec_harmonic, std::vector<MatrixXcdRowMajor> &del_v2_del_umn,
    const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense, MinimumDistList ***mindist_list) const
{
    using namespace Eigen;

    const auto natmin = system_.get_primcell().number_of_atoms;
    const auto nat = system_.get_supercell(0).number_of_atoms;
    const auto natmin3 = natmin * 3;
    const auto nat3 = nat * 3;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk = kmesh_dense->nk;
    const auto ns = system_.get_num_modes();

    NDArray<int, 2> symm_mapping_s;
    NDArray<int, 2> inv_translation_mapping;

    std::vector<FcsClassExtent> fc2_tmp;

    int ixyz1, ixyz2, ixyz3, ixyz4;
    int ixyz1_2, ixyz2_2, ixyz3_2, ixyz4_2;
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

    const auto nmode = static_cast<int>(entries.size());
    if (fc2_deformed.size() != entries.size()) {
        exit("calculate_delv2_delumn_finite_difference", "Inconsistent number of strained force-constant sets.");
    }

    weight_sum.setZero();

    for (imode = 0; imode < nmode; imode++) {
        if (entries[imode].mode == "xx") {
            ixyz1 = ixyz2 = 0;
        } else if (entries[imode].mode == "yy") {
            ixyz1 = ixyz2 = 1;
        } else if (entries[imode].mode == "zz") {
            ixyz1 = ixyz2 = 2;
        } else if (entries[imode].mode == "xy") {
            ixyz1 = 0;
            ixyz2 = 1;
        } else if (entries[imode].mode == "yz") {
            ixyz1 = 1;
            ixyz2 = 2;
        } else if (entries[imode].mode == "zx") {
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
                        dphi2_dumn_realspace_tmp(i1, i2) / entries[imode].smag * entries[imode].weight;
                }
            }
            weight_sum(ixyz1, ixyz2) += entries[imode].weight;
        } else {
            for (i1 = 0; i1 < natmin3; i1++) {
                for (i2 = 0; i2 < nat3; i2++) {
                    dphi2_dumn_realspace_in[ixyz1][ixyz2][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / entries[imode].smag * entries[imode].weight;
                    dphi2_dumn_realspace_in[ixyz2][ixyz1][i1][i2] +=
                        dphi2_dumn_realspace_tmp(i1, i2) / entries[imode].smag * entries[imode].weight;
                }
            }
            weight_sum(ixyz1, ixyz2) += entries[imode].weight;
            weight_sum(ixyz2, ixyz1) += entries[imode].weight;
        }
    }

    // The set of strain patterns in the file decides the completion: when
    // every component is covered (all weight sums 1) the derivative is
    // averaged over the full point group; when only a subset is given (sums
    // 1 or 0) the missing components are generated from the given ones by
    // the operations whose rotation matrices are signed permutations.
    bool all_patterns = true;
    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            if (std::fabs(weight_sum(ixyz1, ixyz2) - 1.0) < eps6) {
                exist_in(ixyz1, ixyz2) = 1;
            } else if (std::fabs(weight_sum(ixyz1, ixyz2)) < eps6) {
                exist_in(ixyz1, ixyz2) = 0;
                all_patterns = false;
            } else {
                exit("calculate_delv2_delumn_finite_difference",
                     "The weights of the strain patterns must sum to 1 (or 0) for each strain component.");
            }
        }
    }
    if (my_rank_ == 0) {
        std::cout << (all_patterns ? "    all strain components are covered: averaging over the point group\n"
                                   : "    a subset of the strain components is covered: completing by symmetry\n");
    }

    const auto t_symm = stage_clock();
    symm_mapping_s.resize(symmetry_.SymmListWithMap_ref.size(), nat);
    symmetry_.make_supercell_mapping_by_symmetry_operations(symm_mapping_s);

    const auto ntran = system_.get_map_p2s(0)[0].size();

    inv_translation_mapping.resize(ntran, ntran);
    symmetry_.make_inverse_translation_mapping(inv_translation_mapping);

    if (all_patterns) {
        // Symmetrize over reference operations sequentially, parallelizing iat1:
        // each atom writes disjoint output rows. Preserve contribution order
        // within each element and skip exactly zero factors.
        const auto nsymm = static_cast<int>(symmetry_.SymmListWithMap_ref.size());
        const auto &map_p2s = system_.get_map_p2s(0);
        const auto &map_s2p = system_.get_map_s2p(0);
        for (int isym = 0; isym < nsymm; isym++) {
            const auto &symop = symmetry_.SymmListWithMap_ref[isym];
            const double *rot = symop.rot.data();
#pragma omp parallel for schedule(dynamic)
            for (int iatom1 = 0; iatom1 < natmin; iatom1++) {
                const int iatom1_2 = symop.mapping[iatom1];
                int itr1 = 0;
                for (int i = 0; i < ntran; i++) {
                    if (map_p2s[iatom1_2][i] == symm_mapping_s[isym][map_p2s[iatom1][0]]) itr1 = i;
                }
                for (int iatom2 = 0; iatom2 < nat; iatom2++) {
                    const int iatom2_s = symm_mapping_s[isym][iatom2];
                    const int iatom2_2 =
                        map_p2s[map_s2p[iatom2_s].atom_num][inv_translation_mapping[itr1][map_s2p[iatom2_s].tran_num]];

                    for (int comb1 = 0; comb1 < 81; comb1++) {
                        const int a1 = comb1 / 27;
                        const int a2 = (comb1 / 9) % 3;
                        const int a3 = (comb1 / 3) % 3;
                        const int a4 = comb1 % 3;
                        const auto val = dphi2_dumn_realspace_in[a1][a2][iatom1 * 3 + a3][iatom2 * 3 + a4];
                        if (val == 0.0) continue;

                        for (int comb2 = 0; comb2 < 81; comb2++) {
                            const int b1 = comb2 / 27;
                            const int b2 = (comb2 / 9) % 3;
                            const int b3 = (comb2 / 3) % 3;
                            const int b4 = comb2 % 3;
                            const auto r1 = rot[b1 * 3 + a1];
                            const auto r2 = rot[b2 * 3 + a2];
                            const auto r3 = rot[b3 * 3 + a3];
                            const auto r4 = rot[b4 * 3 + a4];
                            if (r1 == 0.0 || r2 == 0.0 || r3 == 0.0 || r4 == 0.0) continue;

                            dphi2_dumn_realspace_symm[b1][b2][iatom1_2 * 3 + b3][iatom2_2 * 3 + b4] +=
                                val * r1 * r2 * r3 * r4;
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
    } else {
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
                         "The strain-harmonic-IFC coupling covers only a subset of the strain components, and\n"
                         " the missing ones cannot be completed by symmetry for this crystal (a rotation matrix\n"
                         " is not a signed permutation in Cartesian coordinates): give all six strain patterns.");
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

    print_stage("dV2/du: symmetrization", t_symm, true);

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
