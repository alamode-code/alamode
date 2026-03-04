#include "ifc_derivative.h"
#include <algorithm>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "mpi_common.h"
#include "scph.h"
#include "system.h"

using namespace PHON_NS;

namespace
{
thread_local bool g_skip_v1_impl_compare = false;

bool use_legacy_ifc_derivative_impl()
{
    const auto *env = std::getenv("ALAMODE_USE_LEGACY_IFC_DERIVATIVE");
    if (!env) {
        return false;
    }

    const std::string value(env);
    return !(value.empty() || value == "0" || value == "false" || value == "False" || value == "FALSE");
}

struct ScopedSkipV1Compare
{
    ScopedSkipV1Compare()
    {
        g_skip_v1_impl_compare = true;
    }

    ~ScopedSkipV1Compare()
    {
        g_skip_v1_impl_compare = false;
    }
};

bool are_same_fcs_array_with_cell_verbose(const std::vector<FcsArrayWithCell> &lhs,
                                          const std::vector<FcsArrayWithCell> &rhs, std::string &mismatch_message)
{
    constexpr double tol_fcs_val = eps12;
    constexpr double tol_fcs_emit = eps15 * 10.0;

    mismatch_message.clear();

    auto filter_tiny_entries = [&](const std::vector<FcsArrayWithCell> &src) {
        std::vector<FcsArrayWithCell> out;
        out.reserve(src.size());
        for (const auto &entry: src) {
            if (std::abs(entry.fcs_val) > tol_fcs_emit) {
                out.push_back(entry);
            }
        }
        return out;
    };

    const std::vector<FcsArrayWithCell> *lhs_cmp = &lhs;
    const std::vector<FcsArrayWithCell> *rhs_cmp = &rhs;
    std::vector<FcsArrayWithCell> lhs_filtered, rhs_filtered;

    if (lhs.size() != rhs.size()) {
        std::cout << "  Warning: size mismatch between two FcsArrayWithCell arrays: old=" << lhs.size()
                  << ", new=" << rhs.size() << ". The entries with tiny fcs_val (|fcs_val|<=" << tol_fcs_emit
                  << ") are filtered for the comparison.\n";
        lhs_filtered = filter_tiny_entries(lhs);
        rhs_filtered = filter_tiny_entries(rhs);

        std::cout << "  After filtering tiny entries, size of old array: " << lhs_filtered.size()
                  << ", size of new array: " << rhs_filtered.size() << ".\n";

        if (lhs_filtered.size() != rhs_filtered.size()) {
            mismatch_message = "size mismatch: old=" + std::to_string(lhs.size()) +
                               ", new=" + std::to_string(rhs.size()) +
                               " (after tiny-filter: old=" + std::to_string(lhs_filtered.size()) +
                               ", new=" + std::to_string(rhs_filtered.size()) + ")";
            return false;
        }

        lhs_cmp = &lhs_filtered;
        rhs_cmp = &rhs_filtered;
    }

    for (std::size_t i = 0; i < lhs_cmp->size(); ++i) {
        const auto &a = (*lhs_cmp)[i];
        const auto &b = (*rhs_cmp)[i];

        if (std::abs(a.fcs_val - b.fcs_val) > 1.0e-14) {
            std::ostringstream oss;
            oss << "entry " << i << ": fcs_val mismatch (|old-new|=" << std::scientific << std::setprecision(6)
                << std::abs(a.fcs_val - b.fcs_val) << ", tol=" << std::scientific << std::setprecision(6) << tol_fcs_val
                << ")";
            mismatch_message = oss.str();
            return false;
        }
        if (a.pairs.size() != b.pairs.size()) {
            mismatch_message = "entry " + std::to_string(i) + ": pairs size mismatch";
            return false;
        }
        if (a.atoms_s.size() != b.atoms_s.size()) {
            mismatch_message = "entry " + std::to_string(i) + ": atoms_s size mismatch";
            return false;
        }
        if (a.coords.size() != b.coords.size()) {
            mismatch_message = "entry " + std::to_string(i) + ": coords size mismatch";
            return false;
        }
        if (a.relvecs.size() != b.relvecs.size()) {
            mismatch_message = "entry " + std::to_string(i) + ": relvecs size mismatch";
            return false;
        }
        if (a.relvecs_velocity.size() != b.relvecs_velocity.size()) {
            mismatch_message = "entry " + std::to_string(i) + ": relvecs_velocity size mismatch";
            return false;
        }

        for (std::size_t j = 0; j < a.pairs.size(); ++j) {
            if (a.pairs[j].index != b.pairs[j].index || a.pairs[j].tran != b.pairs[j].tran ||
                a.pairs[j].cell_s != b.pairs[j].cell_s)
            {
                mismatch_message = "entry " + std::to_string(i) + ", pairs[" + std::to_string(j) + "]: mismatch";
                return false;
            }
        }

        for (std::size_t j = 0; j < a.atoms_s.size(); ++j) {
            if (a.atoms_s[j] != b.atoms_s[j]) {
                mismatch_message = "entry " + std::to_string(i) + ", atoms_s[" + std::to_string(j) + "]: mismatch";
                return false;
            }
        }

        for (std::size_t j = 0; j < a.coords.size(); ++j) {
            if (a.coords[j] != b.coords[j]) {
                mismatch_message = "entry " + std::to_string(i) + ", coords[" + std::to_string(j) + "]: mismatch";
                return false;
            }
        }

        for (std::size_t j = 0; j < a.relvecs.size(); ++j) {
            for (int k = 0; k < 3; ++k) {
                if (a.relvecs[j][k] != b.relvecs[j][k]) {
                    mismatch_message = "entry " + std::to_string(i) + ", relvecs[" + std::to_string(j) + "][" +
                                       std::to_string(k) + "]: mismatch";
                    return false;
                }
            }
        }

        for (std::size_t j = 0; j < a.relvecs_velocity.size(); ++j) {
            for (int k = 0; k < 3; ++k) {
                if (a.relvecs_velocity[j][k] != b.relvecs_velocity[j][k]) {
                    mismatch_message = "entry " + std::to_string(i) + ", relvecs_velocity[" + std::to_string(j) + "][" +
                                       std::to_string(k) + "]: mismatch";
                    return false;
                }
            }
        }
    }

    return true;
}
} // namespace

DerivativeIFC::DerivativeIFC(PHON *phon) : Pointers(phon)
{}

void DerivativeIFC::compute_del_v1_del_umn(std::complex<double> **del_v1_del_umn,
                                           const std::complex<double> *const *const *const evec_harmonic) const
{
    if (use_legacy_ifc_derivative_impl()) {
        compute_del_v1_del_umn_legacy(del_v1_del_umn, evec_harmonic);
        return;
    }

    // Calculates the first-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC2 in the unstrained system.
    if (!g_skip_v1_impl_compare && std::getenv("ALAMODE_COMPARE_V1_DERIVATIVE_IMPL") != nullptr) {
        static std::once_flag compare_once;
        std::call_once(compare_once, [&]() {
            std::string report;
            const bool consistent = compare_v1_derivative_implementations(evec_harmonic, report);
            if (mympi->my_rank == 0) {
                std::cout << report << '\n';
            }
            if (!consistent) {
                exit("compute_del_v1_del_umn", "Legacy and unified v1-derivative implementations are inconsistent.");
            }
        });
    }

    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del_v1_del_umn_in_real_space(9, ns);

    del_v1_del_umn_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[0];
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

#pragma omp parallel
    {
        std::vector<FcsArrayWithCell> delta_fcs;

#pragma omp for collapse(2) schedule(dynamic)
        for (int mu = 0; mu < 3; ++mu) {
            for (int nu = 0; nu < 3; ++nu) {
                compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{mu, nu}}, -1.0);
                const int ixyz = mu * 3 + nu;
                for (const auto &entry: delta_fcs) {
                    const int ind1 = entry.pairs[0].index;
                    del_v1_del_umn_in_real_space(ixyz, ind1) += entry.fcs_val;
                }
            }
        }
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 9; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del_v1_del_umn[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                 del_v1_del_umn_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

void DerivativeIFC::compute_del_v1_del_umn_legacy(std::complex<double> **del_v1_del_umn,
                                                  const std::complex<double> *const *const *const evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del_v1_del_umn_in_real_space(9, ns);

    del_v1_del_umn_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[0];
    const auto convmat = system->get_primcell().lattice_vector;
    Eigen::Vector3d vec;

    for (const auto &it: force_constants) {
        const int ind1 = it.pairs[0].index;
        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            vec = convmat * it.relvecs_velocity[0];
            del_v1_del_umn_in_real_space(it.coords[1] * 3 + ixyz1, ind1) += it.fcs_val * vec[ixyz1];
        }
    }

    for (int ixyz = 0; ixyz < 9; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del_v1_del_umn[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                 del_v1_del_umn_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

void DerivativeIFC::compute_del2_v1_del_umn2(std::complex<double> **del2_v1_del_umn2,
                                             const std::complex<double> *const *const *const evec_harmonic) const
{
    if (use_legacy_ifc_derivative_impl()) {
        compute_del2_v1_del_umn2_legacy(del2_v1_del_umn2, evec_harmonic);
        return;
    }

    // Calculates the second-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC3 in the unstrained system.
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del2_v1_del_umn2_in_real_space(81, ns);

    del2_v1_del_umn2_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[1];
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

#pragma omp parallel
    {
        std::vector<FcsArrayWithCell> delta_fcs;
#pragma omp for collapse(4) schedule(dynamic)
        for (int mu1 = 0; mu1 < 3; ++mu1) {
            for (int nu1 = 0; nu1 < 3; ++nu1) {
                for (int mu2 = 0; mu2 < 3; ++mu2) {
                    for (int nu2 = 0; nu2 < 3; ++nu2) {
                        compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{mu1, nu1}, {mu2, nu2}}, -1.0);
                        const int ixyz_comb = mu1 * 27 + nu1 * 9 + mu2 * 3 + nu2;
                        for (const auto &entry: delta_fcs) {
                            const int ind1 = entry.pairs[0].index;
                            del2_v1_del_umn2_in_real_space(ixyz_comb, ind1) += entry.fcs_val;
                        }
                    }
                }
            }
        }
    }


#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 81; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del2_v1_del_umn2[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del2_v1_del_umn2[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                   del2_v1_del_umn2_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

void DerivativeIFC::compute_del2_v1_del_umn2_legacy(std::complex<double> **del2_v1_del_umn2,
                                                    const std::complex<double> *const *const *const evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del2_v1_del_umn2_in_real_space(81, ns);

    del2_v1_del_umn2_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[1];
    const auto convmat = system->get_primcell().lattice_vector;
    Eigen::Vector3d vec1, vec2;

    for (const auto &it: force_constants) {

        const int ind1 = it.pairs[0].index;

        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                int const ixyz_comb = it.coords[1] * 27 + ixyz1 * 9 + it.coords[2] * 3 + ixyz2;
                vec1 = convmat * it.relvecs_velocity[0];
                vec2 = convmat * it.relvecs_velocity[1];
                del2_v1_del_umn2_in_real_space(ixyz_comb, ind1) += it.fcs_val * vec1[ixyz1] * vec2[ixyz2];
            }
        }
    }

    for (int ixyz = 0; ixyz < 81; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del2_v1_del_umn2[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del2_v1_del_umn2[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                   del2_v1_del_umn2_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

void DerivativeIFC::compute_del3_v1_del_umn3(std::complex<double> **del3_v1_del_umn3,
                                             const std::complex<double> *const *const *const evec_harmonic) const
{
    if (use_legacy_ifc_derivative_impl()) {
        compute_del3_v1_del_umn3_legacy(del3_v1_del_umn3, evec_harmonic);
        return;
    }

    // Calculates the third-order derivative of IFC1 with respect to strain in real space and transforms it to the reciprocal space representation.
    // It can be obtained from the IFC4 in the unstrained system.
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del3_v1_del_umn3_in_real_space(729, ns);

    del3_v1_del_umn3_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[2];
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

#pragma omp parallel
    {
        std::vector<FcsArrayWithCell> delta_fcs;
#pragma omp for collapse(6) schedule(dynamic)
        for (int mu1 = 0; mu1 < 3; ++mu1) {
            for (int nu1 = 0; nu1 < 3; ++nu1) {
                for (int mu2 = 0; mu2 < 3; ++mu2) {
                    for (int nu2 = 0; nu2 < 3; ++nu2) {
                        for (int mu3 = 0; mu3 < 3; ++mu3) {
                            for (int nu3 = 0; nu3 < 3; ++nu3) {
                                compute_del_v_strain_in_real_space(fcs_aligned,
                                                                   delta_fcs,
                                                                   {{mu1, nu1}, {mu2, nu2}, {mu3, nu3}},
                                                                   -1.0);
                                const int ixyz_comb = mu1 * 243 + nu1 * 81 + mu2 * 27 + nu2 * 9 + mu3 * 3 + nu3;
                                for (const auto &entry: delta_fcs) {
                                    const int ind1 = entry.pairs[0].index;
                                    del3_v1_del_umn3_in_real_space(ixyz_comb, ind1) += entry.fcs_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }


#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ixyz = 0; ixyz < 729; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del3_v1_del_umn3[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del3_v1_del_umn3[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                   del3_v1_del_umn3_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

void DerivativeIFC::compute_del3_v1_del_umn3_legacy(std::complex<double> **del3_v1_del_umn3,
                                                    const std::complex<double> *const *const *const evec_harmonic) const
{
    const auto natmin = system->get_primcell().number_of_atoms;
    const auto ns = dynamical->neval;
    const auto invsqrt_mass = system->get_invsqrt_mass();
    Eigen::MatrixXd del3_v1_del_umn3_in_real_space(729, ns);

    del3_v1_del_umn3_in_real_space.setZero();

    const auto &force_constants = fcs_phonon->force_constant_with_cell[2];
    const auto convmat = system->get_primcell().lattice_vector;
    Eigen::Vector3d vec1, vec2, vec3;

    for (const auto &it: force_constants) {

        int ind1 = it.pairs[0].index;

        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                for (int ixyz3 = 0; ixyz3 < 3; ixyz3++) {
                    const int ixyz_comb =
                        it.coords[1] * 243 + ixyz1 * 81 + it.coords[2] * 27 + ixyz2 * 9 + it.coords[3] * 3 + ixyz3;
                    vec1 = convmat * it.relvecs_velocity[0];
                    vec2 = convmat * it.relvecs_velocity[1];
                    vec3 = convmat * it.relvecs_velocity[2];
                    del3_v1_del_umn3_in_real_space(ixyz_comb, ind1) +=
                        it.fcs_val * vec1[ixyz1] * vec2[ixyz2] * vec3[ixyz3];
                }
            }
        }
    }

    for (int ixyz = 0; ixyz < 729; ixyz++) {
        for (int is1 = 0; is1 < ns; is1++) {
            del3_v1_del_umn3[ixyz][is1] = 0.0;
            for (int i = 0; i < natmin; i++) {
                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    del3_v1_del_umn3[ixyz][is1] += evec_harmonic[0][is1][i * 3 + ixyz1] * invsqrt_mass[i] *
                                                   del3_v1_del_umn3_in_real_space(ixyz, i * 3 + ixyz1);
                }
            }
        }
    }
}

bool DerivativeIFC::compare_v1_derivative_implementations(const std::complex<double> *const *const *const evec_harmonic,
                                                          std::string &report) const
{
    ScopedSkipV1Compare scoped_skip_v1_compare;

    const auto ns = dynamical->neval;
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<std::vector<std::complex<double>>> legacy_v1_storage(9, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> legacy_v1_ptrs(9);
    std::vector<std::vector<std::complex<double>>> unified_v1_storage(9, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> unified_v1_ptrs(9);
    for (int i = 0; i < 9; ++i) {
        legacy_v1_ptrs[i] = legacy_v1_storage[i].data();
        unified_v1_ptrs[i] = unified_v1_storage[i].data();
    }

    const auto t1 = std::chrono::steady_clock::now();
    compute_del_v1_del_umn_legacy(legacy_v1_ptrs.data(), evec_harmonic);
    const auto t2 = std::chrono::steady_clock::now();
    compute_del_v1_del_umn(unified_v1_ptrs.data(), evec_harmonic);
    const auto t3 = std::chrono::steady_clock::now();

    std::vector<std::vector<std::complex<double>>> legacy_v2_storage(81, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> legacy_v2_ptrs(81);
    std::vector<std::vector<std::complex<double>>> unified_v2_storage(81, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> unified_v2_ptrs(81);
    for (int i = 0; i < 81; ++i) {
        legacy_v2_ptrs[i] = legacy_v2_storage[i].data();
        unified_v2_ptrs[i] = unified_v2_storage[i].data();
    }

    const auto t4 = std::chrono::steady_clock::now();
    compute_del2_v1_del_umn2_legacy(legacy_v2_ptrs.data(), evec_harmonic);
    const auto t5 = std::chrono::steady_clock::now();
    compute_del2_v1_del_umn2(unified_v2_ptrs.data(), evec_harmonic);
    const auto t6 = std::chrono::steady_clock::now();

    std::vector<std::vector<std::complex<double>>> legacy_v3_storage(729, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> legacy_v3_ptrs(729);
    std::vector<std::vector<std::complex<double>>> unified_v3_storage(729, std::vector<std::complex<double>>(ns));
    std::vector<std::complex<double> *> unified_v3_ptrs(729);
    for (int i = 0; i < 729; ++i) {
        legacy_v3_ptrs[i] = legacy_v3_storage[i].data();
        unified_v3_ptrs[i] = unified_v3_storage[i].data();
    }

    const auto t7 = std::chrono::steady_clock::now();
    compute_del3_v1_del_umn3_legacy(legacy_v3_ptrs.data(), evec_harmonic);
    const auto t8 = std::chrono::steady_clock::now();
    compute_del3_v1_del_umn3(unified_v3_ptrs.data(), evec_harmonic);
    const auto t9 = std::chrono::steady_clock::now();

    auto max_diff = [](const auto &lhs, const auto &rhs) {
        double diff_max = 0.0;
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            for (std::size_t j = 0; j < lhs[i].size(); ++j) {
                diff_max = std::max(diff_max, std::abs(lhs[i][j] - rhs[i][j]));
            }
        }
        return diff_max;
    };

    auto max_abs = [](const auto &array) {
        double max_val = 0.0;
        for (const auto &row: array) {
            for (const auto &val: row) {
                max_val = std::max(max_val, std::abs(val));
            }
        }
        return max_val;
    };

    auto first_mismatch = [](const auto &lhs, const auto &rhs, const double tol) {
        std::tuple<int, int, std::complex<double>, std::complex<double>, double> result{-1,
                                                                                        -1,
                                                                                        std::complex<double>(0.0, 0.0),
                                                                                        std::complex<double>(0.0, 0.0),
                                                                                        0.0};

        for (int i = 0; i < static_cast<int>(lhs.size()); ++i) {
            for (int j = 0; j < static_cast<int>(lhs[i].size()); ++j) {
                const double diff = std::abs(lhs[i][j] - rhs[i][j]);
                if (diff > tol) {
                    result = {i, j, lhs[i][j], rhs[i][j], diff};
                    return result;
                }
            }
        }
        return result;
    };

    const double maxdiff_v1 = max_diff(legacy_v1_storage, unified_v1_storage);
    const double maxdiff_v2 = max_diff(legacy_v2_storage, unified_v2_storage);
    const double maxdiff_v3 = max_diff(legacy_v3_storage, unified_v3_storage);
    const bool consistent = (maxdiff_v1 <= eps12) && (maxdiff_v2 <= eps12) && (maxdiff_v3 <= eps12);
    const auto mismatch_v1 = first_mismatch(legacy_v1_storage, unified_v1_storage, eps12);
    const auto mismatch_v2 = first_mismatch(legacy_v2_storage, unified_v2_storage, eps12);
    const auto mismatch_v3 = first_mismatch(legacy_v3_storage, unified_v3_storage, eps12);

    auto wall_ms = [](const std::chrono::steady_clock::time_point &tb,
                      const std::chrono::steady_clock::time_point &te) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(te - tb).count();
    };

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "Comparison for v1-derivative implementations\n";
    oss << "  Consistency: " << (consistent ? "PASS" : "FAIL") << "\n";
    oss << "  max|legacy-unified| for del_v1      = " << std::scientific << std::setprecision(15) << maxdiff_v1 << "\n";
    oss << "  max|legacy-unified| for del2_v1     = " << std::scientific << std::setprecision(15) << maxdiff_v2 << "\n";
    oss << "  max|legacy-unified| for del3_v1     = " << std::scientific << std::setprecision(15) << maxdiff_v3 << "\n";
    oss << "  max|legacy| for del_v1      = " << std::scientific << std::setprecision(15) << max_abs(legacy_v1_storage)
        << "\n";
    oss << "  max|legacy| for del2_v1     = " << std::scientific << std::setprecision(15) << max_abs(legacy_v2_storage)
        << "\n";
    oss << "  max|legacy| for del3_v1     = " << std::scientific << std::setprecision(15) << max_abs(legacy_v3_storage)
        << "\n";
    if (std::get<0>(mismatch_v1) >= 0) {
        oss << "  first mismatch del_v1       : idx=(" << std::get<0>(mismatch_v1) << "," << std::get<1>(mismatch_v1)
            << "), old=" << std::get<2>(mismatch_v1) << ", new=" << std::get<3>(mismatch_v1)
            << ", |diff|=" << std::scientific << std::setprecision(15) << std::get<4>(mismatch_v1) << "\n";
    }
    if (std::get<0>(mismatch_v2) >= 0) {
        oss << "  first mismatch del2_v1      : idx=(" << std::get<0>(mismatch_v2) << "," << std::get<1>(mismatch_v2)
            << "), old=" << std::get<2>(mismatch_v2) << ", new=" << std::get<3>(mismatch_v2)
            << ", |diff|=" << std::scientific << std::setprecision(15) << std::get<4>(mismatch_v2) << "\n";
    }
    if (std::get<0>(mismatch_v3) >= 0) {
        oss << "  first mismatch del3_v1      : idx=(" << std::get<0>(mismatch_v3) << "," << std::get<1>(mismatch_v3)
            << "), old=" << std::get<2>(mismatch_v3) << ", new=" << std::get<3>(mismatch_v3)
            << ", |diff|=" << std::scientific << std::setprecision(15) << std::get<4>(mismatch_v3) << "\n";
    }
    oss << "  wall time [ms] legacy  del_v1       = " << wall_ms(t1, t2) << "\n";
    oss << "  wall time [ms] unified del_v1       = " << wall_ms(t2, t3) << "\n";
    oss << "  wall time [ms] legacy  del2_v1      = " << wall_ms(t4, t5) << "\n";
    oss << "  wall time [ms] unified del2_v1      = " << wall_ms(t5, t6) << "\n";
    oss << "  wall time [ms] legacy  del3_v1      = " << wall_ms(t7, t8) << "\n";
    oss << "  wall time [ms] unified del3_v1      = " << wall_ms(t8, t9) << "\n";
    oss << "  wall time [ms] total (alloc + runs) = " << wall_ms(t0, t9);

    report = oss.str();

    return consistent;
}

void DerivativeIFC::compute_del_v2_del_umn(std::complex<double> ***del_v2_del_umn,
                                           const std::complex<double> *const *const *const evec_harmonic,
                                           const unsigned int nk, double **xk_in) const
{
    using namespace Eigen;

    const auto ns = dynamical->neval;
    int is1, is2;

    std::vector<FcsArrayWithCell> delta_fcs;

    std::complex<double> **mat_tmp;
    allocate(mat_tmp, ns, ns);

    MatrixXcd Dymat(ns, ns);
    MatrixXcd evec_tmp(ns, ns);

    std::vector<FcsArrayWithCell> fcs_aligned;

    fcs_aligned.clear();

    for (const auto &it: fcs_phonon->force_constant_with_cell[1]) {
        fcs_aligned.emplace_back(it);
    }
    sort_by_heading_indices const operator_fcs(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

    for (int ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (int ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            compute_del_v_strain_in_real_space1(fcs_aligned, delta_fcs, ixyz1, ixyz2);

            for (int ik = 0; ik < nk; ik++) {

                dynamical->calc_analytic_k(xk_in[ik], delta_fcs, mat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        Dymat(is1, is2) = mat_tmp[is1][is2];
                        evec_tmp(is1, is2) = evec_harmonic[ik][is2][is1];
                    }
                }
                Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        del_v2_del_umn[ixyz1 * 3 + ixyz2][ik][is1 * ns + is2] = Dymat(is1, is2);
                    }
                }
            }
        }
    }

    deallocate(mat_tmp);
}

void DerivativeIFC::compute_del2_v2_del_umn2(std::complex<double> ***del2_v2_del_umn2,
                                             const std::complex<double> *const *const *const evec_harmonic,
                                             const unsigned int nk, double **xk_in) const
{
    using namespace Eigen;

    const auto ns = dynamical->neval;

    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.clear();
    for (const auto &it: fcs_phonon->force_constant_with_cell[2]) {
        fcs_aligned.emplace_back(it);
    }
    sort_by_heading_indices const operator_fcs(2);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

#pragma omp parallel
    {
        int ixyz;
        int is1, is2;
        std::vector<FcsArrayWithCell> delta_fcs;

        std::complex<double> **mat_tmp;
        allocate(mat_tmp, ns, ns);

        MatrixXcd Dymat(ns, ns);
        MatrixXcd evec_tmp(ns, ns);

#pragma omp for
        for (ixyz = 0; ixyz < 81; ixyz++) {
            int itmp = ixyz;
            const int ixyz22 = itmp % 3;
            itmp /= 3;
            const int ixyz21 = itmp % 3;
            itmp /= 3;
            const int ixyz12 = itmp % 3;
            const int ixyz11 = itmp / 3;

            compute_del_v_strain_in_real_space2(fcs_aligned, delta_fcs, ixyz11, ixyz12, ixyz21, ixyz22);

            for (int ik = 0; ik < nk; ik++) {
                dynamical->calc_analytic_k(xk_in[ik], delta_fcs, mat_tmp);

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        Dymat(is1, is2) = mat_tmp[is1][is2];
                        evec_tmp(is1, is2) = evec_harmonic[ik][is2][is1];
                    }
                }
                Dymat = evec_tmp.adjoint() * Dymat * evec_tmp;

                for (is1 = 0; is1 < ns; is1++) {
                    for (is2 = 0; is2 < ns; is2++) {
                        del2_v2_del_umn2[ixyz][ik][is1 * ns + is2] = Dymat(is1, is2);
                    }
                }
            }
        }

        deallocate(mat_tmp);
    }
}

void DerivativeIFC::compute_del_v3_del_umn(std::complex<double> ****del_v3_del_umn, double **omega2_harmonic,
                                           const std::complex<double> *const *const *const evec_harmonic,
                                           const KpointMeshUniform *kmesh_coarse_in,
                                           const KpointMeshUniform *kmesh_dense_in,
                                           const PhaseFactorStorage *phase_storage_in) const
{
    int ngroup_tmp;
    double *invmass_v3_tmp;
    int **evec_index_v3_tmp;
    std::vector<double> *fcs_group_tmp;
    std::vector<RelativeVector> *relvec_tmp;
    std::complex<double> *phi3_reciprocal_tmp;

    int i;
    int ixyz1, ixyz2;

    double *invsqrt_mass_p;
    allocate(invsqrt_mass_p, system->get_primcell().number_of_atoms);
    for (i = 0; i < system->get_primcell().number_of_atoms; ++i) {
        invsqrt_mass_p[i] = std::sqrt(1.0 / system->get_mass_prim()[i]);
    }

    std::vector<FcsArrayWithCell> delta_fcs;
    std::vector<FcsArrayWithCell> fcs_aligned;
    fcs_aligned.clear();
    for (const auto &it: fcs_phonon->force_constant_with_cell[2]) {
        fcs_aligned.emplace_back(it);
    }
    const sort_by_heading_indices operator_fcs(1);
    boost::sort::block_indirect_sort(fcs_aligned.begin(), fcs_aligned.end(), operator_fcs);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {

            compute_del_v_strain_in_real_space1(fcs_aligned, delta_fcs, ixyz1, ixyz2);

            boost::sort::block_indirect_sort(delta_fcs.begin(), delta_fcs.end());

            anharmonic_core->prepare_group_of_force_constants(delta_fcs, ngroup_tmp, fcs_group_tmp);

            allocate(invmass_v3_tmp, ngroup_tmp);
            allocate(evec_index_v3_tmp, ngroup_tmp, 3);
            allocate(relvec_tmp, ngroup_tmp);
            allocate(phi3_reciprocal_tmp, ngroup_tmp);

            anharmonic_core->prepare_relative_vector(delta_fcs, ngroup_tmp, fcs_group_tmp, relvec_tmp);

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

            scph->compute_V3_elements_for_given_IFCs(del_v3_del_umn[ixyz1 * 3 + ixyz2],
                                                     omega2_harmonic,
                                                     ngroup_tmp,
                                                     fcs_group_tmp,
                                                     relvec_tmp,
                                                     invmass_v3_tmp,
                                                     evec_index_v3_tmp,
                                                     evec_harmonic,
                                                     true,
                                                     kmesh_coarse_in,
                                                     kmesh_dense_in,
                                                     phase_storage_in);

            deallocate(fcs_group_tmp);
            deallocate(invmass_v3_tmp);
            deallocate(evec_index_v3_tmp);
            deallocate(relvec_tmp);
            deallocate(phi3_reciprocal_tmp);
        }
    }
    deallocate(invsqrt_mass_p);
}

bool DerivativeIFC::check_del_v_strain_in_real_space_equivalence(
    const std::vector<FcsArrayWithCell> &fcs_aligned, const std::vector<std::pair<int, int>> &strain_components) const
{
    std::string mismatch_message;
    return check_del_v_strain_in_real_space_equivalence_verbose(fcs_aligned, strain_components, mismatch_message);
}

bool DerivativeIFC::check_del_v_strain_in_real_space_equivalence_verbose(
    const std::vector<FcsArrayWithCell> &fcs_aligned, const std::vector<std::pair<int, int>> &strain_components,
    std::string &mismatch_message) const
{
    std::vector<FcsArrayWithCell> delta_fcs_old;
    std::vector<FcsArrayWithCell> delta_fcs_new;

    if (strain_components.size() == 1) {
        compute_del_v_strain_in_real_space1_legacy(fcs_aligned,
                                                   delta_fcs_old,
                                                   strain_components[0].first,
                                                   strain_components[0].second);
    } else if (strain_components.size() == 2) {
        compute_del_v_strain_in_real_space2_legacy(fcs_aligned,
                                                   delta_fcs_old,
                                                   strain_components[0].first,
                                                   strain_components[0].second,
                                                   strain_components[1].first,
                                                   strain_components[1].second);
    } else {
        exit("check_del_v_strain_in_real_space_equivalence",
             "Legacy comparison is implemented only for m = 1 and m = 2.");
    }

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs_new, strain_components, eps15);

    return are_same_fcs_array_with_cell_verbose(delta_fcs_old, delta_fcs_new, mismatch_message);
}

bool DerivativeIFC::compare_del_v_strain_in_real_space_with_timing(
    const std::vector<FcsArrayWithCell> &fcs_aligned, const std::vector<std::pair<int, int>> &strain_components,
    std::string &report) const
{
    using clock = std::chrono::steady_clock;

    std::vector<FcsArrayWithCell> delta_fcs_old;
    std::vector<FcsArrayWithCell> delta_fcs_new;
    std::string mismatch_message;

    const auto t0 = clock::now();
    if (strain_components.size() == 1) {
        compute_del_v_strain_in_real_space1_legacy(fcs_aligned,
                                                   delta_fcs_old,
                                                   strain_components[0].first,
                                                   strain_components[0].second);
    } else if (strain_components.size() == 2) {
        compute_del_v_strain_in_real_space2_legacy(fcs_aligned,
                                                   delta_fcs_old,
                                                   strain_components[0].first,
                                                   strain_components[0].second,
                                                   strain_components[1].first,
                                                   strain_components[1].second);
    } else {
        exit("compare_del_v_strain_in_real_space_with_timing",
             "Legacy timing-comparison is implemented only for m = 1 and m = 2.");
    }
    const auto t1 = clock::now();

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs_new, strain_components, eps15);
    const auto t2 = clock::now();

    const bool consistent = are_same_fcs_array_with_cell_verbose(delta_fcs_old, delta_fcs_new, mismatch_message);

    const auto old_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const auto new_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::ostringstream oss;
    oss << "Comparison for compute_del_v_strain_in_real_space (m=" << strain_components.size() << ")\n";
    oss << "  Consistency: " << (consistent ? "PASS" : "FAIL") << "\n";
    if (!consistent) {
        oss << "  Mismatch detail: " << mismatch_message << "\n";
    }
    oss << "  legacy delta_fcs size: " << delta_fcs_old.size() << "\n";
    oss << "  unified delta_fcs size: " << delta_fcs_new.size() << "\n";
    oss << "  wall time [ms] legacy : " << old_ms << "\n";
    oss << "  wall time [ms] unified: " << new_ms;

    report = oss.str();

    return consistent;
}

void DerivativeIFC::compute_del_v_strain_in_real_space1_legacy(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                               std::vector<FcsArrayWithCell> &delta_fcs,
                                                               const int ixyz1, const int ixyz2) const
{
    unsigned int i, j;
    Eigen::Vector3d vec;
    double fcs_tmp = 0.0;

    std::vector<AtomCellSuper> pairs_vec;
    std::vector<int> index_old, index_now;
    std::vector<int> index_with_cell, index_with_cell_old;
    AtomCellSuper pairs_tmp{};

    std::vector<int> relvecs_int_old, relvecs_int_now;
    std::vector<unsigned int> atoms_s_now, atoms_s_old;

    std::vector<Eigen::Vector3d> relvecs_now, relvecs_old;
    std::vector<Eigen::Vector3d> relvecs_vel_now, relvecs_vel_old;

    delta_fcs.clear();

    const auto convmat = system->get_primcell().lattice_vector;
    const auto norder = fcs_aligned[0].pairs.size();
    const auto nelems = norder - 1;

    index_old.clear();
    for (i = 0; i < nelems; ++i) {
        index_old.push_back(-1);
    }
    for (i = 0; i < nelems - 1; ++i) {
        for (j = 0; j < 3; ++j) {
            relvecs_int_old.push_back(1000000);
        }
    }
    index_with_cell.clear();

    relvecs_now.resize(nelems - 1);
    relvecs_old.resize(nelems - 1);
    relvecs_vel_now.resize(nelems - 1);
    relvecs_vel_old.resize(nelems - 1);

    for (i = 0; i < 3 * (norder - 2) + 1; ++i)
        index_with_cell_old.push_back(-1);

    for (const auto &it: fcs_aligned) {
        if (it.pairs[norder - 1].index % 3 != ixyz1) {
            continue;
        }

        index_now.clear();
        relvecs_int_now.clear();
        index_with_cell.clear();
        atoms_s_now.clear();

        index_now.push_back(it.pairs[0].index);
        index_with_cell.push_back(it.pairs[0].index);
        atoms_s_now.emplace_back(it.atoms_s[0]);

        for (i = 1; i < nelems; ++i) {
            index_now.push_back(it.pairs[i].index);
            for (j = 0; j < 3; ++j) {
                relvecs_int_now.push_back(nint(it.relvecs[i - 1][j]));
            }

            index_with_cell.push_back(it.pairs[i].index);
            index_with_cell.push_back(it.pairs[i].tran);
            index_with_cell.push_back(it.pairs[i].cell_s);
            atoms_s_now.emplace_back(it.atoms_s[i]);
            relvecs_now[i - 1] = it.relvecs[i - 1];
            relvecs_vel_now[i - 1] = it.relvecs_velocity[i - 1];
        }

        if ((index_now != index_old) || (relvecs_int_now != relvecs_int_old)) {

            if (index_old[0] != -1) {

                if (std::abs(fcs_tmp) > eps15) {

                    pairs_vec.clear();
                    pairs_tmp.index = index_with_cell_old[0];
                    pairs_tmp.tran = 0;
                    pairs_tmp.cell_s = 0;
                    pairs_vec.push_back(pairs_tmp);
                    for (i = 1; i < nelems; ++i) {
                        pairs_tmp.index = index_with_cell_old[3 * i - 2];
                        pairs_tmp.tran = index_with_cell_old[3 * i - 1];
                        pairs_tmp.cell_s = index_with_cell_old[3 * i];
                        pairs_vec.push_back(pairs_tmp);
                    }
                    delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s_old, relvecs_old, relvecs_vel_old);
                }
            }

            fcs_tmp = 0.0;
            index_old = index_now;
            relvecs_int_old = relvecs_int_now;
            atoms_s_old = atoms_s_now;
            relvecs_old = relvecs_now;
            relvecs_vel_old = relvecs_vel_now;
            index_with_cell_old = index_with_cell;
        }

        vec.setZero();
        vec = convmat * it.relvecs_velocity[norder - 2];
        fcs_tmp += it.fcs_val * vec[ixyz2];
    }

    if (std::abs(fcs_tmp) > eps15) {
        pairs_vec.clear();
        pairs_tmp.index = index_with_cell[0];
        pairs_tmp.tran = 0;
        pairs_tmp.cell_s = 0;
        pairs_vec.push_back(pairs_tmp);
        for (i = 1; i < norder - 1; ++i) {
            pairs_tmp.index = index_with_cell[3 * i - 2];
            pairs_tmp.tran = index_with_cell[3 * i - 1];
            pairs_tmp.cell_s = index_with_cell[3 * i];
            pairs_vec.push_back(pairs_tmp);
        }
        delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s_now, relvecs_now, relvecs_vel_now);
    }
}

void DerivativeIFC::compute_del_v_strain_in_real_space2_legacy(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                               std::vector<FcsArrayWithCell> &delta_fcs,
                                                               const int ixyz11, const int ixyz12, const int ixyz21,
                                                               const int ixyz22) const
{
    unsigned int i, j;
    Eigen::Vector3d vec1, vec2;

    double fcs_tmp = 0.0;

    std::vector<AtomCellSuper> pairs_vec;
    std::vector<int> index_old, index_now;
    std::vector<int> index_with_cell, index_with_cell_old;
    AtomCellSuper pairs_tmp{};
    std::vector<int> relvecs_int_old, relvecs_int_now;
    std::vector<unsigned int> atoms_s_now, atoms_s_old;
    std::vector<Eigen::Vector3d> relvecs_now, relvecs_old;
    std::vector<Eigen::Vector3d> relvecs_vel_now, relvecs_vel_old;

    delta_fcs.clear();

    const auto convmat = system->get_primcell().lattice_vector;
    const auto norder = fcs_aligned[0].pairs.size();
    const auto nelems = norder - 2;

    index_old.clear();
    for (i = 0; i < nelems; ++i) {
        index_old.push_back(-1);
    }
    for (i = 0; i < nelems - 1; ++i) {
        for (j = 0; j < 3; ++j) {
            relvecs_int_old.push_back(1000000);
        }
    }
    index_with_cell.clear();

    relvecs_now.resize(nelems - 1);
    relvecs_old.resize(nelems - 1);
    relvecs_vel_now.resize(nelems - 1);
    relvecs_vel_old.resize(nelems - 1);

    for (i = 0; i < 3 * (norder - 3) + 1; ++i)
        index_with_cell_old.push_back(-1);

    for (const auto &it: fcs_aligned) {

        if (it.coords[norder - 2] != ixyz11 || it.coords[norder - 1] != ixyz21) {
            continue;
        }

        index_now.clear();
        relvecs_int_now.clear();
        index_with_cell.clear();
        atoms_s_now.clear();

        index_now.push_back(it.pairs[0].index);
        index_with_cell.push_back(it.pairs[0].index);
        atoms_s_now.emplace_back(it.atoms_s[0]);

        for (i = 1; i < nelems; ++i) {
            index_now.push_back(it.pairs[i].index);
            for (j = 0; j < 3; ++j) {
                relvecs_int_now.push_back(nint(it.relvecs[i - 1][j]));
            }

            index_with_cell.push_back(it.pairs[i].index);
            index_with_cell.push_back(it.pairs[i].tran);
            index_with_cell.push_back(it.pairs[i].cell_s);
            atoms_s_now.emplace_back(it.atoms_s[i]);
            relvecs_now[i - 1] = it.relvecs[i - 1];
            relvecs_vel_now[i - 1] = it.relvecs_velocity[i - 1];
        }

        if ((index_now != index_old) || (relvecs_int_now != relvecs_int_old)) {

            if (index_old[0] != -1) {

                if (std::abs(fcs_tmp) > eps15) {

                    pairs_vec.clear();
                    pairs_tmp.index = index_with_cell_old[0];
                    pairs_tmp.tran = 0;
                    pairs_tmp.cell_s = 0;
                    pairs_vec.push_back(pairs_tmp);
                    for (i = 1; i < nelems; ++i) {
                        pairs_tmp.index = index_with_cell_old[3 * i - 2];
                        pairs_tmp.tran = index_with_cell_old[3 * i - 1];
                        pairs_tmp.cell_s = index_with_cell_old[3 * i];
                        pairs_vec.push_back(pairs_tmp);
                    }
                    delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s_old, relvecs_old, relvecs_vel_old);
                }
            }

            fcs_tmp = 0.0;
            index_old = index_now;
            relvecs_int_old = relvecs_int_now;
            atoms_s_old = atoms_s_now;
            relvecs_old = relvecs_now;
            relvecs_vel_old = relvecs_vel_now;
            index_with_cell_old = index_with_cell;
        }

        vec1.setZero();
        vec2.setZero();

        vec1 = convmat * it.relvecs_velocity[nelems - 1];
        vec2 = convmat * it.relvecs_velocity[nelems];

        fcs_tmp += it.fcs_val * vec1[ixyz12] * vec2[ixyz22];
    }

    if (std::abs(fcs_tmp) > eps15) {

        pairs_vec.clear();
        pairs_tmp.index = index_with_cell_old[0];
        pairs_tmp.tran = 0;
        pairs_tmp.cell_s = 0;
        pairs_vec.push_back(pairs_tmp);
        for (i = 1; i < nelems; ++i) {
            pairs_tmp.index = index_with_cell_old[3 * i - 2];
            pairs_tmp.tran = index_with_cell_old[3 * i - 1];
            pairs_tmp.cell_s = index_with_cell_old[3 * i];
            pairs_vec.push_back(pairs_tmp);
        }
        delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s_old, relvecs_old, relvecs_vel_old);
    }
}

void DerivativeIFC::compute_del_v_strain_in_real_space(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                       std::vector<FcsArrayWithCell> &delta_fcs,
                                                       const std::vector<std::pair<int, int>> &strain_components,
                                                       const double emit_threshold) const
{

    if (fcs_aligned.empty()) {
        delta_fcs.clear();
        return;
    }

    const auto m = strain_components.size();

    if (m == 0) {
        exit("compute_del_v_strain_in_real_space", "Inconsistent derivative-order input.");
    }

    const auto norder = fcs_aligned[0].pairs.size();

    if (m >= norder) {
        exit("compute_del_v_strain_in_real_space", "Derivative order m must be smaller than IFC order n.");
    }

    for (const auto &comp: strain_components) {
        if (comp.first < 0 || comp.first >= 3 || comp.second < 0 || comp.second >= 3) {
            exit("compute_del_v_strain_in_real_space", "Strain tensor indices (mu,nu) must be 0, 1, or 2.");
        }
    }

    const auto convmat = system->get_primcell().lattice_vector;
    const auto nelems = norder - m;

    delta_fcs.clear();

    std::vector<int> index_prev(nelems, -1), index_curr;
    std::vector<int> relvecs_int_prev(3 * (nelems - 1), 1000000), relvecs_int_curr;
    std::vector<int> index_with_cell_prev(3 * (nelems - 1) + 1, -1), index_with_cell_curr;
    std::vector<unsigned int> atoms_s_prev, atoms_s_curr;
    std::vector<Eigen::Vector3d> relvecs_prev(nelems - 1), relvecs_curr(nelems - 1);
    std::vector<Eigen::Vector3d> relvecs_vel_prev(nelems - 1), relvecs_vel_curr(nelems - 1);

    auto tail_mu_matches = [&](const FcsArrayWithCell &it) {
        for (std::size_t j = 0; j < m; ++j) {
            if (it.coords[nelems + j] != static_cast<unsigned int>(strain_components[j].first)) {
                return false;
            }
        }
        return true;
    };

    auto compute_term = [&](const FcsArrayWithCell &it) {
        double term = it.fcs_val;
        for (std::size_t j = 0; j < m; ++j) {
            const auto nu = strain_components[j].second;
            const Eigen::Vector3d vec = convmat * it.relvecs_velocity[(nelems - 1) + j];
            term *= vec[nu];
        }
        return term;
    };

    AtomCellSuper pairs_tmp{};
    std::vector<AtomCellSuper> pairs_vec;
    double fcs_tmp = 0.0;
    bool has_group = false;

    auto emit_group = [&](const std::vector<int> &index_with_cell_ref,
                          const std::vector<unsigned int> &atoms_s_ref,
                          const std::vector<Eigen::Vector3d> &relvecs_ref,
                          const std::vector<Eigen::Vector3d> &relvecs_vel_ref) {
        if (((emit_threshold >= 0.0) && (std::abs(fcs_tmp) <= emit_threshold)) || index_with_cell_ref.empty() ||
            index_with_cell_ref[0] < 0)
        {
            return;
        }

        pairs_vec.clear();
        pairs_tmp.index = index_with_cell_ref[0];
        pairs_tmp.tran = 0;
        pairs_tmp.cell_s = 0;
        pairs_vec.push_back(pairs_tmp);

        for (std::size_t i = 1; i < nelems; ++i) {
            pairs_tmp.index = index_with_cell_ref[3 * i - 2];
            pairs_tmp.tran = index_with_cell_ref[3 * i - 1];
            pairs_tmp.cell_s = index_with_cell_ref[3 * i];
            pairs_vec.push_back(pairs_tmp);
        }

        delta_fcs.emplace_back(fcs_tmp, pairs_vec, atoms_s_ref, relvecs_ref, relvecs_vel_ref);
    };

    for (const auto &it: fcs_aligned) {

        if (!tail_mu_matches(it)) {
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
                emit_group(index_with_cell_prev, atoms_s_prev, relvecs_prev, relvecs_vel_prev);
            }

            fcs_tmp = 0.0;
            index_prev = index_curr;
            relvecs_int_prev = relvecs_int_curr;
            atoms_s_prev = atoms_s_curr;
            relvecs_prev = relvecs_curr;
            relvecs_vel_prev = relvecs_vel_curr;
            index_with_cell_prev = index_with_cell_curr;
            has_group = true;
        }

        fcs_tmp += compute_term(it);
    }

    if (!has_group) {
        return;
    }

    emit_group(index_with_cell_curr, atoms_s_curr, relvecs_curr, relvecs_vel_curr);
}

void DerivativeIFC::compute_del_v_strain_in_real_space1(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                        std::vector<FcsArrayWithCell> &delta_fcs, const int ixyz1,
                                                        const int ixyz2) const
{
    if (use_legacy_ifc_derivative_impl()) {
        compute_del_v_strain_in_real_space1_legacy(fcs_aligned, delta_fcs, ixyz1, ixyz2);
        return;
    }

    if (std::getenv("ALAMODE_CHECK_STRAIN_RENORM") != nullptr) {
        std::string report;
        const auto same = compare_del_v_strain_in_real_space_with_timing(fcs_aligned, {{ixyz1, ixyz2}}, report);
        if (mympi->my_rank == 0) {
            std::cout << report << '\n';
        }
        if (!same) {
            exit("compute_del_v_strain_in_real_space1", "Old and new implementations are not identical.");
        }
    }

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{ixyz1, ixyz2}}, eps15);
}

void DerivativeIFC::compute_del_v_strain_in_real_space2(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                        std::vector<FcsArrayWithCell> &delta_fcs, const int ixyz11,
                                                        const int ixyz12, const int ixyz21, const int ixyz22) const
{
    if (use_legacy_ifc_derivative_impl()) {
        compute_del_v_strain_in_real_space2_legacy(fcs_aligned, delta_fcs, ixyz11, ixyz12, ixyz21, ixyz22);
        return;
    }

    if (std::getenv("ALAMODE_CHECK_STRAIN_RENORM") != nullptr) {
        std::string report;
        const auto same =
            compare_del_v_strain_in_real_space_with_timing(fcs_aligned, {{ixyz11, ixyz12}, {ixyz21, ixyz22}}, report);
        if (mympi->my_rank == 0) {
            std::cout << report << '\n';
        }
        if (!same) {
            exit("compute_del_v_strain_in_real_space2", "Old and new implementations are not identical.");
        }
    }

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{ixyz11, ixyz12}, {ixyz21, ixyz22}}, eps15);
}
