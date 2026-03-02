#include "ifc_derivative.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "constants.h"
#include "error.h"
#include "mpi_common.h"
#include "system.h"

using namespace PHON_NS;

namespace {
bool are_same_fcs_array_with_cell_verbose(const std::vector<FcsArrayWithCell> &lhs,
                                          const std::vector<FcsArrayWithCell> &rhs,
                                          std::string &mismatch_message)
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
        std::cout << "  Warning: size mismatch between two FcsArrayWithCell arrays: old=" << lhs.size() << ", new="
                  << rhs.size()
                  << ". The entries with tiny fcs_val (|fcs_val|<=" << tol_fcs_emit
                  << ") are filtered for the comparison.\n";
        lhs_filtered = filter_tiny_entries(lhs);
        rhs_filtered = filter_tiny_entries(rhs);

        std::cout << "  After filtering tiny entries, size of old array: " << lhs_filtered.size()
                  << ", size of new array: " << rhs_filtered.size() << ".\n";

        if (lhs_filtered.size() != rhs_filtered.size()) {
            mismatch_message = "size mismatch: old=" + std::to_string(lhs.size()) + ", new=" + std::to_string(rhs.size()) +
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
                << std::abs(a.fcs_val - b.fcs_val) << ", tol=" << std::scientific << std::setprecision(6)
                << tol_fcs_val << ")";
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
                    mismatch_message = "entry " + std::to_string(i) + ", relvecs_velocity[" + std::to_string(j) +
                                       "][" + std::to_string(k) + "]: mismatch";
                    return false;
                }
            }
        }
    }

    return true;
}
} // namespace

DerivativeIFC::DerivativeIFC(PHON *phon): Pointers(phon)
{}

bool DerivativeIFC::check_del_v_strain_in_real_space_equivalence(
    const std::vector<FcsArrayWithCell> &fcs_aligned,
    const std::vector<std::pair<int, int>> &strain_components) const
{
    std::string mismatch_message;
    return check_del_v_strain_in_real_space_equivalence_verbose(fcs_aligned, strain_components, mismatch_message);
}

bool DerivativeIFC::check_del_v_strain_in_real_space_equivalence_verbose(
    const std::vector<FcsArrayWithCell> &fcs_aligned,
    const std::vector<std::pair<int, int>> &strain_components,
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

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs_new, strain_components);

    return are_same_fcs_array_with_cell_verbose(delta_fcs_old, delta_fcs_new, mismatch_message);
}

void DerivativeIFC::compute_del_v_strain_in_real_space1_legacy(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                               std::vector<FcsArrayWithCell> &delta_fcs,
                                                               const int ixyz1,
                                                               const int ixyz2) const
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
                                                               const int ixyz11,
                                                               const int ixyz12,
                                                               const int ixyz21,
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
                                                       const std::vector<std::pair<int, int>> &strain_components) const
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
        if (std::abs(fcs_tmp) <= eps15 || index_with_cell_ref.empty() || index_with_cell_ref[0] < 0) {
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
                                                        std::vector<FcsArrayWithCell> &delta_fcs,
                                                        const int ixyz1,
                                                        const int ixyz2) const
{
    if (std::getenv("ALAMODE_CHECK_STRAIN_RENORM") != nullptr) {
        std::string mismatch_message;
        const auto same =
            check_del_v_strain_in_real_space_equivalence_verbose(fcs_aligned, {{ixyz1, ixyz2}}, mismatch_message);
        if (!same && mympi->my_rank == 0) {
            std::cout << "Error in strain IFC renormalization check (m=1): " << mismatch_message << '\n';
        }
        if (!same) {
            exit("compute_del_v_strain_in_real_space1", "Old and new implementations are not identical.");
        }
    }

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{ixyz1, ixyz2}});
}

void DerivativeIFC::compute_del_v_strain_in_real_space2(const std::vector<FcsArrayWithCell> &fcs_aligned,
                                                        std::vector<FcsArrayWithCell> &delta_fcs,
                                                        const int ixyz11,
                                                        const int ixyz12,
                                                        const int ixyz21,
                                                        const int ixyz22) const
{
    if (std::getenv("ALAMODE_CHECK_STRAIN_RENORM") != nullptr) {
        std::string mismatch_message;
        const auto same = check_del_v_strain_in_real_space_equivalence_verbose(
            fcs_aligned, {{ixyz11, ixyz12}, {ixyz21, ixyz22}}, mismatch_message);
        if (!same && mympi->my_rank == 0) {
            std::cout << "Error in strain IFC renormalization check (m=2): " << mismatch_message << '\n';
        }
        if (!same) {
            exit("compute_del_v_strain_in_real_space2", "Old and new implementations are not identical.");
        }
    }

    compute_del_v_strain_in_real_space(fcs_aligned, delta_fcs, {{ixyz11, ixyz12}, {ixyz21, ixyz22}});
}
