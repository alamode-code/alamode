/*
 scph_result_io.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph_result_io.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mpi.h>
#include <sstream>
#include <utility>
#include "constants.h"
#include "error.h"
#include "fcs_hdf5_schema.h"
#include "hdf5_parser.h"

using namespace PHON_NS;

struct ScphResultIOH5::Impl
{
    std::string filename;

    // Map each requested temperature to its row in the file's grid;
    // a missing temperature is fatal (mirrors the legacy text loader).
    auto temperature_rows(const HighFive::File &fh,
                          const std::vector<double> &temps_requested) const -> std::vector<size_t>
    {
        const auto temps_file = H5Easy::load<std::vector<double>>(fh, "/settings/temperatures");
        std::vector<size_t> rows;
        rows.reserve(temps_requested.size());
        for (const auto t: temps_requested) {
            bool found = false;
            for (size_t i = 0; i < temps_file.size(); ++i) {
                if (std::abs(temps_file[i] - t) < eps6) {
                    rows.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                exit("scph_result_io", "The temperature information is not consistent");
            }
        }
        return rows;
    }

    static auto write_dymat_dataset(HighFive::File &fh, const std::string &path,
                                    const std::complex<double> *const *const *const *dymat, const size_t nt,
                                    const size_t ns, const size_t ncell) -> void
    {
        auto dset = h5_create_dataset_compressed<std::complex<double>>(fh, path, {nt, ns, ns, ncell}, 1);
        dset.write_raw(&dymat[0][0][0][0]);
    }
};

ScphResultIOH5::ScphResultIOH5(std::string filename) : impl(std::make_unique<Impl>())
{
    impl->filename = std::move(filename);
}

ScphResultIOH5::~ScphResultIOH5() = default;

bool ScphResultIOH5::is_restartable() const
{
    if (!std::filesystem::exists(impl->filename)) return false;
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    if (!fh.hasAttribute("schema")) return false;
    std::string schema_name;
    fh.getAttribute("schema").read(schema_name);
    if (schema_name != h5_schema_scph_state) return false;
    int complete = 0;
    if (fh.hasAttribute("complete")) fh.getAttribute("complete").read(complete);
    return complete == 1;
}

void ScphResultIOH5::validate_settings(const ScphSettingsH5 &settings) const
{
    using namespace H5Easy;
    const File fh(impl->filename, File::ReadOnly);
    check_h5_schema(fh, h5_schema_scph_state, h5_version_scph_state);

    const auto kmesh_interp = load<std::vector<unsigned int>>(fh, "/settings/kmesh_interpolate");
    const auto kmesh_dense = load<std::vector<unsigned int>>(fh, "/settings/kmesh_dense");
    for (auto i = 0; i < 3; ++i) {
        if (kmesh_interp[i] != settings.kmesh_interpolate[i]) {
            exit("scph_result_io", "The number of KMESH_INTERPOLATE is not consistent");
        }
        if (kmesh_dense[i] != settings.kmesh_dense[i]) {
            exit("scph_result_io", "The number of KMESH_SCPH (KMESH_QHA) is not consistent");
        }
    }
    // The stored corrections are taken against the harmonic matrix of the writing run,
    // which depends on its nonanalytic treatment.
    if (load<int>(fh, "/settings/nonanalytic") != settings.nonanalytic) {
        exit("scph_result_io", "The NONANALYTIC tag is not consistent with the restart file");
    }
    if (load<int>(fh, "/settings/selfenergy_offdiag") != settings.selfenergy_offdiag) {
        exit("scph_result_io", "The SELF_OFFDIAG tag is not consistent");
    }
    // Absent relax_str: a file written before the tag was stored (no relaxation).
    const auto relax_str_file = fh.exist("/settings/relax_str") ? load<int>(fh, "/settings/relax_str") : 0;
    if ((relax_str_file != 0) != (settings.relax_str != 0)) {
        exit("scph_result_io", "The RELAX_STR tag is not consistent with the restart file");
    }
    // Absent EFIELD: a zero-field run (or a file written before the tag existed).
    std::vector<double> efield_file(3, 0.0);
    if (fh.exist("/settings/efield")) efield_file = load<std::vector<double>>(fh, "/settings/efield");
    if (efield_file.size() != 3) exit("scph_result_io", "/settings/efield of the restart file must have 3 entries");
    for (auto i = 0; i < 3; ++i) {
        if (efield_file[i] != settings.efield[i]) {
            exit("scph_result_io",
                 "The EFIELD tag is not consistent with the restart file. A field sweep is not a restart:\n"
                 " seed the next field from the previous structure with tools/efield_seed.py instead.");
        }
    }
    if (efield_file[0] != 0.0 || efield_file[1] != 0.0 || efield_file[2] != 0.0) {
        if (!fh.exist("/settings/born_charges")) {
            exit("scph_result_io", "The restart file has a nonzero EFIELD but no /settings/born_charges");
        }
        const auto dset = fh.getDataSet("/settings/born_charges");
        std::vector<double> zstar_file(dset.getElementCount());
        dset.read(zstar_file.data());
        auto same = zstar_file.size() == settings.born_charges.size();
        for (size_t i = 0; same && i < zstar_file.size(); ++i) {
            same = std::abs(zstar_file[i] - settings.born_charges[i]) <=
                   1.0e-10 * std::max({std::abs(zstar_file[i]), std::abs(settings.born_charges[i]), 1.0});
        }
        if (!same) {
            exit("scph_result_io", "The Born effective charges (BORNINFO) are not consistent with the restart file");
        }
    }
    // POL_REF and e0 (STRAINFILE /Piezoelectric): absent means zero.
    const auto same_values = [&fh](const std::string &path, const double *expected, const std::size_t n) {
        std::vector<double> stored(n, 0.0);
        if (fh.exist(path)) {
            const auto dset = fh.getDataSet(path);
            if (dset.getElementCount() != n) return false;
            dset.read(stored.data());
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (std::abs(stored[i] - expected[i]) >
                1.0e-10 * std::max({std::abs(stored[i]), std::abs(expected[i]), 1.0}))
                return false;
        }
        return true;
    };
    if (!same_values("/settings/pol_ref", settings.pol_ref.data(), 3)) {
        exit("scph_result_io", "The POL_REF tag is not consistent with the restart file");
    }
    // Under EFIELD the stored tensors are the space-group averaged ones; files written
    // before that averaging hold the raw data and are rejected here.
    const std::string piezo_note =
        "\n (Under EFIELD, e0, B and Lambda are averaged over the space group of the reference"
        "\n structure; a restart file written with the unsymmetrized tensors is not compatible.)";
    if (!same_values("/settings/piezo_clamped_ion", settings.piezo0.data(), 27)) {
        exit(
            "scph_result_io",
            ("The clamped-ion piezoelectric tensor (STRAINFILE /Piezoelectric) is not consistent with the restart file" +
             piezo_note)
                .c_str());
    }
    if (!same_values("/settings/piezo_second_order", settings.piezo2.data(), 243)) {
        exit("scph_result_io",
             ("The second-order clamped-ion piezoelectric tensor (STRAINFILE /Piezoelectric/second_order) is not\n"
              " consistent with the restart file" +
              piezo_note)
                 .c_str());
    }
    // Lambda: absent (in the file or in this run) means zero
    const auto path_lambda = std::string("/settings/born_charge_strain_derivative");
    auto same_lambda = true;
    if (fh.exist(path_lambda)) {
        const auto n = fh.getDataSet(path_lambda).getElementCount();
        auto expected = settings.born_strain;
        if (expected.empty()) expected.assign(n, 0.0);
        same_lambda = expected.size() == n && same_values(path_lambda, expected.data(), n);
    } else {
        same_lambda = std::all_of(settings.born_strain.begin(), settings.born_strain.end(), [](const double x) {
            return x == 0.0;
        });
    }
    if (!same_lambda) {
        exit("scph_result_io",
             ("The strain derivative of the Born charges (STRAINFILE /Piezoelectric/born_charge_strain_derivative)\n"
              " is not consistent with the restart file" +
              piezo_note)
                 .c_str());
    }
    // STRAIN_FC5: absent means 0 (also every file written before the tag existed)
    const auto fc5_file = fh.exist("/settings/strain_fc5") ? load<int>(fh, "/settings/strain_fc5") : 0;
    if (fc5_file != settings.strain_fc5) {
        exit("scph_result_io", "The STRAIN_FC5 tag is not consistent with the restart file");
    }
    if (settings.strain_fc5) {
        const auto diag_file = fh.exist("/settings/strain_fc5_diag") ? load<int>(fh, "/settings/strain_fc5_diag") : 0;
        if (diag_file != settings.strain_fc5_diag) {
            exit("scph_result_io", "The STRAIN_FC5_CHANNELS tag is not consistent with the restart file");
        }
        if (!same_values("/settings/fc5_fingerprint", settings.fc5_fingerprint.data(), settings.fc5_fingerprint.size()))
        {
            exit("scph_result_io", "The quintic IFCs (STRAIN_FC5) are not consistent with the restart file");
        }
    }
}

bool ScphResultIOH5::delta_on_full_harmonic() const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    if (!fh.exist("/settings/delta_baseline")) return false; // legacy file
    const auto baseline = H5Easy::load<int>(fh, "/settings/delta_baseline");
    if (baseline != 1) {
        exit("scph_result_io",
             "Unsupported /settings/delta_baseline in the restart file (written by a newer version?)");
    }
    return true;
}

bool ScphResultIOH5::used_strain_fc5() const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    return fh.exist("/settings/strain_fc5") && H5Easy::load<int>(fh, "/settings/strain_fc5") != 0;
}

void ScphResultIOH5::load_dymat(const std::string &name, const std::vector<double> &temps_requested,
                                const unsigned int ns, const unsigned int ncell,
                                std::complex<double> ****dymat_out) const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    const auto rows = impl->temperature_rows(fh, temps_requested);

    const auto dset = fh.getDataSet("/dymat/" + name);
    const auto dims = dset.getDimensions();
    if (dims.size() != 4 || dims[1] != ns || dims[2] != ns || dims[3] != ncell) {
        exit("scph_result_io", "Unexpected shape of a /dymat dataset in the restart file");
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        dset.select({rows[i], 0, 0, 0}, {1, ns, ns, ncell}).read(&dymat_out[i][0][0][0]);
    }
}

void ScphResultIOH5::load_v0(const std::vector<double> &temps_requested, std::vector<double> &v0_out) const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    const auto rows = impl->temperature_rows(fh, temps_requested);
    const auto v0_file = H5Easy::load<std::vector<double>>(fh, "/V0");
    for (size_t i = 0; i < rows.size(); ++i) {
        v0_out[i] = v0_file[rows[i]];
    }
}

bool ScphResultIOH5::load_structure(const double temp_requested, std::vector<double> &u_tensor_out,
                                    std::vector<double> &u0_out, std::string &spg_label_out, Eigen::Matrix3d &lavec_out,
                                    Eigen::MatrixXd &xf_out) const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    if (!fh.exist("/structure/u_tensor")) return false;

    std::vector<int> kinds;
    std::vector<std::string> elements;
    get_structures_from_h5(fh, "PrimitiveCell", lavec_out, xf_out, kinds, elements);

    const auto row = impl->temperature_rows(fh, {temp_requested}).front();

    const auto dset_u = fh.getDataSet("/structure/u_tensor");
    u_tensor_out.assign(9, 0.0);
    dset_u.select({row, 0, 0}, {1, 3, 3}).read(u_tensor_out.data());

    const auto dset_u0 = fh.getDataSet("/structure/u0");
    const auto dims = dset_u0.getDimensions();
    if (dims.size() != 3 || dims[2] != 3) {
        exit("scph_result_io", "Unexpected shape of /structure/u0 in the state file");
    }
    u0_out.assign(dims[1] * 3, 0.0);
    dset_u0.select({row, 0, 0}, {1, dims[1], 3}).read(u0_out.data());

    if (fh.exist("/structure/spg_label")) {
        const auto labels = H5Easy::load<std::vector<std::string>>(fh, "/structure/spg_label");
        spg_label_out = row < labels.size() ? labels[row] : std::string{};
    } else {
        spg_label_out.clear();
    }
    return true;
}

std::vector<unsigned int> ScphResultIOH5::compare_provenance(const ScphProvenanceH5 &current) const
{
    std::vector<unsigned int> mismatched;

    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    // Every dataset must be there: a file written by an older build can
    // carry some of them, and a missing one should mean "no fingerprint to
    // compare", not a raw HDF5 abort.
    for (const auto *name: {"fcs_nrows", "fcs_sum_abs", "fcs_sum_signed", "fcs_sum_sq"}) {
        if (!fh.exist(std::string("/provenance/") + name)) return mismatched;
    }

    const auto nrows_file = H5Easy::load<std::vector<unsigned long long>>(fh, "/provenance/fcs_nrows");
    const auto abs_file = H5Easy::load<std::vector<double>>(fh, "/provenance/fcs_sum_abs");
    const auto signed_file = H5Easy::load<std::vector<double>>(fh, "/provenance/fcs_sum_signed");
    const auto sq_file = H5Easy::load<std::vector<double>>(fh, "/provenance/fcs_sum_sq");

    if (abs_file.size() != nrows_file.size() || signed_file.size() != nrows_file.size() ||
        sq_file.size() != nrows_file.size())
    {
        warn("scph_result_io", "The /provenance datasets of the state file disagree in length; skipping the check.");
        return mismatched;
    }

    // Summation order can differ between runs, so compare relatively rather
    // than bit-for-bit. `scale` is the magnitude the sum is accumulated
    // from, not the sum itself: sum v is cancellation-prone, and for a
    // model whose quartic terms nearly cancel a relative test against its
    // own near-zero value would fail on reordering noise alone.
    const auto differs = [](const double a, const double b, const double scale) {
        return std::abs(a - b) > 1.0e-10 * std::max(scale, 1.0e-30);
    };

    // Order 0 is skipped on purpose. RELAXED_STRUCTURE requires
    // FC2_TEMPERATURE, so the consumer's FC2 is the renormalized one read
    // back from this very file and never matches the bare FC2 the producer
    // loaded; comparing it would warn on every correct run. The orders that
    // must agree are the anharmonic ones.
    const auto n = std::min(current.fcs_nrows.size(), nrows_file.size());
    for (size_t order = 1; order < n; ++order) {
        const auto scale_abs = std::max(current.fcs_sum_abs[order], abs_file[order]);
        if (current.fcs_nrows[order] != static_cast<size_t>(nrows_file[order]) ||
            differs(current.fcs_sum_abs[order], abs_file[order], scale_abs) ||
            differs(current.fcs_sum_signed[order], signed_file[order], scale_abs) ||
            differs(current.fcs_sum_sq[order], sq_file[order], std::max(current.fcs_sum_sq[order], sq_file[order])))
        {
            mismatched.push_back(static_cast<unsigned int>(order));
        }
    }
    return mismatched;
}

void ScphResultIOH5::write_state(const ScphSettingsH5 &settings, const ScphCellsH5 &cells,
                                 const std::complex<double> *const *const *const *delta_main,
                                 const std::complex<double> *const *const *const *delta_harm_renorm,
                                 const std::vector<double> *v0, const ScphFc2RowsH5 *fc2,
                                 const std::vector<unsigned char> *converged_scph,
                                 const std::vector<unsigned char> *converged_structure,
                                 const ScphStructureH5 *structure, const ScphProvenanceH5 *provenance,
                                 const std::vector<double> *data_temperature) const
{
    using namespace H5Easy;

    const auto part = h5_part_filename(impl->filename);
    if (std::filesystem::exists(part)) {
        warn("scph_result_io", "Removing a stale .part file of a previously interrupted run.");
        std::filesystem::remove(part);
    }

    const auto natmin = static_cast<size_t>(cells.xf_prim.rows());
    const size_t ns = 3 * natmin;
    const auto nt = settings.temperatures.size();
    const unsigned int nk1 = cells.ncell_grid[0], nk2 = cells.ncell_grid[1], nk3 = cells.ncell_grid[2];
    const size_t ncell = static_cast<size_t>(nk1) * nk2 * nk3;

    {
        File fh(part, File::ReadWrite | File::Create | File::Excl);
        write_input_variables_h5(fh, settings.input_variables);

        stamp_h5_schema(fh, h5_schema_scph_state, h5_version_scph_state);
        fh.createAttribute("mode", settings.mode);

        // Settings
        dump(fh,
             "/settings/kmesh_interpolate",
             std::vector<unsigned int>{settings.kmesh_interpolate[0],
                                       settings.kmesh_interpolate[1],
                                       settings.kmesh_interpolate[2]});
        dump(fh,
             "/settings/kmesh_dense",
             std::vector<unsigned int>{settings.kmesh_dense[0], settings.kmesh_dense[1], settings.kmesh_dense[2]});
        dump(fh, "/settings/temperatures", settings.temperatures);
        dumpAttribute(fh, "/settings/temperatures", "unit", std::string("K"));
        dump(fh, "/settings/nonanalytic", settings.nonanalytic);
        // /dymat corrections are taken against the full harmonic matrix of the run
        // (with its nonanalytic term); absent: against the analytic fc2 part only
        dump(fh, "/settings/delta_baseline", 1);
        dump(fh, "/settings/selfenergy_offdiag", settings.selfenergy_offdiag);
        dump(fh, "/settings/relax_str", settings.relax_str);
        if (settings.efield[0] != 0.0 || settings.efield[1] != 0.0 || settings.efield[2] != 0.0) {
            dump(fh, "/settings/efield", std::vector<double>(settings.efield.begin(), settings.efield.end()));
            dumpAttribute(fh, "/settings/efield", "unit", std::string("eV/Angstrom"));
            const auto nat = settings.born_charges.size() / 9;
            fh.createDataSet<double>("/settings/born_charges", HighFive::DataSpace({nat, 3, 3}))
                .write_raw(settings.born_charges.data());
        }
        const auto nonzero = [](const auto &v) {
            return std::any_of(v.begin(), v.end(), [](const double x) { return x != 0.0; });
        };
        if (nonzero(settings.pol_ref)) {
            dump(fh, "/settings/pol_ref", std::vector<double>(settings.pol_ref.begin(), settings.pol_ref.end()));
            dumpAttribute(fh, "/settings/pol_ref", "unit", std::string("C/m^2"));
        }
        if (nonzero(settings.piezo0)) {
            fh.createDataSet<double>("/settings/piezo_clamped_ion", HighFive::DataSpace({3, 3, 3}))
                .write_raw(settings.piezo0.data());
            dumpAttribute(fh, "/settings/piezo_clamped_ion", "unit", std::string("e/bohr^2"));
        }
        if (nonzero(settings.piezo2)) {
            fh.createDataSet<double>("/settings/piezo_second_order", HighFive::DataSpace({3, 3, 3, 3, 3}))
                .write_raw(settings.piezo2.data());
            dumpAttribute(fh, "/settings/piezo_second_order", "unit", std::string("e/bohr^2"));
        }
        if (settings.strain_fc5) {
            dump(fh, "/settings/strain_fc5", settings.strain_fc5);
            dump(fh, "/settings/strain_fc5_diag", settings.strain_fc5_diag);
            dump(fh,
                 "/settings/fc5_fingerprint",
                 std::vector<double>(settings.fc5_fingerprint.begin(), settings.fc5_fingerprint.end()));
        }
        if (nonzero(settings.born_strain)) {
            const auto nat = settings.born_strain.size() / 81;
            fh.createDataSet<double>("/settings/born_charge_strain_derivative", HighFive::DataSpace({nat, 3, 3, 3, 3}))
                .write_raw(settings.born_strain.data());
            dumpAttribute(fh, "/settings/born_charge_strain_derivative", "unit", std::string("e"));
        }

        // Primitive cell (identity mapping) and the virtual supercell
        // (primitive cell tiled by KMESH_INTERPOLATE, cell-major atom order:
        // atom index = icell * natmin + iat with icell = ix*nk2*nk3 + iy*nk3 + iz).
        std::vector<std::vector<int>> mapping_prim(natmin, std::vector<int>(1));
        for (size_t i = 0; i < natmin; ++i) mapping_prim[i][0] = static_cast<int>(i);
        write_cell_group_h5(fh,
                            "PrimitiveCell",
                            cells.lavec_prim,
                            cells.xf_prim,
                            cells.kinds,
                            cells.elements,
                            cells.spin_polarized,
                            cells.magmom,
                            cells.noncollinear,
                            cells.time_reversal_symmetry,
                            1,
                            mapping_prim,
                            units::FcUnitSystem::ry_bohr,
                            cells.masses_amu);

        Eigen::Matrix3d lavec_super = cells.lavec_prim;
        lavec_super.col(0) *= static_cast<double>(nk1);
        lavec_super.col(1) *= static_cast<double>(nk2);
        lavec_super.col(2) *= static_cast<double>(nk3);

        Eigen::MatrixXd xf_super(ncell * natmin, 3);
        std::vector<int> kinds_super(ncell * natmin);
        std::vector<std::vector<double>> magmom_super;
        std::vector<std::vector<int>> mapping_super(natmin, std::vector<int>(ncell));
        size_t icell = 0;
        for (unsigned int ix = 0; ix < nk1; ++ix) {
            for (unsigned int iy = 0; iy < nk2; ++iy) {
                for (unsigned int iz = 0; iz < nk3; ++iz) {
                    for (size_t iat = 0; iat < natmin; ++iat) {
                        const auto idx = icell * natmin + iat;
                        xf_super(idx, 0) = (cells.xf_prim(iat, 0) + ix) / nk1;
                        xf_super(idx, 1) = (cells.xf_prim(iat, 1) + iy) / nk2;
                        xf_super(idx, 2) = (cells.xf_prim(iat, 2) + iz) / nk3;
                        kinds_super[idx] = cells.kinds[iat];
                        mapping_super[iat][icell] = static_cast<int>(idx);
                        if (cells.spin_polarized) magmom_super.push_back(cells.magmom[iat]);
                    }
                    ++icell;
                }
            }
        }
        write_cell_group_h5(fh,
                            "SuperCell",
                            lavec_super,
                            xf_super,
                            kinds_super,
                            cells.elements,
                            cells.spin_polarized,
                            magmom_super,
                            cells.noncollinear,
                            cells.time_reversal_symmetry,
                            ncell,
                            mapping_super,
                            units::FcUnitSystem::ry_bohr);

        // Dynamical-matrix corrections (restart payload)
        Impl::write_dymat_dataset(fh, "/dymat/delta", delta_main, nt, ns, ncell);
        if (delta_harm_renorm) {
            Impl::write_dymat_dataset(fh, "/dymat/delta_harm_renorm", delta_harm_renorm, nt, ns, ncell);
        }

        if (v0) {
            dump(fh, "/V0", *v0);
            dumpAttribute(fh, "/V0", "unit", std::string("Ry"));
        }

        // Relaxed structure per temperature (RELAX_STR != 0 only). The
        // reference cell it deforms is the PrimitiveCell group above.
        if (structure) {
            if (structure->u_tensor.size() != nt * 9 || structure->u0.size() != nt * ns ||
                structure->spg_label.size() != nt)
            {
                exit("scph_result_io", "Inconsistent size of the relaxed-structure payload");
            }
            auto dset_u = h5_create_dataset_compressed<double>(fh, "/structure/u_tensor", {nt, 3, 3}, 1);
            dset_u.write_raw(structure->u_tensor.data());
            dset_u.createAttribute("unit", std::string("dimensionless"));

            auto dset_u0 = h5_create_dataset_compressed<double>(fh, "/structure/u0", {nt, natmin, 3}, 1);
            dset_u0.write_raw(structure->u0.data());
            dset_u0.createAttribute("unit", std::string("bohr"));

            dump(fh, "/structure/spg_label", structure->spg_label);
        }

        // Fingerprint of the IFCs behind that structure.
        if (provenance && !provenance->fcs_nrows.empty()) {
            std::vector<unsigned long long> nrows(provenance->fcs_nrows.begin(), provenance->fcs_nrows.end());
            dump(fh, "/provenance/fcs_nrows", nrows);
            dump(fh, "/provenance/fcs_sum_abs", provenance->fcs_sum_abs);
            dump(fh, "/provenance/fcs_sum_signed", provenance->fcs_sum_signed);
            dump(fh, "/provenance/fcs_sum_sq", provenance->fcs_sum_sq);
            dumpAttribute(fh,
                          "/provenance/fcs_nrows",
                          "definition",
                          std::string("per IFC order, as loaded and before replication: entry count, "
                                      "sum |v|, sum v, sum v^2"));
        }

        // Per-temperature convergence flags (1 = converged). Consumers
        // refuse unconverged temperatures unless ALLOW_UNCONVERGED is set.
        if (converged_scph) {
            dump(fh, "/convergence/scph", *converged_scph);
        }
        if (converged_structure) {
            dump(fh, "/convergence/structure", *converged_structure);
        }
        if (data_temperature) {
            dump(fh, "/convergence/data_temperature", *data_temperature);
            dumpAttribute(fh, "/convergence/data_temperature", "unit", std::string("K"));
            dumpAttribute(fh,
                          "/convergence/data_temperature",
                          "definition",
                          std::string("temperature whose result the row holds: T when converged, the source "
                                      "temperature when copied after a failed optimization, NaN for harmonic"));
        }

        if (fc2) {
            // Base harmonic FC2 (readable by any current anphon as a plain
            // force-constant file) ...
            write_fc_order_group_h5(fh,
                                    0,
                                    fc2->atom_indices,
                                    fc2->atom_indices_super,
                                    fc2->coord_indices,
                                    fc2->shift_vectors,
                                    fc2->base_values,
                                    units::FcUnitSystem::ry_bohr,
                                    1);
            // ... plus the total renormalized FC2 per temperature on the
            // same rows, selected downstream via FC2_TEMPERATURE.
            const auto nrows = static_cast<size_t>(fc2->atom_indices.rows());
            const std::string path_tdep = "/ForceConstants/Order2_temperature_dependent/force_constant_values";
            auto dset = h5_create_dataset_compressed<double>(fh, path_tdep, {nt, nrows}, 1);
            dset.write_raw(fc2->values_per_temperature.data());
            dset.createAttribute("unit", std::string("Ry/bohr^2"));
            dset.createAttribute("index_datasets", std::string("/ForceConstants/Order2"));
            dset.createAttribute("variant", fc2->variant);
        }

        fh.createAttribute("complete", 1);
        h5_flush_and_fsync(fh);
    }

    h5_publish_file(part, impl->filename);
}

void ScphResultIOH5::check_convergence(const std::vector<double> &temps_requested, const bool allow_unconverged) const
{
    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    // Absent datasets mean the file predates the flags (legacy import);
    // nothing can be checked then.
    if (!fh.exist("/convergence")) return;

    const auto rows = impl->temperature_rows(fh, temps_requested);

    const auto collect_bad = [&](const std::string &name, std::vector<double> &bad) {
        if (!fh.exist("/convergence/" + name)) return;
        std::vector<unsigned char> flags;
        fh.getDataSet("/convergence/" + name).read(flags);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] < flags.size() && !flags[rows[i]]) bad.push_back(temps_requested[i]);
        }
    };

    std::vector<double> bad_scph, bad_str;
    collect_bad("scph", bad_scph);
    collect_bad("structure", bad_str);
    if (bad_scph.empty() && bad_str.empty()) return;

    std::cout << "\n The state file " << impl->filename << " contains data whose iterations did NOT converge:\n";
    const auto list_temps = [](const char *label, const std::vector<double> &bad) {
        if (bad.empty()) return;
        std::cout << "  " << label << " :";
        for (const auto t: bad) std::cout << ' ' << t << " K";
        std::cout << '\n';
    };
    list_temps("SCPH iteration       ", bad_scph);
    list_temps("structural relaxation", bad_str);

    std::string notes;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto note = scph_row_data_note(fh, rows[i]);
        if (!note.empty()) notes += " Note: " + note + ".\n";
    }
    std::cout << notes;

    if (allow_unconverged) {
        warn("scph_result_io",
             notes.empty() ? "Using unconverged renormalized data because ALLOW_UNCONVERGED = 1."
                           : ("Using the flagged data because ALLOW_UNCONVERGED = 1:\n" + notes +
                              " These rows are not results at their own temperature.")
                                 .c_str());
    } else {
        exit("scph_result_io",
             "Refusing to use unconverged renormalized IFCs/structure.\n"
             " Rerun with tighter/longer iterations (MAXITER, MAX_STR_ITER, ...) to converge them,\n"
             " or set ALLOW_UNCONVERGED = 1 in &general to use the data anyway.");
    }
}

void ScphResultIOH5::load_convergence(const std::vector<double> &temps_requested, std::vector<unsigned char> &scph_out,
                                      std::vector<unsigned char> &structure_out) const
{
    scph_out.assign(temps_requested.size(), 1);
    structure_out.assign(temps_requested.size(), 1);

    const HighFive::File fh(impl->filename, HighFive::File::ReadOnly);
    if (!fh.exist("/convergence")) return;

    const auto rows = impl->temperature_rows(fh, temps_requested);
    const auto load = [&](const std::string &name, std::vector<unsigned char> &out) {
        if (!fh.exist("/convergence/" + name)) return;
        std::vector<unsigned char> flags;
        fh.getDataSet("/convergence/" + name).read(flags);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] < flags.size()) out[i] = flags[rows[i]];
        }
    };
    load("scph", scph_out);
    load("structure", structure_out);
}

std::string PHON_NS::scph_row_data_note(const HighFive::File &fh, const size_t row)
{
    const std::string path = "/convergence/data_temperature";
    if (!fh.exist(path)) return {};
    const auto source = H5Easy::load<std::vector<double>>(fh, path);
    const auto temps = H5Easy::load<std::vector<double>>(fh, "/settings/temperatures");
    if (row >= source.size() || row >= temps.size()) return {};
    const auto temp = temps[row];

    std::ostringstream ss;
    if (std::isnan(source[row])) {
        ss << "the stored data for " << temp << " K are the harmonic ones (the structural optimization at " << temp
           << " K failed before any temperature converged)";
    } else if (std::fabs(source[row] - temp) > eps6) {
        ss << "the stored data for " << temp << " K are a copy of the converged result at " << source[row]
           << " K (the structural optimization at " << temp << " K failed)";
    }
    return ss.str();
}

const std::string &ScphResultIOH5::get_filename() const
{
    return impl->filename;
}
