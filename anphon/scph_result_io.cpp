/*
 scph_result_io.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph_result_io.h"
#include <cmath>
#include <filesystem>
#include <mpi.h>
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
    if (load<int>(fh, "/settings/nonanalytic") != settings.nonanalytic) {
        warn("scph_result_io", "The NONANALYTIC tag is not consistent");
    }
    if (load<int>(fh, "/settings/selfenergy_offdiag") != settings.selfenergy_offdiag) {
        exit("scph_result_io", "The SELF_OFFDIAG tag is not consistent");
    }
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

void ScphResultIOH5::write_state(const ScphSettingsH5 &settings, const ScphCellsH5 &cells,
                                 const std::complex<double> *const *const *const *delta_main,
                                 const std::complex<double> *const *const *const *delta_harm_renorm,
                                 const std::vector<double> *v0, const ScphFc2RowsH5 *fc2,
                                 const std::vector<unsigned char> *converged_scph,
                                 const std::vector<unsigned char> *converged_structure,
                                 const ScphStructureH5 *structure, const ScphProvenanceH5 *provenance) const
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
        dump(fh, "/settings/selfenergy_offdiag", settings.selfenergy_offdiag);
        dump(fh, "/settings/relax_str", settings.relax_str);

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

    if (allow_unconverged) {
        warn("scph_result_io", "Using unconverged renormalized data because ALLOW_UNCONVERGED = 1.");
    } else {
        exit("scph_result_io",
             "Refusing to use unconverged renormalized IFCs/structure.\n"
             " Rerun with tighter/longer iterations (MAXITER, MAX_STR_ITER, ...) to converge them,\n"
             " or set ALLOW_UNCONVERGED = 1 in &general to use the data anyway.");
    }
}

const std::string &ScphResultIOH5::get_filename() const
{
    return impl->filename;
}
