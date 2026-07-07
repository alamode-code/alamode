/*
 kappa_result_io.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "kappa_result_io.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mpi.h>
#include <utility>
#include "constants.h"
#include "error.h"
#include "hdf5_parser.h"

using namespace PHON_NS;

namespace
{
const std::string str_scattering = "/scattering/";

// format_version 2 = temperature-resolved layout (basis from FC2_TEMPERATURE):
// frequencies/velocities/kappa gain a leading temperature dimension and the
// completion flags become per (mode, temperature).
constexpr int kappa_version_tdep = 2;

auto channel_path(const std::string &tag) -> std::string
{
    return str_scattering + tag;
}
} // namespace

struct KappaResultIOH5::Impl
{
    std::string filename;
    std::unique_ptr<HighFive::File> file;
    KappaFileMetaH5 fmeta;
    size_t ntemp = 0; // temperatures of THIS run

    // /iterativebte group description (set only under SOLVER = IBTE); when
    // present, every rebuild recreates or carries the group.
    IbteMetaH5 imeta;
    bool has_imeta = false;

    // Temperature-resolved (accumulation) state
    bool tdep = false;
    std::vector<double> file_temps; // full grid of the file (== run temps in v1 mode)
    std::vector<double> file_fc2temps;
    std::vector<size_t> run_cols; // file column of each run temperature

    auto nt_file() const -> size_t
    {
        return tdep ? file_temps.size() : ntemp;
    }

    auto compute_run_cols() -> void
    {
        run_cols.clear();
        for (const auto t: fmeta.temperatures) {
            auto found = false;
            for (size_t i = 0; i < file_temps.size(); ++i) {
                if (std::abs(file_temps[i] - t) < eps6) {
                    run_cols.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                exit("kappa_result_io", "Internal error: run temperature missing from the file grid");
            }
        }
    }

    auto write_metadata(HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        dump(fh, "/metadata/temperatures", tdep ? file_temps : fmeta.temperatures);
        dumpAttribute(fh, "/metadata/temperatures", "unit", std::string("K"));
        dump(fh, "/metadata/classical", fmeta.classical);
        dump(fh, "/metadata/ismear", fmeta.ismear);
        dump(fh, "/metadata/smearing_width", fmeta.smearing_width);
        dumpAttribute(fh, "/metadata/smearing_width", "unit", std::string("cm^-1"));
        dump(fh, "/metadata/fcs_file", fmeta.fcs_file);
        dump(fh, "/metadata/isotope", fmeta.isotope);
        if (fmeta.isotope > 0 && !fmeta.isotope_factors.empty()) {
            dump(fh, "/metadata/isotope_factors", fmeta.isotope_factors);
            dumpAttribute(fh,
                          "/metadata/isotope_factors",
                          "description",
                          std::string("Tamura g2 mass-variance factor per element"));
        }
        if (tdep) {
            // The basis temperature (FC2_TEMPERATURE) each row was computed with.
            dump(fh, "/metadata/fc2_temperatures", file_fc2temps);
            dumpAttribute(fh, "/metadata/fc2_temperatures", "unit", std::string("K"));
            dump(fh, "/metadata/fc2_source", fmeta.fc2_source);
        }

        const std::string cell = "/metadata/PrimitiveCell";
        dump(fh, cell + "/lattice_vector", Eigen::Matrix3d(fmeta.lattice_vector.transpose()));
        dumpAttribute(fh, cell + "/lattice_vector", "unit", std::string("bohr"));
        dump(fh, cell + "/number_of_atoms", static_cast<size_t>(fmeta.x_fractional.rows()));
        dump(fh, cell + "/number_of_elements", fmeta.elements.size());
        dump(fh, cell + "/fractional_coordinate", fmeta.x_fractional);
        dump(fh, cell + "/atomic_kinds", fmeta.atomic_kinds);
        dump(fh, cell + "/elements", fmeta.elements);
        dump(fh, cell + "/volume", fmeta.volume);
        dumpAttribute(fh, cell + "/volume", "unit", std::string("bohr^3"));
    }

    // Mirror of the legacy check_consistency_restart: hard exits for
    // system/temperature mismatches, warnings for the rest. In the
    // temperature-resolved mode the grids merge instead of having to match.
    auto validate_metadata(const HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        const auto &efile = fh;
        const auto natoms = load<size_t>(efile, "/metadata/PrimitiveCell/number_of_atoms");
        const auto nelems = load<size_t>(efile, "/metadata/PrimitiveCell/number_of_elements");
        if (natoms != static_cast<size_t>(fmeta.x_fractional.rows()) || nelems != fmeta.elements.size()) {
            exit("kappa_result_io", "SYSTEM information in the kappa.h5 file is not consistent");
        }

        if (!tdep) {
            const auto temps = load<std::vector<double>>(efile, "/metadata/temperatures");
            auto temps_ok = temps.size() == fmeta.temperatures.size();
            if (temps_ok) {
                for (size_t i = 0; i < temps.size(); ++i) {
                    if (std::abs(temps[i] - fmeta.temperatures[i]) >= eps6) {
                        temps_ok = false;
                        break;
                    }
                }
            }
            if (!temps_ok) {
                exit("kappa_result_io", "Temperature information in the kappa.h5 file is not consistent");
            }
        }

        if (load<int>(efile, "/metadata/classical") != fmeta.classical) {
            warn("kappa_result_io", "CLASSICAL val is not consistent");
        }
        if (load<std::string>(efile, "/metadata/fcs_file") != fmeta.fcs_file) {
            warn("kappa_result_io", "FCSFILE is not consistent");
        }
        const auto ismear_file = load<int>(efile, "/metadata/ismear");
        if (ismear_file != fmeta.ismear) {
            warn("kappa_result_io", "Smearing method is not consistent");
        }
        if (fh.exist("/metadata/isotope_factors") && fmeta.isotope > 0) {
            const auto factors = load<std::vector<double>>(efile, "/metadata/isotope_factors");
            auto factors_ok = factors.size() == fmeta.isotope_factors.size();
            if (factors_ok) {
                for (size_t i = 0; i < factors.size(); ++i) {
                    if (std::abs(factors[i] - fmeta.isotope_factors[i]) >= eps8) factors_ok = false;
                }
            }
            if (!factors_ok) {
                warn("kappa_result_io", "ISOFACT is not consistent");
            }
        }
        if (ismear_file != -1 &&
            std::abs(load<double>(efile, "/metadata/smearing_width") - fmeta.smearing_width) >= eps4)
        {
            warn("kappa_result_io", "Smearing width is not consistent");
        }
    }

    // The layout mode of an existing file must match the run: mixing a
    // temperature-resolved (FC2_TEMPERATURE) run with a v1 file, or vice
    // versa, would silently reinterpret the datasets.
    auto validate_layout_mode(const HighFive::File &fh) const -> void
    {
        int tdep_file = 0;
        if (fh.hasAttribute("temperature_resolved")) {
            fh.getAttribute("temperature_resolved").read(tdep_file);
        }
        if ((tdep_file != 0) != tdep) {
            exit("kappa_result_io",
                 tdep ? "The existing kappa.h5 file was written without FC2_TEMPERATURE.\n"
                        " Use a different PREFIX or remove the file."
                      : "The existing kappa.h5 file is temperature-resolved (FC2_TEMPERATURE).\n"
                        " Use a different PREFIX or remove the file.");
        }
    }

    // Create the group, attributes, static datasets, and preallocated
    // gamma/flag (and, in tdep mode, frequency/velocity) datasets of a
    // channel. In tdep mode the frequency/velocity content is written later
    // (write_basis_slices); only the shapes are taken from cmeta here.
    auto create_channel(HighFive::File &fh, const KappaChannelMetaH5 &cmeta) const -> void
    {
        using namespace H5Easy;
        const auto path = channel_path(cmeta.tag);
        auto group = fh.createGroup(path);

        const std::vector<unsigned int> kmesh{cmeta.nk_i[0], cmeta.nk_i[1], cmeta.nk_i[2]};
        group.createAttribute("kmesh", kmesh);
        group.createAttribute("nk_irred", cmeta.nk_irred);
        group.createAttribute("nbranches", cmeta.ns);

        dump(fh, path + "/xk_irred", cmeta.xk_irred);
        dump(fh, path + "/weights", cmeta.weights);

        // CSR encoding of the irreducible stars
        std::vector<int> offsets(cmeta.nk_irred + 1, 0);
        std::vector<int> knum_flat;
        for (size_t i = 0; i < cmeta.equiv_knum.size(); ++i) {
            offsets[i + 1] = offsets[i] + static_cast<int>(cmeta.equiv_knum[i].size());
            knum_flat.insert(knum_flat.end(), cmeta.equiv_knum[i].begin(), cmeta.equiv_knum[i].end());
        }
        dump(fh, path + "/equiv_offsets", offsets);
        dump(fh, path + "/equiv_knum", knum_flat);

        const size_t nequiv_total = knum_flat.size();
        const size_t nrows = static_cast<size_t>(cmeta.nk_irred) * cmeta.ns;
        const auto nt = nt_file();

        if (tdep) {
            h5_create_dataset_prealloc<double>(fh, path + "/frequencies", {nt, cmeta.nk_irred, cmeta.ns})
                .createAttribute("unit", std::string("cm^-1"));
            h5_create_dataset_prealloc<double>(fh, path + "/velocities", {nt, nequiv_total, cmeta.ns, 3})
                .createAttribute("unit", std::string("m/s"));
            auto dset_gamma = h5_create_dataset_prealloc<double>(fh, path + "/gamma", {nrows, nt});
            dset_gamma.createAttribute("unit", std::string("cm^-1"));
            h5_create_dataset_prealloc<unsigned char>(fh, path + "/gamma_computed", {nrows, nt});
        } else {
            dump(fh, path + "/frequencies", cmeta.frequencies);
            dumpAttribute(fh, path + "/frequencies", "unit", std::string("cm^-1"));
            auto dset_vel =
                fh.createDataSet<double>(path + "/velocities", HighFive::DataSpace({nequiv_total, cmeta.ns, 3}));
            dset_vel.write_raw(cmeta.velocities.data());
            dset_vel.createAttribute("unit", std::string("m/s"));

            auto dset_gamma = h5_create_dataset_prealloc<double>(fh, path + "/gamma", {nrows, nt});
            dset_gamma.createAttribute("unit", std::string("cm^-1"));
            h5_create_dataset_prealloc<unsigned char>(fh, path + "/gamma_computed", {nrows});
        }

        // The isotope linewidths live on the 3ph mesh; the group is created
        // alongside the 3ph channel and filled at the end of the run.
        if (cmeta.tag == "3ph" && fmeta.isotope > 0 && !fh.exist("/scattering/isotope")) {
            auto dset_iso =
                tdep ? h5_create_dataset_prealloc<double>(fh,
                                                          "/scattering/isotope/gamma",
                                                          {nt, cmeta.nk_irred, cmeta.ns})
                     : h5_create_dataset_prealloc<double>(fh, "/scattering/isotope/gamma", {cmeta.nk_irred, cmeta.ns});
            dset_iso.createAttribute("unit", std::string("cm^-1"));
        }
    }

    // Fill the frequency/velocity slices of this run's temperature columns
    // (tdep mode only). Pure raw-data writes into preallocated datasets.
    auto write_basis_slices(const KappaChannelMetaH5 &cmeta) const -> void
    {
        if (!tdep) return;
        const auto path = channel_path(cmeta.tag);
        auto dset_freq = file->getDataSet(path + "/frequencies");
        auto dset_vel = file->getDataSet(path + "/velocities");
        const size_t nequiv_total = cmeta.velocities.size() / (static_cast<size_t>(cmeta.ns) * 3);
        for (const auto col: run_cols) {
            dset_freq.select({col, 0, 0}, {1, cmeta.nk_irred, cmeta.ns}).write_raw(cmeta.frequencies.data());
            dset_vel.select({col, 0, 0, 0}, {1, nequiv_total, cmeta.ns, 3}).write_raw(cmeta.velocities.data());
        }
        h5_flush_and_fsync(*file);
    }

    auto channel_matches(const HighFive::File &fh, const KappaChannelMetaH5 &cmeta) const -> bool
    {
        const auto group = fh.getGroup(channel_path(cmeta.tag));
        std::vector<unsigned int> kmesh;
        unsigned int nk_irred = 0, ns_file = 0;
        group.getAttribute("kmesh").read(kmesh);
        group.getAttribute("nk_irred").read(nk_irred);
        group.getAttribute("nbranches").read(ns_file);
        return kmesh.size() == 3 && kmesh[0] == cmeta.nk_i[0] && kmesh[1] == cmeta.nk_i[1] &&
               kmesh[2] == cmeta.nk_i[2] && nk_irred == cmeta.nk_irred && ns_file == cmeta.ns;
    }

    // Human-readable list of the scattering processes entering the total
    // linewidth behind the kappa tensors, e.g. "3ph+4ph+isotope+boundary".
    auto scattering_process_string(const bool with_4ph) const -> std::string
    {
        std::string procs = "3ph";
        if (with_4ph) procs += "+4ph";
        if (fmeta.isotope > 0) procs += "+isotope";
        if (fmeta.boundary_length > 0.0) procs += "+boundary";
        return procs;
    }

    auto create_kappa_group(HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        const auto nt = nt_file();
        const auto procs_full = scattering_process_string(fmeta.with_kappa_3ph_only);
        const auto label = [&procs_full](HighFive::DataSet dset) {
            dset.createAttribute("unit", std::string("W/mK"));
            dset.createAttribute("scattering_processes", procs_full);
        };

        label(h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_peierls", {nt, 3, 3}));
        if (fmeta.with_kappa_3ph_only) {
            auto dset = h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_3ph_only", {nt, 3, 3});
            dset.createAttribute("unit", std::string("W/mK"));
            dset.createAttribute("scattering_processes", scattering_process_string(false));
        }
        if (fmeta.with_kappa_coherent) {
            label(h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_coherent", {nt, 3, 3}));
            // Wigner total = Peierls (intraband) + coherent (interband).
            // Written only when the coherent term is actually computed, so
            // "total" never silently means "Peierls only".
            label(h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_total", {nt, 3, 3}));
        }
        if (fmeta.with_kappa_spec) {
            dump(fh, "/kappa/energy_axis", fmeta.energy_axis);
            dumpAttribute(fh, "/kappa/energy_axis", "unit", std::string("cm^-1"));
            auto dset = h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_spec", {fmeta.energy_axis.size(), nt, 3});
            dset.createAttribute("unit", std::string("W/mK/cm^-1"));
            dset.createAttribute("scattering_processes", procs_full);
        }
        // Validity marker as a raw-data dataset (not an attribute) so setting
        // it never touches file metadata. Per temperature in tdep mode.
        h5_create_dataset_prealloc<unsigned char>(fh, "/kappa/valid", {tdep ? nt : 1});

        // Machine-readable summary of the same information on the group.
        auto gk = fh.getGroup("/kappa");
        gk.createAttribute("includes_isotope_scattering", fmeta.isotope > 0 ? 1 : 0);
        gk.createAttribute("includes_boundary_scattering", fmeta.boundary_length > 0.0 ? 1 : 0);
        gk.createAttribute("includes_4ph_scattering", fmeta.with_kappa_3ph_only ? 1 : 0);
        gk.createAttribute("boundary_length", fmeta.boundary_length); // m; 0 = off
    }

    // The /kappa layout is fixed at setup; a restart run with different
    // output options needs a rebuild.
    auto kappa_group_matches(const HighFive::File &fh) const -> bool
    {
        const auto need = [&](const std::string &name, const bool required) {
            return fh.exist("/kappa/" + name) == required;
        };
        if (!need("kappa_peierls", true)) return false;
        if (!need("kappa_total", fmeta.with_kappa_coherent)) return false;
        if (!need("kappa_3ph_only", fmeta.with_kappa_3ph_only)) return false;
        if (!need("kappa_coherent", fmeta.with_kappa_coherent)) return false;
        if (!need("kappa_spec", fmeta.with_kappa_spec)) return false;
        const auto nt = nt_file();
        if (fmeta.with_kappa_spec) {
            const auto dims = fh.getDataSet("/kappa/kappa_spec").getDimensions();
            if (dims.size() != 3 || dims[0] != fmeta.energy_axis.size() || dims[1] != nt) return false;
        }
        {
            const auto dims = fh.getDataSet("/kappa/kappa_peierls").getDimensions();
            if (dims.size() != 3 || dims[0] != nt) return false;
        }
        const auto dims_valid = fh.getDataSet("/kappa/valid").getDimensions();
        if (dims_valid[0] != (tdep ? nt : 1)) return false;

        // The scattering-process labels must describe THIS run, since the
        // kappa tensors are recomputed and overwritten; a changed process
        // set (e.g. ISOTOPE toggled) forces a rebuild of the group.
        const auto gk = fh.getGroup("/kappa");
        if (!gk.hasAttribute("includes_isotope_scattering")) return false;
        int iso = 0, boundary = 0, fourph = 0;
        double blen = 0.0;
        gk.getAttribute("includes_isotope_scattering").read(iso);
        gk.getAttribute("includes_boundary_scattering").read(boundary);
        gk.getAttribute("includes_4ph_scattering").read(fourph);
        gk.getAttribute("boundary_length").read(blen);
        if (iso != (fmeta.isotope > 0 ? 1 : 0)) return false;
        if (boundary != (fmeta.boundary_length > 0.0 ? 1 : 0)) return false;
        if (fourph != (fmeta.with_kappa_3ph_only ? 1 : 0)) return false;
        if (std::abs(blen - fmeta.boundary_length) >= eps10) return false;
        return true;
    }

    // The isotope group must exist exactly when isotope scattering is on;
    // toggling ISOTOPE across restarts triggers one rebuild.
    auto isotope_group_matches(const HighFive::File &fh) const -> bool
    {
        return fh.exist("/scattering/isotope") == (fmeta.isotope > 0);
    }

    auto create_ibte_group(HighFive::File &fh) const -> void
    {
        const auto nt = nt_file();
        auto group = fh.createGroup("/iterativebte");
        const std::vector<unsigned int> kmesh{imeta.nk_i[0], imeta.nk_i[1], imeta.nk_i[2]};
        group.createAttribute("kmesh", kmesh);
        group.createAttribute("nk_irred", imeta.nk_irred);
        group.createAttribute("nbranches", imeta.ns);

        h5_create_dataset_prealloc<double>(fh, "/iterativebte/Q", {nt, imeta.nk_irred, imeta.ns})
            .createAttribute("description",
                             std::string("diagonal (out-scattering) part of the collision operator, internal units"));
        h5_create_dataset_prealloc<double>(fh, "/iterativebte/dF", {nt, imeta.nk_irred, imeta.ns, 3})
            .createAttribute("unit", std::string("bohr/K"));
        h5_create_dataset_prealloc<double>(fh, "/iterativebte/kappa", {nt, 3, 3})
            .createAttribute("unit", std::string("W/mK"));
        h5_create_dataset_prealloc<unsigned char>(fh, "/iterativebte/converged", {nt});
        h5_create_dataset_prealloc<unsigned char>(fh, "/iterativebte/computed", {nt});
    }

    auto ibte_group_matches(const HighFive::File &fh) const -> bool
    {
        const auto group = fh.getGroup("/iterativebte");
        std::vector<unsigned int> kmesh;
        unsigned int nk_irred = 0, ns_file = 0;
        group.getAttribute("kmesh").read(kmesh);
        group.getAttribute("nk_irred").read(nk_irred);
        group.getAttribute("nbranches").read(ns_file);
        if (!(kmesh.size() == 3 && kmesh[0] == imeta.nk_i[0] && kmesh[1] == imeta.nk_i[1] &&
              kmesh[2] == imeta.nk_i[2] && nk_irred == imeta.nk_irred && ns_file == imeta.ns))
        {
            return false;
        }
        const auto dims = fh.getDataSet("/iterativebte/computed").getDimensions();
        return dims.size() == 1 && dims[0] == nt_file();
    }

    auto stamp(HighFive::File &fh) const -> void
    {
        stamp_h5_schema(fh, h5_schema_kappa_result, tdep ? kappa_version_tdep : h5_version_kappa_result);
        if (fh.hasAttribute("temperature_resolved")) fh.deleteAttribute("temperature_resolved");
        fh.createAttribute("temperature_resolved", tdep ? 1 : 0);
    }

    // ---- v1 rebuild: carry channels verbatim (H5Ocopy) ----
    auto rebuild(const std::vector<KappaChannelMetaH5> &to_create, const std::vector<std::string> &tags_drop,
                 const bool drop_ibte = false) -> void
    {
        file.reset();

        const auto part = h5_part_filename(filename);
        if (std::filesystem::exists(part)) {
            warn("kappa_result_io", "Removing a stale .part file of a previously interrupted run.");
            std::filesystem::remove(part);
        }

        {
            HighFive::File newfile(part, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Excl);

            if (std::filesystem::exists(filename)) {
                const HighFive::File oldfile(filename, HighFive::File::ReadOnly);
                if (oldfile.exist("/scattering")) {
                    newfile.createGroup("/scattering");
                    for (const auto &tag: oldfile.getGroup("/scattering").listObjectNames()) {
                        if (std::find(tags_drop.begin(), tags_drop.end(), tag) != tags_drop.end()) continue;
                        const auto path = channel_path(tag);
                        if (H5Ocopy(oldfile.getId(),
                                    path.c_str(),
                                    newfile.getId(),
                                    path.c_str(),
                                    H5P_DEFAULT,
                                    H5P_DEFAULT) < 0)
                        {
                            exit("kappa_result_io", "Failed to carry over a scattering channel", tag.c_str());
                        }
                    }
                }
                // The /iterativebte results survive structural changes (e.g.
                // the 4ph channel appearing) exactly like the gamma channels.
                if (!drop_ibte && oldfile.exist("/iterativebte")) {
                    if (H5Ocopy(oldfile.getId(),
                                "/iterativebte",
                                newfile.getId(),
                                "/iterativebte",
                                H5P_DEFAULT,
                                H5P_DEFAULT) < 0)
                    {
                        exit("kappa_result_io", "Failed to carry over the /iterativebte group");
                    }
                }
            }

            write_metadata(newfile);
            for (const auto &cmeta: to_create) {
                if (!newfile.exist(channel_path(cmeta.tag))) {
                    create_channel(newfile, cmeta);
                }
            }
            ensure_isotope_dataset(newfile);
            create_kappa_group(newfile);
            if (has_imeta && !newfile.exist("/iterativebte")) {
                create_ibte_group(newfile);
            }
            stamp(newfile);
            h5_flush_and_fsync(newfile);
        }

        h5_publish_file(part, filename);
        file = std::make_unique<HighFive::File>(filename, HighFive::File::ReadWrite);
    }

    // Create /scattering/isotope/gamma when isotope scattering is on but the
    // group was neither carried over nor created alongside a fresh 3ph
    // channel (dims come from the 3ph group present in the new file).
    auto ensure_isotope_dataset(HighFive::File &fh) const -> void
    {
        if (fmeta.isotope == 0 || fh.exist("/scattering/isotope") || !fh.exist("/scattering/3ph")) return;
        unsigned int nk_irred = 0, ns = 0;
        const auto g = fh.getGroup("/scattering/3ph");
        g.getAttribute("nk_irred").read(nk_irred);
        g.getAttribute("nbranches").read(ns);
        auto dset_iso =
            tdep ? h5_create_dataset_prealloc<double>(fh, "/scattering/isotope/gamma", {nt_file(), nk_irred, ns})
                 : h5_create_dataset_prealloc<double>(fh, "/scattering/isotope/gamma", {nk_irred, ns});
        dset_iso.createAttribute("unit", std::string("cm^-1"));
    }

    // ---- tdep rebuild: recreate every channel on the merged temperature
    //      grid and remap the old columns/slices into their new positions ----
    auto rebuild_tdep(const std::vector<KappaChannelMetaH5> &to_create, const std::vector<std::string> &tags_drop,
                      const std::vector<double> &old_temps, const std::vector<unsigned char> &old_valid) -> void
    {
        file.reset();

        // old column -> new column
        std::vector<size_t> col_map(old_temps.size());
        for (size_t j = 0; j < old_temps.size(); ++j) {
            size_t pos = file_temps.size();
            for (size_t i = 0; i < file_temps.size(); ++i) {
                if (std::abs(file_temps[i] - old_temps[j]) < eps6) {
                    pos = i;
                    break;
                }
            }
            col_map[j] = pos;
        }
        const auto nt_old = old_temps.size();
        const auto nt_new = file_temps.size();

        const auto part = h5_part_filename(filename);
        if (std::filesystem::exists(part)) {
            warn("kappa_result_io", "Removing a stale .part file of a previously interrupted run.");
            std::filesystem::remove(part);
        }

        {
            HighFive::File newfile(part, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Excl);

            if (std::filesystem::exists(filename)) {
                const HighFive::File oldfile(filename, HighFive::File::ReadOnly);
                if (oldfile.exist("/scattering")) {
                    for (const auto &tag: oldfile.getGroup("/scattering").listObjectNames()) {
                        if (std::find(tags_drop.begin(), tags_drop.end(), tag) != tags_drop.end()) continue;
                        if (tag == "isotope") continue; // handled below, after the channels exist

                        // Reconstruct the channel skeleton from the old file.
                        KappaChannelMetaH5 cmeta;
                        cmeta.tag = tag;
                        const auto opath = channel_path(tag);
                        const auto ogroup = oldfile.getGroup(opath);
                        std::vector<unsigned int> kmesh;
                        ogroup.getAttribute("kmesh").read(kmesh);
                        for (auto i = 0; i < 3; ++i) cmeta.nk_i[i] = kmesh[i];
                        ogroup.getAttribute("nk_irred").read(cmeta.nk_irred);
                        ogroup.getAttribute("nbranches").read(cmeta.ns);
                        cmeta.xk_irred = H5Easy::load<Eigen::MatrixXd>(oldfile, opath + "/xk_irred");
                        cmeta.weights = H5Easy::load<std::vector<double>>(oldfile, opath + "/weights");
                        const auto offsets = H5Easy::load<std::vector<int>>(oldfile, opath + "/equiv_offsets");
                        const auto knum = H5Easy::load<std::vector<int>>(oldfile, opath + "/equiv_knum");
                        cmeta.equiv_knum.resize(cmeta.nk_irred);
                        for (unsigned int i = 0; i < cmeta.nk_irred; ++i) {
                            cmeta.equiv_knum[i].assign(knum.begin() + offsets[i], knum.begin() + offsets[i + 1]);
                        }
                        create_channel(newfile, cmeta);

                        // Remap the temperature-resolved payload.
                        const size_t nrows = static_cast<size_t>(cmeta.nk_irred) * cmeta.ns;
                        const size_t nequiv = knum.size();
                        const size_t nfreq = static_cast<size_t>(cmeta.nk_irred) * cmeta.ns;
                        const size_t nvel = nequiv * cmeta.ns * 3;

                        std::vector<double> gamma_old(nrows * nt_old), buf;
                        std::vector<unsigned char> flags_old(nrows * nt_old);
                        oldfile.getDataSet(opath + "/gamma").read(gamma_old.data());
                        oldfile.getDataSet(opath + "/gamma_computed").read(flags_old.data());

                        std::vector<double> gamma_new(nrows * nt_new, 0.0);
                        std::vector<unsigned char> flags_new(nrows * nt_new, 0);
                        for (size_t r = 0; r < nrows; ++r) {
                            for (size_t j = 0; j < nt_old; ++j) {
                                gamma_new[r * nt_new + col_map[j]] = gamma_old[r * nt_old + j];
                                flags_new[r * nt_new + col_map[j]] = flags_old[r * nt_old + j];
                            }
                        }
                        newfile.getDataSet(opath + "/gamma").write_raw(gamma_new.data());
                        newfile.getDataSet(opath + "/gamma_computed").write_raw(flags_new.data());

                        buf.resize(nfreq);
                        auto dset_freq_old = oldfile.getDataSet(opath + "/frequencies");
                        auto dset_freq_new = newfile.getDataSet(opath + "/frequencies");
                        auto dset_vel_old = oldfile.getDataSet(opath + "/velocities");
                        auto dset_vel_new = newfile.getDataSet(opath + "/velocities");
                        std::vector<double> vbuf(nvel);
                        for (size_t j = 0; j < nt_old; ++j) {
                            dset_freq_old.select({j, 0, 0}, {1, cmeta.nk_irred, cmeta.ns}).read(buf.data());
                            dset_freq_new.select({col_map[j], 0, 0}, {1, cmeta.nk_irred, cmeta.ns})
                                .write_raw(buf.data());
                            dset_vel_old.select({j, 0, 0, 0}, {1, nequiv, cmeta.ns, 3}).read(vbuf.data());
                            dset_vel_new.select({col_map[j], 0, 0, 0}, {1, nequiv, cmeta.ns, 3}).write_raw(vbuf.data());
                        }
                    }
                }

                // Remap the final-kappa rows and their validity flags.
                create_kappa_group(newfile);
                for (const std::string name: {"kappa_peierls", "kappa_total", "kappa_3ph_only", "kappa_coherent"}) {
                    if (!oldfile.exist("/kappa/" + name) || !newfile.exist("/kappa/" + name)) continue;
                    std::vector<double> row(9);
                    for (size_t j = 0; j < nt_old; ++j) {
                        oldfile.getDataSet("/kappa/" + name).select({j, 0, 0}, {1, 3, 3}).read(row.data());
                        newfile.getDataSet("/kappa/" + name)
                            .select({col_map[j], 0, 0}, {1, 3, 3})
                            .write_raw(row.data());
                    }
                }
                if (oldfile.exist("/kappa/kappa_spec") && newfile.exist("/kappa/kappa_spec")) {
                    const auto ne = fmeta.energy_axis.size();
                    std::vector<double> col(ne * 3);
                    for (size_t j = 0; j < nt_old; ++j) {
                        oldfile.getDataSet("/kappa/kappa_spec").select({0, j, 0}, {ne, 1, 3}).read(col.data());
                        newfile.getDataSet("/kappa/kappa_spec")
                            .select({0, col_map[j], 0}, {ne, 1, 3})
                            .write_raw(col.data());
                    }
                }
                std::vector<unsigned char> valid_new(nt_new, 0);
                for (size_t j = 0; j < nt_old && j < old_valid.size(); ++j) {
                    valid_new[col_map[j]] = old_valid[j];
                }
                newfile.getDataSet("/kappa/valid").write_raw(valid_new.data());
            } else {
                create_kappa_group(newfile);
            }

            write_metadata(newfile);
            for (const auto &cmeta: to_create) {
                if (!newfile.exist(channel_path(cmeta.tag))) {
                    create_channel(newfile, cmeta);
                }
            }
            ensure_isotope_dataset(newfile);

            // Remap the isotope-linewidth slices into their new columns.
            if (fmeta.isotope > 0 && std::filesystem::exists(filename) &&
                std::find(tags_drop.begin(), tags_drop.end(), std::string("isotope")) == tags_drop.end() &&
                newfile.exist("/scattering/isotope/gamma"))
            {
                const HighFive::File oldfile(filename, HighFive::File::ReadOnly);
                if (oldfile.exist("/scattering/isotope/gamma")) {
                    const auto dset_old = oldfile.getDataSet("/scattering/isotope/gamma");
                    const auto d = dset_old.getDimensions();
                    const auto dnew = newfile.getDataSet("/scattering/isotope/gamma").getDimensions();
                    if (d.size() == 3 && dnew.size() == 3 && d[1] == dnew[1] && d[2] == dnew[2]) {
                        auto dset_new = newfile.getDataSet("/scattering/isotope/gamma");
                        std::vector<double> slab(d[1] * d[2]);
                        for (size_t j = 0; j < nt_old && j < d[0]; ++j) {
                            dset_old.select({j, 0, 0}, {1, d[1], d[2]}).read(slab.data());
                            dset_new.select({col_map[j], 0, 0}, {1, d[1], d[2]}).write_raw(slab.data());
                        }
                    }
                }
            }

            stamp(newfile);
            h5_flush_and_fsync(newfile);
        }

        h5_publish_file(part, filename);
        file = std::make_unique<HighFive::File>(filename, HighFive::File::ReadWrite);
    }

    auto reset_kappa_valid() -> void
    {
        if (tdep) {
            // Only this run's rows lose validity; other temperatures keep theirs.
            const unsigned char zero = 0;
            auto dset = file->getDataSet("/kappa/valid");
            for (const auto col: run_cols) {
                dset.select({col}, {1}).write_raw(&zero);
            }
        } else {
            const unsigned char zero = 0;
            file->getDataSet("/kappa/valid").write_raw(&zero);
        }
        h5_flush_and_fsync(*file);
    }

    auto write_tensor_txx(const std::string &path, const double *const *const *tensor) -> void
    {
        if (!tensor) return;
        if (tdep) {
            auto dset = file->getDataSet(path);
            for (size_t j = 0; j < run_cols.size(); ++j) {
                dset.select({run_cols[j], 0, 0}, {1, 3, 3}).write_raw(&tensor[j][0][0]);
            }
        } else {
            file->getDataSet(path).write_raw(&tensor[0][0][0]);
        }
    }

    auto write_tensor_flat(const std::string &path, const std::vector<double> &flat) -> void
    {
        auto dset = file->getDataSet(path);
        if (tdep) {
            for (size_t j = 0; j < run_cols.size(); ++j) {
                dset.select({run_cols[j], 0, 0}, {1, 3, 3}).write_raw(flat.data() + j * 9);
            }
        } else {
            dset.write_raw(flat.data());
        }
    }
};

KappaResultIOH5::KappaResultIOH5(std::string filename) : impl(std::make_unique<Impl>())
{
    impl->filename = std::move(filename);
}

KappaResultIOH5::~KappaResultIOH5() = default;

void KappaResultIOH5::open_or_create(const KappaFileMetaH5 &fmeta, const KappaChannelMetaH5 &channel,
                                     const bool reset_channel)
{
    impl->fmeta = fmeta;
    impl->ntemp = fmeta.temperatures.size();
    impl->tdep = fmeta.temperature_resolved;

    auto need_rebuild = false;
    std::vector<std::string> drops;
    std::vector<double> old_temps;
    std::vector<unsigned char> old_valid;

    impl->file_temps = fmeta.temperatures;
    impl->file_fc2temps.assign(impl->file_temps.size(), fmeta.fc2_temperature);

    if (std::filesystem::exists(impl->filename)) {
        const HighFive::File oldfile(impl->filename, HighFive::File::ReadOnly);
        check_h5_schema(oldfile, h5_schema_kappa_result, kappa_version_tdep);
        impl->validate_layout_mode(oldfile);
        impl->validate_metadata(oldfile);

        if (impl->tdep) {
            // Merge the run's temperatures into the file grid (sorted union);
            // any new column forces a rebuild.
            old_temps = H5Easy::load<std::vector<double>>(oldfile, "/metadata/temperatures");
            const auto old_fc2temps = H5Easy::load<std::vector<double>>(oldfile, "/metadata/fc2_temperatures");
            oldfile.getDataSet("/kappa/valid").read(old_valid);

            auto merged = old_temps;
            auto merged_fc2 = old_fc2temps;
            for (size_t i = 0; i < fmeta.temperatures.size(); ++i) {
                auto found = false;
                for (const auto t: merged) {
                    if (std::abs(t - fmeta.temperatures[i]) < eps6) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    merged.push_back(fmeta.temperatures[i]);
                    merged_fc2.push_back(fmeta.fc2_temperature);
                    need_rebuild = true;
                }
            }
            // keep sorted by temperature
            std::vector<size_t> order(merged.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&merged](const size_t a, const size_t b) {
                return merged[a] < merged[b];
            });
            impl->file_temps.clear();
            impl->file_fc2temps.clear();
            for (const auto i: order) {
                impl->file_temps.push_back(merged[i]);
                impl->file_fc2temps.push_back(merged_fc2[i]);
            }
        }

        if (oldfile.exist(channel_path(channel.tag))) {
            const auto match = impl->channel_matches(oldfile, channel);
            if (!match && !reset_channel) {
                exit("kappa_result_io",
                     "KPOINT information of an existing channel in the kappa.h5 file is not consistent.\n"
                     " Set RESTART = 0 (or RESTART_4PH = 0) to discard it, or remove the file.");
            }
            if (!match || reset_channel) {
                drops.push_back(channel.tag);
                need_rebuild = true;
            }
        } else {
            need_rebuild = true;
        }
        if (!impl->kappa_group_matches(oldfile)) need_rebuild = true;
        if (!impl->isotope_group_matches(oldfile)) need_rebuild = true;
    } else {
        need_rebuild = true;
    }

    // The isotope linewidths live on the 3ph mesh: discard them together
    // with a dropped 3ph channel, and when isotope scattering is off.
    if (fmeta.isotope == 0 || std::find(drops.begin(), drops.end(), std::string("3ph")) != drops.end()) {
        drops.emplace_back("isotope");
    }

    impl->compute_run_cols();

    if (need_rebuild) {
        if (impl->tdep) {
            impl->rebuild_tdep({channel}, drops, old_temps, old_valid);
        } else {
            impl->rebuild({channel}, drops);
        }
    } else {
        impl->file = std::make_unique<HighFive::File>(impl->filename, HighFive::File::ReadWrite);
    }
    impl->write_basis_slices(channel);
    impl->reset_kappa_valid();
}

void KappaResultIOH5::ensure_channel(const KappaChannelMetaH5 &channel, const bool reset_channel)
{
    if (!impl->file) {
        exit("kappa_result_io", "ensure_channel called before open_or_create");
    }
    std::vector<unsigned char> valid_now;
    if (impl->tdep) {
        impl->file->getDataSet("/kappa/valid").read(valid_now);
    }

    if (impl->file->exist(channel_path(channel.tag))) {
        const auto match = impl->channel_matches(*impl->file, channel);
        if (!match && !reset_channel) {
            exit("kappa_result_io",
                 "KPOINT information of an existing channel in the kappa.h5 file is not consistent.\n"
                 " Set RESTART = 0 (or RESTART_4PH = 0) to discard it, or remove the file.");
        }
        if (!match || reset_channel) {
            if (impl->tdep) {
                impl->rebuild_tdep({channel}, {channel.tag}, impl->file_temps, valid_now);
            } else {
                impl->rebuild({channel}, {channel.tag});
            }
        }
    } else {
        if (impl->tdep) {
            impl->rebuild_tdep({channel}, {}, impl->file_temps, valid_now);
        } else {
            impl->rebuild({channel}, {});
        }
    }
    impl->write_basis_slices(channel);
}

bool KappaResultIOH5::open_or_create_for_ibte(const KappaFileMetaH5 &fmeta, const IbteMetaH5 &imeta, const bool reset)
{
    impl->fmeta = fmeta;
    impl->ntemp = fmeta.temperatures.size();
    impl->tdep = fmeta.temperature_resolved;
    impl->file_temps = fmeta.temperatures;
    impl->file_fc2temps.assign(impl->file_temps.size(), fmeta.fc2_temperature);

    if (impl->tdep) {
        // No temperature-resolved layout of /iterativebte yet.
        return false;
    }

    impl->imeta = imeta;
    impl->has_imeta = true;

    auto need_rebuild = false;
    auto drop_ibte = false;

    if (std::filesystem::exists(impl->filename)) {
        const HighFive::File oldfile(impl->filename, HighFive::File::ReadOnly);
        check_h5_schema(oldfile, h5_schema_kappa_result, kappa_version_tdep);
        impl->validate_layout_mode(oldfile);
        impl->validate_metadata(oldfile);

        if (oldfile.exist("/iterativebte")) {
            if (!impl->ibte_group_matches(oldfile) || reset) {
                drop_ibte = true;
                need_rebuild = true;
            }
        } else {
            need_rebuild = true;
        }
    } else {
        need_rebuild = true;
    }

    impl->compute_run_cols();

    if (need_rebuild) {
        impl->rebuild({}, {}, drop_ibte);
    } else {
        impl->file = std::make_unique<HighFive::File>(impl->filename, HighFive::File::ReadWrite);
    }
    return true;
}

std::vector<unsigned char> KappaResultIOH5::load_ibte_computed() const
{
    std::vector<unsigned char> flags(impl->ntemp, 0);
    if (!impl->file || !impl->file->exist("/iterativebte")) return flags;
    impl->file->getDataSet("/iterativebte/computed").read(flags);
    return flags;
}

void KappaResultIOH5::load_ibte_kappa(const size_t itemp, double *kappa9, unsigned char &converged) const
{
    impl->file->getDataSet("/iterativebte/kappa").select({itemp, 0, 0}, {1, 3, 3}).read(kappa9);
    impl->file->getDataSet("/iterativebte/converged").select({itemp}, {1}).read(&converged);
}

void KappaResultIOH5::load_ibte_dF(const size_t itemp, double *dF_flat) const
{
    const size_t nkirr = impl->imeta.nk_irred;
    const size_t ns = impl->imeta.ns;
    impl->file->getDataSet("/iterativebte/dF").select({itemp, 0, 0, 0}, {1, nkirr, ns, 3}).read(dF_flat);
}

void KappaResultIOH5::store_ibte_temperature(const size_t itemp, const double *Q, const double *dF,
                                             const double *kappa9, const unsigned char converged)
{
    if (!impl->file || !impl->has_imeta) {
        exit("kappa_result_io", "store_ibte_temperature called before open_or_create_for_ibte");
    }
    const size_t nkirr = impl->imeta.nk_irred;
    const size_t ns = impl->imeta.ns;

    // Data (Q, dF, kappa, converged) first, the computed flag second, each
    // made durable, so a persisted flag proves the temperature hit the disk.
    impl->file->getDataSet("/iterativebte/Q").select({itemp, 0, 0}, {1, nkirr, ns}).write_raw(Q);
    impl->file->getDataSet("/iterativebte/dF").select({itemp, 0, 0, 0}, {1, nkirr, ns, 3}).write_raw(dF);
    impl->file->getDataSet("/iterativebte/kappa").select({itemp, 0, 0}, {1, 3, 3}).write_raw(kappa9);
    impl->file->getDataSet("/iterativebte/converged").select({itemp}, {1}).write_raw(&converged);
    h5_flush_and_fsync(*impl->file);

    const unsigned char one = 1;
    impl->file->getDataSet("/iterativebte/computed").select({itemp}, {1}).write_raw(&one);
    h5_flush_and_fsync(*impl->file);
}

std::vector<int> KappaResultIOH5::load_computed_gamma(const std::string &tag, double **damping) const
{
    const auto path = channel_path(tag);
    const auto dset_flags = impl->file->getDataSet(path + "/gamma_computed");
    const auto dset = impl->file->getDataSet(path + "/gamma");
    const auto dims = dset.getDimensions();
    std::vector<double> buf(dims[0] * dims[1]);
    dset.read(buf.data());

    std::vector<int> rows_done;
    const auto factor = kayser_to_Ry; // cm^-1 -> internal

    if (impl->tdep) {
        // A mode counts as done only when every temperature column of THIS
        // run is flagged.
        std::vector<unsigned char> flags(dims[0] * dims[1]);
        dset_flags.read(flags.data());
        const auto nt = dims[1];
        for (size_t row = 0; row < dims[0]; ++row) {
            auto done = true;
            for (const auto col: impl->run_cols) {
                if (!flags[row * nt + col]) {
                    done = false;
                    break;
                }
            }
            if (!done) continue;
            for (size_t j = 0; j < impl->run_cols.size(); ++j) {
                damping[row][j] = buf[row * nt + impl->run_cols[j]] * factor;
            }
            rows_done.push_back(static_cast<int>(row));
        }
        return rows_done;
    }

    std::vector<unsigned char> flags;
    dset_flags.read(flags);
    for (size_t row = 0; row < flags.size(); ++row) {
        if (!flags[row]) continue;
        for (size_t k = 0; k < dims[1]; ++k) {
            damping[row][k] = buf[row * dims[1] + k] * factor;
        }
        rows_done.push_back(static_cast<int>(row));
    }
    return rows_done;
}

void KappaResultIOH5::store_gamma_batch(const std::string &tag, const unsigned int first_row, const unsigned int nrow,
                                        const double *const *damping)
{
    if (nrow == 0) return;
    const auto path = channel_path(tag);
    const auto ntemp = impl->ntemp;
    const auto factor = Ry_to_kayser; // internal -> cm^-1

    auto dset_gamma = impl->file->getDataSet(path + "/gamma");
    auto dset_flags = impl->file->getDataSet(path + "/gamma_computed");

    // Data first, flags second, each made durable before the next step:
    // a persisted flag therefore proves its row hit the disk.
    if (impl->tdep) {
        std::vector<double> col(nrow);
        for (size_t j = 0; j < impl->run_cols.size(); ++j) {
            for (unsigned int r = 0; r < nrow; ++r) {
                col[r] = damping[first_row + r][j] * factor;
            }
            dset_gamma.select({first_row, impl->run_cols[j]}, {nrow, 1}).write_raw(col.data());
        }
        h5_flush_and_fsync(*impl->file);

        const std::vector<unsigned char> ones(nrow, 1);
        for (const auto col_idx: impl->run_cols) {
            dset_flags.select({first_row, col_idx}, {nrow, 1}).write_raw(ones.data());
        }
        h5_flush_and_fsync(*impl->file);
        return;
    }

    std::vector<double> buf(static_cast<size_t>(nrow) * ntemp);
    for (unsigned int r = 0; r < nrow; ++r) {
        for (size_t k = 0; k < ntemp; ++k) {
            buf[r * ntemp + k] = damping[first_row + r][k] * factor;
        }
    }
    dset_gamma.select({first_row, 0}, {nrow, ntemp}).write_raw(buf.data());
    h5_flush_and_fsync(*impl->file);

    const std::vector<unsigned char> ones(nrow, 1);
    dset_flags.select({first_row}, {nrow}).write_raw(ones.data());
    h5_flush_and_fsync(*impl->file);
}

void KappaResultIOH5::store_gamma_rows(const std::string &tag, const std::vector<int> &rows,
                                       const double *const *damping)
{
    if (rows.empty()) return;
    if (impl->tdep) {
        exit("kappa_result_io", "Legacy text import is not supported for temperature-resolved files");
    }
    const auto path = channel_path(tag);
    const auto ntemp = impl->ntemp;
    const auto factor = Ry_to_kayser;

    auto dset_gamma = impl->file->getDataSet(path + "/gamma");
    std::vector<double> buf(ntemp);
    for (const auto row: rows) {
        for (size_t k = 0; k < ntemp; ++k) {
            buf[k] = damping[row][k] * factor;
        }
        dset_gamma.select({static_cast<size_t>(row), 0}, {1, ntemp}).write_raw(buf.data());
    }
    h5_flush_and_fsync(*impl->file);

    auto dset_flags = impl->file->getDataSet(path + "/gamma_computed");
    const unsigned char one = 1;
    for (const auto row: rows) {
        dset_flags.select({static_cast<size_t>(row)}, {1}).write_raw(&one);
    }
    h5_flush_and_fsync(*impl->file);
}

void KappaResultIOH5::store_kappa(const double *const *const *kappa_peierls, const double *const *const *kappa_3ph_only,
                                  const double *const *const *kappa_coherent, const double *const *const *kappa_spec,
                                  const double *const *gamma_isotope)
{
    impl->write_tensor_txx("/kappa/kappa_peierls", kappa_peierls);

    if (impl->fmeta.isotope > 0 && gamma_isotope) {
        auto dset = impl->file->getDataSet("/scattering/isotope/gamma");
        const auto dims = dset.getDimensions();
        const auto factor = Ry_to_kayser; // internal -> cm^-1
        const auto d1 = impl->tdep ? dims[1] : dims[0];
        const auto d2 = impl->tdep ? dims[2] : dims[1];
        std::vector<double> buf(d1 * d2);
        for (size_t i = 0; i < d1; ++i) {
            for (size_t k = 0; k < d2; ++k) {
                buf[i * d2 + k] = gamma_isotope[i][k] * factor;
            }
        }
        if (impl->tdep) {
            for (const auto col: impl->run_cols) {
                dset.select({col, 0, 0}, {1, d1, d2}).write_raw(buf.data());
            }
        } else {
            dset.write_raw(buf.data());
        }
    }
    if (impl->fmeta.with_kappa_3ph_only) impl->write_tensor_txx("/kappa/kappa_3ph_only", kappa_3ph_only);
    if (impl->fmeta.with_kappa_coherent) {
        impl->write_tensor_txx("/kappa/kappa_coherent", kappa_coherent);
        if (kappa_peierls && kappa_coherent) {
            std::vector<double> total(impl->ntemp * 9);
            for (size_t j = 0; j < impl->ntemp; ++j) {
                for (auto a = 0; a < 3; ++a) {
                    for (auto b = 0; b < 3; ++b) {
                        total[j * 9 + 3 * a + b] = kappa_peierls[j][a][b] + kappa_coherent[j][a][b];
                    }
                }
            }
            impl->write_tensor_flat("/kappa/kappa_total", total);
        }
    }
    if (impl->fmeta.with_kappa_spec && kappa_spec) {
        const auto ne = impl->fmeta.energy_axis.size();
        auto dset = impl->file->getDataSet("/kappa/kappa_spec");
        if (impl->tdep) {
            std::vector<double> col(ne * 3);
            for (size_t j = 0; j < impl->run_cols.size(); ++j) {
                for (size_t e = 0; e < ne; ++e) {
                    for (auto x = 0; x < 3; ++x) col[e * 3 + x] = kappa_spec[e][j][x];
                }
                dset.select({0, impl->run_cols[j], 0}, {ne, 1, 3}).write_raw(col.data());
            }
        } else {
            dset.write_raw(&kappa_spec[0][0][0]);
        }
    }
    h5_flush_and_fsync(*impl->file);

    const unsigned char one = 1;
    auto dset_valid = impl->file->getDataSet("/kappa/valid");
    if (impl->tdep) {
        for (const auto col: impl->run_cols) {
            dset_valid.select({col}, {1}).write_raw(&one);
        }
    } else {
        dset_valid.write_raw(&one);
    }
    h5_flush_and_fsync(*impl->file);
}

const std::string &KappaResultIOH5::get_filename() const
{
    return impl->filename;
}
