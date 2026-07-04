/*
 kappa_result_io.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "kappa_result_io.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <utility>
#include "constants.h"
#include "error.h"
#include "hdf5_parser.h"

using namespace PHON_NS;

namespace
{
const std::string str_scattering = "/scattering/";

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
    size_t ntemp = 0;

    auto write_metadata(HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        dump(fh, "/metadata/temperatures", fmeta.temperatures);
        dumpAttribute(fh, "/metadata/temperatures", "unit", std::string("K"));
        dump(fh, "/metadata/classical", fmeta.classical);
        dump(fh, "/metadata/ismear", fmeta.ismear);
        dump(fh, "/metadata/smearing_width", fmeta.smearing_width);
        dumpAttribute(fh, "/metadata/smearing_width", "unit", std::string("cm^-1"));
        dump(fh, "/metadata/fcs_file", fmeta.fcs_file);

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
    // system/temperature mismatches, warnings for the rest.
    auto validate_metadata(const HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        const auto &efile = fh;
        const auto natoms = load<size_t>(efile, "/metadata/PrimitiveCell/number_of_atoms");
        const auto nelems = load<size_t>(efile, "/metadata/PrimitiveCell/number_of_elements");
        if (natoms != static_cast<size_t>(fmeta.x_fractional.rows()) || nelems != fmeta.elements.size()) {
            exit("kappa_result_io", "SYSTEM information in the kappa.h5 file is not consistent");
        }

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
        if (ismear_file != -1 &&
            std::abs(load<double>(efile, "/metadata/smearing_width") - fmeta.smearing_width) >= eps4) {
            warn("kappa_result_io", "Smearing width is not consistent");
        }
    }

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

        dump(fh, path + "/frequencies", cmeta.frequencies);
        dumpAttribute(fh, path + "/frequencies", "unit", std::string("cm^-1"));

        const size_t nequiv_total = knum_flat.size();
        auto dset_vel = fh.createDataSet<double>(
            path + "/velocities", HighFive::DataSpace({nequiv_total, cmeta.ns, 3}));
        dset_vel.write_raw(cmeta.velocities.data());
        dset_vel.createAttribute("unit", std::string("m/s"));

        const size_t nrows = static_cast<size_t>(cmeta.nk_irred) * cmeta.ns;
        auto dset_gamma = h5_create_dataset_prealloc<double>(fh, path + "/gamma", {nrows, ntemp});
        dset_gamma.createAttribute("unit", std::string("cm^-1"));
        h5_create_dataset_prealloc<unsigned char>(fh, path + "/gamma_computed", {nrows});
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

    auto create_kappa_group(HighFive::File &fh) const -> void
    {
        using namespace H5Easy;
        h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_total", {ntemp, 3, 3})
            .createAttribute("unit", std::string("W/mK"));
        if (fmeta.with_kappa_3ph_only) {
            h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_3ph_only", {ntemp, 3, 3})
                .createAttribute("unit", std::string("W/mK"));
        }
        if (fmeta.with_kappa_coherent) {
            h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_coherent", {ntemp, 3, 3})
                .createAttribute("unit", std::string("W/mK"));
        }
        if (fmeta.with_kappa_coherent_block) {
            h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_coherent_block", {ntemp, 3, 3})
                .createAttribute("unit", std::string("W/mK"));
        }
        if (fmeta.with_kappa_spec) {
            dump(fh, "/kappa/energy_axis", fmeta.energy_axis);
            dumpAttribute(fh, "/kappa/energy_axis", "unit", std::string("cm^-1"));
            h5_create_dataset_prealloc<double>(fh, "/kappa/kappa_spec",
                                               {fmeta.energy_axis.size(), ntemp, 3})
                .createAttribute("unit", std::string("W/mK/cm^-1"));
        }
        // Validity marker as a raw-data dataset (not an attribute) so setting
        // it never touches file metadata.
        h5_create_dataset_prealloc<unsigned char>(fh, "/kappa/valid", {1});
    }

    // The /kappa layout is fixed at setup; a restart run with different
    // output options needs a rebuild.
    auto kappa_group_matches(const HighFive::File &fh) const -> bool
    {
        const auto need = [&](const std::string &name, const bool required) {
            return fh.exist("/kappa/" + name) == required;
        };
        if (!need("kappa_total", true)) return false;
        if (!need("kappa_3ph_only", fmeta.with_kappa_3ph_only)) return false;
        if (!need("kappa_coherent", fmeta.with_kappa_coherent)) return false;
        if (!need("kappa_coherent_block", fmeta.with_kappa_coherent_block)) return false;
        if (!need("kappa_spec", fmeta.with_kappa_spec)) return false;
        if (fmeta.with_kappa_spec) {
            const auto dims = fh.getDataSet("/kappa/kappa_spec").getDimensions();
            if (dims.size() != 3 || dims[0] != fmeta.energy_axis.size() || dims[1] != ntemp) return false;
        } else {
            const auto dims = fh.getDataSet("/kappa/kappa_total").getDimensions();
            if (dims.size() != 3 || dims[0] != ntemp) return false;
        }
        return true;
    }

    // Rebuild the file as <filename>.part and atomically swap it in:
    // fresh metadata, /kappa, and schema attributes; channels listed in
    // tags_drop are discarded, every other existing channel is copied
    // verbatim (H5Ocopy preserves data and creation properties), and the
    // channels in to_create are created empty unless already carried over.
    auto rebuild(const std::vector<KappaChannelMetaH5> &to_create,
                 const std::vector<std::string> &tags_drop) -> void
    {
        file.reset();

        const auto part = h5_part_filename(filename);
        if (std::filesystem::exists(part)) {
            warn("kappa_result_io", "Removing a stale .part file of a previously interrupted run.");
            std::filesystem::remove(part);
        }

        {
            HighFive::File newfile(part, HighFive::File::ReadWrite | HighFive::File::Create |
                                             HighFive::File::Excl);

            if (std::filesystem::exists(filename)) {
                const HighFive::File oldfile(filename, HighFive::File::ReadOnly);
                if (oldfile.exist("/scattering")) {
                    newfile.createGroup("/scattering");
                    for (const auto &tag: oldfile.getGroup("/scattering").listObjectNames()) {
                        if (std::find(tags_drop.begin(), tags_drop.end(), tag) != tags_drop.end()) continue;
                        const auto path = channel_path(tag);
                        if (H5Ocopy(oldfile.getId(), path.c_str(), newfile.getId(), path.c_str(),
                                    H5P_DEFAULT, H5P_DEFAULT) < 0) {
                            exit("kappa_result_io", "Failed to carry over a scattering channel", tag.c_str());
                        }
                    }
                }
            }

            write_metadata(newfile);
            for (const auto &cmeta: to_create) {
                if (!newfile.exist(channel_path(cmeta.tag))) {
                    create_channel(newfile, cmeta);
                }
            }
            create_kappa_group(newfile);
            stamp_h5_schema(newfile, h5_schema_kappa_result, h5_version_kappa_result);
            h5_flush_and_fsync(newfile);
        }

        h5_publish_file(part, filename);
        file = std::make_unique<HighFive::File>(filename, HighFive::File::ReadWrite);
    }

    auto reset_kappa_valid() -> void
    {
        const unsigned char zero = 0;
        file->getDataSet("/kappa/valid").write_raw(&zero);
        h5_flush_and_fsync(*file);
    }

    auto write_tensor_txx(const std::string &path, const double *const *const *tensor) -> void
    {
        if (!tensor) return;
        file->getDataSet(path).write_raw(&tensor[0][0][0]);
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

    auto need_rebuild = false;
    std::vector<std::string> drops;

    if (std::filesystem::exists(impl->filename)) {
        const HighFive::File oldfile(impl->filename, HighFive::File::ReadOnly);
        check_h5_schema(oldfile, h5_schema_kappa_result, h5_version_kappa_result);
        impl->validate_metadata(oldfile);

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
    } else {
        need_rebuild = true;
    }

    if (need_rebuild) {
        impl->rebuild({channel}, drops);
    } else {
        impl->file = std::make_unique<HighFive::File>(impl->filename, HighFive::File::ReadWrite);
        impl->reset_kappa_valid();
    }
}

void KappaResultIOH5::ensure_channel(const KappaChannelMetaH5 &channel, const bool reset_channel)
{
    if (!impl->file) {
        exit("kappa_result_io", "ensure_channel called before open_or_create");
    }
    if (impl->file->exist(channel_path(channel.tag))) {
        const auto match = impl->channel_matches(*impl->file, channel);
        if (match && !reset_channel) return;
        if (!match && !reset_channel) {
            exit("kappa_result_io",
                 "KPOINT information of an existing channel in the kappa.h5 file is not consistent.\n"
                 " Set RESTART = 0 (or RESTART_4PH = 0) to discard it, or remove the file.");
        }
        impl->rebuild({channel}, {channel.tag});
    } else {
        impl->rebuild({channel}, {});
    }
}

std::vector<int> KappaResultIOH5::load_computed_gamma(const std::string &tag, double **damping) const
{
    const auto path = channel_path(tag);
    std::vector<unsigned char> flags;
    impl->file->getDataSet(path + "/gamma_computed").read(flags);

    const auto dset = impl->file->getDataSet(path + "/gamma");
    const auto dims = dset.getDimensions();
    std::vector<double> buf(dims[0] * dims[1]);
    dset.read(buf.data());

    std::vector<int> rows_done;
    const auto factor = time_ry / Hz_to_kayser; // cm^-1 -> internal
    for (size_t row = 0; row < flags.size(); ++row) {
        if (!flags[row]) continue;
        for (size_t k = 0; k < dims[1]; ++k) {
            damping[row][k] = buf[row * dims[1] + k] * factor;
        }
        rows_done.push_back(static_cast<int>(row));
    }
    return rows_done;
}

void KappaResultIOH5::store_gamma_batch(const std::string &tag, const unsigned int first_row,
                                        const unsigned int nrow, const double *const *damping)
{
    if (nrow == 0) return;
    const auto path = channel_path(tag);
    const auto ntemp = impl->ntemp;
    const auto factor = Hz_to_kayser / time_ry; // internal -> cm^-1

    std::vector<double> buf(static_cast<size_t>(nrow) * ntemp);
    for (unsigned int r = 0; r < nrow; ++r) {
        for (size_t k = 0; k < ntemp; ++k) {
            buf[r * ntemp + k] = damping[first_row + r][k] * factor;
        }
    }

    // Data first, flags second, each made durable before the next step:
    // a persisted flag therefore proves its row hit the disk.
    impl->file->getDataSet(path + "/gamma")
        .select({first_row, 0}, {nrow, ntemp})
        .write_raw(buf.data());
    h5_flush_and_fsync(*impl->file);

    const std::vector<unsigned char> ones(nrow, 1);
    impl->file->getDataSet(path + "/gamma_computed").select({first_row}, {nrow}).write_raw(ones.data());
    h5_flush_and_fsync(*impl->file);
}

void KappaResultIOH5::store_gamma_rows(const std::string &tag, const std::vector<int> &rows,
                                       const double *const *damping)
{
    if (rows.empty()) return;
    const auto path = channel_path(tag);
    const auto ntemp = impl->ntemp;
    const auto factor = Hz_to_kayser / time_ry;

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

void KappaResultIOH5::store_kappa(const double *const *const *kappa_total,
                                  const double *const *const *kappa_3ph_only,
                                  const double *const *const *kappa_coherent,
                                  const double *const *const *kappa_coherent_block,
                                  const double *const *const *kappa_spec)
{
    impl->write_tensor_txx("/kappa/kappa_total", kappa_total);
    if (impl->fmeta.with_kappa_3ph_only) impl->write_tensor_txx("/kappa/kappa_3ph_only", kappa_3ph_only);
    if (impl->fmeta.with_kappa_coherent) impl->write_tensor_txx("/kappa/kappa_coherent", kappa_coherent);
    if (impl->fmeta.with_kappa_coherent_block) {
        impl->write_tensor_txx("/kappa/kappa_coherent_block", kappa_coherent_block);
    }
    if (impl->fmeta.with_kappa_spec) impl->write_tensor_txx("/kappa/kappa_spec", kappa_spec);
    h5_flush_and_fsync(*impl->file);

    const unsigned char one = 1;
    impl->file->getDataSet("/kappa/valid").write_raw(&one);
    h5_flush_and_fsync(*impl->file);
}

const std::string &KappaResultIOH5::get_filename() const
{
    return impl->filename;
}
