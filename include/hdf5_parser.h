/*
 hdf5_parser.h

 Copyright (c) 2023 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/


#pragma once

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef H5_USE_EIGEN
#define H5_USE_EIGEN 1
#endif
#include <highfive/H5Easy.hpp>

#include "constants.h"
#include "units.h"
#include "version.h"

// ===========================================================================
//  Schema tagging, format versioning, and crash-safe write utilities
//
//  Every ALAMODE-written HDF5 file that follows the versioned-schema
//  convention carries two root attributes: "schema" (a namespaced string
//  such as "alamode:kappa_result") and "format_version" (an integer major
//  version; minor evolutions of a schema must be purely additive).
//  Files written before this convention carry neither attribute and are
//  treated as legacy version 0, which is accepted only where explicitly
//  allowed (the force-constant files produced by older alm versions).
// ===========================================================================

// Schema identifiers and current format versions of the ALAMODE HDF5 files.
inline const std::string h5_schema_force_constants = "alamode:force_constants";
inline const std::string h5_schema_kappa_result = "alamode:kappa_result";
inline const std::string h5_schema_scph_state = "alamode:scph_state";
inline const std::string h5_schema_eigenvalues = "alamode:eigenvalues";
inline const std::string h5_schema_eigenvectors = "alamode:eigenvectors";
inline constexpr int h5_version_eigen = 1;
constexpr int h5_version_force_constants = 1;
constexpr int h5_version_kappa_result = 1;
constexpr int h5_version_scph_state = 1;

// Write the root schema attributes plus provenance (code version, creation
// date). Existing attributes are replaced, so this is safe to call again on
// a file whose content is being rebuilt.
inline auto stamp_h5_schema(HighFive::File &file, const std::string &schema_name, const int format_version) -> void
{
    const auto replace_string_attr = [&file](const std::string &key, const std::string &val) {
        if (file.hasAttribute(key)) file.deleteAttribute(key);
        file.createAttribute(key, val);
    };
    if (file.hasAttribute("format_version")) file.deleteAttribute("format_version");
    file.createAttribute("format_version", format_version);
    replace_string_attr("schema", schema_name);
    replace_string_attr("alamode_version", ALAMODE_VERSION);

    const std::time_t now = std::time(nullptr);
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y-%b-%d %T", std::localtime(&now));
    replace_string_attr("created_date", std::string(time_str));
}

// Validate the root schema attributes of an ALAMODE HDF5 file and return the
// detected format version (0 = legacy file without attributes). Unknown
// schemas and versions newer than max_supported_version are fatal: such a
// file was written by a different or newer code and must not be reinterpreted.
inline auto check_h5_schema(const HighFive::File &file, const std::string &expected_schema,
                            const int max_supported_version, const bool allow_legacy_unversioned = false) -> int
{
    if (!file.hasAttribute("schema")) {
        if (allow_legacy_unversioned) return 0;
        std::cout << "Error: file " << file.getName() << " has no schema attribute; expected schema \""
                  << expected_schema << "\".\n";
        exit(1);
    }
    std::string schema_name;
    file.getAttribute("schema").read(schema_name);
    if (schema_name != expected_schema) {
        std::cout << "Error: file " << file.getName() << " has schema \"" << schema_name << "\" but \""
                  << expected_schema << "\" is expected here.\n";
        exit(1);
    }
    int version = 0;
    file.getAttribute("format_version").read(version);
    if (version > max_supported_version) {
        std::cout << "Error: file " << file.getName() << " uses format_version " << version << " of schema \""
                  << expected_schema << "\", but this build supports up to version " << max_supported_version
                  << ". Please use a newer ALAMODE.\n";
        exit(1);
    }
    return version;
}

// Flush HDF5 buffers and force the data down to the storage device.
// H5Fflush alone only hands the bytes to the OS page cache, which is enough
// to survive a killed process but not a kernel crash or power loss; the
// fsync closes that gap.
inline auto h5_flush_and_fsync(HighFive::File &file) -> void
{
    file.flush();
#ifndef _WIN32
    void *vfd_handle = nullptr;
    if (H5Fget_vfd_handle(file.getId(), H5P_DEFAULT, &vfd_handle) >= 0 && vfd_handle) {
        const int fd = *static_cast<const int *>(vfd_handle);
        if (fd >= 0) ::fsync(fd);
    }
#endif
}

// Make a just-renamed directory entry durable.
inline auto h5_fsync_parent_directory(const std::string &filepath) -> void
{
#ifndef _WIN32
    const std::filesystem::path p(filepath);
    const auto dir = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#endif
}

// Name of the staging file used while building a new HDF5 file. Kept in the
// same directory as the target so the final rename never crosses filesystems.
inline auto h5_part_filename(const std::string &path_final) -> std::string
{
    return path_final + ".part";
}

// Atomically publish a fully-written staging file under its final name.
// The rename is atomic on POSIX filesystems; on other platforms this is
// best-effort. The caller must have flushed and closed the file first.
inline auto h5_publish_file(const std::string &path_part, const std::string &path_final) -> void
{
    std::error_code ec;
    std::filesystem::rename(path_part, path_final, ec);
    if (ec) {
        std::cout << "Error: failed to rename " << path_part << " to " << path_final << ": " << ec.message() << '\n';
        exit(1);
    }
    h5_fsync_parent_directory(path_final);
}

// Create a dataset whose file space is fully allocated and zero-filled at
// creation time (contiguous layout, H5D_ALLOC_TIME_EARLY). After creation,
// writes into it are pure in-place raw-data updates that can never damage
// the file structure — the property the crash-safe restart files rely on.
template <typename T>
inline auto h5_create_dataset_prealloc(HighFive::File &file, const std::string &path,
                                       const std::vector<size_t> &dims) -> HighFive::DataSet
{
    HighFive::RawPropertyList<HighFive::PropertyType::DATASET_CREATE> props;
    props.add(H5Pset_alloc_time, H5D_ALLOC_TIME_EARLY);
    const T fill_value{};
    const auto dtype = HighFive::create_datatype<T>();
    props.add(H5Pset_fill_value, dtype.getId(), &fill_value);
    props.add(H5Pset_fill_time, H5D_FILL_TIME_ALLOC);
    return file.createDataSet<T>(path, HighFive::DataSpace(dims), props);
}

// Resolve a physical temperature to a row index of a temperature-grid
// dataset (e.g. /settings/temperatures of an alamode:scph_state file).
// A missing match is fatal and lists the temperatures the file provides.
inline auto h5_resolve_temperature_index(const H5Easy::File &file, const double temperature, const double tolerance,
                                         const std::string &path_temperatures) -> int
{
    const auto temps = H5Easy::load<std::vector<double>>(file, path_temperatures);
    for (size_t i = 0; i < temps.size(); ++i) {
        if (std::abs(temps[i] - temperature) < tolerance) return static_cast<int>(i);
    }
    std::cout << "Error: temperature " << temperature << " K is not available in " << file.getName()
              << ".\n Available temperatures (K):";
    for (const auto &t: temps)
        std::cout << ' ' << t;
    std::cout << '\n';
    exit(1);
}

// Read the "unit" attribute of a dataset; returns an empty string when the
// attribute is absent (files written before the attribute existed).
inline auto get_unit_attribute_h5(const H5Easy::File &file, const std::string &path) -> std::string
{
    if (!file.getDataSet(path).hasAttribute("unit")) return "";
    return H5Easy::loadAttribute<std::string>(file, path, "unit");
}

// Multiply-by factor that converts a length-type dataset ("bohr"/"angstrom")
// into bohr. An absent attribute is interpreted as bohr (legacy files).
inline auto h5_length_factor_to_bohr(const H5Easy::File &file, const std::string &path,
                                     std::string *unit_detected = nullptr) -> double
{
    const auto unit = get_unit_attribute_h5(file, path);
    if (unit_detected) *unit_detected = unit;
    if (unit.empty() || unit == "bohr") return 1.0;
    if (unit == "angstrom") return 1.0 / Bohr_in_Angstrom;
    std::cout << "Error: unsupported unit \"" << unit << "\" of dataset " << path << " in file " << file.getName()
              << ". Supported units are \"bohr\" and \"angstrom\".\n";
    exit(1);
}

// Multiply-by factor that converts force constant values of the given order
// ("Ry/bohr^m" / "eV/angstrom^m" with m = order + 2) into Ry/bohr^m.
// An absent attribute is interpreted as Ry/bohr^m (legacy files).
inline auto h5_fc_factor_to_ry_bohr(const H5Easy::File &file, const std::string &path, const int order,
                                    std::string *unit_detected = nullptr) -> double
{
    const auto unit = get_unit_attribute_h5(file, path);
    if (unit_detected) *unit_detected = unit;
    const auto str_m = std::to_string(order + 2);
    if (unit.empty() || unit == "Ry/bohr^" + str_m) return 1.0;
    if (unit == "eV/angstrom^" + str_m) return 1.0 / units::fc_ry_bohr_to_ev_ang(order + 2);
    std::cout << "Error: unsupported unit \"" << unit << "\" of dataset " << path << " in file " << file.getName()
              << ". Supported units are \"Ry/bohr^" << str_m << "\" and \"eV/angstrom^" << str_m << "\".\n";
    exit(1);
}

// Resolve a user-facing cell-type string ("prim"/"super"/...) to the HDF5 group
// name ("PrimitiveCell"/"SuperCell"). Shared by the get_*_from_h5 helpers below.
inline auto resolve_h5_cell_name(const std::string &celltype) -> std::string
{
    if (celltype == "prim" || celltype == "primitive" || celltype == "PrimitiveCell") {
        return "PrimitiveCell";
    }
    if (celltype == "super" || celltype == "supercell" || celltype == "SuperCell") {
        return "SuperCell";
    }
    std::cout << "resolve_h5_cell_name: Invalid celltype " << celltype << '\n';
    exit(1);
}

inline auto get_structures_from_h5(const H5Easy::File &file, const std::string &celltype, Eigen::Matrix3d &lavec,
                                   Eigen::MatrixXd &x_fractional, std::vector<int> &kind_index,
                                   std::vector<std::string> &element_names,
                                   std::string *length_unit_detected = nullptr) -> void
{
    using namespace H5Easy;
    const std::string search_cell = resolve_h5_cell_name(celltype);

    lavec = load<Eigen::Matrix3d>(file, "/" + search_cell + "/lattice_vector");
    lavec.transposeInPlace();
    // Convert the lattice into the internal unit (bohr) if the file declares
    // another unit; files without the attribute are assumed to be in bohr.
    lavec *= h5_length_factor_to_bohr(file, "/" + search_cell + "/lattice_vector", length_unit_detected);
    x_fractional = load<Eigen::MatrixXd>(file, "/" + search_cell + "/fractional_coordinate");
    kind_index = load<std::vector<int>>(file, "/" + search_cell + "/atomic_kinds");
    element_names = load<std::vector<std::string>>(file, "/" + search_cell + "/elements");
}

inline auto get_mapping_table_from_h5(const H5Easy::File &file, const std::string &celltype,
                                      std::vector<std::vector<int>> &mapping_table) -> void
{
    const std::string search_cell = resolve_h5_cell_name(celltype);
    mapping_table = H5Easy::load<std::vector<std::vector<int>>>(file, "/" + search_cell + "/mapping_table");
}

inline auto get_magnetism_from_h5(const H5Easy::File &file, const std::string &celltype, int &lspin,
                                  std::vector<std::vector<double>> &magmom, int &noncollinear,
                                  int &time_reversal_symmetry) -> void
{
    using namespace H5Easy;
    const std::string search_cell = resolve_h5_cell_name(celltype);

    lspin = load<int>(file, "/" + search_cell + "/spin_polarized");
    if (lspin > 0) {
        magmom = load<std::vector<std::vector<double>>>(file, "/" + search_cell + "/magnetic_moments");
        noncollinear = loadAttribute<int>(file, "/" + search_cell + "/magnetic_moments", "noncollinear");
        time_reversal_symmetry =
            loadAttribute<int>(file, "/" + search_cell + "/magnetic_moments", "time_reversal_symmetry");
    } else {
        noncollinear = 0;
        time_reversal_symmetry = 1;
    }
}

// Read the force constants of the given order (order 0 = harmonic FC2).
// When temperature_index >= 0, the values are taken from row
// temperature_index of /ForceConstants/Order2_temperature_dependent
// (the renormalized FC2 written by an SCPH/QHA run) while the index and
// shift datasets are shared with the base /ForceConstants/Order2 group,
// which holds the same row set by construction.
inline auto get_force_constants_from_h5(const H5Easy::File &file, const int order, Eigen::MatrixXi &atom_indices,
                                        Eigen::MatrixXi &atom_indices_super, Eigen::MatrixXi &coord_indices,
                                        Eigen::MatrixXd &shift_vectors, Eigen::ArrayXd &fcs_values,
                                        std::string *shift_unit_detected = nullptr,
                                        std::string *fc_unit_detected = nullptr,
                                        const int temperature_index = -1) -> void
{
    using namespace H5Easy;
    const std::string str_ordername = "Order" + std::to_string(order + 2);
    atom_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices");
    atom_indices_super = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/atom_indices_supercell");
    coord_indices = load<Eigen::MatrixXi>(file, "/ForceConstants/" + str_ordername + "/coord_indices");
    shift_vectors = load<Eigen::MatrixXd>(file, "/ForceConstants/" + str_ordername + "/shift_vectors");

    std::string path_values = "/ForceConstants/" + str_ordername + "/force_constant_values";

    if (temperature_index >= 0) {
        if (order != 0) {
            std::cout << "Error: temperature-dependent force constants are available only for Order2 "
                      << "(harmonic FC2), not " << str_ordername << ".\n";
            exit(1);
        }
        path_values = "/ForceConstants/" + str_ordername + "_temperature_dependent/force_constant_values";
        if (!file.exist(path_values)) {
            std::cout << "Error: file " << file.getName() << " does not contain temperature-dependent force constants ("
                      << path_values << ").\n";
            exit(1);
        }
        const auto dset = file.getDataSet(path_values);
        const auto dims = dset.getDimensions();
        if (dims.size() != 2 || static_cast<size_t>(temperature_index) >= dims[0]) {
            std::cout << "Error: temperature index " << temperature_index << " is out of range for dataset "
                      << path_values << " in file " << file.getName() << ".\n";
            exit(1);
        }
        if (dims[1] != static_cast<size_t>(atom_indices.rows())) {
            std::cout << "Error: dataset " << path_values << " in file " << file.getName() << " has " << dims[1]
                      << " rows per temperature but the shared index datasets have " << atom_indices.rows()
                      << " rows.\n";
            exit(1);
        }
        fcs_values.resize(static_cast<Eigen::Index>(dims[1]));
        dset.select({static_cast<size_t>(temperature_index), 0}, {1, dims[1]}).read(fcs_values.data());
    } else {
        fcs_values = load<Eigen::ArrayXd>(file, path_values);
    }

    // Convert into the internal units (bohr, Ry/bohr^m) if the file declares
    // other units; datasets without the attribute are assumed to already be in
    // the internal units (files written before the attributes existed).
    std::string unit_shift, unit_fc;
    const auto factor_shift =
        h5_length_factor_to_bohr(file, "/ForceConstants/" + str_ordername + "/shift_vectors", &unit_shift);
    const auto factor_fc = h5_fc_factor_to_ry_bohr(file, path_values, order, &unit_fc);
    // Reject partially annotated files when a non-default unit is declared:
    // one dataset in a non-default unit while its sibling carries no unit
    // attribute is ambiguous, most likely a hand-edited or third-party file.
    if ((unit_shift.empty() && factor_fc != 1.0) || (unit_fc.empty() && factor_shift != 1.0)) {
        std::cout << "Error: /ForceConstants/" << str_ordername << " in file " << file.getName()
                  << " declares a non-default unit on one dataset while its sibling"
                  << " (shift_vectors / force_constant_values) has no unit attribute.\n";
        exit(1);
    }
    if (factor_shift != 1.0) shift_vectors *= factor_shift;
    if (factor_fc != 1.0) fcs_values *= factor_fc;
    if (shift_unit_detected) *shift_unit_detected = unit_shift;
    if (fc_unit_detected) *fc_unit_detected = unit_fc;
}
