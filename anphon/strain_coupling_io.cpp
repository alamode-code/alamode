/*
 strain_coupling_io.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "strain_coupling_io.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include "constants.h"
#include "fcs_phonon.h"
#include "hdf5_parser.h"
#include "strain_file_parsers.h"

using namespace PHON_NS;
using namespace PHON_NS::strain_coupling;

namespace
{
std::string lowercase(std::string s)
{
    for (auto &c: s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string entry_group(const std::size_t k)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "/StrainHarmonic/entry_%03zu", k);
    return buf;
}

void require(const HighFive::File &file, const std::string &path, const char *what)
{
    if (!file.exist(path)) {
        throw std::runtime_error(file.getName() + " has no " + path + " (" + what + ").");
    }
}

std::vector<std::size_t> dims_of(const HighFive::File &file, const std::string &path)
{
    return file.getDataSet(path).getDimensions();
}

void require_dims(const HighFive::File &file, const std::string &path, const std::vector<std::size_t> &expected)
{
    const auto dims = dims_of(file, path);
    if (dims != expected) {
        std::string want, have;
        for (const auto d: expected) want += (want.empty() ? "" : ",") + std::to_string(d);
        for (const auto d: dims) have += (have.empty() ? "" : ",") + std::to_string(d);
        throw std::runtime_error(path + " of " + file.getName() + " has shape (" + have + "), expected (" + want +
                                 ").");
    }
}

// The "unit" attribute of a dataset, "" when absent.
std::string unit_of(const HighFive::File &file, const std::string &path)
{
    const auto dset = file.getDataSet(path);
    if (!dset.hasAttribute("unit")) return "";
    std::string unit;
    dset.getAttribute("unit").read(unit);
    return unit;
}

void require_unit(const HighFive::File &file, const std::string &path, const char *expected)
{
    const auto unit = unit_of(file, path);
    if (unit.empty()) {
        throw std::runtime_error(path + " of " + file.getName() + " has no 'unit' attribute (expected \"" + expected +
                                 "\").");
    }
    if (lowercase(unit) != lowercase(expected)) {
        throw std::runtime_error(path + " of " + file.getName() + " is in \"" + unit + "\"; only \"" + expected +
                                 "\" is accepted.");
    }
}

// Read a contiguous double dataset of the given shape.
std::vector<double> load_doubles(const HighFive::File &file, const std::string &path,
                                 const std::vector<std::size_t> &expected)
{
    require_dims(file, path, expected);
    std::size_t n = 1;
    for (const auto d: expected) n *= d;
    std::vector<double> values(n);
    if (n > 0) file.getDataSet(path).read(values.data());
    for (const auto v: values) {
        if (!std::isfinite(v))
            throw std::runtime_error(path + " of " + file.getName() + " contains non-finite values.");
    }
    return values;
}

// The "convention" attribute of a dataset must be `expected`.
void require_convention(const HighFive::File &file, const std::string &path, const char *expected)
{
    const auto dset = file.getDataSet(path);
    std::string convention;
    if (dset.hasAttribute("convention")) dset.getAttribute("convention").read(convention);
    if (convention != expected) {
        throw std::runtime_error(path + " of " + file.getName() + " has the convention \"" + convention +
                                 "\"; only \"" + expected + "\" is accepted.");
    }
}

// values[a] == values[partner(a)] for every flat index a, within 1e-8 + 1e-6 max |values|.
template <typename Partner>
void require_symmetric(const HighFive::File &file, const std::string &path, const std::vector<double> &values,
                       Partner partner, const char *what)
{
    double scale = 0.0;
    for (const auto v: values) scale = std::max(scale, std::fabs(v));
    const auto tol = 1.0e-8 + 1.0e-6 * scale;
    double dev = 0.0;
    for (std::size_t a = 0; a < values.size(); ++a) dev = std::max(dev, std::fabs(values[a] - values[partner(a)]));
    if (dev > tol) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), " (largest deviation %.3e, tolerance %.3e).", dev, tol);
        throw std::runtime_error(path + " of " + file.getName() + " is not symmetric " + what + buf);
    }
}

// A cell group in the layout of the alm force-constant files (lattice rows
// in the file, columns in memory).
strain_parsers::ReferenceCell read_cell_group(const HighFive::File &file, const std::string &group)
{
    require(file, group, "cell group");
    for (const char *name: {"lattice_vector", "fractional_coordinate", "atomic_kinds", "elements"}) {
        require(file, group + "/" + name, "dataset of the cell group");
    }
    const auto unit = lowercase(unit_of(file, group + "/lattice_vector"));
    if (!(unit.empty() || unit == "bohr" || unit == "angstrom")) {
        throw std::runtime_error(group + "/lattice_vector of " + file.getName() + " is in \"" + unit +
                                 "\"; bohr or angstrom expected.");
    }
    require_dims(file, group + "/lattice_vector", {3, 3});
    Eigen::Matrix3d lattice = H5Easy::load<Eigen::Matrix3d>(file, group + "/lattice_vector");
    lattice.transposeInPlace();
    lattice *= h5_length_factor_to_bohr(file, group + "/lattice_vector");
    if (!lattice.allFinite() || std::fabs(lattice.determinant()) < 1.0e-8) {
        throw std::runtime_error(group + "/lattice_vector of " + file.getName() + " is singular or not finite.");
    }

    const auto xf = H5Easy::load<Eigen::MatrixXd>(file, group + "/fractional_coordinate");
    const auto kinds = H5Easy::load<std::vector<int>>(file, group + "/atomic_kinds");
    const auto elements = H5Easy::load<std::vector<std::string>>(file, group + "/elements");
    const auto natom = static_cast<std::size_t>(xf.rows());
    if (natom == 0 || xf.cols() != 3 || !xf.allFinite()) {
        throw std::runtime_error(group + "/fractional_coordinate of " + file.getName() +
                                 " must be a finite (natom, 3) array with natom > 0.");
    }
    if (kinds.size() != natom) {
        throw std::runtime_error(group + "/atomic_kinds of " + file.getName() +
                                 " does not have one entry per atom of fractional_coordinate.");
    }
    strain_parsers::ReferenceCell cell;
    cell.lattice = lattice;
    cell.x_fractional = xf;
    cell.symbols.resize(natom);
    for (std::size_t i = 0; i < natom; ++i) {
        if (kinds[i] < 0 || static_cast<std::size_t>(kinds[i]) >= elements.size()) {
            throw std::runtime_error(group + "/atomic_kinds of " + file.getName() + " refers to element index " +
                                     std::to_string(kinds[i]) + " but only " + std::to_string(elements.size()) +
                                     " elements are listed.");
        }
        cell.symbols[i] = elements[kinds[i]];
    }
    return cell;
}

// modes / smag / weight arrays of /StrainForce or /StrainHarmonic.
struct ModeTable
{
    std::vector<std::string> modes;
    std::vector<double> smag, weight;
};

ModeTable read_mode_table(const HighFive::File &file, const std::string &group)
{
    for (const char *name: {"modes", "smag", "weight"}) require(file, group + "/" + name, "strain-mode table");
    ModeTable t;
    t.modes = H5Easy::load<std::vector<std::string>>(file, group + "/modes");
    t.smag = H5Easy::load<std::vector<double>>(file, group + "/smag");
    t.weight = H5Easy::load<std::vector<double>>(file, group + "/weight");
    if (t.modes.empty()) throw std::runtime_error(group + " of " + file.getName() + " lists no strain modes.");
    if (t.smag.size() != t.modes.size() || t.weight.size() != t.modes.size()) {
        throw std::runtime_error(group + " of " + file.getName() +
                                 ": modes, smag and weight must have the same length.");
    }
    for (std::size_t k = 0; k < t.modes.size(); ++k) {
        if (!strain_parsers::is_strain_mode_name(t.modes[k])) {
            throw std::runtime_error(group + "/modes of " + file.getName() + ": invalid strain mode \"" + t.modes[k] +
                                     "\" (xx, yy, zz, yz, zx, xy).");
        }
        if (!std::isfinite(t.smag[k]) || t.smag[k] == 0.0 || !std::isfinite(t.weight[k])) {
            throw std::runtime_error(group + " of " + file.getName() + ": entry " + std::to_string(k + 1) +
                                     " has an invalid smag or weight.");
        }
    }
    // The optional displacement_gradient dataset must agree with mode/smag.
    const auto path = group + "/displacement_gradient";
    if (file.exist(path)) {
        const auto u = load_doubles(file, path, {t.modes.size(), 3, 3});
        for (std::size_t k = 0; k < t.modes.size(); ++k) {
            const Eigen::Matrix3d expected = strain_parsers::displacement_gradient(t.modes[k], t.smag[k]);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if (std::fabs(u[(k * 3 + i) * 3 + j] - expected(i, j)) > 1.0e-10) {
                        throw std::runtime_error(path + " of " + file.getName() + ": entry " + std::to_string(k + 1) +
                                                 " does not match its mode and magnitude.");
                    }
                }
            }
        }
    }
    return t;
}
} // namespace

struct StrainCouplingFile::Impl
{
    std::string filename;
    std::unique_ptr<HighFive::File> file;
};

StrainCouplingFile::StrainCouplingFile(std::string filename) : impl(std::make_unique<Impl>())
{
    impl->filename = std::move(filename);
    try {
        impl->file = std::make_unique<HighFive::File>(impl->filename, HighFive::File::ReadOnly);
    } catch (const HighFive::Exception &e) {
        throw std::runtime_error("STRAINFILE " + impl->filename + " cannot be opened as an HDF5 file.\n " + e.what());
    }
}

StrainCouplingFile::~StrainCouplingFile() = default;

const std::string &StrainCouplingFile::filename() const
{
    return impl->filename;
}

ContainerSummary StrainCouplingFile::probe() const
{
    const auto &file = *impl->file;
    ContainerSummary s;

    if (!file.hasAttribute("schema")) {
        throw std::runtime_error(impl->filename + " is not a strain-coupling container: it has no 'schema' attribute" +
                                 " (expected \"" + h5_schema_strain_coupling + "\").");
    }
    std::string schema;
    file.getAttribute("schema").read(schema);
    if (schema != h5_schema_strain_coupling) {
        throw std::runtime_error(impl->filename + " has schema \"" + schema + "\" but STRAINFILE needs \"" +
                                 h5_schema_strain_coupling + "\".");
    }
    if (!file.hasAttribute("format_version")) {
        throw std::runtime_error(impl->filename + " has no 'format_version' attribute.");
    }
    file.getAttribute("format_version").read(s.format_version);
    if (s.format_version > h5_version_strain_coupling) {
        throw std::runtime_error(impl->filename + " uses format_version " + std::to_string(s.format_version) +
                                 " of the strain-coupling schema, but this build supports up to version " +
                                 std::to_string(h5_version_strain_coupling) + ". Please use a newer ALAMODE.");
    }
    if (file.hasAttribute("created_date")) file.getAttribute("created_date").read(s.created_date);
    if (file.hasAttribute("strainkit_version")) file.getAttribute("strainkit_version").read(s.writer);

    require(file, "/ReferenceCell", "the reference structure");
    require(file, "/ReferenceCell/number_of_atoms", "atom count of the reference structure");
    s.natom_reference = H5Easy::load<std::size_t>(file, "/ReferenceCell/number_of_atoms");

    s.has_elastic = file.exist("/Elastic");
    s.has_stress = s.has_elastic && file.exist("/Elastic/stress");
    s.has_c2c3 = s.has_elastic && file.exist("/Elastic/soec") && file.exist("/Elastic/toec");
    if (s.has_elastic && (file.exist("/Elastic/soec") != file.exist("/Elastic/toec"))) {
        throw std::runtime_error(impl->filename + ": /Elastic must contain both soec and toec or neither.");
    }
    s.has_strain_force = file.exist("/StrainForce");
    s.has_strain_harmonic = file.exist("/StrainHarmonic");
    s.has_piezo = file.exist("/Piezoelectric");
    s.has_piezo_clamped_ion = s.has_piezo && file.exist("/Piezoelectric/clamped_ion");
    s.has_piezo_second_order = s.has_piezo && file.exist("/Piezoelectric/second_order");
    s.has_born_charge_strain_derivative = s.has_piezo && file.exist("/Piezoelectric/born_charge_strain_derivative");
    return s;
}

strain_parsers::ReferenceCell StrainCouplingFile::read_reference_cell() const
{
    return read_cell_group(*impl->file, "/ReferenceCell");
}

ElasticSet StrainCouplingFile::read_elastic() const
{
    const auto &file = *impl->file;
    if (!file.exist("/Elastic")) {
        throw std::runtime_error(impl->filename + " has no /Elastic group.\n " + missing_group_hint("/Elastic"));
    }
    ElasticSet set;
    set.source = impl->filename + ":/Elastic";
    if (file.exist("/Elastic/stress")) {
        require_unit(file, "/Elastic/stress", "GPa");
        const auto v = load_doubles(file, "/Elastic/stress", {3, 3});
        for (std::size_t i = 0; i < 9; ++i) set.stress_gpa[i] = v[i];
        set.has_stress = true;
    }
    if (file.exist("/Elastic/soec") || file.exist("/Elastic/toec")) {
        require(file, "/Elastic/soec", "second-order elastic constants");
        require(file, "/Elastic/toec", "third-order elastic constants");
        require_unit(file, "/Elastic/soec", "GPa");
        require_unit(file, "/Elastic/toec", "GPa");
        set.soec_gpa = load_doubles(file, "/Elastic/soec", {9, 9});
        set.toec_gpa = load_doubles(file, "/Elastic/toec", {9, 9, 9});
        set.has_c2c3 = true;
    }
    return set;
}

StrainForceSet StrainCouplingFile::read_strain_force() const
{
    const auto &file = *impl->file;
    if (!file.exist("/StrainForce")) {
        throw std::runtime_error(impl->filename + " has no /StrainForce group.\n " +
                                 missing_group_hint("/StrainForce"));
    }
    const auto table = read_mode_table(file, "/StrainForce");
    const auto n = table.modes.size();

    StrainForceSet set;
    set.origin = impl->filename + ":/StrainForce";
    set.cell_description = "The force rows of " + set.origin + " follow the atoms of its Cell group";
    set.cell = read_cell_group(file, "/StrainForce/Cell");
    set.has_cell = true;
    set.natom_rows = set.cell.natom();

    require(file, "/StrainForce/forces", "strain-force blocks");
    require_unit(file, "/StrainForce/forces", "eV/angstrom");
    const auto forces = load_doubles(file, "/StrainForce/forces", {n, set.natom_rows, 3});

    set.blocks.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        auto &block = set.blocks[k];
        block.mode = table.modes[k];
        block.smag = table.smag[k];
        block.weight = table.weight[k];
        block.forces.assign(forces.begin() + static_cast<std::ptrdiff_t>(k * set.natom_rows * 3),
                            forces.begin() + static_cast<std::ptrdiff_t>((k + 1) * set.natom_rows * 3));
    }
    return set;
}

StrainHarmonicSet StrainCouplingFile::read_strain_harmonic() const
{
    const auto &file = *impl->file;
    if (!file.exist("/StrainHarmonic")) {
        throw std::runtime_error(impl->filename + " has no /StrainHarmonic group.\n " +
                                 missing_group_hint("/StrainHarmonic"));
    }
    const auto table = read_mode_table(file, "/StrainHarmonic");
    const auto n = table.modes.size();

    StrainHarmonicSet set;
    set.origin = impl->filename + ":/StrainHarmonic";
    set.entries.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        auto &entry = set.entries[k];
        const auto group = entry_group(k + 1);
        require(file, group, "strained-supercell entry listed in /StrainHarmonic/modes");
        require(file, group + "/ForceConstants/Order2", "harmonic force constants of the entry");
        entry.mode = table.modes[k];
        entry.smag = table.smag[k];
        entry.weight = table.weight[k];
        entry.label = group;
        entry.supercell = read_cell_group(file, group + "/SuperCell");
        entry.has_cell = true;

        // The per-entry attributes are redundant copies of the table and must agree with it.
        const auto g = file.getGroup(group);
        if (g.hasAttribute("mode")) {
            std::string mode;
            g.getAttribute("mode").read(mode);
            if (mode != entry.mode) {
                throw std::runtime_error(group + " of " + impl->filename + " is labeled mode \"" + mode +
                                         "\" but /StrainHarmonic/modes lists \"" + entry.mode + "\".");
            }
        }
        for (const char *name: {"smag", "weight"}) {
            if (!g.hasAttribute(name)) continue;
            double v = 0.0;
            g.getAttribute(name).read(v);
            const auto ref = std::string(name) == "smag" ? entry.smag : entry.weight;
            if (std::fabs(v - ref) > 1.0e-12 * std::max(1.0, std::fabs(ref))) {
                throw std::runtime_error(group + " of " + impl->filename + ": the attribute " + name +
                                         " disagrees with the /StrainHarmonic table.");
            }
        }
    }
    // Extra entry groups that are not listed in the table are refused: they
    // would silently be ignored otherwise.
    const auto extra = entry_group(n + 1);
    if (file.exist(extra)) {
        throw std::runtime_error(impl->filename + " contains " + extra +
                                 " which is not listed in /StrainHarmonic/modes.");
    }
    return set;
}

std::array<double, 27> StrainCouplingFile::read_piezo() const
{
    const auto &file = *impl->file;
    const std::string path = "/Piezoelectric/clamped_ion";
    require(file, path, "clamped-ion piezoelectric tensor");
    require_unit(file, path, "C/m^2");
    require_convention(file, path, "proper, clamped-ion");
    const auto v = load_doubles(file, path, {3, 3, 3});
    std::array<double, 27> e{};
    for (std::size_t i = 0; i < 27; ++i) e[i] = v[i];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = j + 1; k < 3; ++k) {
                if (std::fabs(e[(i * 3 + j) * 3 + k] - e[(i * 3 + k) * 3 + j]) > 1.0e-8) {
                    throw std::runtime_error(path + " of " + file.getName() +
                                             " is not symmetric in its last two indices.");
                }
            }
        }
    }
    return e;
}

std::array<double, 243> StrainCouplingFile::read_piezo_second_order() const
{
    const auto &file = *impl->file;
    const std::string path = "/Piezoelectric/second_order";
    require(file, path, "second-order clamped-ion piezoelectric tensor");
    require_unit(file, path, "C/m^2");
    require_convention(file, path, "proper, clamped-ion, per linear strain u");
    const auto v = load_doubles(file, path, {3, 3, 3, 3, 3});
    // flat index ((i * 3 + j) * 3 + k) * 9 + l * 3 + m = i * 81 + jk * 9 + lm
    const auto swap_pair = [](const std::size_t p) { return (p % 3) * 3 + p / 3; };
    require_symmetric(
        file,
        path,
        v,
        [&](const std::size_t a) { return a / 81 * 81 + swap_pair(a / 9 % 9) * 9 + a % 9; },
        "in its indices jk");
    require_symmetric(
        file,
        path,
        v,
        [&](const std::size_t a) { return a / 9 * 9 + swap_pair(a % 9); },
        "in its indices lm");
    require_symmetric(
        file,
        path,
        v,
        [](const std::size_t a) { return a / 81 * 81 + a % 9 * 9 + a / 9 % 9; },
        "under the exchange of the index pairs jk and lm");
    std::array<double, 243> b{};
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = v[i];
    return b;
}

std::vector<double> StrainCouplingFile::read_born_charge_strain_derivative(const std::size_t natom_reference) const
{
    const auto &file = *impl->file;
    const std::string path = "/Piezoelectric/born_charge_strain_derivative";
    require(file, path, "strain derivative of the reduced Born effective charges");
    require_unit(file, path, "e");
    require_convention(file, path, "reduced (F^-1 Z*), per linear strain u");
    const auto v = load_doubles(file, path, {natom_reference, 3, 3, 3, 3});
    require_symmetric(
        file,
        path,
        v,
        [](const std::size_t a) { return a / 9 * 9 + (a % 3) * 3 + a % 9 / 3; },
        "in its strain indices mn");
    return v;
}

void StrainCouplingFile::load_harmonic_fc2(const StrainHarmonicEntry &entry, const Fcs_phonon &fcs_phonon,
                                           std::vector<FcsArrayWithCell> &fc2_out) const
{
    fcs_phonon.parse_fcs_from_h5(*impl->file, entry.label, 0, fc2_out);
}

std::string StrainCouplingFile::missing_group_hint(const std::string &group)
{
    if (group == "/Elastic") {
        return "Add it with: elastic.py fit --strain-file FILE (or strainfile.py pack for existing text files).";
    }
    if (group == "/StrainForce") {
        return "Add it with: elastic.py fit --strain-file FILE, strainifc.py collect --coupling force --strain-file FILE, or strainfile.py pack.";
    }
    if (group == "/StrainHarmonic") {
        return "Add it with: strainifc.py collect --coupling harmonic --strain-file FILE (or strainfile.py pack).";
    }
    if (group == "/Piezoelectric") {
        return "Add it with: strainfile.py piezo FILE --outcar OUTCAR (VASP LEPSILON) or --voigt TABLE --structure POSCAR.";
    }
    return "See the strain-coupling tools (tools/strainfile.py) to add it.";
}
