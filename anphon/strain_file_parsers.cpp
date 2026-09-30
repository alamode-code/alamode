/*
 strain_file_parsers.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "strain_file_parsers.h"
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include "strain_reference_cell.h"

namespace PHON_NS
{
namespace strain_parsers
{
namespace
{
std::string lowercase(std::string s)
{
    for (auto &c: s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::runtime_error short_read(const char *filename, const char *what, const std::size_t nvalues)
{
    return std::runtime_error(std::string(filename) + " ended before all " + std::to_string(nvalues) +
                              " values of the " + what + " section were read.");
}
} // namespace

const char *unit_name(const ElasticUnit unit)
{
    switch (unit) {
    case ElasticUnit::GPa:
        return "GPa";
    case ElasticUnit::Ry:
        return "Ry";
    default:
        return "legacy (V*C in Ry, no unit token)";
    }
}

bool parse_double_strict(const std::string &token, double &value)
{
    if (token.empty()) return false;
    const char *begin = token.c_str();
    char *end = nullptr;
    errno = 0;
    const double v = std::strtod(begin, &end);
    if (end != begin + token.size() || errno == ERANGE) return false;
    value = v;
    return true;
}

ElasticUnit read_section(std::istream &fin, const char *filename, const char *what, const std::size_t nvalues,
                         double *values)
{
    std::string tok;

    // The section label. Its spelling is deliberately not checked: the readers
    // never did, and rejecting it now would break files that work today.
    if (!(fin >> tok)) {
        throw std::runtime_error(std::string(filename) + " ended before the " + what + " section.");
    }
    if (!(fin >> tok)) throw short_read(filename, what, nvalues);

    auto unit = ElasticUnit::Legacy;
    std::size_t i = 0;
    double v;
    if (parse_double_strict(tok, v)) {
        // Legacy file: the token after the label is already the first value.
        values[i++] = v;
    } else {
        const auto low = lowercase(tok);
        if (low == "gpa") {
            unit = ElasticUnit::GPa;
        } else if (low == "ry") {
            unit = ElasticUnit::Ry;
        } else {
            throw std::runtime_error("Unknown unit '" + tok + "' after the " + what + " label of " + filename +
                                     ".\n Accepted units: GPa (cell-independent) and Ry (V*C per cell).");
        }
    }
    for (; i < nvalues; ++i) {
        if (!(fin >> values[i])) throw short_read(filename, what, nvalues);
    }
    return unit;
}

C1Data parse_c1_array(std::istream &fin)
{
    C1Data data;
    data.unit = read_section(fin, "C1_array.in", "C1", 9, data.values.data());
    return data;
}

ElasticData parse_elastic_constants(std::istream &fin)
{
    ElasticData data;
    data.c2.assign(81, 0.0);
    data.c3.assign(729, 0.0);
    data.unit_soec = read_section(fin, "elastic_constants.in", "SOEC", 81, data.c2.data());
    data.unit_toec = read_section(fin, "elastic_constants.in", "TOEC", 729, data.c3.data());
    return data;
}

bool has_trailing_data(std::istream &fin)
{
    std::string tok;
    return static_cast<bool>(fin >> tok);
}

bool is_strain_mode_name(const std::string &mode)
{
    return mode == "xx" || mode == "yy" || mode == "zz" || mode == "xy" || mode == "yz" || mode == "zx";
}

strain_coupling::StrainForceSet parse_strain_force(std::istream &fin, const std::size_t natom_default,
                                                   const char *filename)
{
    strain_coupling::StrainForceSet set;
    set.origin = filename;
    const std::string fname(filename);

    // Optional header. The first token is carried into the block loop, so no
    // seeking is needed.
    std::string first_token;
    const bool has_first = static_cast<bool>(fin >> first_token);
    set.has_cell = has_first && is_reference_cell_tag(first_token);
    set.natom_rows = natom_default;
    if (set.has_cell) {
        set.cell = parse_reference_cell(fin, filename);
        set.natom_rows = set.cell.natom();
    }

    bool pending_first = has_first && !set.has_cell;
    while (true) {
        strain_coupling::StrainForceBlock block;
        bool ok;
        if (pending_first) {
            pending_first = false;
            block.mode = first_token;
            ok = static_cast<bool>(fin >> block.smag >> block.weight);
        } else {
            ok = static_cast<bool>(fin >> block.mode >> block.smag >> block.weight);
        }
        if (!ok) break;
        if (!is_strain_mode_name(block.mode)) {
            throw std::runtime_error("Invalid name of strain mode in " + fname + ".");
        }
        if (!std::isfinite(block.smag) || block.smag == 0.0 || !std::isfinite(block.weight)) {
            throw std::runtime_error(fname + ": block " + std::to_string(set.blocks.size() + 1) +
                                     " has an invalid smag or weight.");
        }
        block.forces.resize(set.natom_rows * 3);
        for (auto &v: block.forces) {
            if (!(fin >> v)) {
                throw std::runtime_error(
                    fname + " ended in the middle of a block. Every block must consist of a\n"
                            " 'mode smag weight' line followed by one line of three force components per atom\n"
                            " of the cell the file describes.");
            }
        }
        set.blocks.push_back(std::move(block));
    }
    // The loop ends at the end of the stream, or at a token that does not
    // start a valid 'mode smag weight' line (a partial line ending exactly at
    // the end of the stream is not distinguished).
    set.trailing_data = !fin.eof();
    return set;
}

strain_coupling::StrainHarmonicSet parse_strain_harmonic(std::istream &fin, const char *filename)
{
    strain_coupling::StrainHarmonicSet set;
    set.origin = filename;
    const std::string fname(filename);
    strain_coupling::StrainHarmonicEntry entry;
    while (fin >> entry.mode >> entry.smag >> entry.weight >> entry.label) {
        if (!is_strain_mode_name(entry.mode)) {
            throw std::runtime_error("Invalid name of strain mode in " + fname + ".");
        }
        if (!std::isfinite(entry.smag) || entry.smag == 0.0 || !std::isfinite(entry.weight)) {
            throw std::runtime_error(fname + ": entry " + std::to_string(set.entries.size() + 1) +
                                     " has an invalid smag or weight.");
        }
        set.entries.push_back(entry);
    }
    set.trailing_data = !fin.eof();
    return set;
}
} // namespace strain_parsers
} // namespace PHON_NS
