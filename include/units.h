/*
 units.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "constants.h"

// Helpers for the user-selectable input/output units of ALM and ANPHON.
// The internal canonical units are always Rydberg atomic units:
// lengths in bohr, forces in Ry/bohr, force constants in Ry/bohr^n.
namespace units
{

enum class LengthUnit
{
    bohr,
    angstrom
};
enum class ForceUnit
{
    ry_per_bohr,
    ev_per_angstrom,
    ha_per_bohr
};
enum class FcUnitSystem
{
    ry_bohr,
    ev_angstrom
};

inline auto to_lower_copy(std::string s) -> std::string
{
    std::transform(s.begin(), s.end(), s.begin(), [](const unsigned char c) { return std::tolower(c); });
    return s;
}

inline auto parse_length_unit(const std::string &name) -> LengthUnit
{
    const auto s = to_lower_copy(name);
    if (s == "bohr") return LengthUnit::bohr;
    if (s == "angstrom") return LengthUnit::angstrom;
    throw std::invalid_argument("Invalid length unit '" + name + "'. Valid options are: bohr, angstrom.");
}

inline auto parse_force_unit(const std::string &name) -> ForceUnit
{
    const auto s = to_lower_copy(name);
    if (s == "ry/bohr") return ForceUnit::ry_per_bohr;
    if (s == "ev/angstrom") return ForceUnit::ev_per_angstrom;
    if (s == "ha/bohr") return ForceUnit::ha_per_bohr;
    throw std::invalid_argument("Invalid force unit '" + name + "'. Valid options are: Ry/bohr, eV/angstrom, Ha/bohr.");
}

inline auto parse_fc_unit_system(const std::string &name) -> FcUnitSystem
{
    const auto s = to_lower_copy(name);
    if (s == "ry/bohr") return FcUnitSystem::ry_bohr;
    if (s == "ev/angstrom") return FcUnitSystem::ev_angstrom;
    throw std::invalid_argument("Invalid force constant unit system '" + name +
                                "'. Valid options are: Ry/bohr, eV/angstrom.");
}

// Multiply-by factors into the canonical units.
inline auto length_to_bohr(const LengthUnit u) -> double
{
    return u == LengthUnit::angstrom ? 1.0 / Bohr_in_Angstrom : 1.0;
}

inline auto force_to_ry_bohr(const ForceUnit u) -> double
{
    switch (u) {
    case ForceUnit::ev_per_angstrom:
        return Bohr_in_Angstrom / Ryd_in_eV;
    case ForceUnit::ha_per_bohr:
        return 2.0;
    default:
        return 1.0;
    }
}

// Factor from Ry/bohr^m to eV/angstrom^m for a force constant with m = order + 2.
inline auto fc_ry_bohr_to_ev_ang(const int m) -> double
{
    return Ryd_in_eV / std::pow(Bohr_in_Angstrom, m);
}

inline auto canonical_name(const LengthUnit u) -> std::string
{
    return u == LengthUnit::angstrom ? "angstrom" : "bohr";
}

inline auto canonical_name(const ForceUnit u) -> std::string
{
    switch (u) {
    case ForceUnit::ev_per_angstrom:
        return "eV/angstrom";
    case ForceUnit::ha_per_bohr:
        return "Ha/bohr";
    default:
        return "Ry/bohr";
    }
}

inline auto canonical_name(const FcUnitSystem u) -> std::string
{
    return u == FcUnitSystem::ev_angstrom ? "eV/angstrom" : "Ry/bohr";
}

} // namespace units
