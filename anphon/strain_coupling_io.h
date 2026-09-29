/*
 strain_coupling_io.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "strain_coupling_types.h"

// Reader of the strain-coupling container (HDF5 schema "alamode:strain_coupling",
// the file named by STRAINFILE) that replaces the loose text inputs of the
// SCPH/QHA cell relaxation:
//
//   /ReferenceCell/      the reference structure (mandatory)
//   /Elastic/            stress (3,3), soec (9,9), toec (9,9,9), all in GPa
//   /StrainForce/        modes, smag, weight, forces [n, natom, 3] in eV/Angstrom, Cell/
//   /StrainHarmonic/     modes, smag, weight, entry_NNN/ (alm force-constant layout)
//   /Piezoelectric/      clamped_ion (3,3,3): proper clamped-ion tensor e_ijk in C/m^2 (EFIELD)
//
// The class returns the plain structs of strain_coupling_types.h, so the
// consumers do not depend on HDF5. Every method throws std::runtime_error with
// a message that names the offending dataset; the callers turn that into
// exit(). No HDF5 type appears in this header.

namespace PHON_NS
{
class Fcs_phonon;
class FcsArrayWithCell;

namespace strain_coupling
{
struct ContainerSummary
{
    int format_version{0};
    bool has_elastic{false};
    bool has_stress{false};
    bool has_c2c3{false};
    bool has_strain_force{false};
    bool has_strain_harmonic{false};
    bool has_piezo{false};
    std::size_t natom_reference{0};
    std::string created_date;
    std::string writer;
};

class StrainCouplingFile
{
public:
    // Opens the file read-only; throws when it cannot be opened as HDF5.
    explicit StrainCouplingFile(std::string filename);
    ~StrainCouplingFile();

    StrainCouplingFile(const StrainCouplingFile &) = delete;
    StrainCouplingFile &operator=(const StrainCouplingFile &) = delete;

    [[nodiscard]] const std::string &filename() const;

    // Schema and version check plus the inventory of the groups.
    [[nodiscard]] ContainerSummary probe() const;

    [[nodiscard]] strain_parsers::ReferenceCell read_reference_cell() const;

    // /Elastic; has_stress / has_c2c3 tell which datasets were present.
    [[nodiscard]] ElasticSet read_elastic() const;

    // /StrainForce with its Cell (the rows are mapped onto the current cell by the caller).
    [[nodiscard]] StrainForceSet read_strain_force() const;

    // /StrainHarmonic metadata and the SuperCell of every entry; the force
    // constants are read separately with load_harmonic_fc2.
    [[nodiscard]] StrainHarmonicSet read_strain_harmonic() const;

    // /Piezoelectric/clamped_ion: e_ijk [C/m^2], flat index (i * 3 + j) * 3 + k;
    // checks the unit, the convention attribute, the shape, finiteness and the
    // symmetry in jk.
    [[nodiscard]] std::array<double, 27> read_piezo() const;

    // Read the harmonic force constants of one entry (entry.label is its group).
    void load_harmonic_fc2(const StrainHarmonicEntry &entry, const Fcs_phonon &fcs_phonon,
                           std::vector<FcsArrayWithCell> &fc2_out) const;

    // The tool command that produces a missing group.
    static std::string missing_group_hint(const std::string &group);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace strain_coupling
} // namespace PHON_NS
