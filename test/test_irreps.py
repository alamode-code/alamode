#!/usr/bin/env python

"""Integration test for the IRREPS = 1 analysis of anphon (MODE = phonons).

Runs the Gamma-point irreducible-representation analysis and checks the
group-theoretical output, which is exact (labels, degeneracies, activities),
unlike frequencies:

  * Si: Oh / Fd-3m, Gamma_optic = T2g (Raman-only), acoustic T1u.  The
    harmonic force constants are generated with alm from the committed
    example/Si/reference/DFSET_harmonic, so this case is self-contained.
  * cubic BaTiO3: Oh / Pm-3m, Gamma_optic = 3T1u + T2u with an imaginary
    (soft) T1u and a silent T2u.  Uses the same
    example/BaTiO3/scph_relax/reference/cBTO222.h5 fixture as
    test_batio3.py and fails with a message when it is absent.
  * wurtzite ZnO (optional): C6v / P6_3mc, Gamma_optic = A1 + 2B1 + E1 +
    2E2 with silent B1 modes -- the convention-sensitive hexagonal check.
    Skipped when example/ZnO/qha_relax/ZnO442_harmonic.xml is absent.
"""

import os
import shutil
import subprocess
import sys


def gen_anphon_input(fname, prefix, fcsfile, cell=None):
    lines = [
        "&general",
        "  PREFIX = %s" % prefix,
        "  MODE = phonons",
        "  FCSFILE = %s" % fcsfile,
        "/",
    ]
    if cell is not None:
        lines += ["&cell"] + ["  %s" % row for row in cell] + ["/"]
    lines += [
        "&kpoint",
        "  1",
        "  G 0.0 0.0 0.0 X 0.5 0.5 0.0 21",
        "/",
        "&analysis",
        "  IRREPS = 1",
        "/",
        "",
    ]
    with open(fname, "w") as f:
        f.write("\n".join(lines))


def parse_irreps_file(fname):
    """Parse PREFIX.irreps into (header dict, list of multiplet dicts)."""
    header = {}
    groups = []
    in_table = False
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("multiplet, first & last branch"):
                    in_table = True
                elif body.startswith("Characters") or body.startswith("IR oscillator"):
                    in_table = False
                if body.startswith("Point group:"):
                    header["Point group"] = body.split(":", 1)[-1].strip()
                for key in ("Gamma_total", "Gamma_acoustic", "Gamma_optic"):
                    if body.startswith(key) and "=" in body:
                        header[key] = body.split("=", 1)[-1].strip()
                continue
            if not in_table:
                continue
            entries = line.split()
            if len(entries) < 7:
                continue
            groups.append(
                {
                    "first": int(entries[1]),
                    "last": int(entries[2]),
                    "freq": float(entries[3]),
                    "irrep": entries[4],
                    "deg": int(entries[5]),
                    "activity": entries[6],
                }
            )
    return header, groups


def check(cond, message):
    if not cond:
        print("FAIL: %s" % message)
        return 1
    return 0


def run_case(anphonbin, prefix, fcsfile, expects, cell=None):
    infile = "%s_irreps.in" % prefix
    gen_anphon_input(infile, prefix, fcsfile, cell=cell)
    with open("%s_irreps.log" % prefix, "w") as f:
        ret = subprocess.run([anphonbin, infile], stdout=f)
    if ret.returncode != 0:
        print("FAIL: anphon exited with %d for %s" % (ret.returncode, prefix))
        return 1

    header, groups = parse_irreps_file("%s.irreps" % prefix)

    nfail = 0
    nfail += check(
        header.get("Point group", "").startswith(expects["point_group"]),
        "%s point group: expected %s, got '%s'"
        % (prefix, expects["point_group"], header.get("Point group")),
    )
    for key in ("Gamma_total", "Gamma_acoustic", "Gamma_optic"):
        nfail += check(
            header.get(key) == expects[key],
            "%s %s: expected '%s', got '%s'"
            % (prefix, key, expects[key], header.get(key)),
        )
    nfail += check(
        len(groups) == len(expects["groups"]),
        "%s: expected %d multiplets, got %d"
        % (prefix, len(expects["groups"]), len(groups)),
    )
    for ref, got in zip(expects["groups"], groups):
        irrep, deg, activity, freq_sign = ref
        nfail += check(
            got["irrep"] == irrep and got["deg"] == deg and got["activity"] == activity,
            "%s multiplet: expected (%s, %d, %s), got (%s, %d, %s)"
            % (prefix, irrep, deg, activity, got["irrep"], got["deg"], got["activity"]),
        )
        if freq_sign != 0:
            nfail += check(
                got["freq"] * freq_sign > 0.0,
                "%s multiplet %s: expected frequency sign %+d, got %f"
                % (prefix, irrep, freq_sign, got["freq"]),
            )
    if nfail == 0:
        print(
            "%s: OK (%s, Gamma_optic = %s)"
            % (prefix, expects["point_group"], expects["Gamma_optic"])
        )
    return nfail


def ensure_si_fixture(almbin, project_root):
    """Generate si222.h5 with alm from the committed DFSET (self-contained)."""
    if os.path.exists("si222.h5"):
        return 0
    sys.path.insert(0, os.path.join(project_root, "test"))
    from test_si import gen_alminput_si

    shutil.copy(
        "%s/example/Si/reference/DFSET_harmonic" % project_root, "DFSET_harmonic"
    )
    gen_alminput_si("ALM_irreps.in", 1, dfset="DFSET_harmonic", prefix="si222")
    with open("ALM_irreps.log", "w") as f:
        ret = subprocess.run([almbin, "ALM_irreps.in"], stdout=f)
    if ret.returncode != 0 or not os.path.exists("si222.h5"):
        print("FAIL: could not generate the Si force-constant fixture with alm.")
        return 1
    return 0


def ensure_bto_fixture(project_root):
    """Copy cBTO222.h5 from the same location test_batio3.py uses."""
    if os.path.exists("cBTO222.h5"):
        return 0
    src = os.path.join(
        project_root, "example", "BaTiO3", "scph_relax", "reference", "cBTO222.h5"
    )
    if not os.path.exists(src):
        print("File %s not found" % src)
        return 1
    shutil.copy(src, "cBTO222.h5")
    return 0


def runtest_irreps(almbin, anphonbin, project_root):
    nfail = 0

    # freq_sign: +1 = real, -1 = imaginary (soft), 0 = no check (acoustic)
    if ensure_si_fixture(almbin, project_root) != 0:
        return 1
    nfail += run_case(
        anphonbin,
        prefix="si222",
        fcsfile="si222.h5",
        # The supercell stored by alm is not the primitive cell; give the
        # fcc primitive lattice explicitly.
        cell=[
            "10.203",
            "0.0 0.5 0.5",
            "0.5 0.0 0.5",
            "0.5 0.5 0.0",
        ],
        expects={
            "point_group": "Oh (m-3m)",
            "Gamma_total": "T2g + T1u",
            "Gamma_acoustic": "T1u",
            "Gamma_optic": "T2g",
            "groups": [
                ("T1u", 3, "acoustic", 0),
                ("T2g", 3, "Raman", 1),
            ],
        },
    )

    if ensure_bto_fixture(project_root) != 0:
        return nfail + 1
    nfail += run_case(
        anphonbin,
        prefix="cBTO",
        fcsfile="cBTO222.h5",
        expects={
            "point_group": "Oh (m-3m)",
            "Gamma_total": "4T1u + T2u",
            "Gamma_acoustic": "T1u",
            "Gamma_optic": "3T1u + T2u",
            "groups": [
                ("T1u", 3, "IR", -1),
                ("T1u", 3, "acoustic", 0),
                ("T1u", 3, "IR", 1),
                ("T2u", 3, "silent", 1),
                ("T1u", 3, "IR", 1),
            ],
        },
    )

    # Optional hexagonal, convention-sensitive case: the silent wurtzite
    # modes must come out as B1 (International-Tables sigma_v convention).
    zno_xml = os.path.join(
        project_root, "example", "ZnO", "qha_relax", "ZnO442_harmonic.xml"
    )
    if os.path.exists(zno_xml):
        shutil.copy(zno_xml, "ZnO442_harmonic.xml")
        nfail += run_case(
            anphonbin,
            prefix="ZnO",
            fcsfile="ZnO442_harmonic.xml",
            cell=[
                "1.0",
                "6.1148879046662375 0.0 0.0",
                "-3.0574439523331187 5.2956482667351600 0.0",
                "0.0 0.0 9.8732748090602650",
            ],
            expects={
                "point_group": "C6v (6mm)",
                "Gamma_total": "2A1 + 2B1 + 2E1 + 2E2",
                "Gamma_acoustic": "A1 + E1",
                "Gamma_optic": "A1 + 2B1 + E1 + 2E2",
                "groups": [
                    ("A1(+)E1", 3, "acoustic", 0),
                    ("E2", 2, "Raman", 1),
                    ("B1", 1, "silent", 1),
                    ("A1", 1, "IR+Raman", 1),
                    ("E1", 2, "IR+Raman", 1),
                    ("E2", 2, "Raman", 1),
                    ("B1", 1, "silent", 1),
                ],
            },
        )
    else:
        print("ZnO (optional): skipped, %s not found" % zno_xml)

    return nfail


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)

    dirname = "%s/test/irreps" % project_root
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)

    almbin = "%s/_build/alm/alm" % project_root
    anphonbin = "%s/_build/anphon/anphon" % project_root

    info = runtest_irreps(almbin, anphonbin, project_root)

    if info == 0:
        print("All IRREPS checks passed.")
        sys.exit(0)
    else:
        sys.exit(1)
