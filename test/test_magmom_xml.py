#!/usr/bin/env python
"""Magnetic moments in an XML force-constant file with a supercell (GitHub #340).

alm fits the harmonic IFCs of the 64-atom Si supercell twice, without and with
MAGMOM = 64*1, and writes PREFIX.xml (FCS_ALAMODE = 1). Equal moments on all
atoms do not change the symmetry, so anphon must give the same phonon bands
from both files. In 2.0.0 the moments of the supercell atoms were read past
the end of the primitive-cell array.

Run:  python test_magmom_xml.py   (from test/, binaries in ../_build)
"""

import os
import shutil
import subprocess
import sys

import numpy as np

from test_si import gen_alminput_si


def run(binary, inp):
    with open(inp.replace(".in", ".log"), "w") as f:
        return subprocess.run(
            [binary, inp], stdout=f, stderr=subprocess.STDOUT
        ).returncode


def runtest(almbin, anphonbin, project_root):
    shutil.copy(
        "%s/example/Si/reference/DFSET_harmonic" % project_root, "DFSET_harmonic"
    )
    for prefix, magmom in (("si_nomag", ""), ("si_mag", "; MAGMOM = 64*1")):
        gen_alminput_si("alm_%s.in" % prefix, 1, dfset="DFSET_harmonic", prefix=prefix)
        with open("alm_%s.in" % prefix) as f:
            text = f.read()
        assert "KD = Si" in text
        with open("alm_%s.in" % prefix, "w") as f:
            f.write(text.replace("KD = Si", "KD = Si" + magmom))
        if run(almbin, "alm_%s.in" % prefix) != 0:
            print("alm failed for %s" % prefix)
            return 1
        with open("ph_%s.in" % prefix, "w") as f:
            f.write(
                "&general\n PREFIX = %s; MODE = phonons; FCSFILE = %s.xml\n/\n"
                % (prefix, prefix)
            )
            f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
            f.write(
                "&kpoint\n  1\n  G 0.0 0.0 0.0 X 0.5 0.5 0.0 11\n  G 0.0 0.0 0.0 L 0.5 0.5 0.5 11\n/\n"
            )
        if os.path.exists("%s.bands" % prefix):
            os.remove("%s.bands" % prefix)
        if run(anphonbin, "ph_%s.in" % prefix) != 0:
            print("anphon failed for %s" % prefix)
            return 1

    with open("si_mag.xml") as f:
        if "MagneticMoments" not in f.read():
            print("si_mag.xml has no MagneticMoments")
            return 1

    ref = np.loadtxt("si_nomag.bands")
    now = np.loadtxt("si_mag.bands")
    if ref.shape != now.shape:
        print("band shapes differ: %s vs %s" % (ref.shape, now.shape))
        return 1
    diff = np.abs(ref - now).max()
    print("max |difference| of the bands with and without moments: %.3e cm^-1" % diff)
    return 0 if diff < 1.0e-6 else 1


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    almbin = "%s/_build/alm/alm" % project_root
    anphonbin = "%s/_build/anphon/anphon" % project_root
    if len(sys.argv) > 2:
        almbin, anphonbin = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])

    workdir = "%s/test/si_magmom" % project_root
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    info = runtest(almbin, anphonbin, project_root)
    print("Magnetic moments in XML --> %s" % ("pass" if info == 0 else "failed"))
    sys.exit(info)
