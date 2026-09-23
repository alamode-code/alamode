#!/usr/bin/env python
"""displace.py --evec reads anphon's HDF5 (PREFIX.evec.h5, legacy .evec.hdf5)
and text (PREFIX.evec) eigenvector files identically.

Runs anphon on the Si example (KPMODE = 0 list commensurate with the 2x2x2
supercell, PRINTEVEC = 1) once with the default FILE_FORMAT = h5 and once
with FILE_FORMAT = text, checks that both files load to the same arrays, and
that displace.py --random_normalcoord produces identical structures.

Run from the build directory: python3 ../test/test_evec_formats.py
"""

import filecmp
import os
import shutil
import subprocess
import sys

import numpy as np

if __name__ == "__main__":
    project_root = os.path.dirname(os.getcwd())
    exdir = os.path.join(project_root, "example/Si/anharm_IFCs/2_generate_config")
    tools = os.path.join(project_root, "tools")
    anphonbin = os.path.join(project_root, "_build/anphon/anphon")
    workdir = os.path.join(project_root, "test/si/evec_formats")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    os.chdir(workdir)
    for f in ["POSCAR_primitive_cell", "POSCAR_supercell", "si222_harmonic.h5"]:
        shutil.copy(os.path.join(exdir, f), ".")

    displace = [sys.executable, os.path.join(tools, "displace.py"), "--VASP"]
    displace += ["POSCAR_supercell", "--prim", "POSCAR_primitive_cell"]
    displace += ["--random_normalcoord"]
    hint = subprocess.run(displace, capture_output=True, text=True).stdout
    anphon_tail = hint[hint.index("&cell") :]
    assert "PRINTEVEC = 1" in anphon_tail

    general = "&general\n PREFIX = %s\n MODE = phonons\n"
    general += " FCSFILE = si222_harmonic.h5\n KD = Si\n%s/\n"
    for prefix, extra in [("si_h5", ""), ("si_txt", " FILE_FORMAT = text\n")]:
        with open(prefix + ".in", "w") as f:
            f.write(general % (prefix, extra) + anphon_tail)
        with open(prefix + ".log", "w") as f:
            subprocess.run([anphonbin, prefix + ".in"], stdout=f, check=True)
    shutil.copy("si_h5.evec.h5", "si_legacy.evec.hdf5")

    sys.path.insert(0, tools)
    from GenDisplacement import AlamodeDisplace

    loaded = {}
    for fname in ["si_txt.evec", "si_h5.evec.h5", "si_legacy.evec.hdf5"]:
        obj = AlamodeDisplace.__new__(AlamodeDisplace)
        obj._load_phonon_results(fname)
        loaded[fname] = obj
    ref = loaded["si_txt.evec"]
    for obj in loaded.values():
        assert obj._nmode == ref._nmode
        assert np.allclose(obj._qpoints, ref._qpoints, atol=1e-12)
        assert np.allclose(obj._mass, ref._mass, rtol=1e-6)
        assert np.allclose(obj._omega2, ref._omega2, rtol=1e-6, atol=1e-12)
        assert np.allclose(obj._evec, ref._evec, atol=1e-6)
        assert obj._qlist_real == ref._qlist_real
        assert sorted(obj._qlist_uniq) == sorted(ref._qlist_uniq)

    opts = ["--temp", "300", "--random_seed", "1", "-nd", "3"]
    for fname in loaded:
        prefix = fname.replace(".", "_") + "_"
        cmd = displace + ["--evec", fname, "--prefix", prefix] + opts
        subprocess.run(cmd, capture_output=True, check=True)
    for i in range(1, 4):
        h5, legacy = "si_h5_evec_h5_%d.POSCAR" % i, "si_legacy_evec_hdf5_%d.POSCAR" % i
        assert filecmp.cmp(h5, legacy, shallow=False), (h5, legacy)
        # The text file keeps 7 significant digits, HDF5 full doubles.
        x_txt = np.loadtxt("si_txt_evec_%d.POSCAR" % i, skiprows=8)
        x_h5 = np.loadtxt(h5, skiprows=8)
        assert np.allclose(x_txt, x_h5, rtol=0, atol=1e-7), abs(x_txt - x_h5).max()

    print("displace.py --evec h5/text --> pass")
