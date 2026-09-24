## Simple script for testing (not unit test)

After change directory to `test/`, please run the test script as:

```
> python test_si.py
```

If the test passes, you will see the output as 
```
Silicon ALM --> pass
Silicon ANPHON --> pass
```

### Coverage notes

- `test_qha.py` covers MODE = QHA structural optimization (fresh run, h5
  restart, legacy-text restart, band-parallel V4 with IALGO = 1 on 2 MPI
  ranks, and perturbative QHA with RELAX_STR = 3).
- `test_fc3file.py` covers the `FC3FILE` tag: harmonic FC2 from XML with
  cubic FC3 from HDF5 (Si, same supercell, checked against the FCSFILE
  reference), a 4x4x2 harmonic / 3x3x2 cubic combination (ZnO, checked
  against the `FC2FILE` + `FCSFILE` route), and rejection of an unsupported
  file extension. It reuses the Si fixtures of `test_si.py` and generates
  them with alm when absent.
- `test_fourph.py` covers the four-phonon channel (`INCLUDE_4PH = 1`) on a
  2x2x2 BaTiO3 mesh for the Lorentzian, Gaussian and adaptive smearing
  schemes, against linewidths of the reference (pre-factorization)
  implementation stored in `example/BaTiO3/kappa_4ph/reference_for_test`,
  plus a 2-rank MPI consistency run when `mpirun` is available.
- `test_kpmode0.py` covers SCPH postprocess on a general k-point list
  (KPMODE = 0) with the non-analytic correction enabled — the only fixture
  exercising the kpoint_general branch.
- `test_scph_h5.py` also covers `BUBBLE > 0` with `RELAX_STR != 0` (BaTiO3, tetragonal
  at 280 K): the in-run on-shell bubble, restarted from the state file, must match
  `MODE = selfenergy` + `RELAXED_STRUCTURE = 1` mode by mode, differ from the undeformed
  reference, and be reproduced on 2 MPI ranks when `mpirun` is available.
- `test_scph_hessian.py` covers `BUBBLE = 4` (curvature of the SCP free energy) on a 1x1x2
  BaTiO3 cell relaxed from a P1 start: the analytic force Jacobian (static bubble + quartic
  ladder, GMRES) against finite differences of the SCP force (`BUBBLE_FD_CHECK`), its
  symmetry, the necessity of the ladder, and a 2-rank run. `test_scph_hessian_kernels`
  (C++) covers the occupation factor, divided differences and the Daleckii-Krein map.
- `test_dfc2_fold.py` covers `DFC2FILE` from an SCPH run whose `&cell` is an
  integer supercell of the primitive cell (BaTiO3 1x1x2 cell, 2 2 1 meshes):
  the folded correction must reproduce a primitive-cell SCPH with the same q
  sampling (2 2 2 / 2 2 2).
- Known gap: the SCP-failure rescue path in the SCPH structural loop
  (`Relaxation::rescue_step_after_scp_failure`, called from the driver in
  scph.cpp) is not exercised by any fixture; none of the test systems fails
  the SCP fixed point. Accepted as uncovered.