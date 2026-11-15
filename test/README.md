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
- `test_magmom_xml.py` covers magnetic moments in an XML force-constant file with a
  supercell (GitHub #340): Si with `MAGMOM = 64*1` must give the same bands as without.
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
- `test_ibte_trisym.py` covers the triplet reduction (`TRISYM`) of the BTE collision operator with
  the tetrahedron method (Si, 4x4x4, `ISMEAR = -1`): a cubic IBTE tensor and kxx against a reference,
  the 3ph diagonal and (without isotope in-scattering) kappa equal for `TRISYM = 1` and `0`, VBTE equal
  to IBTE, and a 2-rank run when `mpirun` is available.
- `test_velsym.py` covers the little-group symmetrization of the group velocities:
  cubic SrTiO3 (FC2 renormalized by a 2x2x2 SCPH run at 300 K) on a 4x4x4 mesh must give
  cubic Peierls and coherent kappa tensors with `NONANALYTIC = 2` (about 1e-2 off-diagonal
  without it), invariant finite-difference velocities, an unchanged `NONANALYTIC = 3` kappa,
  and the same result on 2 MPI ranks. `test_velocity_symmetry` (C++, built with anphon) covers
  the average itself: conventions against a symmetric force-constant model, idempotence,
  invariance, tilted degenerate blocks and gauge invariance of the block traces.
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
  sampling (2 2 2 / 2 2 2). It also runs SCPH with `NONANALYTIC = 3` on a skewed
  10-atom cell (2 4 4 meshes, equivalent to the cubic 4x4x4 one and not
  commensurate with the 2x2x2 IFCs), whose frequencies must equal the 5-atom ones
  at the folded q: this needs one Bloch gauge for the IFCs and the Ewald terms.
- `test_cell_images` (C++, built with anphon) covers the minimum-image search of
  `System::get_minimum_distances` beyond the 27 neighboring cells (`images_beyond_27`)
  on orthogonal, hexagonal and skewed supercells against a brute-force search.
- `test_strain_dfc2_projection` (C++, built with anphon) covers the projection of the
  IFC-derived strain derivatives of FC2 (`strain_dfc2_projection.h`) on a random set. It checks
  index symmetry, the ASR per component, Hermiticity at a generic q, idempotence and
  covariance. `test_kpmode0.py` checks the resulting ASR of SrTiO3 runs with RELAX_STR = 2
  and with a fixed sheared strain.
- `test_relaxed_cell_convention.py` covers the geometry convention of `RELAXED_STRUCTURE = 1`
  (BaTiO3, polar tetragonal and sheared states): D(q) equal to the state's for NONANALYTIC = 1-3
  and the bands against `.scph_bands`, velocities against physical-q finite differences, the
  velocity-operator connection with u0, reduced vs full mesh (`ALAMODE_NOSYM`), the
  `relaxed_cell_convention` restart stamp, the Born charges taken from the state
  (`ALAMODE_DUMP_BORN`), and the u = u0 = 0 limit.
- `test_fe_bubble.py` covers `FE_BUBBLE = 1`: the bubble free energy of Si (`MODE = phonons`,
  6x6x6 mesh) must be finite and negative and, with `CLASSICAL = 1`, proportional to T^2 (to the
  7 digits of `.thermo`). On BaTiO3 (`MODE = SCPH`, 2x2x2 SCPH, 4x4x4 mesh) the bubble column of
  `.scph_thermo` must be included in `F_total`. Both runs are repeated on 2 MPI ranks when
  `mpirun` is available (1e-10 relative for SCPH), and `RELAX_STR = 1` must be refused.
- Known gap: the SCP-failure rescue path in the SCPH structural loop
  (`Relaxation::rescue_step_after_scp_failure`, called from the driver in
  scph.cpp) is not exercised by any fixture; none of the test systems fails
  the SCP fixed point. Accepted as uncovered.