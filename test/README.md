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
- `test_kpmode0.py` covers SCPH postprocess on a general k-point list
  (KPMODE = 0) with the non-analytic correction enabled — the only fixture
  exercising the kpoint_general branch.
- Known gap: the SCP-failure rescue path in the SCPH structural loop
  (`Relaxation::rescue_step_after_scp_failure`, called from the driver in
  scph.cpp) is not exercised by any fixture; none of the test systems fails
  the SCP fixed point. Accepted as uncovered.