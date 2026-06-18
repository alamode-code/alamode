# ALM Constraint Elimination: Coordinate-Factorization Backend (RREF Replacement)

**Status:** design / pre-implementation
**Scope decision (agreed):** Policy A (reproduce RREF's free/dependent partition); keep
inter-order rotational constraints neglected in the algebraic path (per-order, as today);
deliver a design doc before any code.

---

## 1. Goal

Replace the Gauss–Jordan (RREF) elimination used to build ALM's algebraic constraint map
with a numerically stable, factorization-based backend that produces the **identical**
downstream data structures. The reduced coordinate `z` must remain a *subset of the
original symmetry-irreducible IFC coordinates*, because the elastic-net / adaptive-lasso
L1/L2 penalties are applied directly in `z` (basis is part of the model). An arbitrary
orthonormal QR/SVD nullspace basis is therefore **not** an acceptable substitute for
LMODEL = 2/3; it is only safe for pure OLS.

Non-goals (this phase): replacing the OLS constrained solvers (`dgglse`, KKT
`solveGQRSparse`); enforcing inter-order rotational invariance in the algebraic path;
changing the L1 penalty semantics.

**Performance is a co-equal goal.** A large maxorder=5 adaptive-lasso fit
(TaIrTe4, 4×1×1) is dominated by the reduction phase. The fixes below are ordered so the
biggest, lowest-risk speedup lands first and is independent of the stability refactor.

---

## 1a. Performance: the dominant reduction bottlenecks

Reasoning + independent Codex review converge on three costs, the first being near-pure
waste in the LASSO path.

**(P1) Wasted merged dense QR in the algebraic path — VERIFIED, zero-risk to remove.**
`update_constraint_matrix` *always* calls `build_constraint_matrix_sparse`
(`constraint.cpp:419`), which densifies the **full merged** constraint matrix to a dense
`P × N` buffer (`N` = total irreducible params across *all* orders) and runs dense
`dgeqp3` (`least_squares.cpp:204-223`, via `constraint.cpp:604`). For maxorder=5 TaIrTe4,
`N` is ~1e4–1e5, so this is an `O(P·N)` dense allocation (can be many GB) and
`O(P·N·min(P,N))` factorization. In the algebraic (ICONST≥10 / LASSO) path the result
(`const_mat_sparse`/`const_rhs_vec`) is **never consumed**: the solve uses the per-order
mapping (`project_constraints`/`recover_original_forceconstants`), and
`get_exist_constraint()` checks `const_self`/`const_fix`/`const_relate` directly and never
reads `number_of_constraints` in the algebraic branch (`constraint.cpp:858-879`).
*Fix:* guard `build_constraint_matrix_sparse`, the `number_of_constraints` assignment, and
`build_constraint_matrix_dense` under `if (!constraint_algebraic)` in
`update_constraint_matrix`. (Optional: set `number_of_constraints =
Σ(const_fix[o].size()+const_relate[o].size())` after the mapping, for reporting only.)
**Likely the single biggest win; independent of the RREF→factorization work; ship first.**

**(P2) Per-order `rref_sparse` — `O(r·P·N)` hashed Gauss–Jordan with unbounded fill.**
Still runs per order in the algebraic path to build the mapping. Each pivot scans **all**
rows (`rref.cpp:182`) and does per-element `unordered_map` find/insert/erase
(`rref.cpp:190-214`); fill-in is unbounded. The coordinate-factorization backend (§5),
done sparsely, removes both the all-row scan and the fill growth — this is where the
stability work and the performance work coincide.

**(P3) Densifying "sparse" rank reduction (`least_squares.cpp:204-223`).** Where row-rank
reduction is genuinely needed (the non-algebraic path, and the per-order `qrd` mode), it
still densifies to `P × N` + dense `dgeqp3`. Replace with SuiteSparseQR on `Cᵀ` (best) or
`Eigen::SparseQR` + `COLAMDOrdering` (already have Eigen; lower integration risk).

**Ordering tension (P2/§5).** Policy A needs *natural* column order to reproduce RREF's
partition, but sparse performance wants a fill-reducing order. Compromise: select `J_B` in
natural order, then let the sparse solve of the *fixed* `C_B` use its own internal
fill-reducing ordering — public coordinates are preserved while the factorization still
gets fill reduction.

Prioritized fix table:

| Rank | Fix | Expected gain | Risk | Effort |
|---:|---|---|---|---|
| 1 | P1: skip merged QR + dense build when `constraint_algebraic` | Very high (removes `O(P·N)` mem + dense QR from every LASSO setup) | Low (output unused in algebraic path) | Small |
| 2 | P3: true sparse rank-revealing QR for the row-rank job | High (quadratic mem + cubic time → sparse fill cost) | Medium (match rank/pivot semantics; tests) | Medium |
| 3 | P2/§5: coordinate factorization replaces `rref_sparse` | High (kills all-row Gauss–Jordan + fill) + stability | Medium-high (reproduce partition & signs) | Medium-large |
| 4 | If `rref_sparse` is kept: column→row incidence to avoid all-row scans; flat sorted-vector rows for small nnz | Medium-high | Medium (fill/cleanup is delicate) | Medium |
| 5 | Reserve triplet/vector capacity (`constraint.cpp:538`); replace `boost::bimap` with dual flat vectors | Low-medium | Low | Small |

---

## 2. Current flow and the exact output contract

Algebraic path (ICONST ≥ 10, `constraint_algebraic == 1`):

1. Per order, `const_self[order]` (a `ConstraintSparseForm` = `vector<map<col,val>>`) holds
   the merged homogeneous constraints `C x = 0` (translation ⊕ rotation-self ⊕ symmetry).
2. `rref_sparse(nparam, const_self[order], tol)` reduces it to reduced row echelon form
   (`rref.cpp:128`). This does **double duty**: drops redundant rows *and* produces the
   echelon structure parsed in the next step.
3. `get_mapping_constraint` (`constraint.cpp:637`) parses each echelon row into:
   - `const_fix[order]`  — `ConstraintTypeFix{p_index_target, val_to_fix}`
   - `const_relate[order]` — `ConstraintTypeRelate{p_index_target, alpha[], p_index_orig[]}`
   - `index_bimap[order]` — bimap(compact free index ↔ original irreducible index)
4. The reduced ("compact") sensing matrix is built column-by-column in
   `project_constraints` (`optimize.cpp:2923`) / `project_energy_row` (`:2982`); the
   solution is expanded back by `recover_original_forceconstants` (`:3032`).

**The contract every backend must satisfy** (semantics consumed downstream):

- `index_bimap[order]`: `left` = compact free-parameter index (within order),
  `right` = original irreducible index. `N_new = Σ index_bimap[order].size()`.
- `const_fix[order][k]`: original index `p_index_target` is fixed to `val_to_fix`.
  - From the homogeneous map: `val_to_fix == 0`.
  - From FC2FIX/FC3FIX: an **entire order** is pre-fixed with nonzero `val_to_fix`, and
    `get_mapping_constraint` *skips* that order (`constraint.cpp:652`). This is the affine
    `x0` term and **must be preserved**.
- `const_relate[order][k]`: `x[target] = − Σ_j alpha[j] · x[orig[j]]`
  (recovery negates at `optimize.cpp:3072`). **Invariant:** every `orig[j]` is itself a
  *free* parameter (present in `index_bimap.right`). `project_constraints` relies on this
  with an unchecked `index_bimap.right.at(orig[j])` (`optimize.cpp:2967`).

---

## 3. Key equivalence that de-risks the migration

For a **fixed** partition of columns into dependent `J_B` and free `J_F`, the relation

```
x_B = − C_B^{-1} C_F x_F ,   C_B = C[:,J_B] ,  C_F = C[:,J_F]
```

is **unique**. RREF with pivot columns `J_B` produces exactly `[I | C_B^{-1} C_F]`, so the
elimination coefficients RREF emits equal `C_B^{-1} C_F` mathematically. RREF guarantees
`C_B` is square and full rank by construction (a column becomes a pivot only when a nonzero
pivot exists).

Therefore: **if the new backend selects the same partition as RREF, the emitted
`const_relate`/`const_fix` are identical to RREF's up to floating point** — only *how* the
numbers are computed changes (stable triangular solve vs. accumulating Gauss–Jordan
divisions). This makes the legacy-compatibility test (§7) an equality test at ~1e-10, and
guarantees `const_relate` sparsity is unchanged (so the compact sensing matrix stays as
sparse as today).

**Caveat (honest framing of Policy A):** the stability gain is *partial*. Policy A removes
round-off accumulation, but if the instability originates in an ill-conditioned `C_B`
(a poor column partition), Policy A inherits it — a stable solve of an ill-conditioned
system is still ill-conditioned. Curing that requires Policy B (rank-revealing column
pivoting), which changes which IFCs are free and thus perturbs the penalized model; it is
deferred to an expert-only opt-in.

---

## 4. Matching RREF's partition: which columns are dependent

RREF processes columns left-to-right; a column becomes a pivot (→ dependent, `J_B`) iff it
is **linearly independent of the columns to its left** in the row space. Equivalently,
`J_B` = the *leftmost maximal independent set of columns* of `C`; `J_F` = the rest. The
per-row target chosen by `get_mapping_constraint` is `ConstVec[0].col` = the smallest
column in the reduced row = the pivot column (`constraint.cpp:669`). So:

- target/dependent indices = leftmost-independent columns,
- origins/free indices = the remaining columns (always to the right of, and disjoint from,
  the pivots — the §2 invariant).

This column-independence property is **independent of pivoting strategy**, so we can match
RREF's `J_B` exactly while using *partial row pivoting* for the numerics.

---

## 5. Reference algorithm (per order)

Input: `C = const_self[order]` (raw, not RREF'd), `m × n`, `n = nequiv[order].size()`,
tolerance `tol`. Output: `const_fix[order]`, `const_relate[order]` (then the existing
`index_bimap` construction runs unchanged).

**Rank-revealing LU with natural column order + partial row pivoting + column skipping**
(this is RREF's column/row selection, but stably pivoted; it yields rank, independent rows,
the partition, and the elimination coefficients in one pass):

```
used_rows = ∅ ; J_B = [] ; pivot_row_of[col] = {}
for col = 0 .. n-1 (natural / left-to-right order):     # matches RREF leftmost rule
    among rows ∉ used_rows, find r* = argmax |C[r, col]|  # partial pivot (stability)
    if |C[r*, col]| <= tol:
        continue                                          # col is FREE (J_F)
    J_B.append(col); used_rows.add(r*); pivot_row_of[col] = r*
    eliminate col from all other rows ∉ used_rows         # forward elimination
# rank r = |J_B| ; rows ∉ used_rows are redundant (consistent, residual ~0)
J_F = [c for c in 0..n-1 if c ∉ J_B]
# Back-substitute the upper-triangular pivot system to express each dependent
# variable purely in terms of free variables  → relation matrix T (x_B = T x_F).
```

Notes:
- For per-order sizes a dense kernel is fine. A sparse realization (scalability track) can
  use a sparse LU/QR with **natural column ordering** (COLAMD reordering must *not* be used
  here — it would change `J_B` and break Policy A); confirm tooling during implementation.
- The column-skip criterion uses `tol`; see §9 for the tolerance policy (scale-aware,
  single source).
- This naturally produces `T` with columns indexed by `J_F` only, so the §2 free-origin
  invariant holds automatically.

**Emit the triple** (signs matter):

`T = − C_B^{-1} C_F`, and RREF stores `alpha = C_B^{-1} C_F = −T` (recovery negates):

```
for each dependent column d = J_B[i]:
    if row i of T is all (|·| <= tol):           # equivalent to RREF single-entry row
        const_fix[order].push_back({target=d, val_to_fix=0.0})
    else:
        alpha = [], orig = []
        for each free column f = J_F[j] with |T[i,j]| > tol:
            alpha.push_back(-T[i,j])              # alpha = (C_B^{-1} C_F)[i,j]
            orig.push_back(f)
        const_relate[order].push_back({target=d, alpha, orig})
# index_bimap built by the EXISTING loop (constraint.cpp:702-730) — reuse verbatim.
```

Sanity check (must hold in tests): constraint `x0 + 2·x1 = 0`, free `z = x1` ⇒
`x0 = −2 z`, i.e. `alpha = [2]`, `orig = [x1]`, recovery gives `x0 = −2·x1`. ✓

---

## 6. Affine x0 / FC2FIX-FC3FIX handling

No change in structure. Orders pre-fixed from file (`fix_harmonic`/`fix_cubic`) arrive with
`const_fix[order]` already populated (nonzero `val_to_fix`); the new builder must **skip**
those orders exactly like `get_mapping_constraint` (`constraint.cpp:652`), leaving the
file-fix entries untouched. The RHS shift `b ← b − A x0` is already handled by
`project_constraints`/`project_energy_row` and is unaffected.

---

## 7. Integration & file touch-points

Minimize surface area: replace only the per-order *relation-derivation* block; reuse the
`index_bimap` construction and all downstream code.

| File | Change |
|---|---|
| `alm/fcs.h` (`ReductionAlgo`, ~176) | add enum value `coord_factorization` |
| `alm/constraint.cpp` `set_reduction_algorithm` (~742) | map `ialgo == 3` → `coord_factorization` |
| `alm/input_parser.cpp` (~1169, 1430) | document `ALGO_REDUCTION = 3`; default stays `rref` until validated |
| `alm/constraint.cpp` `update_constraint_matrix` (~407, 439-454) | when algebraic & algo==coord_factorization: **skip `rref_sparse`**; call new per-order builder, then the existing index_bimap loop |
| `alm/constraint.cpp` / `.h` | new private method `build_coordinate_map_order(const ConstraintSparseForm& C_in, size_t nparam, double tol, vector<ConstraintTypeFix>& fix_out, vector<ConstraintTypeRelate>& relate_out)` |
| `alm/least_squares.cpp` / `.h` | numerical kernel `coordinate_elimination(const ConstraintSparseForm& C, size_t n, double tol, vector<size_t>& J_B, vector<size_t>& J_F, /*T*/ DenseOrSparse& T, int& rank)` next to `get_independent_rows*` |
| `alm/rref.cpp` / `.h` | **leave intact** as `rref` legacy backend |

Refactor `get_mapping_constraint` so the `index_bimap` construction
(`constraint.cpp:702-730`) is callable independently of how `const_fix`/`const_relate` were
produced (extract a small helper, or have both backends populate the two vectors then call
the shared bimap builder). This guarantees `index_bimap` semantics are byte-identical
across backends.

---

## 8. Test matrix

Add to the existing test/example harness; run for **LMODEL = 1, 2, 3** because L1 semantics
depend on `index_bimap` + relation coefficients.

1. **Constraint residual** — random `z`, recover `x`, assert `‖C x‖ ≤ tol` per order.
2. **Legacy compatibility** (the crux of Policy A) — for identical inputs, assert the new
   backend's `const_fix`/`const_relate`/`index_bimap` equal the `rref` backend's:
   same `J_B`/`J_F` sets, same `index_bimap`, `alpha` within ~1e-10. Expected to pass
   tightly per §3.
3. **Reduced sensing matrix** — `‖A·P_old − A·P_new‖ / ‖A·P_old‖ ≤ tol` (or compare
   `A P z` for random `z`).
4. **End-to-end regression** on representative examples (Si, Si_LAMMPS, BaTiO3, ZnO):
   compare `N_new`, nonzero IFC counts, constraint residuals, fit residuals, solution_path
   output, and recovered `.xml`/`.fcs`. For OLS, compare *residuals* (the constrained
   solution may be non-unique); for enet/adaptive-lasso, expect match within solver
   tolerance given §3.
5. **Stress** — a deliberately ill-conditioned constraint block to show the new backend
   stays finite/accurate where RREF degrades (documents the partial-stability claim of §3).

---

## 9. Tolerance policy

Today RREF is called with `eps8` (translation/symmetry/Huang), `eps6` (rotation,
`constraint.cpp:1801`), a `1e-12` default in `rref.h`, plus an internal `eps10`
zero-criterion (`rref.cpp:141`). Consolidate the new backend on **one** scale-aware
threshold for rank/zero decisions: `tol = max(m,n) · ‖C‖ · ε` (matching
`find_independent_rows_dense`, `least_squares.cpp:64`), with an optional user override.
Keep the legacy per-call `eps*` only inside the `rref` legacy path.

---

## 10. Parallel low-risk track (robustness fixes from review)

Independent of the RREF replacement; can ship separately:

- `solveGQRSparse` returns `void` and leaves `x` unset if all solvers fail
  (`least_squares.cpp:795-801`) while the caller uses `x` (`optimize.cpp:337`) → return a
  status, fail early.
- QR-reduction `info` codes ignored (`constraint.cpp:1353,1807,2557`, `fcs.cpp:487`).
- Uninitialized RHS `Eigen::VectorXd dvec(nrows)` read in the `ConstraintSparseForm`
  overload (`least_squares.cpp:167-168`) — harmless today (reduced RHS discarded) but UB.
- `get_standardizer` divide-by-zero for constant/zero columns (`optimize.cpp:1845-1855`).
- `find_independent_rows_dense` zero-size guard before reading `A_data[0]`
  (`least_squares.cpp:61`).

---

## 11. Rollout

0. **Ship P1 first (performance, standalone).** Guard the merged reduction + dense build
   under `if (!constraint_algebraic)`. No new backend, no semantics change; verify a LASSO
   example produces identical output and times the reduction phase before/after. This alone
   is expected to remove the dominant cost of the reported maxorder=5 run.
1. Land the coordinate-factorization kernel + per-order builder behind `ALGO_REDUCTION = 3`
   (Policy A); default stays `rref`. Addresses P2 + stability.
2. Pass tests §8.1–§8.3, then §8.4 regression across examples.
3. Flip default to `coord_factorization`; keep `rref` selectable for reproducibility.
4. P3: replace the densifying row-rank QR with a true sparse rank-revealing factorization
   (benefits the non-algebraic path and any retained `qrd` use).
5. (Later, separate proposals) Policy B expert mode (rank-revealing pivot), documented as
   potentially changing penalized-model results; inter-order rotational invariance via a
   multi-order block `C`.

---

## 12. Risks

- **Sparse natural-ordering tooling**: matching `J_B` requires a factorization with natural
  column order; common sparse libraries default to fill-reducing reordering. Resolve during
  implementation (custom elimination, or a library that supports natural ordering); dense
  per-order kernel is the safe fallback.
- **Non-unique OLS solutions**: do not require bitwise-identical IFCs for OLS regression;
  compare residuals.
- **Tolerance changes** can shift rank decisions at the margin; the stress test and
  regression suite are the guardrails.
