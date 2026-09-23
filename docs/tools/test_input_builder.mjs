// Checks the generator of the docs input-file builder (docs/source/_static/input_builder.js)
// and runs the generated inputs through the alm and anphon binaries.
//
//   node docs/tools/test_input_builder.mjs            # from the repository root
//   ALM=/path/to/alm ANPHON=/path/to/anphon node docs/tools/test_input_builder.mjs
//
// The binaries default to _build/alm/alm and _build/anphon/anphon; the binary
// checks are skipped when they are not found. anphon runs under `mpirun -np 2`
// when mpirun is on the PATH. Work files go to a temporary directory.
import { createRequire } from "node:module";
import { execFileSync, execSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync, copyFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const ib = createRequire(import.meta.url)(join(repo, "docs/source/_static/input_builder.js"));
const si = join(repo, "example/Si");

// -- generator checks (no binaries needed) ----------------------------------
const poscar = ib.parsePoscar(readFileSync(join(si, "anharm_IFCs/2_generate_config/POSCAR_supercell"), "utf8"));
assert.equal(poscar.kd, "Si");
assert.equal(poscar.positions.split("\n").length, 64);
assert.ok(Math.abs(Number(poscar.scale) - 5.431 / ib.BOHR_IN_ANGSTROM) < 1e-8);

// Cartesian + Selective dynamics give the same fractional coordinates as Direct.
const cart = ib.parsePoscar("x\n1.0\n0 2.7 2.7\n2.7 0 2.7\n2.7 2.7 0\nSi\n2\nSelective dynamics\nCartesian\n" +
  "0 0 0 T T T\n1.35 1.35 1.35 F F F\n");
assert.deepEqual(cart.positions.split("\n")[1].split(/\s+/).slice(1).map(Number), [0.25, 0.25, 0.25]);
assert.throws(() => ib.parsePoscar("x\n1.0\n1 0 0\n0 1 0\n0 0 1\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n"), /poscar_vasp4/);

// Three per-axis scale factors scale the x, y, z components of the lattice and of Cartesian positions.
const col = (p, line) => p.positions.split("\n")[line].split(/\s+/).map(Number);
const axes = ib.parsePoscar("x\n2 3 4\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nCartesian\n0.5 0.5 0.5\n"); // 1, 1.5, 2 angstrom
assert.deepEqual(axes.lattice.split("\n").map((r) => r.trim().split(/\s+/).map(Number)), [[2, 0, 0], [0, 3, 0], [0, 0, 4]]);
assert.ok(Math.abs(Number(axes.scale) - 1 / ib.BOHR_IN_ANGSTROM) < 1e-8);
assert.deepEqual(col(axes, 0), [1, 0.5, 0.5, 0.5]);
assert.throws(() => ib.parsePoscar("x\n2 -3 4\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n"), /poscar_lattice/);
// ... and give the same fractional coordinates as one isotropic factor.
const iso = (sc) => ib.parsePoscar(`x\n${sc}\n0 .5 .5\n.5 0 .5\n.5 .5 0\nSi\n2\nCartesian\n0 0 0\n.25 .25 .25\n`);
assert.equal(iso("5.4 5.4 5.4").positions, iso("5.4").positions);

// Repeated element names (POTCAR suffixes stripped) merge into one species.
const dup = ib.parsePoscar("x\n1.0\n4 0 0\n0 4 0\n0 0 4\nSi_pv Ge Si\n1 1 1\nDirect\n0 0 0\n.5 0 0\n0 .5 0\n");
assert.equal(dup.kd, "Si Ge");
assert.deepEqual(dup.positions.split("\n").map((l) => Number(l.split(/\s+/)[0])), [1, 2, 1]);
assert.deepEqual(ib.cutoffPairs(dup.kd.split(" ")), ["Si-Si", "Si-Ge", "Ge-Ge"]);

const alm = (over) => ({
  program: "alm", prefix: "si222", mode: "suggest", ...poscar, norder: "2",
  cutoffs: { "Si-Si": ["None", "7.3"] }, ...over,
});
const anphon = (over) => ({
  program: "anphon", prefix: "si222", mode: "phonons", fcsfile: "si222.h5", omitCell: true,
  kpmode: "1", path: ib.PRESETS.fcc, mesh: "20 20 20", solver: "RTA", ...over,
});
const gen = (s) => {
  const r = ib.generateInput(s);
  assert.deepEqual(r.errors, [], `unexpected errors: ${JSON.stringify(r.errors)}\n${r.text}`);
  return r.text;
};
const errs = (s) => ib.generateInput(s).errors.map((e) => e[0]);

assert.deepEqual(errs(alm({ positions: "", scale: "", lattice: "", kd: "" }))
  .filter((e) => e === "struct_missing"), ["struct_missing"]);
assert.ok(errs(alm({ cutoffs: { "Si-Si": ["None", ""] } })).includes("cutoff_missing"));
assert.ok(errs(alm({ mode: "optimize" })).includes("dfset_missing"));
assert.ok(errs(anphon({ mode: "kappa", kpmode: "1" })).includes("kappa_mesh"));
assert.ok(errs(anphon({ fcsfile: "si222.xml" })).includes("xml_needs_cell"));
assert.ok(errs(anphon({ mode: "SCPH", kmeshInterp: "4 4 4", kmeshScf: "6 6 6" })).includes("kmesh_multiple"));
assert.match(gen(anphon({ mode: "SCPH", kmeshInterp: "4 4 4", kmeshScf: "8 8 8", relaxStr: "1" })), /&relax\n\/\n/);
// RELAX_STR = 2, 3 need &strain data the builder does not write.
assert.deepEqual(ib.RELAX_OPTIONS, { SCPH: ["0", "1"], QHA: ["1"] });
assert.ok(errs(anphon({ mode: "QHA", kmeshInterp: "4 4 4", kmeshScf: "8 8 8", relaxStr: "3" })).includes("relax_bad"));
assert.ok(!/RELAX_STR/.test(gen(anphon({ mode: "QHA", kmeshInterp: "4 4 4", kmeshScf: "8 8 8", relaxStr: "1" }))));
// Hand-typed duplicate KD and one-point band segments are rejected.
assert.ok(errs(alm({ kd: "Si Si" })).includes("kd_dup"));
assert.ok(errs(anphon({ path: "G 0 0 0 X 0.5 0.5 0 1" })).includes("path_npts"));
assert.ok(!errs(anphon({ path: "G 0 0 0 X 0.5 0.5 0 2" })).includes("path_npts"));
assert.match(ib.format("ja", ["path_npts", 1]), /2 以上/);
assert.ok(!/&cell/.test(gen(anphon({}))));
console.log("generator checks: OK");

// -- binary checks --------------------------------------------------------------
const almBin = process.env.ALM || join(repo, "_build/alm/alm");
const anphonBin = process.env.ANPHON || join(repo, "_build/anphon/anphon");
if (!existsSync(almBin) || !existsSync(anphonBin)) {
  console.log(`binary checks: SKIPPED (${almBin} or ${anphonBin} not found)`);
  process.exit(0);
}
let mpirun = null;
try { mpirun = execSync("command -v mpirun", { encoding: "utf8" }).trim() || null; } catch { /* serial */ }

const work = mkdtempSync(join(tmpdir(), "input_builder_"));
const run = (bin, text, name, mpi) => {
  writeFileSync(join(work, name + ".in"), text);
  const [cmd, args] = mpi && mpirun ? [mpirun, ["-np", "2", bin, name + ".in"]] : [bin, [name + ".in"]];
  try {
    execFileSync(cmd, args, { cwd: work, stdio: ["ignore", "pipe", "pipe"], encoding: "utf8" });
  } catch (e) {
    throw new Error(`${name} failed:\n${text}\n${e.stdout}\n${e.stderr}`);
  }
};
const expect = (name, ...files) => files.forEach((f) =>
  assert.ok(existsSync(join(work, f)), `${name}: ${f} was not written (work dir ${work})`));
copyFileSync(join(si, "reference/DFSET_harmonic"), join(work, "DFSET_harmonic"));
copyFileSync(join(si, "reference/DFSET_cubic"), join(work, "DFSET_cubic"));

// a. alm suggest, 2x2x2 supercell POSCAR, NORDER = 2
run(almBin, gen(alm({})), "a_suggest");
expect("a", "si222.pattern_HARMONIC", "si222.pattern_ANHARM3");
// a'. the "Load example (Si)" settings: conventional cell + SUPERCELL + PRIMCELL = Auto
const conv = ib.parsePoscar("Si\n5.431\n1 0 0\n0 1 0\n0 0 1\nSi\n8\nDirect\n0 0 0\n0 .5 .5\n.5 0 .5\n.5 .5 0\n" +
  ".25 .25 .25\n.25 .75 .75\n.75 .25 .75\n.75 .75 .25\n");
run(almBin, gen(alm({ ...conv, prefix: "si_ex", primcell: "Auto", supercell: "2 2 2" })), "a_example");
expect("a'", "si_ex.pattern_HARMONIC", "si_ex.pattern_ANHARM3");
// a''. repeated element names in the POSCAR ("Si Si") merge into one species
const merged = ib.parsePoscar("Si\n5.431\n1 0 0\n0 1 0\n0 0 1\nSi Si\n4 4\nDirect\n0 0 0\n0 .5 .5\n.5 0 .5\n.5 .5 0\n" +
  ".25 .25 .25\n.25 .75 .75\n.75 .25 .75\n.75 .75 .25\n");
run(almBin, gen(alm({ ...merged, prefix: "si_merged", primcell: "Auto", supercell: "2 2 2" })), "a_merged");
expect("a''", "si_merged.pattern_HARMONIC", "si_merged.pattern_ANHARM3");
// b. alm optimize, harmonic
run(almBin, gen(alm({ mode: "optimize", norder: "1", cutoffs: { "Si-Si": ["None"] }, primcell: "Auto",
  dfset: "DFSET_harmonic" })), "b_harmonic");
expect("b", "si222.h5");
// b'. cubic fit on top of the harmonic one (FC2FIX), for d
run(almBin, gen(alm({ prefix: "si222_cubic", mode: "optimize", primcell: "Auto", dfset: "DFSET_cubic",
  fc2fix: "si222.h5" })), "b_cubic");
expect("b'", "si222_cubic.h5");
// c. anphon phonons, &cell omitted, KPMODE = 1
run(anphonBin, gen(anphon({})), "c_phonons", true);
expect("c", "si222.bands");
// d. anphon kappa on a small mesh
run(anphonBin, gen(anphon({ prefix: "si222_kappa", mode: "kappa", fcsfile: "si222_cubic.h5", kpmode: "2",
  mesh: "6 6 6", tmin: "300", tmax: "300", dt: "10" })), "d_kappa", true);
expect("d", "si222_kappa.kl");
console.log(`binary checks: OK (${work})`);
if (!process.env.KEEP) rmSync(work, { recursive: true });
