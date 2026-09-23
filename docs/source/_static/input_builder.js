/* Input-file builder for alm and anphon (docs/source/input_builder.rst).
   The generator functions (parsePoscar, generateInput) are pure and exported
   for node; docs/tools/test_input_builder.mjs runs their output through the
   alm and anphon binaries. The form is built in the browser only. */
(function () {
  "use strict";

  const BOHR_IN_ANGSTROM = 0.52917721092; // include/constants.h

  // -- labels (tag names such as MODE stay untranslated) ----------------------

  const LABELS = {
    en: {
      program: "Program", mode: "Mode",
      structure: "Structure",
      poscar: "Paste a VASP 5 POSCAR (with element names) to fill the fields below.",
      readPoscar: "Read POSCAR", loadExample: "Load example (Si)",
      scale: "Lattice constant a (bohr)",
      vectors: "Lattice vectors in units of a (three rows)",
      kd: "Elements (KD)",
      positions: "&position: species index and fractional coordinates, one atom per line",
      poscarRead: "Read {0} atoms of {1}.",
      interaction: "Force constants",
      cutoff: "Cutoff radii in bohr, or None (no cutoff)",
      pair: "Pair", orders: ["Harmonic", "Cubic", "Quartic"],
      cutoffHint: "Enter the elements first.",
      cells: "Cell transformation (optional)",
      primcellHint: "Auto, or 1, 3 or 9 entries",
      supercellHint: "1, 3 or 9 integers",
      optimize: "Fitting (&optimize)",
      forceconst: "Force constants",
      omitCell: "Omit &cell (use the cell stored in FCSFILE)",
      cellNote: "A given &cell must be the primitive cell or a supercell of it.",
      kpoints: "k points (&kpoint)",
      kpPath: "Band path (KPMODE = 1)", kpMesh: "Uniform mesh (KPMODE = 2)",
      pathHelp: "One segment per line: label k1 k2 k3 label k1 k2 k3 number-of-points (fractional coordinates of the reciprocal lattice vectors). The fcc preset assumes the primitive vectors a/2(0,1,1), a/2(1,0,1), a/2(1,1,0).",
      preset: "Preset",
      presets: { "": "—", fcc: "fcc: G-X-G-L", sc: "Simple cubic: G-X-M-G-R-M" },
      kappa: "Thermal conductivity (&kappa)",
      include4ph: "Four-phonon scattering (INCLUDE_4PH = 1)",
      temps: "Temperatures (K)",
      scph: "Self-consistent phonons (&scph)", qha: "Quasi-harmonic approximation (&qha)",
      relaxNote: "Cell relaxation (RELAX_STR = 2 or 3) needs strain-coupling data (&strain, STRAINFILE); write it by hand following the anphon input reference.",
      output: "Generated input", copy: "Copy", copied: "Copied", download: "Download",
      ok: "No problems found.",
      errors: {
        prefix_missing: "PREFIX is empty.",
        struct_missing: "The structure is missing: paste a POSCAR or fill in the lattice, the elements and the positions.",
        cell_missing: "The lattice is missing: paste a POSCAR or fill in a and the lattice vectors.",
        lattice_bad: "The lattice needs a positive a and three rows of three numbers.",
        lattice_singular: "The lattice vectors are linearly dependent.",
        kd_missing: "The element names (KD) are missing.",
        kd_dup: "The element {0} appears more than once in KD.",
        pos_bad: "Position line {0} must have a species index and three fractional coordinates.",
        pos_kd: "Position line {0}: the species index must be between 1 and {1}.",
        cutoff_missing: "The {1} cutoff for {0} is missing (a number or None).",
        cutoff_bad: "The {1} cutoff for {0} must be a positive number or None.",
        matrix_bad: "{0} must have 1, 3 or 9 entries.",
        dfset_missing: "MODE = optimize needs a DFSET file.",
        fc2fix_norder1: "FC2FIX fixes the harmonic terms of an anharmonic fit; it has no use with NORDER = 1.",
        lasso_nocv: "With LMODEL = {0}, set CV (cross-validation) or L1_ALPHA.",
        fcsfile_missing: "FCSFILE is empty.",
        xml_needs_cell: "An XML FCSFILE has no stored cell: &cell cannot be omitted.",
        path_missing: "The band path is empty.",
        path_bad: "Band path line {0} must have 9 columns: label k1 k2 k3 label k1 k2 k3 N.",
        path_npts: "Band path line {0}: the number of points N must be at least 2.",
        relax_bad: "RELAX_STR = {0} is not supported here (use {1}).",
        mesh_bad: "{0} must be three positive integers.",
        kappa_mesh: "MODE = kappa needs a uniform mesh (KPMODE = 2).",
        kmesh_multiple: "Each entry of {0} must be a multiple of KMESH_INTERPOLATE.",
        fourph_warn: "INCLUDE_4PH = 1 needs quartic force constants in FCSFILE (NORDER = 3 in alm).",
        num_bad: "{0} must be a number.",
        poscar: "POSCAR: {0}",
        poscar_short: "too few lines.",
        poscar_lattice: "the scale factor or the lattice vectors are not numbers.",
        poscar_vasp4: "the element names are missing (VASP 4 format); add them above the atom counts.",
        poscar_counts: "the atom counts do not match the element names.",
        poscar_coords: "an atomic position is not three numbers.",
      },
    },
    ja: {
      program: "プログラム", mode: "モード",
      structure: "結晶構造",
      poscar: "VASP 5 形式 (元素名つき) の POSCAR を貼り付けると、下の欄が埋まります。",
      readPoscar: "POSCAR を読み込む", loadExample: "例を読み込む (Si)",
      scale: "格子定数 a (bohr)",
      vectors: "a を単位とする格子ベクトル (3 行)",
      kd: "元素 (KD)",
      positions: "&position: 元素番号と分率座標 (1 行に 1 原子)",
      poscarRead: "{1} の {0} 原子を読み込みました。",
      interaction: "力の定数",
      cutoff: "カットオフ半径 (bohr) または None (カットオフなし)",
      pair: "ペア", orders: ["2 次", "3 次", "4 次"],
      cutoffHint: "先に元素を入力してください。",
      cells: "セルの変換 (任意)",
      primcellHint: "Auto、または 1, 3, 9 個の値",
      supercellHint: "1, 3, 9 個の整数",
      optimize: "フィッティング (&optimize)",
      forceconst: "力の定数",
      omitCell: "&cell を省略する (FCSFILE に保存されたセルを使う)",
      cellNote: "&cell を与える場合は、基本単位格子かその超格子でなければなりません。",
      kpoints: "k 点 (&kpoint)",
      kpPath: "バンド経路 (KPMODE = 1)", kpMesh: "一様メッシュ (KPMODE = 2)",
      pathHelp: "1 行に 1 区間: ラベル k1 k2 k3 ラベル k1 k2 k3 点数 (逆格子ベクトルを単位とする分率座標)。fcc のプリセットは基本並進ベクトル a/2(0,1,1), a/2(1,0,1), a/2(1,1,0) を仮定しています。",
      preset: "プリセット",
      presets: { "": "—", fcc: "fcc: G-X-G-L", sc: "単純立方: G-X-M-G-R-M" },
      kappa: "熱伝導率 (&kappa)",
      include4ph: "4 フォノン散乱 (INCLUDE_4PH = 1)",
      temps: "温度 (K)",
      scph: "自己無撞着フォノン (&scph)", qha: "準調和近似 (&qha)",
      relaxNote: "セルの緩和 (RELAX_STR = 2, 3) には歪み結合のデータ (&strain, STRAINFILE) が必要です。anphon の入力変数の説明に従って手で書いてください。",
      output: "生成された入力ファイル", copy: "コピー", copied: "コピーしました", download: "ダウンロード",
      ok: "問題は見つかりませんでした。",
      errors: {
        prefix_missing: "PREFIX が空です。",
        struct_missing: "構造がありません。POSCAR を貼り付けるか、格子・元素・原子位置を入力してください。",
        cell_missing: "格子がありません。POSCAR を貼り付けるか、a と格子ベクトルを入力してください。",
        lattice_bad: "格子には正の a と、3 つの数からなる 3 行が必要です。",
        lattice_singular: "格子ベクトルが一次従属です。",
        kd_missing: "元素名 (KD) がありません。",
        kd_dup: "元素 {0} が KD に 2 回以上現れています。",
        pos_bad: "原子位置の {0} 行目には、元素番号と 3 つの分率座標が必要です。",
        pos_kd: "原子位置の {0} 行目: 元素番号は 1 から {1} の間でなければなりません。",
        cutoff_missing: "{0} の {1} のカットオフがありません (数値または None)。",
        cutoff_bad: "{0} の {1} のカットオフは正の数か None でなければなりません。",
        matrix_bad: "{0} は 1, 3, 9 個の値でなければなりません。",
        dfset_missing: "MODE = optimize には DFSET ファイルが必要です。",
        fc2fix_norder1: "FC2FIX は非調和フィットで調和項を固定するためのもので、NORDER = 1 では使いません。",
        lasso_nocv: "LMODEL = {0} では CV (交差検証) か L1_ALPHA を設定してください。",
        fcsfile_missing: "FCSFILE が空です。",
        xml_needs_cell: "XML 形式の FCSFILE にはセルが保存されていないため、&cell は省略できません。",
        path_missing: "バンド経路が空です。",
        path_bad: "バンド経路の {0} 行目は 9 列 (ラベル k1 k2 k3 ラベル k1 k2 k3 点数) でなければなりません。",
        path_npts: "バンド経路の {0} 行目: 点数 N は 2 以上でなければなりません。",
        relax_bad: "RELAX_STR = {0} はここでは使えません ({1} を使ってください)。",
        mesh_bad: "{0} は 3 つの正の整数でなければなりません。",
        kappa_mesh: "MODE = kappa には一様メッシュ (KPMODE = 2) が必要です。",
        kmesh_multiple: "{0} の各値は KMESH_INTERPOLATE の倍数でなければなりません。",
        fourph_warn: "INCLUDE_4PH = 1 には FCSFILE に 4 次の力の定数が必要です (alm で NORDER = 3)。",
        num_bad: "{0} は数値でなければなりません。",
        poscar: "POSCAR: {0}",
        poscar_short: "行数が足りません。",
        poscar_lattice: "スケール因子か格子ベクトルが数値ではありません。",
        poscar_vasp4: "元素名がありません (VASP 4 形式)。原子数の行の上に元素名を追加してください。",
        poscar_counts: "原子数の数が元素名の数と一致しません。",
        poscar_coords: "原子位置が 3 つの数値になっていません。",
      },
    },
  };

  // Band paths from the tutorial inputs (example/Si/si_phband.in, example/SrTiO3/reference/phband.in).
  const PRESETS = {
    fcc: "G 0.0 0.0 0.0 X 0.5 0.5 0.0 51\nX 0.5 0.5 1.0 G 0.0 0.0 0.0 51\nG 0.0 0.0 0.0 L 0.5 0.5 0.5 51",
    sc: "G 0.0 0.0 0.0 X 0.5 0.0 0.0 51\nX 0.5 0.0 0.0 M 0.5 0.5 0.0 51\nM 0.5 0.5 0.0 G 0.0 0.0 0.0 51\n" +
        "G 0.0 0.0 0.0 R 0.5 0.5 0.5 51\nR 0.5 0.5 0.5 M 0.5 0.5 0.0 51",
  };

  // -- pure helpers -----------------------------------------------------------

  // RELAX_STR = 2, 3 need &strain / strain-coupling data that the builder does not write.
  const RELAX_OPTIONS = { SCPH: ["0", "1"], QHA: ["1"] };

  const words = (s) => String(s || "").trim().split(/\s+/).filter(Boolean);
  const isNum = (s) => s !== "" && isFinite(Number(s));
  const stripComment = (s) => s.replace(/[#!].*$/, "").trim();
  const nonEmptyLines = (s) => String(s || "").split(/\r?\n/).map(stripComment).filter(Boolean);
  const f10 = (x) => (x < 0 ? "" : " ") + x.toFixed(10);

  function det3(m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  function inv3(m) {
    const d = det3(m);
    const c = (i, j) => {
      const r = [0, 1, 2].filter((k) => k !== i), s = [0, 1, 2].filter((k) => k !== j);
      return ((i + j) % 2 ? -1 : 1) * (m[r[0]][s[0]] * m[r[1]][s[1]] - m[r[0]][s[1]] * m[r[1]][s[0]]);
    };
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / d));
  }

  // VASP 5 POSCAR -> fields of the structure form (lattice in bohr).
  function parsePoscar(text) {
    const L = String(text).replace(/\s+$/, "").split(/\r?\n/).map((s) => s.trim());
    if (L.length < 8) throw new Error("poscar_short");
    const nums = (s) => words(s).map(Number);
    // One scale (negative = target volume) or three per-axis factors, as in VASP.
    const w1 = words(L[1]), iend = w1.findIndex((x) => !isNum(x));
    const sc = w1.slice(0, iend < 0 ? w1.length : iend).map(Number);
    const lat = [2, 3, 4].map((i) => nums(L[i]).slice(0, 3));
    if (sc.length === 0 || (sc.length >= 3 ? sc.slice(0, 3).some((x) => !(x > 0)) : sc[0] === 0) ||
        lat.some((r) => r.length < 3 || r.some((x) => !isFinite(x)))) {
      throw new Error("poscar_lattice");
    }
    const names = words(L[5]).map((s) => s.split(/[_/]/)[0]);
    const species = [...new Set(names)]; // "Si Si" -> one species
    if (species.every((s) => /^\d+$/.test(s))) throw new Error("poscar_vasp4");
    const counts = nums(L[6]);
    if (counts.length !== names.length || counts.some((n) => !Number.isInteger(n) || n < 1)) {
      throw new Error("poscar_counts");
    }
    let i = 7;
    if (/^s/i.test(L[i])) i++; // Selective dynamics
    const cartesian = /^[ck]/i.test(L[i]);
    i++;
    const factor = sc.length >= 3 ? 1 : sc[0] > 0 ? sc[0] : Math.cbrt(-sc[0] / Math.abs(det3(lat)));
    const axis = sc.length >= 3 ? sc.slice(0, 3) : [factor, factor, factor];
    const latA = lat.map((r) => r.map((x, j) => x * axis[j])); // angstrom
    const inv = inv3(latA);
    const positions = [];
    counts.forEach((n, ikd) => {
      for (let k = 0; k < n; k++, i++) {
        let x = nums(L[i] || "").slice(0, 3);
        if (x.length < 3 || x.some((v) => !isFinite(v))) throw new Error("poscar_coords");
        if (cartesian) {
          x = x.map((v, j) => v * axis[j]);
          x = [0, 1, 2].map((j) => x[0] * inv[0][j] + x[1] * inv[1][j] + x[2] * inv[2][j]);
        }
        positions.push([species.indexOf(names[ikd]) + 1].concat(x));
      }
    });
    return {
      scale: (factor / BOHR_IN_ANGSTROM).toFixed(10),
      lattice: (sc.length >= 3 ? latA : lat).map((r) => r.map(f10).join(" ")).join("\n"),
      kd: species.join(" "),
      positions: positions.map((p) => p[0] + " " + p.slice(1).map(f10).join(" ")).join("\n"),
    };
  }

  function cutoffPairs(kd) {
    const pairs = [];
    for (let i = 0; i < kd.length; i++) for (let j = i; j < kd.length; j++) pairs.push(kd[i] + "-" + kd[j]);
    return pairs;
  }

  // Settings object (all values are strings or booleans, as read from the form)
  // -> { text, filename, errors: [[key, ...args]], warnings: [...] }.
  function generateInput(s) {
    const errors = [], warnings = [];
    const err = (...a) => errors.push(a), warn = (...a) => warnings.push(a);
    const alm = s.program === "alm";
    const out = [];
    const block = (name, lines) => out.push("&" + name, ...lines.map((l) => "  " + l), "/", "");
    const prefix = String(s.prefix || "").trim();
    if (!prefix) err("prefix_missing");

    // structure
    const needCell = alm || !s.omitCell;
    let cell = null, kd = words(s.kd), positions = [];
    const latRows = nonEmptyLines(s.lattice).map(words);
    const scaleGiven = String(s.scale || "").trim() !== "";
    if (needCell) {
      if (!scaleGiven && latRows.length === 0) {
        err(alm ? "struct_missing" : "cell_missing");
      } else if (!(Number(s.scale) > 0) || latRows.length !== 3 || latRows.some((r) => r.length !== 3 || !r.every(isNum))) {
        err("lattice_bad");
      } else if (Math.abs(det3(latRows.map((r) => r.map(Number)))) < 1e-8) {
        err("lattice_singular");
      } else {
        cell = [String(s.scale).trim()].concat(latRows.map((r) => r.join(" ")));
      }
    }
    if (alm) {
      if (kd.length === 0) err("kd_missing");
      kd.filter((k, i) => kd.indexOf(k) !== i).forEach((k) => err("kd_dup", k));
      nonEmptyLines(s.positions).forEach((line, i) => {
        const w = words(line);
        if (w.length < 4 || !w.slice(0, 4).every(isNum)) return err("pos_bad", i + 1);
        const ikd = Number(w[0]);
        if (!Number.isInteger(ikd) || ikd < 1 || ikd > kd.length) return err("pos_kd", i + 1, kd.length);
        positions.push(w.slice(0, 4).join(" "));
      });
      if (positions.length === 0 && !errors.some((e) => e[0] === "struct_missing")) err("struct_missing");
    }

    const general = ["PREFIX = " + prefix, "MODE = " + s.mode];
    if (alm) {
      general.push("NAT = " + positions.length + "; NKD = " + kd.length, "KD = " + kd.join(" "));
      for (const tag of ["PRIMCELL", "SUPERCELL"]) {
        const v = words(s[tag.toLowerCase()]);
        if (v.length === 0) continue;
        const ok = tag === "PRIMCELL"
          ? (v.length === 1 && /^a(uto)?$/i.test(v[0])) || ([1, 3, 9].includes(v.length) && v.every((x) => /^-?\d+(\.\d*)?(\/\d+)?$/.test(x)))
          : [1, 3, 9].includes(v.length) && v.every((x) => /^-?\d+$/.test(x));
        if (!ok) err("matrix_bad", tag);
        general.push(tag + " = " + v.join(" "));
      }
      block("general", general);

      const norder = Number(s.norder) || 1;
      block("interaction", ["NORDER = " + norder]);
      const cut = [];
      for (const pair of cutoffPairs(kd)) {
        const vals = (s.cutoffs && s.cutoffs[pair]) || [];
        const row = [];
        for (let o = 0; o < norder; o++) {
          const v = String(vals[o] || "").trim();
          if (v === "") err("cutoff_missing", pair, o);
          else if (!/^none$/i.test(v) && !(Number(v) > 0)) err("cutoff_bad", pair, o);
          row.push(/^none$/i.test(v) ? "None" : v || "?");
        }
        cut.push(pair + " " + row.join(" "));
      }
      block("cutoff", cut);
      if (cell) block("cell", cell);
      block("position", positions);

      if (s.mode === "optimize") {
        const opt = [];
        const dfset = String(s.dfset || "").trim();
        if (!dfset) err("dfset_missing");
        opt.push("DFSET = " + dfset);
        const lmodel = s.lmodel || "ols";
        if (lmodel !== "ols") {
          opt.push("LMODEL = " + lmodel);
          const cv = String(s.cv || "").trim(), alpha = String(s.l1alpha || "").trim();
          if (cv) opt.push("CV = " + cv);
          if (alpha) opt.push("L1_ALPHA = " + alpha);
          if (!cv && !alpha) warn("lasso_nocv", lmodel);
          if (cv && !/^-?\d+$/.test(cv)) err("num_bad", "CV");
          if (alpha && !isNum(alpha)) err("num_bad", "L1_ALPHA");
        }
        const fc2fix = String(s.fc2fix || "").trim();
        if (fc2fix) {
          opt.push("FC2FIX = " + fc2fix);
          if (norder === 1) warn("fc2fix_norder1");
        }
        block("optimize", opt);
      }
    } else {
      const fcsfile = String(s.fcsfile || "").trim();
      if (!fcsfile) err("fcsfile_missing");
      general.push("FCSFILE = " + fcsfile);
      if (s.omitCell && /\.xml$/i.test(fcsfile)) err("xml_needs_cell");
      const thermal = ["kappa", "SCPH", "QHA"].includes(s.mode);
      if (thermal) {
        const temps = ["TMIN", "TMAX", "DT"].filter((t) => String(s[t.toLowerCase()] || "").trim() !== "");
        temps.forEach((t) => { if (!isNum(String(s[t.toLowerCase()]).trim())) err("num_bad", t); });
        if (temps.length) general.push(temps.map((t) => t + " = " + String(s[t.toLowerCase()]).trim()).join("; "));
      }
      block("general", general);
      if (cell) block("cell", cell);

      const mesh = (tag, v) => {
        const w = words(v);
        if (w.length !== 3 || !w.every((x) => /^\d+$/.test(x) && Number(x) > 0)) { err("mesh_bad", tag); return null; }
        return w.map(Number);
      };
      if (s.mode === "SCPH" || s.mode === "QHA") {
        const tagMesh = s.mode === "SCPH" ? "KMESH_SCPH" : "KMESH_QHA";
        const ki = mesh("KMESH_INTERPOLATE", s.kmeshInterp), km = mesh(tagMesh, s.kmeshScf);
        if (ki && km && km.some((k, i) => k % ki[i] !== 0)) err("kmesh_multiple", tagMesh);
        const lines = ["KMESH_INTERPOLATE = " + words(s.kmeshInterp).join(" "), tagMesh + " = " + words(s.kmeshScf).join(" ")];
        const relaxDefault = s.mode === "QHA" ? "1" : "0";
        const relax = String(s.relaxStr || relaxDefault);
        const relaxOk = RELAX_OPTIONS[s.mode];
        if (!relaxOk.includes(relax)) err("relax_bad", relax, relaxOk.join(", "));
        if (relax !== relaxDefault) lines.push("RELAX_STR = " + relax);
        block(s.mode === "SCPH" ? "scph" : "qha", lines);
        if (relax !== "0") block("relax", []); // required when RELAX_STR != 0; all tags have defaults
      }

      const kp = [s.kpmode === "1" ? "1" : "2"];
      if (s.mode === "kappa" && kp[0] !== "2") err("kappa_mesh");
      if (kp[0] === "1") {
        const lines = nonEmptyLines(s.path);
        if (lines.length === 0) err("path_missing");
        lines.forEach((line, i) => {
          const w = words(line);
          const ok = w.length === 9 && [1, 2, 3, 5, 6, 7].every((j) => isNum(w[j])) && /^\d+$/.test(w[8]);
          if (!ok) err("path_bad", i + 1);
          else if (Number(w[8]) < 2) err("path_npts", i + 1); // anphon/kpoint.cpp
          kp.push(w.join(" "));
        });
      } else if (mesh("&kpoint", s.mesh)) {
        kp.push(words(s.mesh).join(" "));
      }
      block("kpoint", kp);

      if (s.mode === "kappa") {
        const kap = [];
        if (s.solver && s.solver !== "RTA") kap.push("SOLVER = " + s.solver);
        if (s.include4ph) {
          kap.push("INCLUDE_4PH = 1");
          warn("fourph_warn");
          if (words(s.kmeshCoarse).length) {
            mesh("KMESH_COARSE", s.kmeshCoarse);
            kap.push("KMESH_COARSE = " + words(s.kmeshCoarse).join(" "));
          }
        }
        if (kap.length) block("kappa", kap);
      }
    }

    return {
      text: out.join("\n"),
      filename: (prefix ? prefix + "_" : "") + s.program + ".in",
      errors, warnings,
    };
  }

  function format(lang, e) {
    const L = LABELS[lang] || LABELS.en;
    const args = e.slice(1).map((a, i) => (e[0].startsWith("cutoff") && i === 1 ? L.orders[a].toLowerCase() : a));
    const tmpl = L.errors[e[0]] || LABELS.en.errors[e[0]] || e[0];
    return tmpl.replace(/\{(\d)\}/g, (_, i) => args[i]);
  }

  const api = { parsePoscar, generateInput, cutoffPairs, format, PRESETS, RELAX_OPTIONS, BOHR_IN_ANGSTROM };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // -- browser UI ---------------------------------------------------------------

  const SI_EXAMPLE = [
    "Si conventional cell", "5.431",
    "1.0 0.0 0.0", "0.0 1.0 0.0", "0.0 0.0 1.0",
    "Si", "8", "Direct",
    "0.00 0.00 0.00", "0.00 0.50 0.50", "0.50 0.00 0.50", "0.50 0.50 0.00",
    "0.25 0.25 0.25", "0.25 0.75 0.75", "0.75 0.25 0.75", "0.75 0.75 0.25",
  ].join("\n");

  function init(root) {
    const lang = (document.documentElement.lang || "en").toLowerCase().startsWith("ja") ? "ja" : "en";
    const T = LABELS[lang];
    const cutoffCache = {};

    const field = (id, label, attrs = "", show = "") =>
      `<label class="aib-field"${show ? ` data-show="${show}"` : ""}><span>${label}</span><input id="ib-${id}" ${attrs}></label>`;
    const select = (id, label, opts, show = "") =>
      `<label class="aib-field"${show ? ` data-show="${show}"` : ""}><span>${label}</span><select id="ib-${id}">` +
      opts.map((o) => (Array.isArray(o) ? `<option value="${o[0]}">${o[1]}</option>` : `<option>${o}</option>`)).join("") +
      "</select></label>";
    // For static label text in the form template only; user text goes through the DOM (textContent, .value).
    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    const el = (tag, props = {}, ...kids) => {
      const e = Object.assign(document.createElement(tag), props);
      e.append(...kids);
      return e;
    };

    root.innerHTML = `
<div class="aib-grid">
<form class="aib-form" onsubmit="return false">
  <fieldset><legend>${T.program}</legend>
    <div class="aib-row">
      <label class="aib-radio"><input type="radio" name="ib-program" value="alm" checked> alm</label>
      <label class="aib-radio"><input type="radio" name="ib-program" value="anphon"> anphon</label>
      <button type="button" id="ib-example" class="aib-button">${T.loadExample}</button>
    </div>
    <div class="aib-row">
      ${field("prefix", "PREFIX", 'value="" placeholder="si222" spellcheck="false"')}
      ${select("mode", "MODE", [])}
    </div>
  </fieldset>

  <fieldset data-show="anphon"><legend>${T.forceconst}</legend>
    <div class="aib-row">${field("fcsfile", "FCSFILE", 'placeholder="si222.h5" spellcheck="false"')}</div>
    <label class="aib-check"><input type="checkbox" id="ib-omitcell" checked> ${esc(T.omitCell)}</label>
    <p class="aib-note" data-show="cell">${esc(T.cellNote)}</p>
  </fieldset>

  <fieldset data-show="cell"><legend>${T.structure}</legend>
    <p class="aib-note">${T.poscar}</p>
    <textarea wrap="off" id="ib-poscar" rows="5" spellcheck="false" placeholder="Si&#10;5.431&#10;..."></textarea>
    <div class="aib-row"><button type="button" id="ib-readposcar" class="aib-button">${T.readPoscar}</button>
      <span id="ib-poscarmsg" class="aib-note"></span></div>
    <div class="aib-row">
      ${field("scale", T.scale, 'spellcheck="false"')}
      ${field("kd", T.kd, 'placeholder="Si" spellcheck="false"', "pos")}
    </div>
    <label class="aib-field aib-wide"><span>${T.vectors}</span><textarea wrap="off" id="ib-lattice" rows="3" spellcheck="false"></textarea></label>
    <label class="aib-field aib-wide" data-show="pos"><span>${esc(T.positions)}</span><textarea wrap="off" id="ib-positions" rows="5" spellcheck="false"></textarea></label>
  </fieldset>

  <fieldset data-show="alm"><legend>${T.interaction}</legend>
    <div class="aib-row">${select("norder", "NORDER", ["1", "2", "3"])}</div>
    <p class="aib-note">${T.cutoff}</p>
    <div id="ib-cutoffs"></div>
    <p class="aib-note">${T.cells}</p>
    <div class="aib-row">
      ${field("primcell", "PRIMCELL", `list="ib-primcell-list" placeholder="${T.primcellHint}" spellcheck="false"`)}
      <datalist id="ib-primcell-list"><option value="Auto"></option></datalist>
      ${field("supercell", "SUPERCELL", `placeholder="${T.supercellHint}" spellcheck="false"`)}
    </div>
  </fieldset>

  <fieldset data-show="alm:optimize"><legend>${esc(T.optimize)}</legend>
    <div class="aib-row">
      ${field("dfset", "DFSET", 'placeholder="DFSET_harmonic" spellcheck="false"')}
      ${select("lmodel", "LMODEL", ["ols", "enet", "adaptive-lasso"])}
    </div>
    <div class="aib-row" data-show="lasso">
      ${field("cv", "CV", 'placeholder="4" spellcheck="false"')}
      ${field("l1alpha", "L1_ALPHA", 'spellcheck="false"')}
    </div>
    <div class="aib-row">${field("fc2fix", "FC2FIX", 'placeholder="si222.h5" spellcheck="false"')}</div>
  </fieldset>

  <fieldset data-show="anphon"><legend>${esc(T.kpoints)}</legend>
    <div class="aib-row">
      <label class="aib-radio"><input type="radio" name="ib-kpmode" value="1" checked> ${T.kpPath}</label>
      <label class="aib-radio"><input type="radio" name="ib-kpmode" value="2"> ${T.kpMesh}</label>
    </div>
    <div data-show="kp1">
      <div class="aib-row">${select("preset", T.preset, Object.entries(T.presets))}</div>
      <p class="aib-note">${T.pathHelp}</p>
      <textarea wrap="off" id="ib-path" rows="4" spellcheck="false"></textarea>
    </div>
    <div class="aib-row" data-show="kp2">${field("mesh", "k1 k2 k3", 'value="20 20 20" spellcheck="false"')}</div>
  </fieldset>

  <fieldset data-show="anphon:kappa"><legend>${esc(T.kappa)}</legend>
    <div class="aib-row">${select("solver", "SOLVER", ["RTA", "IBTE", "VBTE"])}</div>
    <label class="aib-check"><input type="checkbox" id="ib-include4ph"> ${T.include4ph}</label>
    <div class="aib-row" data-show="fourph">${field("kmeshcoarse", "KMESH_COARSE", 'placeholder="10 10 10" spellcheck="false"')}</div>
  </fieldset>

  <fieldset data-show="anphon:SCPH anphon:QHA"><legend id="ib-scflegend"></legend>
    <div class="aib-row">
      ${field("kmeshinterp", "KMESH_INTERPOLATE", 'value="4 4 4" spellcheck="false"')}
      ${field("kmeshscf", "KMESH_SCPH", 'value="8 8 8" spellcheck="false"')}
      ${select("relaxstr", "RELAX_STR", [])}
    </div>
    <p class="aib-note">${esc(T.relaxNote)}</p>
  </fieldset>

  <fieldset data-show="anphon:kappa anphon:SCPH anphon:QHA"><legend>${T.temps}</legend>
    <div class="aib-row">
      ${field("tmin", "TMIN", 'placeholder="0" size="6"')}
      ${field("tmax", "TMAX", 'placeholder="1000" size="6"')}
      ${field("dt", "DT", 'placeholder="10" size="6"')}
    </div>
  </fieldset>
</form>

<div class="aib-output">
  <div class="aib-outhead"><strong>${T.output}</strong>
    <span><button type="button" id="ib-copy" class="aib-button">${T.copy}</button>
    <button type="button" id="ib-download" class="aib-button">${T.download}</button></span></div>
  <ul id="ib-messages" class="aib-messages"></ul>
  <pre id="ib-text" class="aib-pre"></pre>
</div>
</div>`;

    const $ = (id) => root.querySelector("#ib-" + id);
    const val = (id) => $(id).value;
    const radio = (name) => root.querySelector(`input[name="ib-${name}"]:checked`).value;
    const setOptions = (sel, opts, keep) => {
      const old = sel.value;
      sel.replaceChildren(...opts.map((o) => new Option(o)));
      if (keep && opts.includes(old)) sel.value = old;
    };
    let lastProgram = null, lastMode = null, current = null;

    function settings() {
      const kd = words(val("kd"));
      const cutoffs = {};
      root.querySelectorAll("#ib-cutoffs input").forEach((inp) => {
        (cutoffs[inp.dataset.pair] = cutoffs[inp.dataset.pair] || [])[Number(inp.dataset.order)] = inp.value;
      });
      return {
        program: radio("program"), prefix: val("prefix"), mode: val("mode"),
        scale: val("scale"), lattice: val("lattice"), kd: kd.join(" "), positions: val("positions"),
        norder: val("norder"), cutoffs, primcell: val("primcell"), supercell: val("supercell"),
        dfset: val("dfset"), lmodel: val("lmodel"), cv: val("cv"), l1alpha: val("l1alpha"), fc2fix: val("fc2fix"),
        fcsfile: val("fcsfile"), omitCell: $("omitcell").checked, kpmode: radio("kpmode"), path: val("path"),
        mesh: val("mesh"), solver: val("solver"), include4ph: $("include4ph").checked, kmeshCoarse: val("kmeshcoarse"),
        kmeshInterp: val("kmeshinterp"), kmeshScf: val("kmeshscf"), relaxStr: val("relaxstr"),
        tmin: val("tmin"), tmax: val("tmax"), dt: val("dt"),
      };
    }

    function renderCutoffs() {
      root.querySelectorAll("#ib-cutoffs input").forEach((inp) => {
        (cutoffCache[inp.dataset.pair] = cutoffCache[inp.dataset.pair] || [])[inp.dataset.order] = inp.value;
      });
      const pairs = cutoffPairs(words(val("kd")));
      const norder = Number(val("norder"));
      if (pairs.length === 0) {
        $("cutoffs").replaceChildren(el("p", { className: "aib-note", textContent: T.cutoffHint }));
        return;
      }
      const orders = T.orders.slice(0, norder);
      const head = el("tr", {}, el("th", { textContent: T.pair }), ...orders.map((o) => el("th", { textContent: o })));
      const rows = pairs.map((p) => el("tr", {}, el("td", {}, el("code", { textContent: p })),
        ...orders.map((_, o) => {
          const inp = el("input", { value: (cutoffCache[p] || [])[o] ?? (o === 0 ? "None" : ""), size: 7, spellcheck: false });
          Object.assign(inp.dataset, { pair: p, order: o });
          return el("td", {}, inp);
        })));
      $("cutoffs").replaceChildren(el("table", { className: "aib-cutoffs" }, head, ...rows));
    }

    function update() {
      const program = radio("program");
      if (program !== lastProgram) {
        setOptions($("mode"), program === "alm" ? ["suggest", "optimize"] : ["phonons", "kappa", "SCPH", "QHA"], false);
        lastProgram = program;
      }
      const mode = val("mode");
      if (mode !== lastMode) {
        if (mode === "kappa") root.querySelector('input[name="ib-kpmode"][value="2"]').checked = true;
        if (mode === "SCPH" || mode === "QHA") {
          setOptions($("relaxstr"), RELAX_OPTIONS[mode], false);
          $("scflegend").textContent = mode === "SCPH" ? T.scph : T.qha;
          $("kmeshscf").previousElementSibling.textContent = mode === "SCPH" ? "KMESH_SCPH" : "KMESH_QHA";
        }
        lastMode = mode;
      }
      const s = settings();
      const active = new Set([program, program + ":" + mode]);
      if (program === "alm" || !s.omitCell) active.add("cell");
      if (program === "alm") active.add("pos");
      if (program === "alm" && s.lmodel !== "ols") active.add("lasso");
      if (s.include4ph) active.add("fourph");
      active.add("kp" + s.kpmode);
      root.querySelectorAll("[data-show]").forEach((el) => {
        el.hidden = !el.dataset.show.split(" ").some((c) => active.has(c));
      });

      current = generateInput(s);
      $("text").textContent = current.text;
      const items = current.errors.map((e) => ["aib-error", e]).concat(current.warnings.map((e) => ["aib-warning", e]));
      $("messages").replaceChildren(...(items.length
        ? items.map(([cls, e]) => el("li", { className: cls, textContent: format(lang, e) }))
        : [el("li", { className: "aib-ok", textContent: T.ok })]));
    }

    function readPoscar() {
      const text = val("poscar");
      if (!text.trim()) return;
      try {
        const p = parsePoscar(text);
        $("scale").value = p.scale;
        $("lattice").value = p.lattice;
        $("kd").value = p.kd;
        $("positions").value = p.positions;
        $("poscarmsg").textContent = T.poscarRead.replace("{0}", p.positions.split("\n").length).replace("{1}", p.kd);
        $("poscarmsg").className = "aib-note";
      } catch (e) {
        $("poscarmsg").textContent = format(lang, ["poscar", format(lang, [e.message])]);
        $("poscarmsg").className = "aib-note aib-error";
      }
      renderCutoffs();
      update();
    }

    function loadExample() {
      root.querySelector('input[name="ib-program"][value="alm"]').checked = true;
      update();
      $("prefix").value = "si222";
      $("mode").value = "suggest";
      $("poscar").value = SI_EXAMPLE;
      $("norder").value = "2";
      $("primcell").value = "Auto";
      $("supercell").value = "2 2 2";
      cutoffCache["Si-Si"] = ["None", "7.3"];
      $("fcsfile").value = "si222.h5";
      $("path").value = PRESETS.fcc;
      readPoscar();
    }

    root.addEventListener("input", (ev) => {
      if (ev.target.id === "ib-kd") renderCutoffs();
      if (ev.target.id === "ib-fcsfile") $("omitcell").checked = /\.h5$/i.test(ev.target.value.trim()) || !ev.target.value.trim();
      update();
    });
    root.addEventListener("change", (ev) => {
      if (ev.target.id === "ib-norder") renderCutoffs();
      if (ev.target.id === "ib-preset" && ev.target.value) $("path").value = PRESETS[ev.target.value];
      if (ev.target.id === "ib-poscar") return readPoscar();
      update();
    });
    $("readposcar").addEventListener("click", readPoscar);
    $("example").addEventListener("click", loadExample);
    $("copy").addEventListener("click", () => {
      const done = () => {
        $("copy").textContent = T.copied;
        setTimeout(() => { $("copy").textContent = T.copy; }, 1500);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(current.text).then(done);
      } else {
        const range = document.createRange();
        range.selectNodeContents($("text"));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        if (document.execCommand("copy")) done();
      }
    });
    $("download").addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([current.text], { type: "text/plain" }));
      a.download = current.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });

    renderCutoffs();
    update();
    const q = new URLSearchParams(location.search);
    if (q.get("example") === "si") loadExample();
    if (q.get("program") === "anphon") {
      root.querySelector('input[name="ib-program"][value="anphon"]').checked = true;
      update();
    }
  }

  const start = () => document.querySelectorAll(".alamode-input-builder").forEach(init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
