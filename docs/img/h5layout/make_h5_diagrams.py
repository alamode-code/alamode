#!/usr/bin/env python3
"""Bilingual, transparent HDF5 layout figures (about 700 px wide in HTML).

DOT=/path/to/dot PDFTOCAIRO=/path/to/pdftocairo python3 make_h5_diagrams.py

Identifiers, dimensions and units were checked with h5py against:
  test/fourph/cBTO222.h5, bto_ismear0.kappa.h5
  test/si_units/si222u_a.h5
  test/kappa_restart/si222_ibte.kappa.h5
  test/scph_h5/cBTO222_scph.scph.h5, defo_1.kappa.h5, kbto.kappa.h5
  test/qha/ZnO_h5c2.qha.h5, strain_IFC/ZnO.strain.h5
  test/mode_analysis/si/si_se_h5.selfenergy.h5
Additional kappa datasets are grounded in anphon/kappa_result_io.cpp
(create_kappa_group), as requested: coherent/total/spec/energy_axis.
These are representative layouts, not a claim that all optional branches coexist.
The supplied dump.py collapses numbered targets and input-variable attributes.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "flowcharts"))
from make_flowcharts import L, THEMES, FONT, STYLE, label, render_source  # noqa: E402


def row(path, en, ja=None, condition=None, kind="data"):
    return dict(
        path=path,
        text=L(en, ja if ja is not None else en),
        condition=condition,
        kind=kind,
    )


COMMON = "@ format_version, alamode_version\n@ alamode_git_commit, created_date"
CELL = (
    "lattice_vector (3, 3) [bohr]\nfractional_coordinate (ncell, 3)\n"
    "atomic_kinds (ncell,) · elements (nelem,)\nmapping_table (nprim, ntrans)\n"
    "number_of_atoms () · number_of_elements ()\nnumber_of_primitive_translations ()\nspin_polarized ()"
)
ORDER = (
    "force_constant_values (N,) [Ry/bohr^n]\n"
    "atom_indices (N, n)\natom_indices_supercell (N, n)\n"
    "coord_indices (N, n)\nshift_vectors (N, 3(n−1)) [bohr]\n@ basis = Cartesian"
)

FCS = dict(
    title=L("Force constants · PREFIX.h5", "力定数 · PREFIX.h5"),
    root=L(
        "/\n@ schema = alamode:force_constants\n" + COMMON,
        "/\n@ schema = alamode:force_constants\n" + COMMON,
    ),
    rows=[
        row(
            "/ForceConstants/\nOrder2",
            ORDER + "\nn = 2; N = entries in this order",
            ORDER + "\nn = 2; N = この次数の項数",
        ),
        row(
            "/ForceConstants/\nOrder3, Order4",
            "Same layout as Order2; n = 3, 4\nEach order has its own N",
            "Order2 と同じ構成; n = 3, 4\nN は次数ごとに異なる",
            L("if higher orders are fitted", "高次の力定数を求めた場合"),
        ),
        row(
            "/PrimitiveCell\n/SuperCell",
            CELL + "\nSame dataset names in both groups",
            CELL + "\n両グループに同名のデータセット",
        ),
        row(
            "/metadata/\ninput_variables",
            "@ input tags and their values",
            "@ 入力タグと設定値",
            kind="attrs",
        ),
        row(
            "/",
            "version ()\ncreated date ()\nLegacy string datasets",
            "version ()\ncreated date ()\n従来形式の文字列データセット",
        ),
    ],
    note=L(
        "N: IFC entries · n: IFC order · nelem: elements\nncell: atoms in that cell · nprim: mapping rows · ntrans: translations",
        "N: 力定数の項数 · n: 力定数の次数 · nelem: 元素数\nncell: セル内の原子数 · nprim: 対応表の行数 · ntrans: 並進数",
    ),
)

KAPPA = dict(
    title=L("Thermal conductivity · PREFIX.kappa.h5", "熱伝導率 · PREFIX.kappa.h5"),
    root=L(
        "/\n@ schema = alamode:kappa_result\n"
        + COMMON
        + "\n@ temperature_resolved = 0 / 1",
        "/\n@ schema = alamode:kappa_result\n"
        + COMMON
        + "\n@ temperature_resolved = 0 / 1",
    ),
    rows=[
        row(
            "/metadata",
            "temperatures (nT,) [K] · fcs_file ()\nclassical () · ismear () · isotope ()\nsmearing_width () [cm^-1]\nisotope_factors (nelem,)",
        ),
        row(
            "/metadata/\nPrimitiveCell",
            "lattice_vector (3, 3) [bohr] · volume () [bohr^3]\nfractional_coordinate (natom, 3)\natomic_kinds (natom,) · elements (nelem,)\nnumber_of_atoms () · number_of_elements ()",
        ),
        row(
            "/metadata/\ninput_variables",
            "@ input tags and their values",
            "@ 入力タグと設定値",
            kind="attrs",
        ),
        row(
            "/metadata/\ninput_variables_runs/\n280.00, ...",
            "@ input tags per temperature run\n/metadata also has:\nfc2_source () · fc2_temperatures (nT,) [K]",
            "@ 各温度の計算に用いた入力タグ\n/metadata には以下も保存:\nfc2_source () · fc2_temperatures (nT,) [K]",
            L("FC2_TEMPERATURE runs", "FC2_TEMPERATURE を使う計算"),
            kind="mixed",
        ),
        row(
            "/scattering/3ph",
            "@ kmesh, nk_irred, nbranches\nfrequencies (nk_irred, nbranch) [cm^-1]\ngamma (nk_irred*nbranch, nT) [cm^-1]\ngamma_computed (nk_irred*nbranch,)\nvelocities (nk, nbranch, 3) [m/s]\nvelocity_diad (nk, nbranch, 3, 3) [(m/s)^2]\nxk_irred (nk_irred, 3) · weights (nk_irred,)\nequiv_knum (nk,) · equiv_offsets (nk_irred+1,)",
        ),
        row(
            "/scattering/3ph",
            "When temperature_resolved = 1:\nfrequencies (nT, nk_irred, nbranch) [cm^-1]\nvelocities (nT, nk, nbranch, 3) [m/s]\nvelocity_diad (nT, nk, nbranch, 3, 3) [(m/s)^2]\ngamma_computed (nk_irred*nbranch, nT)\ngamma keeps the shape shown above",
            "temperature_resolved = 1 の場合:\nfrequencies (nT, nk_irred, nbranch) [cm^-1]\nvelocities (nT, nk, nbranch, 3) [m/s]\nvelocity_diad (nT, nk, nbranch, 3, 3) [(m/s)^2]\ngamma_computed (nk_irred*nbranch, nT)\ngamma の形状は上記と同じ",
        ),
        row(
            "/scattering/4ph",
            "Same layout as 3ph at temperature_resolved = 0,\nexcept velocity_diad (not in the 4ph dump)\nOwn kmesh, nk_irred and nbranches",
            "temperature_resolved = 0 の 3ph と同じ構成\nただし 4ph のダンプに velocity_diad はない\nkmesh, nk_irred, nbranches は個別に保持",
            L("INCLUDE_4PH=1", "INCLUDE_4PH=1"),
        ),
        row(
            "/scattering/isotope",
            "gamma (nk_irred, nbranch) [cm^-1]\nWith temperature_resolved = 1:\ngamma (nT, nk_irred, nbranch) [cm^-1]",
            "gamma (nk_irred, nbranch) [cm^-1]\ntemperature_resolved = 1 の場合:\ngamma (nT, nk_irred, nbranch) [cm^-1]",
            L("ISOTOPE>0", "ISOTOPE>0"),
        ),
        row(
            "/kappa",
            "kappa_peierls (nT, 3, 3) [W/mK]\nvalid (nT,) · uint8 if temperature_resolved = 1\nvalid (1,) · uint8 otherwise\n@ formulation, boundary_length\n@ includes_4ph_scattering\n@ includes_isotope_scattering\n@ includes_boundary_scattering",
            "kappa_peierls (nT, 3, 3) [W/mK]\nvalid (nT,) · uint8: temperature_resolved = 1\nそれ以外は valid (1,) · uint8\n@ formulation, boundary_length\n@ includes_4ph_scattering\n@ includes_isotope_scattering\n@ includes_boundary_scattering",
        ),
        row(
            "/kappa",
            "kappa_3ph_only (nT, 3, 3) [W/mK]\n@ scattering_processes = 3ph+isotope",
            condition=L("INCLUDE_4PH=1", "INCLUDE_4PH=1"),
        ),
        row(
            "/kappa",
            "kappa_coherent (nT, 3, 3) [W/mK]\nkappa_total (nT, 3, 3) [W/mK]",
            condition=L("KAPPA_COHERENT>0", "KAPPA_COHERENT>0"),
        ),
        row(
            "/kappa",
            "kappa_spec (n_e, nT, 3) [W/mK/cm^-1]\nenergy_axis (n_e,) [cm^-1]",
            condition=L("KAPPA_SPEC=1", "KAPPA_SPEC=1"),
        ),
        row(
            "/iterativebte",
            "Q (nT, nk_irred, nbranch)\ndF (nT, nk_irred, nbranch, 3) [bohr/K]\nkappa (nT, 3, 3) [W/mK]\ncomputed (nT,) · converged (nT,)\n@ kmesh, nk_irred, nbranches",
            condition=L("SOLVER=IBTE", "SOLVER=IBTE"),
        ),
    ],
    note=L(
        "nT: temperatures · nk: full mesh · nk_irred: irreducible points\nnbranch: branches · natom: cell atoms · nelem: elements · n_e: energy bins",
        "nT: 温度点数 · nk: 全メッシュ点数 · nk_irred: 既約点数\nnbranch: 分枝数 · natom: セル内の原子数 · nelem: 元素数 · n_e: エネルギー点数",
    ),
)

SCPH = dict(
    title=L(
        "Finite-temperature state · PREFIX.scph.h5 / PREFIX.qha.h5",
        "有限温度の状態 · PREFIX.scph.h5 / PREFIX.qha.h5",
    ),
    root=L(
        "/\n@ schema = alamode:scph_state\n"
        + COMMON
        + "\n@ mode = SCPH / QHA · complete",
        "/\n@ schema = alamode:scph_state\n"
        + COMMON
        + "\n@ mode = SCPH / QHA · complete",
    ),
    rows=[
        row(
            "/ForceConstants/\nOrder2",
            ORDER.replace("^n", "^2")
            .replace("(N, n)", "(N, 2)")
            .replace("3(n−1)", "3"),
        ),
        row(
            "/ForceConstants/\nOrder2_temperature_dependent",
            "force_constant_values (nT, N) [Ry/bohr^2]\n@ index_datasets = /ForceConstants/Order2\n@ variant = scph / qha\nShares the index datasets of Order2",
            "force_constant_values (nT, N) [Ry/bohr^2]\n@ index_datasets = /ForceConstants/Order2\n@ variant = scph / qha\nOrder2 のインデックスを共有",
        ),
        row(
            "/PrimitiveCell\n/SuperCell",
            CELL + "\nPrimitiveCell also has masses (natom,) [amu]",
            CELL + "\nPrimitiveCell には masses (natom,) [amu] も保存",
        ),
        row(
            "/settings",
            "temperatures (nT,) [K]\nkmesh_dense (3,) · kmesh_interpolate (3,)\nnonanalytic () · relax_str ()\nselfenergy_offdiag ()",
        ),
        row(
            "/dymat",
            "delta (nT, nbranch, nbranch, nq)\ncomplex128; no unit attribute",
            "delta (nT, nbranch, nbranch, nq)\ncomplex128; unit 属性なし",
        ),
        row(
            "/dymat",
            "delta_harm_renorm (nT, nbranch, nbranch, nq)\ncomplex128; no unit attribute",
            "delta_harm_renorm (nT, nbranch, nbranch, nq)\ncomplex128; unit 属性なし",
            L(
                "relaxed harmonic correction stored",
                "構造変化による調和項の補正を保存した場合",
            ),
        ),
        row(
            "/structure",
            "u0 (nT, natom, 3) [bohr]\nu_tensor (nT, 3, 3) [dimensionless]\nspg_label (nT,)",
            condition=L("RELAX_STR>0", "RELAX_STR>0"),
        ),
        row("/convergence", "scph (nT,)\nConvergence flags", "scph (nT,)\n収束フラグ"),
        row(
            "/convergence",
            "structure (nT,)",
            condition=L(
                "structural convergence stored", "構造の収束判定を保存した場合"
            ),
        ),
        row(
            "/",
            "V0 (nT,) [Ry]",
            condition=L("reference energy stored", "基準エネルギーを保存した場合"),
        ),
        row(
            "/provenance",
            "fcs_nrows (3,) · fcs_sum_abs (3,)\nfcs_sum_signed (3,) · fcs_sum_sq (3,)\nThree entries for IFC orders 2, 3, 4",
            "fcs_nrows (3,) · fcs_sum_abs (3,)\nfcs_sum_signed (3,) · fcs_sum_sq (3,)\n力定数の次数 2, 3, 4 に対応する 3 要素",
            L("when fingerprints are stored", "照合用の情報を保存する場合"),
        ),
        row(
            "/metadata/\ninput_variables",
            "@ input tags and their values",
            "@ 入力タグと設定値",
            kind="attrs",
        ),
    ],
    note=L(
        "nT: temperatures · N: FC2 entries · nq: stored q points · nbranch: branches\nnatom: PrimitiveCell atoms · ncell: cell atoms · nelem: elements\nnprim: mapping rows · ntrans: translations",
        "nT: 温度点数 · N: FC2 の項数 · nq: 保存した q 点数 · nbranch: 分枝数\nnatom: PrimitiveCell の原子数 · ncell: セル内の原子数 · nelem: 元素数\nnprim: 対応表の行数 · ntrans: 並進数",
    ),
)

SELFENERGY = dict(
    title=L(
        "Self-energy · PREFIX.selfenergy.h5", "自己エネルギー · PREFIX.selfenergy.h5"
    ),
    root=L(
        "/\nNo root attributes\nThe format version is /metadata/version",
        "/\nルート属性なし\nフォーマットの版は /metadata/version",
    ),
    rows=[
        row(
            "/metadata",
            "temperatures (nT,) [K]\nkmesh (3,)\nismear () · smearing_width () [cm^-1]\nmode () · target_mode ()\nnumber_of_targets () · version ()",
        ),
        row(
            "/metadata/\ninput_variables",
            "@ input tags and their values",
            "@ 入力タグと設定値",
            kind="attrs",
        ),
        row(
            "/path",
            "kaxis (npath,)\nxk (npath, 3)\nCoordinates along the band path",
            "kaxis (npath,)\nxk (npath, 3)\nバンド経路に沿った座標",
            L("KPMODE = 1", "KPMODE = 1"),
        ),
        row(
            "/targets/\n00001, 00002, ...",
            "branch () · frequency () [cm^-1]\nxk (3,) · on_mesh () · kaxis ()\nOne group per target",
            "branch () · frequency () [cm^-1]\nxk (3,) · on_mesh () · kaxis ()\n対象ごとに 1 グループ",
        ),
        row(
            "/targets/0000N",
            "linewidth (nT,) [cm^-1]\nfull width 2Γ",
            "linewidth (nT,) [cm^-1]\n全幅 2Γ",
            L("LINEWIDTH = 1 or SHIFT = 1", "LINEWIDTH = 1 または SHIFT = 1"),
        ),
        row(
            "/targets/0000N",
            "shift_tadpole (nT,) · shift_bubble (nT,) [cm^-1]\nshift_loop (nT,) with QUARTIC = 1",
            "shift_tadpole (nT,) · shift_bubble (nT,) [cm^-1]\nQUARTIC = 1 では shift_loop (nT,)",
            L("SHIFT = 1", "SHIFT = 1"),
        ),
        row(
            "/targets/0000N",
            "self_omega (nω,) [cm^-1]\nself_real (nT, nω) · self_imag (nT, nω)",
            "self_omega (nω,) [cm^-1]\nself_real (nT, nω) · self_imag (nT, nω)",
            L("SELF_W = 1", "SELF_W = 1"),
        ),
        row(
            "/targets/0000N",
            "fstate_energy (ne,) [cm^-1]\nfstate_absorption (nT, ne) · fstate_emission (nT, ne)",
            "fstate_energy (ne,) [cm^-1]\nfstate_absorption (nT, ne) · fstate_emission (nT, ne)",
            L("FSTATE_W = 1", "FSTATE_W = 1"),
        ),
        row(
            "/spectrum",
            "omega (nω,) · xk (nq, 3) · kaxis (nq,)\nkmesh_coarse (3,)\nT001, T002, ...: temperature ()\ntotal (nq, nω) · branch001, ... (nq, nω)",
            "omega (nω,) · xk (nq, 3) · kaxis (nq,)\nkmesh_coarse (3,)\nT001, T002, ...: temperature ()\ntotal (nq, nω) · branch001, ... (nq, nω)",
            L("INTERPOLATE = 1", "INTERPOLATE = 1"),
        ),
    ],
    note=L(
        "nT: temperatures · npath, nq: points on the path\nnω, ne: frequency points",
        "nT: 温度点数 · npath, nq: 経路上の点数\nnω, ne: 振動数の点数",
    ),
)

OVERVIEW = dict(
    title=L("HDF5 files · writers and readers", "HDF5 ファイル · 出力元と読み込み先"),
    lanes=[
        (
            "alm",
            L("alm\nMODE=optimize", "alm\nMODE=optimize"),
            L("PREFIX.h5\nforce constants", "PREFIX.h5\n力定数"),
            "anphon",
            L("anphon\nFCSFILE", "anphon\nFCSFILE"),
        ),
        (
            "anphon",
            L("anphon\nMODE=phonons\nNEWFCS=1", "anphon\nMODE=phonons\nNEWFCS=1"),
            L(
                "PREFIX_+.h5 / PREFIX_-.h5\nsame force-constant schema",
                "PREFIX_+.h5 / PREFIX_-.h5\n同じ力定数スキーマ",
            ),
            "anphon",
            L("anphon\nFCSFILE", "anphon\nFCSFILE"),
        ),
        (
            "anphon",
            L("anphon\nMODE=kappa", "anphon\nMODE=kappa"),
            L("PREFIX.kappa.h5\nthermal conductivity", "PREFIX.kappa.h5\n熱伝導率"),
            "tool",
            L("analyzer.py\nanalysis", "analyzer.py\n解析"),
        ),
        (
            "anphon",
            L("anphon\nMODE=SCPH / QHA", "anphon\nMODE=SCPH / QHA"),
            L(
                "PREFIX.scph.h5\nPREFIX.qha.h5\nfinite-temperature state",
                "PREFIX.scph.h5\nPREFIX.qha.h5\n有限温度の状態",
            ),
            "anphon",
            L(
                "anphon\nFC2_TEMPERATURE with\nFCSFILE / FC2FILE / DFC2FILE\nRELAXED_STRUCTURE\nrestart",
                "anphon\nFC2_TEMPERATURE と\nFCSFILE / FC2FILE / DFC2FILE\nRELAXED_STRUCTURE\nリスタート",
            ),
        ),
        (
            "anphon",
            L("anphon\nMODE=selfenergy", "anphon\nMODE=selfenergy"),
            L(
                "PREFIX.selfenergy.h5\nself-energy results",
                "PREFIX.selfenergy.h5\n自己エネルギーの結果",
            ),
            None,
            None,
        ),
        (
            "tool",
            L("tools/strainkit\nstrainfile.py", "tools/strainkit\nstrainfile.py"),
            L("PREFIX.strain.h5\nstrain couplings", "PREFIX.strain.h5\nひずみとの結合"),
            "anphon",
            L("anphon\nSTRAINFILE", "anphon\nSTRAINFILE"),
        ),
    ],
)
CHARTS = dict(
    h5_overview=OVERVIEW,
    h5_fcs=FCS,
    h5_kappa=KAPPA,
    h5_scph=SCPH,
    h5_selfenergy=SELFENERGY,
)


def node(nid, text, lang, th, role="file", shape=None, optional=False):
    fill, border = th["roles"][role]
    attrs = dict(STYLE[role])
    if shape:
        attrs.update(shape=shape, style="filled")
    if optional:
        attrs["style"] += ",dashed"
    attrs.update(fillcolor=fill, color=border)
    return "%s [label=%s, %s];" % (
        nid,
        label(text, lang),
        ", ".join('%s="%s"' % item for item in attrs.items()),
    )


def source(name, chart, lang, theme):
    th = THEMES[theme]
    lines = [
        "digraph %s {" % name,
        'graph [bgcolor="transparent", rankdir=TB, nodesep=0.28, ranksep=0.22, pad=0.12, ordering=out];',
        'node [fontname="%s", fontsize=12, fontcolor="%s", margin="0.10,0.06"];'
        % (FONT[lang], th["text"]),
        'edge [color="%s", arrowsize=0.65];' % th["edge"],
        node("title", chart["title"][lang], lang, th, role="anphon", shape="box"),
    ]
    if "lanes" in chart:
        for i, (role, writer, file, reader_role, reader) in enumerate(chart["lanes"]):
            w, f, r = "w%d" % i, "f%d" % i, "r%d" % i
            lines += [
                node(w, writer[lang], lang, th, role),
                node(f, file[lang], lang, th),
            ]
            ids = [w, f]
            lines += ["%s -> %s;" % (w, f)]
            if reader:
                lines += [
                    node(r, reader[lang], lang, th, reader_role),
                    "%s -> %s;" % (f, r),
                ]
                ids.append(r)
            if i == 2:
                lines += [
                    node("restart", "anphon\nRESTART", lang, th, "anphon"),
                    "f2 -> restart;",
                    "r2 -> restart [style=invis];",
                ]
                # Reader nodes stack, keeping three columns at readable width.
                lines += ["{rank=same; w2; f2; r2;}"]
            else:
                lines += ["{rank=same; %s;}" % ";".join(ids)]
            if i:
                lines += ["w%d -> %s [style=invis, weight=100];" % (i - 1, w)]
        lines += ["title -> w0 [style=invis];", "restart -> r3 [style=invis];"]
    else:
        lines += [
            node("root", chart["root"][lang], lang, th, shape="folder"),
            "title -> root [style=invis];",
        ]
        for i, item in enumerate(chart["rows"]):
            g, d = "g%d" % i, "d%d" % i
            group_text = item["path"]
            if item["condition"]:
                group_text += "\n" + item["condition"][lang]
            lines += [
                node(
                    g,
                    group_text,
                    lang,
                    th,
                    role="anphon",
                    shape="folder",
                    optional=bool(item["condition"]),
                ),
                node(
                    d,
                    item["text"][lang],
                    lang,
                    th,
                    shape="note" if item["kind"] == "attrs" else "box",
                    optional=bool(item["condition"]),
                ),
                "{rank=same; %s; %s;}" % (g, d),
                "%s -> %s [arrowhead=none];" % (g, d),
            ]
            if i:
                lines += ["g%d -> %s [arrowhead=none, weight=100];" % (i - 1, g)]
            else:
                lines += ["root -> g0 [arrowhead=none];"]
        legend = L(
            "Folder: group/path · box: datasets · note: attributes\nDashed: optional · @: attribute · (): scalar · [ ]: stored unit",
            "フォルダー: グループ/パス · 四角: データセット · メモ: 属性\n破線: 任意 · @: 属性 · (): スカラー · [ ]: 保存された単位",
        )
        lines += [
            node(
                "legend",
                legend[lang] + "\n" + chart["note"][lang],
                lang,
                th,
                shape="note",
            ),
            "g%d -> legend [style=invis];" % (len(chart["rows"]) - 1),
        ]
    return "\n".join(lines + ["}"])


if __name__ == "__main__":
    for name, chart in CHARTS.items():
        for lang in ("en", "ja"):
            for theme in THEMES:
                render_source(name, source(name, chart, lang, theme), lang, theme, HERE)
        print("rendered", name)
