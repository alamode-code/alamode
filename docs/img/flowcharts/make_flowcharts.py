#!/usr/bin/env python3
"""Render the flowcharts of the "Running ALAMODE" page.

Each chart is defined once, with its labels in English and Japanese, and
rendered by Graphviz to PDF (LaTeX) and, through poppler's pdftocairo, to SVG (HTML) for both languages:
NAME.svg / NAME.pdf and NAME.ja.svg / NAME.ja.pdf. A dark-palette set,
NAME.dark.svg and NAME.dark.ja.svg, is shown instead in the dark mode of the
HTML theme (only-light / only-dark classes in quickstart.rst). Sphinx picks the
language variant through figure_language_filename and the format through
the ``NAME.*`` wildcard. The rendered files are committed, so building the
documentation does not need Graphviz or poppler; rerun this script after editing.

    python3 make_flowcharts.py            # uses `dot` from PATH
    DOT=/path/to/dot python3 make_flowcharts.py
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOT = os.environ.get("DOT", "dot")
PDFTOCAIRO = os.environ.get("PDFTOCAIRO", "pdftocairo")  # poppler
FONT = {"en": "Helvetica", "ja": "Hiragino Sans"}
MONO = "Courier"

# Role -> node shape. Shapes, border styles, and explicit program names
# supplement colour so that the charts also work in grey scale.
STYLE = {
    "alm": dict(shape="box", style="rounded,filled", penwidth="1.6"),
    "anphon": dict(shape="box", style="rounded,filled", penwidth="1.6"),
    "dft": dict(shape="box", style="filled"),
    "tool": dict(shape="box", style="filled,dashed"),
    "file": dict(shape="note", style="filled"),
    "ask": dict(shape="diamond", style="filled", margin="0.02"),
    "end": dict(shape="box", style="rounded,filled"),
    "set": dict(shape="box", style="filled"),
}

# Theme -> text and edge colours, and role -> (fill, border). The background is
# transparent, so the dark set is drawn on the dark page of the HTML theme.
THEMES = {
    "light": dict(
        text="#000000",
        edge="#444444",
        roles={
            "alm": ("#DCEBFA", "#2B6CB0"),
            "anphon": ("#DDF2E3", "#2F855A"),
            "dft": ("#ECECEC", "#555555"),
            "tool": ("#FFF4D6", "#B7791F"),
            "file": ("#FFFFFF", "#777777"),
            "ask": ("#FCE9E9", "#C53030"),
            "end": ("#EFE6FB", "#6B46C1"),
            "set": ("#FFFFFF", "#2F855A"),
        },
    ),
    "dark": dict(
        text="#E8ECF1",
        edge="#AEB6C0",
        roles={
            "alm": ("#1D3A5C", "#6CA8E8"),
            "anphon": ("#1C4230", "#6CCB8F"),
            "dft": ("#3A3F46", "#A3ABB5"),
            "tool": ("#4A3A14", "#E0AE4E"),
            "file": ("#2B3038", "#9AA3AD"),
            "ask": ("#4D2226", "#F08A8A"),
            "end": ("#382A57", "#B39AF0"),
            "set": ("#2B3038", "#6CCB8F"),
        },
    ),
}


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(text, lang, plain=False):
    """First line bold, following lines smaller; `code` spans in monospace.
    plain=True keeps every line in one regular weight (decision nodes)."""
    out = []
    for i, line in enumerate(text.split("\n")):
        parts = line.split("`")
        html = "".join(
            ('<FONT FACE="%s">%s</FONT>' % (MONO, esc(p)) if k % 2 else esc(p))
            for k, p in enumerate(parts)
        )
        if plain:
            out.append(html)
        else:
            out.append(
                "<B>%s</B>" % html
                if i == 0
                else '<FONT POINT-SIZE="10">%s</FONT>' % html
            )
    # Separate rows give mixed Latin/Japanese fonts enough vertical clearance.
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
        'CELLPADDING="2">'
        + "".join("<TR><TD>%s</TD></TR>" % row for row in out)
        + "</TABLE>>"
    )


def render(name, graph, nodes, edges, lang, same=(), theme="light"):
    font = FONT[lang]
    th = THEMES[theme]
    lines = [
        "digraph %s {" % name,
        '  graph [bgcolor="transparent", fontname="%s", %s];' % (font, graph),
        '  node  [fontname="%s", fontsize="12", margin="0.15,0.08", fontcolor="%s"];'
        % (font, th["text"]),
        '  edge  [fontname="%s", fontsize="10", color="%s", fontcolor="%s", '
        'arrowsize="0.8"];' % (font, th["edge"], th["text"]),
    ]
    for nid, (role, text) in nodes.items():
        fill, border = th["roles"][role]
        attrs = dict(STYLE[role], fillcolor=fill, color=border)
        lines.append(
            "  %s [label=%s, %s];"
            % (
                nid,
                label(text[lang], lang, role == "ask"),
                ", ".join('%s="%s"' % kv for kv in attrs.items()),
            )
        )
    for e in edges:
        src, dst = e[0], e[1]
        extra = dict(e[2]) if len(e) > 2 else {}
        if "label" in extra:
            extra["label"] = " %s " % extra["label"][lang]
        attrs = ", ".join('%s="%s"' % kv for kv in extra.items())
        lines.append("  %s -> %s [%s];" % (src, dst, attrs))
    for group in same:
        lines.append("  { rank=same; %s; }" % "; ".join(group))
    lines.append("}")
    source = "\n".join(lines)
    render_source(name, source, lang, theme)


def render_source(name, source, lang, theme="light", output_dir=HERE):
    """Render DOT through PDF to outlined SVG, using the shared naming scheme."""
    # NAME[.dark][.ja].ext -- Sphinx's default figure_language_filename is
    # {root}.{language}{ext}, so the language must come last.
    stem = name + ("" if theme == "light" else "." + theme)
    stem += "" if lang == "en" else "." + lang
    pdf = Path(output_dir) / (stem + ".pdf")
    subprocess.run([DOT, "-Tpdf", "-o", str(pdf)], input=source.encode(), check=True)
    # The SVG is converted from the PDF so that the text becomes glyph outlines:
    # Graphviz's own SVG places each font run at a fixed x, and runs overlap
    # when the browser substitutes fonts with other widths.
    subprocess.run(
        [PDFTOCAIRO, "-svg", str(pdf), str(pdf.with_suffix(".svg"))], check=True
    )
    if theme != "light":
        pdf.unlink()  # the dark set is used by the HTML pages only


def L(en, ja):
    return {"en": en, "ja": ja}


YES, NO = L("yes", "はい"), L("no", "いいえ")

# ---------------------------------------------------------------- overview
OVERVIEW = dict(
    graph='rankdir="TB", nodesep="0.35", ranksep="0.32"',
    nodes={
        "dft0": (
            "dft",
            L(
                "DFT\nconverge settings, relax the primitive cell",
                "DFT\n計算条件の収束確認と基本セルの構造最適化",
            ),
        ),
        "sug": (
            "alm",
            L("alm\nsuggest displacement patterns", "alm\n変位パターンを提案"),
        ),
        "dftf": (
            "dft",
            L(
                "DFT\nforces of the displaced supercells",
                "DFT\n変位させたスーパーセルの原子に働く力",
            ),
        ),
        "fit": ("alm", L("alm\nfit the force constants", "alm\n力定数をフィット")),
        "fcs": (
            "file",
            L("PREFIX.h5\nforce constants (IFCs)", "PREFIX.h5\n力定数 (IFC)"),
        ),
        "strain": (
            "tool",
            L(
                "Strain tools (optional)\nstrained-cell DFT → STRAINFILE",
                "ひずみ用ツール (任意)\nひずみを与えたセルの DFT → STRAINFILE",
            ),
        ),
        "anp": ("anphon", L("anphon\nphonon properties", "anphon\nフォノン物性")),
        "out": (
            "file",
            L(
                "Output files\n.bands  .dos  .thermo  .kappa.h5  ...",
                "出力ファイル\n.bands  .dos  .thermo  .kappa.h5  ...",
            ),
        ),
        "tools": (
            "tool",
            L(
                "Analysis tools\nplotband.py  plotdos.py  analyzer.py",
                "解析ツール\nplotband.py  plotdos.py  analyzer.py",
            ),
        ),
    },
    edges=[
        ("dft0", "sug"),
        ("sug", "dftf"),
        ("dftf", "fit"),
        ("fit", "fcs"),
        ("fcs", "anp"),
        (
            "strain",
            "anp",
            dict(style="dashed", label=L("cell relaxation", "セルの最適化")),
        ),
        ("anp", "out"),
        ("out", "tools"),
    ],
)

# --------------------------------------------------------------- alm steps
ALM = dict(
    graph='rankdir="TB", nodesep="0.3", ranksep="0.3"',
    nodes={
        "start": (
            "dft",
            L("DFT: relaxed primitive cell", "DFT: 構造最適化した基本セル"),
        ),
        "cell": (
            "set",
            L(
                "Choose a supercell\n`SUPERCELL` of alm (recommended),\nor build it with pymatgen, ASE, ...",
                "スーパーセルを決める\nalm の `SUPERCELL` (推奨),\nまたは pymatgen, ASE などで作成",
            ),
        ),
        "sug": (
            "alm",
            L(
                "alm   `MODE = suggest`  (NORDER = 1)\nharmonic displacement patterns",
                "alm   `MODE = suggest`  (NORDER = 1)\n調和項の変位パターン",
            ),
        ),
        "dA": (
            "tool",
            L(
                "displace.py  `--pattern_file`\nsmall displacements, about 0.01 Å",
                "displace.py  `--pattern_file`\n小さな変位 (約 0.01 Å)",
            ),
        ),
        "dftA": (
            "dft",
            L("DFT forces → extract.py → DFSET", "DFT で力を計算 → extract.py → DFSET"),
        ),
        "fitA": (
            "alm",
            L(
                "alm   `MODE = optimize`  (NORDER = 1)\n`LMODEL = ols`",
                "alm   `MODE = optimize`  (NORDER = 1)\n`LMODEL = ols`",
            ),
        ),
        "fc2": (
            "file",
            L(
                "harmonic.h5\nharmonic force constants (FC2)",
                "harmonic.h5\n調和力定数 (FC2)",
            ),
        ),
        "q": (
            "ask",
            L("Anharmonic\nIFCs\nneeded?", "非調和\n力定数が\n必要?"),
        ),
        "done": (
            "anphon",
            L(
                "anphon\nphonons, DOS, thermodynamics",
                "anphon\nフォノン分散, DOS, 熱力学量",
            ),
        ),
        "evec": (
            "anphon",
            L(
                "anphon   `MODE = phonons`  (KPMODE = 0)\n`FCSFILE = harmonic.h5`\n`PRINTEVEC = 1`  → PREFIX.evec.h5",
                "anphon   `MODE = phonons`  (KPMODE = 0)\n`FCSFILE = harmonic.h5`\n`PRINTEVEC = 1`  → PREFIX.evec.h5",
            ),
        ),
        "dB": (
            "tool",
            L(
                "displace.py\n`--random_normalcoord`\n`--evec PREFIX.evec.h5 --temp T`\nrandom displacements sampled at temperature T\n(or `--random` on MD snapshots)",
                "displace.py\n`--random_normalcoord`\n`--evec PREFIX.evec.h5 --temp T`\n温度 T でサンプリングしたランダム変位\n(または MD スナップショットに `--random`)",
            ),
        ),
        "dftB": (
            "dft",
            L("DFT forces → extract.py → DFSET", "DFT で力を計算 → extract.py → DFSET"),
        ),
        "fitB": (
            "alm",
            L(
                "alm   `MODE = optimize`  (NORDER = 2, 3, ...)\n`LMODEL = enet` or `adaptive-lasso`\n`FC2FIX = harmonic.h5`",
                "alm   `MODE = optimize`  (NORDER = 2, 3, ...)\n`LMODEL = enet` または `adaptive-lasso`\n`FC2FIX = harmonic.h5`",
            ),
        ),
        "fcs": (
            "file",
            L(
                "PREFIX.h5\nharmonic and anharmonic force constants",
                "PREFIX.h5\n調和および非調和力定数",
            ),
        ),
    },
    edges=[
        ("start", "cell"),
        ("cell", "sug"),
        ("sug", "dA"),
        ("dA", "dftA"),
        ("dftA", "fitA"),
        ("fitA", "fc2"),
        ("fc2", "q"),
        ("q", "done", dict(label=NO, constraint="false")),
        (
            "q",
            "evec",
            dict(
                label=L("yes\n(kappa, SCPH, QHA, ...)", "はい\n(kappa, SCPH, QHA など)")
            ),
        ),
        ("evec", "dB"),
        ("dB", "dftB"),
        ("dftB", "fitB"),
        ("fitB", "fcs"),
    ],
    same=[["q", "done"]],
)

# ------------------------------------------------------------- anphon mode
MODES = dict(
    graph='rankdir="LR", nodesep="0.22", ranksep="0.5"',
    nodes={
        "fcs": ("file", L("PREFIX.h5\nforce constants", "PREFIX.h5\n力定数")),
        "q": ("ask", L("What do\nyou want to\ncalculate?", "何を\n計算する?")),
        "ph": (
            "anphon",
            L(
                "Dispersion, DOS, thermodynamics\n`MODE = phonons`\nFC2  (+ FC3 for Grüneisen)",
                "分散関係, DOS, 熱力学関数\n`MODE = phonons`\nFC2  (グリュナイゼン定数は + FC3)",
            ),
        ),
        "ka": (
            "anphon",
            L(
                "Thermal conductivity\n`MODE = kappa`\nFC2, FC3  (+ FC4 for 4-phonon)",
                "格子熱伝導率\n`MODE = kappa`\nFC2, FC3  (4フォノン散乱は + FC4)",
            ),
        ),
        "se": (
            "anphon",
            L(
                "Linewidth or spectrum at chosen q\n`MODE = selfenergy`\nFC2, FC3",
                "指定した q 点の線幅・スペクトル\n`MODE = selfenergy`\nFC2, FC3",
            ),
        ),
        "sc": (
            "anphon",
            L(
                "Phonons at finite T, strong anharmonicity\n`MODE = SCPH`\nFC2, FC3, FC4",
                "有限温度のフォノン (強い非調和性)\n`MODE = SCPH`\nFC2, FC3, FC4",
            ),
        ),
        "qh": (
            "anphon",
            L(
                "Thermal expansion, structure vs T\n`MODE = QHA`\nFC2, FC3, FC4",
                "熱膨張, 構造の温度変化\n`MODE = QHA`\nFC2, FC3, FC4",
            ),
        ),
    },
    edges=[("fcs", "q")] + [("q", m) for m in ("ph", "ka", "se", "sc", "qh")],
)

# ------------------------------------------- finite temperature (SCPH, QHA)
FINITE_T = dict(
    graph='rankdir="TB", nodesep="0.35", ranksep="0.3"',
    nodes={
        "start": (
            "anphon",
            L("anphon\n`MODE = SCPH` or `QHA`", "anphon\n`MODE = SCPH` または `QHA`"),
        ),
        "q1": (
            "ask",
            L("Relax the\nstructure\nat finite T?", "有限温度で\n構造を\n最適化する?"),
        ),
        "r0": (
            "set",
            L(
                "`RELAX_STR = 0`\nfixed structure (SCPH only)",
                "`RELAX_STR = 0`\n構造を固定 (SCPH のみ)",
            ),
        ),
        "q2": ("ask", L("Relax the\ncell as well?", "セルも\n最適化する?")),
        "r1": (
            "set",
            L(
                "`RELAX_STR = 1`\natomic positions only",
                "`RELAX_STR = 1`\n原子位置のみ",
            ),
        ),
        "r2": (
            "set",
            L(
                "`RELAX_STR = 2`\npositions and cell",
                "`RELAX_STR = 2`\n原子位置とセル",
            ),
        ),
        "q3": (
            "ask",
            L(
                "Skip the\nstrained-cell\nDFT?",
                "ひずみセルの\nDFT を\n省略する?",
            ),
        ),
        "tl": (
            "tool",
            L(
                "Strain tools\nstrained-cell DFT\nstrainifc.py, elastic.py → strainfile.py",
                "ひずみ用ツール\nひずみセルの DFT を準備\nstrainifc.py, elastic.py → strainfile.py",
            ),
        ),
        "sf": (
            "file",
            L(
                "STRAINFILE\n`STRAIN_COUPLING = 7`  (default)",
                "STRAINFILE\n`STRAIN_COUPLING = 7`  (既定値)",
            ),
        ),
        "c0": (
            "set",
            L(
                "`STRAIN_COUPLING = 0`\nelastic / harmonic couplings from IFCs;\nstrain–force coupling set to zero",
                "`STRAIN_COUPLING = 0`\n弾性定数・調和項の結合を IFC から推定\nひずみと力の結合はゼロ",
            ),
        ),
        "run": (
            "file",
            L(
                "PREFIX.scph.h5 / .qha.h5\nFC2 and structure at each T",
                "PREFIX.scph.h5 / .qha.h5\n各温度の FC2 と結晶構造",
            ),
        ),
        "q4": (
            "ask",
            L(
                "Use the\nresult at T in\nanother run?",
                "温度 T の\n結果を別の\n計算に使う?",
            ),
        ),
        "end": ("end", L("Analyze the results", "結果を解析")),
        "fu": (
            "set",
            L(
                "Reuse the state file\n`FCSFILE = PREFIX.h5` (original IFCs)\n`DFC2FILE = PREFIX.scph.h5` (or .qha.h5)\n`FC2_TEMPERATURE = T` (stored temperature)\nIf relaxed: `RELAXED_STRUCTURE = 1`\nand retain the original FC4",
                "保存した結果を再利用\n`FCSFILE = PREFIX.h5` (元の IFC)\n`DFC2FILE = PREFIX.scph.h5` (または .qha.h5)\n`FC2_TEMPERATURE = T` (保存済みの温度)\n構造最適化後は `RELAXED_STRUCTURE = 1`\n元の FC4 も必要",
            ),
        ),
        "re": (
            "anphon",
            L(
                "anphon\n`MODE = kappa` or `phonons`\nFor kappa: `TMIN = TMAX = T`",
                "anphon\n`MODE = kappa` または `phonons`\nkappa の場合: `TMIN = TMAX = T`",
            ),
        ),
    },
    edges=[
        ("start", "q1"),
        ("q1", "r0", dict(label=NO)),
        ("q1", "q2", dict(label=YES)),
        ("q2", "r1", dict(label=NO)),
        ("q2", "r2", dict(label=YES)),
        ("r2", "q3"),
        ("q3", "tl", dict(label=NO)),
        (
            "q3",
            "c0",
            dict(
                label=L(
                    "yes (only if the strain–force\ncoupling is zero by symmetry)",
                    "はい (対称性により\nひずみと力の結合が\nゼロの場合のみ)",
                )
            ),
        ),
        ("tl", "sf"),
        ("r0", "run"),
        ("r1", "run"),
        ("sf", "run"),
        ("c0", "run"),
        ("run", "q4"),
        ("q4", "fu", dict(label=YES)),
        ("q4", "end", dict(label=NO, constraint="false")),
        ("fu", "re"),
    ],
    same=[["r0", "r1", "r2"], ["sf", "c0"], ["q4", "end"]],
)

CHARTS = {
    "overview": OVERVIEW,
    "alm_workflow": ALM,
    "anphon_modes": MODES,
    "finite_temperature": FINITE_T,
}

if __name__ == "__main__":
    for name, c in CHARTS.items():
        for lang in ("en", "ja"):
            for theme in THEMES:
                args = (c["graph"], c["nodes"], c["edges"], lang, c.get("same", ()))
                render(name, *args, theme=theme)
        print("rendered", name)
