#!/usr/bin/env python3
"""Render the flowcharts of the "Running ALAMODE" page.

Each chart is defined once, with its labels in English and Japanese, and
rendered by Graphviz to SVG (HTML) and PDF (LaTeX) for both languages:
NAME.svg / NAME.pdf and NAME.ja.svg / NAME.ja.pdf. Sphinx picks the
language variant through figure_language_filename and the format through
the ``NAME.*`` wildcard. The rendered files are committed, so building the
documentation does not need Graphviz; rerun this script after editing.

    python3 make_flowcharts.py            # uses `dot` from PATH
    DOT=/path/to/dot python3 make_flowcharts.py
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOT = os.environ.get("DOT", "dot")
FONT = {"en": "Helvetica", "ja": "Hiragino Sans"}
MONO = "Courier"

# role -> node attributes. Roles differ in shape as well as colour, so the
# charts stay readable in grey scale.
STYLE = {
    "alm": dict(
        shape="box",
        style="rounded,filled",
        fillcolor="#DCEBFA",
        color="#2B6CB0",
        penwidth="1.6",
    ),
    "anphon": dict(
        shape="box",
        style="rounded,filled",
        fillcolor="#DDF2E3",
        color="#2F855A",
        penwidth="1.6",
    ),
    "dft": dict(shape="box", style="filled", fillcolor="#ECECEC", color="#555555"),
    "tool": dict(
        shape="box", style="filled,dashed", fillcolor="#FFF4D6", color="#B7791F"
    ),
    "file": dict(shape="note", style="filled", fillcolor="#FFFFFF", color="#777777"),
    "ask": dict(
        shape="diamond",
        style="filled",
        fillcolor="#FCE9E9",
        color="#C53030",
        margin="0.02",
    ),
    "end": dict(
        shape="box", style="rounded,filled", fillcolor="#EFE6FB", color="#6B46C1"
    ),
    "set": dict(shape="box", style="filled", fillcolor="#FFFFFF", color="#2F855A"),
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
    return "<" + "<BR/>".join(out) + ">"


def render(name, graph, nodes, edges, lang, same=()):
    font = FONT[lang]
    lines = [
        "digraph %s {" % name,
        '  graph [bgcolor="white", fontname="%s", %s];' % (font, graph),
        '  node  [fontname="%s", fontsize="12", margin="0.15,0.08"];' % font,
        '  edge  [fontname="%s", fontsize="10", color="#444444", arrowsize="0.8"];'
        % font,
    ]
    for nid, (role, text) in nodes.items():
        attrs = ", ".join('%s="%s"' % kv for kv in STYLE[role].items())
        lines.append(
            "  %s [label=%s, %s];"
            % (nid, label(text[lang], lang, role == "ask"), attrs)
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
    suffix = "" if lang == "en" else "." + lang
    for fmt in ("svg", "pdf"):
        subprocess.run(
            [DOT, "-T" + fmt, "-o", str(HERE / ("%s%s.%s" % (name, suffix, fmt)))],
            input=source.encode(),
            check=True,
        )


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
                "Strained-cell DFT  (optional)\nonly to relax the cell at finite T",
                "ひずみを与えたセルの DFT  (任意)\n有限温度でセルを最適化するときのみ",
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
            "end",
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
        ("strain", "anp", dict(style="dashed", label=L("STRAINFILE", "STRAINFILE"))),
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
                "Choose a supercell\na conventional cell is often a good start",
                "スーパーセルを決める\n通常は慣用単位胞程度から",
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
            "end",
            L("phonons, DOS,\nthermodynamics", "フォノン分散, DOS,\n熱力学量"),
        ),
        "evec": (
            "anphon",
            L(
                "anphon   `MODE = phonons`  (KPMODE = 0)\n`FCSFILE = harmonic.h5`  → PREFIX.evec",
                "anphon   `MODE = phonons`  (KPMODE = 0)\n`FCSFILE = harmonic.h5`  → PREFIX.evec",
            ),
        ),
        "dB": (
            "tool",
            L(
                "displace.py  `--random_normalcoord --evec PREFIX.evec --temp T`\nrandom displacements sampled at temperature T\n(or `--random` on MD snapshots)",
                "displace.py  `--random_normalcoord --evec PREFIX.evec --temp T`\n温度 T でサンプリングしたランダム変位\n(または MD スナップショットに `--random`)",
            ),
        ),
        "dftB": (
            "dft",
            L("DFT forces → extract.py → DFSET", "DFT で力を計算 → extract.py → DFSET"),
        ),
        "fitB": (
            "alm",
            L(
                "alm   `MODE = optimize`  (NORDER = 2, 3, ...)\n`LMODEL = enet` or `adaptive-lasso`,  `FC2FIX = harmonic.h5`",
                "alm   `MODE = optimize`  (NORDER = 2, 3, ...)\n`LMODEL = enet` または `adaptive-lasso`,  `FC2FIX = harmonic.h5`",
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
        ("q", "done", dict(label=NO)),
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
                "Dispersion, DOS, thermodynamics\n`MODE = phonons`    FC2  (+ FC3 for Grüneisen)",
                "分散関係, DOS, 熱力学関数\n`MODE = phonons`    FC2  (グリュナイゼン定数は + FC3)",
            ),
        ),
        "ka": (
            "anphon",
            L(
                "Thermal conductivity\n`MODE = kappa`    FC2, FC3  (+ FC4 for 4-phonon)",
                "格子熱伝導率\n`MODE = kappa`    FC2, FC3  (4フォノン散乱は + FC4)",
            ),
        ),
        "se": (
            "anphon",
            L(
                "Linewidth or spectrum at chosen q\n`MODE = selfenergy`    FC2, FC3",
                "指定した q 点の線幅・スペクトル\n`MODE = selfenergy`    FC2, FC3",
            ),
        ),
        "sc": (
            "anphon",
            L(
                "Phonons at finite T, strong anharmonicity\n`MODE = SCPH`    FC2, FC3, FC4",
                "有限温度のフォノン (強い非調和性)\n`MODE = SCPH`    FC2, FC3, FC4",
            ),
        ),
        "qh": (
            "anphon",
            L(
                "Thermal expansion, structure vs T\n`MODE = QHA`    FC2, FC3, FC4",
                "熱膨張, 構造の温度変化\n`MODE = QHA`    FC2, FC3, FC4",
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
                "`RELAX_STR = 2`\npositions and cell  (QHA: also 3)",
                "`RELAX_STR = 2`\n原子位置とセル  (QHA は 3 も可)",
            ),
        ),
        "q3": (
            "ask",
            L(
                "DFT data of\nstrained cells\navailable?",
                "ひずみを与えた\nセルの DFT\nデータがある?",
            ),
        ),
        "tl": (
            "tool",
            L(
                "strainifc.py, elastic.py\n→ strainfile.py",
                "strainifc.py, elastic.py\n→ strainfile.py",
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
                "`STRAIN_COUPLING = 0`\nestimated from the IFCs;\nhigh-symmetry crystals only",
                "`STRAIN_COUPLING = 0`\n力定数から推定;\n高対称な結晶のみ",
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
        "fu": (
            "set",
            L(
                "`FC2_TEMPERATURE = T`\n+ `RELAXED_STRUCTURE = 1`  if the structure was relaxed",
                "`FC2_TEMPERATURE = T`\n構造を最適化した場合は + `RELAXED_STRUCTURE = 1`",
            ),
        ),
        "re": (
            "anphon",
            L(
                "anphon\n`MODE = kappa` or `phonons`  on the result at T",
                "anphon\n温度 T の結果で `MODE = kappa` または `phonons`",
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
        ("q3", "tl", dict(label=YES)),
        ("q3", "c0", dict(label=NO)),
        ("tl", "sf"),
        ("r0", "run"),
        ("r1", "run"),
        ("sf", "run"),
        ("c0", "run"),
        ("run", "q4"),
        ("q4", "fu", dict(label=YES)),
        ("fu", "re"),
    ],
    same=[["r1", "r2"]],
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
            render(name, c["graph"], c["nodes"], c["edges"], lang, c.get("same", ()))
        print("rendered", name)
