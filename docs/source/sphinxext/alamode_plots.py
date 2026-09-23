"""Interactive Plotly figures built from ALAMODE output files.

``.. interactive-plot::`` reads .bands / .dos(.bz2) / .scph_bands / column files
from the repository at build time and embeds them as compact JSON. HTML output
draws the figure with Plotly (``_static/alamode_plots.js``) and keeps the
``:fallback:`` image in a <noscript>; every other builder (LaTeX, text, ...)
gets the fallback image as an ordinary figure (or image, without a caption).

Kinds and their options::

    .. interactive-plot::
       :kind: dispersion | dispersion-compare | dispersion-temperature | xy
       :fallback: ../../img/some.png        (required; figure options also apply)
       :data: file.bands [file2.bands ...]  (dispersion kinds)
       :labels: NA = 0; NA = 1             (legend entries, ';'-separated)
       :dos: file.dos.bz2                   (dispersion: side DOS panel)
       :reference: harmonic.bands           (dispersion-temperature: dashed overlay)
       :reference-label: Harmonic
       :tmin: 50                            (dispersion-temperature: drop lower T)
       :temperature: 300                    (dispersion-temperature: initial slider T)
       :series:                             (xy: one "file xcol:ycol label" per line,
          file.kl 1:2 kappa_xx               columns 1-based as in gnuplot)
       :xlabel: / :ylabel: / :log: x|y|xy / :markers:
       :height: 420

       Optional caption (translatable).

Data paths are relative to the rst file or, failing that, to the repository root.
``python alamode_plots.py`` runs a self-check on the Si and SrTiO3 references.
"""

import bz2
import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.directives.patches import Figure
from sphinx.transforms.post_transforms import SphinxPostTransform

REPO_ROOT = Path(__file__).resolve().parents[3]
FREQ_DIGITS = 2
K_DIGITS = 4


# -- parsers ---------------------------------------------------------------


def _read_lines(path):
    path = Path(path)
    if path.suffix == ".bz2":
        with bz2.open(path, "rt") as f:
            return f.read().splitlines()
    return path.read_text().splitlines()


def _ticks(lines):
    """High-symmetry labels and positions from the first two comment lines."""
    labels = lines[0].lstrip("#").split()
    pos = [round(float(x), K_DIGITS) for x in lines[1].lstrip("#").split()]
    labels = ["Γ" if s.upper() in ("G", "GAMMA") else s for s in labels]
    # merge labels that share a position (path discontinuities such as X|U)
    ticks, names = [], []
    for p, s in zip(pos, labels):
        if ticks and abs(ticks[-1] - p) < 1e-4:
            names[-1] += "|" + s
        else:
            ticks.append(p)
            names.append(s)
    return {"ticks": ticks, "ticklabels": names}


def _rows(lines):
    return [
        [float(v) for v in ln.split()]
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def parse_bands(path):
    lines = _read_lines(path)
    rows = _rows(lines)
    out = _ticks(lines)
    out["x"] = [round(r[0], K_DIGITS) for r in rows]
    out["branches"] = [
        [round(r[i], FREQ_DIGITS) for r in rows] for i in range(1, len(rows[0]))
    ]
    return out


def parse_dos(path):
    rows = _rows(_read_lines(path))
    return {
        "e": [round(r[0], FREQ_DIGITS) for r in rows],
        "dos": [float(f"{r[1]:.4g}") for r in rows],
    }


def parse_scph_bands(path, tmin=None):
    lines = _read_lines(path)
    out = _ticks(lines)
    blocks = {}
    for r in _rows(lines):
        blocks.setdefault(r[0], []).append(r[1:])
    temps = sorted(t for t in blocks if tmin is None or t >= tmin)
    first = blocks[temps[0]]
    out["x"] = [round(r[0], K_DIGITS) for r in first]
    out["temps"] = temps
    out["frames"] = [
        [[round(r[i], FREQ_DIGITS) for r in blocks[t]] for i in range(1, len(first[0]))]
        for t in temps
    ]
    return out


def parse_columns(path, cx, cy):
    """Columns cx, cy (1-based) of a whitespace-separated file, sorted by x."""
    pts = sorted((r[cx - 1], r[cy - 1]) for r in _rows(_read_lines(path)))
    return [float(f"{x:.6g}") for x, _ in pts], [float(f"{y:.5g}") for _, y in pts]


# -- directive -------------------------------------------------------------


class MissingData(Exception):
    """A data file of an interactive plot does not exist."""


class fallback_node(nodes.General, nodes.Element):
    """Holds the JSON payload; its only child is the fallback figure/image.

    ``css`` names the <figure> class and the prefix of its canvas/data classes.
    """

    css = ""


class interactive_plot(fallback_node):
    css = "alamode-plot"


class InteractivePlot(Figure):
    """Base for figures drawn by JavaScript from a JSON payload (see run)."""

    name = "interactive-plot"
    node_class = interactive_plot
    required_arguments = 0
    option_spec = dict(
        Figure.option_spec,
        kind=lambda v: directives.choice(
            v, ("dispersion", "dispersion-compare", "dispersion-temperature", "xy")
        ),
        fallback=directives.unchanged_required,
        data=directives.unchanged,
        labels=directives.unchanged,
        dos=directives.unchanged,
        reference=directives.unchanged,
        **{"reference-label": directives.unchanged},
        tmin=float,
        temperature=float,
        series=directives.unchanged,
        xlabel=directives.unchanged,
        ylabel=directives.unchanged,
        log=lambda v: directives.choice(v, ("x", "y", "xy")),
        markers=directives.flag,
        height=directives.positive_int,
    )

    def _path(self, name):
        env = self.state.document.settings.env
        here = Path(env.doc2path(env.docname)).parent
        candidates = [(base / name).resolve() for base in (here, REPO_ROOT)]
        for p in candidates:
            if p.is_file():
                env.note_dependency(str(p))
                return p
        # Depend on the missing path too, so that the page is rebuilt once it appears.
        env.note_dependency(str(candidates[-1]))
        raise MissingData(name)

    def _payload(self):
        o = self.options
        kind = o.get("kind", "dispersion")
        labels = [s.strip() for s in o.get("labels", "").split(";") if s.strip()]
        files = o.get("data", "").split()
        d = {"kind": kind, "height": o.get("height", 420)}
        if kind in ("dispersion", "dispersion-compare"):
            if not files:
                raise self.error(
                    f"interactive-plot: :data: is required for kind {kind}"
                )
            d["series"] = [
                dict(
                    parse_bands(self._path(f)),
                    name=labels[i] if i < len(labels) else Path(f).name,
                )
                for i, f in enumerate(files)
            ]
            if "dos" in o:
                d["dos"] = parse_dos(self._path(o["dos"]))
        elif kind == "dispersion-temperature":
            d.update(parse_scph_bands(self._path(files[0]), o.get("tmin")))
            d["name"] = labels[0] if labels else "SCPH"
            d["t0"] = o.get("temperature", d["temps"][0])
            if "reference" in o:
                d["reference"] = dict(
                    parse_bands(self._path(o["reference"])),
                    name=o.get("reference-label", "Harmonic"),
                )
                if d["reference"]["ticklabels"] != d["ticklabels"]:
                    raise self.error(
                        "interactive-plot: reference .bands is on a different k path"
                    )
        else:
            d["series"] = []
            for line in o.get("series", "").strip().splitlines():
                fname, cols, *name = line.split(maxsplit=2)
                cx, cy = (int(c) for c in cols.split(":"))
                x, y = parse_columns(self._path(fname), cx, cy)
                d["series"].append(
                    {"x": x, "y": y, "name": name[0] if name else Path(fname).name}
                )
            d.update(
                xlabel=o.get("xlabel", ""),
                ylabel=o.get("ylabel", ""),
                log=o.get("log", ""),
            )
            d["markers"] = "markers" in o
        if kind != "xy":
            d["ylabel"] = o.get("ylabel", "Frequency (cm⁻¹)")
        return d

    def run(self):
        if "fallback" not in self.options:
            raise self.error(f"{self.name}: :fallback: image is required")
        try:
            payload = self._payload()
        except MissingData as e:
            payload = None
            self.state.document.reporter.warning(
                f"{self.name}: data file not found: {e}; showing the static image",
                line=self.lineno,
            )
        self.arguments = [self.options.pop("fallback")]
        (fig,) = super().run()[-1:]
        fallback = fig
        if isinstance(fig, nodes.figure) and not any(
            isinstance(c, nodes.caption) for c in fig
        ):
            # no caption: behave like ``.. image::`` so the PDF layout is unchanged
            fallback = fig[0]
            if fig.get("align"):
                fallback["align"] = fig["align"]
        if payload is None:
            return [fallback]
        node = self.node_class()
        node["payload"] = payload
        node += fallback
        return [node]


class StripInteractivePlots(SphinxPostTransform):
    """Non-HTML builders: keep only the fallback figure (all fallback_node kinds)."""

    default_priority = 200

    def run(self, **kwargs):
        if self.app.builder.format == "html":
            return
        for node in list(self.document.findall(fallback_node)):
            node.replace_self(node.children)


# -- HTML ------------------------------------------------------------------


def visit_html(self, node):
    fb = node[0]
    children = list(fb.children) if isinstance(fb, nodes.figure) else [fb]
    caption = next((c for c in children if isinstance(c, nodes.caption)), None)
    ids = " ".join(fb.get("ids", []))
    css = node.css
    data = json.dumps(
        node["payload"], separators=(",", ":"), ensure_ascii=False
    ).replace("</", "<\\/")
    self.body.append(
        f'<figure class="{css} align-{fb.get("align", "center")}"'
        + (f' id="{ids.split()[0]}"' if ids else "")
        + f'>\n<div class="{css}-canvas"></div>\n'
        + f'<script type="application/json" class="{css}-data">{data}</script>\n<noscript>'
    )
    for c in children:
        if not isinstance(c, (nodes.caption, nodes.legend)):
            c.walkabout(self)
    self.body.append("</noscript>\n")
    if caption is not None:
        caption.walkabout(self)
        self.body.append("</figcaption>\n")
    self.body.append("</figure>\n")
    raise nodes.SkipNode


def add_scripts(app, pagename, templatename, context, doctree):
    if (
        doctree is not None
        and next(doctree.findall(interactive_plot), None) is not None
    ):
        app.add_js_file("vendor/plotly-basic.min.js", loading_method="defer")
        app.add_js_file("alamode_plots.js", loading_method="defer")


def setup(app):
    app.add_node(interactive_plot, html=(visit_html, None))
    app.add_directive("interactive-plot", InteractivePlot)
    app.add_post_transform(StripInteractivePlots)
    app.connect("html-page-context", add_scripts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    ref = REPO_ROOT / "example"
    si = parse_bands(ref / "Si/reference/si222.bands")
    assert si["ticklabels"] == ["Γ", "X", "Γ", "L"] and len(si["branches"]) == 6
    assert all(len(b) == len(si["x"]) for b in si["branches"])
    dos = parse_dos(ref / "Si/reference/si222_20.dos.bz2")
    assert len(dos["e"]) == len(dos["dos"]) > 50
    x, y = parse_columns(ref / "Si/reference/si222.kl", 1, 2)
    assert x[0] == 0 and x[-1] == 1000 and len(y) == len(x)
    sto = parse_scph_bands(ref / "SrTiO3/reference/STO_scph2-2.scph_bands", tmin=50)
    assert (
        sto["temps"][0] == 50 and sto["temps"][-1] == 1000 and len(sto["frames"]) == 20
    )
    assert all(len(f) == 15 and len(f[0]) == len(sto["x"]) for f in sto["frames"])
    harm = parse_bands(ref / "SrTiO3/reference/STO222_NA3.bands")
    assert harm["ticklabels"] == sto["ticklabels"] and len(harm["x"]) == len(sto["x"])
    assert (
        min(harm["branches"][0])
        < 0
        <= min(min(b) for f in sto["frames"] for b in f) + 1e-3
    )
    print("alamode_plots self-check OK")
