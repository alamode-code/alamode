"""Interactive 3D crystal structures and phonon-mode animations (3Dmol.js).

``.. structure-viewer::`` reads a structure at build time and embeds it as JSON;
HTML output draws it with 3Dmol.js (``_static/alamode_structures.js``), other
builders get the ``:fallback:`` image (see alamode_plots.py, whose directive,
fallback handling and HTML writer this reuses)::

    .. structure-viewer::
       :structure: si_alm1.in | POSCAR      (ALAMODE input or VASP POSCAR)
       :animation: mode.xyz                  (multi-frame XYZ from anphon ANIME)
       :amplitude: 4                         (magnify the displacements of :animation:)
       :bonds: Ti O                          (elements drawn with bonds; default all)
       :fallback: ../../img/some.png         (required; figure options also apply)
       :height: 400

       Optional caption (translatable).

With ``:animation:``, ``:structure:`` only supplies the cell: an anphon input
whose ``&cell`` is multiplied by its ``ANIME_CELLSIZE`` (or any file above).
ALAMODE inputs are in Bohr, POSCAR and XYZ in Angstrom.
``python alamode_structures.py`` runs a self-check on example files.
"""

import math
import re
from pathlib import Path

from alamode_plots import (
    REPO_ROOT,
    InteractivePlot,
    fallback_node,
    visit_html,
)
from docutils.parsers.rst import directives
from sphinx.directives.patches import Figure

BOHR = 0.529177210903
DIGITS = 3
TOL = 1e-3


# -- parsers ---------------------------------------------------------------


def _frac_to_cart(lattice, frac):
    return [
        [sum(f[i] * lattice[i][k] for i in range(3)) for k in range(3)] for f in frac
    ]


def _is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _det(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def parse_poscar(path):
    """VASP 5 POSCAR (element names on line 6) -> (lattice, elements, coords) in A."""
    lines = Path(path).read_text().splitlines()
    s = [float(x) for x in lines[1].split()[:3] if _is_number(x)]
    raw = [[float(x) for x in ln.split()[:3]] for ln in lines[2:5]]
    if len(s) == 3:  # one factor per Cartesian axis
        scale = s
    elif s[0] < 0:  # negative: the target cell volume
        f = (-s[0] / abs(_det(raw))) ** (1 / 3)
        scale = [f, f, f]
    else:
        scale = [s[0]] * 3
    lattice = [[scale[k] * v[k] for k in range(3)] for v in raw]
    names, counts = lines[5].split(), [int(n) for n in lines[6].split()]
    elements = [e for e, n in zip(names, counts) for _ in range(n)]
    i = 8 if lines[7].strip()[0] in "sS" else 7  # Selective dynamics
    cartesian = lines[i].strip()[0] in "cCkK"
    pos = [
        [float(x) for x in ln.split()[:3]]
        for ln in lines[i + 1 : i + 1 + len(elements)]
    ]
    coords = (
        [[scale[k] * p[k] for k in range(3)] for p in pos]
        if cartesian
        else _frac_to_cart(lattice, pos)
    )
    return lattice, elements, coords


def _fields(text):
    """&name ... / blocks of an ALAMODE input, keyed by lower-case name."""
    text = re.sub(r"#.*", "", text)
    return {
        m[1].lower(): m[2] for m in re.finditer(r"&(\w+)(.*?)^\s*/", text, re.M | re.S)
    }


def _keys(body):
    return {
        k.strip().upper(): v.strip()
        for k, v in re.findall(r"(\w+)\s*=\s*([^;\n]*)", body)
    }


def parse_alamode(path):
    """alm/anphon input -> (lattice, elements, coords) in A; no atoms without &position.

    The cell is multiplied by ANIME_CELLSIZE when &analysis gives one.
    """
    f = _fields(Path(path).read_text())
    cell = [ln.split() for ln in f["cell"].strip().splitlines() if ln.strip()]
    a = float(cell[0][0]) * BOHR
    lattice = [[a * float(x) for x in row[:3]] for row in cell[1:4]]
    size = _keys(f.get("analysis", "")).get("ANIME_CELLSIZE")
    if size:
        lattice = [[n * x for x in v] for n, v in zip(map(int, size.split()), lattice)]
    if "position" not in f:
        return lattice, [], []
    kd = _keys(f["general"])["KD"].split()
    rows = [ln.split() for ln in f["position"].strip().splitlines() if ln.strip()]
    elements = [kd[int(r[0]) - 1] for r in rows]
    return (
        lattice,
        elements,
        _frac_to_cart(lattice, [[float(x) for x in r[1:4]] for r in rows]),
    )


def parse_structure(path):
    head = Path(path).read_text()
    return parse_alamode(path) if "&cell" in head.lower() else parse_poscar(path)


def parse_xyz(path):
    """Multi-frame XYZ -> (elements, [frame coords]) in A."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        block = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        elements = [b[0] for b in block]
        frames.append([[float(x) for x in b[1:4]] for b in block])
        i += n + 2
    return elements, frames


# -- geometry --------------------------------------------------------------


def _inverse(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    adj = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    return [[x / det for x in row] for row in adj]


def build_payload(lattice, elements, frames, amplitude=1.0):
    """Wrap atoms into the cell, add their images on the cell faces, magnify
    the motion, and give the displacement arrows of the largest-amplitude frame."""
    n, nf = len(elements), len(frames)
    mean = [[sum(fr[i][k] for fr in frames) / nf for k in range(3)] for i in range(n)]
    inv = _inverse(lattice)
    out_el, shifts, src = [], [], []
    for i in range(n):
        frac = [sum(mean[i][k] * inv[k][j] for k in range(3)) for j in range(3)]
        base = [-math.floor(x + TOL) for x in frac]
        on_face = [abs(x + b) < TOL for x, b in zip(frac, base)]
        for img in range(8):
            steps = [(img >> j) & 1 for j in range(3)]
            if any(s and not on_face[j] for j, s in enumerate(steps)):
                continue
            t = [b + s for b, s in zip(base, steps)]
            shifts.append(
                [sum(t[j] * lattice[j][k] for j in range(3)) for k in range(3)]
            )
            out_el.append(elements[i])
            src.append(i)

    def at(fr, a):
        i = src[a]
        return [
            mean[i][k] + amplitude * (fr[i][k] - mean[i][k]) + shifts[a][k]
            for k in range(3)
        ]

    payload = {
        "lattice": [[round(x, DIGITS) for x in v] for v in lattice],
        "elements": out_el,
        "frames": [
            [round(x, DIGITS) for a in range(len(src)) for x in at(fr, a)]
            for fr in frames
        ],
    }
    if nf > 1:
        disp = [
            [[fr[i][k] - mean[i][k] for k in range(3)] for i in range(n)]
            for fr in frames
        ]
        big = max(disp, key=lambda d: sum(x * x for v in d for x in v))
        dmax = max(math.sqrt(sum(x * x for x in v)) for v in big)
        payload["arrows"] = [
            [round(m + t, DIGITS) for m, t in zip(mean[src[a]], shifts[a])]
            + [round(x / dmax, DIGITS) for x in big[src[a]]]
            for a in range(len(src))
            if math.sqrt(sum(x * x for x in big[src[a]])) > 0.2 * dmax
        ]
    return payload


# -- directive -------------------------------------------------------------


class structure_viewer(fallback_node):
    css = "alamode-structure"


class StructureViewer(InteractivePlot):
    name = "structure-viewer"
    node_class = structure_viewer
    option_spec = dict(
        Figure.option_spec,
        fallback=directives.unchanged_required,
        height=directives.positive_int,
        structure=directives.unchanged_required,
        animation=directives.unchanged,
        amplitude=float,
        bonds=directives.unchanged,
    )

    def _payload(self):
        o = self.options
        if "structure" not in o:
            raise self.error("structure-viewer: :structure: is required")
        lattice, elements, coords = parse_structure(self._path(o["structure"]))
        frames = [coords]
        if "animation" in o:
            elements, frames = parse_xyz(self._path(o["animation"]))
        if not elements:
            raise self.error(f"structure-viewer: no atoms in {o['structure']}")
        d = build_payload(lattice, elements, frames, o.get("amplitude", 1.0))
        d["height"] = o.get("height", 400)
        if "bonds" in o:
            d["bonds"] = o["bonds"].split()
        return d


def add_scripts(app, pagename, templatename, context, doctree):
    if (
        doctree is not None
        and next(doctree.findall(structure_viewer), None) is not None
    ):
        app.add_js_file("vendor/3Dmol-min.js", loading_method="defer")
        app.add_js_file("alamode_structures.js", loading_method="defer")


def setup(app):
    app.setup_extension(
        "alamode_plots"
    )  # post-transform stripping the node for non-HTML
    app.add_node(structure_viewer, html=(visit_html, None))
    app.add_directive("structure-viewer", StructureViewer)
    app.connect("html-page-context", add_scripts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    ex = REPO_ROOT / "example"
    lat, el, xyz = parse_structure(ex / "Si/reference/si_alm1.in")
    assert len(el) == len(xyz) == 64 and set(el) == {"Si"}
    assert abs(lat[0][0] - 20.406 * BOHR) < 1e-6
    d = build_payload(lat, el, [xyz])
    # face images: 64 atoms plus their periodic copies on the cell faces
    assert len(d["frames"][0]) == 3 * len(d["elements"]) > 3 * 64 and "arrows" not in d
    modes = REPO_ROOT / "docs/tools/mode_animations"
    lat, _, _ = parse_structure(REPO_ROOT / "docs/tools/mode_animations/sto_R.in")
    assert abs(lat[2][2] - 2 * 7.363 * BOHR) < 1e-6
    el, frames = parse_xyz(modes / "sto_R_soft.xyz")
    assert len(frames) == 20 and len(el) == 40 and el[:5] == ["Sr", "Ti", "O", "O", "O"]
    d = build_payload(lat, el, frames, 4)
    f0 = d["frames"][0]
    moved = [max(abs(f[j] - f0[j]) for f in d["frames"]) for j in range(len(f0))]
    moving = {
        e for a, e in enumerate(d["elements"]) if max(moved[3 * a : 3 * a + 3]) > 0.1
    }
    assert moving == {"O"}, moving  # R-point octahedron rotation: Sr and Ti stay still
    assert len(d["arrows"]) > 0 and all(len(a) == 6 for a in d["arrows"])
    print("alamode_structures self-check OK")
