Third-party files vendored for the documentation.

- `plotly-basic.min.js`: plotly.js 4.1.1, "basic" partial bundle (scatter, bar,
  pie), unmodified. MIT License, see `plotly.LICENSE`.
  Source: https://github.com/plotly/plotly.js (dist/plotly-basic.min.js).
  Loaded only on pages that contain an `interactive-plot` directive
  (see `docs/source/sphinxext/alamode_plots.py`).
- `3Dmol-min.js`: 3Dmol.js 2.5.5, unmodified. BSD-3-Clause License, see
  `3dmol.LICENSE` (and the banner in `3Dmol-min.js.LICENSE.txt`).
  Source: https://www.npmjs.com/package/3dmol (build/3Dmol-min.js).
  Loaded only on pages that contain a `structure-viewer` directive
  (see `docs/source/sphinxext/alamode_structures.py`).
