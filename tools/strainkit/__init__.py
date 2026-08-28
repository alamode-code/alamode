"""strainkit -- helpers for strain-related ALAMODE workflows.

Two command-line tools are built on this package:

* ``tools/strainifc.py`` -- strain-IFC coupling inputs (``strain_harmonic.in``,
  ``strain_force.in``) for the SCPH/QHA cell relaxation of anphon.  This is a
  port of the ``strainIFCcoupling`` scripts by Ryota Masuki
  (https://github.com/r-masuki/strainIFCcoupling, MIT license) onto the
  in-repo ``alm`` Python package.
* ``tools/elastic.py`` -- DFT finite-strain workflow for the first-, second- and
  third-order elastic constants (``elastic_constants.in``, ``C1_array.in``).

All modules keep the atom ordering of the user's template structure; nothing
is ever reordered silently.
"""

__version__ = "0.1.0"
