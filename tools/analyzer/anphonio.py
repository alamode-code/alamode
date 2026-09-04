import numpy as np


class ParseResult:
    def __init__(self, filename):
        self.kgrid = np.zeros(3, dtype=int)
        self.nat = 0
        self.q_coord = None
        self.gamma = None
        self.temperatures = None
        self.omega = None
        self.multiplicity = None
        self.volume = None
        self.vel = None
        self.lattice_vector = None
        self.atomic_kinds = None
        self.x_fractional = None
        self.classical = None

        self.read_result(filename)

    def read_result(self, filename):
        total_irq = 0
        which_phonon = 0
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if line == "":
                    raise EOFError(
                        "Reached end of file before finishing parsing '{}'. "
                        "The file may be incomplete or corrupted.".format(filename)
                    )
                if "#SYSTEM" in line:
                    line = f.readline().rstrip().split()
                    self.nat = int(line[0])  # number of atoms
                    line = f.readline().rstrip()
                    self.volume = float(line)
                    # Optional cell geometry: 3 lattice-vector rows followed by one row per
                    # atom, either "x y z" (anphon writes the fractional coordinates only) or
                    # "kind x y z". Atomic kinds are set to None when they are not stored.
                    try:
                        lattice = np.zeros((3, 3), dtype=float)
                        for i in range(3):
                            lattice[i, :] = np.array([float(t) for t in f.readline().strip().split()])
                        kinds = np.zeros(self.nat, dtype=int)
                        xf = np.zeros((self.nat, 3), dtype=float)
                        has_kinds = None
                        for i in range(self.nat):
                            line = f.readline().strip().split()
                            if has_kinds is None:
                                has_kinds = len(line) == 4
                            if has_kinds:
                                kinds[i] = int(line[0])
                                xf[i, :] = np.array([float(t) for t in line[1:4]])
                            else:
                                xf[i, :] = np.array([float(t) for t in line[0:3]])
                        self.lattice_vector = lattice
                        self.x_fractional = xf
                        self.atomic_kinds = kinds if has_kinds else None
                    except Exception:
                        self.lattice_vector = None
                        self.x_fractional = None
                        self.atomic_kinds = None

                elif "#KPOINT" in line:
                    line = f.readline().rstrip().split()
                    self.kgrid = np.array(line, dtype=int)

                    line = f.readline().rstrip().split()
                    total_irq = int(line[0])
                    self.q_coord = np.zeros((total_irq, 3))
                    weight = np.zeros(total_irq)
                    for i in range(total_irq):
                        line = f.readline().rstrip().split()
                        self.q_coord[i, :] = np.array(line[1:4], dtype=float)
                        weight[i] = float(line[4])

                elif "#CLASSICAL" in line:
                    line = f.readline().rstrip().split()
                    self.classical = int(line[0])

                elif "#TEMPERATURE" in line:
                    line = f.readline().rstrip().split()
                    t_min = float(line[0])
                    t_max = float(line[1])
                    t_stp = float(line[2])
                    nstemp = int((t_max - t_min) / t_stp + 1)
                    self.temperatures = np.arange(t_min, t_max + t_stp, t_stp)
                    self.gamma = np.zeros((total_irq, self.nat * 3, nstemp))  # <-
                    self.vel = np.zeros((total_irq, self.nat * 3, 48, 3))
                    self.multiplicity = np.zeros(total_irq)

                elif "#K-point (irreducible)" in line:
                    self.omega = np.zeros((total_irq, 3 * self.nat))
                    for _i in range(3 * self.nat * total_irq):
                        line = f.readline().rstrip().split()
                        ik = int(line[0])
                        im = int(line[1])
                        o = float(line[2])
                        self.omega[ik - 1, im - 1] = o

                elif "#GAMMA_EACH" in line:
                    which_phonon += 1
                    line = f.readline().rstrip().split()
                    ik = int(line[0]) - 1
                    im = int(line[1]) - 1
                    line = f.readline().rstrip().split()
                    degen = int(line[0])
                    self.multiplicity[ik] = degen
                    for i in range(degen):
                        line = f.readline().split()
                        self.vel[ik, im, i, :] = np.array(line, dtype=float)
                    for it in range(nstemp):
                        line = f.readline().rstrip()
                        tmp_g = float(line)
                        self.gamma[ik, im, it] = tmp_g

                    if which_phonon == 3 * self.nat * total_irq:
                        break  # every thing is read


KAPPA_H5_SCHEMA = "alamode:kappa_result"
KAPPA_H5_MAX_FORMAT_VERSION = 2


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "h5py is required to read PREFIX.kappa.h5 files (pip install h5py)"
        ) from exc
    return h5py


def _check_kappa_h5_header(f, filename):
    """Validates the schema/format_version attributes of an open kappa.h5 file."""
    schema = f.attrs.get("schema", None)
    if isinstance(schema, bytes):
        schema = schema.decode()
    if schema != KAPPA_H5_SCHEMA:
        raise RuntimeError(
            "{} is not an anphon kappa result file (schema attribute = {!r}, expected {!r})".format(
                filename, schema, KAPPA_H5_SCHEMA
            )
        )
    version = int(f.attrs.get("format_version", 0))
    if version < 1 or version > KAPPA_H5_MAX_FORMAT_VERSION:
        raise RuntimeError(
            "{}: unsupported format_version {} (this reader supports 1..{})".format(
                filename, version, KAPPA_H5_MAX_FORMAT_VERSION
            )
        )
    return version, bool(int(f.attrs.get("temperature_resolved", 0)))


def probe_kappa_h5(filename):
    """Returns a dict with the file-level information of a PREFIX.kappa.h5 file without loading
    the arrays: format_version, temperature_resolved, temperatures, has_4ph, has_isotope."""
    h5py = _import_h5py()
    with h5py.File(filename, "r") as f:
        version, tdep = _check_kappa_h5_header(f, filename)
        return {
            "format_version": version,
            "temperature_resolved": tdep,
            "temperatures": np.array(f["metadata/temperatures"][...], dtype=float).ravel(),
            "has_4ph": "scattering/4ph" in f,
            # isotope linewidths are written when kappa is finalized: require /kappa/valid
            "has_isotope": "scattering/isotope/gamma" in f
            and "kappa/valid" in f
            and bool(np.any(np.array(f["kappa/valid"][...]).ravel().astype(bool))),
        }


class ParseKappaH5:
    """Reader for the unified ``PREFIX.kappa.h5`` file written by anphon (FILE_FORMAT = h5).

    The object exposes the same attributes as :class:`ParseResult` (``kgrid``, ``nat``,
    ``q_coord``, ``gamma``, ``temperatures``, ``omega``, ``multiplicity``, ``volume``, ``vel``,
    ``lattice_vector``, ``atomic_kinds``, ``x_fractional``, ``classical``) for one scattering
    channel (``"3ph"`` or ``"4ph"``), so that it can be used as a drop-in replacement for the
    legacy text readers in :class:`analyzer.calculator.Calculator`.

    The file stores the linewidths of the three-phonon channel (dense mesh) and, when
    ``INCLUDE_4PH = 1`` was used, of the four-phonon channel (``KMESH_COARSE`` mesh) side by
    side, together with the isotope linewidths (``/scattering/isotope``) and the metadata of the
    calculation. Two layouts exist:

    * ``format_version = 1``: harmonic phonons; all temperatures of the run share the same
      frequencies/velocities (``frequencies`` has shape ``(nk_irred, ns)``). All temperature
      columns of ``gamma`` are loaded.
    * ``format_version = 2`` (``temperature_resolved = 1``, runs with ``FC2_TEMPERATURE``): the
      phonon basis depends on the temperature. Because every quantity of the Calculator is
      referred to one basis, only ONE temperature is loaded from such a file: the one closest
      to ``temperature`` (mandatory when the file holds several temperatures); ``temperatures``
      then has length 1 and ``gamma`` a single column. Frequency slices of files written before
      the row-major fix of anphon (no ``layout`` attribute on the dataset) are transposed on
      reading.

    Units are converted to those of the text files: cm^-1 for frequencies and linewidths, m/s
    for group velocities, bohr for the lattice vectors, bohr^3 for the volume.
    Isotope linewidths are exposed through ``gamma_isotope`` only when the run reached the
    stage that writes them (``/kappa/valid`` set for the selected temperature).
    """

    def __init__(self, filename, channel="3ph", temperature=None, verbose=True):
        h5py = _import_h5py()

        if channel not in ("3ph", "4ph"):
            raise ValueError("channel must be '3ph' or '4ph'")

        self.filename = filename
        self.channel = channel
        self.kgrid = np.zeros(3, dtype=int)
        self.nat = 0
        self.q_coord = None
        self.gamma = None
        self.temperatures = None
        self.omega = None
        self.multiplicity = None
        self.volume = None
        self.vel = None
        self.lattice_vector = None
        self.atomic_kinds = None
        self.x_fractional = None
        self.classical = None
        # file-level information that has no counterpart in the text files
        self.format_version = 1
        self.temperature_resolved = False
        self.has_4ph = False
        self.has_isotope = False
        self.gamma_isotope = None
        self.temperature_index = 0
        self.temperatures_all = None
        self.gamma_computed = None
        self.n_missing = 0

        with h5py.File(filename, "r") as f:
            self.format_version, self.temperature_resolved = _check_kappa_h5_header(f, filename)

            cell = f["metadata/PrimitiveCell"]
            self.nat = int(cell["number_of_atoms"][()])
            self.volume = float(cell["volume"][()])
            # The HDF5 file stores the lattice vectors as rows; the text files (and hence the
            # Calculator, which transposes before calling spglib) store them as columns.
            self.lattice_vector = np.array(cell["lattice_vector"][...], dtype=float).T
            self.atomic_kinds = np.array(cell["atomic_kinds"][...], dtype=int)
            self.x_fractional = np.array(cell["fractional_coordinate"][...], dtype=float)
            self.classical = int(f["metadata/classical"][()]) if "metadata/classical" in f else 0
            temps_all = np.array(f["metadata/temperatures"][...], dtype=float).ravel()
            if temps_all.size == 0:
                raise RuntimeError("{}: the file contains no temperatures".format(filename))
            if not np.all(np.isfinite(temps_all)):
                raise RuntimeError("{}: the temperature grid contains non-finite values".format(filename))
            self.temperatures_all = temps_all
            self.has_4ph = "scattering/4ph" in f
            if "scattering/{}".format(channel) not in f:
                raise RuntimeError(
                    "{} does not contain the '{}' scattering channel".format(filename, channel)
                )

            # ---- temperature selection ----
            nt = temps_all.size
            if temperature is not None and not np.isfinite(temperature):
                raise ValueError("temperature must be a finite number, got {}".format(temperature))
            if temperature is None:
                if self.temperature_resolved and nt > 1:
                    raise ValueError(
                        "{} holds temperature-dependent phonons at {} temperatures {}; "
                        "the temperature to load must be given (the Calculator cannot mix "
                        "several phonon bases)".format(filename, nt, temps_all.tolist())
                    )
                it = 0
            else:
                it = int(np.argmin(np.abs(temps_all - temperature)))
                if verbose and abs(temps_all[it] - temperature) > 0 and self.temperature_resolved:
                    print(
                        "# Warning: {} K is not on the temperature grid of {}; using {} K".format(
                            temperature, filename, temps_all[it]
                        )
                    )
            self.temperature_index = it
            if self.temperature_resolved:
                self.temperatures = temps_all[it : it + 1]
            else:
                self.temperatures = temps_all

            # ---- validity of the finalized quantities (isotope group) ----
            valid = None
            if "kappa/valid" in f:
                valid = np.array(f["kappa/valid"][...]).ravel().astype(bool)
            iso_present = "scattering/isotope/gamma" in f
            if iso_present:
                if valid is None or valid.size == 0:
                    iso_ok = False
                elif self.temperature_resolved:
                    iso_ok = bool(valid[it]) if it < valid.size else False
                else:
                    iso_ok = bool(valid[0])
                if not iso_ok and verbose:
                    print(
                        "# Warning: {} contains an isotope group but the run did not reach the "
                        "stage that writes it (kappa not finalized); isotope linewidths ignored".format(
                            filename
                        )
                    )
                self.has_isotope = iso_ok

            # ---- the scattering channel ----
            g = f["scattering/{}".format(channel)]
            self.kgrid = np.array(g.attrs["kmesh"], dtype=int)
            ns = int(g.attrs["nbranches"])
            nk = int(g.attrs["nk_irred"])
            self.q_coord = np.array(g["xk_irred"][...], dtype=float).reshape((nk, 3))
            offsets = np.array(g["equiv_offsets"][...], dtype=int).ravel()
            knum = np.array(g["equiv_knum"][...], dtype=int).ravel()
            if offsets.size != nk + 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0) or offsets[-1] != knum.size:
                raise RuntimeError(
                    "{}: inconsistent equiv_offsets/equiv_knum in the {} channel".format(filename, channel)
                )
            self.multiplicity = np.diff(offsets).astype(float)

            freq = np.array(g["frequencies"][...], dtype=float)
            vel = np.array(g["velocities"][...], dtype=float)
            if self.temperature_resolved:
                if freq.ndim != 3 or vel.ndim != 4:
                    raise RuntimeError(
                        "{}: unexpected frequency/velocity layout for a temperature-resolved file".format(filename)
                    )
                freq = freq[it]
                vel = vel[it]
                # Files written before the row-major fix hold the column-major Eigen buffer;
                # the fixed writer stamps the dataset with layout = "row-major".
                layout = g["frequencies"].attrs.get("layout", None)
                if isinstance(layout, bytes):
                    layout = layout.decode()
                if layout != "row-major":
                    freq = freq.ravel().reshape((ns, nk)).T
                    if verbose:
                        print(
                            "# Note: legacy frequency layout in {} (written before the row-major fix); "
                            "the slice has been transposed on reading.".format(filename)
                        )
            self.omega = np.ascontiguousarray(freq.reshape((nk, ns)))
            if np.any(np.diff(self.omega, axis=1) < -1.0e-6) and verbose:
                print(
                    "# Warning: the branch frequencies of some k points in {} are not in ascending "
                    "order; check the file layout".format(filename)
                )

            # velocities are stored for every equivalent k point, concatenated in the order of
            # equiv_knum; rearrange them like the text files: (nk_irred, ns, max_multiplicity, 3)
            vel = vel.reshape((knum.size, ns, 3))
            nmax = int(self.multiplicity.max())
            self.vel = np.zeros((nk, ns, nmax, 3), dtype=float)
            for ik in range(nk):
                lo, hi = offsets[ik], offsets[ik + 1]
                self.vel[ik, :, : hi - lo, :] = np.transpose(vel[lo:hi], (1, 0, 2))

            gamma = np.array(g["gamma"][...], dtype=float)
            if gamma.ndim == 1:
                gamma = gamma[:, None]
            gamma = gamma.reshape((nk, ns, gamma.shape[-1]))
            if "gamma_computed" in g:
                done = np.array(g["gamma_computed"][...]).astype(bool)
                if done.ndim == 1:
                    done = np.repeat(done[:, None], gamma.shape[-1], axis=1)
                done = done.reshape((nk, ns, done.shape[-1]))
            else:
                done = np.ones(gamma.shape, dtype=bool)
                if verbose:
                    print("# Warning: {} has no gamma_computed flags; all entries assumed computed".format(filename))
            if self.temperature_resolved:
                gamma = gamma[:, :, it : it + 1]
                done = done[:, :, it : it + 1]
            self.gamma = gamma
            self.gamma_computed = done
            self.n_missing = int(np.sum(~done))
            if self.n_missing:
                if verbose:
                    print(
                        "# Warning: {} entries of the {} linewidths in {} were not computed "
                        "(incomplete or interrupted run); they are set to NaN and the modes are "
                        "excluded from sums".format(self.n_missing, channel, filename)
                    )
                self.gamma[~done] = np.nan

            if channel == "3ph" and self.has_isotope:
                giso = np.array(f["scattering/isotope/gamma"][...], dtype=float)
                if giso.ndim == 3:  # (nt, nk_irred, ns)
                    giso = giso[it]
                self.gamma_isotope = giso.reshape((nk, ns))
