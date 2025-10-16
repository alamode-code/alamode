#
# GenDisplacement.py
#
# Class to generate displacements of atoms
#
# Copyright (c) 2020 Terumasa Tadano
#
# This file is distributed under the terms of the MIT license.
# Please see the file 'LICENCE.txt' in the root directory
# or http://opensource.org/licenses/mit-license.php for information.
#
from __future__ import print_function

import cmath
import copy
import math
import random

import numpy as np


class AlamodeDisplace(object):
    def __init__(
        self,
        displacement_mode,
        codeobj_base,
        random_seed=12345,
        file_evec=None,
        file_primitive=None,
        verbosity=1,
        omega_min=None,
        truncate_k=None,
        cap_sigma_by_nn=False,
        nn_frac=None
    ):
        self._BOHR_TO_ANGSTROM = 0.5291772108
        self._K_BOLTZMANN = 1.3806488e-23
        self._RYDBERG_TO_JOULE = 4.35974394e-18 / 2.0
        amu = 1.660538782e-27
        electron_mass = 9.10938215e-31
        self._AMU_RYD = amu / electron_mass / 2.0
        self._KAYSER_TO_RYD = 1.0 / 109737.3
    
        self._pattern = []
        self._primitive_lattice_vector = None
        self._inverse_primitive_lattice_vector = None
        self._primitive_kd = None
        self._nat_primitive = 0
        self._xp_fractional = None
        self._counter = 1
        self._displacement_magnitude = 0.02
        self._md_snapshots = None  # displacements in angstrom unit
        self._verbosity = verbosity
        self._classical = False
        self._commensurate_qpoints = []
        self._mapping_shift = None
        self._mapping_s2p = None
        self._qpoints = None
        self._omega2 = None
        self._evec = None
        self._qlist_real = []
        self._qlist_uniq = []
        self._mass = None
        self._nmode = None
        self._updated_structure = {}

        # ===== Safety & robustness knobs (defaults chosen to be conservative) =====
        # Floor for tiny positive ω (Rydberg a.u.). Tweak per system.
        if omega_min is not None:
            self._omega_min = omega_min * self._KAYSER_TO_RYD
        else:
            self._omega_min = 10.0 * self._KAYSER_TO_RYD

        # Truncated normal (±kσ) to kill rare catastrophic draws.
        # Implementation default is "clip". See _trunc_normal for alternatives.
        if truncate_k is not None:
            self._truncate_k = truncate_k
        else:
            self._truncate_k = 4.0
        self._truncate_mode = "clip"  # "clip" | "reject" | "reject_varmatch"

        # Cap σ so a single (q,ν) cannot move any atom more than
        # a fraction of its nearest-neighbor distance in real space.
        if cap_sigma_by_nn is not None:
            self._cap_sigma_by_nn = cap_sigma_by_nn
        else:
            self._cap_sigma_by_nn = True

        if nn_frac is not None:
            self._nn_frac = nn_frac
        else:
            self._nn_frac = 0.20  # each mode limited so σ*C ≤ nn_frac * d_nn per atom

        # Γ-acoustic identification by projection (no index assumptions)
        self._gamma_overlap_thresh = 0.95

        # Cache for diagnostics
        self._omega_eff_cache = None

        self._displacement_mode = displacement_mode.lower()
        self._supercell = codeobj_base
        self._random_seed = random_seed

        if file_primitive:
            primitive_cell = copy.deepcopy(codeobj_base)
            primitive_cell.load_initial_structure(file_primitive)

            self._primitive_lattice_vector = primitive_cell._lattice_vector
            self._inverse_primitive_lattice_vector = (
                primitive_cell._inverse_lattice_vector
            )
            self._xp_fractional = primitive_cell.x_fractional
            self._nat_primitive = primitive_cell.nat
            self._primitive_kd = primitive_cell._kd
            self._find_commensurate_q()
            self._generate_mapping_s2p()

        else:
            if (
                self._displacement_mode == "random_normalcoordinate"
                or self._displacement_mode == "pes"
            ):
                raise RuntimeError(
                    "The --prim option is necessary when '--random_normalcoord' "
                    "or '--pes' is invoked."
                )

        if file_evec:
            self._load_phonon_results(file_evec)
        else:
            if self._displacement_mode == "pes":
                raise RuntimeError(
                    "The --evec option is necessary when '--pes' is invoked."
                )

            if self._displacement_mode == "random_normalcoordinate":
                print(
                    "The --evec option is necessary when '--random_normalcoord'\n"
                    "option is used. \n"
                    "Please generate a PREFIX.evec file by using the ANPHON code\n"
                    "with the following inputs and then run displace.py again with\n"
                    "--evec=PREFIX.evec option:\n\n"
                )

                print("&cell")
                print("1.0")
                for elem in self._primitive_lattice_vector.transpose():
                    for i in range(3):
                        print("%20.15f" % (elem[i] / self._BOHR_TO_ANGSTROM), end="")
                    print("")
                print("/")
                print("&kpoint")
                print("0")
                for elem in self._commensurate_qpoints:
                    print("%20.15f %20.15f %20.15f" % (elem[0], elem[1], elem[2]))
                print("/")
                print("&analysis")
                print(" PRINTEVEC = 1")
                print("/")
                exit(0)

    def generate(
        self,
        file_pattern=None,
        file_mddata=None,
        option_every=None,
        magnitude=0.00,
        number_of_displacements=1,
        temperature=None,
        classical=False,
        option_pes=None,
        option_qrange=None,
        ignore_imag=False,
        imag_evec=False,
    ):
        self._counter = 1
        self._displacement_magnitude = magnitude
        self._classical = classical

        header_list = []
        disp_list = []

        if self._displacement_mode == "fd":
            if not file_pattern:
                raise RuntimeError("pattern file must be given with --pattern option")
            self._parse_displacement_patterns(file_pattern)

            if self._verbosity > 0:
                print(" Displacement mode              : Finite displacement\n")
                print(
                    " %d displacement pattern are generated from\n"
                    " the given *.pattern_* files" % len(self._pattern)
                )
                print("")

            try:
                import yaml

                with open(file_pattern[0], "r") as f:
                    obj = yaml.safe_load(f)
                    nat_from_pattern = (
                        len(obj["structure"]["supercell"]["coordinates"]) // 3
                    )
                    if nat_from_pattern % self._supercell.nat != 0:
                        raise RuntimeError(
                            "The number of atoms in prefix.pattern_* is not integral multiple of that of "
                            "the reference structure."
                        )

                    if nat_from_pattern // self._supercell.nat > 1:
                        self._set_updated_structure(obj["structure"]["supercell"])
            except ImportError:
                pass

            for pattern in self._pattern:
                header, disp = self._get_finite_displacement(pattern)
                self._counter += 1
                header_list.append(header)
                disp_list.append(disp)

            return header_list, disp_list, self._updated_structure

        if self._displacement_mode == "random":
            if self._verbosity > 0:
                print(" Displacement mode              : Random displacement\n")
                print(
                    " %d randomly-displaced configurations are generated from\n"
                    " the original supercell structure" % number_of_displacements
                )
                print(
                    " The direction of displacement is random, but the displacement\n"
                    " magnitude of each atom is fixed to %.2f Angstrom"
                    % self._displacement_magnitude
                )
                print("")

            disp_random = self._get_random_displacements(
                number_of_displacements, "gauss"
            )
            for i in range(number_of_displacements):
                header = "Random disp. with mag %f : %i" % (
                    self._displacement_magnitude,
                    self._counter,
                )
                header_list.append(header)
                disp_list.append(disp_random[i])
                self._counter += 1

            return header_list, disp_list, self._updated_structure

        if self._displacement_mode == "md":
            list_every = self._sample_md_snapshots(file_mddata, option_every)

            if self._verbosity > 0:
                print(" Displacement mode              : MD sampling\n")
                print(
                    " Sampling range and interval: [%d:%d], interval = %d"
                    % (list_every[0] + 1, list_every[1], list_every[2])
                )
                print(
                    " %d snapshots are sampled from the LOAD_MDDATA file(s)"
                    % len(self._md_snapshots)
                )
                print("")

            ndisp = len(self._md_snapshots)

            for i in range(ndisp):
                header = "Snapshot sampled from MD data: %i" % self._counter
                disp_tmp = self._md_snapshots[i]

                # Convert disp_tmp in fractional coordinates
                for j in range(self._supercell.nat):
                    disp_tmp[j] = np.dot(
                        disp_tmp[j], self._supercell.inverse_lattice_vector.transpose()
                    )

                header_list.append(header)
                disp_list.append(disp_tmp)
                self._counter += 1

            return header_list, disp_list, self._updated_structure

        if self._displacement_mode == "md_plus_random":
            list_every = self._sample_md_snapshots(file_mddata, option_every)

            if self._verbosity > 0:
                print(
                    " Displacement mode              : MD sampling + random displacement\n"
                )
                print(
                    " Sampling range and interval: [%d:%d], interval = %d"
                    % (list_every[0] + 1, list_every[1], list_every[2])
                )
                print(
                    " %d snapshots are sampled from the LOAD_MDDATA file(s)"
                    % len(self._md_snapshots)
                )
                print(
                    " Then, random displacements of %.2f Angstrom are\n"
                    " added to each atom in each snapshot.\n"
                    " The direction of displacement is random."
                    % self._displacement_magnitude
                )
                print("")

            ndisp = len(self._md_snapshots)
            disp_random = self._get_random_displacements(ndisp, "gauss")

            for i in range(ndisp):
                header = "Random disp. with mag %f on top of sampled snapshots: %i" % (
                    self._displacement_magnitude,
                    self._counter,
                )
                disp_tmp = self._md_snapshots[i]

                # Convert disp_tmp in fractional coordinates
                for j in range(self._supercell.nat):
                    disp_tmp[j] = np.dot(
                        disp_tmp[j], self._supercell.inverse_lattice_vector.transpose()
                    )

                disp_tmp += disp_random[i]
                header_list.append(header)
                disp_list.append(disp_tmp)
                self._counter += 1

            return header_list, disp_list, self._updated_structure

        if self._displacement_mode == "random_normalcoordinate":
            if self._verbosity > 0:
                print(
                    " Displacement mode              : Random displacement in normal coordinate\n"
                )
            if temperature is None:
                raise RuntimeError("The temperature must be given with the --temp option.")
            if self._verbosity > 0:
                print(
                    " %d randomly-displaced configurations are generated from\n"
                    " the original supercell structure" % number_of_displacements
                )
                print(
                    " The normal coordinate distribution is generated randomly from \n"
                    " the following conditions:\n\n"
                    " Temperature : %f K" % temperature
                )
                if self._classical:
                    print(" Statistics  : Classical (no zero-point vibration)")
                else:
                    print(" Statistics  : Quantum (with zero-point vibration)")
                print("")

                if self._omega_min > 0.0:
                    print(
                        " A frequency floor of %.2f cm^-1 is applied to all modes"
                        % (self._omega_min / self._KAYSER_TO_RYD)
                    )
                    print("")
                    
                if self._cap_sigma_by_nn:
                    print(
                        " σ is capped so that each mode cannot move any atom more than"
                    )
                    print(
                        " %.2f times the nearest-neighbor distance in real space"
                        % self._nn_frac
                    )
                    print("")

            disp_random = self._get_random_displacements_normalcoordinate(
                number_of_displacements, temperature, ignore_imag
            )
            for i in range(number_of_displacements):
                header = "Random disp. in normal coordinates at T = %f K: %i" % (
                    temperature,
                    self._counter,
                )
                header_list.append(header)
                disp_list.append(disp_random[:, :, i])
                self._counter += 1

            return header_list, disp_list, self._updated_structure

        if self._displacement_mode == "pes":
            if option_pes is None or option_qrange is None:
                raise RuntimeError("--pes and --Qrange options should be given.")

            target_q, target_mode = [int(t) - 1 for t in option_pes.split()]
            Qmin, Qmax = option_qrange.split()
            Qlist = np.linspace(float(Qmin), float(Qmax), number_of_displacements)

            if self._verbosity > 0:
                print(
                    " Displacement mode              : Along specific normal coordinate\n"
                )
                print(
                    " %d configurations are generated from\n"
                    " the original supercell structure" % number_of_displacements
                )
                print(
                    " The normal coordinate of the following phonon mode is excited:\n\n"
                    " xk = %f %f %f"
                    % (
                        self._qpoints[target_q, 0],
                        self._qpoints[target_q, 1],
                        self._qpoints[target_q, 2],
                    )
                )
                print(" branch : %d" % (target_mode + 1))
                print("")

            disp_pes = self._generate_displacements_pes(
                np.array(Qlist), target_q, target_mode, imag_evec
            )

            for i in range(number_of_displacements):
                header = (
                    "Displacement along normal coordinate Q (q= %f %f %f, branch : %d), "
                    "magnitude = %f (u^{1/2} ang)"
                    % (
                        self._qpoints[target_q, 0],
                        self._qpoints[target_q, 1],
                        self._qpoints[target_q, 2],
                        target_mode + 1,
                        Qlist[i],
                    )
                )

                header_list.append(header)
                disp_list.append(disp_pes[:, :, i])
                self._counter += 1

            return header_list, disp_list, self._updated_structure

        else:
            raise RuntimeError("This cannot happen")

    def _sample_md_snapshots(self, file_mddata, str_every):
        disp_merged = []

        try:
            # Get displacements in angstrom unit
            disp_merged = self._supercell.get_displacements(file_mddata, "angstrom")
        except ValueError:
            try:
                for target in file_mddata:
                    disp = np.loadtxt(target, dtype=float)
                    disp *= self._BOHR_TO_ANGSTROM
                    disp_merged.extend(
                        np.reshape(
                            disp,
                            (len(disp) // self._supercell.nat, self._supercell.nat, 3),
                        )
                    )
            except ValueError as err:
                raise RuntimeError("Failed to read the MD files") from err

        list_str_every = str_every.strip().split(":")
        start = 0
        end = len(disp_merged)
        if len(list_str_every) == 1:
            interval = int(list_str_every[0])
        elif len(list_str_every) == 3:
            start = int(list_str_every[0]) - 1
            end = int(list_str_every[1])
            interval = int(list_str_every[2])

            if start > end:
                raise RuntimeError(
                    "In the --every option, start must not be larger than end."
                )

            if start > len(disp_merged) or end > len(disp_merged):
                raise RuntimeError(
                    "The range specified by --every is larger than the loaded MD data."
                )

        else:
            raise RuntimeError("Invalid format of the --every option.")

        self._md_snapshots = disp_merged[start:end:interval]

        return [start, end, interval]

    def _set_updated_structure(self, yamlobj):
        lavec = np.reshape(np.array(yamlobj["lattice_vectors"]), (3, 3)).transpose()
        nat_updated = len(yamlobj["coordinates"]) // 3
        xf = np.reshape(np.array(yamlobj["coordinates"], dtype=float), (nat_updated, 3))
        self._updated_structure["lattice_vector"] = lavec
        self._updated_structure["inverse_lattice_vector"] = np.linalg.inv(lavec)
        self._updated_structure["kd"] = yamlobj["atomic_index"]
        self._updated_structure["x_fractional"] = xf
        self._updated_structure["nat"] = nat_updated

    def _parse_displacement_patterns(self, files_in):
        import yaml

        self._pattern = []

        for file in files_in:
            pattern_tmp = []

            f = open(file, "r")

            try:
                # parse assuming the YAML format
                obj = yaml.safe_load(f)

                if obj["displacements"]["basis"] == "F":
                    raise RuntimeError("DBASIS must be 'C'")

                for pattern in obj["displacements"]["patterns"]:
                    atoms = pattern["atoms"]
                    directions = pattern["directions"]
                    natom_move = len(atoms)

                    pattern_set = []
                    for i in range(natom_move):
                        disp = []
                        disp.append(atoms[i])
                        for j in range(3):
                            disp.append(float(directions[3 * i + j]))

                        pattern_set.append(disp)

                    pattern_tmp.append(pattern_set)

            except ValueError:
                # If failed, assume the old format

                tmp, basis = f.readline().rstrip().split(":")
                if basis == "F":
                    raise RuntimeError("DBASIS must be 'C'") from None

                while True:
                    line = f.readline()
                    if not line:
                        break

                    line_split_by_colon = line.rstrip().split(":")
                    is_entry = len(line_split_by_colon) == 2

                    if is_entry:
                        pattern_set = []
                        natom_move = int(line_split_by_colon[1])
                        for _ in range(natom_move):
                            disp = []
                            line = f.readline()
                            line_split = line.rstrip().split()
                            disp.append(int(line_split[0]))
                            for j in range(3):
                                disp.append(float(line_split[j + 1]))

                            pattern_set.append(disp)
                        pattern_tmp.append(pattern_set)

            for entry in pattern_tmp:
                if entry not in self._pattern:
                    self._pattern.append(entry)

            f.close()

    def _get_finite_displacement(self, pattern):
        header = "Disp. Num. %i" % self._counter
        header += " ( %f Angstrom" % self._displacement_magnitude

        if self._updated_structure:
            nat_ref = self._updated_structure["nat"]
            disp = np.zeros((nat_ref, 3))
            invlavec = (
                self._updated_structure["inverse_lattice_vector"]
                / self._BOHR_TO_ANGSTROM
            )
        else:
            nat_ref = self._supercell.nat
            disp = np.zeros((nat_ref, 3))
            invlavec = self._supercell.inverse_lattice_vector

        for displace in pattern:
            atom = displace[0] - 1
            header += ", %i : " % displace[0]
            str_direction = ""

            for i in range(3):
                if abs(displace[i + 1]) > 1.0e-10:
                    if displace[i + 1] > 0.0:
                        str_direction += "+" + self._char_xyz(i)
                    else:
                        str_direction += "-" + self._char_xyz(i)

                disp[atom][i] += displace[i + 1] * self._displacement_magnitude
            header += str_direction
        header += ")"

        if invlavec is not None:
            for i in range(nat_ref):
                disp[i] = np.dot(disp[i], invlavec.T)

        return header, disp

    def _get_random_displacements(self, ndata, mode="gauss"):
        """
        Return random displacements in fractional coordinates
        """
        disp_xyz = np.zeros(3)
        disp_random = np.zeros((ndata, self._supercell.nat, 3))

        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)

        if mode == "gauss":
            for idata in range(ndata):
                for i in range(self._supercell.nat):
                    for j in range(3):
                        # Generate a random number following the Gaussian distribution
                        disp_xyz[j] = random.gauss(0.0, 1.0)

                    # Normalize the random displacement so that it has the norm
                    # of self._displacement_magnitude.
                    norm = np.linalg.norm(disp_xyz)
                    disp_random[idata, i, :] = (
                        disp_xyz[:] / norm * self._displacement_magnitude
                    )

                    # Transform to the fractional coordinate
                    disp_random[idata, i] = np.dot(
                        disp_random[idata, i],
                        self._supercell.inverse_lattice_vector.transpose(),
                    )

        elif mode == "uniform":
            for idata in range(ndata):
                for i in range(self._supercell.nat):
                    for j in range(3):
                        # Generate a random number following the Gaussian distribution
                        disp_xyz[j] = random.uniform(
                            -self._displacement_magnitude, self._displacement_magnitude
                        )

                    # Transform to the fractional coordinate
                    disp_random[idata, i] = np.dot(
                        disp_xyz[:], self._supercell.inverse_lattice_vector.transpose()
                    )
        else:
            raise RuntimeError(
                "Invalid option for the random number distribution types."
            )

        return disp_random

    def _get_random_displacements_normalcoordinate(
        self, ndata, temperature, ignore_imag
    ):
        if temperature is None:
            raise RuntimeError("The temperature must be given with the --temp option.")

        nq = len(self._qpoints)
        Q_R = np.zeros((nq, self._nmode, ndata))
        Q_I = np.zeros((nq, self._nmode, ndata))

        # get sigma in units of bohr*amu_ry^(1/2)
        sigma = self._get_gaussian_sigma(temperature, ignore_imag)

        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)

        for iq in range(nq):
            for imode in range(self._nmode):
                if sigma[iq, imode] < 1.0e-12:
                    Q_R[iq, imode, :] = 0.0
                    Q_I[iq, imode, :] = 0.0
                else:
                    Q_R[iq, imode, :] = self._trunc_normal(
                        ndata, sigma[iq, imode], self._truncate_k
                    )
                    Q_I[iq, imode, :] = self._trunc_normal(
                        ndata, sigma[iq, imode], self._truncate_k
                    )
                # print(self._omega2[iq, imode], Q_R[iq, imode, :], Q_I[iq, imode, :])

        disp = np.zeros((self._supercell.nat, 3, ndata))

        for iat in range(self._supercell.nat):
            xshift = self._mapping_shift[iat]
            jat = self._mapping_s2p[iat]
            for iq in self._qlist_real:
                xq_tmp = self._qpoints[iq, :]
                phase_real = math.cos(2.0 * math.pi * np.dot(xq_tmp, xshift))
                for imode in range(self._nmode):
                    for icrd in range(3):
                        disp[iat, icrd, :] += (
                            Q_R[iq, imode, :]
                            * self._evec[iq, imode, 3 * jat + icrd].real
                            * phase_real
                        )

            for iq in self._qlist_uniq:
                xq_tmp = self._qpoints[iq, :]
                phase = cmath.exp(
                    complex(0.0, 1.0) * 2.0 * math.pi * np.dot(xq_tmp, xshift)
                )
                for imode in range(self._nmode):
                    for icrd in range(3):
                        ctmp = self._evec[iq, imode, 3 * jat + icrd] * phase
                        disp[iat, icrd, :] += math.sqrt(2.0) * (
                            Q_R[iq, imode, :] * ctmp.real
                            - Q_I[iq, imode, :] * ctmp.imag
                        )

        factor = np.zeros(self._supercell.nat)
        for iat in range(self._supercell.nat):
            factor[iat] = 1.0 / math.sqrt(
                self._mass[self._supercell.atomic_kinds[iat]]
                * self._AMU_RYD
                * float(nq)
            )

        for idata in range(ndata):
            for i in range(3):
                # convert the unit of disp from bohr*amu_ry^(1/2) to angstrom
                disp[:, i, idata] = (
                    factor[:] * disp[:, i, idata] * self._BOHR_TO_ANGSTROM
                )

            # Transform to the fractional coordinate
            for iat in range(self._supercell.nat):
                disp[iat, :, idata] = np.dot(
                    disp[iat, :, idata],
                    self._supercell.inverse_lattice_vector.transpose(),
                )

        return disp

    def _get_gaussian_sigma(self, temp, ignore_imag):
        """
        Computes the deviation of Q, i.e., sqrt{<Q^2>} from the phonon frequency
        and input temperature. Since omega is defined in the Rydberg atomic unit,
        the return value (sigma) is also in the Rydberg atomic unit (bohr*amu_ry^(1/2)).
        """

        nq = len(self._qpoints)
        nmode = self._nmode
        omega_raw = np.zeros((nq, nmode))
        omega_eff = np.zeros((nq, nmode))
        sigma = np.zeros((nq, nmode))

        # 1) Build ω from ω^2, handling imaginary mode policy
        for iq in range(nq):
            for imode in range(nmode):
                if self._omega2[iq, imode] < 0.0:
                    if ignore_imag:
                        omega_raw[iq, imode] = 0.0
                        if self._verbosity > 0:
                            print(
                                "Warning: Detected imaginary mode at iq = %d, imode = %d.\n"
                                "This mode will be ignored.\n" % (iq + 1, imode + 1)
                            )
                    else:
                        omega_raw[iq, imode] = math.sqrt(-self._omega2[iq, imode])
                        if self._verbosity > 0:
                            print(
                                "Warning: Detected imaginary mode at iq = %d, imode = %d.\n"
                                "Use the absolute frequency for this mode.\n"
                                % (iq + 1, imode + 1)
                            )
                else:
                    omega_raw[iq, imode] = math.sqrt(self._omega2[iq, imode])

        # 2) Γ-acoustic identification by projection; set huge ω to suppress σ
        gamma_acoustic = self._identify_gamma_acoustic(
            overlap_thresh=self._gamma_overlap_thresh,
            max_modes=3,
            verbose=True,
        )

        # 3) Apply ω floor + Γ-acoustic suppression to build ω_eff
        n_floored = 0
        floored_modes = []
        for iq in range(nq):
            for imode in range(nmode):
                w = omega_raw[iq, imode]
                # Check if this (q,mode) is tagged as Γ-acoustic
                is_gamma_acoustic = iq in gamma_acoustic and gamma_acoustic[iq][imode]
                
                # If Γ-acoustic, force σ→0 by huge ω (skip frequency floor)
                if is_gamma_acoustic:
                    weff = 1.0e6
                else:
                    # Apply floor for tiny positive ω (only for non-acoustic modes)
                    weff = max(w, self._omega_min) if w > 0.0 else w
                    if w > 0.0 and w < self._omega_min:
                        n_floored += 1
                        if self._verbosity > 0:
                            omega_kayser = w / self._KAYSER_TO_RYD
                            floored_modes.append((iq + 1, imode + 1, omega_kayser))
                
                omega_eff[iq, imode] = weff

        if self._verbosity > 0 and n_floored > 0:
            print(f" [{n_floored} modes affected by frequency floor]")
            print(f"  Frequency floor: {self._omega_min / self._KAYSER_TO_RYD:.2f} cm^-1")
            if n_floored <= 10:
                print("  Affected modes:")
                for iq, imode, omega_kayser in floored_modes:
                    print(f"    iq={iq}, imode={imode}: omega = {omega_kayser:.4f} cm^-1")
            else:
                print(f"  ({n_floored} modes total, showing first 10)")
                print("  Affected modes:")
                for iq, imode, omega_kayser in floored_modes[:10]:
                    print(f"    iq={iq}, imode={imode}: omega = {omega_kayser:.4f} cm^-1")
            print("")

        # 4) Compute σ from ω_eff (quantum or classical)
        for iq in range(nq):
            for imode in range(nmode):
                if omega_eff[iq, imode] > 1.0e-12:
                    if self._classical:
                        sigma[iq, imode] = math.sqrt(
                            self._n_classical(omega_eff[iq, imode], temp)
                            / omega_eff[iq, imode]
                        )
                    else:
                        sigma[iq, imode] = math.sqrt(
                            (1.0 + 2.0 * self._n_bose(omega_eff[iq, imode], temp))
                            / (2.0 * omega_eff[iq, imode])
                        )
                else:
                    sigma[iq, imode] = 0.0

        # 5) Option 1: cap σ so typical displacement (σ*C) per mode
        #    doesn't exceed nn-fraction. Note: truncation at ±k*σ may
        #    occasionally allow slightly larger displacements.
        if self._cap_sigma_by_nn:
            dnn = self._nearest_neighbor_distances()  # Å, per atom
            Cgain = self._per_mode_realspace_gain()  # Å per unit Q
            f = self._nn_frac
            n_capped = 0
            capped_modes = []
            for iq in range(nq):
                for imode in range(nmode):
                    if sigma[iq, imode] <= 0.0:
                        continue
                    Ci = Cgain[iq, imode, :]  # (nat,)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        caps = np.where(Ci > 0.0, (f * dnn) / Ci, np.inf)
                    sig_cap = np.min(caps)
                    if np.isfinite(sig_cap) and sig_cap < sigma[iq, imode]:
                        if self._verbosity > 0:
                            sigma_old = sigma[iq, imode]
                            # Use omega_eff (floor-corrected) for clarity
                            omega_kayser = (omega_eff[iq, imode]
                                            / self._KAYSER_TO_RYD)
                            capped_modes.append((iq + 1, imode + 1, omega_kayser,
                                                 sigma_old, sig_cap))
                        n_capped += 1
                        sigma[iq, imode] = sig_cap

            if self._verbosity > 0 and n_capped > 0:
                print(f" [{n_capped} modes affected by nearest-neighbor sigma capping]")
                print(f"  Parameters: nn_frac={f:.2f}")
                if n_capped <= 10:
                    print("  Affected modes (sigma: old -> new):")
                    for iq, imode, omega, s_old, s_new in capped_modes:
                        print(f"    iq={iq}, imode={imode} "
                              f"(omega={omega:.2f} cm^-1): "
                              f"{s_old:.6f} -> {s_new:.6f}")
                else:
                    print(f"  ({n_capped} modes total, showing first 10)")
                    print("  Affected modes (sigma: old -> new):")
                    for iq, imode, omega, s_old, s_new in capped_modes[:10]:
                        print(f"    iq={iq}, imode={imode} "
                              f"(omega={omega:.2f} cm^-1): "
                              f"{s_old:.6f} -> {s_new:.6f}")
                print("")

        # Cache ω_eff for diagnostics
        self._omega_eff_cache = omega_eff.copy()
        return sigma

    def _generate_displacements_pes(self, Q_array, iq, imode, use_imaginary_part):
        tol_zero = 1.0e-3
        Q_R = Q_array  # in units of u^{1/2} Angstrom

        ndata = len(Q_R)
        xq_tmp = self._qpoints[iq, :]

        is_commensurate = False
        for xq in self._commensurate_qpoints:
            xtmp = (xq - xq_tmp) % 1.0
            if math.sqrt(np.dot(xtmp, xtmp)) < tol_zero:
                is_commensurate = True
                break
        if not is_commensurate:
            raise RuntimeWarning(
                "The q point specified by --pes is not commensurate with the supercell."
            )

        disp = np.zeros((self._supercell.nat, 3, ndata))

        for iat in range(self._supercell.nat):
            xshift = self._mapping_shift[iat, :]
            jat = self._mapping_s2p[iat]
            phase_base = 2.0 * math.pi * np.dot(xq_tmp, xshift)
            cexp_phase = cmath.exp(1.0j * phase_base)

            if use_imaginary_part:
                for icrd in range(3):
                    disp[iat, icrd, :] += (
                        Q_R[:]
                        * (self._evec[iq, imode, 3 * jat + icrd] * cexp_phase).imag
                    )
            else:
                for icrd in range(3):
                    disp[iat, icrd, :] += (
                        Q_R[:]
                        * (self._evec[iq, imode, 3 * jat + icrd] * cexp_phase).real
                    )

        factor = np.zeros(self._supercell.nat)
        for iat in range(self._supercell.nat):
            # factor[iat] = 1.0 / math.sqrt(self._mass[self._supercell.atomic_kinds[iat]] * float(nq))
            factor[iat] = 1.0 / math.sqrt(self._mass[self._supercell.atomic_kinds[iat]])

        for idata in range(ndata):
            for i in range(3):
                disp[:, i, idata] = factor[:] * disp[:, i, idata]

            # Transform to the fractional coordinate
            # The lattice vector is defined in units of Angstrom, so this operation is fine.
            for iat in range(self._supercell.nat):
                disp[iat, :, idata] = np.dot(
                    disp[iat, :, idata],
                    self._supercell.inverse_lattice_vector.transpose(),
                )

        if use_imaginary_part:
            if np.max(np.abs(disp)) < 1.0e-10:
                print(
                    "Warning. All displacements are zero for the phonon modes specified by --pes."
                )

        return disp

    def _find_fraction(self, x, tol=1e-6):
        """
        Find a fraction a/b that approximates the floating value x within a tolerance.
        """
        negative = x < 0
        x = abs(x)

        h1, h2, k1, k2 = 1, 0, 0, 1
        b = x

        while True:
            a = int(b)
            aux = h1
            h1 = a * h1 + h2
            h2 = aux
            aux = k1
            k1 = a * k1 + k2
            k2 = aux

            if abs(x - h1 / k1) < tol:
                if negative:
                    h1 = -h1
                return h1, k1

            b = 1 / (b - a)

    # The following get_commensurate_qlist is unused here, but kept for API parity.
    def get_commensurate_qlist(self):
        tol_zero = 1.0e-3

        nqmax = (
            self.supercell.get_global_number_of_atoms()
            // self.primitive.get_global_number_of_atoms()
        )
        convertor = np.dot(
            np.linalg.inv(self.supercell.get_cell()).T, self.primitive.get_cell().T
        )

        for i in range(3):
            for j in range(3):
                numerator, denominator = self.find_fraction(
                    convertor[i, j], tol=tol_zero
                )
                convertor[i, j] = float(numerator) / float(denominator)

        comb = np.array(
            [
                [Lx * sx, Ly * sy, Lz * sz]
                for Lx in range(10)
                for Ly in range(10)
                for Lz in range(10)
                for sx in (1, -1)
                for sy in (1, -1)
                for sz in (1, -1)
            ]
        )

        qlist = []

        for entry in comb:
            qvec = np.dot(entry, convertor) % 1.0
            qvec = np.where(qvec >= 0.5, qvec - 1.0, qvec)
            qvec = np.where(qvec < -0.5, qvec + 1.0, qvec)

            # Check if qvec is close to any vector in qlist
            is_duplicate = False
            for elem in qlist:
                diff = (elem - qvec + 1.0) % 1.0
                diff = np.where(diff >= 0.5, diff - 1.0, diff)
                if np.linalg.norm(diff) < tol_zero:
                    is_duplicate = True
                    break

            if not is_duplicate:
                qlist.append(qvec)
                if len(qlist) == nqmax:
                    break

        return qlist

    def _find_commensurate_q(self):
        tol_zero = 1.0e-3

        nqmax = self._supercell.nat // self._nat_primitive
        convertor = np.dot(
            self._supercell.inverse_lattice_vector, self._primitive_lattice_vector
        )

        for i in range(3):
            for j in range(3):
                numerator, denominator = self._find_fraction(
                    convertor[i, j], tol=tol_zero
                )
                convertor[i, j] = float(numerator) / float(denominator)

        comb = np.array(
            [
                [Lx * sx, Ly * sy, Lz * sz]
                for Lx in range(10)
                for Ly in range(10)
                for Lz in range(10)
                for sx in (1, -1)
                for sy in (1, -1)
                for sz in (1, -1)
            ]
        )

        qlist = []

        for entry in comb:
            qvec = np.dot(entry, convertor) % 1.0
            qvec = np.where(qvec >= 0.5, qvec - 1.0, qvec)
            qvec = np.where(qvec < -0.5, qvec + 1.0, qvec)

            # Check if qvec is close to any vector in qlist
            is_duplicate = False
            for elem in qlist:
                diff = (elem - qvec + 1.0) % 1.0
                diff = np.where(diff >= 0.5, diff - 1.0, diff)
                if np.linalg.norm(diff) < tol_zero:
                    is_duplicate = True
                    break

            if not is_duplicate:
                qlist.append(qvec)
                if len(qlist) == nqmax:
                    break

        self._commensurate_qpoints = qlist

    def _generate_mapping_s2p(self):
        tol_zero = 1.0e-3
        convertor = np.dot(
            self._supercell.lattice_vector.transpose(),
            self._inverse_primitive_lattice_vector.transpose(),
        )

        for i in range(3):
            for j in range(3):
                convertor[i, j] = float(round(convertor[i, j]))

        shift = np.zeros((self._supercell.nat, 3))
        map_s2p = np.zeros(self._supercell.nat, dtype=int)

        for iat in range(self._supercell.nat):
            xtmp = self._supercell.x_fractional[iat, :]
            xnew = np.dot(xtmp, convertor)

            iloc = -1

            for jat in range(self._nat_primitive):
                xp = self._xp_fractional[jat, :]
                xdiff = np.array((xnew - xp) % 1.0)
                for i in range(3):
                    if xdiff[i] >= 0.5:
                        xdiff[i] -= 1.0
                diff = math.sqrt(np.dot(xdiff[:], xdiff[:]))
                if diff < tol_zero:
                    iloc = jat
                    break
            if iloc == -1:
                raise RuntimeError("Equivalent atom not found")

            map_s2p[iat] = iloc
            shift[iat, :] = [
                float(round(xnew[i] - self._xp_fractional[iloc, i])) for i in range(3)
            ]

        self._mapping_shift = shift
        self._mapping_s2p = map_s2p

    def _n_bose(self, omega, temperature):
        if abs(temperature) < 1.0e-15 or omega < 1.0e-10:
            return 0.0
        else:
            temperature_au = (
                self._K_BOLTZMANN * temperature / self._RYDBERG_TO_JOULE
            )
            x = omega / temperature_au
            # Prevent overflow when x is too large
            # (e.g., when omega=1e6 for acoustic mode suppression)
            # For x > ~700, exp(x) would overflow.
            # At large x, n_BE ≈ exp(-x) ≈ 0
            if x > 100.0:
                return 0.0
            return 1.0 / (math.exp(x) - 1.0)

    def _n_classical(self, omega, temperature):
        if abs(temperature) < 1.0e-15 or omega < 1.0e-10:
            return 0.0
        else:
            temperature_au = self._K_BOLTZMANN * temperature / self._RYDBERG_TO_JOULE
            return temperature_au / omega

    def _load_phonon_results(self, file_in):
        tol_zero = 1.0e-3

        f = open(file_in, "r")

        # skip 10 lines
        for _ in range(10):
            f.readline()

        nmode = int(f.readline().split(":")[1])
        nq = int(f.readline().split(":")[1])
        _nkd = int(f.readline().split(":")[1])
        mass = [float(t) for t in f.readline().split(":")[1].split()]
        # skip 3 lines
        for _ in range(3):
            f.readline()

        omega2 = np.zeros((nq, nmode))

        evec = np.zeros((nq, nmode, nmode), dtype=np.complex128)
        xq = np.zeros((nq, 3))

        for iq in range(nq):
            xq_tmp = [float(a) for a in (f.readline().split(":")[1]).split()]
            xq[iq, :] = xq_tmp[:]
            for imode in range(nmode):
                omega2[iq, imode] = float(f.readline().split(":")[1])
                for jmode in range(nmode):
                    line = f.readline().split()
                    evec[iq, imode, jmode] = complex(float(line[0]), float(line[1]))
                f.readline()
            f.readline()

        qlist_real = []
        qlist_uniq = []

        # Prepare q point list
        for iq in range(nq):
            xq_tmp = xq[iq, :]
            xq_minus = -xq_tmp
            xdiff = (xq_tmp - xq_minus) % 1.0
            norm = math.sqrt(np.dot(xdiff, xdiff))
            if norm < tol_zero:
                qlist_real.append(iq)

            flag_uniq = True

            for jq in qlist_uniq:
                xq_tmp2 = xq[jq, :]
                xdiff = (xq_tmp2 - xq_tmp) % 1.0
                xdiff2 = (xq_tmp2 + xq_tmp) % 1.0
                norm = math.sqrt(np.dot(xdiff, xdiff))
                norm2 = math.sqrt(np.dot(xdiff2, xdiff2))

                if norm < tol_zero or norm2 < tol_zero:
                    flag_uniq = False
                    break
            if flag_uniq:
                qlist_uniq.append(iq)

        qlist_uniq = list(set(qlist_uniq) - set(qlist_real))

        f.close()

        self._qpoints = xq
        self._omega2 = omega2
        self._evec = evec
        self._qlist_real = qlist_real
        self._qlist_uniq = qlist_uniq
        self._mass = mass
        self._nmode = nmode

    @staticmethod
    def _char_xyz(entry):
        if entry % 3 == 0:
            return "x"
        if entry % 3 == 1:
            return "y"
        if entry % 3 == 2:
            return "z"

    def _trunc_normal(self, size, sigma, k):
        """
        Generate N(0, σ^2) with either clipping or rejection so that exactly 'size'
        samples are returned. Mode controlled by self._truncate_mode.
        """
        if k is None or k <= 0:
            return np.random.normal(0.0, sigma, size=size)

        if self._truncate_mode == "clip":
            x = np.random.normal(0.0, sigma, size=size)
            lim = k * sigma
            np.clip(x, -lim, lim, out=x)
            return x

        def _Phi(z):
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        def _phi(z):
            return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)

        if self._truncate_mode == "reject":
            lim = k * sigma
            out = np.empty(size, dtype=float)
            n = 0
            while n < size:
                m = size - n
                x = np.random.normal(0.0, sigma, size=m)
                keep = np.abs(x) <= lim
                nk = np.count_nonzero(keep)
                if nk > 0:
                    out[n : n + nk] = x[keep][:nk]
                    n += nk
            return out

        if self._truncate_mode == "reject_varmatch":
            # Adjust pre-draw scale so truncated variance ≈ σ^2
            denom = (2.0 * _Phi(k) - 1.0)
            var_std = 1.0 - (2.0 * k * _phi(k)) / denom
            scale = sigma / math.sqrt(max(var_std, 1e-14))
            lim = k * scale
            out = np.empty(size, dtype=float)
            n = 0
            while n < size:
                m = size - n
                x = np.random.normal(0.0, scale, size=m)
                keep = np.abs(x) <= lim
                nk = np.count_nonzero(keep)
                if nk > 0:
                    out[n : n + nk] = x[keep][:nk]
                    n += nk
            return out

        # Fallback
        return np.random.normal(0.0, sigma, size=size)

    def _gamma_indices(self, tol=1.0e-10):
        """
        Return indices iq where q ≡ (0,0,0) modulo 1 (wrapped to [-0.5,0.5)).
        """
        q = self._qpoints
        qred = q % 1.0
        qred = np.where(qred >= 0.5, qred - 1.0, qred)
        is_gamma = np.linalg.norm(qred, axis=1) < tol
        return np.where(is_gamma)[0]

    def _translation_bases_prim(self):
        """
        Build 3 mass-weighted rigid translations (Tx, Ty, Tz) on the primitive basis,
        shaped as (3*n_prim,), and L2-normalized.
        """
        nprim = self._nat_primitive
        T = np.zeros((3, 3 * nprim), dtype=float)
        for jat in range(nprim):
            mj = self._mass[self._primitive_kd[jat]]  # amu
            w = 1.0 / math.sqrt(mj)
            base = 3 * jat
            T[0, base + 0] = w
            T[1, base + 1] = w
            T[2, base + 2] = w
        for a in range(3):
            na = np.linalg.norm(T[a])
            if na > 0:
                T[a] /= na
        return T

    def _identify_gamma_acoustic(self, overlap_thresh=0.95, max_modes=3, verbose=True):
        """
        Identify Γ-acoustic modes by checking if all atoms have the same displacement vector.
        For acoustic modes, the eigenvector should represent uniform translation:
        all displacement vectors should be identical (within tolerance).
        
        Returns dict { iq_gamma: mask }, mask has shape (nmode,) bool.
        """
        gamma_ids = self._gamma_indices()
        out = {}
        if len(gamma_ids) == 0:
            return out
        
        for iq in gamma_ids:
            nmode = self._nmode
            nprim = self._nat_primitive
            mask = np.zeros(nmode, dtype=bool)
            uniformity_scores = np.zeros(nmode)
            
            for imode in range(nmode):
                # Extract eigenvector for this mode (3*nprim components)
                evec = self._evec[iq, imode, :].copy()
                
                # Reshape to (nprim, 3) - displacement vectors for each atom
                disp_vectors = evec.reshape(nprim, 3)
                
                # For acoustic modes, need to account for mass-weighting
                # Divide by sqrt(mass) to get actual displacement directions
                disp_physical = np.zeros_like(disp_vectors, dtype=complex)
                for jat in range(nprim):
                    mj = self._mass[self._primitive_kd[jat]]
                    if mj > 0:
                        disp_physical[jat] = disp_vectors[jat] / math.sqrt(mj)
                
                # Use real part for Gamma point modes
                disp_real = disp_physical.real
                
                # Skip if all displacements are essentially zero
                max_disp = np.max(np.abs(disp_real))
                if max_disp < 1e-10:
                    uniformity_scores[imode] = 0.0
                    continue
                
                # For acoustic mode, all displacement vectors should be the same
                # Compare each atom's displacement with the first atom
                ref_disp = disp_real[0]
                
                # Compute maximum relative difference from reference
                max_rel_diff = 0.0
                for jat in range(1, nprim):
                    diff = np.linalg.norm(disp_real[jat] - ref_disp)
                    ref_norm = np.linalg.norm(ref_disp)
                    if ref_norm > 1e-10:
                        rel_diff = diff / ref_norm
                        max_rel_diff = max(max_rel_diff, rel_diff)
                    else:
                        # If reference is near zero, check absolute difference
                        if np.linalg.norm(disp_real[jat]) > 1e-10:
                            max_rel_diff = 1.0  # Different from zero reference
                
                # Uniformity score: 1 - max_relative_difference
                # Perfect acoustic mode: max_rel_diff = 0 → score = 1.0
                # Non-uniform mode: max_rel_diff → ∞ → score → 0
                uniformity_scores[imode] = 1.0 / (1.0 + max_rel_diff)
            
            # Mark modes with high uniformity score as acoustic
            mask = uniformity_scores >= overlap_thresh
            
            # If no modes pass the threshold, take the top 3 with highest scores
            if mask.sum() == 0:
                order = np.argsort(-uniformity_scores)
                chosen = order[:max_modes]
                if uniformity_scores[chosen[0]] > 0.80:
                    mask[chosen] = True
            
            out[iq] = mask
            
            if verbose and self._verbosity > 0:
                k = int(mask.sum())
                top3 = (np.sort(uniformity_scores)[-3:][::-1] if nmode >= 3
                        else np.sort(uniformity_scores)[::-1])
                print(f" [Gamma acoustic] iq={iq+1}: "
                      f"tagged {k} modes. Top uniformity scores: {top3}")
                # Print identified acoustic modes and their frequencies
                acoustic_modes = np.where(mask)[0]
                if len(acoustic_modes) > 0:
                    print("  Identified acoustic modes:")
                    for imode in acoustic_modes:
                        omega2 = self._omega2[iq, imode]
                        if omega2 >= 0:
                            omega_kayser = (math.sqrt(omega2) /
                                            self._KAYSER_TO_RYD)
                        else:
                            omega_kayser = -(math.sqrt(-omega2) /
                                             self._KAYSER_TO_RYD)
                        print(f"    Mode {imode+1}: "
                              f"frequency = {omega_kayser:10.4f} cm^-1, "
                              f"uniformity = {uniformity_scores[imode]:.4f}")
        return out

    def _pairwise_minimum_image(self, x_cart):
        """
        Compute nearest-neighbor distance per atom (Å) using minimum image
        within the current supercell cell.
        """
        inv = self._supercell.inverse_lattice_vector
        lat = self._supercell.lattice_vector
        xf = x_cart @ inv.T  # fractional
        nat = x_cart.shape[0]
        dmin = np.full(nat, np.inf)
        for i in range(nat):
            df = xf - xf[i]
            df -= np.round(df)  # wrap to [-0.5, 0.5)
            dr = df @ lat.T
            dist = np.linalg.norm(dr, axis=1)
            dist[i] = np.inf
            dmin[i] = dist.min()
        return dmin

    def _nearest_neighbor_distances(self):
        """
        Return per-atom nearest-neighbor distances (Å) in the current supercell.
        """
        x_cart = self._supercell.x_fractional @ self._supercell.lattice_vector.T
        return self._pairwise_minimum_image(x_cart)

    def _per_mode_realspace_gain(self):
        """
        C[iq, imode, iat] in Å per unit Q (bohr*amu_ry^(1/2)).
        Mirrors the construction used in _get_random_displacements_normalcoordinate.
        """
        nq = len(self._qpoints)
        nmode = self._nmode
        nat = self._supercell.nat

        # per-atom factor and mapping
        factor = np.zeros(nat)
        for iat in range(nat):
            factor[iat] = 1.0 / math.sqrt(
                self._mass[self._supercell.atomic_kinds[iat]]
                * self._AMU_RYD
                * float(nq)
            )

        is_real = np.zeros(nq, dtype=bool)
        is_real[self._qlist_real] = True

        C = np.zeros((nq, nmode, nat))
        for iq in range(nq):
            for imode in range(nmode):
                if is_real[iq]:
                    for iat in range(nat):
                        jat = self._mapping_s2p[iat]
                        e = self._evec[iq, imode, 3*jat: 3*jat + 3].real
                        g_pre = np.linalg.norm(e)
                        C[iq, imode, iat] = (
                            factor[iat] * g_pre * self._BOHR_TO_ANGSTROM
                        )
                else:
                    for iat in range(nat):
                        jat = self._mapping_s2p[iat]
                        e = self._evec[iq, imode, 3*jat: 3*jat + 3]
                        g_pre = math.sqrt(2.0) * np.linalg.norm(e)
                        C[iq, imode, iat] = (
                            factor[iat] * g_pre * self._BOHR_TO_ANGSTROM
                        )
        return C
