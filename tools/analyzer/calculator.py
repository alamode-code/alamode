import numpy as np
import spglib

from analyzer.anphonio import ParseResult
from analyzer.interpolate import Interpolator


class Calculator:
    def __init__(
        self,
        file_result_3ph,
        file_result_4ph=None,
        file_isotope=None,
        average_gamma=True,
        tolerance=1.0e-3,
    ):
        """
        The constructor of the Calculator class.

        Parameters:
            file_result_3ph (str): File path to the 3-phonon results.
            file_result_4ph (str): File path to the 4-phonon results.
            file_isotope (str): File path to the isotope self-energy.
            average_gamma (bool): If True, averages the gamma values at degenerate phonon modes (default: True).
            tolerance (float): Tolerance for symmetry detection (default: 1.0e-3).
        """
        self.file_result_3ph = file_result_3ph
        self.file_result_4ph = file_result_4ph
        self.file_isotope = file_isotope
        self.omega = None  # Frequency array
        self.omega4 = None
        self.gamma3 = None  # linewidth due to 3-phonon scattering
        self.gamma4 = None  # linewidth due to 4-phonon scattering
        self.gamma4_interpolated = None
        self.gamma_iso = None  # linediwth due to isotope scattering
        self.vel = None  # Velocity array
        self.vel4 = None
        self.qpoint_weight = None  # Weight array
        self.qpoint_weight4 = None
        self.qpoints = None
        self.qpoints4 = None
        self.qgrid = None
        self.qgrid4 = None
        self.volume = None
        self.volume4 = None
        self.temperatures = None
        self.temperatures4 = None
        self.average_gamma = average_gamma
        self.rotations = None
        self.tolerance = tolerance
        self.classical = None

        self._BOHR = 0.52917721092
        self._k_Boltzmann = 1.3806488e-23
        self._Hz_to_kayser = 1.0e-2 / (2.0 * np.pi * 299792458)
        self._Ryd = 4.35974394e-18 / 2.0
        self._time_ry = 6.62606896e-34 / (2.0 * np.pi * self._Ryd)
        self._factor_gamma_to_tau = 1.0e12 * self._Hz_to_kayser * 0.5
        self._kayser_to_Ryd = self._time_ry / self._Hz_to_kayser
        self._T_to_Ryd = self._k_Boltzmann / self._Ryd

        if self.file_result_3ph is not None:
            self.set_variables_3ph()

        if self.file_result_4ph is not None:
            self.set_variables_4ph()
            self.interpol_gamma4()

        if self.file_isotope is not None:
            self.set_variables_iso()

    def set_variables_3ph(self):
        """
        Sets the variables related to 3-phonon scattering amplitudes.
        """
        result = ParseResult(self.file_result_3ph)
        self.omega = result.omega
        if self.average_gamma:
            self.gamma3 = self.average_gamma_at_degenerate_point(
                self.omega, result.gamma
            )
        else:
            self.gamma3 = result.gamma
        self.vel = result.vel
        self.volume = result.volume
        self.qpoint_weight = result.multiplicity
        self.qpoints = result.q_coord
        self.qgrid = result.kgrid
        self.temperatures = result.temperatures
        self.classical = result.classical

    def set_variables_4ph(self):
        """
        Sets the variables related to 4-phonon scattering.
        """
        result = ParseResult(self.file_result_4ph)
        self.omega4 = result.omega
        if self.average_gamma:
            self.gamma4 = self.average_gamma_at_degenerate_point(
                self.omega4, result.gamma
            )
        else:
            self.gamma4 = result.gamma
        self.vel4 = result.vel
        self.volume4 = result.volume
        self.qpoint_weight4 = result.multiplicity
        self.qpoints4 = result.q_coord
        self.qgrid4 = result.kgrid
        self.temperatures4 = result.temperatures

        if result.lattice_vector is not None:
            cell = (result.lattice_vector.T, result.x_fractional, result.atomic_kinds)
            self.rotations = spglib.get_symmetry_dataset(cell, symprec=self.tolerance)[
                "rotations"
            ]

    def set_variables_iso(self):
        """
        Sets the variables related to isotope scattering.
        """
        result = np.loadtxt(self.file_isotope)
        nk = round(np.max(result[:, 0]))
        ns = round(np.max(result[:, 1]))
        omega = result[:, 2].reshape((nk, ns))
        gamma = result[:, 3].reshape((nk, ns, 1))
        if self.average_gamma:
            self.gamma_iso = self.average_gamma_at_degenerate_point(omega, gamma)[
                :, :, 0
            ]
        else:
            self.gamma_iso = gamma[:, :, 0]

    def interpol_gamma4(self):
        """
        Interpolates the 4-phonon scattering gamma values.
        """
        interpol = Interpolator(
            self.qgrid4,
            self.qpoints4,
            weight_q=self.qpoint_weight4,
            rotations=self.rotations,
        )
        self.gamma4_interpolated = np.zeros(self.gamma3.shape, dtype=float)
        for i, xq in enumerate(self.qpoints):
            self.gamma4_interpolated[i, :, :] = interpol.run2(self.gamma4, xq)

    def check_data_load(self, four_phonon, isotope):
        """
        Checks if the necessary data is loaded for the given calculation.

        Parameters:
            four_phonon (bool): If True, checks for 4-phonon scattering data.
            isotope (bool): If True, checks for isotope scattering data.
        """
        if self.gamma3 is None:
            self.set_variables_3ph()

        if four_phonon and (self.gamma4 is None):
            if self.file_result_4ph is None:
                raise RuntimeError(
                    "file_result_4ph must be given when initializing the class"
                )
            else:
                self.set_variables_4ph()

        if isotope and (self.gamma_iso is None):
            if self.file_isotope is None:
                raise RuntimeError(
                    "file_isotope must be given when initializing the class"
                )
            else:
                self.set_variables_iso()

    def average_gamma_at_degenerate_point(self, frequencies, gamma, tol_omega=1e-3):
        """
        Averages the gamma values at degenerate points across the phonon band structure.

        Parameters:
            frequencies (numpy.ndarray): Array of frequencies.
            gamma (numpy.ndarray): Array of gamma values.
            tol_omega (float): Tolerance for considering frequencies as degenerate.
        """
        nk, ns = frequencies.shape
        nt = gamma.shape[-1]

        gamma_avg = np.zeros((nk, ns, nt), dtype=float)

        # Loop over all k-points
        for i in range(nk):
            degeneracy_at_k = []
            omega_prev = frequencies[i, 0]
            ideg = 1

            # Identify degenerate modes
            for j in range(1, ns):
                omega_now = frequencies[i, j]
                if abs(omega_now - omega_prev) < tol_omega:
                    ideg += 1
                else:
                    degeneracy_at_k.append(ideg)
                    ideg = 1
                    omega_prev = omega_now
            degeneracy_at_k.append(ideg)  # Append the last set of degeneracies

            is_index = 0

            # Average the tau values for each set of degenerate modes
            for deg in degeneracy_at_k:
                damp_sum = np.zeros(nt)
                for k in range(is_index, is_index + deg):
                    damp_sum += gamma[i, k, :]

                for k in range(is_index, is_index + deg):
                    gamma_avg[i, k, :] = damp_sum / deg

                is_index += deg

        return gamma_avg

    def get_linewidth(self, temperature, four_phonon=False, isotope=False):
        """
        Gets the linewidth for a given temperature.

        Parameters:
            temperature (float): The temperature to get the linewidth for.
            four_phonon (bool): If True, includes 4-phonon scattering linewidth.
            isotope (bool): If True, includes isotope scattering linewidth.

        Returns:
            numpy.ndarray: The linewidth.
        """
        self.check_data_load(four_phonon, isotope)

        tempdiff = np.abs(self.temperatures - temperature)
        index = np.argsort(tempdiff)

        if tempdiff[index[0]] > 0:
            raise RuntimeWarning(
                "The data at exactly at {} K not found. "
                "Return the values at {} K instead".format(
                    temperature, self.temperatures[index[0]]
                )
            )
        out = self.gamma3[:, :, index[0]]
        if four_phonon:
            out += self.gamma4_interpolated[:, :, index[0]]
        if isotope:
            out += self.gamma_iso

        return out

    def print_linewidth(self, temperature, four_phonon=False, isotope=False):
        """
        Prints the linewidth for a given temperature.

        Parameters:
            temperature (float): The temperature to print the linewidth for.
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
        """
        tempdiff = np.abs(self.temperatures - temperature)
        index = np.argsort(tempdiff)

        if tempdiff[index[0]] > 0:
            raise RuntimeWarning(
                "The data at exactly at {} K not found. "
                "Return the values at {} K instead".format(
                    temperature, self.temperatures[index[0]]
                )
            )
        gamma3 = self.gamma3[:, :, index[0]]
        if four_phonon:
            gamma4 = self.gamma4_interpolated[:, :, index[0]]
        else:
            gamma4 = None

        if isotope:
            gamma_iso = self.gamma_iso[:, :]
        else:
            gamma_iso = None

        nk, nmode = gamma3.shape

        print(
            "# k-point index, mode index, frequency (cm^-1), 3-phonon linewidth", end=""
        )
        if four_phonon:
            print(", 4-phonon linewidth", end="")
        if isotope:
            print(", isotope linewidth", end="")
        print(", total linewidth")
        for ik in range(nk):
            for imode in range(nmode):
                total = gamma3[ik, imode]
                if four_phonon:
                    total += gamma4[ik, imode]
                if isotope:
                    total += gamma_iso[ik, imode]
                print(
                    "{:4d} {:4d} {:12.6f} {:12.6f}".format(
                        ik + 1, imode + 1, self.omega[ik, imode], gamma3[ik, imode]
                    ),
                    end="",
                )
                if four_phonon:
                    print("{:12.6f}".format(gamma4[ik, imode]), end="")
                if isotope:
                    print("{:12.6f}".format(gamma_iso[ik, imode]), end="")
                print("{:12.6f}".format(total))

    def get_linewidth_mode(self, index_k, index_mode, four_phonon=False, isotope=False):
        """
        Gets the linewidth for a specific mode at a specific k-point.

        Parameters:
            index_k (int): The index of the k-point to analyze (starts from 0).
            index_mode (int): The index of the mode to analyze (starts from 0).
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.

        Returns:
            tuple: The linewidths for 3-phonon, 4-phonon, and isotope scatterings.
        """
        gamma3 = self.gamma3[index_k, index_mode, :]
        if four_phonon:
            gamma4 = self.gamma4_interpolated[index_k, index_mode, :]
        else:
            gamma4 = None

        if isotope:
            gamma_iso = self.gamma_iso[index_k, index_mode]
        else:
            gamma_iso = None

        return gamma3, gamma4, gamma_iso

    def print_linewidth_mode(
        self, index_k, index_mode, four_phonon=False, isotope=False
    ):
        """
        Prints the linewidth for a specific mode at a specific k-point.

        Parameters:
            index_k (int): The index of the k-point to analyze (starts from 0).
            index_mode (int): The index of the mode to analyze (starts from 0).
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
        """
        gamma3, gamma4, gamma_iso = self.get_linewidth_mode(
            index_k, index_mode, four_phonon, isotope
        )

        print(
            "# Phonon linewidth of mode {:d} at k-point {:d}".format(
                index_mode + 1, index_k + 1
            )
        )
        print("# Phonon frequency: {:12.6f}".format(self.omega[index_k, index_mode]))
        print("# temperature, 3-phonon linewidth", end="")
        if four_phonon:
            print(", 4-phonon linewidth", end="")
        if isotope:
            print(", isotope linewidth", end="")
        print(", total linewidth")

        for itemp in range(len(self.temperatures)):
            total = gamma3[itemp]
            if four_phonon:
                total += gamma4[itemp]
            if isotope:
                total += gamma_iso
            print(
                "{:12.6f} {:12.6f}".format(self.temperatures[itemp], gamma3[itemp]),
                end="",
            )
            if four_phonon:
                print("{:12.6f}".format(gamma4[itemp]), end="")
            if isotope:
                print("{:12.6f}".format(gamma_iso), end="")
            print("{:12.6f}".format(total))

    def get_lifetime(self, temperature, four_phonon=False, isotope=False):
        """
        Gets the phonon lifetime for a given temperature.

        Parameters:
            temperature (float): The temperature to get the phonon lifetime for.
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.

        Returns:
            numpy.ndarray: phonon lifetime.
        """
        gamma = self.get_linewidth(temperature, four_phonon, isotope)

        with np.errstate(divide="ignore"):
            tau = self._factor_gamma_to_tau / gamma
        for i in range(self.omega.shape[0]):
            for j in range(self.omega.shape[1]):
                if self.omega[i, j] <= 1.0e-8 or gamma[i, j] <= 1.0e-8:
                    tau[i, j] = 0.0
        return tau

    def print_lifetime(self, temperature, four_phonon=False, isotope=False):
        """
        Prints the phonon lifetime for a given temperature.

        Parameters:
            temperature (float): The temperature to print the phonon lifetime for.
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
        """
        tau_3ph = self.get_lifetime(temperature)
        if four_phonon and isotope:
            tau_total = self.get_lifetime(temperature, four_phonon=True, isotope=True)
        elif four_phonon:
            tau_total = self.get_lifetime(temperature, four_phonon=True)
        elif isotope:
            tau_total = self.get_lifetime(temperature, isotope=True)

        print("# Phonon lifetime (ps) at {:6.2f} K".format(temperature))
        print(
            "# k-point index, mode index, frequency (cm^-1), 3-phonon lifetime (ps)",
            end="",
        )
        if four_phonon and isotope:
            print(", lifetime with isotope and 4ph (ps)")
        elif four_phonon:
            print(", lifetime with 4ph (ps)")
        elif isotope:
            print(", lifetime with isotope (ps)")

        for ik in range(self.gamma3.shape[0]):
            for imode in range(self.gamma3.shape[1]):
                print(
                    "{:4d} {:4d} {:12.6f} {:12.6f}".format(
                        ik + 1, imode + 1, self.omega[ik, imode], tau_3ph[ik, imode]
                    ),
                    end="",
                )
                if four_phonon or isotope:
                    print("{:12.6f}".format(tau_total[ik, imode]), end="")
                print()

    def print_lifetime_mode(
        self, index_k, index_mode, four_phonon=False, isotope=False
    ):
        """
        Prints the phonon lifetime for a specific mode at a specific k-point.

        Parameters:
            index_k (int): The index of the k-point to analyze (starts from 0).
            index_mode (int): The index of the mode to analyze (starts from 0).
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
        """

        gamma3, gamma4, gamma_iso = self.get_linewidth_mode(
            index_k, index_mode, four_phonon, isotope
        )

        print(
            "# Phonon lifetime (ps) of mode {:d} at k-point {:d}".format(
                index_mode + 1, index_k + 1
            )
        )
        print("# Phonon frequency: {:12.6f}".format(self.omega[index_k, index_mode]))
        print("# temperature, 3-phonon lifetime (ps)", end="")
        if four_phonon and isotope:
            print(", lifetime with isotope and 4ph (ps)")
        elif four_phonon:
            print(", lifetime with 4ph (ps)")
        elif isotope:
            print(", lifetime with isotope (ps)")

        for itemp in range(len(self.temperatures)):
            total = gamma3[itemp]
            if four_phonon:
                total += gamma4[itemp]
            if isotope:
                total += gamma_iso
            if gamma3[itemp] <= 1.0e-12:
                print("{:12.6f} {:12.6f}".format(self.temperatures[itemp], 0.0), end="")
            else:
                print(
                    "{:12.6f} {:12.6f}".format(
                        self.temperatures[itemp],
                        self._factor_gamma_to_tau / gamma3[itemp],
                    ),
                    end="",
                )

            if four_phonon or isotope:
                if total <= 1.0e-12:
                    print("{:12.6f}".format(0.0), end="")
                else:
                    print("{:12.6f}".format(self._factor_gamma_to_tau / total), end="")
            print("")

    def get_thermal_conductivity(
        self, four_phonon=False, isotope=False, len_boundary=None, gb_shape="sphere"
    ):
        """
        Gets the thermal conductivity.

        Parameters:
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
            len_boundary (float): The length of the boundary in nanometer (nm).
            gb_shape (str): The shape of the grain boundary (either sphere or cube).

        Returns:
            numpy.ndarray: Thermal conductivity tensor.
        """
        nt = len(self.temperatures)
        kappa = np.zeros((nt, 3, 3), dtype=float)

        nk, nmode = self.omega.shape

        vvprod = np.zeros((nk, nmode, 3, 3), dtype=float)
        for i in range(nk):
            for j in range(nmode):
                for k in range(3):
                    for l in range(3):
                        vvprod[i, j, k, l] = np.dot(
                            self.vel[i, j, :, k], self.vel[i, j, :, l]
                        )

        if len_boundary is None:
            for it, temp in enumerate(self.temperatures):
                tau = self.get_lifetime(
                    temperature=temp, four_phonon=four_phonon, isotope=isotope
                )
                cv = self.heat_capacity(self.omega, temp)

                for i in range(3):
                    for j in range(3):
                        product = cv * tau * vvprod[:, :, i, j]
                        kappa[it, i, j] = np.sum(product, axis=(0, 1))

        else:
            if gb_shape == "sphere":
                assert len_boundary > 0.0, "The boundary length must be positive"
                velnorm = np.linalg.norm(self.vel[:, :, 0, :], axis=2)

                for it, temp in enumerate(self.temperatures):
                    tau = self.get_lifetime(
                        temperature=temp, four_phonon=four_phonon, isotope=isotope
                    )
                    cv = self.heat_capacity(self.omega, temp)

                    mfp = velnorm * tau * 0.001

                    for i in range(3):
                        for j in range(3):
                            product = (
                                cv
                                * tau
                                * vvprod[:, :, i, j]
                                * len_boundary
                                / (len_boundary + 2.0 * mfp)
                            )
                            kappa[it, i, j] = np.sum(product, axis=(0, 1))

            elif gb_shape == "cube":
                assert len(len_boundary) == 3, (
                    "The boundary length must be a list of "
                    "3 elements when using 'cube' shape"
                )

                velnorm = np.abs(self.vel)

                mfp = np.zeros_like(velnorm)

                for it, temp in enumerate(self.temperatures):
                    tau = self.get_lifetime(
                        temperature=temp, four_phonon=four_phonon, isotope=isotope
                    )
                    cv = self.heat_capacity(self.omega, temp)

                    for i in range(velnorm.shape[2]):
                        for j in range(3):
                            mfp[:, :, i, j] = tau * velnorm[:, :, i, j] * 0.001

                    for i in range(3):
                        for j in range(3):
                            product = (
                                cv
                                * tau
                                * np.sum(
                                    self.vel[:, :, :, i]
                                    * self.vel[:, :, :, j]
                                    * len_boundary[i]
                                    / (len_boundary[i] + 2.0 * mfp[:, :, :, i]),
                                    axis=(2),
                                )
                            )
                            kappa[it, i, j] = np.sum(product, axis=(0, 1))

        factor_toSI = (
            1.0e18
            / (self._BOHR**3 * self.volume)
            / (self.qgrid[0] * self.qgrid[1] * self.qgrid[2])
        )

        kappa *= factor_toSI

        return kappa

    def print_thermal_conductivity(
        self, four_phonon=False, isotope=False, len_boundary=None
    ):
        """
        Prints the thermal conductivity.

        Parameters:
            four_phonon (bool): If True, includes 4-phonon scattering data.
            isotope (bool): If True, includes isotope scattering data.
            len_boundary (float): The length of the boundary in nanometer (nm).
        """
        kappa = self.get_thermal_conductivity(
            four_phonon, isotope, len_boundary=len_boundary
        )

        print("# Thermal conductivity (W/mK)")
        if four_phonon:
            print("# Including 4-phonon scattering")
        if isotope:
            print("# Including isotope scattering")
        if len_boundary is not None:
            print(
                "# Including boundary scattering with length {:12.4f} nm".format(
                    len_boundary
                )
            )
        print("# temperature, xx, xy, xz, yx, yy, yz, zx, zy, zz")
        for it, temp in enumerate(self.temperatures):
            print("{:12.2f}".format(temp), end="")
            for i in range(3):
                for j in range(3):
                    print("{:15.3f}".format(kappa[it, i, j]), end="")
            print()

    def get_cumulative_kappa(
        self,
        temperature,
        four_phonon=False,
        isotope=False,
        nsamples=200,
        gridtype="log",
    ):
        """
        Gets the cumulative thermal conductivity.

        Parameters:
            temperature (float): The temperature to get the cumulative thermal conductivity for.
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
            nsamples (int): The number of sampling points for the mean-free-path.
            gridtype (str): The type of grid ('linear' or 'log') for sampling the mean-free-path.

        Returns:
            tuple: The cumulative thermal conductivity and the mean-free-path.
        """

        tau = self.get_lifetime(
            temperature=temperature, four_phonon=four_phonon, isotope=isotope
        )
        cv = self.heat_capacity(self.omega, temperature)

        nk, nmode = self.omega.shape
        vvprod = np.zeros((nk, nmode, 3, 3), dtype=float)
        for i in range(nk):
            for j in range(nmode):
                for k in range(3):
                    for l in range(3):
                        vvprod[i, j, k, l] = np.dot(
                            self.vel[i, j, :, k], self.vel[i, j, :, l]
                        )

        velnorm = np.linalg.norm(self.vel[:, :, 0, :], axis=2)

        mfp = velnorm * tau * 0.001

        max_mfp = np.max(mfp)
        min_mfp = np.min(mfp[np.where(mfp > 0.0)])

        if gridtype == "linear":
            length_vec = np.linspace(min_mfp, max_mfp, nsamples)
        elif gridtype == "log" or gridtype == "logarithmic":
            length_vec = np.geomspace(min_mfp, max_mfp, nsamples)

        kappa = np.zeros((nsamples, 3, 3), dtype=float)

        for ilen, len_boundary in enumerate(length_vec):
            tau_mod = np.where(mfp <= len_boundary, tau, 0.0)
            for i in range(3):
                for j in range(3):
                    product = cv * tau_mod * vvprod[:, :, i, j]
                    kappa[ilen, i, j] = np.sum(product)

        factor_toSI = (
            1.0e18
            / (self._BOHR**3 * self.volume)
            / (self.qgrid[0] * self.qgrid[1] * self.qgrid[2])
        )

        kappa *= factor_toSI

        return kappa, length_vec

    def print_cumulative_kappa(
        self,
        temperature,
        four_phonon=False,
        isotope=False,
        nsamples=200,
        gridtype="log",
    ):
        """
        Prints the cumulative thermal conductivity.

        Parameters:
            temperature (float): The temperature to print the cumulative thermal conductivity for.
            four_phonon (bool): If True, includes 4-phonon scattering.
            isotope (bool): If True, includes isotope scattering.
            nsamples (int): The number of sampling points for the mean-free-path.
            gridtype (str): The type of grid ('linear' or 'log') for sampling the mean-free-path.
        """
        kappa, length_vec = self.get_cumulative_kappa(
            temperature, four_phonon, isotope, nsamples=nsamples, gridtype=gridtype
        )

        print("# Cumulative thermal conductivity (W/mK)")
        if four_phonon:
            print("# Including 4-phonon scattering")
        if isotope:
            print("# Including isotope scattering")
        print("# Temperature = {:12.2f} K".format(temperature))
        print("# mfp (nm), xx, yy, zz")
        for ilen, len_boundary in enumerate(length_vec):
            print("{:12.4f}".format(len_boundary), end="")
            for i in range(3):
                print("{:15.3f}".format(kappa[ilen, i, i]), end="")
            print()

    def heat_capacity(self, omegas, temp):
        """
        Calculates the volumetric heat capacity.

        Parameters:
            omegas (numpy.ndarray): The frequencies.
            temp (float): The temperature.

        Returns:
            numpy.ndarray: The heat capacity at constant volume.
        """
        if self.classical:
            return self._k_Boltzmann * np.ones_like(omegas)
        else:
            if np.abs(temp) < 1.0e-12:
                return np.zeros_like(omegas)
            else:
                x = omegas * self._kayser_to_Ryd / (self._T_to_Ryd * temp)
                ret = self._k_Boltzmann * (x / (2.0 * np.sinh(0.5 * x))) ** 2

                for i in range(omegas.shape[0]):
                    for j in range(omegas.shape[1]):
                        if omegas[i, j] < 1.0e-8:
                            ret[i, j] = 0.0
                return ret

    # Additional methods will be defined here.
