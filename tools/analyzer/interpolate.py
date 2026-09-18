import numpy as np

# Clamp applied to the linewidths before taking the log, identical to anphon's
# (eps = DBL_EPSILON in Ry, converted to the cm^-1 unit of the stored data), so
# that the interpolated values reproduce PREFIX.interpolated_gamma.
_RY_TO_KAYSER = (
    1.0e-2
    / (2.0 * np.pi * 299792458)
    / (6.62606896e-34 / (2.0 * np.pi * 4.35974394e-18 / 2.0))
)
EPS_LOG_CLAMP = np.finfo(float).eps * _RY_TO_KAYSER

# log-space value assigned to the acoustic branches at Gamma by modified-log-linear:
# anphon uses -100 on the Ry-valued linewidths, shifted here to the cm^-1 data
_LOG_LIMIT_GAMMA = -100.0 + np.log(_RY_TO_KAYSER)


class Interpolator:
    def __init__(self, qgrid_coarse, q_coords_irred, weight_q=None, rotations=None):
        self.kgrid = qgrid_coarse
        self.xqc_irred = q_coords_irred
        self.xqc_full = self.make_fullgrid()
        self.weight_xqc = weight_q
        self.bz2irb = None

        if rotations is not None:
            self.unfold(rotations)

    def make_fullgrid(self):
        totk = self.kgrid[0] * self.kgrid[1] * self.kgrid[2]
        xk = np.zeros((totk, 3))
        for i in range(self.kgrid[0]):
            for j in range(self.kgrid[1]):
                for k in range(self.kgrid[2]):
                    ik = k + j * self.kgrid[2] + i * self.kgrid[1] * self.kgrid[2]
                    xk[ik, 0] = i / self.kgrid[0]
                    xk[ik, 1] = j / self.kgrid[1]
                    xk[ik, 2] = k / self.kgrid[2]
        return xk

    def get_knum(self, xk):
        i = (xk[0] * self.kgrid[0] + 2 * self.kgrid[0]) % self.kgrid[0]
        j = (xk[1] * self.kgrid[1] + 2 * self.kgrid[1]) % self.kgrid[1]
        k = (xk[2] * self.kgrid[2] + 2 * self.kgrid[2]) % self.kgrid[2]
        return round(k + j * self.kgrid[2] + i * self.kgrid[1] * self.kgrid[2])

    def unfold(self, rotations):
        # find the map between fullk and k
        nsymm = len(rotations)
        qsym = np.zeros(rotations.shape)
        for i in range(nsymm):
            qsym[i, :, :] = np.linalg.inv(rotations[i, :, :]).T

        count = 0
        found = np.zeros(len(self.xqc_full), dtype=int)
        self.bz2irb = np.zeros(len(self.xqc_full), dtype=int)
        for i, k in enumerate(self.xqc_irred):
            cl = 0
            for rot in qsym:
                k2 = rot.dot(k)
                k2 = k2 - np.round(k2)
                k2_index = self.get_knum(k2)
                if found[k2_index] == 0:
                    found[k2_index] = 1
                    self.bz2irb[k2_index] = i
                    count += 1
                    cl += 1
            if self.weight_xqc is not None:
                if cl < int(self.weight_xqc[i]):
                    # Time-reversal-reduced stars also contain the images -R*k.
                    for rot in qsym:
                        k2 = -rot.dot(k)
                        k2 = k2 - np.round(k2)
                        k2_index = self.get_knum(k2)
                        if found[k2_index] == 0:
                            found[k2_index] = 1
                            self.bz2irb[k2_index] = i
                            count += 1
                            cl += 1
                if cl != int(self.weight_xqc[i]):
                    raise RuntimeError(
                        "Symmetry unfolding failed for irreducible q-point {}: "
                        "the number of star members ({}) does not match the "
                        "expected weight ({}).".format(i, cl, int(self.weight_xqc[i]))
                    )

    def run(
        self,
        data_coarse,
        xk,
        interpolation_method="log-linear",
        rotations=None,
        eps=EPS_LOG_CLAMP,
    ):
        if self.bz2irb is None:
            if rotations is None:
                raise RuntimeError("need to give rotations to run interpolation")
            else:
                self.unfold(rotations)

        # according to the slides
        #
        corners = self.find_corner_fractional(xk)
        # corners(8,3) in fractional
        x000 = corners[0]
        x100 = corners[1]
        x110 = corners[2]
        x010 = corners[3]
        x001 = corners[4]
        x101 = corners[5]
        x111 = corners[6]
        x011 = corners[7]
        delta = (xk - x000) / (x111 - x000)

        _, nmodes, ntemps = np.shape(data_coarse)
        out = np.zeros((nmodes, ntemps))

        if interpolation_method == "linear":
            for imode in range(nmodes):
                for itemp in range(ntemps):
                    v000 = data_coarse[self.bz2irb[self.get_knum(x000)], imode, itemp]
                    v100 = data_coarse[self.bz2irb[self.get_knum(x100)], imode, itemp]
                    v110 = data_coarse[self.bz2irb[self.get_knum(x110)], imode, itemp]
                    v010 = data_coarse[self.bz2irb[self.get_knum(x010)], imode, itemp]
                    v001 = data_coarse[self.bz2irb[self.get_knum(x001)], imode, itemp]
                    v101 = data_coarse[self.bz2irb[self.get_knum(x101)], imode, itemp]
                    v111 = data_coarse[self.bz2irb[self.get_knum(x111)], imode, itemp]
                    v011 = data_coarse[self.bz2irb[self.get_knum(x011)], imode, itemp]

                    c0 = v000
                    c1 = v001 - v000
                    c2 = v100 - v000
                    c3 = v010 - v000
                    c4 = v101 - v100 - v001 + v000
                    c5 = v110 - v010 - v100 + v000
                    c6 = v011 - v010 - v001 + v000
                    c7 = v111 - v110 - v011 - v101 + v001 + v010 + v100 - v000

                    v = (
                        c0
                        + c1 * delta[2]
                        + c2 * delta[0]
                        + c3 * delta[1]
                        + c4 * delta[0] * delta[2]
                        + c5 * delta[0] * delta[1]
                        + c6 * delta[1] * delta[2]
                        + c7 * delta[0] * delta[1] * delta[2]
                    )

                    out[imode, itemp] = v

        elif interpolation_method == "log-linear":
            for imode in range(nmodes):
                for itemp in range(ntemps):
                    v000 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x000)], imode, itemp],
                            eps,
                        )
                    )
                    v100 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x100)], imode, itemp],
                            eps,
                        )
                    )
                    v110 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x110)], imode, itemp],
                            eps,
                        )
                    )
                    v010 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x010)], imode, itemp],
                            eps,
                        )
                    )
                    v001 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x001)], imode, itemp],
                            eps,
                        )
                    )
                    v101 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x101)], imode, itemp],
                            eps,
                        )
                    )
                    v111 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x111)], imode, itemp],
                            eps,
                        )
                    )
                    v011 = np.log(
                        max(
                            data_coarse[self.bz2irb[self.get_knum(x011)], imode, itemp],
                            eps,
                        )
                    )

                    c0 = v000
                    c1 = v001 - v000
                    c2 = v100 - v000
                    c3 = v010 - v000
                    c4 = v101 - v100 - v001 + v000
                    c5 = v110 - v010 - v100 + v000
                    c6 = v011 - v010 - v001 + v000
                    c7 = v111 - v110 - v011 - v101 + v001 + v010 + v100 - v000

                    v = (
                        c0
                        + c1 * delta[2]
                        + c2 * delta[0]
                        + c3 * delta[1]
                        + c4 * delta[0] * delta[2]
                        + c5 * delta[0] * delta[1]
                        + c6 * delta[1] * delta[2]
                        + c7 * delta[0] * delta[1] * delta[2]
                    )

                    out[imode, itemp] = np.exp(v)

        return out

    def run2(
        self,
        data_coarse,
        xk,
        interpolation_method="log-linear",
        rotations=None,
        eps=EPS_LOG_CLAMP,
    ):
        """Interpolate data_coarse[nk_irred, nmodes, ntemps] to the point xk.

        interpolation_method follows the INTERPOLATOR tag of anphon:
        linear, log-linear (trilinear in log space) and modified-log-linear
        (same, but the acoustic branches of a cell that has Gamma as a corner
        are extrapolated from the neighboring cells instead, see
        TriLinearInterpolator::interpolate_avoidgamma in anphon).
        """
        if self.bz2irb is None:
            if rotations is None:
                raise RuntimeError("need to give rotations to run interpolation")
            else:
                self.unfold(rotations)

        if interpolation_method not in ("linear", "log-linear", "modified-log-linear"):
            raise ValueError(
                "unknown interpolation_method: {}".format(interpolation_method)
            )
        use_log = interpolation_method != "linear"

        data = np.log(np.maximum(data_coarse, eps)) if use_log else data_coarse
        corners = self.find_corner_fractional(xk)
        v = self._trilinear(data, corners, xk)

        if interpolation_method == "modified-log-linear":
            nac = min(3, v.shape[0])
            if np.all(np.abs(xk - np.round(xk)) < 1.0e-8):
                v[:nac] = _LOG_LIMIT_GAMMA
            elif self._contains_gamma(corners):
                v[:nac] = self._avoid_gamma(data[:, :nac], corners, xk)

        return np.exp(v) if use_log else v

    def _trilinear(self, data, corners, xk):
        # Trilinear polynomial of the cell spanned by corners, evaluated at xk
        # (xk may lie outside the cell: that is the extrapolation used by
        # modified-log-linear).
        delta = (xk - corners[0]) / (corners[6] - corners[0])
        w = np.array(
            [
                1.0,
                delta[2],
                delta[0],
                delta[1],
                delta[0] * delta[2],
                delta[0] * delta[1],
                delta[1] * delta[2],
                delta[0] * delta[1] * delta[2],
            ]
        ).reshape((8,) + (1,) * (data.ndim - 1))
        cv = data[self.bz2irb[[self.get_knum(c) for c in corners]]]
        c = np.empty_like(cv)
        c[0] = cv[0]
        c[1] = cv[4] - cv[0]
        c[2] = cv[1] - cv[0]
        c[3] = cv[3] - cv[0]
        c[4] = cv[5] - cv[4] - cv[1] + cv[0]
        c[5] = cv[2] - cv[3] - cv[1] + cv[0]
        c[6] = cv[7] - cv[3] - cv[4] + cv[0]
        c[7] = cv[6] - cv[2] - cv[7] - cv[5] + cv[1] + cv[3] + cv[4] - cv[0]
        return np.sum(c * w, axis=0)

    def _contains_gamma(self, corners):
        return any(self.get_knum(c) == 0 for c in corners)

    def _avoid_gamma(self, data, corners, xk):
        # Average of the trilinear extrapolations from the mirrored neighbor
        # cells that do not contain Gamma, mirrored about the corner closest to xk.
        others = [c for c in corners if self.get_knum(c) != 0]
        closest = min(others, key=lambda c: np.sum((xk - c) ** 2))
        total = None
        count = 0
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    center = closest + np.array([sx, sy, sz]) * (xk - closest)
                    corners_n = self.find_corner_fractional(center)
                    if self._contains_gamma(corners_n):
                        continue
                    val = self._trilinear(data, corners_n, xk)
                    total = val if total is None else total + val
                    count += 1
        if count == 0:
            # every neighbor cell touches Gamma (coarse mesh of 2 or less along
            # some direction): nothing to extrapolate from, keep the plain value
            return self._trilinear(data, corners, xk)
        return total / count

    def find_corner_fractional(self, xk):
        #
        # return fractional coordinate of the cell that contain xkf
        #
        nk1 = self.kgrid[0]
        nk2 = self.kgrid[1]
        nk3 = self.kgrid[2]

        i = np.floor(xk[0] * self.kgrid[0])
        j = np.floor(xk[1] * self.kgrid[1])
        k = np.floor(xk[2] * self.kgrid[2])
        # (i,j,k) with the below x will contain xkf

        ii = i + 1
        jj = j + 1
        kk = k + 1

        x1 = np.array([i / nk1, j / nk2, k / nk3])
        x2 = np.array([ii / nk1, j / nk2, k / nk3])
        x3 = np.array([ii / nk1, jj / nk2, k / nk3])
        x4 = np.array([i / nk1, jj / nk2, k / nk3])
        x5 = np.array([i / nk1, j / nk2, kk / nk3])
        x6 = np.array([ii / nk1, j / nk2, kk / nk3])
        x7 = np.array([ii / nk1, jj / nk2, kk / nk3])
        x8 = np.array([i / nk1, jj / nk2, kk / nk3])

        return np.vstack((x1, x2, x3, x4, x5, x6, x7, x8))
