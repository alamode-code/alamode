import numpy as np


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
                if cl != int(self.weight_xqc[i]):
                    print("unfolding goes wrong")

    def run(
        self,
        data_coarse,
        xk,
        interpolation_method="log-linear",
        rotations=None,
        eps=1.0e-12,
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
        eps=1.0e-12,
    ):
        if self.bz2irb is None:
            if rotations is None:
                raise RuntimeError("need to give rotations to run interpolation")
            else:
                self.unfold(rotations)

        corners = self.find_corner_fractional(xk)
        x000 = corners[0]
        x111 = corners[6]
        delta = (xk - x000) / (x111 - x000)
        delta_reshaped = np.array(
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
        ).reshape(8, 1, 1)

        corner_indices = np.array([self.get_knum(corner) for corner in corners])
        corner_values = data_coarse[self.bz2irb[corner_indices]]
        c = np.zeros(corner_values.shape)

        if interpolation_method == "log-linear":
            corner_values = np.log(np.maximum(corner_values, eps))

        c[0] = corner_values[0]
        c[1] = corner_values[4] - corner_values[0]
        c[2] = corner_values[1] - corner_values[0]
        c[3] = corner_values[3] - corner_values[0]
        c[4] = corner_values[5] - corner_values[4] - corner_values[1] + corner_values[0]
        c[5] = corner_values[2] - corner_values[3] - corner_values[1] + corner_values[0]
        c[6] = corner_values[7] - corner_values[3] - corner_values[4] + corner_values[0]
        c[7] = (
            corner_values[6]
            - corner_values[2]
            - corner_values[7]
            - corner_values[5]
            + corner_values[1]
            + corner_values[3]
            + corner_values[4]
            - corner_values[0]
        )

        v = np.sum(c * delta_reshaped, axis=0)

        if interpolation_method == "log-linear":
            v = np.exp(v)

        return v

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
