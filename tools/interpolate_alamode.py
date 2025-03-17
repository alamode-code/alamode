import numpy as np
import xml.etree.ElementTree as ET
from ase.data import atomic_numbers

Hz_to_kayser = 5.30883746e-12


def write_gamma_to_file(freq, gamma, filename):
    totq, nmode = freq.shape
    with open(filename, "w") as f:
        for iq in range(totq):
            for im in range(nmode):
                f.write("{:>12.5f}{:>20.8e}\n".format(freq[iq, im], gamma[iq, im]))


def read_symmetry(filename):
    rot = []
    with open(filename, "r") as f:
        aline = f.readline()
        aline = f.readline()
        while aline:
            rotation = np.array(aline.split()[:9], dtype=float).reshape((3, 3))
            rot.append(rotation)
            aline = f.readline()
    return np.array(rot)


def get_supercell_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    structure = root.find("Structure")
    lattice = structure.find("LatticeVector")
    cell = np.zeros((3, 3))
    for i, a in enumerate(lattice):
        cell[i, :] = np.array(a.text.split(), dtype=float)
    pos = structure.find("Position")
    positions = []
    symbols = []
    for p in pos:
        elm = atomic_numbers[p.attrib["element"]]
        symbols.append(elm)
        positions.append(np.array(p.text.split(), dtype=float))
    positions = np.array(positions)
    return cell, positions, symbols


def read_alamode(filename):
    # read the cell
    with open(filename, "r") as f:
        aline = f.readline()
        factor = 0
        cell = np.zeros((3, 3))
        while aline:
            if "&cell" in aline:
                factor = float(f.readline())
                for i in range(3):
                    cell[i, :] = np.array(f.readline().split())
                break
            aline = f.readline()

    return cell * factor * 0.529177


# cell, pos, atomic_number = get_supercell_xml("opt_rc12_ndata12.xml")
# primitivecell = read_alamode("RTA.in")


class Result:
    def __init__(self, filename):
        self.kgrid = np.zeros(3, dtype=int)
        self.nat = 0
        self.q_coord = None
        self.gamma = None
        self.temperatures = None
        self.omega = None
        self.degeneracy = None
        self.volume = None
        self.read_result(filename)
        self.fullk = self.make_fullgrid()
        self.bz2irb = None

    def read_result(self, filename):
        total_irq = 0
        which_phonon = 0
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if "#SYSTEM" in line:
                    line = f.readline().rstrip().split()
                    self.nat = int(line[0])  # number of atoms
                    line = f.readline().rstrip()
                    self.volume = float(line)

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

                elif "#TEMPERATURE" in line:
                    line = f.readline().rstrip().split()
                    t_min = float(line[0])
                    t_max = float(line[1])
                    t_stp = float(line[2])
                    nstemp = int((t_max - t_min) / t_stp + 1)
                    self.temperatures = np.arange(t_min, t_max + t_stp, t_stp)
                    self.gamma = np.zeros((total_irq, self.nat * 3, nstemp))  # <-
                    self.vel = np.zeros((total_irq, self.nat * 3, 48, 3))
                    self.degeneracy = np.zeros(total_irq)

                elif "#K-point (irreducible)" in line:
                    self.omega = np.zeros((total_irq, 3 * self.nat))
                    for i in range(3 * self.nat * total_irq):
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
                    self.degeneracy[ik] = degen
                    for i in range(degen):
                        line = f.readline().split()
                        self.vel[ik, im, i, :] = np.array(line, dtype=float)
                    for it in range(nstemp):
                        line = f.readline().rstrip()
                        tmp_g = float(line)
                        self.gamma[ik, im, it] = tmp_g

                    if which_phonon == 3 * self.nat * total_irq:
                        break  # every thing is read

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
        return int(k + j * self.kgrid[2] + i * self.kgrid[1] * self.kgrid[2])

    def unfold(self, rotation):
        # find the map between fullk and k
        nsymm = len(rotation)
        qsym = np.zeros(rotation.shape)
        for i in range(nsymm):
            qsym[i, :, :] = np.linalg.inv(rotation[i, :, :]).T

        count = 0
        found = np.zeros(len(self.fullk), dtype=int)
        self.bz2irb = np.zeros(len(self.fullk), dtype=int)
        for i, k in enumerate(self.q_coord):
            cl = 0
            for rot in qsym:
                k2 = rot.dot(k)
                k2 = k2 - np.floor(k2)
                k2_index = self.get_knum(k2)
                if found[k2_index] == 0:
                    found[k2_index] = 1
                    self.bz2irb[k2_index] = i
                    count += 1
                    cl += 1
            if cl != int(self.degeneracy[i]):
                print("unfolding goes wrong")

    def interpolate_gamma(self, xk):
        if self.bz2irb is None:
            print("unfolding not done")
            return
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

        out = np.zeros((self.nat * 3, len(self.temperatures)))

        for imode in range(self.nat * 3):
            for itemp in range(len(self.temperatures)):
                v000 = self.gamma[self.bz2irb[self.get_knum(x000)], imode, itemp]
                v100 = self.gamma[self.bz2irb[self.get_knum(x100)], imode, itemp]
                v110 = self.gamma[self.bz2irb[self.get_knum(x110)], imode, itemp]
                v010 = self.gamma[self.bz2irb[self.get_knum(x010)], imode, itemp]
                v001 = self.gamma[self.bz2irb[self.get_knum(x001)], imode, itemp]
                v101 = self.gamma[self.bz2irb[self.get_knum(x101)], imode, itemp]
                v111 = self.gamma[self.bz2irb[self.get_knum(x111)], imode, itemp]
                v011 = self.gamma[self.bz2irb[self.get_knum(x011)], imode, itemp]

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

        return out

    def find_corner_fractional(self, xk):
        #
        # return fractional coordinate of the cell that contain xkf
        #
        nk1 = self.kgrid[0]
        nk2 = self.kgrid[1]
        nk3 = self.kgrid[2]
        nk23 = nk2 * nk3

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

    def average_gamma(self):
        nk, nmode, ntemp = self.gamma.shape
        tmp_gamma = np.zeros((nmode, ntemp))
        counter = np.zeros(nmode)
        for ik in range(nk):
            tmp_gamma[:, :] = 0
            counter[:] = 0
            for imode in range(nmode):
                for jmode in range(nmode):
                    if np.abs(self.omega[ik, imode] - self.omega[ik, jmode]) < 1:
                        counter[imode] += 1
                        tmp_gamma[imode, :] += self.gamma[ik, jmode, :]
            self.gamma[ik, imode, :] = tmp_gamma[imode, :] / counter[imode]
        return

    def calculate_kappa(self):
        bohr = 5.29177e-11
        hbar = 1.054e-34
        hart = 4.359e-18

        self.average_gamma()
        kappa = np.zeros((len(self.temperatures), 3, 3))
        for it, temp in enumerate(self.temperatures):
            for ik in range(len(self.q_coord)):
                for imode in range(self.nat * 3):
                    if self.omega[ik, imode] > 0:
                        if self.gamma[ik, imode, it] > 1e-10:
                            tau = Hz_to_kayser / (
                                2 * self.gamma[ik, imode, it] * 2.4188e-17
                            )  # in a.u.
                        else:
                            tau = 0
                        cv = self.calculate_cv(self.omega[ik, imode], temp)
                        # print(cv)
                        tmp = np.zeros((3, 3))
                        for iw in range(int(self.degeneracy[ik])):
                            for ix in range(3):
                                for iy in range(3):
                                    tmp[ix, iy] += (
                                        self.vel[ik, imode, iw, ix]
                                        * self.vel[ik, imode, iw, iy]
                                        / (2.187e6) ** 2
                                    )

                        kappa[it, :, :] += cv * tmp[:, :] * tau

        weight = 1 / (self.kgrid[0] * self.kgrid[1] * self.kgrid[2] * self.volume)
        factor = hart / (bohr * (hbar / hart))
        kappa[:, :, :] *= weight * factor
        return kappa

    def calculate_cv(self, omega, T):
        # energy in cm^-1
        # T in K
        e = omega * 4.55633e-6  # hartree
        kbT = T * 3.1668085e-6  # hartree
        x = e / kbT
        return e * x * np.e**x / ((np.e**x - 1) ** 2 * T)


def main(prefix):
    rot = read_symmetry("SYMM_INFO_PRIM")
    # print(rot)
    r3 = Result(prefix + ".result")
    r4 = Result(prefix + ".4ph.result101010")
    r4.unfold(rot)

    for ik, k in enumerate(r3.q_coord):
        gamma_k = r4.interpolate_gamma(k)
        r3.gamma[ik, :, :] += gamma_k

    print(r3.calculate_kappa()[:, 0, 0])


# write_gamma_to_file(r3.omega, gamma[:,:,-1], "interpolated.dat")
# write_gamma_to_file(r4.omega, r4.gamma[:,:,-1], "result4.dat")
# write_gamma_to_file(r3.omega, r3.gamma[:,:,-1], "result3.dat")
