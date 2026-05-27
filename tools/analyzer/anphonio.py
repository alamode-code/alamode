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
                    try:
                        self.lattice_vector = np.zeros((3, 3), dtype=float)
                        for i in range(3):
                            self.lattice_vector[i, :] = np.array(
                                [float(t) for t in f.readline().strip().split()]
                            )
                        self.atomic_kinds = np.zeros(self.nat, dtype=int)
                        self.x_fractional = np.zeros((self.nat, 3), dtype=float)
                        for i in range(self.nat):
                            line = f.readline().strip().split()
                            self.atomic_kinds[i] = int(line[0])
                            self.x_fractional[i, :] = np.array(
                                [float(t) for t in line[1:]]
                            )
                    except Exception:
                        pass

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
