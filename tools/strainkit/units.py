"""Physical constants and unit conversions.

The values mirror ``include/constants.h`` of ALAMODE so that files written by
these tools are interpreted by anphon with exactly the constants anphon uses.
(They differ from ``ase.units`` at the 1e-8 relative level, which is
physically irrelevant, but we deliberately do not mix two sets of constants.)
"""

BOHR_IN_ANGSTROM = 0.52917721092  # include/constants.h
RYD_IN_EV = 13.605693122994  # include/constants.h
EV_IN_J = 1.6021766208e-19  # value used in anphon (ifc_derivative.cpp)
EV_PER_ANG3_TO_GPA = EV_IN_J * 1.0e30 * 1.0e-9  # 160.21766208


def ang_to_bohr(x):
    return x / BOHR_IN_ANGSTROM


def bohr_to_ang(x):
    return x * BOHR_IN_ANGSTROM


def ev_to_ry(x):
    return x / RYD_IN_EV


def ry_to_ev(x):
    return x * RYD_IN_EV


def ev_per_ang_to_ry_per_bohr(f):
    """eV/Angstrom -> Ry/bohr (forces)."""
    return f * BOHR_IN_ANGSTROM / RYD_IN_EV


def ry_per_bohr_to_ev_per_ang(f):
    return f * RYD_IN_EV / BOHR_IN_ANGSTROM


def ev_per_ang3_to_gpa(x):
    return x * EV_PER_ANG3_TO_GPA


def gpa_to_ev_per_ang3(x):
    return x / EV_PER_ANG3_TO_GPA


def volume_times_c_to_ry(volume_ang3, c_ev_per_ang3):
    """V * C in Ry, the unit anphon expects in elastic_constants.in / C1_array.in.

    ``volume_ang3`` is the volume of the anphon primitive cell in Angstrom^3 and
    ``c_ev_per_ang3`` the constant (or stress) in eV/Angstrom^3.
    """
    return volume_ang3 * c_ev_per_ang3 / RYD_IN_EV


def ry_to_c_ev_per_ang3(value_ry, volume_ang3):
    """Inverse of :func:`volume_times_c_to_ry`."""
    return value_ry * RYD_IN_EV / volume_ang3
