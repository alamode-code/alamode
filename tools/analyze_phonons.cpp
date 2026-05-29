/*
 analyze_phonons.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "analyze_phonons.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "memory.h"

using namespace std;

int main(int argc, char *argv[])
{
    string str;

    cout << "# Result analyzer ver. 1.0.6\n";
    cout << "# Input file : " << argv[1] << '\n';
    calc = argv[2];
    average_gamma = atoi(argv[3]);

    // Read information from .result file

    ifs.open(argv[1], std::ios::in);

    if (!ifs) {
        cout << "ERROR: Cannot open file " << argv[1] << '\n';
        exit(1);
    }

    if (!locate_tag("#SYSTEM")) {
        cout << "ERROR: Cannot find #SYSTEM tag\n";
        exit(1);
    }

    ifs >> nat >> nkd;
    ifs >> volume;

    ns = nat * 3;

    if (!locate_tag("#TEMPERATURE")) {
        cout << "ERROR: Cannot find #TEMPERATURE tag\n";
        exit(1);
    }

    ifs >> tmin >> tmax >> dt;
    if (abs(dt) < eps && (tmin == tmax)) dt = 1.0;
    nt = static_cast<int>((tmax - tmin) / dt) + 1;
    allocate(temp, nt);
    for (i = 0; i < nt; ++i)
        temp[i] = tmin + dt * static_cast<double>(i);

    if (!locate_tag("#CLASSICAL")) {
        classical = false;
    } else {
        ifs >> classical;
    }

    if (!locate_tag("#KPOINT")) {
        cout << "ERROR: Cannot find #KPOINT tag\n";
        exit(1);
    }

    ifs >> nkx >> nky >> nkz;
    ifs >> nk;

    if (!locate_tag("##Phonon Frequency")) {
        cout << "ERROR: Cannot find ##Phonon Frequency tag\n";
        exit(1);
    }
    getline(ifs, str);

    allocate(omega, nk, ns);
    allocate(tau, nt, nk, ns);
    allocate(vel, nk, ns, 48, 3);
    allocate(n_weight, nk);

    int idummy, jdummy;

    for (i = 0; i < nk; ++i) {
        for (j = 0; j < ns; ++j) {
            ifs >> idummy >> jdummy >> omega[i][j];
        }
    }

    if (!locate_tag("##Phonon Relaxation Time")) {
        cout << "ERROR: Cannot find ##Phonon Relaxation Time tag\n";
        exit(1);
    }

    double vel_tmp[3];
    double damp_tmp;

    for (i = 0; i < nk; ++i) {
        for (j = 0; j < ns; ++j) {
            getline(ifs, str);
            getline(ifs, str);

            ifs >> n_weight[i];

            for (k = 0; k < n_weight[i]; ++k) {
                ifs >> vel_tmp[0] >> vel_tmp[1] >> vel_tmp[2];

                for (l = 0; l < 3; ++l)
                    vel[i][j][k][l] = vel_tmp[l];
            }

            for (k = 0; k < nt; ++k) {
                ifs >> damp_tmp;
                if (omega[i][j] < eps6) {
                    tau[k][i][j] = 0.0; // Neglect contributions from imaginary branches
                } else {
                    tau[k][i][j] = 1.0e+12 * Hz_to_kayser * 0.5 / damp_tmp;
                }
            }
            ifs.ignore();
            getline(ifs, str);
        }
    }

    ifs.close();

    if (average_gamma) average_gamma_at_degenerate_point(omega, tau, nt, nk, ns);

    if (calc == "tau") {

        // Print phonon lifetimes at uniform q-grid at the given temperature

        beg_k = atoi(argv[4]) - 1;
        end_k = atoi(argv[5]);

        if (end_k == 0) end_k = nk;

        if (beg_k < 0 || beg_k >= nk || end_k < 0 || end_k > nk) {
            cout << "ERROR: kpoint index out-of-range [" << 1 << ":" << nk << "] : " << setw(5) << beg_k + 1 << setw(5)
                 << end_k << '\n';
            exit(1);
        }

        beg_s = atoi(argv[6]) - 1;
        end_s = atoi(argv[7]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }

        int itemp;
        double target_temp;

        target_temp = atof(argv[8]);
        if (fmod(target_temp - tmin, dt) > eps12) {
            cout << "ERROR: No information is found at the given temperature." << '\n';
            exit(1);
        }
        itemp = static_cast<int>((target_temp - tmin) / dt);

        isotope = atoi(argv[9]);
        file_isotope = argv[10];
        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        calc_tau(itemp);

    } else if (calc == "tau_temp") {

        // Print temperature dependence of phonon lifetime

        int target_k, target_s;

        target_k = atoi(argv[4]) - 1;
        target_s = atoi(argv[5]) - 1;

        if (target_k < 0 || target_k >= nk) {
            cout << "ERROR: kpoint index out-of-range [" << 1 << ":" << nk << "] : " << setw(5) << target_k + 1 << '\n';
            exit(1);
        }
        if (target_s < 0 || target_s >= ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << target_s + 1 << '\n';
            exit(1);
        }

        isotope = atoi(argv[6]);
        file_isotope = argv[7];
        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        calc_tau_temp(target_k, target_s);

    } else if (calc == "kappa") {

        // Print thermal conductivity

        beg_s = atoi(argv[4]) - 1;
        end_s = atoi(argv[5]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }

        isotope = atoi(argv[6]);
        file_isotope = argv[7];
        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        calc_kappa();

    } else if (calc == "cumulative") {

        // Print the cumulative thermal conductivity
        // isotropic case

        double max_len, d_len;
        int itemp;
        double target_temp;

        beg_s = atoi(argv[4]) - 1;
        end_s = atoi(argv[5]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }
        isotope = atoi(argv[6]);
        file_isotope = argv[7];

        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        max_len = atof(argv[8]);
        d_len = atof(argv[9]);

        target_temp = atof(argv[10]);
        if (fmod(target_temp - tmin, dt) > eps12) {
            cout << "ERROR: No information is found at the given temperature." << '\n';
            exit(1);
        }
        itemp = static_cast<int>((target_temp - tmin) / dt);

        int nsample = atoi(argv[11]);
        std::string gridtype = argv[12];

        calc_kappa_cumulative(max_len, d_len, itemp, nsample, gridtype);

    } else if (calc == "cumulative2") {

        // Print the cumulative thermal conductivity
        // Direction specific case

        double max_len, d_len;
        int size_flag[3];
        int itemp;
        double target_temp;

        beg_s = atoi(argv[4]) - 1;
        end_s = atoi(argv[5]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }

        isotope = atoi(argv[6]);
        file_isotope = argv[7];

        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        max_len = atof(argv[8]);
        d_len = atof(argv[9]);

        target_temp = atof(argv[10]);
        if (fmod(target_temp - tmin, dt) > eps12) {
            cout << "ERROR: No information is found at the given temperature.\n";
            exit(1);
        }
        itemp = static_cast<int>((target_temp - tmin) / dt);

        for (i = 0; i < 3; ++i) {
            size_flag[i] = atoi(argv[11 + i]);
        }

        int nsample = atoi(argv[14]);
        std::string gridtype = argv[15];

        calc_kappa_cumulative2(max_len, d_len, itemp, size_flag, nsample, gridtype);

    } else if (calc == "kappa_matthiessen") {

        // Print the thermal conductivity with boundary effects
        // considered by the Matthiessen's rule.

        double max_len, d_len;
        int size_flag[3];
        int itemp;
        double target_temp;

        beg_s = atoi(argv[4]) - 1;
        end_s = atoi(argv[5]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }

        isotope = atoi(argv[6]);
        file_isotope = argv[7];

        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        max_len = atof(argv[8]);
        d_len = atof(argv[9]);

        target_temp = atof(argv[10]);
        if (fmod(target_temp - tmin, dt) > eps12) {
            cout << "ERROR: No information is found at the given temperature.\n";
            exit(1);
        }
        itemp = static_cast<int>((target_temp - tmin) / dt);

        for (i = 0; i < 3; ++i) {
            size_flag[i] = atoi(argv[11 + i]);
        }

        calc_kappa_boundary2(max_len, d_len, itemp, size_flag);

    } else if (calc == "kappa_boundary") {

        // Print the temperature dependence of thermal conductivity
        // with boundary effects considered by the Matthiessen's rule.

        double len_boundary;

        beg_s = atoi(argv[4]) - 1;
        end_s = atoi(argv[5]);

        if (end_s == 0) end_s = ns;

        if (beg_s < 0 || beg_s >= ns || end_s < 0 || end_s > ns) {
            cout << "ERROR: mode index out-of-range [" << 1 << ":" << ns << "] : " << setw(5) << beg_s + 1 << setw(5)
                 << end_s << '\n';
            exit(1);
        }
        isotope = atoi(argv[6]);
        file_isotope = argv[7];

        if (isotope) update_tau_isotope(file_isotope, omega, tau, nt, nk, ns);

        len_boundary = atof(argv[8]);

        calc_kappa_boundary(len_boundary);
    }

    return 0;
}

void calc_tau(int itemp)
{
    int ik, is;

    double vel_norm;

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);
    double kappa[3][3];
    double tau_tmp, c_tmp;

    cout << "# Phonon lifetime at temperature " << temp[itemp] << " K.\n";
    cout << "# kpoint range " << beg_k + 1 << " " << end_k << '\n';
    cout << "# mode   range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "#  ik,  is, Frequency [cm^{-1}], Lifetime [ps], |Velocity| [m/s], MFP [nm], ";
    cout << "Multiplicity, Thermal conductivity par mode (xx, xy, ...) [W/mK]\n";

    for (ik = beg_k; ik < end_k; ++ik) {
        for (is = beg_s; is < end_s; ++is) {

            tau_tmp = tau[itemp][ik][is];
            if (classical) {
                c_tmp = k_Boltzmann;
            } else {
                c_tmp = Cv(omega[ik][is], temp[itemp]);
            }

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    kappa[i][j] = 0.0;
                }
            }

            for (k = 0; k < n_weight[ik]; ++k) {
                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        kappa[i][j] += c_tmp * vel[ik][is][k][i] * vel[ik][is][k][j] * tau_tmp;
                    }
                }
            }

            vel_norm = sqrt(vel[ik][is][0][0] * vel[ik][is][0][0] + vel[ik][is][0][1] * vel[ik][is][0][1] +
                            vel[ik][is][0][2] * vel[ik][is][0][2]);
            cout << setw(5) << ik + 1 << setw(5) << is + 1;
            cout << setw(15) << omega[ik][is] << setw(15) << tau_tmp;
            cout << setw(15) << vel_norm << setw(15) << tau_tmp * vel_norm * 0.001;
            cout << setw(5) << n_weight[ik];

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    cout << setw(15) << kappa[i][j] * factor / static_cast<double>(n_weight[ik]);
                }
            }
            cout << '\n';
        }
    }
}

void calc_tau_temp(int target_k, int target_s)
{
    double vel_norm;

    vel_norm = sqrt(vel[target_k][target_s][0][0] * vel[target_k][target_s][0][0] +
                    vel[target_k][target_s][0][1] * vel[target_k][target_s][0][1] +
                    vel[target_k][target_s][0][2] * vel[target_k][target_s][0][2]);

    cout << "# Temperature dependence of the damping function will be printed\n";
    cout << "# for phonon specified by kpoint " << target_k + 1 << " and mode " << target_s + 1 << '\n';
    cout << "# Frequency = " << omega[target_k][target_s] << " [cm^-1]\n";
    cout << "# Velocity  = " << vel_norm << " [m/s]\n";
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "# Temperature [k], Lifetime [ps], MFP [nm]\n";

    for (i = 0; i < nt; ++i) {
        cout << setw(9) << temp[i];
        cout << setw(15) << tau[i][target_k][target_s];
        cout << setw(15) << tau[i][target_k][target_s] * vel_norm * 0.001 << '\n';
    }
}

void calc_kappa()
{
    int it, ik, is;
    double ***kappa;
    double c_tmp, tau_tmp;

    allocate(kappa, nt, 3, 3);

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);

    cout << "# Temperature dependence of thermal conductivity will be printed.\n";
    cout << "# mode range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "# Temperature [K], kappa [W/mK] (xx, xy, xz, yx, yy, yz, zx, zy, zz)\n";

    for (i = 0; i < nt; ++i) {
        for (j = 0; j < 3; ++j) {
            for (k = 0; k < 3; ++k) {
                kappa[i][j][k] = 0.0;
            }
        }
    }

    for (it = 0; it < nt; ++it) {
        for (ik = 0; ik < nk; ++ik) {
            for (is = beg_s; is < end_s; ++is) {

                tau_tmp = tau[it][ik][is];

                if (classical) {
                    c_tmp = k_Boltzmann;
                } else {
                    c_tmp = Cv(omega[ik][is], temp[it]);
                }

                for (i = 0; i < n_weight[ik]; ++i) {
                    for (j = 0; j < 3; ++j) {
                        for (k = 0; k < 3; ++k) {
                            kappa[it][j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp;
                        }
                    }
                }
            }
        }

        cout << setw(10) << temp[it];
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                cout << setw(15) << kappa[it][i][j] * factor;
            }
        }
        cout << '\n';
    }

    deallocate(kappa);
}


void calc_kappa_cumulative(double max_length, double delta_length, int itemp, const int nsample,
                           const std::string &gridtype)
{
    double length;
    double kappa[3][3];
    int il, ik, is;
    int nsame;
    double tau_tmp, c_tmp, vel_tmp, mfp_tmp;

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);


    cout << "# Cumulative thermal conductivity at temperature " << temp[itemp] << " K.\n";
    cout << "# mode range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings." << '\n';
    cout << "# Each phonon contribute to the total thermal conductivity if\n";
    cout << "# (v_x)^2+(v_y)^2+(v_z)^2 < L^2 is satisfied.\n";
    cout << "# L [nm], kappa [W/mK] (xx, xy, ...)\n";

    std::vector<double> length_vec;
    int nlength;

    if (delta_length > eps) {
        nlength = static_cast<int>(max_length / delta_length);
        length_vec.resize(nlength);
        for (il = 0; il < nlength; ++il) {
            length_vec[il] = delta_length * static_cast<double>(il);
        }
    } else {
        nlength = nsample;

        double mfp_min = DBL_MAX;
        double mfp_max = 0;

        for (ik = 0; ik < nk; ++ik) {
            for (is = beg_s; is < end_s; ++is) {
                tau_tmp = tau[itemp][ik][is];
                vel_tmp = pow(vel[ik][is][0][0], 2) + pow(vel[ik][is][0][1], 2) + pow(vel[ik][is][0][2], 2);
                mfp_tmp = tau_tmp * sqrt(vel_tmp) * 0.001; // in nm unit
                if (mfp_tmp > eps6) {
                    mfp_min = min(mfp_min, mfp_tmp);
                    mfp_max = max(mfp_max, mfp_tmp);
                }
            }
        }

        length_vec.resize(nlength);

        if (gridtype == "linear") {
            const double dl = (mfp_max - mfp_min) / static_cast<double>(nlength);
            for (il = 0; il < nlength; ++il) {
                length_vec[il] = mfp_min + dl * static_cast<double>(il);
            }
        } else if (gridtype == "log" || gridtype == "logarithmic") {
            for (il = 0; il < nlength; ++il) {
                length_vec[il] =
                    mfp_min * std::pow(mfp_max / mfp_min, static_cast<double>(il) / static_cast<double>(nlength - 1));
            }
        }
    }

    for (il = 0; il < nlength; ++il) {
        length = length_vec[il];

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                kappa[i][j] = 0.0;
            }
        }

#pragma omp parallel
        {
            double kappa_omp[3][3];
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    kappa_omp[i][j] = 0.0;
                }
            }

#pragma omp for private(nsame, tau_tmp, c_tmp, vel_tmp, mfp_tmp, is, i, j, k)
            for (ik = 0; ik < nk; ++ik) {
                nsame = n_weight[ik];

                for (is = beg_s; is < end_s; ++is) {
                    tau_tmp = tau[itemp][ik][is];

                    if (classical) {
                        c_tmp = k_Boltzmann;
                    } else {
                        c_tmp = Cv(omega[ik][is], temp[itemp]);
                    }

                    vel_tmp = pow(vel[ik][is][0][0], 2) + pow(vel[ik][is][0][1], 2) + pow(vel[ik][is][0][2], 2);
                    mfp_tmp = tau_tmp * sqrt(vel_tmp) * 0.001;

                    if (mfp_tmp < length) {
                        for (i = 0; i < nsame; ++i) {
                            for (j = 0; j < 3; ++j) {
                                for (k = 0; k < 3; ++k) {
                                    kappa_omp[j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp;
                                }
                            }
                        }
                    }
                }
            }
#pragma omp critical
            {
                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        kappa[i][j] += kappa_omp[i][j];
                    }
                }
            }
        }

        cout << setw(15) << length;
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                cout << setw(15) << kappa[i][j] * factor;
            }
        }
        cout << '\n';
    }
}

void calc_kappa_cumulative2(double max_length, double delta_length, int itemp, int flag[3], const int nsample,
                            const std::string &gridtype)
{
    double length;
    double kappa[3][3];
    int il, ik, is;
    int nsame;
    double tau_tmp, c_tmp;
    double mfp_tmp[3];
    bool is_longer_than_L[3];

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);


    cout << "# Cumulative thermal conductivity at temperature " << temp[itemp] << " K.\n";
    cout << "# mode range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "# Each phonon contribute to the total thermal conductivity if\n";
    cout << "# |v_{x,y,z}| < L is satisfied.\n";
    cout << "# Boundary direction flag  :" << flag[0] << " " << flag[1] << " " << flag[2] << '\n';
    cout << "# L [nm], kappa [W/mK] (xx, xy, ...)\n";


    std::vector<double> length_vec;
    int nlength;

    if (delta_length > eps) {
        nlength = static_cast<int>(max_length / delta_length);
        length_vec.resize(nlength);
        for (il = 0; il < nlength; ++il) {
            length_vec[il] = delta_length * static_cast<double>(il);
        }
    } else {
        nlength = nsample;

        double mfp_min = DBL_MAX;
        double mfp_max = 0;
        double vel_tmp;
        double mfp_norm;

        for (ik = 0; ik < nk; ++ik) {
            for (is = beg_s; is < end_s; ++is) {
                tau_tmp = tau[itemp][ik][is];
                vel_tmp = pow(vel[ik][is][0][0], 2) + pow(vel[ik][is][0][1], 2) + pow(vel[ik][is][0][2], 2);
                mfp_norm = tau_tmp * sqrt(vel_tmp) * 0.001; // in nm unit
                if (mfp_norm > eps6) {
                    mfp_min = min(mfp_min, mfp_norm);
                    mfp_max = max(mfp_max, mfp_norm);
                }
            }
        }

        length_vec.resize(nlength);

        if (gridtype == "linear") {
            const double dl = (mfp_max - mfp_min) / static_cast<double>(nlength);
            for (il = 0; il < nlength; ++il) {
                length_vec[il] = mfp_min + dl * static_cast<double>(il);
            }
        } else if (gridtype == "log" || gridtype == "logarithmic") {
            for (il = 0; il < nlength; ++il) {
                length_vec[il] =
                    mfp_min * std::pow(mfp_max / mfp_min, static_cast<double>(il) / static_cast<double>(nlength - 1));
            }
        }
    }

    for (il = 0; il < nlength; ++il) {
        length = length_vec[il];

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                kappa[i][j] = 0.0;
            }
        }

#pragma omp parallel
        {

            double kappa_omp[3][3];

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    kappa_omp[i][j] = 0.0;
                }
            }

#pragma omp for private(nsame, tau_tmp, c_tmp, mfp_tmp, is_longer_than_L, is, i, j, k)
            for (ik = 0; ik < nk; ++ik) {
                nsame = n_weight[ik];

                for (is = beg_s; is < end_s; ++is) {
                    tau_tmp = tau[itemp][ik][is];

                    if (classical) {
                        c_tmp = k_Boltzmann;
                    } else {
                        c_tmp = Cv(omega[ik][is], temp[itemp]);
                    }

                    for (i = 0; i < nsame; ++i) {

                        for (j = 0; j < 3; ++j) {
                            mfp_tmp[j] = tau_tmp * abs(vel[ik][is][i][j]) * 0.001;
                            is_longer_than_L[j] = mfp_tmp[j] > length;
                        }

                        if ((flag[0] && is_longer_than_L[0]) | (flag[1] && is_longer_than_L[1]) |
                            (flag[2] && is_longer_than_L[2]))
                            continue;

                        for (j = 0; j < 3; ++j) {
                            for (k = 0; k < 3; ++k) {
                                kappa_omp[j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp;
                            }
                        }
                    }
                }
            }

#pragma omp critical
            {
                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        kappa[i][j] += kappa_omp[i][j];
                    }
                }
            }
        }

        cout << setw(15) << length;
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                cout << setw(15) << kappa[i][j] * factor;
            }
        }
        cout << '\n';
    }
}

void calc_kappa_boundary(const double len_boundary)
{
    int it, ik, is;
    double ***kappa;
    double c_tmp, tau_tmp;
    double mfp_tmp, vel_norm;

    allocate(kappa, nt, 3, 3);

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);

    cout << "# Temperature dependence of thermal conductivity with boundary effects.\n";
    cout << "# mode range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "# Size of boundary " << len_boundary << " [nm]\n";
    cout << "# Temperature [K], kappa [W/mK] (xx, xy, xz, yx, yy, yz, zx, zy, zz)\n";

    for (i = 0; i < nt; ++i) {
        for (j = 0; j < 3; ++j) {
            for (k = 0; k < 3; ++k) {
                kappa[i][j][k] = 0.0;
            }
        }
    }

    for (it = 0; it < nt; ++it) {
        for (ik = 0; ik < nk; ++ik) {
            for (is = beg_s; is < end_s; ++is) {

                tau_tmp = tau[it][ik][is];

                if (classical) {
                    c_tmp = k_Boltzmann;
                } else {
                    c_tmp = Cv(omega[ik][is], temp[it]);
                }

                vel_norm = vel[ik][is][0][0] * vel[ik][is][0][0] + vel[ik][is][0][1] * vel[ik][is][0][1] +
                           vel[ik][is][0][2] * vel[ik][is][0][2];
                mfp_tmp = std::sqrt(vel_norm) * tau_tmp * 0.001;

                for (i = 0; i < n_weight[ik]; ++i) {
                    for (j = 0; j < 3; ++j) {
                        for (k = 0; k < 3; ++k) {
                            kappa[it][j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp * len_boundary /
                                               (len_boundary + 2.0 * mfp_tmp);
                        }
                    }
                }
            }
        }

        cout << setw(10) << temp[it];
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                cout << setw(15) << kappa[it][i][j] * factor;
            }
        }
        cout << '\n';
    }

    deallocate(kappa);
}

void calc_kappa_boundary2(double max_length, double delta_length, int itemp, int flag[3])
{
    int nlength = static_cast<int>(max_length / delta_length);
    double length;
    double kappa[3][3];
    int il, ik, is;
    int nsame;
    double tau_tmp, c_tmp;
    double mfp_tmp[3];

    double factor = 1.0e+18 / (pow(Bohr_in_Angstrom, 3) * static_cast<double>(nkx * nky * nkz) * volume);


    cout << "# Size dependent thermal conductivity at temperature " << temp[itemp] << " K.\n";
    cout << "# Relaxation time will be modified following Matthiesen's rule \n";
    cout << "# mode range " << beg_s + 1 << " " << end_s << '\n';
    if (isotope) cout << "# With phonon-isotope scatterings.\n";
    cout << "# Size change flag  :" << flag[0] << " " << flag[1] << " " << flag[2] << '\n';
    cout << "# L [nm], kappa [W/mK] (xx, xy, ...)\n";


    for (il = 0; il < nlength; ++il) {
        length = delta_length * static_cast<double>(il);

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                kappa[i][j] = 0.0;
            }
        }

        for (ik = 0; ik < nk; ++ik) {
            nsame = n_weight[ik];

            for (is = beg_s; is < end_s; ++is) {
                tau_tmp = tau[itemp][ik][is];

                if (classical) {
                    c_tmp = k_Boltzmann;
                } else {
                    c_tmp = Cv(omega[ik][is], temp[itemp]);
                }

                for (i = 0; i < nsame; ++i) {

                    for (j = 0; j < 3; ++j) {
                        mfp_tmp[j] = tau_tmp * abs(vel[ik][is][i][j]) * 0.001;

                        if (flag[j]) {
                            for (k = 0; k < 3; ++k) {
                                kappa[j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp * length /
                                               (length + 2.0 * mfp_tmp[j]);
                            }
                        } else {
                            for (k = 0; k < 3; ++k) {
                                kappa[j][k] += c_tmp * vel[ik][is][i][j] * vel[ik][is][i][k] * tau_tmp;
                            }
                        }
                    }
                }
            }
        }

        cout << setw(15) << length;
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                cout << setw(15) << kappa[i][j] * factor;
            }
        }
        cout << '\n';
    }
}


int locate_tag(string key)
{
    int ret = 0;
    string line, line2;

    ifs.clear();
    ifs.seekg(0, std::ios_base::beg);

    while (getline(ifs, line)) {

        line2 = line;
        line = trim(line2);

        if (line == key) {
            ret = 1;
            break;
        }
    }
    return ret;
}

double Cv(double omega, double temp)
{
    double x;

    if (abs(temp) < 1.0e-12 || omega < eps8) return 0.0;

    x = omega * kayser_to_Ryd / (temp * T_to_Ryd);
    return k_Boltzmann * pow(x / (2.0 * sinh(0.5 * x)), 2.0);
}

void update_tau_isotope(const std::string file, double **omega, double ***tau, const int nt, const int nk, const int ns)
{
    int i;
    int ik, is, jk, js;
    double omega_tmp, tau_tmp;
    ifstream ifs;
    string line;
    double **tau_isotope;

    ifs.open(file.c_str(), ios::in);

    if (!ifs) {
        cout << "ERROR: Cannot open file " << file << '\n';
        exit(1);
    }

    allocate(tau_isotope, nk, ns);

    for (i = 0; i < 3; ++i)
        getline(ifs, line);

    for (ik = 0; ik < nk; ++ik) {
        getline(ifs, line);
        getline(ifs, line);
        for (is = 0; is < ns; ++is) {
            ifs >> jk >> js >> omega_tmp >> tau_tmp;
            if (jk < 1 || jk > nk) {
                cout << "ERROR: In file " << file << ", k point index is out-of-range. \n";
                exit(1);
            }
            if (js < 1 || js > ns) {
                cout << "ERROR: In file " << file << ", mode index is out-of-range. \n";
                exit(1);
            }

            if (omega[ik][is] < eps6) {
                tau_isotope[ik][is] = 0.0; // Neglect contributions from imaginary branches
            } else {
                tau_isotope[ik][is] = 1.0e+12 * Hz_to_kayser * 0.5 / tau_tmp;
            }
        }
        ifs.ignore();
        getline(ifs, line);
    }
    ifs.close();

    average_gamma_isotope_at_degenerate_point(omega, tau_isotope, nk, ns);

    // Now, update the original tau

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (omega[ik][is] >= eps6) {
                for (i = 0; i < nt; ++i) {
                    tau_tmp = 1.0 / tau[i][ik][is] + 1.0 / tau_isotope[ik][is];
                    tau[i][ik][is] = 1.0 / tau_tmp;
                }
            }
        }
    }

    deallocate(tau_isotope);
}


void average_gamma_at_degenerate_point(double **e, double ***tau, const int nt, const int nk, const int ns)
{
    int ideg, is;
    double omega_prev, omega_now;
    double tol_omega = 1.0e-3;

    vector<int> degeneracy_at_k;
    double *damp_sum;

    allocate(damp_sum, nt);

    for (i = 0; i < nk; ++i) {

        degeneracy_at_k.clear();
        omega_prev = e[i][0];
        ideg = 1;

        for (j = 1; j < ns; ++j) {
            omega_now = e[i][j];

            if (abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_at_k.push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_at_k.push_back(ideg);

        is = 0;

        for (j = 0; j < degeneracy_at_k.size(); ++j) {
            ideg = degeneracy_at_k[j];

            if (ideg > 1) {

                for (k = 0; k < nt; ++k)
                    damp_sum[k] = 0.0;

                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < nt; ++l)
                        damp_sum[l] += 1.0 / tau[l][i][k];
                }


                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < nt; ++l)
                        tau[l][i][k] = static_cast<double>(ideg) / damp_sum[l];
                }
            }

            is += ideg;
        }
    }

    deallocate(damp_sum);
}


void average_gamma_isotope_at_degenerate_point(double **e, double **tau, const int nk, const int ns)
{
    int ideg, is;
    double omega_prev, omega_now;
    double tol_omega = 1.0e-3;

    vector<int> degeneracy_at_k;
    double damp_sum;

    for (i = 0; i < nk; ++i) {

        degeneracy_at_k.clear();
        omega_prev = e[i][0];
        ideg = 1;

        for (j = 1; j < ns; ++j) {
            omega_now = e[i][j];

            if (abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_at_k.push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_at_k.push_back(ideg);

        is = 0;

        for (j = 0; j < degeneracy_at_k.size(); ++j) {
            ideg = degeneracy_at_k[j];

            if (ideg > 1) {

                damp_sum = 0.0;

                for (k = is; k < is + ideg; ++k) {
                    damp_sum += 1.0 / tau[i][k];
                }

                for (k = is; k < is + ideg; ++k) {
                    tau[i][k] = static_cast<double>(ideg) / damp_sum;
                }
            }

            is += ideg;
        }
    }
}
