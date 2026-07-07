/*
 kappa_result_io_text.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "kappa_result_io_text.h"
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <mpi.h>
#include "constants.h"
#include "error.h"
#include "kpoint.h"
#include "system.h"

using namespace PHON_NS;

void KappaResultIOText::write_header(std::fstream &fs_result, const std::string &file_result,
                                     const KpointMeshUniform *kmesh_in, const Cell &primcell,
                                     const bool with_cell_geometry, const bool classical_in, const int ismear_in,
                                     const double epsilon_in, const double tmin_in, const double tmax_in,
                                     const double delta_t_in, const std::string &file_fcs_in)
{
    fs_result.open(file_result.c_str(), std::ios::out);
    if (!fs_result) {
        exit("setup_result_io", "Could not open file_result3");
    }

    fs_result << "## General information\n";
    fs_result << "#SYSTEM\n";
    fs_result << primcell.number_of_atoms << " " << primcell.number_of_elems << '\n';
    fs_result << primcell.volume << '\n';
    if (with_cell_geometry) {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                fs_result << std::setw(20) << std::scientific << primcell.lattice_vector(i, j);
            }
            fs_result << '\n';
        }
        for (auto i = 0; i < primcell.number_of_atoms; ++i) {
            for (auto j = 0; j < 3; ++j) {
                fs_result << std::setw(20) << std::scientific << primcell.x_fractional(i, j);
            }
            fs_result << '\n';
        }
    }
    fs_result << "#END SYSTEM\n";

    fs_result << "#KPOINT\n";
    fs_result << kmesh_in->nk_i[0] << " " << kmesh_in->nk_i[1] << " " << kmesh_in->nk_i[2] << '\n';
    fs_result << kmesh_in->nk_irred << '\n';

    for (int i = 0; i < kmesh_in->nk_irred; ++i) {
        fs_result << std::setw(6) << i + 1 << ":";
        for (int j = 0; j < 3; ++j) {
            fs_result << std::setw(15) << std::scientific << kmesh_in->kpoint_irred_all[i][0].kval[j];
        }
        fs_result << std::setw(12) << std::fixed << kmesh_in->weight_k[i] << '\n';
    }

    fs_result.unsetf(std::ios::fixed);

    fs_result << "#END KPOINT\n";

    fs_result << "#CLASSICAL\n";
    fs_result << classical_in << '\n';
    fs_result << "#END CLASSICAL\n";

    fs_result << "#FCSXML\n";
    fs_result << file_fcs_in << '\n';
    fs_result << "#END  FCSXML\n";

    fs_result << "#SMEARING\n";
    fs_result << ismear_in << '\n';
    fs_result << epsilon_in * Ry_to_kayser << '\n';
    fs_result << "#END SMEARING\n";

    fs_result << "#TEMPERATURE\n";
    fs_result << tmin_in << " " << tmax_in << " " << delta_t_in << '\n';
    fs_result << "#END TEMPERATURE\n";

    fs_result << "##END General information\n";
}

void KappaResultIOText::check_consistency(std::fstream &fs_result, const std::string &file_result_in,
                                          const unsigned int nk_in[3], const unsigned int nk_irred_in,
                                          const Cell &primcell, const bool classical_in, const int ismear_in,
                                          const double epsilon_in, const double tmin_in, const double tmax_in,
                                          const double delta_t_in, const std::string &file_fcs_in)
{
    // Read-only: this function only validates the header. The text path
    // reopens the stream for appending afterwards; the h5 import path must
    // never modify the legacy file.
    fs_result.open(file_result_in.c_str(), std::ios::in);
    if (!fs_result) {
        exit("check_consistency_restart", "Could not open file_result_in");
    }

    // Check the consistency

    std::string line_tmp, str_tmp;
    int natmin_tmp, nkd_tmp;
    int nk_tmp[3], nksym_tmp;
    int ismear, is_classical;
    double epsilon_tmp, T1, T2, delta_T;

    bool found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#SYSTEM") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) exit("check_consistency_restart", "Could not find #SYSTEM tag");

    fs_result >> natmin_tmp >> nkd_tmp;

    if (!(natmin_tmp == primcell.number_of_atoms && nkd_tmp == primcell.number_of_elems)) {
        exit("check_consistency_restart", "SYSTEM information is not consistent");
    }

    found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#KPOINT") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) exit("check_consistency_restart", "Could not find #KPOINT tag");

    fs_result >> nk_tmp[0] >> nk_tmp[1] >> nk_tmp[2];
    fs_result >> nksym_tmp;

    if (!(nk_in[0] == nk_tmp[0] && nk_in[1] == nk_tmp[1] && nk_in[2] == nk_tmp[2] && nk_irred_in == nksym_tmp)) {
        exit("check_consistency_restart", "KPOINT information is not consistent");
    }

    found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#CLASSICAL") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) {
        std::cout << " Could not find the #CLASSICAL tag in the restart file.\n";
        std::cout << " CLASSIACAL = 0 is assumed.\n";
        is_classical = 0;
    } else {
        fs_result >> is_classical;
    }
    if (static_cast<bool>(is_classical) != classical_in) {
        warn("check_consistency_restart", "CLASSICAL val is not consistent");
    }

    found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#FCSXML") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) exit("check_consistency_restart", "Could not find #FCSXML tag");

    fs_result >> str_tmp;
    if (str_tmp != file_fcs_in) {
        warn("check_consistency_restart", "FCSXML is not consistent");
    }

    found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#SMEARING") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) exit("check_consistency_restart", "Could not find #SMEARING tag");

    fs_result >> ismear;
    fs_result >> epsilon_tmp;

    if (ismear != ismear_in) {
        warn("check_consistency_restart", "Smearing method is not consistent");
    }
    if (ismear != -1 && std::abs(epsilon_tmp - epsilon_in * Ry_to_kayser) >= eps4) {
        std::cout << "epsilon from file : " << std::setw(15) << std::setprecision(10) << epsilon_tmp * Ry_to_kayser
                  << '\n';
        std::cout << "epsilon from input: " << std::setw(15) << std::setprecision(10) << epsilon_in * Ry_to_kayser
                  << '\n';
        warn("check_consistency_restart", "Smearing width is not consistent");
    }

    found_tag = false;
    while (fs_result >> line_tmp) {
        if (line_tmp == "#TEMPERATURE") {
            found_tag = true;
            break;
        }
    }
    if (!found_tag) exit("check_consistency_restart", "Could not find #TEMPERATURE tag");

    fs_result >> T1 >> T2 >> delta_T;

    if (!(T1 == tmin_in && T2 == tmax_in && delta_T == delta_t_in)) {
        exit("check_consistency_restart", "Temperature information is not consistent");
    }
}

void KappaResultIOText::load_gamma_blocks(std::fstream &fs_result, const std::string &file_result,
                                          const unsigned int nk_irred, const unsigned int ns,
                                          const unsigned int ntemp, double **damping,
                                          std::vector<int> &vks_done_out, const char *label,
                                          const bool allow_truncate)
{
    std::string line_tmp;
    unsigned int nk_tmp, ns_tmp;
    unsigned int multiplicity;
    double vel_dummy[3];
    bool truncate_tail = false;
    std::streampos truncate_pos = std::streampos(0);

    fs_result.clear();
    fs_result.seekg(0, std::ios::beg);

    while (true) {
        const auto block_start = fs_result.tellg();
        if (!(fs_result >> line_tmp)) break;
        if (line_tmp != "#GAMMA_EACH") continue;

        truncate_pos = block_start;

        if (!(fs_result >> nk_tmp >> ns_tmp >> multiplicity)) {
            truncate_tail = true;
            break;
        }

        if (nk_tmp < 1 || nk_tmp > nk_irred || ns_tmp < 1 || ns_tmp > ns) {
            const auto message =
                std::string("Invalid k-point or branch index in the ") + label + " restart (.result) file.";
            exit("prepare_restart", message.c_str());
        }

        const auto nks_tmp = (nk_tmp - 1) * ns + ns_tmp - 1;

        for (unsigned int i = 0; i < multiplicity; ++i) {
            if (!(fs_result >> vel_dummy[0] >> vel_dummy[1] >> vel_dummy[2])) {
                truncate_tail = true;
                break;
            }
        }
        if (truncate_tail) break;

        std::vector<double> damping_tmp(ntemp);
        for (unsigned int i = 0; i < ntemp; ++i) {
            if (!(fs_result >> damping_tmp[i])) {
                truncate_tail = true;
                break;
            }
            damping_tmp[i] *= kayser_to_Ry;
        }
        if (truncate_tail) break;

        std::string end_tag, end_name;
        if (!(fs_result >> end_tag >> end_name) || end_tag != "#END" || end_name != "GAMMA_EACH") {
            truncate_tail = true;
            break;
        }

        for (unsigned int i = 0; i < ntemp; ++i) {
            damping[nks_tmp][i] = damping_tmp[i];
        }
        vks_done_out.push_back(nks_tmp);
    }

    if (truncate_tail) {
        const auto message =
            std::string("Ignoring an incomplete ") + label + " #GAMMA_EACH block at the end of " + file_result + ".";
        warn("prepare_restart", message.c_str());

        // When the caller only imports the data (h5 migration), the legacy
        // file must stay byte-identical; the incomplete tail is simply
        // skipped in memory.
        if (!allow_truncate) return;

        fs_result.clear();
        fs_result.close();

        const auto truncate_offset = static_cast<off_t>(static_cast<std::streamoff>(truncate_pos));
        if (truncate(file_result.c_str(), truncate_offset) != 0) {
            const auto error_message = std::string("Could not truncate incomplete restart block in ") + file_result +
                                       ": " + std::strerror(errno);
            exit("prepare_restart", error_message.c_str());
        }

        fs_result.open(file_result.c_str(), std::ios::in | std::ios::out);
        if (!fs_result) {
            exit("prepare_restart", "Could not reopen restart file after truncating incomplete block.");
        }
    }
}

void KappaResultIOText::write_gamma_batch(std::fstream &fs_result, const unsigned int ik, const unsigned int nshift,
                                          const unsigned int np, const KpointMeshUniform *kmesh_in,
                                          const unsigned int ns, const unsigned int ntemp,
                                          const double *const *const *vel_in, const double *const *damp_in,
                                          const char *label)
{
    unsigned int k;
    std::ostringstream result_block;
    for (unsigned int j = 0; j < np; ++j) {

        const auto iks_g = ik * np + j + nshift;

        if (iks_g >= kmesh_in->nk_irred * ns) break;

        result_block << "#GAMMA_EACH\n";
        result_block << iks_g / ns + 1 << " " << iks_g % ns + 1 << '\n';

        const auto nk_equiv = kmesh_in->kpoint_irred_all[iks_g / ns].size();

        result_block << nk_equiv << '\n';
        for (k = 0; k < nk_equiv; ++k) {
            const auto ktmp = kmesh_in->kpoint_irred_all[iks_g / ns][k].knum;
            result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][0];
            result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][1];
            result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][2] << '\n';
        }

        for (k = 0; k < ntemp; ++k) {
            result_block << std::setw(15) << damp_in[iks_g][k] * Hz_to_kayser / time_ry << '\n';
        }
        result_block << "#END GAMMA_EACH\n";
    }
    fs_result << result_block.str();
    fs_result.flush();
    if (!fs_result) {
        const auto message = std::string("Could not write ") + label + " restart block.";
        exit("write_result_gamma", message.c_str());
    }
}

void KappaResultIOText::write_frequency_block(std::fstream &fs_result, const KpointMeshUniform *kmesh_in,
                                              const double *const *eval_in, const unsigned int ns)
{
    fs_result << "##Phonon Frequency\n";
    fs_result << "#K-point (irreducible), Branch, Omega (cm^-1)\n";
    for (int i = 0; i < kmesh_in->nk_irred; ++i) {
        const auto ik = kmesh_in->kpoint_irred_all[i][0].knum;
        for (auto is = 0; is < ns; ++is) {
            fs_result << std::setw(6) << i + 1 << std::setw(6) << is + 1;
            fs_result << std::setw(15) << in_kayser(eval_in[ik][is])
                      << '\n';
        }
    }
    fs_result << "##END Phonon Frequency\n\n";
    fs_result << "##Phonon Relaxation Time\n";
}

void KappaResultIOText::write_frequency_velocity_block(std::fstream &fs_result, const KpointMeshUniform *kmesh_in,
                                                       const double *const *eval_in,
                                                       const double *const *const *vel_in, const unsigned int ns)
{
    fs_result << "##Phonon Frequency" << '\n';
    fs_result << "#K-point (irreducible), Branch, Omega (cm^-1), Group velocity (m/s)" << '\n';

    double factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;
    for (int i = 0; i < kmesh_in->nk_irred; ++i) {
        const int ik = kmesh_in->kpoint_irred_all[i][0].knum;
        for (auto is = 0; is < ns; ++is) {
            fs_result << std::setw(6) << i + 1 << std::setw(6) << is + 1;
            fs_result << std::setw(15) << in_kayser(eval_in[ik][is]);
            fs_result << std::setw(15) << vel_in[ik][is][0] * factor << std::setw(15) << vel_in[ik][is][1] * factor
                      << std::setw(15) << vel_in[ik][is][2] * factor << '\n';
        }
    }

    fs_result << "##END Phonon Frequency" << '\n' << '\n';
    fs_result << "##Q and W at each temperature" << '\n';
}

void KappaResultIOText::write_ibte_Q_dF_block(std::fstream &fs_result, const double etemp,
                                              const KpointMeshUniform *kmesh_in, const double *const *Q_all,
                                              const double *const *const *df, const unsigned int ns)
{
    fs_result << std::setw(10) << etemp << '\n';

    for (auto ik = 0; ik < kmesh_in->nk_irred; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            auto k1 = kmesh_in->kpoint_irred_all[ik][0].knum;
            fs_result << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << '\n';
            fs_result << std::setw(15) << std::scientific << std::setprecision(5) << Q_all[ik][is]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][0]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][1]
                      << std::setw(15) << std::scientific << std::setprecision(5) << df[k1][is][2] << '\n';
        }
    }
    fs_result << '\n';
}
