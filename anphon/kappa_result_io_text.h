/*
 kappa_result_io_text.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace PHON_NS
{
class Cell;
class KpointMeshUniform;

// Legacy text .result format (FILE_FORMAT = text and the one-way h5 import).
// All functions are rank-0-only by contract; they perform no MPI communication
// — rank guards stay at the call sites.
class KappaResultIOText
{
public:
    static void write_header(std::fstream &fs_result, const std::string &file_result,
                             const KpointMeshUniform *kmesh_in, const Cell &primcell, bool with_cell_geometry,
                             const bool classical_in, const int ismear_in, const double epsilon_in,
                             const double tmin_in, const double tmax_in, const double delta_t_in,
                             const std::string &file_fcs_in);

    static void check_consistency(std::fstream &fs_result, const std::string &file_result_in,
                                  const unsigned int nk_in[3], const unsigned int nk_irred_in,
                                  const Cell &primcell, const bool classical_in, const int ismear_in,
                                  const double epsilon_in, const double tmin_in, const double tmax_in,
                                  const double delta_t_in, const std::string &file_fcs_in);

    static void load_gamma_blocks(std::fstream &fs_result, const std::string &file_result,
                                  const unsigned int nk_irred, const unsigned int ns, const unsigned int ntemp,
                                  double **damping, std::vector<int> &vks_done_out, const char *label,
                                  const bool allow_truncate);

    static void write_gamma_batch(std::fstream &fs_result, const unsigned int ik, const unsigned int nshift,
                                  const unsigned int np, const KpointMeshUniform *kmesh_in, const unsigned int ns,
                                  const unsigned int ntemp, const double *const *const *vel_in,
                                  const double *const *damp_in, const char *label);

    static void write_frequency_block(std::fstream &fs_result, const KpointMeshUniform *kmesh_in,
                                      const double *const *eval_in, const unsigned int ns);

    static void write_frequency_velocity_block(std::fstream &fs_result, const KpointMeshUniform *kmesh_in,
                                               const double *const *eval_in, const double *const *const *vel_in,
                                               const unsigned int ns);

    static void write_ibte_Q_dF_block(std::fstream &fs_result, const double etemp,
                                      const KpointMeshUniform *kmesh_in, const double *const *Q_all,
                                      const double *const *const *df, const unsigned int ns);
};
} // namespace PHON_NS
