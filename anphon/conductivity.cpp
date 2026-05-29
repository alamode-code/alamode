/*
 conductivity.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "conductivity.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "integration.h"
#include "interpolation.h"
#include "isotope.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "phonon_velocity.h"
#include "progress_bar.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

Conductivity::Conductivity(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Conductivity::~Conductivity()
{
    deallocate_variables();
}

void Conductivity::set_default_variables()
{
    calc_kappa_spec = 0;
    ntemp = 0;
    damping3 = nullptr;
    damping4 = nullptr;
    kappa = nullptr;
    kappa_3only = nullptr;
    kappa_spec = nullptr;
    kappa_coherent = nullptr;
    kappa_coherent_block = nullptr;
    temperature = nullptr;
    vel = nullptr;
    vel_4ph = nullptr;
    velmat = nullptr;
    calc_coherent = 0;
    calc_coherent_block = 0;
    file_coherent_elems = "";
    nshift_restart = 0;
    nshift_restart4 = 0;
    kmesh_4ph = nullptr;
    fph_rta = 0;
    dymat_4ph = nullptr;
    phase_storage_4ph = nullptr;
    restart_flag_3ph = false;
    restart_flag_4ph = false;
    file_result3 = "";
    file_result4 = "";
    interpolator = "log-linear";
    len_boundary = 0.0;
    write_interpolation = 0;
}

void Conductivity::deallocate_variables()
{
    if (damping3) {
        deallocate(damping3);
    }
    if (damping4) {
        deallocate(damping4);
    }
    if (kappa) {
        deallocate(kappa);
    }
    if (kappa_3only) {
        deallocate(kappa_3only);
    }
    if (kappa_spec) {
        deallocate(kappa_spec);
    }
    if (kappa_coherent) {
        deallocate(kappa_coherent);
    }
    if (kappa_coherent_block) {
        deallocate(kappa_coherent_block);
    }
    if (temperature) {
        deallocate(temperature);
    }
    if (vel) {
        deallocate(vel);
    }
    if (velmat) {
        deallocate(velmat);
    }
    if (vel_4ph) {
        deallocate(vel_4ph);
    }
    delete kmesh_4ph;
    delete dymat_4ph;
    delete phase_storage_4ph;
}

void Conductivity::setup_kappa()
{
    MPI_Bcast(&calc_coherent, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&calc_coherent_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&restart_flag_3ph, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    // quartic_mode is boardcasted in fcs_phonon->setup
    if (anharmonic_core->quartic_mode > 0) {
        fph_rta = 1;
    } else {
        fph_rta = 0;
    }

    nk_3ph = dos->kmesh_dos->nk;
    ns = dynamical->neval;

    ntemp = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    allocate(temperature, ntemp);

    for (auto i = 0; i < ntemp; ++i) {
        temperature[i] = system->Tmin + static_cast<double>(i) * system->dT;
    }

    const auto nks_total = dos->kmesh_dos->nk_irred * ns;
    const auto nks_each_thread = nks_total / mympi->nprocs;
    const auto nrem = nks_total - nks_each_thread * mympi->nprocs;

    if (nrem > 0) {
        allocate(damping3, (nks_each_thread + 1) * mympi->nprocs, ntemp);
    } else {
        allocate(damping3, nks_total, ntemp);
    }

    if (len_boundary > eps) {
        if (mympi->my_rank == 0) {
            std::cout << "\n    Bounday scattering effect will be considered with len_boundary = " << len_boundary
                      << "\n\n";
        }
    }

    const auto factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;

    if (mympi->my_rank == 0) {
        allocate(vel, nk_3ph, ns, 3);
        if (calc_coherent || calc_coherent_block) {
            allocate(velmat, nk_3ph, ns, ns, 3);
        }
    } else {
        allocate(vel, 1, 1, 1);
        if (calc_coherent || calc_coherent_block) {
            allocate(velmat, 1, 1, 1, 3);
        }
    }

    phonon_velocity->get_phonon_group_velocity_mesh_mpi(*dos->kmesh_dos, system->get_primcell().lattice_vector, vel);

    if (mympi->my_rank == 0) {
        for (auto i = 0; i < nk_3ph; ++i) {
            for (auto j = 0; j < ns; ++j) {
                for (auto k = 0; k < 3; ++k)
                    vel[i][j][k] *= factor;
            }
        }
    }

    if (calc_coherent || calc_coherent_block) {
        phonon_velocity->calc_phonon_velmat_mesh(velmat);
        check_velocity_matrix_consistency(dos->kmesh_dos, dos->dymat_dos->get_eigenvalues());
        if (calc_coherent == 2) {
            file_coherent_elems = input->job_title + ".kc_elem";
        }
    }

    vks_job.clear();

    for (auto i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
        for (auto j = 0; j < ns; ++j) {
            vks_job.insert(i * ns + j);
        }
    }

    // prepare IO for 3ph only
    setup_result_io(1);
    prepare_restart(1);

    // setting up related to setup_kappa_4ph are inside here
    if (fph_rta > 0) setup_kappa_4ph();
}

void Conductivity::setup_kappa_4ph()
{
    if (!temperature) {
        ntemp = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
        allocate(temperature, ntemp);

        for (auto i = 0; i < ntemp; ++i) {
            temperature[i] = system->Tmin + static_cast<double>(i) * system->dT;
        }
    }

    ns = dynamical->neval;
    nk_3ph = dos->kmesh_dos->nk;
    MPI_Bcast(&nk_coarse[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&restart_flag_4ph, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    const auto nks_total = dos->kmesh_dos->nk_irred * ns;
    const auto nks_each_thread = nks_total / mympi->nprocs;
    const auto nrem = nks_total - nks_each_thread * mympi->nprocs;

    // Set KMESH_COARSE for 4-ph calculation.
    // If nk_coarse is not set, use the same k-mesh as the 3-ph calculation.
    unsigned int nkc_tmp[3] = {};
    if (nk_coarse[0] * nk_coarse[1] * nk_coarse[2] > 0) {
        for (auto i = 0; i < 3; i++)
            nkc_tmp[i] = nk_coarse[i];
    } else {
        for (auto i = 0; i < 3; i++)
            nkc_tmp[i] = dos->kmesh_dos->nk_i[i];
    }

    if (mympi->my_rank == 0) {
        std::cout << "\n";
        std::cout << " Four-phonon scattering rate will be calculated additionally.\n";
        std::cout << " KMESH for 4-ph:\n";
        std::cout << "   nk1 : " << std::setw(5) << nkc_tmp[0] << '\n';
        std::cout << "   nk2 : " << std::setw(5) << nkc_tmp[1] << '\n';
        std::cout << "   nk3 : " << std::setw(5) << nkc_tmp[2] << '\n';
    }

    if (nrem > 0) {
        allocate(damping4, (nks_each_thread + 1) * mympi->nprocs, ntemp);
    } else {
        allocate(damping4, nks_total, ntemp);
    }

    double **eval_tmp;
    std::complex<double> ***evec_tmp;

    const auto neval = dynamical->neval;

    kmesh_4ph = new KpointMeshUniform(nkc_tmp);
    kmesh_4ph->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector);
    auto nk_4ph = kmesh_4ph->nk;
    dymat_4ph = new DymatEigenValue(true, false, nk_4ph, neval);

    allocate(eval_tmp, nk_4ph, neval);
    allocate(evec_tmp, nk_4ph, neval, neval);

    dynamical->get_eigenvalues_dymat(nk_4ph,
                                     kmesh_4ph->xk,
                                     kmesh_4ph->kvec_na,
                                     fcs_phonon->force_constant_with_cell[0],
                                     ewald->fc2_without_dipole,
                                     true,
                                     eval_tmp,
                                     evec_tmp);

    if (!dynamical->get_projection_directions().empty()) {
        if (mympi->my_rank == 0) {
            for (auto ik = 0; ik < nk_4ph; ++ik) {
                dynamical->project_degenerate_eigenvectors(system->get_primcell().lattice_vector,
                                                           fcs_phonon->force_constant_with_cell[0],
                                                           kmesh_4ph->xk[ik],
                                                           dynamical->get_projection_directions(),
                                                           evec_tmp[ik]);
            }
        }

        MPI_Bcast(&evec_tmp[0][0][0], nk_4ph * neval * neval, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    }

    dymat_4ph->set_eigenvals_and_eigenvecs(nk_4ph, eval_tmp, evec_tmp);
    deallocate(eval_tmp);
    deallocate(evec_tmp);

    phase_storage_4ph = new PhaseFactorStorage(kmesh_4ph->nk_i);
    phase_storage_4ph->create(true);

    if (mympi->my_rank == 0) {
        allocate(vel_4ph, kmesh_4ph->nk, ns, 3);
    } else {
        allocate(vel_4ph, 1, 1, 1);
    }
    phonon_velocity->get_phonon_group_velocity_mesh_mpi(*kmesh_4ph, system->get_primcell().lattice_vector, vel_4ph);
    const auto factor = Bohr_in_Angstrom * 1.0e-10 / time_ry;

    if (mympi->my_rank == 0) {
        for (auto i = 0; i < kmesh_4ph->nk; ++i) {
            for (auto j = 0; j < ns; ++j) {
                for (auto k = 0; k < 3; ++k)
                    vel_4ph[i][j][k] *= factor;
            }
        }
    }

    vks_job4.clear();
    for (auto i = 0; i < kmesh_4ph->nk_irred; ++i) {
        for (auto j = 0; j < ns; ++j) {
            vks_job4.insert(i * ns + j);
        }
    }

    if (!integration->adaptive_sigma4) {
        integration->adaptive_sigma4 = new AdaptiveSmearingSigma(kmesh_4ph->nk, ns, integration->adaptive_factor);
        integration->adaptive_sigma4->setup(phonon_velocity,
                                            kmesh_4ph,
                                            system->get_primcell().lattice_vector,
                                            system->get_primcell().reciprocal_lattice_vector);
    }

    // prepare IO for four phonon
    setup_result_io(-1);
    prepare_restart(-1);
}


void Conductivity::prepare_restart(const int mode)
{
    // prepare restart for either 3ph or 4ph
    int i;
    int nks_done = 0, *arr_done;

    if (mode == 1) {
        // 3ph
        nshift_restart = 0;
        vks_done.clear();
        if (mympi->my_rank == 0) {
            if (!restart_flag_3ph) {

                fs_result3 << "##Phonon Frequency\n";
                fs_result3 << "#K-point (irreducible), Branch, Omega (cm^-1)\n";
                for (i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
                    const auto ik = dos->kmesh_dos->kpoint_irred_all[i][0].knum;
                    for (auto is = 0; is < dynamical->neval; ++is) {
                        fs_result3 << std::setw(6) << i + 1 << std::setw(6) << is + 1;
                        fs_result3 << std::setw(15) << writes->in_kayser(dos->dymat_dos->get_eigenvalues()[ik][is])
                                   << '\n';
                    }
                }
                fs_result3 << "##END Phonon Frequency\n\n";
                fs_result3 << "##Phonon Relaxation Time\n";
            } else {
                load_restart_gamma_blocks(fs_result3,
                                          file_result3,
                                          dos->kmesh_dos->nk_irred,
                                          damping3,
                                          vks_done,
                                          "3-phonon");
            }

            fs_result3.clear();
            fs_result3.close();
            fs_result3.open(file_result3.c_str(), std::ios::app | std::ios::out);
            if (!fs_result3) {
                exit("prepare_restart", "Could not open 3-phonon result file for append.");
            }
        }

        if (mympi->my_rank == 0) {
            nks_done = vks_done.size();
        }
        MPI_Bcast(&nks_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        nshift_restart = nks_done;

        if (nks_done > 0) {
            allocate(arr_done, nks_done);

            if (mympi->my_rank == 0) {
                for (i = 0; i < nks_done; ++i) {
                    arr_done[i] = vks_done[i];
                }
            }
            MPI_Bcast(&arr_done[0], nks_done, MPI_INT, 0, MPI_COMM_WORLD);

            // Remove vks_done elements from vks_job

            for (i = 0; i < nks_done; ++i) {

                const auto it_set = vks_job.find(arr_done[i]);

                if (it_set == vks_job.end()) {
                    std::cout << " rank = " << mympi->my_rank << " arr_done = " << arr_done[i] << '\n';
                    exit("prepare_restart", "This cannot happen");
                } else {
                    vks_job.erase(it_set);
                }
            }
            deallocate(arr_done);
        }
        vks_done.clear();

    } else if (mode == -1) {
        // 4ph
        nshift_restart4 = 0;
        vks_done4.clear();
        if (mympi->my_rank == 0) {
            if (!restart_flag_4ph) {
                fs_result4 << "##Phonon Frequency\n";
                fs_result4 << "#K-point (irreducible), Branch, Omega (cm^-1)\n";
                for (i = 0; i < kmesh_4ph->nk_irred; ++i) {
                    const int ik = kmesh_4ph->kpoint_irred_all[i][0].knum;
                    for (auto is = 0; is < dynamical->neval; ++is) {
                        fs_result4 << std::setw(6) << i + 1 << std::setw(6) << is + 1;
                        fs_result4 << std::setw(15) << writes->in_kayser(dymat_4ph->get_eigenvalues()[ik][is]) << '\n';
                    }
                }

                fs_result4 << "##END Phonon Frequency\n\n";
                fs_result4 << "##Phonon Relaxation Time\n";
            } else {
                load_restart_gamma_blocks(fs_result4,
                                          file_result4,
                                          kmesh_4ph->nk_irred,
                                          damping4,
                                          vks_done4,
                                          "4-phonon");
            }
            fs_result4.clear();
            fs_result4.close();
            fs_result4.open(file_result4.c_str(), std::ios::app | std::ios::out);
            if (!fs_result4) {
                exit("prepare_restart", "Could not open 4-phonon result file for append.");
            }
        }

        if (mympi->my_rank == 0) {
            nks_done = vks_done4.size();
        }
        MPI_Bcast(&nks_done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        nshift_restart4 = nks_done;

        if (nks_done > 0) {
            allocate(arr_done, nks_done);

            if (mympi->my_rank == 0) {
                for (i = 0; i < nks_done; ++i) {
                    arr_done[i] = vks_done4[i];
                }
            }
            MPI_Bcast(&arr_done[0], nks_done, MPI_INT, 0, MPI_COMM_WORLD);

            // Remove vks_done elements from vks_job

            for (i = 0; i < nks_done; ++i) {

                const auto it_set = vks_job4.find(arr_done[i]);

                if (it_set == vks_job4.end()) {
                    std::cout << " rank = " << mympi->my_rank << " arr_done = " << arr_done[i] << '\n';
                    exit("prepare_restart", "This cannot happen");
                } else {
                    vks_job4.erase(it_set);
                }
            }
            deallocate(arr_done);
        }
        vks_done4.clear();
    } else {
        exit("prepare_restart", "this could not happen");
    }
}


void Conductivity::load_restart_gamma_blocks(std::fstream &fs_result, const std::string &file_result,
                                             const unsigned int nk_irred, double **damping,
                                             std::vector<int> &vks_done_out, const char *label)
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
            damping_tmp[i] *= time_ry / Hz_to_kayser;
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


void Conductivity::setup_result_io(const int mode)
{
    // check consistency or write header for result, for either 3ph or 4ph calculation
    if (mympi->my_rank == 0) {

        if (mode == 1) {
            // 3ph
            if (conductivity->restart_flag_3ph) {
                std::cout << "\n";
                std::cout << " RESTART = 1 : Restart from the interrupted run.\n";
                std::cout << "               Phonon lifetimes will be load from file " << file_result3 << '\n';
                std::cout << "               and check the consistency of the computational settings.\n";

                check_consistency_restart(fs_result3,
                                          file_result3,
                                          dos->kmesh_dos->nk_i,
                                          dos->kmesh_dos->nk_irred,
                                          system->get_primcell(),
                                          thermodynamics->classical,
                                          integration->ismear,
                                          integration->epsilon,
                                          system->Tmin,
                                          system->Tmax,
                                          system->dT,
                                          fcs_phonon->file_fcs);

            } else {

                write_header_result(fs_result3,
                                    file_result3,
                                    dos->kmesh_dos,
                                    system->get_primcell(),
                                    thermodynamics->classical,
                                    integration->ismear,
                                    integration->epsilon,
                                    system->Tmin,
                                    system->Tmax,
                                    system->dT,
                                    fcs_phonon->file_fcs);
            }
        } else if (mode == -1) {

            if (conductivity->restart_flag_4ph) {
                std::cout << "\n";
                std::cout << " RESTART_4PH = 1 : Restart from the interrupted run.\n";
                std::cout << "                   Phonon lifetimes will be load from file " << file_result4 << '\n';
                std::cout << "                   and check the consistency of the computational settings.\n";

                check_consistency_restart(fs_result4,
                                          file_result4,
                                          kmesh_4ph->nk_i,
                                          kmesh_4ph->nk_irred,
                                          system->get_primcell(),
                                          thermodynamics->classical,
                                          integration->ismear,
                                          integration->epsilon,
                                          system->Tmin,
                                          system->Tmax,
                                          system->dT,
                                          fcs_phonon->file_fcs);

            } else {

                write_header_result(fs_result4,
                                    file_result4,
                                    kmesh_4ph,
                                    system->get_primcell(),
                                    thermodynamics->classical,
                                    integration->ismear,
                                    integration->epsilon,
                                    system->Tmin,
                                    system->Tmax,
                                    system->dT,
                                    fcs_phonon->file_fcs);
            }
        } else {
            exit("set_up_result_io", "this could not happen");
        }
    }
}


void Conductivity::calc_anharmonic_imagself3()
{
    unsigned int i;
    unsigned int *nks_thread = nullptr;
    double *damping3_loc = nullptr;

    // Distribute (k,s) to individual MPI threads

    const auto nks_g = vks_job.size();
    vks_l.clear();

    unsigned int icount = 0;

    for (const auto &it: vks_job) {
        if (icount % mympi->nprocs == mympi->my_rank) {
            vks_l.push_back(it);
        }
        ++icount;
    }

    if (mympi->my_rank == 0) {
        allocate(nks_thread, mympi->nprocs);
    }

    auto nks_tmp = vks_l.size();
    // Only root's recvbuf is significant; pass nks_thread directly (it is the
    // start of the buffer on root and nullptr on other ranks) to avoid forming
    // &nks_thread[my_rank] from a null pointer on non-root ranks.
    MPI_Gather(&nks_tmp, 1, MPI_UNSIGNED, nks_thread, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " Start computing 3-phonon (bubble) self-energies ... \n";
        std::cout << " Total Number of phonon modes to be calculated : " << nks_g << '\n';
        std::cout << " They are distributed to " << std::setw(6) << mympi->nprocs << " MPI processes\n";
        std::cout << '\n' << std::flush;
        deallocate(nks_thread);
    }

    unsigned int nk_tmp;

    if (nks_g % mympi->nprocs != 0) {
        nk_tmp = nks_g / mympi->nprocs + 1;
    } else {
        nk_tmp = nks_g / mympi->nprocs;
    }

    if (vks_l.size() < nk_tmp) {
        vks_l.push_back(-1);
    }

    allocate(damping3_loc, ntemp);

    auto startTime = std::chrono::system_clock::now();
    auto lastUpdate = startTime;
    bool isConsole = isOutputToConsole();

    for (i = 0; i < nk_tmp; ++i) {

        const auto iks = vks_l[i];

        if (iks == -1) {

            for (unsigned int j = 0; j < ntemp; ++j)
                damping3_loc[j] = eps; // do nothing

        } else {

            const auto knum = dos->kmesh_dos->kpoint_irred_all[iks / ns][0].knum;
            const auto snum = iks % ns;

            const auto omega = dos->dymat_dos->get_eigenvalues()[knum][snum];

            if (integration->ismear >= 0) {
                anharmonic_core->calc_damping_smearing(ntemp,
                                                       temperature,
                                                       omega,
                                                       iks / ns,
                                                       snum,
                                                       dos->kmesh_dos,
                                                       dos->dymat_dos->get_eigenvalues(),
                                                       dos->dymat_dos->get_eigenvectors(),
                                                       damping3_loc);
            } else if (integration->ismear == -1) {
                anharmonic_core->calc_damping_tetrahedron(ntemp,
                                                          temperature,
                                                          omega,
                                                          iks / ns,
                                                          snum,
                                                          dos->kmesh_dos,
                                                          dos->dymat_dos->get_eigenvalues(),
                                                          dos->dymat_dos->get_eigenvectors(),
                                                          damping3_loc);
            }
        }

        MPI_Gather(&damping3_loc[0],
                   ntemp,
                   MPI_DOUBLE,
                   damping3[nshift_restart + i * mympi->nprocs],
                   ntemp,
                   MPI_DOUBLE,
                   0,
                   MPI_COMM_WORLD);

        if (mympi->my_rank == 0) {
            write_result_gamma(i, nshift_restart, vel, damping3, 1);

            auto currentTime = std::chrono::system_clock::now();
            long long totalElapsedTime =
                std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
            long long avgTimePerStep = (i == 0) ? 0 : totalElapsedTime / i;
            long long timeRemaining = (i == 0) ? 0 : avgTimePerStep * (nks_tmp - i - 1);
            displayProgressBar(i, nks_tmp - 1, std::cout, timeRemaining, isConsole, "3-phonon");
            lastUpdate = currentTime;
            if (i == nk_tmp - 1) std::cout << "\n done. \n\n" << std::flush;
        }
    }
    deallocate(damping3_loc);
}


void Conductivity::calc_anharmonic_imagself4()
{
    unsigned int i;
    unsigned int *nks_thread = nullptr;

    // Distribute (k,s) to individual MPI threads

    const auto nks_g = vks_job4.size();
    vks_l.clear();

    unsigned int icount = 0;

    for (const auto &it: vks_job4) {
        if (icount % mympi->nprocs == mympi->my_rank) {
            vks_l.push_back(it);
        }
        ++icount;
    }

    if (mympi->my_rank == 0) {
        allocate(nks_thread, mympi->nprocs);
    }

    double *damping4_loc;

    auto nks_tmp = vks_l.size();
    // Only root's recvbuf is significant; pass nks_thread directly (it is the
    // start of the buffer on root and nullptr on other ranks) to avoid forming
    // &nks_thread[my_rank] from a null pointer on non-root ranks.
    MPI_Gather(&nks_tmp, 1, MPI_UNSIGNED, nks_thread, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " Start computing 4-phonon self-energies ... \n";
        std::cout << " WARNING: This is very very expensive!! Please be patient.\n";
        std::cout << " Total Number of phonon modes to be calculated : " << nks_g << '\n';
        std::cout << " They are distributed to " << std::setw(6) << mympi->nprocs << " MPI processes\n";
        std::cout << '\n' << std::flush;
        deallocate(nks_thread);
    }

    unsigned int nk_tmp;

    if (nks_g % mympi->nprocs != 0) {
        nk_tmp = nks_g / mympi->nprocs + 1;
    } else {
        nk_tmp = nks_g / mympi->nprocs;
    }

    if (vks_l.size() < nk_tmp) {
        vks_l.push_back(-1);
    }

    allocate(damping4_loc, ntemp);

    auto startTime = std::chrono::system_clock::now();
    auto lastUpdate = startTime;
    bool isConsole = isOutputToConsole();

    for (i = 0; i < nk_tmp; ++i) {

        const auto iks = vks_l[i];

        if (iks == -1) {

            for (unsigned int j = 0; j < ntemp; ++j)
                damping4_loc[j] = eps; // do nothing

        } else {

            const auto knum = kmesh_4ph->kpoint_irred_all[iks / ns][0].knum;
            const auto snum = iks % ns;

            const auto omega = dymat_4ph->get_eigenvalues()[knum][snum];

            if (integration->ismear_4ph == 0 || integration->ismear_4ph == 1 || integration->ismear_4ph == 2) {
                anharmonic_core->calc_damping4_smearing_batch(ntemp,
                                                              temperature,
                                                              omega,
                                                              iks / ns,
                                                              snum,
                                                              kmesh_4ph,
                                                              dymat_4ph->get_eigenvalues(),
                                                              dymat_4ph->get_eigenvectors(),
                                                              phase_storage_4ph,
                                                              damping4_loc);

            } else if (integration->ismear_4ph == -1) {
                // TODO: Implement tetrahedron method for 4ph scattering
                //                anharmonic_core->calc_damping_tetrahedron(ntemp,
                //                                                          Temperature,
                //                                                          omega,
                //                                                          iks / ns,
                //                                                          snum,
                //                                                          damping4_loc);
            }
        }

        MPI_Gather(&damping4_loc[0],
                   ntemp,
                   MPI_DOUBLE,
                   damping4[nshift_restart4 + i * mympi->nprocs],
                   ntemp,
                   MPI_DOUBLE,
                   0,
                   MPI_COMM_WORLD);

        if (mympi->my_rank == 0) {
            write_result_gamma(i, nshift_restart4, vel_4ph, damping4, -1);

            auto currentTime = std::chrono::system_clock::now();
            long long totalElapsedTime =
                std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
            long long avgTimePerStep = (i == 0) ? 0 : totalElapsedTime / i;
            long long timeRemaining = (i == 0) ? 0 : avgTimePerStep * (nks_tmp - i - 1);
            displayProgressBar(i, nks_tmp - 1, std::cout, timeRemaining, isConsole, "4-phonon");
            lastUpdate = currentTime;
            if (i == nk_tmp - 1) std::cout << "\n done. \n\n" << std::flush;
        }
    }
    deallocate(damping4_loc);
}


void Conductivity::calc_anharmonic_imagself()
{
    calc_anharmonic_imagself3();
    if (fph_rta > 0) {
        calc_anharmonic_imagself4();
    }
}


void Conductivity::write_result_gamma(const unsigned int ik, const unsigned int nshift, double ***vel_in,
                                      double **damp_in, int mode)
{
    const unsigned int np = mympi->nprocs;
    unsigned int k;

    if (mode == 1) {
        // damping 3
        std::ostringstream result_block;
        for (unsigned int j = 0; j < np; ++j) {

            const auto iks_g = ik * np + j + nshift;

            if (iks_g >= dos->kmesh_dos->nk_irred * ns) break;

            result_block << "#GAMMA_EACH\n";
            result_block << iks_g / ns + 1 << " " << iks_g % ns + 1 << '\n';

            const auto nk_equiv = dos->kmesh_dos->kpoint_irred_all[iks_g / ns].size();

            result_block << nk_equiv << '\n';
            for (k = 0; k < nk_equiv; ++k) {
                const auto ktmp = dos->kmesh_dos->kpoint_irred_all[iks_g / ns][k].knum;
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][0];
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][1];
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][2] << '\n';
            }

            for (k = 0; k < ntemp; ++k) {
                result_block << std::setw(15) << damp_in[iks_g][k] * Hz_to_kayser / time_ry << '\n';
            }
            result_block << "#END GAMMA_EACH\n";
        }
        fs_result3 << result_block.str();
        fs_result3.flush();
        if (!fs_result3) {
            exit("write_result_gamma", "Could not write 3-phonon restart block.");
        }

    } else if (mode == -1) {
        // damping 4
        std::ostringstream result_block;
        for (unsigned int j = 0; j < np; ++j) {

            const auto iks_g = ik * np + j + nshift;

            if (iks_g >= kmesh_4ph->nk_irred * ns) break;

            result_block << "#GAMMA_EACH\n";
            result_block << iks_g / ns + 1 << " " << iks_g % ns + 1 << '\n';

            const auto nk_equiv = kmesh_4ph->kpoint_irred_all[iks_g / ns].size();

            result_block << nk_equiv << '\n';
            for (k = 0; k < nk_equiv; ++k) {
                const auto ktmp = kmesh_4ph->kpoint_irred_all[iks_g / ns][k].knum;
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][0];
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][1];
                result_block << std::setw(15) << vel_in[ktmp][iks_g % ns][2] << '\n';
            }

            for (k = 0; k < ntemp; ++k) {
                result_block << std::setw(15) << damp_in[iks_g][k] * Hz_to_kayser / time_ry << '\n';
            }
            result_block << "#END GAMMA_EACH\n";
        }
        fs_result4 << result_block.str();
        fs_result4.flush();
        if (!fs_result4) {
            exit("write_result_gamma", "Could not write 4-phonon restart block.");
        }
    }
}


void Conductivity::compute_kappa()
{
    unsigned int i;
    unsigned int iks;

    if (mympi->my_rank == 0) {

        std::string file_kl;
        std::ofstream ofs_kl;

        double **lifetime;
        double **gamma_total;

        allocate(lifetime, dos->kmesh_dos->nk_irred * ns, ntemp);
        allocate(gamma_total, dos->kmesh_dos->nk_irred * ns, ntemp);

        average_self_energy_at_degenerate_point(ntemp, dos->kmesh_dos, dos->dymat_dos->get_eigenvalues(), damping3);

        for (iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {
            for (i = 0; i < ntemp; ++i) {
                gamma_total[iks][i] = damping3[iks][i];
            }
        }

        double vel_norm;
        if (len_boundary > eps) {
            for (iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {
                vel_norm = 0.0;
                auto knum = dos->kmesh_dos->kpoint_irred_all[iks / ns][0].knum;
                auto snum = iks % ns;
                for (auto j = 0; j < 3; ++j) {
                    vel_norm += vel[knum][snum][j] * vel[knum][snum][j];
                }
                vel_norm = std::sqrt(vel_norm);

                for (i = 0; i < ntemp; ++i) {
                    gamma_total[iks][i] += (vel_norm / len_boundary) * time_ry; // same unit as gamma
                }
            }
        }

        if (isotope->include_isotope) {
            for (iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {
                const auto snum = iks % ns;
                const auto gamma_iso = isotope->gamma_isotope[iks / ns][snum];
                for (i = 0; i < ntemp; ++i) {
                    gamma_total[iks][i] += gamma_iso;
                }
            }
        }

        average_self_energy_at_degenerate_point(ntemp, dos->kmesh_dos, dos->dymat_dos->get_eigenvalues(), gamma_total);

        // kappa_spec must be allocated before the FPH_RTA block below, because
        // compute_kappa_intraband() writes into it on the intermediate 3-phonon-only
        // call when KAPPA_SPEC = 1. The final full call overwrites it cleanly.
        if (calc_kappa_spec) {
            allocate(kappa_spec, dos->n_energy, ntemp, 3);
        }

        if (fph_rta > 0) {

            // calculate kappa_3ph
            double **lifetime_3only;
            allocate(lifetime_3only, dos->kmesh_dos->nk_irred * ns, ntemp);
            lifetime_from_gamma(gamma_total, lifetime_3only);

            allocate(kappa_3only, ntemp, 3, 3);
            compute_kappa_intraband(dos->kmesh_dos,
                                    dos->dymat_dos->get_eigenvalues(),
                                    lifetime_3only,
                                    kappa_3only,
                                    kappa_spec);

            deallocate(lifetime_3only);

            average_self_energy_at_degenerate_point(ntemp, kmesh_4ph, dymat_4ph->get_eigenvalues(), damping4);

            double **damping4_dense = nullptr;

            allocate(damping4_dense, dos->kmesh_dos->nk_irred * ns, ntemp);

            interpolate_data(kmesh_4ph, dos->kmesh_dos, damping4, damping4_dense);

            for (auto ik = 0; ik < dos->kmesh_dos->nk_irred; ++ik) {
                for (auto is = 0; is < ns; ++is) {
                    for (auto itemp = 0; itemp < ntemp; ++itemp) {
                        gamma_total[ik * ns + is][itemp] += damping4_dense[ik * ns + is][itemp];
                    }
                }
            }

            average_self_energy_at_degenerate_point(ntemp,
                                                    dos->kmesh_dos,
                                                    dos->dymat_dos->get_eigenvalues(),
                                                    gamma_total);
        }

        lifetime_from_gamma(gamma_total, lifetime);

        allocate(kappa, ntemp, 3, 3);

        // kappa_spec is already allocated above (before the FPH_RTA block).
        compute_kappa_intraband(dos->kmesh_dos, dos->dymat_dos->get_eigenvalues(), lifetime, kappa, kappa_spec);
        deallocate(lifetime);

        if (calc_coherent) {
            allocate(kappa_coherent, ntemp, 3, 3);
            compute_kappa_coherent(dos->kmesh_dos, dos->dymat_dos->get_eigenvalues(), gamma_total, kappa_coherent);
        }

        if (calc_coherent_block) {
            allocate(kappa_coherent_block, ntemp, 3, 3);
            compute_kappa_coherent_block(dos->kmesh_dos,
                                         dos->dymat_dos->get_eigenvalues(),
                                         gamma_total,
                                         kappa_coherent_block);
        }

        deallocate(gamma_total);
    }
}


void Conductivity::average_self_energy_at_degenerate_point(const int m, const KpointMeshUniform *kmesh_in,
                                                           const double *const *eval_in, double **damping) const
{
    int j, k, l;
    const auto nkr = kmesh_in->nk_irred;

    double *eval_tmp;
    const auto tol_omega = 1.0e-7; // Approximately equal to 0.01 cm^{-1}

    std::vector<int> degeneracy_at_k;

    allocate(eval_tmp, ns);

    double *damping_sum;

    allocate(damping_sum, m);

    for (auto i = 0; i < nkr; ++i) {
        const auto ik = kmesh_in->kpoint_irred_all[i][0].knum;

        for (j = 0; j < ns; ++j)
            eval_tmp[j] = eval_in[ik][j];

        degeneracy_at_k.clear();

        auto omega_prev = eval_tmp[0];
        auto ideg = 1;

        for (j = 1; j < ns; ++j) {
            const auto omega_now = eval_tmp[j];

            if (std::abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_at_k.push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_at_k.push_back(ideg);

        int is = 0;
        for (j = 0; j < degeneracy_at_k.size(); ++j) {
            ideg = degeneracy_at_k[j];

            if (ideg > 1) {

                for (l = 0; l < m; ++l)
                    damping_sum[l] = 0.0;

                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < m; ++l) {
                        damping_sum[l] += damping[ns * i + k][l];
                    }
                }

                for (k = is; k < is + ideg; ++k) {
                    for (l = 0; l < m; ++l) {
                        damping[ns * i + k][l] = damping_sum[l] / static_cast<double>(ideg);
                    }
                }
            }

            is += ideg;
        }
    }
    deallocate(damping_sum);
}

void Conductivity::compute_kappa_intraband(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                           const double *const *lifetime, double ***kappa_intra,
                                           double ***kappa_spec_out) const
{
    int i, is, ik;
    double ****kappa_mode;
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->get_primcell().volume);

    const auto nk_irred = kmesh_in->nk_irred;
    allocate(kappa_mode, ntemp, 9, ns, nk_irred);

    for (i = 0; i < ntemp; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {

                if (temperature[i] < eps) {
                    // Set kappa as zero when T = 0.
                    for (is = 0; is < ns; ++is) {
                        for (ik = 0; ik < nk_irred; ++ik) {
                            kappa_mode[i][3 * j + k][is][ik] = 0.0;
                        }
                    }
                } else {
                    for (is = 0; is < ns; ++is) {
                        for (ik = 0; ik < nk_irred; ++ik) {
                            const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            const auto omega = eval_in[knum][is];
                            auto vv_tmp = 0.0;
                            const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();

                            // Accumulate group velocity (diad product) for the reducible k points
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                vv_tmp += vel[ktmp][is][j] * vel[ktmp][is][k];
                            }

                            if (thermodynamics->classical) {
                                kappa_mode[i][3 * j + k][is][ik] = thermodynamics->Cv_classical(omega, temperature[i]) *
                                                                   vv_tmp * lifetime[ns * ik + is][i];
                            } else {
                                kappa_mode[i][3 * j + k][is][ik] =
                                    thermodynamics->Cv(omega, temperature[i]) * vv_tmp * lifetime[ns * ik + is][i];
                            }

                            // Convert to SI unit
                            kappa_mode[i][3 * j + k][is][ik] *= factor_toSI;
                        }
                    }
                }

                kappa_intra[i][j][k] = 0.0;

                for (is = 0; is < ns; ++is) {
                    for (ik = 0; ik < nk_irred; ++ik) {
                        kappa_intra[i][j][k] += kappa_mode[i][3 * j + k][is][ik];
                    }
                }

                kappa_intra[i][j][k] /= static_cast<double>(nk_3ph);
            }
        }
    }

    if (calc_kappa_spec) {
        //allocate(kappa_spec_out, dos->n_energy, ntemp, 3);
        compute_frequency_resolved_kappa(ntemp,
                                         integration->ismear,
                                         dos->kmesh_dos,
                                         dos->dymat_dos->get_eigenvalues(),
                                         kappa_mode,
                                         kappa_spec_out);
    }

    deallocate(kappa_mode);
}

void Conductivity::compute_kappa_coherent(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                          const double *const *gamma_total, double ***kappa_coherent_out) const
{
    // Compute the coherent part of thermal conductivity
    // based on the Michelle's paper.
    int ib;
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->get_primcell().volume);
    const auto common_factor = factor_toSI * 1.0e+12 * time_ry / static_cast<double>(kmesh_in->nk);
    const auto common_factor_output = factor_toSI * 1.0e+12 * time_ry;
    const int ns2 = ns * ns;
    const auto czero = std::complex<double>(0.0, 0.0);
    std::vector<std::complex<double>> kappa_tmp(ns2, czero);
    std::complex<double> **kappa_save = nullptr;

    const auto nk_irred = kmesh_in->nk_irred;

    std::ofstream ofs;
    if (calc_coherent == 2) {
        ofs.open(file_coherent_elems.c_str(), std::ios::out);
        if (!ofs) exit("compute_kappa_coherent", "cannot open file_kc");
        ofs << "# Temperature [K], 1st and 2nd xyz components, ibranch, jbranch, ik_irred, "
               "omega1 [cm^-1], omega2 [cm^-1], kappa_elems real, kappa_elems imag\n";
        allocate(kappa_save, ns2, nk_irred);
    }

    for (auto i = 0; i < ntemp; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
            for (unsigned int k = 0; k < 3; ++k) {

                kappa_coherent_out[i][j][k] = 0.0;

                if (temperature[i] > eps) {
#pragma omp parallel for
                    for (ib = 0; ib < ns2; ++ib) {
                        kappa_tmp[ib] = czero;
                        const int is = ib / ns;
                        const int js = ib % ns;

                        if (js == is) continue; // skip the diagonal component

                        for (auto ik = 0; ik < nk_irred; ++ik) {
                            const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            const auto omega1 = eval_in[knum][is];
                            const auto omega2 = eval_in[knum][js];

                            if (omega1 < eps8 || omega2 < eps8) continue;
                            auto vv_tmp = czero;
                            const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();

                            // Accumulate group velocity (diad product) for the reducible k points
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                vv_tmp += velmat[ktmp][is][js][j] * velmat[ktmp][js][is][k];
                            }
                            auto kcelem_tmp =
                                2.0 * (omega1 * omega2) / (omega1 + omega2) *
                                (thermodynamics->Cv(omega1, temperature[i]) / omega1 +
                                 thermodynamics->Cv(omega2, temperature[i]) / omega2) *
                                2.0 * (gamma_total[ik * ns + is][i] + gamma_total[ik * ns + js][i]) /
                                (4.0 * std::pow(omega1 - omega2, 2.0) +
                                 4.0 * std::pow(gamma_total[ik * ns + is][i] + gamma_total[ik * ns + js][i], 2.0)) *
                                vv_tmp;
                            kappa_tmp[ib] += kcelem_tmp;

                            if (calc_coherent == 2 && j == k) {
                                kappa_save[ib][ik] = kcelem_tmp * common_factor_output;
                            }
                        }
                    } // end OpenMP parallelization over ib

                    for (ib = 0; ib < ns2; ++ib) {
                        if (std::abs(kappa_tmp[ib].imag()) > eps10) {
                            warn("compute_kappa_coherent", "The kappa_coherent_out has imaginary component.");
                        }
                        kappa_coherent_out[i][j][k] += kappa_tmp[ib].real();
                    }

                    if (calc_coherent == 2 && j == k) {
                        for (ib = 0; ib < ns2; ++ib) {

                            const int is = ib / ns;
                            const int js = ib % ns;

                            for (auto ik = 0; ik < nk_irred; ++ik) {
                                if (is == js) kappa_save[ib][ik] = czero;

                                ofs << std::setw(5) << temperature[i];
                                ofs << std::setw(3) << j + 1 << std::setw(3) << k + 1;
                                ofs << std::setw(4) << is + 1;
                                ofs << std::setw(4) << js + 1;
                                ofs << std::setw(6) << ik + 1;
                                const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                                const auto omega1 = eval_in[knum][is];
                                const auto omega2 = eval_in[knum][js];
                                ofs << std::setw(15) << writes->in_kayser(omega1);
                                ofs << std::setw(15) << writes->in_kayser(omega2);
                                ofs << std::setw(15) << kappa_save[ib][ik].real();
                                ofs << std::setw(15) << kappa_save[ib][ik].imag();
                                ofs << '\n';
                            }
                        }
                        ofs << '\n';
                    }
                }
                kappa_coherent_out[i][j][k] *= common_factor;
            }
        }
    }

    if (calc_coherent == 2) {
        ofs.close();
        deallocate(kappa_save);
    }
}

void Conductivity::compute_kappa_coherent_block(const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                                const double *const *gamma_total, double ***kappa_coherent_out) const
{
    const auto factor_toSI = 1.0e+18 / (std::pow(Bohr_in_Angstrom, 3) * system->get_primcell().volume);
    const auto common_factor = factor_toSI * 1.0e+12 * time_ry / static_cast<double>(kmesh_in->nk);
    const auto czero = std::complex<double>(0.0, 0.0);
    const auto nk_irred = kmesh_in->nk_irred;
    const auto tol_omega = 1.0e-7;

    for (auto itemp = 0; itemp < ntemp; ++itemp) {
        for (auto mu = 0; mu < 3; ++mu) {
            for (auto nu = 0; nu < 3; ++nu) {
                kappa_coherent_out[itemp][mu][nu] = 0.0;
            }
        }
    }

    for (auto ik = 0; ik < nk_irred; ++ik) {
        const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;

        std::vector<std::pair<int, int>> blocks;
        auto begin = 0;
        auto omega_prev = eval_in[knum][0];
        for (auto is = 1; is < ns; ++is) {
            const auto omega_now = eval_in[knum][is];
            if (std::abs(omega_now - omega_prev) >= tol_omega) {
                blocks.emplace_back(begin, is);
                begin = is;
                omega_prev = omega_now;
            }
        }
        blocks.emplace_back(begin, static_cast<int>(ns));

        for (auto itemp = 0; itemp < ntemp; ++itemp) {
            if (temperature[itemp] <= eps) continue;

            for (auto iblock = 0; iblock < blocks.size(); ++iblock) {
                const auto first_i = blocks[iblock].first;
                const auto last_i = blocks[iblock].second;
                const auto n_i = last_i - first_i;

                auto omega_i = 0.0;
                auto gamma_i = 0.0;
                for (auto is = first_i; is < last_i; ++is) {
                    omega_i += eval_in[knum][is];
                    gamma_i += gamma_total[ik * ns + is][itemp];
                }
                omega_i /= static_cast<double>(n_i);
                gamma_i /= static_cast<double>(n_i);

                if (omega_i < eps8 || !std::isfinite(omega_i) || !std::isfinite(gamma_i)) continue;

                for (auto jblock = 0; jblock < blocks.size(); ++jblock) {
                    const auto first_j = blocks[jblock].first;
                    const auto last_j = blocks[jblock].second;
                    const auto n_j = last_j - first_j;

                    if (iblock == jblock) continue;

                    auto omega_j = 0.0;
                    auto gamma_j = 0.0;
                    for (auto js = first_j; js < last_j; ++js) {
                        omega_j += eval_in[knum][js];
                        gamma_j += gamma_total[ik * ns + js][itemp];
                    }
                    omega_j /= static_cast<double>(n_j);
                    gamma_j /= static_cast<double>(n_j);

                    const auto gamma_sum = gamma_i + gamma_j;
                    const auto domega = omega_i - omega_j;
                    if (std::abs(domega) < tol_omega) continue;

                    const auto denominator = domega * domega + gamma_sum * gamma_sum;
                    if (omega_j < eps8 || denominator <= 0.0 || !std::isfinite(omega_j) || !std::isfinite(gamma_j) ||
                        !std::isfinite(denominator))
                    {
                        continue;
                    }

                    const auto population_factor = (thermodynamics->Cv(omega_i, temperature[itemp]) * omega_j +
                                                    thermodynamics->Cv(omega_j, temperature[itemp]) * omega_i) /
                                                   (omega_i + omega_j);
                    if (!std::isfinite(population_factor)) continue;

                    const auto prefactor = population_factor * gamma_sum / denominator;
                    if (!std::isfinite(prefactor)) continue;

                    for (auto mu = 0; mu < 3; ++mu) {
                        for (auto nu = 0; nu < 3; ++nu) {
                            auto trace_vv = czero;
                            const auto nk_equiv = kmesh_in->kpoint_irred_all[ik].size();
                            for (auto ieq = 0; ieq < nk_equiv; ++ieq) {
                                const auto ktmp = kmesh_in->kpoint_irred_all[ik][ieq].knum;
                                for (auto is = first_i; is < last_i; ++is) {
                                    for (auto js = first_j; js < last_j; ++js) {
                                        trace_vv += velmat[ktmp][is][js][mu] * velmat[ktmp][js][is][nu];
                                    }
                                }
                            }
                            if (!std::isfinite(trace_vv.real()) || !std::isfinite(trace_vv.imag())) continue;
                            kappa_coherent_out[itemp][mu][nu] += (prefactor * trace_vv).real();
                        }
                    }
                }
            }
        }
    }

    for (auto itemp = 0; itemp < ntemp; ++itemp) {
        for (auto mu = 0; mu < 3; ++mu) {
            for (auto nu = 0; nu < 3; ++nu) {
                kappa_coherent_out[itemp][mu][nu] *= common_factor;
            }
        }
    }
}

void Conductivity::check_velocity_matrix_consistency(const KpointMeshUniform *kmesh_in,
                                                     const double *const *eval_in) const
{
    if (mympi->my_rank != 0 || !std::getenv("ALAMODE_CHECK_VELMAT")) return;

    auto max_abs = 0.0;
    auto max_rel = 0.0;
    auto max_imag_diag = 0.0;
    auto max_hermiticity = 0.0;
    auto max_hermiticity_rel = 0.0;
    unsigned int max_k = 0;
    unsigned int max_mode = 0;
    unsigned int max_mu = 0;
    unsigned int max_herm_k = 0;
    unsigned int max_herm_i = 0;
    unsigned int max_herm_j = 0;
    unsigned int max_herm_mu = 0;

    for (auto ik = 0u; ik < kmesh_in->nk; ++ik) {
        for (auto is = 0u; is < ns; ++is) {
            if (eval_in[ik][is] < eps8) continue;

            for (auto mu = 0u; mu < 3; ++mu) {
                const auto v_diag = velmat[ik][is][is][mu];
                const auto diff = std::abs(v_diag.real() - vel[ik][is][mu]);
                const auto scale = std::max({std::abs(v_diag.real()), std::abs(vel[ik][is][mu]), 1.0});
                const auto rel = diff / scale;

                if (diff > max_abs) {
                    max_abs = diff;
                    max_k = ik;
                    max_mode = is;
                    max_mu = mu;
                }
                if (rel > max_rel) {
                    max_rel = rel;
                }

                max_imag_diag = std::max(max_imag_diag, std::abs(v_diag.imag()));
            }
        }

        for (auto is = 0u; is < ns; ++is) {
            for (auto js = is + 1; js < ns; ++js) {
                for (auto mu = 0u; mu < 3; ++mu) {
                    const auto lhs = velmat[ik][is][js][mu];
                    const auto rhs = std::conj(velmat[ik][js][is][mu]);
                    const auto diff = std::abs(lhs - rhs);
                    const auto scale = std::max({std::abs(lhs), std::abs(rhs), 1.0});
                    const auto rel = diff / scale;

                    if (diff > max_hermiticity) {
                        max_hermiticity = diff;
                        max_herm_k = ik;
                        max_herm_i = is;
                        max_herm_j = js;
                        max_herm_mu = mu;
                    }
                    if (rel > max_hermiticity_rel) {
                        max_hermiticity_rel = rel;
                    }
                }
            }
        }
    }

    const auto filename = input->job_title + ".velmat_check";
    std::ofstream ofs(filename.c_str(), std::ios::out);
    if (!ofs) exit("check_velocity_matrix_consistency", "Could not open velmat_check file");

    ofs << "# Velocity-matrix consistency diagnostic\n";
    ofs << "# max |Re[v_ii] - group_velocity|\n";
    ofs << std::scientific << std::setprecision(16) << max_abs << ' ' << max_rel << ' ' << max_k + 1 << ' '
        << max_mode + 1 << ' ' << max_mu + 1 << ' ' << writes->in_kayser(eval_in[max_k][max_mode]) << ' '
        << vel[max_k][max_mode][max_mu] << ' ' << velmat[max_k][max_mode][max_mode][max_mu].real() << ' '
        << velmat[max_k][max_mode][max_mode][max_mu].imag() << '\n';
    ofs << "# max |v_ij - conj(v_ji)|\n";
    ofs << max_hermiticity << ' ' << max_hermiticity_rel << ' ' << max_herm_k + 1 << ' ' << max_herm_i + 1 << ' '
        << max_herm_j + 1 << ' ' << max_herm_mu + 1 << ' ' << writes->in_kayser(eval_in[max_herm_k][max_herm_i]) << ' '
        << writes->in_kayser(eval_in[max_herm_k][max_herm_j]) << ' '
        << velmat[max_herm_k][max_herm_i][max_herm_j][max_herm_mu].real() << ' '
        << velmat[max_herm_k][max_herm_i][max_herm_j][max_herm_mu].imag() << ' '
        << velmat[max_herm_k][max_herm_j][max_herm_i][max_herm_mu].real() << ' '
        << velmat[max_herm_k][max_herm_j][max_herm_i][max_herm_mu].imag() << '\n';
    ofs.close();

    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    std::cout << " Velocity-matrix diagnostic (ALAMODE_CHECK_VELMAT=1):\n"
              << "   max |Re[v_ii] - group_velocity| = " << std::scientific << max_abs << " at k=" << max_k + 1
              << ", mode=" << max_mode + 1 << ", component=" << max_mu + 1 << '\n'
              << "   max relative difference          = " << max_rel << '\n'
              << "   max |Im[v_ii]|                   = " << max_imag_diag << '\n'
              << "   max |v_ij - conj(v_ji)|          = " << max_hermiticity << '\n'
              << "   details are stored in the file " << filename << '\n';
    std::cout.flags(flags);
    std::cout.precision(precision);
}

void Conductivity::compute_frequency_resolved_kappa(const int ntemp, const int smearing_method,
                                                    const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                                    const double *const *const *const *kappa_mode,
                                                    double ***kappa_spec_out) const
{
    int i, j;
    unsigned int *kmap_identity;
    double **eval;

    std::cout << '\n';
    std::cout << " KAPPA_SPEC = 1 : Calculating thermal conductivity spectra ... ";

    allocate(kmap_identity, nk_3ph);
    allocate(eval, ns, nk_3ph);

    for (i = 0; i < nk_3ph; ++i)
        kmap_identity[i] = i;

    for (i = 0; i < nk_3ph; ++i) {
        for (j = 0; j < ns; ++j) {
            eval[j][i] = writes->in_kayser(eval_in[i][j]);
        }
    }

#ifdef _OPENMP
#pragma omp parallel private(j)
#endif
    {
        int k;
        int knum;
        double *weight;
        allocate(weight, nk_3ph);

#ifdef _OPENMP
#pragma omp for
#endif
        for (i = 0; i < dos->n_energy; ++i) {

            for (j = 0; j < ntemp; ++j) {
                for (k = 0; k < 3; ++k) {
                    kappa_spec_out[i][j][k] = 0.0;
                }
            }

            for (int is = 0; is < ns; ++is) {
                if (smearing_method == -1) {
                    integration->calc_weight_tetrahedron(nk_3ph,
                                                         kmap_identity,
                                                         eval[is],
                                                         dos->energy_dos[i],
                                                         dos->tetra_nodes_dos->get_ntetra(),
                                                         dos->tetra_nodes_dos->get_tetras(),
                                                         weight);
                } else {
                    integration->calc_weight_smearing(nk_3ph,
                                                      nk_3ph,
                                                      kmap_identity,
                                                      eval[is],
                                                      dos->energy_dos[i],
                                                      smearing_method,
                                                      weight);
                }

                for (j = 0; j < ntemp; ++j) {
                    for (k = 0; k < 3; ++k) {
                        for (int ik = 0; ik < kmesh_in->nk_irred; ++ik) {
                            knum = kmesh_in->kpoint_irred_all[ik][0].knum;
                            kappa_spec_out[i][j][k] += kappa_mode[j][3 * k + k][is][ik] * weight[knum];
                        }
                    }
                }
            }
        }
        deallocate(weight);
    }

    deallocate(kmap_identity);
    deallocate(eval);

    std::cout << " done!\n";
}

void Conductivity::set_kmesh_coarse(const unsigned int *nk_in)
{
    for (auto i = 0; i < 3; ++i)
        nk_coarse[i] = nk_in[i];
}

KpointMeshUniform *Conductivity::get_kmesh_coarse() const
{
    return kmesh_4ph;
}

void Conductivity::set_conductivity_params(const std::string &file_result3_in, const std::string &file_result4_in,
                                           const bool restart_3ph_in, const bool restart_4ph_in)
{
    file_result3 = file_result3_in;
    file_result4 = file_result4_in;
    restart_flag_3ph = restart_3ph_in;
    restart_flag_4ph = restart_4ph_in;
}

bool Conductivity::get_restart_conductivity(const int order) const
{
    if (order == 3) return restart_flag_3ph;
    if (order == 4) return restart_flag_4ph;

    return false;
}

std::string Conductivity::get_filename_results(const int order) const
{
    if (order == 3) return file_result3;
    if (order == 4) return file_result4;

    return "";
}

void Conductivity::check_consistency_restart(std::fstream &fs_result, const std::string &file_result_in,
                                             const unsigned int nk_in[3], const unsigned int nk_irred_in,
                                             const Cell &primcell, const bool classical_in, const int ismear_in,
                                             const double epsilon_in, const double tmin_in, const double tmax_in,
                                             const double delta_t_in, const std::string &file_fcs_in)
{
    const auto Ry_to_kayser = Hz_to_kayser / time_ry;

    fs_result.open(file_result_in.c_str(), std::ios::in | std::ios::out);
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

void Conductivity::write_header_result(std::fstream &fs_result, const std::string &file_result,
                                       const KpointMeshUniform *kmesh_in, const Cell &primcell, const bool classical_in,
                                       const int ismear_in, const double epsilon_in, const double tmin_in,
                                       const double tmax_in, const double delta_t_in, const std::string &file_fcs_in)
{
    const auto Ry_to_kayser = Hz_to_kayser / time_ry;

    fs_result.open(file_result.c_str(), std::ios::out);
    if (!fs_result) {
        exit("setup_result_io", "Could not open file_result3");
    }

    fs_result << "## General information\n";
    fs_result << "#SYSTEM\n";
    fs_result << primcell.number_of_atoms << " " << primcell.number_of_elems << '\n';
    fs_result << primcell.volume << '\n';
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

void Conductivity::interpolate_data(const KpointMeshUniform *kmesh_coarse_in, const KpointMeshUniform *kmesh_dense_in,
                                    const double *const *val_coarse_in, double **val_dense_out) const
{
    double ***damping4_coarse = nullptr;
    double ***damping4_interpolated = nullptr;
    allocate(damping4_interpolated, ns, ntemp, kmesh_dense_in->nk);
    allocate(damping4_coarse, ns, ntemp, kmesh_coarse_in->nk);

    auto interpol = new TriLinearInterpolator(kmesh_coarse_in->nk_i, kmesh_dense_in->nk_i);
    interpol->setup();

    if (interpolator == "linear") {

        for (auto ik = 0; ik < kmesh_coarse_in->nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                for (auto itemp = 0; itemp < ntemp; ++itemp) {
                    damping4_coarse[is][itemp][ik] =
                        val_coarse_in[kmesh_coarse_in->kmap_to_irreducible[ik] * ns + is][itemp];
                }
            }
        }

        for (auto is = 0; is < ns; ++is) {
            for (auto itemp = 0; itemp < ntemp; ++itemp) {
                interpol->interpolate(damping4_coarse[is][itemp], damping4_interpolated[is][itemp]);
            }
        }

        for (auto ik = 0; ik < kmesh_dense_in->nk_irred; ++ik) {
            auto knum = kmesh_dense_in->kpoint_irred_all[ik][0].knum;
            for (auto is = 0; is < ns; ++is) {
                for (auto itemp = 0; itemp < ntemp; ++itemp) {
                    val_dense_out[ik * ns + is][itemp] = damping4_interpolated[is][itemp][knum];
                }
            }
        }

    } else if (interpolator == "log-linear" || interpolator == "modified-log-linear") {

        double val_tmp;
        for (auto ik = 0; ik < kmesh_coarse_in->nk; ++ik) {
            for (auto is = 0; is < ns; ++is) {
                for (auto itemp = 0; itemp < ntemp; ++itemp) {
                    val_tmp = val_coarse_in[kmesh_coarse_in->kmap_to_irreducible[ik] * ns + is][itemp];
                    if (val_tmp < eps) val_tmp = eps; // TODO: reconsider appropriate cutoff value here.
                    damping4_coarse[is][itemp][ik] = std::log(val_tmp);
                }
            }
        }

        for (auto is = 0; is < ns; ++is) {
            for (auto itemp = 0; itemp < ntemp; ++itemp) {
                if (interpolator == "modified-log-linear") {
                    interpol->interpolate_avoidgamma(damping4_coarse[is][itemp], damping4_interpolated[is][itemp], is);
                } else {
                    interpol->interpolate(damping4_coarse[is][itemp], damping4_interpolated[is][itemp]);
                }
            }
        }

        for (auto ik = 0; ik < kmesh_dense_in->nk_irred; ++ik) {
            auto knum = kmesh_dense_in->kpoint_irred_all[ik][0].knum;
            for (auto is = 0; is < ns; ++is) {
                for (auto itemp = 0; itemp < ntemp; ++itemp) {
                    val_dense_out[ik * ns + is][itemp] = std::exp(damping4_interpolated[is][itemp][knum]);
                }
            }
        }
    }

    if (write_interpolation > 0) {

        auto file_interpolate = input->job_title + ".interpolated_gamma";

        std::ofstream ofs_itp;

        ofs_itp.open(file_interpolate.c_str(), std::ios::out);

        if (!ofs_itp) exit("interpolation", "Could not open file_interpolate");

        ofs_itp << "# Result of interpolated gamma.\n";
        ofs_itp << "# Frequency (cm^-1), Gamma at each temperature \n";

        for (auto is = 0; is < ns; ++is) {
            for (auto ik = 0; ik < kmesh_dense_in->nk_irred; ++ik) {
                auto knum = kmesh_dense_in->kpoint_irred_all[ik][0].knum;
                ofs_itp << std::setw(10) << std::setprecision(2)
                        << writes->in_kayser(dos->dymat_dos->get_eigenvalues()[knum][is]);

                for (auto itemp = 0; itemp < ntemp; ++itemp) {
                    ofs_itp << std::setw(15) << std::scientific << std::setprecision(4)
                            << val_dense_out[ik * ns + is][itemp] * Hz_to_kayser / time_ry;
                }

                ofs_itp << '\n';
            }
        }

        ofs_itp.close();
    }

    deallocate(damping4_coarse);
    deallocate(damping4_interpolated);
    delete interpol;
}

void Conductivity::lifetime_from_gamma(double **&gamma, double **&lifetime)
{
    unsigned int i;
    double damp_tmp;

    for (unsigned int iks = 0; iks < dos->kmesh_dos->nk_irred * ns; ++iks) {

        if (dynamical->is_imaginary[iks / ns][iks % ns]) {
            for (i = 0; i < ntemp; ++i) {
                lifetime[iks][i] = 0.0;
                gamma[iks][i] = 1.0e+100; // very big number
            }
        } else {
            for (i = 0; i < ntemp; ++i) {
                damp_tmp = gamma[iks][i];
                if (damp_tmp > 1.0e-100) {
                    lifetime[iks][i] = 1.0e+12 * time_ry * 0.5 / damp_tmp;
                } else {
                    lifetime[iks][i] = 0.0;
                    gamma[iks][i] = 1.0e+100;
                }
            }
        }
    }
}
