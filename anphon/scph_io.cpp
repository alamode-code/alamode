/*
 scph_io.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Functions for writing SCPH outputs.
 This file currently contains SCPH-specific force-constant output routines.

 Functions included:
 - write_anharmonic_correction_fc2: Write anharmonic force constant corrections
*/

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "parsephon.h"
#include "scph_qha_common.h"
#include "system.h"

using namespace PHON_NS;

void ScphQhaCommon::write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat, const unsigned int NT,
                                                    const KpointMeshUniform *kmesh_coarse_in,
                                                    MinimumDistList ***mindist_list_in, const bool is_qha,
                                                    const int type)
{
    // Output anharmonically-renormalized IFC to file

    unsigned int i, j;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    double ***delta_fc2;
    const auto ns = dynamical->neval;
    unsigned int is, js, icell;
    unsigned int iat, jat;

    std::string file_fc2;
    std::ofstream ofs_fc2;

    if (is_qha) {
        file_fc2 = input->job_title + ".qha_dfc2";
    } else {
        if (type == 0) {
            file_fc2 = input->job_title + ".scph_dfc2";
        } else if (type == 1) {
            file_fc2 = input->job_title + ".scph+bubble(0)_dfc2";
        } else if (type == 2) {
            file_fc2 = input->job_title + ".scph+bubble(w)_dfc2";
        } else if (type == 3) {
            file_fc2 = input->job_title + ".scph+bubble(wQP)_dfc2";
        }
    }

    ofs_fc2.open(file_fc2.c_str(), std::ios::out);
    if (!ofs_fc2) exit("write_anharmonic_correction_fc2", "Cannot open file_fc2");

    const auto ncell = kmesh_coarse_in->nk_i[0] * kmesh_coarse_in->nk_i[1] * kmesh_coarse_in->nk_i[2];

    allocate(delta_fc2, ns, ns, ncell);

    ofs_fc2.precision(10);
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << system->get_primcell().lattice_vector(j, i);
        }
        ofs_fc2 << '\n';
    }
    ofs_fc2 << std::setw(5) << system->get_primcell().number_of_atoms << std::setw(5)
            << system->get_primcell().number_of_elems << '\n';
    for (i = 0; i < system->get_primcell().number_of_elems; ++i) {
        ofs_fc2 << std::setw(5) << system->symbol_kd[i];
    }
    ofs_fc2 << '\n';

    for (i = 0; i < system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << system->get_primcell().x_fractional(i, j);
        }
        ofs_fc2 << std::setw(5) << system->get_primcell().kind[i] + 1 << '\n';
    }

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + dT * static_cast<double>(iT);

        ofs_fc2 << "# Temp = " << temp << '\n';

        for (is = 0; is < ns; ++is) {
            iat = is / 3;

            for (js = 0; js < ns; ++js) {
                jat = js / 3;

                for (icell = 0; icell < ncell; ++icell) {
                    delta_fc2[is][js][icell] = delta_dymat[iT][is][js][icell].real() *
                                               std::sqrt(system->get_mass_prim()[iat] * system->get_mass_prim()[jat]);
                }
            }
        }

        for (icell = 0; icell < ncell; ++icell) {

            for (is = 0; is < ns; ++is) {
                iat = is / 3;
                const auto icrd = is % 3;

                for (js = 0; js < ns; ++js) {
                    jat = js / 3;
                    const auto jcrd = js % 3;

                    const auto nmulti = mindist_list_in[iat][jat][icell].shift.size();

                    for (auto it = mindist_list_in[iat][jat][icell].shift.cbegin();
                         it != mindist_list_in[iat][jat][icell].shift.cend();
                         ++it)
                    {

                        ofs_fc2 << std::setw(4) << (*it).sx;
                        ofs_fc2 << std::setw(4) << (*it).sy;
                        ofs_fc2 << std::setw(4) << (*it).sz;
                        ofs_fc2 << std::setw(5) << iat << std::setw(3) << icrd;
                        ofs_fc2 << std::setw(4) << jat << std::setw(3) << jcrd;
                        ofs_fc2 << std::setprecision(15) << std::setw(25)
                                << delta_fc2[is][js][icell] / static_cast<double>(nmulti) << '\n';
                    }
                }
            }
        }

        ofs_fc2 << '\n';
    }

    deallocate(delta_fc2);

    ofs_fc2.close();
    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_fc2;

    if (is_qha) {
        std::cout << " : Anharmonic corrections to the second-order IFCs (QHA)\n";
    } else {
        if (type == 0) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH)\n";
        } else if (type == 1) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(0))\n";
        } else if (type == 2) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(w))\n";
        } else if (type == 3) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(wQP))\n";
        }
    }
}
