/*
 scph_v3v4_elements.h

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <vector>

namespace PHON_NS
{
class AnharmonicCore;
class KpointMeshUniform;
class PhaseFactorStorage;
class RelativeVector;

// Transform a given set of real-space cubic IFCs into normal-mode V3
// elements on the dense k mesh. MPI-collective: every rank must call it
// with the same inputs. A free function with fully explicit inputs so that
// consumers (e.g. DerivativeIFC during QHA runs) do not depend on a live
// Scph instance.
void compute_V3_elements_for_given_IFCs(std::complex<double> ***v3_out, double **omega2_harmonic_in,
                                        int ngroup_v3_in, std::vector<double> *fcs_group_v3_in,
                                        std::vector<RelativeVector> *relvec_v3_in, double *invmass_v3_in,
                                        int **evec_index_v3_in, const std::complex<double> *const *const *evec_in,
                                        bool self_offdiag, unsigned int ns_in,
                                        const KpointMeshUniform *kmesh_coarse_in,
                                        const KpointMeshUniform *kmesh_dense_in,
                                        const PhaseFactorStorage *phase_storage_in,
                                        AnharmonicCore &anharmonic_core_in, int my_rank, int nprocs);

// Set V3 (fc_order = 3) or V4 (fc_order = 4) elements involving acoustic
// modes at the Gamma point exactly to zero.
void zerofill_elements_acoustic_at_gamma(double **omega2, std::complex<double> ***v_elems, int fc_order,
                                         unsigned int ns_in, unsigned int nk_dense_in,
                                         unsigned int nk_irred_coarse_in);
} // namespace PHON_NS
