/*
 gruneisen.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>
#include "fcs_phonon.h"
#include "ndarray.h"
#include "pointers.h"

namespace PHON_NS
{

class Gruneisen: protected Pointers
{
public:
    Gruneisen(class PHON *);

    ~Gruneisen();

    double delta_a;
    // 0: none, 1: isotropic, 2: diagonal (xx, yy, zz) generalized,
    // 3: all 6 Voigt (xx, yy, zz, yz, xz, xy) generalized parameters
    int gruneisen_mode;
    bool print_newfcs;

    // Optional &strain field (MODE = phonons): when given, NEWFCS applies this
    // anisotropic strain tensor instead of the default isotropic strain
    // (+- delta_a, kept as an undocumented legacy tag).
    bool strain_newfcs_given;
    Eigen::Matrix3d strain_newfcs;

    void setup();

    NDArray<std::complex<double>, 2> gruneisen_bs;
    NDArray<std::complex<double>, 2> gruneisen_dos;

    // Generalized Gruneisen parameters gamma_{mu nu}(k, s) for gruneisen_mode >= 2,
    // shaped [nk, ns, ncomp] with ncomp = 3 (mode 2) or 6 (mode 3, Voigt order).
    NDArray<std::complex<double>, 3> gruneisen_tensor_bs;
    NDArray<std::complex<double>, 3> gruneisen_tensor_dos;

    int number_of_strain_components() const
    {
        return gruneisen_mode == 2 ? 3 : (gruneisen_mode == 3 ? 6 : 0);
    }

    void calc_gruneisen();

    void write_new_fcsxml_all() const;

private:
    void set_default_variables();

    void deallocate_variables();

    NDArray<double, 2> xshift_s;
    std::vector<FcsArrayWithCell> delta_fc2;

    // Strain-derivative IFCs used by NEWFCS (along the isotropic direction, or
    // along strain_newfcs when the &strain field is given)
    std::vector<FcsArrayWithCell> delta_fc2_newfcs, delta_fc3_newfcs;

    // Strain-derivative IFCs per component (3 or 6 lists, see gruneisen_mode)
    std::vector<std::vector<FcsArrayWithCell>> delta_fc2_strain;

    void prepare_delta_fcs(const std::vector<FcsArrayWithCell> &fcs_in, std::vector<FcsArrayWithCell> &delta_fcs,
                           const Eigen::Matrix3d &strain_dir) const;

    void prepare_delta_fcs_strain(const std::vector<FcsArrayWithCell> &fcs_in,
                                  std::vector<std::vector<FcsArrayWithCell>> &delta_fcs_strain) const;

    void calc_gruneisen_at_kpoints(unsigned int nk, const NDArray<double, 2> &xk, const double *const *eval,
                                   const std::complex<double> *const *const *evec,
                                   NDArray<std::complex<double>, 2> &gamma_iso,
                                   NDArray<std::complex<double>, 3> &gamma_tensor) const;

    // void impose_ASR_on_harmonic_IFC(std::vector<FcsArrayWithCell> &,
    //                    int);

    //  double calc_stress_energy2(const std::vector<FcsArrayWithCell>);
    void calc_stress_energy3(std::vector<FcsArrayWithCell>, double ****);

    void print_stress_energy();
};
} // namespace PHON_NS
