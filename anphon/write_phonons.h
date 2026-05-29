/*
 write_phonons.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <fstream>
#include <string>
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{
class Writes: protected Pointers
{
public:
    Writes(class PHON *);

    ~Writes();

    void writePhononInfo();

    void printPhononEnergies() const;

    void writeGruneisen();

    void setupResultIo();

    void writeInputVars();

    void writeKappa() const;

    void writeSelfenergyIsotope() const;

    bool print_zmode;

    double in_kayser(const double) const;

    void setWriteOptions(const bool print_msd_, const bool print_xsf_, const bool print_anime_,
                         const std::string &anime_format_, const int anime_steps_,
                         const unsigned int anime_cellsize_[3], const double anime_kpoint_[3], const bool print_ucorr_,
                         const int shift_ucorr_[3], const bool print_zmode_, const bool print_eval_);

    void writePhononEnergies(const unsigned int nk_in, const double *const *const *eval_in, const bool is_qha = false,
                             const int bubble = 0) const;

    void writePhononBands(const unsigned int nk_in, const double *kaxis_in, const double *const *const *eval_in,
                          const bool is_qha = false, const int bubble = 0) const;

    void writePhononDos(double **dos_in, const bool is_qha = false, const int bubble = 0) const;

    void writeThermodynamicFunc(double *heat_capacity, double *heat_capacity_correction, double *FE_QHA,
                                double *dFE_scph, double *FE_total, double *entropy, const bool is_qha = false) const;

    void writeMSD(double **msd_in, const bool is_qha = false, const int bubble = 0) const;

    void writeDispCorrelation(double ***ucorr_in, const bool is_qha = false, const int bubble = 0) const;

    void writeDielecFunc(double ****dielec_in, const bool is_qha = false) const;

    unsigned int getVerbosity() const;

    void setVerbosity(unsigned int verbosity_in);

    bool getPrintMSD() const;

    bool getPrintUcorr() const;

    std::array<int, 3> getShiftUcorr() const;

    int nbands;

private:
    void writePhononBands() const;

    void writePhononVel() const;

    void writePhononVelAll() const;

    void writePhononDos() const;

    void writeTwoPhononDos() const;

    void writeLongitudinalProjDos() const;

    void writeScatteringPhaseSpace() const;

    void writeScatteringAmplitude() const;

    void writeNormalModeDirection() const;

    void writeNormalModeDirectionEach(const std::string &fname_axsf, const unsigned int nk_in,
                                      const std::complex<double> *const *const *evec_in) const;

    void writeNormalModeAnimation(const double[3], const unsigned int[3]) const;

    void writeEigenvectors() const;

    void writeEigenvalues() const;

    void writeEigenvectorsEach(const std::string &fname_evec, const unsigned int nk_in, const double *const *xk_in,
                               const double *const *eval_in, const std::complex<double> *const *const *evec_in) const;

    void writeEigenvaluesEach(const std::string &fname_eval, const unsigned int nk_in, const double *const *xk_in,
                              const double *const *eval_in) const;

    void printNormalmodeBorncharge() const;

#ifdef _HDF5

    void writeEigenvectorsHdf5() const;

    void writeEigenvectorsEachHdf5(const std::string &fname_evec, const unsigned int nk_in, const double *const *xk_in,
                                   const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                   const unsigned int kpmode_in) const;

    void writeEigenvaluesHdf5() const;

    void writeEigenvaluesEachHdf5(const std::string &fname_eval, const unsigned int nk_in, const double *const *xk_in,
                                  const double *const *eval_in, const unsigned int kpmode_in) const;

#endif

    void writeThermodynamicFunc() const;

    void writeMSD() const;

    void writeDispCorrelation() const;

    void writeParticipationRatio() const;

    void writeParticipationRatioEach(const std::string &fname_pr, const std::string &fname_apr,
                                     const unsigned int nk_in, const double *const *xk_in, const double *const *eval_in,
                                     const std::complex<double> *const *const *evec_in) const;

    void writeParticipationRatioMesh(const std::string &fname_pr, const std::string &fname_apr,
                                     const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                                     const std::complex<double> *const *const *evec_in) const;

    void writeDielectricFunction() const;

    double Ry_to_kayser;
    unsigned int verbosity;

    int anime_frames;

    bool print_xsf;
    bool print_msd;
    bool print_ucorr;
    bool print_anime;
    bool print_eval;

    unsigned int anime_cellsize[3];
    double anime_kpoint[3];
    int shift_ucorr[3];

    std::string anime_format;

public:
};
} // namespace PHON_NS
