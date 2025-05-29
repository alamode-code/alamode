/*
 selfenergy.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include "kpoint.h"
#include <complex>
#include <vector>
#include <string>

namespace PHON_NS
{
class Selfenergy: protected Pointers
{
public:
    Selfenergy(class PHON *phon);

    ~Selfenergy();

    void setup_selfenergy();

    void selfenergy_tadpole(const unsigned int N,
                            const double *T,
                            const double omega,
                            const unsigned int knum,
                            const unsigned int snum,
                            const KpointMeshUniform *kmesh_in,
                            const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in,
                            std::complex<double> *ret) const;

    void selfenergy_a(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_b(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_c(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_c_mod(const unsigned int N,
                          const double *T,
                          const double omega,
                          const unsigned int knum,
                          const unsigned int snum,
                          const KpointMeshUniform *kmesh_in,
                          const double *const *eval_in,
                          const std::complex<double> *const *const *evec_in,
                          std::complex<double> *ret) const;

    void selfenergy_d(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_e(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_f(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_g(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_h(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_i(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

    void selfenergy_j(const unsigned int N,
                      const double *T,
                      const double omega,
                      const unsigned int knum,
                      const unsigned int snum,
                      const KpointMeshUniform *kmesh_in,
                      const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in,
                      std::complex<double> *ret) const;

private:
    unsigned int ns;
    double epsilon;

    void mpi_reduce_complex(unsigned int,
                            std::complex<double> *,
                            std::complex<double> *) const;
};
}
