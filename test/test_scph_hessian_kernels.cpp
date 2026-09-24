/*
 test_scph_hessian_kernels.cpp

 Unit test for anphon/scph_hessian_kernels.h, the kernels of the SCPH
 free-energy Hessian (BUBBLE = 4):

 - f(lambda) and df/dlambda against finite differences (ground state, finite
   temperature, classical);
 - the sign of the divided differences (all <= 0) and their continuity across
   a near degeneracy;
 - the Daleckii-Krein map against a central finite difference of the matrix
   function G(Phi) = C f(Lambda) C^dagger, for random complex Hermitian
   matrices with distinct and with exactly degenerate eigenvalues; the zero
   rows and columns of frozen modes (a fixed translational subspace);
 - restarted block GMRES for (I - K) y = b: columns converging at different
   steps (a happy breakdown in one of them), a zero right-hand side, restarts,
   a random non-Hermitian K against a direct solve, and a non-finite K;
 - the one-mode double well V = -a x^2/2 + b x^4/4 (hbar = m = 1): the static
   bubble with the quartic ladder resummed,
     d2F/du0^2 = K - (V3^2 Lambda/2) / (1 + V4 Lambda/2),  Lambda = -f'(K)/2,
   must equal the finite-difference curvature of the Gaussian (SCPH) free
   energy F(u0), at an arbitrary displacement, for the three regimes.

 Built by the anphon CMake project as `test_scph_hessian_kernels`. Exits 0 on
 success and prints the first failing check otherwise.
*/

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "scph_hessian_kernels.h"

using PHON_NS::scph_hessian::apply_dk;
using PHON_NS::scph_hessian::divided_difference;
using PHON_NS::scph_hessian::divided_difference_matrix;
using PHON_NS::scph_hessian::gmres_identity_minus;
using PHON_NS::scph_hessian::OccupationFactor;
using cplx = std::complex<double>;

namespace
{

int failures = 0;

void check(const bool ok, const std::string &what, const double got = 0.0, const double want = 0.0)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s (got %.12e, want %.12e)\n", what.c_str(), got, want);
}

std::vector<OccupationFactor> regimes()
{
    return {OccupationFactor{0.0, false}, OccupationFactor{0.7, false}, OccupationFactor{0.7, true}};
}

const char *name(const OccupationFactor &occ)
{
    if (occ.classical) return "classical";
    return occ.kT > 0.0 ? "quantum, T > 0" : "quantum, T = 0";
}

void test_derivative()
{
    for (const auto &occ: regimes()) {
        for (const double lambda: {0.05, 0.9, 4.0}) {
            const double h = 1.0e-5 * lambda;
            const double fd = (occ.f(lambda + h) - occ.f(lambda - h)) / (2.0 * h);
            check(std::abs(fd - occ.dfdlambda(lambda)) < 1.0e-7 * std::abs(fd),
                  std::string("df/dlambda, ") + name(occ),
                  occ.dfdlambda(lambda),
                  fd);
            check(occ.dfdlambda(lambda) < 0.0, std::string("f decreasing, ") + name(occ));
        }
        // Across the switch to the midpoint derivative (rel_tol = 1e-6) the
        // divided difference must stay equal to f'(midpoint) to second order in
        // the gap, from both sides.
        const double a = 1.3;
        for (const double gap: {5.0e-7, 2.0e-6, 1.0e-4}) {
            const double b = a * (1.0 + gap);
            const double dd = divided_difference(occ, a, b);
            const double mid = occ.dfdlambda(0.5 * (a + b));
            check(std::abs(dd - mid) < 1.0e-8 * std::abs(mid),
                  std::string("divided difference vs midpoint derivative, ") + name(occ),
                  dd,
                  mid);
        }
    }
}

Eigen::MatrixXcd random_hermitian(std::mt19937 &rng, const int n)
{
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::MatrixXcd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) A(i, j) = cplx(g(rng), g(rng));
    }
    return 0.5 * (A + A.adjoint());
}

// G(Phi) = C f(Lambda) C^dagger for a positive-definite Hermitian Phi.
Eigen::MatrixXcd matrix_function(const OccupationFactor &occ, const Eigen::MatrixXcd &Phi)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Phi);
    Eigen::VectorXd fvals(Phi.rows());
    for (Eigen::Index s = 0; s < Phi.rows(); ++s) fvals(s) = occ.f(es.eigenvalues()(s));
    return es.eigenvectors() * fvals.cast<cplx>().asDiagonal() * es.eigenvectors().adjoint();
}

void test_daleckii_krein()
{
    std::mt19937 rng(12345);
    const int n = 6;
    for (const auto &occ: regimes()) {
        for (const bool degenerate: {false, true}) {
            // Phi = U diag(lambda) U^dagger with a controlled spectrum
            Eigen::HouseholderQR<Eigen::MatrixXcd> qr(random_hermitian(rng, n) +
                                                      cplx(0.0, 1.0) * random_hermitian(rng, n));
            const Eigen::MatrixXcd U = qr.householderQ();
            Eigen::VectorXd lambda(n);
            lambda << 0.3, 0.8, 1.1, 1.7, 2.4, 3.0;
            if (degenerate) {
                lambda(2) = lambda(1);
                lambda(5) = lambda(4);
            }
            const Eigen::MatrixXcd Phi = U * lambda.cast<cplx>().asDiagonal() * U.adjoint();
            const Eigen::MatrixXcd dPhi = random_hermitian(rng, n);

            const double t = 1.0e-5;
            const Eigen::MatrixXcd fd =
                (matrix_function(occ, Phi + t * dPhi) - matrix_function(occ, Phi - t * dPhi)) / (2.0 * t);
            // an arbitrary orthonormal basis of a degenerate eigenspace must do
            const Eigen::MatrixXd L = divided_difference_matrix(occ, lambda);
            const Eigen::MatrixXcd dk = apply_dk(U, L, dPhi);
            const double err = (dk - fd).norm() / fd.norm();
            check(err < 1.0e-7,
                  std::string("Daleckii-Krein vs finite difference, ") + name(occ) +
                      (degenerate ? ", degenerate" : ", distinct"),
                  err,
                  0.0);
            check((L.array() <= 0.0).all(), std::string("L <= 0, ") + name(occ));
        }

        // frozen modes: zero rows and columns, and no response inside them
        Eigen::VectorXd lambda(4);
        lambda << 0.5, 1.0, 2.0, 3.0;
        const std::vector<bool> frozen{true, false, false, true};
        const Eigen::MatrixXd L = divided_difference_matrix(occ, lambda, &frozen);
        check(L.row(0).isZero() && L.col(3).isZero() && L(1, 2) != 0.0, std::string("frozen modes, ") + name(occ));
    }
}

// One-mode Gaussian theory, hbar = m = 1: width s2 = f(K)/2 at K = W^2,
// self-consistent K = <V''> = -a + 3 b (u0^2 + s2).
struct OneMode
{
    double a, b;
    OccupationFactor occ;

    double solve_K(const double u0) const
    {
        double K = std::max(2.0 * b * u0 * u0, 0.5);
        for (int it = 0; it < 100000; ++it) {
            const double K_new = -a + 3.0 * b * (u0 * u0 + 0.5 * occ.f(K));
            if (std::abs(K_new - K) < 1.0e-15 * std::abs(K)) return K_new;
            K = 0.5 * K + 0.5 * K_new;
        }
        return K;
    }

    // F(u0) = F_h(W) - K s2 / 2 + <V>, with F_h the harmonic free energy.
    double free_energy(const double u0) const
    {
        const double K = solve_K(u0);
        const double w = std::sqrt(K);
        const double s2 = 0.5 * occ.f(K);
        double Fh;
        if (occ.classical) {
            Fh = occ.kT * std::log(w / occ.kT);
        } else {
            Fh = 0.5 * w + (occ.kT > 0.0 ? occ.kT * std::log(-std::expm1(-w / occ.kT)) : 0.0);
        }
        const double V = -0.5 * a * (u0 * u0 + s2) + 0.25 * b * (std::pow(u0, 4) + 6.0 * u0 * u0 * s2 + 3.0 * s2 * s2);
        return Fh - 0.5 * K * s2 + V;
    }

    double hessian_resummed(const double u0) const
    {
        const double K = solve_K(u0);
        const double Lambda = -0.5 * occ.dfdlambda(K);
        const double V3 = 6.0 * b * u0, V4 = 6.0 * b;
        return K - 0.5 * V3 * V3 * Lambda / (1.0 + 0.5 * V4 * Lambda);
    }
};

void test_one_mode()
{
    for (const auto &occ: regimes()) {
        const OneMode model{1.0, 1.0, occ};
        for (const double u0: {0.4, 0.8, 1.2}) {
            const double h = 1.0e-3;
            const double d2F =
                (model.free_energy(u0 + h) - 2.0 * model.free_energy(u0) + model.free_energy(u0 - h)) / (h * h);
            const double H = model.hessian_resummed(u0);
            check(std::abs(d2F - H) < 1.0e-5 * std::max(1.0, std::abs(H)),
                  std::string("one-mode free-energy curvature, ") + name(occ),
                  H,
                  d2F);
        }
    }
}

void test_gmres()
{
    using Eigen::MatrixXcd;
    std::vector<double> residual;
    int napply = 0;

    // I - K = diag(1, 2); column 0 converges in one step (happy breakdown),
    // column 1 needs two.
    {
        MatrixXcd K = MatrixXcd::Zero(2, 2);
        K(1, 1) = -1.0;
        MatrixXcd B(2, 2);
        B << 1.0, 1.0, 0.0, 1.0;
        MatrixXcd Y;
        const bool ok = gmres_identity_minus(
            B,
            Y,
            [&](const MatrixXcd &in, MatrixXcd &out) { out = K * in; },
            10,
            5,
            1.0e-12,
            residual,
            napply);
        MatrixXcd want(2, 2);
        want << 1.0, 1.0, 0.0, 0.5;
        check(ok && (Y - want).norm() < 1.0e-12, "GMRES: columns converging at different steps", (Y - want).norm());
    }

    // random non-Hermitian K with spectral radius < 1; a zero column and an
    // eigenvector column; a short restart forces several cycles
    {
        std::mt19937 rng(777);
        std::normal_distribution<double> g(0.0, 1.0);
        const int n = 30;
        MatrixXcd K(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) K(i, j) = cplx(g(rng), g(rng));
        }
        Eigen::ComplexEigenSolver<MatrixXcd> es(K);
        K *= 0.8 / es.eigenvalues().cwiseAbs().maxCoeff();
        Eigen::ComplexEigenSolver<MatrixXcd> es2(K);
        MatrixXcd B(n, 4);
        for (int i = 0; i < n; ++i) B(i, 0) = cplx(g(rng), g(rng));
        B.col(1).setZero();
        B.col(2) = es2.eigenvectors().col(0);
        for (int i = 0; i < n; ++i) B(i, 3) = cplx(g(rng), g(rng));
        MatrixXcd Y;
        const bool ok = gmres_identity_minus(
            B,
            Y,
            [&](const MatrixXcd &in, MatrixXcd &out) { out = K * in; },
            4,
            400,
            1.0e-12,
            residual,
            napply);
        const MatrixXcd direct = (MatrixXcd::Identity(n, n) - K).partialPivLu().solve(B);
        const double err = (Y - direct).norm() / direct.norm();
        check(ok && err < 1.0e-9, "GMRES: random K with restarts vs direct solve", err);
        check(Y.col(1).isZero(), "GMRES: zero right-hand side");
    }

    // a non-finite operator must fail, not report convergence
    {
        MatrixXcd B = MatrixXcd::Ones(3, 1), Y;
        const bool ok = gmres_identity_minus(
            B,
            Y,
            [&](const MatrixXcd &in, MatrixXcd &out) {
                out = in;
                out(0, 0) = std::numeric_limits<double>::quiet_NaN();
            },
            5,
            3,
            1.0e-12,
            residual,
            napply);
        check(!ok, "GMRES: non-finite operator is reported as a failure");
    }
}

} // namespace

int main()
{
    test_derivative();
    test_daleckii_krein();
    test_gmres();
    test_one_mode();
    if (failures == 0) std::printf("scph_hessian_kernels: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
