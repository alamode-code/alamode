import numpy as np
import pytest

from strainkit import elasticfit as ef


def _model(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, 6), rng.normal(0, 1.0, 21), rng.normal(0, 5.0, 56)


def _data(kind, s0, c2, c3, volume, use_e, use_s, smag=0.01, nmag=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in ef.strain_points(ef.direction_set(kind), smag, nmag):
        e, s = ef.evaluate_model(s0, c2, c3, p.eta6, volume, e0=-100.0)
        if noise:
            e += noise * volume * rng.normal()
            s = s + noise * rng.normal(size=6)
        out.append(
            ef.FitData(p.label, p.eta6, e if use_e else None, s if use_s else None)
        )
    return out


@pytest.mark.parametrize(
    "kind,mode,expected_rank",
    [
        ("minimal", "stress", 83),
        ("minimal", "both", 84),
        ("full", "energy", 84),
        ("full", "both", 84),
    ],
)
def test_exact_recovery(kind, mode, expected_rank):
    s0, c2, c3 = _model(1)
    V = 45.0
    fit = ef.fit_elastic(
        _data(kind, s0, c2, c3, V, mode != "stress", mode != "energy"), V, mode
    )
    assert fit.rank == expected_rank == fit.expected_rank
    assert np.abs(fit.sigma0 - s0).max() < 1e-9
    assert np.abs(fit.c2 - c2).max() < 1e-9
    assert np.abs(fit.c3 - c3).max() < 1e-8
    assert abs(fit.de0) < 1e-9


def test_energy_minimal_rank_deficient():
    s0, c2, c3 = _model(2)
    fit = ef.fit_elastic(
        _data("minimal", s0, c2, c3, 30.0, True, False), 30.0, "energy"
    )
    assert fit.rank < fit.expected_rank


def test_noise_robustness():
    s0, c2, c3 = _model(3)
    V = 40.0
    fit = ef.fit_elastic(
        _data("minimal", s0, c2, c3, V, True, True, noise=1e-6), V, "both"
    )
    assert np.abs(fit.c2 - c2).max() * ef.EV_PER_ANG3_TO_GPA < 0.2


def test_second_pk_round_trip():
    rng = np.random.default_rng(4)
    F = np.eye(3) + 0.02 * rng.normal(size=(3, 3))
    sigma = rng.normal(size=(3, 3))
    sigma = 0.5 * (sigma + sigma.T)
    S = ef.second_pk_from_cauchy(sigma, F)
    assert np.allclose(ef.cauchy_from_second_pk(S, F), sigma)
    assert np.allclose(S, S.T)


def test_direction_set_sizes():
    assert len(ef.direction_set("minimal")) == 21
    assert len(ef.direction_set("full")) == 56
    assert len(ef.strain_points(ef.direction_set("minimal"), 0.01, 2)) == 85
    with pytest.raises(ValueError):
        ef.direction_set("foo")


def test_both_mode_weights_are_residual_based():
    """'both' mode: each block is weighted by its own single-block residual RMS
    (the stress block has no dE0 column and must not be treated as rank deficient)."""
    s0, c2, c3 = _model(6)
    V = 40.0
    data = _data("minimal", s0, c2, c3, V, True, True, noise=1e-5, seed=3)
    fit = ef.fit_elastic(data, V, "both")
    assert fit.block_rms is not None and set(fit.block_rms) == {"energy", "stress"}
    assert all(
        0 < v < 1e-2 for v in fit.block_rms.values()
    )  # neither block got the neutral weight 1.0
    assert fit.rank == 84
    assert np.abs(fit.c2 - c2).max() * ef.EV_PER_ANG3_TO_GPA < 0.5


def test_fit_requires_data_for_mode():
    s0, c2, c3 = _model(7)
    with pytest.raises(ValueError, match="no data"):
        ef.fit_elastic(_data("minimal", s0, c2, c3, 30.0, True, False), 30.0, "stress")
