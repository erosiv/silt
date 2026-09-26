"""Tests for RNG tensors (silt.seed / silt.sample_uniform /
silt.sample_normal).

Ported from the former example/rng.py smoke script, with real
assertions in place of print statements. RNG tensors are GPU-only
(curand runs on device), so this whole file requires a GPU.
"""
import pytest

import silt

pytestmark = pytest.mark.gpu


def _rng_tensor(n=4096, seed=0):
    rng = silt.tensor(silt.rng, silt.shape(n), silt.gpu)
    silt.seed(rng, seed, 0)
    return rng


def test_seed_does_not_raise():
    _rng_tensor()


def test_sample_uniform_default_range():
    rng = _rng_tensor(8192)
    sample = silt.sample_uniform(rng)
    data = sample.cpu().numpy()
    assert data.min() >= 0.0
    assert data.max() <= 1.0
    assert 0.4 < data.mean() < 0.6


def test_sample_uniform_custom_range():
    rng = _rng_tensor(8192)
    sample = silt.sample_uniform(rng, -2.0, 2.0)
    data = sample.cpu().numpy()
    assert data.min() >= -2.0
    assert data.max() <= 2.0


def test_sample_normal_default():
    rng = _rng_tensor(16384)
    sample = silt.sample_normal(rng)
    data = sample.cpu().numpy()
    assert abs(data.mean()) < 0.1
    assert 0.85 < data.std() < 1.15


def test_sample_normal_custom_params():
    rng = _rng_tensor(16384)
    sample = silt.sample_normal(rng, 5.0, 2.0)
    data = sample.cpu().numpy()
    assert abs(data.mean() - 5.0) < 0.2
    assert 1.7 < data.std() < 2.3
