"""Tests for silt.tensor: construction, properties, reshape, and the
CPU/GPU device switch."""
import numpy as np
import pytest

import silt
from conftest import run_python


def test_construct_defaults_to_cpu_host():
    t = silt.tensor(silt.float32, silt.shape(4))
    assert t.host == silt.cpu


def test_construct_with_explicit_host():
    t = silt.tensor(silt.float32, silt.shape(4), silt.cpu)
    assert t.host == silt.cpu


def test_properties_agree_with_shape():
    t = silt.tensor(silt.float32, silt.shape(4, 4))
    assert t.type == silt.float32
    assert t.elem == 16
    assert t.size == 16 * 4  # bytes: element count * sizeof(float)
    assert t.shape.elem == 16


def test_reshape_preserves_data():
    t = silt.tensor.from_numpy(np.arange(12, dtype=np.float32))
    t.reshape(3, 4)
    assert t.shape.elem == 12
    np.testing.assert_array_equal(t.numpy().reshape(-1), np.arange(12, dtype=np.float32))


def test_flatten_collapses_to_one_dimension():
    t = silt.tensor(silt.float32, silt.shape(3, 4))
    t.flatten()
    assert t.shape.dim == 1
    assert t.shape.elem == 12


def test_reshape_rejects_mismatched_element_count():
    t = silt.tensor(silt.float32, silt.shape(3, 4))
    with pytest.raises(Exception):
        t.reshape(5, 5)


def test_zero_element_tensor_constructs_without_raising():
    t = silt.tensor(silt.float32, silt.shape(0))
    assert t.elem == 0
    assert t.size == 0


def test_zero_element_tensor_numpy_round_trip():
    t = silt.tensor(silt.float32, silt.shape(0))
    arr = t.numpy()
    assert arr.size == 0


def test_zero_element_tensor_repr_does_not_crash():
    """__repr__ reads refs(), which is only meaningful once allocate()
    has set up an owning refcount -- a zero-element tensor must have
    one (see tensor_t<T>::allocate())."""
    t = silt.tensor(silt.float32, silt.shape(0))
    assert "silt.tensor" in repr(t)


def test_zero_element_tensor_copy_to_round_trips():
    t = silt.tensor(silt.float32, silt.shape(0))
    c = t.copy_to()
    assert c.elem == 0
    assert c.host == silt.cpu


def test_default_constructed_tensor_raises_cleanly_rather_than_crashing():
    """`silt.tensor()` (the bare default constructor) leaves its
    polymorphic `impl` pointer null, and every accessor dereferences it
    unconditionally via `type()` -- so touching any attribute of a
    default-constructed tensor is currently a guaranteed crash rather
    than a clean error.

    Run in a subprocess: touching an attribute of a default-constructed
    tensor is currently expected to crash the interpreter outright, and
    that must not take down the rest of the suite. This documents the
    desired end state -- a clean exception -- which the current code
    does not provide.
    """
    result = run_python(
        """
        import silt
        t = silt.tensor()
        try:
            _ = t.type
        except Exception:
            print("OK")
        else:
            print("NO EXCEPTION RAISED")
        """
    )
    assert not result.crashed, (
        f"silt.tensor() followed by an attribute access crashed the "
        f"interpreter; it should raise a clean Python exception "
        f"instead.\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"silt.tensor() followed by an attribute access did not raise.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
