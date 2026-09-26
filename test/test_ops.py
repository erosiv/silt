"""Tests for the free-function tensor operations in the `silt` module
(set/add/multiply/divide/mix/clamp/clone/cast/min/max) and the indexed
operations (indexed_set/index_radius).

All tensors here are CPU by default (silt.tensor's two-argument
constructor defaults to CPU), so most of this file needs no GPU.
"""
import numpy as np
import pytest

import silt
from conftest import run_python


# -- elementwise ops: basic correctness --------------------------------


def test_set_scalar():
    t = silt.tensor(silt.float32, silt.shape(4))
    silt.set(t, 2.5)
    np.testing.assert_array_equal(t.numpy(), np.full(4, 2.5, dtype=np.float32))


def test_add_scalar():
    t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    silt.add(t, 10.0)
    np.testing.assert_array_equal(t.numpy(), [11.0, 12.0, 13.0])


def test_multiply_scalar():
    t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    silt.multiply(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [2.0, 4.0, 6.0])


def test_divide_scalar():
    t = silt.tensor.from_numpy(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    silt.divide(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])


def test_add_tensor_tensor():
    a = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    silt.add(a, b)
    np.testing.assert_array_equal(a.numpy(), [11.0, 22.0, 33.0])


def test_mix_interpolates():
    a = silt.tensor.from_numpy(np.array([0.0, 0.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 10.0], dtype=np.float32))
    silt.mix(a, b, 0.25)
    np.testing.assert_allclose(a.numpy(), [2.5, 2.5])


def test_clamp_float32():
    t = silt.tensor.from_numpy(np.array([-5.0, 0.5, 5.0], dtype=np.float32))
    silt.clamp(t, 0.0, 1.0)
    np.testing.assert_array_equal(t.numpy(), [0.0, 0.5, 1.0])


def test_clamp_rejects_non_float32():
    # `clamp`'s binding constrains its type parameter to
    # `std::same_as<float>`, so calling it on a float64 tensor should
    # raise rather than silently doing nothing or misinterpreting data.
    t = silt.tensor.from_numpy(np.array([-5.0, 0.5, 5.0], dtype=np.float64))
    with pytest.raises(Exception):
        silt.clamp(t, 0.0, 1.0)


# -- min / max -----------------------------------------------------------


def test_min_basic():
    t = silt.tensor.from_numpy(np.array([3.0, -2.0, 5.0], dtype=np.float32))
    assert silt.min(t) == pytest.approx(-2.0)


def test_max_of_all_negative_tensor():
    """silt::max<T> seeds its accumulator with
    std::numeric_limits<T>::min(), which for a floating-point type is
    the smallest *positive* normal value, not the most negative one, so
    max() of an all-negative tensor is wrong."""
    t = silt.tensor.from_numpy(np.array([-1.0, -5.0, -0.5, -9.0], dtype=np.float32))
    result = silt.max(t)
    assert result == pytest.approx(-0.5), (
        f"silt.max returned {result} for an all-negative tensor; expected -0.5"
    )


def test_max_of_mixed_sign_tensor():
    t = silt.tensor.from_numpy(np.array([-1.0, 5.0, -0.5], dtype=np.float32))
    assert silt.max(t) == pytest.approx(5.0)


# -- clone / cast ----------------------------------------------------------


def test_clone_preserves_host():
    """silt::clone<T> allocates its result on the GPU unconditionally,
    even for a CPU source tensor, so the binary op inside it dispatches
    to a GPU code path using the source's host (non-device) pointer.
    Cloning a CPU tensor should yield a CPU tensor with identical data.

    Run in a subprocess: the underlying bug plausibly crashes the
    interpreter (a device kernel dispatch against a host pointer) on a
    machine that does have a GPU, rather than merely returning the
    wrong host tag.
    """
    result = run_python(
        """
        import numpy as np
        import silt
        t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.5], dtype=np.float32))
        c = silt.clone(t)
        assert c.host == silt.cpu, f"clone returned host={c.host!r}"
        np.testing.assert_array_equal(c.numpy(), t.numpy())
        print("OK")
        """
    )
    assert not result.crashed, (
        f"silt.clone() of a CPU tensor crashed.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"silt.clone() did not preserve the source's host.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_cast_between_floating_point_types():
    """`silt.cast`'s inner lambda declares a local `silt::tensor tensor`
    that shadows the captured parameter of the same name and is used in
    its own initializer (`tensor.as<From>()`), so the cast currently
    reads from an object whose `impl` pointer has not yet been set.
    This compiles cleanly and is undefined behaviour -- plausibly a
    crash, plausibly silently wrong data -- so it is exercised in a
    subprocess.
    """
    result = run_python(
        """
        import numpy as np
        import silt
        src = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        dst = silt.cast(src, silt.float64)
        assert dst.type == silt.float64, f"cast produced type {dst.type!r}"
        np.testing.assert_allclose(dst.numpy(), [1.0, 2.0, 3.0])
        print("OK")
        """
    )
    assert not result.crashed, f"silt.cast crashed.\nstderr={result.stderr}"
    assert "OK" in result.stdout, (
        f"silt.cast produced wrong output.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_cast_is_a_no_op_when_types_already_match():
    src = silt.tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    dst = silt.cast(src, silt.float32)
    assert dst.type == silt.float32
    np.testing.assert_array_equal(dst.numpy(), src.numpy())


def test_cast_from_int_is_currently_unsupported():
    """`silt.cast`'s type parameters are constrained to
    std::floating_point on both sides, so an int<->float cast currently
    always raises, even though the underlying silt::cast<To, From>
    template itself is written to work for any arithmetic type. This
    documents today's limitation so that lifting it later is a
    deliberate, visible change."""
    src = silt.tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(Exception):
        silt.cast(src, silt.float32)


# -- indexed operations (GPU only) -----------------------------------------


@pytest.mark.gpu
def test_index_radius_and_indexed_set():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    t = silt.tensor(silt.float32, s, silt.gpu)
    silt.set(t, 0.0)
    silt.indexed_set(t, 1.0, idx)
    data = t.cpu().numpy()
    assert data.max() == pytest.approx(1.0)
    assert data.min() == pytest.approx(0.0)  # cells outside the radius are untouched
