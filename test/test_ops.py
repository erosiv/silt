"""Tests for the free-function tensor operations in the `silt` module
(set_/add_/multiply_/divide_/mix_/clamp_, their out-of-place counterparts
add/multiply/divide/mix/clamp, tensor.copy_to/cast/min/max) and the indexed
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
    silt.set_(t, 2.5)
    np.testing.assert_array_equal(t.numpy(), np.full(4, 2.5, dtype=np.float32))


def test_add_scalar():
    t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    silt.add_(t, 10.0)
    np.testing.assert_array_equal(t.numpy(), [11.0, 12.0, 13.0])


def test_multiply_scalar():
    t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    silt.multiply_(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [2.0, 4.0, 6.0])


def test_divide_scalar():
    t = silt.tensor.from_numpy(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    silt.divide_(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])


def test_add_tensor_tensor():
    a = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    silt.add_(a, b)
    np.testing.assert_array_equal(a.numpy(), [11.0, 22.0, 33.0])


def test_mix_interpolates():
    a = silt.tensor.from_numpy(np.array([0.0, 0.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 10.0], dtype=np.float32))
    silt.mix_(a, b, 0.25)
    np.testing.assert_allclose(a.numpy(), [2.5, 2.5])


def test_clamp_float32():
    t = silt.tensor.from_numpy(np.array([-5.0, 0.5, 5.0], dtype=np.float32))
    silt.clamp_(t, 0.0, 1.0)
    np.testing.assert_array_equal(t.numpy(), [0.0, 0.5, 1.0])


def test_clamp_rejects_non_float32():
    # `clamp_`'s binding constrains its type parameter to
    # `std::same_as<float>`, so calling it on a float64 tensor should
    # raise rather than silently doing nothing or misinterpreting data.
    t = silt.tensor.from_numpy(np.array([-5.0, 0.5, 5.0], dtype=np.float64))
    with pytest.raises(Exception):
        silt.clamp_(t, 0.0, 1.0)


# -- out-of-place counterparts (add/multiply/divide/mix/clamp) -------------


def test_add_out_of_place_leaves_input_unmutated():
    a = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    result = silt.add(a, b)
    np.testing.assert_array_equal(a.numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.numpy(), [11.0, 22.0, 33.0])


def test_multiply_out_of_place_leaves_input_unmutated():
    t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    result = silt.multiply(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.numpy(), [2.0, 4.0, 6.0])


def test_divide_out_of_place_leaves_input_unmutated():
    t = silt.tensor.from_numpy(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    result = silt.divide(t, 2.0)
    np.testing.assert_array_equal(t.numpy(), [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_mix_out_of_place_leaves_input_unmutated():
    a = silt.tensor.from_numpy(np.array([0.0, 0.0], dtype=np.float32))
    b = silt.tensor.from_numpy(np.array([10.0, 10.0], dtype=np.float32))
    result = silt.mix(a, b, 0.25)
    np.testing.assert_array_equal(a.numpy(), [0.0, 0.0])
    np.testing.assert_allclose(result.numpy(), [2.5, 2.5])


def test_clamp_out_of_place_leaves_input_unmutated():
    t = silt.tensor.from_numpy(np.array([-5.0, 0.5, 5.0], dtype=np.float32))
    result = silt.clamp(t, 0.0, 1.0)
    np.testing.assert_array_equal(t.numpy(), [-5.0, 0.5, 5.0])
    np.testing.assert_array_equal(result.numpy(), [0.0, 0.5, 1.0])


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


# -- copy_to / cast ---------------------------------------------------------


def test_copy_to_preserves_host():
    """tensor.copy_to() (built on tensor_t<T>::copy_to()) always allocates
    its result on the requested host -- defaulting to the source's own
    host when none is given -- rather than assuming GPU. Copying a CPU
    tensor should yield a CPU tensor with identical data.

    Run in a subprocess: a regression here would plausibly crash the
    interpreter (a device kernel dispatch against a host pointer) on a
    machine that does have a GPU, rather than merely returning the
    wrong host tag.
    """
    result = run_python(
        """
        import numpy as np
        import silt
        t = silt.tensor.from_numpy(np.array([1.0, 2.0, 3.5], dtype=np.float32))
        c = t.copy_to()
        assert c.host == silt.cpu, f"copy_to() returned host={c.host!r}"
        np.testing.assert_array_equal(c.numpy(), t.numpy())
        print("OK")
        """
    )
    assert not result.crashed, (
        f"tensor.copy_to() of a CPU tensor crashed.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"tensor.copy_to() did not preserve the source's host.\n"
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
    silt.set_(t, 0.0)
    silt.indexed_set(t, 1.0, idx)
    data = t.to_cpu().numpy()
    assert data.max() == pytest.approx(1.0)
    assert data.min() == pytest.approx(0.0)  # cells outside the radius are untouched


@pytest.mark.gpu
def test_indexed_add_scalar():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    t = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(t, 1.0)
    silt.indexed_add_(t, 10.0, idx)
    data = t.to_cpu().numpy()
    assert data.max() == pytest.approx(11.0)
    assert data.min() == pytest.approx(1.0)  # cells outside the radius are untouched


@pytest.mark.gpu
def test_indexed_add_tensor_tensor():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    a = silt.tensor(silt.float32, s, silt.gpu)
    b = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(a, 1.0)
    silt.set_(b, 10.0)
    silt.indexed_add_(a, b, idx)
    data = a.to_cpu().numpy()
    assert data.max() == pytest.approx(11.0)
    assert data.min() == pytest.approx(1.0)


@pytest.mark.gpu
def test_indexed_multiply_scalar():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    t = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(t, 2.0)
    silt.indexed_multiply_(t, 3.0, idx)
    data = t.to_cpu().numpy()
    assert data.max() == pytest.approx(6.0)
    assert data.min() == pytest.approx(2.0)


@pytest.mark.gpu
def test_indexed_divide_scalar():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    t = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(t, 6.0)
    silt.indexed_divide_(t, 3.0, idx)
    data = t.to_cpu().numpy()
    assert data.min() == pytest.approx(2.0)
    assert data.max() == pytest.approx(6.0)


@pytest.mark.gpu
def test_indexed_divide_tensor_tensor():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    a = silt.tensor(silt.float32, s, silt.gpu)
    b = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(a, 6.0)
    silt.set_(b, 3.0)
    silt.indexed_divide_(a, b, idx)
    data = a.to_cpu().numpy()
    assert data.min() == pytest.approx(2.0)
    assert data.max() == pytest.approx(6.0)


@pytest.mark.gpu
def test_indexed_mix_interpolates_only_within_mask():
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)
    a = silt.tensor(silt.float32, s, silt.gpu)
    b = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(a, 0.0)
    silt.set_(b, 10.0)
    silt.indexed_mix_(a, b, idx, 0.25)
    data = a.to_cpu().numpy()
    assert data.max() == pytest.approx(2.5)
    assert data.min() == pytest.approx(0.0)  # cells outside the mask are untouched


@pytest.mark.gpu
def test_indexed_ops_leave_cells_outside_the_index_set_untouched():
    """Cross-check against a dense numpy mask built from the same
    radius predicate, rather than just checking min/max, so a bug that
    happened to preserve the extremes wouldn't slip through."""
    s = silt.shape(8, 8)
    idx = silt.index_radius(s, [4.0, 4.0], 2.5)

    yy, xx = np.mgrid[0:8, 0:8]
    mask = (xx.astype(np.float32) - 4.0) ** 2 + (yy.astype(np.float32) - 4.0) ** 2 < 2.5 ** 2

    t = silt.tensor(silt.float32, s, silt.gpu)
    silt.set_(t, 1.0)
    silt.indexed_add_(t, 9.0, idx)
    data = t.to_cpu().numpy().reshape(8, 8)

    expected = np.where(mask, 10.0, 1.0)
    np.testing.assert_array_equal(data, expected)


# -- value-based selectors (index_range/index_greater/index_lesser/index_match, GPU only) ---


@pytest.mark.gpu
def test_index_range_selects_closed_interval():
    data = silt.tensor.from_numpy(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    data.to_gpu()
    idx = silt.index_range(data, 1.0, 3.0)
    got = sorted(idx.to_cpu().numpy().tolist())
    assert got == [1, 2, 3]  # both endpoints included


@pytest.mark.gpu
def test_index_greater_is_range_with_positive_infinity():
    data = silt.tensor.from_numpy(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    data.to_gpu()
    greater = silt.index_greater(data, 2.0)
    equivalent_range = silt.index_range(data, 2.0, float("inf"))
    np.testing.assert_array_equal(
        sorted(greater.to_cpu().numpy().tolist()),
        sorted(equivalent_range.to_cpu().numpy().tolist()),
    )
    assert sorted(greater.to_cpu().numpy().tolist()) == [2, 3, 4]


@pytest.mark.gpu
def test_index_lesser_is_range_with_negative_infinity():
    data = silt.tensor.from_numpy(np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    data.to_gpu()
    lesser = silt.index_lesser(data, 2.0)
    equivalent_range = silt.index_range(data, float("-inf"), 2.0)
    np.testing.assert_array_equal(
        sorted(lesser.to_cpu().numpy().tolist()),
        sorted(equivalent_range.to_cpu().numpy().tolist()),
    )
    assert sorted(lesser.to_cpu().numpy().tolist()) == [0, 1, 2]


@pytest.mark.gpu
def test_index_match_is_range_with_lo_equal_hi():
    data = silt.tensor.from_numpy(np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32))
    data.to_gpu()
    matched = silt.index_match(data, 2.0)
    got = sorted(matched.to_cpu().numpy().tolist())
    assert got == [1, 2]  # both cells equal to 2.0, none of the others


@pytest.mark.gpu
def test_index_range_empty_when_no_cell_qualifies():
    data = silt.tensor.from_numpy(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    data.to_gpu()
    idx = silt.index_range(data, 10.0, 20.0)
    assert idx.elem == 0


@pytest.mark.gpu
def test_index_greater_on_int32_uses_int_max_not_float_infinity():
    """int has no representable infinity, so index_greater's alias must
    fall back to numeric_limits<int>::max() as the upper bound rather
    than failing to cast float('inf') into an int."""
    data = silt.tensor.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    data.to_gpu()
    idx = silt.index_greater(data, 3)
    got = sorted(idx.to_cpu().numpy().tolist())
    assert got == [2, 3]


@pytest.mark.gpu
def test_index_lesser_on_int32_uses_int_min_not_float_infinity():
    """int has no representable -infinity, so index_lesser's alias must
    fall back to numeric_limits<int>::min() as the lower bound rather
    than failing to cast float('-inf') into an int."""
    data = silt.tensor.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    data.to_gpu()
    idx = silt.index_lesser(data, 2)
    got = sorted(idx.to_cpu().numpy().tolist())
    assert got == [0, 1]
