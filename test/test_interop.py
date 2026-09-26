"""Tests for numpy/pytorch interop (silt.tensor.numpy/from_numpy/torch/
from_torch).

This file is about correctness only. Performance (elementwise-loop
copies instead of memcpy, and the "no-copy" claim in the docs being
false) is a separate, tracked follow-up and is not exercised here.
"""
import numpy as np
import pytest

import silt
from conftest import run_python


def test_numpy_round_trip_float32():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = silt.tensor.from_numpy(arr)
    np.testing.assert_array_equal(t.numpy(), arr)


def test_numpy_round_trip_float64():
    arr = np.linspace(0, 1, 10, dtype=np.float64)
    t = silt.tensor.from_numpy(arr)
    assert t.type == silt.float64
    np.testing.assert_allclose(t.numpy(), arr)


def test_numpy_round_trip_int32():
    arr = np.arange(5, dtype=np.int32)
    t = silt.tensor.from_numpy(arr)
    np.testing.assert_array_equal(t.numpy(), arr)


def test_from_numpy_rejects_unsupported_dtype():
    """`from_numpy` only special-cases float32/float64/int32
    (interop.hpp); every other dtype should raise a clear error rather
    than silently misreading the buffer. int64 is used here explicitly
    so the test is not sensitive to numpy's platform-dependent default
    integer width."""
    arr = np.arange(5, dtype=np.int64)
    with pytest.raises(Exception):
        silt.tensor.from_numpy(arr)


def test_single_element_tensor_round_trips():
    """A tensor with exactly one element has shape.dim() == 0 (shape's
    dimension counter trims trailing extent-1 dimensions all the way
    down), and `.numpy()` switches on that value to pick the numpy
    array's rank -- so a 1-element tensor currently falls through to
    the `default:` case and raises "too many dimensions" instead of
    producing a scalar or 1-element array."""
    t = silt.tensor(silt.float32, silt.shape(1))
    silt.set(t, 7.0)
    arr = t.numpy()
    assert arr.size == 1
    assert float(np.asarray(arr).reshape(-1)[0]) == pytest.approx(7.0)


def test_trailing_singleton_dimension_round_trips():
    """A (512, 1) tensor has shape.dim() == 1 (trailing extent-1 dims
    are trimmed), so `.numpy()` currently returns a 1D array of length
    512 rather than a (512, 1) array. This documents that reshape, not
    element count, is what's affected -- data should still round-trip
    correctly."""
    t = silt.tensor(silt.float32, silt.shape(512, 1))
    silt.set(t, 2.0)
    arr = np.asarray(t.numpy())
    assert arr.size == 512
    np.testing.assert_array_equal(arr.reshape(-1), np.full(512, 2.0, dtype=np.float32))


def test_non_contiguous_numpy_array_is_read_correctly():
    """`from_numpy` reads `array.data()` and indexes it linearly
    (interop.hpp), as though the array were always contiguous. A
    transposed (or otherwise strided) numpy array is not, so this
    currently produces silently wrong data for any non-contiguous
    input. Run in a subprocess since misreading raw memory this way is
    a memory-safety concern for larger or oddly-strided arrays, not
    just a logic error."""
    result = run_python(
        """
        import numpy as np
        import silt
        base = np.arange(12, dtype=np.float32).reshape(3, 4)
        transposed = base.T  # shape (4, 3), non-contiguous
        assert not transposed.flags['C_CONTIGUOUS']
        t = silt.tensor.from_numpy(transposed)
        got = np.asarray(t.numpy()).reshape(transposed.shape)
        np.testing.assert_array_equal(got, transposed)
        print("OK")
        """
    )
    assert not result.crashed, (
        f"from_numpy on a non-contiguous array crashed.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"non-contiguous numpy array was read incorrectly.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# -- pytorch interop (optional dependency, GPU only) ------------------------

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_torch_round_trip():
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device available")
    t_torch = torch.full((4, 4), 2.5, dtype=torch.float32, device="cuda")
    t = silt.tensor.from_torch(t_torch)
    assert t.host == silt.gpu
    back = t.torch()
    torch.testing.assert_close(back.cpu(), t_torch.cpu())
