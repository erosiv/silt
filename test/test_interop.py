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


def test_numpy_round_trip_int64():
    """int64 is the index-set storage type."""
    arr = np.arange(5, dtype=np.int64)
    t = silt.tensor.from_numpy(arr)
    assert t.dtype == silt.int64
    np.testing.assert_array_equal(t.numpy(), arr)


def test_from_numpy_rejects_unsupported_dtype():
    """`from_numpy` only special-cases float32/float64/int32/int64
    (interop.hpp); every other dtype should raise a clear error rather
    than silently misreading the buffer. int16 is used here explicitly
    so the test is not sensitive to numpy's platform-dependent default
    integer width."""
    arr = np.arange(5, dtype=np.int16)
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
    silt.set_(t, 7.0)
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
    silt.set_(t, 2.0)
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


# -- pytorch interop (optional dependency, CPU and GPU) ---------------------

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


def test_torch_from_cpu_tensor_round_trip():
    """from_torch used to assume every incoming tensor was CUDA data and
    handed its pointer straight to a GPU kernel -- for a CPU torch
    tensor this reinterpreted host memory as device memory. It now
    branches on the tensor's device and, for CPU, copies the same way
    from_numpy does. No CUDA required for this one.
    """
    t_torch = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t = silt.tensor.from_torch(t_torch)
    assert t.host == silt.cpu
    np.testing.assert_array_equal(t.numpy(), t_torch.numpy())


def test_torch_export_of_cpu_tensor():
    """The export direction (tensor.torch()) used to unconditionally
    require a GPU-hosted silt tensor, even though nothing about a torch
    tensor requires CUDA. It now mirrors the source tensor's own host
    into the returned torch tensor's device, matching from_torch's
    symmetric CPU/CUDA handling above. No CUDA required for this one.
    """
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = silt.tensor.from_numpy(arr)  # CPU-hosted silt tensor
    assert t.host == silt.cpu
    t_torch = t.torch()
    assert not t_torch.is_cuda
    np.testing.assert_array_equal(t_torch.numpy(), arr)


def test_torch_from_non_contiguous_cpu_tensor_is_read_correctly():
    """Mirrors test_non_contiguous_numpy_array_is_read_correctly for the
    CPU torch path: a transposed tensor is not contiguous, and must be
    read through its strides rather than scanned as if it were flat.
    """
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = base.t()  # shape (4, 3), non-contiguous
    assert not transposed.is_contiguous()
    t = silt.tensor.from_torch(transposed)
    np.testing.assert_array_equal(t.numpy(), transposed.numpy())


@pytest.mark.gpu
def test_torch_rejects_non_contiguous_cuda_tensor():
    """silt has no strided GPU copy, so a non-contiguous CUDA tensor is
    rejected rather than silently misread (the CUDA counterpart of the
    two tests above)."""
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device available")
    base = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    transposed = base.t()
    assert not transposed.is_contiguous()
    with pytest.raises(Exception):
        silt.tensor.from_torch(transposed)
