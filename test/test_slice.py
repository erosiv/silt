"""Tests for slicing (silt.tensor.__getitem__), backed by silt::slice.

silt.slice itself only exposes read-only introspection (dim, shape,
offset, stride, extent, transform); the construction logic lives in
silt::slice::index(), which is only reachable from Python through
tensor slicing -- so that is what these tests exercise.

Note: a bare `t[a:b:c]` passes a single slice object to __getitem__,
not a tuple, and the C++ binding expects a tuple. Use a trailing comma
(`t[a:b:c,]`) to index a 1D tensor here.
"""
import numpy as np
import pytest

import silt
from conftest import run_python


def test_full_slice_default_matches_original_extent():
    t = silt.tensor(silt.float32, silt.shape(10))
    v = t[0:10:1,]
    assert v.slice.extent[0] == 10


def test_offset_slice_respects_bound():
    t = silt.tensor(silt.float32, silt.shape(10))
    v = t[2:10:1,]
    assert v.slice.offset[0] == 2
    assert v.slice.extent[0] == 8


def test_offset_beyond_bound_raises():
    t = silt.tensor(silt.float32, silt.shape(10))
    with pytest.raises(Exception):
        _ = t[15:20:1,]


def test_strided_slice_includes_the_last_valid_element():
    """silt::slice::index() clamps the requested extent with
    `(bound - offset) / stride`, which truncates rather than rounds up.
    For bound=10, offset=0, stride=3 the valid indices are 0, 3, 6, 9 --
    four elements -- but the clamp currently yields 3, silently
    dropping the last one.
    """
    t = silt.tensor(silt.float32, silt.shape(10))
    v = t[0:10:3,]
    assert v.slice.extent[0] == 4, (
        "strided slice dropped its last valid element: "
        f"got extent {v.slice.extent[0]}, expected 4"
    )


def test_strided_slice_writes_reach_every_selected_element():
    """The same bug, observed through data rather than metadata: with
    the extent under-counted, a `silt.set` on the slice currently
    leaves the last valid stride position untouched, which is the
    user-visible symptom."""
    t = silt.tensor(silt.float32, silt.shape(10))
    silt.set_(t, 0.0)
    v = t[0:10:3,]
    silt.set_(v, 1.0)
    data = t.numpy()
    expected = np.zeros(10, dtype=np.float32)
    expected[0:10:3] = 1.0
    np.testing.assert_array_equal(data, expected)


def test_single_index_selects_one_element():
    t = silt.tensor(silt.float32, silt.shape(10))
    silt.set_(t, 0.0)
    v = t[3,]
    assert v.slice.extent[0] == 1
    silt.set_(v, 9.0)
    data = t.numpy()
    assert data[3] == pytest.approx(9.0)
    assert data[2] == pytest.approx(0.0)
    assert data[4] == pytest.approx(0.0)


def test_view_survives_source_tensor_being_dropped():
    """A view held a raw pointer into its source tensor_t<T> with no
    ownership tie, so once the last Python reference to the source
    tensor was dropped, the view's buffer was freed out from under it
    -- a use-after-free on the very next operation. Run in a subprocess
    since that is undefined behaviour, not just a wrong-answer bug.
    """
    result = run_python(
        """
        import silt
        t = silt.tensor(silt.float32, silt.shape(10))
        silt.set_(t, 0.0)
        v = t[0:10:1,]
        del t
        silt.set_(v, 5.0)
        assert v.elem == 10
        assert v.type == silt.float32
        print("OK")
        """
    )
    assert not result.crashed, (
        f"using a view after its source tensor was dropped crashed.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
