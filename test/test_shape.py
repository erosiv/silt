"""Tests for silt.shape: construction, indexing, and reshape."""
import pytest

import silt
from conftest import run_python


def test_construct_1d():
    s = silt.shape(4)
    assert s.dim == 1
    assert s.elem == 4
    assert s.ext[0] == 4


def test_construct_2d():
    s = silt.shape(4, 8)
    assert s.dim == 2
    assert s.elem == 32
    assert (s[0], s[1]) == (4, 8)


def test_construct_3d():
    s = silt.shape(2, 3, 4)
    assert s.dim == 3
    assert s.elem == 24


def test_construct_4d():
    s = silt.shape(2, 2, 2, 2)
    assert s.dim == 4
    assert s.elem == 16


def test_default_construct_is_unit_shape():
    s = silt.shape()
    assert s.elem == 1


def test_trailing_size_one_dims_are_not_counted_as_active():
    # shape.count_dim() trims trailing extent-1 dimensions, so a
    # (512, 1) shape reports dim == 1, not 2. Documented, deliberate
    # behaviour, pinned down here so a future change to it is a
    # conscious decision rather than an accident.
    s = silt.shape(512, 1)
    assert s.dim == 1
    assert s.elem == 512


def test_reshape_preserves_element_count():
    s = silt.shape(4, 4)
    s2 = s.reshape(2, 8)
    assert s2.elem == 16
    assert (s2[0], s2[1]) == (2, 8)


def test_reshape_rejects_mismatched_element_count():
    s = silt.shape(4, 4)
    with pytest.raises(Exception):
        s.reshape(3, 3)


def test_repr_does_not_crash_or_corrupt():
    """`shape.__repr__` returns a pointer into a std::format temporary
    that is destroyed at the end of the full expression -- a heap-use-
    after-free whose printed value is garbage rather than the intended
    "silt.shape(4, 8)". Run in a subprocess: this is undefined
    behaviour, which may or may not crash depending on allocator state,
    and a crash here must not take down the rest of the suite.
    """
    result = run_python(
        """
        import silt
        s = silt.shape(4, 8)
        text = repr(s)
        assert text == "silt.shape(4, 8)", f"got corrupted repr: {text!r}"
        print("OK")
        """
    )
    assert not result.crashed, (
        f"repr(shape) failed: repr() should return a valid string, "
        f"not a dangling pointer into a destroyed temporary.\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
