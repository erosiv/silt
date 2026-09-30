"""silt -- simple immediate lightweight tensors.

The compiled extension module (``silt.silt``) provides the core types: shape,
slice, tensor, view, and the operation free-functions. This package re-exports
them and is the intended home for the pure-Python convenience layer.
"""

from __future__ import annotations

import builtins as _b  # `int` etc. are shadowed by the dtype constants exported below
from typing import Sequence

from .silt import *  # noqa: F401,F403

# Re-exported here (rather than at the bottom, where it originally lived)
# because the constructors below need it: some dtype/host values' Python
# names collide with builtins (`silt.int`), so those are looked up as
# `_ext.dtype.int` instead of the shadowed bare name.
from . import silt as _ext


def _shape(spec: _ext.shape | Sequence[_b.int]) -> _ext.shape:
    """Accept a silt.shape, or a tuple/list of up to 4 ints."""
    if isinstance(spec, shape):
        return spec
    dims = tuple(spec)
    if not (1 <= len(dims) <= 4):
        raise ValueError(f"shape must have 1 to 4 dimensions, got {len(dims)}")
    return shape(*dims)


def _make(spec: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = float32, host: _ext.host = cpu, fill: float | None = None) -> _ext.tensor:
    """Shared allocation (+ optional scalar fill) behind zeros/ones/full/
    empty/like, so the shape/dtype/host argument handling exists once."""
    t = tensor(dtype, _shape(spec), host)
    if fill is not None:
        set_(t, fill)
    return t


def zeros(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """A new tensor filled with 0."""
    return _make(shape, dtype, host, fill=0)


def ones(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """A new tensor filled with 1."""
    return _make(shape, dtype, host, fill=1)


def full(shape: _ext.shape | Sequence[_b.int], value: float, dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """A new tensor filled with `value`."""
    return _make(shape, dtype, host, fill=value)


def empty(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """Allocate without initializing -- contents are whatever the allocator
    handed back."""
    return _make(shape, dtype, host)


def like(t: _ext.tensor, fill: float | None = None) -> _ext.tensor:
    """A new tensor with the same shape, dtype and host as `t`, optionally
    filled with a scalar value (left uninitialized if `fill` is omitted)."""
    return _make(t.shape, t.dtype, t.host, fill=fill)


def _numpy_dtype(dtype: _ext.dtype):
    # silt has no hard runtime dependency on numpy (see pyproject.toml) --
    # only arange/linspace need it, so the import stays local to them.
    import numpy as _np

    if dtype == float32:
        return _np.float32
    if dtype == float64:
        return _np.float64
    if dtype == _ext.dtype.int:
        return _np.int32
    raise ValueError(f"no numpy dtype for {dtype!r}")


def arange(n: _b.int, dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """A 1D tensor of `n` consecutive values starting at 0, analogous to
    numpy.arange. Requires numpy."""
    import numpy as _np

    data = _np.arange(n, dtype=_numpy_dtype(dtype))
    return tensor.from_numpy(data).to(host)


def linspace(a: float, b: float, n: _b.int, dtype: _ext.dtype = float32, host: _ext.host = cpu) -> _ext.tensor:
    """A 1D tensor of `n` evenly spaced values from `a` to `b` inclusive,
    analogous to numpy.linspace. Requires numpy."""
    import numpy as _np

    data = _np.linspace(a, b, n, dtype=_numpy_dtype(dtype))
    return tensor.from_numpy(data).to(host)


def rand(shape: _ext.shape | Sequence[_b.int], seed_value: _b.int = 0) -> _ext.tensor:
    """Allocate a GPU RNG-state tensor and seed it in one step -- previously
    two easy-to-forget calls: ``tensor(silt.rng, shape, silt.gpu)`` then
    ``silt.seed(t, seed_value, 0)``.

    Named `rand` rather than `rng`: `silt.rng` is already the RNG dtype
    constant (exported to module level like the other dtypes), and reusing
    the name here would shadow it and break existing code such as
    ``silt.tensor(silt.rng, shape, silt.gpu)``.
    """
    t = tensor(_ext.dtype.rng, _shape(shape), _ext.host.gpu)
    seed(t, seed_value, 0)
    return t


# Out-of-place counterparts of the in-place `_`-suffixed ops the extension
# binds (torch convention: `add_` mutates, `add` returns a new tensor).
# Implemented here rather than in C++ because "out-of-place" is just
# "copy, then mutate the copy" -- no need to duplicate the kernel
# dispatch for it.
def add(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place add: a copy of `lhs` plus `rhs` (tensor or scalar)."""
    result = lhs.copy_to()
    add_(result, rhs)
    return result


def multiply(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place multiply: a copy of `lhs` times `rhs` (tensor or scalar)."""
    result = lhs.copy_to()
    multiply_(result, rhs)
    return result


def divide(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place divide: a copy of `lhs` divided by `rhs` (tensor or scalar)."""
    result = lhs.copy_to()
    divide_(result, rhs)
    return result


def mix(lhs: _ext.tensor, rhs: _ext.tensor, w: float) -> _ext.tensor:
    """Out-of-place mix: a copy of `lhs` interpolated toward `rhs` by weight `w`."""
    result = lhs.copy_to()
    mix_(result, rhs, w)
    return result


def sort(t: _ext.tensor) -> _ext.tensor:
    """Out-of-place sort: an ascending sorted copy of `t`."""
    result = t.copy_to()
    sort_(result)
    return result


def histogram(data: _ext.tensor | _ext.view, bins: _b.int, lo: float | None = None, hi: float | None = None) -> _ext.tensor:
    """Counts of `data` in `bins` equal-width bins over [lo, hi], as an int
    tensor on the same host. The last bin is closed; values outside the range
    and NaNs are not counted. `lo` / `hi` default to the data's own min / max
    (tensors only; a view needs an explicit range)."""
    if lo is None:
        lo = _ext.min(data)
    if hi is None:
        hi = _ext.max(data)
    return _ext.histogram(data, bins, lo, hi)


def clamp(lhs: _ext.tensor, min: float, max: float) -> _ext.tensor:
    """Out-of-place clamp: a copy of `lhs` limited to [min, max]."""
    result = lhs.copy_to()
    clamp_(result, min, max)
    return result


def minimum(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place elementwise minimum of a copy of `lhs` and `rhs`."""
    result = lhs.copy_to()
    minimum_(result, rhs)
    return result


def maximum(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place elementwise maximum of a copy of `lhs` and `rhs`."""
    result = lhs.copy_to()
    maximum_(result, rhs)
    return result


# The version lives in the root VERSION file, which is not shipped in the wheel.
# At runtime we read it back from the installed distribution metadata, which
# scikit-build-core populates from that same file at build time.
try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        __version__ = _dist_version("silt-erosiv")
    except PackageNotFoundError:  # not installed, e.g. imported from a build tree
        __version__ = "0+unknown"
except ImportError:  # pragma: no cover -- importlib.metadata is stdlib from 3.8
    __version__ = "0+unknown"

# Dict-based dedupe: wrappers (e.g. histogram) shadow same-named extension names.
__all__ = list(dict.fromkeys(
    [_n for _n in dir(_ext) if not _n.startswith("_")]
    + [
        "__version__",
        "zeros", "ones", "full", "empty", "like", "arange", "linspace", "rand",
        "add", "multiply", "divide", "mix", "sort", "histogram", "clamp", "minimum", "maximum",
    ]
))
