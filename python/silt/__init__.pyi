"""
silt -- simple immediate lightweight tensors.

The compiled extension module (``silt.silt``) provides the core types: shape,
slice, tensor, view, and the operation free-functions. This package re-exports
them and is the intended home for the pure-Python convenience layer.
"""

import _b
import _ext

from . import silt as silt
from .silt import (
    add_ as add_,
    argmax as argmax,
    argmin as argmin,
    argsort as argsort,
    cast as cast,
    clamp_ as clamp_,
    device_memory_info as device_memory_info,
    divide_ as divide_,
    dtype as dtype,
    gather as gather,
    host as host,
    index_box as index_box,
    index_complement as index_complement,
    index_difference as index_difference,
    index_greater as index_greater,
    index_intersection as index_intersection,
    index_lesser as index_lesser,
    index_match as index_match,
    index_polygon as index_polygon,
    index_radius as index_radius,
    index_range as index_range,
    index_slice as index_slice,
    index_sort_unique as index_sort_unique,
    index_symmetric_difference as index_symmetric_difference,
    index_union as index_union,
    indexed_add_ as indexed_add_,
    indexed_argmax as indexed_argmax,
    indexed_argmin as indexed_argmin,
    indexed_divide_ as indexed_divide_,
    indexed_max as indexed_max,
    indexed_mean as indexed_mean,
    indexed_min as indexed_min,
    indexed_mix_ as indexed_mix_,
    indexed_multiply_ as indexed_multiply_,
    indexed_set as indexed_set,
    indexed_sum as indexed_sum,
    max as max,
    maximum_ as maximum_,
    mean as mean,
    memory_reset_peak as memory_reset_peak,
    memory_stats as memory_stats,
    memory_usage as memory_usage,
    min as min,
    minimum_ as minimum_,
    mix_ as mix_,
    multiply_ as multiply_,
    sample_normal as sample_normal,
    sample_uniform as sample_uniform,
    scatter_ as scatter_,
    seed as seed,
    set_ as set_,
    shape as shape,
    slice as slice,
    sort_ as sort_,
    std as std,
    sum as sum,
    synchronize as synchronize,
    tensor as tensor,
    var as var,
    view as view
)


def histogram(data: _ext.tensor | _ext.view, bins: _b.int, lo: float | None = None, hi: float | None = None) -> _ext.tensor:
    """
    Counts of `data` in `bins` equal-width bins over [lo, hi], as an int
    tensor on the same host. The last bin is closed; values outside the range
    and NaNs are not counted. `lo` / `hi` default to the data's own min / max
    (tensors only; a view needs an explicit range).
    """

int: silt.dtype = silt.dtype.int

float32: silt.dtype = silt.dtype.float32

float64: silt.dtype = silt.dtype.float64

rng: silt.dtype = silt.dtype.rng

int64: silt.dtype = silt.dtype.int64

cpu: silt.host = silt.host.cpu

gpu: silt.host = silt.host.gpu

def _shape(spec: _ext.shape | Sequence[_b.int]) -> _ext.shape:
    """Accept a silt.shape, or a tuple/list of up to 4 ints."""

def _make(spec: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu, fill: float | None = None) -> _ext.tensor:
    """
    Shared allocation (+ optional scalar fill) behind zeros/ones/full/
    empty/like, so the shape/dtype/host argument handling exists once.
    """

def zeros(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """A new tensor filled with 0."""

def ones(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """A new tensor filled with 1."""

def full(shape: _ext.shape | Sequence[_b.int], value: float, dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """A new tensor filled with `value`."""

def empty(shape: _ext.shape | Sequence[_b.int], dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """
    Allocate without initializing -- contents are whatever the allocator
    handed back.
    """

def like(t: _ext.tensor, fill: float | None = None) -> _ext.tensor:
    """
    A new tensor with the same shape, dtype and host as `t`, optionally
    filled with a scalar value (left uninitialized if `fill` is omitted).
    """

def _numpy_dtype(dtype: _ext.dtype): ...

def arange(n: _b.int, dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """
    A 1D tensor of `n` consecutive values starting at 0, analogous to
    numpy.arange. Requires numpy.
    """

def linspace(a: float, b: float, n: _b.int, dtype: _ext.dtype = silt.dtype.float32, host: _ext.host = silt.host.cpu) -> _ext.tensor:
    """
    A 1D tensor of `n` evenly spaced values from `a` to `b` inclusive,
    analogous to numpy.linspace. Requires numpy.
    """

def rand(shape: _ext.shape | Sequence[_b.int], seed_value: _b.int = 0) -> _ext.tensor:
    """
    Allocate a GPU RNG-state tensor and seed it in one step -- previously
    two easy-to-forget calls: ``tensor(silt.rng, shape, silt.gpu)`` then
    ``silt.seed(t, seed_value, 0)``.

    Named `rand` rather than `rng`: `silt.rng` is already the RNG dtype
    constant (exported to module level like the other dtypes), and reusing
    the name here would shadow it and break existing code such as
    ``silt.tensor(silt.rng, shape, silt.gpu)``.
    """

def add(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place add: a copy of `lhs` plus `rhs` (tensor or scalar)."""

def multiply(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place multiply: a copy of `lhs` times `rhs` (tensor or scalar)."""

def divide(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """
    Out-of-place divide: a copy of `lhs` divided by `rhs` (tensor or scalar).
    """

def mix(lhs: _ext.tensor, rhs: _ext.tensor, w: float) -> _ext.tensor:
    """
    Out-of-place mix: a copy of `lhs` interpolated toward `rhs` by weight `w`.
    """

def sort(t: _ext.tensor) -> _ext.tensor:
    """Out-of-place sort: an ascending sorted copy of `t`."""

def clamp(lhs: _ext.tensor, min: float, max: float) -> _ext.tensor:
    """Out-of-place clamp: a copy of `lhs` limited to [min, max]."""

def minimum(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place elementwise minimum of a copy of `lhs` and `rhs`."""

def maximum(lhs: _ext.tensor, rhs: _ext.tensor | float) -> _ext.tensor:
    """Out-of-place elementwise maximum of a copy of `lhs` and `rhs`."""

__all__: list = ['add_', 'argmax', 'argmin', 'argsort', 'cast', 'clamp_', 'cpu', 'device_memory_info', 'divide_', 'dtype', 'float32', 'float64', 'gather', 'gpu', 'histogram', 'host', 'index_box', 'index_complement', 'index_difference', 'index_greater', 'index_intersection', 'index_lesser', 'index_match', 'index_polygon', 'index_radius', 'index_range', 'index_slice', 'index_sort_unique', 'index_symmetric_difference', 'index_union', 'indexed_add_', 'indexed_argmax', 'indexed_argmin', 'indexed_divide_', 'indexed_max', 'indexed_mean', 'indexed_min', 'indexed_mix_', 'indexed_multiply_', 'indexed_set', 'indexed_sum', 'int', 'int64', 'max', 'maximum_', 'mean', 'memory_reset_peak', 'memory_stats', 'memory_usage', 'min', 'minimum_', 'mix_', 'multiply_', 'rng', 'sample_normal', 'sample_uniform', 'scatter_', 'seed', 'set_', 'shape', 'slice', 'sort_', 'std', 'sum', 'synchronize', 'tensor', 'var', 'view', '__version__', 'zeros', 'ones', 'full', 'empty', 'like', 'arange', 'linspace', 'rand', 'add', 'multiply', 'divide', 'mix', 'sort', 'clamp', 'minimum', 'maximum']
