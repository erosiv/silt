"""silt -- simple immediate lightweight tensors.

The compiled extension module (``silt.silt``) provides the core types: shape,
slice, tensor, view, and the operation free-functions. This package re-exports
them and is the intended home for the pure-Python convenience layer.
"""

from .silt import *  # noqa: F401,F403

# Out-of-place counterparts of the in-place `_`-suffixed ops the extension
# binds (torch convention: `add_` mutates, `add` returns a new tensor).
# Implemented here rather than in C++ because "out-of-place" is just
# "clone, then mutate the clone" -- no need to duplicate the kernel
# dispatch for it.
def add(lhs, rhs):
    result = clone(lhs)
    add_(result, rhs)
    return result


def multiply(lhs, rhs):
    result = clone(lhs)
    multiply_(result, rhs)
    return result


def divide(lhs, rhs):
    result = clone(lhs)
    divide_(result, rhs)
    return result


def mix(lhs, rhs, w):
    result = clone(lhs)
    mix_(result, rhs, w)
    return result


def clamp(lhs, min, max):
    result = clone(lhs)
    clamp_(result, min, max)
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

# Re-export what the extension exposes, minus the extension module itself, which
# `import *` would otherwise leave in this package's namespace as an attribute.
from . import silt as _ext

__all__ = (
    [_n for _n in dir(_ext) if not _n.startswith("_")]
    + ["__version__", "add", "multiply", "divide", "mix", "clamp"]
)
