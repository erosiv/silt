"""Shared pytest fixtures and helpers for the silt test suite.

Some behaviour under test is undefined behaviour in the current
implementation -- e.g. constructing an uninitialized `silt.tensor()`, or
reading `shape.__repr__`'s dangling pointer. A test that exercises
undefined behaviour must never run in-process, because a crash there
would take down the entire pytest session and hide every other test's
result. `run_python` runs such code in a fresh subprocess instead, so a
crash is reported as a single failed test with the child's output
attached, rather than aborting the run.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest


@dataclass
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def crashed(self) -> bool:
        """True if the subprocess exited non-zero.

        This covers both a genuine interpreter crash (segfault, abort)
        and an ordinary unhandled exception or failed assertion inside
        the subprocess. Either way the outer test should fail, with the
        child's stdout/stderr shown for diagnosis, so both cases are
        folded into this one check.
        """
        return self.returncode != 0


def run_python(code: str, timeout: float = 30.0) -> SubprocessResult:
    """Run `code` in a fresh Python subprocess and capture the outcome.

    Use this for any test that exercises code documented as unsafe or
    undefined in the current implementation, so a crash is reported as
    an ordinary test failure instead of aborting the whole test run.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return SubprocessResult(proc.returncode, proc.stdout, proc.stderr)


def _gpu_available() -> bool:
    """Detect a usable NVIDIA GPU independently of silt itself, since
    silt's own CUDA error handling doesn't reliably distinguish "no GPU"
    from "broken" yet."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    return _gpu_available()


@pytest.fixture(autouse=True)
def _skip_gpu_tests_if_unavailable(request, gpu_available):
    """Auto-skip any test marked `@pytest.mark.gpu` when no GPU is
    detected, e.g. on hosted CI runners, which have no GPU at all."""
    if request.node.get_closest_marker("gpu") and not gpu_available:
        pytest.skip("no usable GPU detected (nvidia-smi not found or reported none)")
