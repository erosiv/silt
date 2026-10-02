"""silt's allocation ledger: every tensor allocation is counted, for all users of silt_lib."""

import gc

import pytest

import silt


def _cpu_bytes():
    return silt.memory_usage().cpu_bytes


def test_cpu_tensor_is_counted_until_released():
    gc.collect()
    before = silt.memory_usage()
    t = silt.zeros((64, 64), silt.float32, silt.cpu)
    during = silt.memory_usage()
    assert during.cpu_bytes == before.cpu_bytes + 64 * 64 * 4
    assert during.cpu_allocations == before.cpu_allocations + 1
    del t
    gc.collect()
    assert silt.memory_usage().cpu_bytes == before.cpu_bytes


def test_views_hold_no_memory_of_their_own():
    t = silt.zeros((8, 8, 3), silt.float32, silt.cpu)
    before = _cpu_bytes()
    v = t[:, :, 0]
    assert _cpu_bytes() == before
    del v


def test_element_size_follows_dtype():
    before = _cpu_bytes()
    a = silt.zeros((10,), silt.float64, silt.cpu)
    assert _cpu_bytes() == before + 10 * 8
    del a


def test_peak_persists_until_reset():
    silt.memory_reset_peak()
    base = silt.memory_usage()
    t = silt.zeros((1024,), silt.float32, silt.cpu)
    del t
    gc.collect()
    assert silt.memory_usage().cpu_peak >= base.cpu_bytes + 1024 * 4
    silt.memory_reset_peak()
    assert silt.memory_usage().cpu_peak == silt.memory_usage().cpu_bytes


def test_repr_lists_the_counters():
    assert "cpu_bytes=" in repr(silt.memory_usage())


@pytest.mark.gpu
def test_gpu_tensor_is_counted_until_released():
    gc.collect()
    before = silt.memory_usage()
    t = silt.zeros((64, 64), silt.float32, silt.gpu)
    assert silt.memory_usage().gpu_bytes == before.gpu_bytes + 64 * 64 * 4
    del t
    gc.collect()
    assert silt.memory_usage().gpu_bytes == before.gpu_bytes


@pytest.mark.gpu
def test_operation_scratch_is_released():
    t = silt.zeros((4096,), silt.float32, silt.gpu)
    before = silt.memory_usage()
    silt.sum(t)
    silt.min(t)
    after = silt.memory_usage()
    assert after.gpu_bytes == before.gpu_bytes
    assert after.gpu_allocations == before.gpu_allocations


@pytest.mark.gpu
def test_device_memory_info_reports_free_and_total():
    free, total = silt.device_memory_info()
    assert 0 < free <= total
