#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <silt/core/error.hpp>
#include <silt/core/memory.hpp>
#include <silt/core/types.hpp>

#include "glm.hpp"

#include <sstream>

//
//
//

//! General Util Binding Function
void bind_util(nb::module_& module) {

  //
  // Type Enumerator Binding
  //

  nb::enum_<silt::dtype>(module, "dtype")
      .value("int", silt::dtype::INT32)
      .value("float32", silt::dtype::FLOAT32)
      .value("float64", silt::dtype::FLOAT64)
      .value("rng", silt::dtype::RNG)
      .value("int64", silt::dtype::INT64)
      .export_values();

  //
  // Device Enumerator Binding
  //

  nb::enum_<silt::host_t>(module, "host")
      .value("cpu", silt::host_t::CPU)
      .value("gpu", silt::host_t::GPU)
      .export_values();

  //
  // Memory Usage Binding
  //

  nb::class_<silt::memory_stats>(module, "memory_stats", "Tensor memory currently held through silt, in bytes.")
      .def_ro("cpu_bytes", &silt::memory_stats::cpu_bytes, "Live CPU bytes.")
      .def_ro("gpu_bytes", &silt::memory_stats::gpu_bytes, "Live GPU bytes.")
      .def_ro("cpu_peak", &silt::memory_stats::cpu_peak, "Peak CPU bytes since start or the last reset.")
      .def_ro("gpu_peak", &silt::memory_stats::gpu_peak, "Peak GPU bytes since start or the last reset.")
      .def_ro("cpu_allocations", &silt::memory_stats::cpu_allocations, "Live CPU allocations.")
      .def_ro("gpu_allocations", &silt::memory_stats::gpu_allocations, "Live GPU allocations.")
      .def_ro("id", &silt::memory_stats::id, "Identifies the silt_lib instance that keeps this ledger.")
      .def("__repr__", [](const silt::memory_stats& m) {
        std::ostringstream out;
        out << "memory_stats(cpu_bytes=" << m.cpu_bytes << ", gpu_bytes=" << m.gpu_bytes
            << ", cpu_peak=" << m.cpu_peak << ", gpu_peak=" << m.gpu_peak
            << ", cpu_allocations=" << m.cpu_allocations << ", gpu_allocations=" << m.gpu_allocations << ")";
        return out.str();
      });

  module.def("memory_usage", &silt::memory_usage,
             "Snapshot of the tensor memory held through silt (every library sharing this silt_lib).");

  module.def("memory_reset_peak", &silt::memory_reset_peak, "Reset the peak counters to the current usage.");

  module.def(
      "device_memory_info",
      []() {
        size_t free = 0, total = 0;
        silt::device_memory_info(free, total);
        return std::make_tuple(free, total);
      },
      "(free, total) bytes of GPU memory as reported by the driver, including allocations made outside silt.");
}
