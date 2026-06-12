#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace {

//
// Slice Unpacking
//

void __unpack_slice(
  nb::handle& handle,
  Py_ssize_t& offset,
  Py_ssize_t& stride,
  Py_ssize_t& extent
) {

  if (PySlice_Check(handle.ptr())) {
    if (PySlice_Unpack(handle.ptr(), &offset, &extent, &stride) < 0) {
      throw nb::python_error();
    }
  }
  
  else {
    offset = nb::cast<Py_ssize_t>(handle);
    stride = 1;
    extent = 1;
  }

}

}
