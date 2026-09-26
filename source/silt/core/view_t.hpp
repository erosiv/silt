#pragma once

#include <silt/core/error.hpp>
#include <silt/core/slice.hpp>
#include <silt/core/types.hpp>
#include <silt/silt.hpp>

namespace silt {

//! view_t is a strict-typed, lightweight data-view for vectorized memory reads.
//!
//! A view_t is intended to be constructed on the fly from another data-type, so
//! that any memory read operations from the view are compiler optimized.
//!
//! The view is non-owning and shares its underlying memory, so that it can be
//! passed cheaply while the owning data-types remaing untouched.
//!
//! A check to see whether the construction of a view is valid should be done at
//! construction time, and not at lookup time, to guarantee performance.
//!
template<typename T>
struct view_t: typedbase {

  typedef T val_t;

  view_t() {
    this->_slice = slice();
    this->_host = CPU;
    this->_data = NULL;
  }

  GPU_ENABLE view_t(const silt::slice slice, const host_t host, T* data): _slice{slice},
                                                                          _host{host},
                                                                          _data{data} {}

  GPU_ENABLE inline silt::slice slice() const { return this->_slice; }  //!< View Sliced Shape
  GPU_ENABLE inline size_t elem() const { return this->_slice.elem(); } //!< Number of Elements
  GPU_ENABLE inline host_t host() const { return this->_host; }         //!< Current Device (CPU / GPU)
  GPU_ENABLE inline const T* data() const { return this->_data; }       //!< Raw Data Pointer (Const)
  GPU_ENABLE inline T* data() { return this->_data; }                   //!< Raw Data Pointer (Mutable)

  //! Type Enumerator Retrieval
  constexpr silt::dtype type() noexcept {
    return silt::typedesc<T>::type;
  }

  //! Const Subscript Operator: With Slice Transform!
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[this->_slice.transform(index)];
  }

  //! Non-Const Subscript Operator: With Slice Transform!
  GPU_ENABLE T& operator[](const size_t index) noexcept {
    return this->_data[this->_slice.transform(index)];
  }

  // Advanced Shape Manipulation

  void reshape(int d0, int d1, int d2, int d3) {
    this->_slice.reshape(d0, d1, d2, d3);
  }

  void index(const int dim, const int offset, const int stride, const int extent) {
    this->_slice.index(dim, offset, stride, extent);
  }

  void reset() {
    this->_slice.reset();
  }

private:
  silt::slice _slice; //!< Sliced Shape of Data
  host_t _host = CPU; //!< Compute Device Location
  T* _data = NULL;    //!< Raw Data Pointer
};

} // namespace silt
