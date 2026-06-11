#ifndef SILT_SLICE
#define SILT_SLICE

#include <silt/silt.hpp>
#include <silt/core/types.hpp>
#include <silt/core/error.hpp>
#include <silt/core/shape.hpp>

namespace silt {

//! slice is a type for index conversions that supports multi-dimensional slicing
//! This type exists primarily so that tensor operations can be executed on subspaces.
//! This is distinct from a shape because it performs additional computations to transform
//! coordinates, whereas the regular shape assumes densely packed regular data.
//!
//! todo: Consider copy-free transpositions and other ordering implementations.
struct slice {

  using vec_t = glm::vec<4, int>;
  
  // Constructors

  GPU_ENABLE slice(int d0, int d1, int d2, int d3):
    _shape{d0, d1, d2, d3}{
      this->reset();
    }

  GPU_ENABLE slice(int d0, int d1, int d2):
    _shape{d0, d1, d2, 1}{
      this->reset();
    }

  GPU_ENABLE slice(int d0, int d1):
    _shape{d0, d1, 1, 1}{
      this->reset();
    }

  GPU_ENABLE slice(int d0):
    _shape{d0, 1, 1, 1}{
      this->reset();
    }

  GPU_ENABLE slice():
    _shape{1, 1, 1, 1}{
      this->reset();
    }

  GPU_ENABLE slice(const silt::shape shape):
    slice(shape[0], shape[1], shape[2], shape[3]){}

  // Member Data Lookup

  GPU_ENABLE inline silt::shape shape() const { return this->_shape; }
  GPU_ENABLE inline int dim()           const { return this->_shape.dim(); }
  GPU_ENABLE inline int maxelem()       const { return this->_shape.elem(); }
  GPU_ENABLE inline vec_t offset()      const { return this->_offset; }
  GPU_ENABLE inline vec_t stride()      const { return this->_stride; }
  GPU_ENABLE inline vec_t extent()      const { return this->_extent; }
  GPU_ENABLE inline int elem()          const {
    return silt::shape::count_elem(this->_extent);
  }

  //
  // Slice Manipulation
  //

  void reshape(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
    this->_shape.reshape(d0, d1, d2, d3);
    this->reset();
  }

  void flatten() {
    this->reshape(this->_shape.elem());
    this->reset();
  }

  //! Slice Index along individual Dimension
  void index(const int dim, const int offset, const int stride, const int extent) {
    this->_offset[dim] = offset;
    this->_stride[dim] = stride;
    this->_extent[dim] = extent;
  }

  GPU_ENABLE void reset() {
    this->_offset = {0, 0, 0, 0};
    this->_stride = {1, 1, 1, 1};
    this->_extent = this->_shape.ext();
  }

  //
  // Coordinate Transform Methods
  //

  GPU_ENABLE int transform(const int index) const {
    vec_t value = this->stride() * this->_unflatten(index);
    return this->_shape.flatten(this->offset() + value);
  }

private:

  //! Compute the Position in Slice-Space
  GPU_ENABLE vec_t _unflatten(const int index) const {
    vec_t value{0};
    int scale = 1;
    for (int d = this->dim() - 1; d >= 0; --d) {
      value[d] = (index / scale) % this->_extent[d];
      scale *= this->_extent[d];
    }
    return value;
  }

  silt::shape _shape; //!< Underlying Shape
  vec_t _offset;      //!< Per-Dimension Offset
  vec_t _stride;      //!< Per-Dimension Stride
  vec_t _extent;      //!< Per-Dimension Extent

};

} // end of namespace silt

#endif