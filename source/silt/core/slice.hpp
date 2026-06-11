#ifndef SILT_SLICE
#define SILT_SLICE

#include <silt/silt.hpp>
#include <silt/core/types.hpp>
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
    shape{d0, d1, d2, d3},
    offset{0, 0, 0, 0},
    stride{1, 1, 1, 1},
    extent{d0, d1, d2, d3},
    _dim{4}{}

  GPU_ENABLE slice(int d0, int d1, int d2):
    shape{d0, d1, d2, 1},
    offset{0, 0, 0, 0},
    stride{1, 1, 1, 1},
    extent{d0, d1, d2, 1},
    _dim{3}{}

  GPU_ENABLE slice(int d0, int d1):
    shape{d0, d1, 1, 1},
    offset{0, 0, 0, 0},
    stride{1, 1, 1, 1},
    extent{d0, d1, 1, 1},
    _dim{2}{}

  GPU_ENABLE slice(int d0):
    shape{d0, 1, 1, 1},
    offset{0, 0, 0, 0},
    stride{1, 1, 1, 1},
    extent{d0, 1, 1, 1},
    _dim{1}{}

  GPU_ENABLE slice():
    shape{1, 1, 1, 1},
    offset{0, 0, 0, 0},
    stride{1, 1, 1, 1},
    extent{1, 1, 1, 1},
    _dim{0}{}

  GPU_ENABLE slice(const silt::shape shape):
    slice(shape[0], shape[1], shape[2], shape[3]){}

  // 

  //! Dimension Subscript Operator
  GPU_ENABLE inline int operator[](const size_t d) const {
    return this->shape[d];
  }

  GPU_ENABLE int maxelem() const {
    return this->shape[0] * this->shape[1] * this->shape[2] * this->shape[3];
  }

  GPU_ENABLE int dim() const {
    return this->_dim;
  }

  GPU_ENABLE int elem() const {
    return this->extent[0] * this->extent[1] * this->extent[2] * this->extent[3];
  }

  GPU_ENABLE int index(const int ind) const {
    return ind;
//    vec_t value = this->unflatten(ind);
//    return this->flatten(value);
  }

  // Reshape the Slice

//  GPU_ENABLE reshape(int d0, int d1, int d2, int d3) {
//
//  }
//
//  //! Slice Index along individual Dimension
//  GPU_ENABLE void slice(const int dim, const int offset, const int stride, const int extlim) {
//    this->offset[dim] = offset;
//    this->stride[dim] = stride;
//    this->extlim[dim] = extlim;
//  }

private:

  int _dim;     //!< Total Number of Active Dimensions
  vec_t shape;  //!< Non-Sliced Tensor Shape
  vec_t offset; //!< Per-Dimension Offset
  vec_t stride; //!< Per-Dimension Stride
  vec_t extent; //!< Per-Dimension Extent

};

} // end of namespace silt

#endif