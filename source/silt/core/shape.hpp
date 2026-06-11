#ifndef SILT_SHAPE
#define SILT_SHAPE

#include <silt/silt.hpp>
#include <silt/core/types.hpp>
#include <silt/core/error.hpp>

namespace silt {

//! shape is a D-dimensional compact extent, with indexing
//! procedures for lookup into linearized tensors and views.
//!
//! shape is intended for 2D and 3D applications, with a D of max 4,
//! for 3D buffers of arbitrary depth.
//! 
//! \todo cleanup this type with cleaner constructors
//! \todo add better flattening / unflattening procedures.
//! \todo consider whether this type should have slice generators.
//!
struct shape {

  using vec_t = glm::vec<4, int>;

  // Constructors

  GPU_ENABLE shape(int d0, int d1, int d2, int d3):
    _ext{d0, d1, d2, d3} {
      this->_dim = shape::count_dim(this->ext());
      this->_elem = shape::count_elem(this->ext());
    }

  GPU_ENABLE shape(int d0, int d1, int d2):
    _ext{d0, d1, d2, 1} {
      this->_dim = shape::count_dim(this->ext());
      this->_elem = shape::count_elem(this->ext());
    }

  GPU_ENABLE shape(int d0, int d1):
    _ext{d0, d1, 1, 1} {
      this->_dim = shape::count_dim(this->ext());
      this->_elem = shape::count_elem(this->ext());
    }

  GPU_ENABLE shape(int d0):
    _ext{d0, 1, 1, 1} {
      this->_dim = shape::count_dim(this->ext());
      this->_elem = shape::count_elem(this->ext());
    }

  GPU_ENABLE shape():
    _ext{1, 1, 1, 1} {
      this->_dim = shape::count_dim(this->ext());
      this->_elem = shape::count_elem(this->ext());
    }

  // Member Lookup

  GPU_ENABLE inline int dim()   const { return this->_dim; }
  GPU_ENABLE inline int elem()  const { return this->_elem; }
  GPU_ENABLE inline vec_t ext() const { return this->_ext; }

  //! Dimension Subscript Operator
  GPU_ENABLE inline int operator[](const size_t d) const {
    return this->_ext[d];
  }

  //! Out-Of-Bounds Check (Compact)
  GPU_ENABLE bool oob(const vec_t pos) const {
    for (size_t d = 0; d < this->_dim; ++d)
      if (pos[d] < 0 || pos[d] >= this->_ext[d])
        return true;
    return false;
  }

  // note: this should not necessarily be done like this...

  GPU_ENABLE bool oob(const silt::ivec2 pos) const {
    for (size_t d = 0; d < 2; ++d)
      if (pos[d] < 0 || pos[d] >= this->_ext[d])
        return true;
    return false;
  }

  GPU_ENABLE bool oob(const silt::ivec3 pos) const {
    for (size_t d = 0; d < 3; ++d)
      if (pos[d] < 0 || pos[d] >= this->_ext[d])
        return true;
    return false;
  }
  
  GPU_ENABLE int flatten(const silt::ivec2 pos) const {
    int index{0};
    for (size_t d = 0; d < 2; ++d) {
      index *= this->_ext[d];
      index += pos[d];
    }
    return index;
  }

  GPU_ENABLE int flatten(const silt::ivec3 pos) const {
    int index{0};
    for (size_t d = 0; d < 3; ++d) {
      index *= this->_ext[d];
      index += pos[d];
    }
    return index;
  }

  //! Flattening Operator
  GPU_ENABLE int flatten(const vec_t pos) const {
    int index{0};
    for (size_t d = 0; d < this->_dim; ++d) {
      index *= this->_ext[d];
      index += pos[d];
    }
    return index;
  }
  
  //! Unflattening Operator
  GPU_ENABLE vec_t unflatten(const int index) const {
    vec_t value{0};
    int scale = 1;
    for (int d = this->_dim - 1; d >= 0; --d) {
      value[d] = (index / scale) % this->_ext[d];
      scale *= this->_ext[d];
    }
    return value;
  }

  //
  // Shape Manipulation
  //

  void reshape(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {

    const int elem = d0 * d1 * d2 * d3;
    if(elem != this->elem())
      throw silt::error::bad_reshape(this->elem(), elem);

    this->_ext = {d0, d1, d2, d3};
    this->_dim = count_dim(this->_ext);

  }

private:

  static GPU_ENABLE int count_elem(const vec_t ext) {
    return ext[0] * ext[1] * ext[2] * ext[3];
  }

  static GPU_ENABLE int count_dim(const vec_t ext) {
    int d = 0;
    if(ext[0] > 1) d = 1;
    if(ext[1] > 1) d = 2;
    if(ext[2] > 1) d = 3;
    if(ext[3] > 1) d = 4;
    return d;
  }

  // Data Members

  int _dim;    //!< Total Number of Active Dimensions
  int _elem;   //!< Total Number of Elements
  vec_t _ext;  //!< Per-Dimension Extent

};

}

#endif