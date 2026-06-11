#ifndef SILT_VIEW
#define SILT_VIEW

#include <silt/silt.hpp>
#include <silt/core/types.hpp>
#include <silt/core/slice.hpp>

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

  view_t() {
    this->_slice = slice();
    this->_host = CPU;
    this->_data = NULL;
  }

  GPU_ENABLE view_t(const silt::slice slice, const host_t host, T* data):
    _slice{slice},
    _host{host},
    _data{data}{}

  GPU_ENABLE inline silt::slice slice()   const { return this->_slice; }        //!< View Sliced Shape
  GPU_ENABLE inline size_t elem()         const { return this->_slice.elem(); } //!< Number of Elements
  GPU_ENABLE inline host_t host()         const { return this->_host; }         //!< Current Device (CPU / GPU)
  GPU_ENABLE inline const T *data()       const { return this->_data; }         //!< Raw Data Pointer (Const)
  GPU_ENABLE inline T *data()                   { return this->_data; }         //!< Raw Data Pointer (Mutable)

  //! Type Enumerator Retrieval
  constexpr silt::dtype type() noexcept {
    return silt::typedesc<T>::type;
  }

  //! Const Subscript Operator
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }
  
  //! Non-Const Subscript Operator
  GPU_ENABLE T &operator[](const size_t index) noexcept {
    return this->_data[index];
  }

private:

  silt::slice _slice; //!< Sliced Shape of Data
  host_t _host = CPU; //!< Compute Device Location
  T* _data = NULL;    //!< Raw Data Pointer

};

//! view is a tag-poylymorphic view_t wrapper type.
struct EXPORT_SHARED view {

  view() = default;

  //! Polymorphic Tensor Copy Constructor
  view(const silt::view& rhs){
    this->impl = rhs.clone();
  }

  //! Polymorphic Tensor Move Constructor
  view(silt::view&& rhs){
    this->impl = rhs.clone();
  }

  //! Strict-Typed Tensor Copy Constructor
  template<typename T>
  view(const silt::view_t<T> &view) {
    this->impl = new silt::view_t<T>(view);
  }

  //! Strict-Typed Tensor Move Constructor
  template<typename T>
  view(silt::view_t<T> &&ten) {
    this->impl = new silt::view_t<T>(ten);
  }

  ~view() { this->clear(); }

  //! Copy Assignment Operator
  view& operator=(const silt::view& rhs) {
    this->clear();
    this->impl = rhs.clone();
    return *this;
  }

  //! Move Assignment Operator
  view& operator=(silt::view &&rhs) {
    this->clear();
    this->impl = rhs.clone();
    return *this;
  }

  //! Polymorphic Strict-Type Cast (Const)
  template<typename T>
  inline const view_t<T> &as() const noexcept {
    return static_cast<view_t<T> &>(*(this->impl));
  }

  //! Polymorphic Strict-Type Cast (Mutable)
  template<typename T>
  inline view_t<T> &as() noexcept {
    return static_cast<view_t<T> &>(*(this->impl));
  }

  //
  // Data Inspection Operations (Type-Deducing)
  //
  
  inline silt::dtype type() const noexcept {
    return this->impl->type();
  }

  silt::slice slice() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().slice();
    });
  }

  silt::host_t host() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().host();
    });
  }

  size_t elem() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().elem();
    });
  }

  void *data() {
    return select(this->type(), [self = this]<typename S>() {
      return (void *)self->as<S>().data();
    });
  }

private:

  void clear() {
    if(this->impl != NULL)
      delete this->impl;
    this->impl = NULL;
  }

  //! Clone the implementation pointer with new
  typedbase* clone() const {
    if(this->impl == NULL) 
      return NULL;
    return select(this->type(), [self = this]<typename S>() -> typedbase* {
      return new silt::view_t<S>(self->as<S>());
    });
  }

  typedbase* impl = NULL; //!< Polymorphic Implementation Pointer

};

}

#endif