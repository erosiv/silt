#pragma once

#include <silt/silt.hpp>
#include <silt/core/tensor_t.hpp>

namespace silt {

//! tensor is a tag-poylymorphic tensor_t wrapper type.
//!
//! This type stores a pointer to a tensor_t<T>, which it
//! can cast to the appropriate type when required.
//!
//! Because of the dll exported type, tensor cannot use
//! std::shared pointer, which is why we do the funny
//! business with the move and copy constructors:
struct EXPORT_SHARED tensor {

  tensor() = default;
  tensor(const silt::dtype type, const silt::shape shape): impl{make(type, shape)} {}
  tensor(const silt::dtype type, const silt::shape shape, const host_t host): impl{make(type, shape, host)} {}

  //! Polymorphic Tensor Copy Constructor
  tensor(const silt::tensor& rhs){
    this->impl = rhs.clone();
  }

  //! Polymorphic Tensor Move Constructor
  tensor(silt::tensor&& rhs) noexcept: impl{rhs.impl} {
    rhs.impl = nullptr;
  }

  //! Strict-Typed Tensor Copy Constructor
  template<typename T>
  tensor(const silt::tensor_t<T> &ten) {
    this->impl = new silt::tensor_t<T>(ten);
  }

  //! Strict-Typed Tensor Move Constructor
  template<typename T>
  tensor(silt::tensor_t<T> &&ten) {
    this->impl = new silt::tensor_t<T>(ten);
  }

  ~tensor() { this->clear(); }

  //! Copy Assignment Operator
  tensor& operator=(const silt::tensor& rhs) {
    this->clear();
    this->impl = rhs.clone();
    return *this;
  }

  //! Move Assignment Operator
  tensor& operator=(silt::tensor &&rhs) noexcept {
    if (this == &rhs) return *this;
    this->clear();
    this->impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //! Polymorphic Strict-Type Cast (Const)
  template<typename T>
  inline const tensor_t<T> &as() const noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  //! Polymorphic Strict-Type Cast (Mutable)
  template<typename T>
  inline tensor_t<T> &as() noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  //
  // Data Inspection Operations (Type-Deducing)
  //

  inline silt::dtype type() const {
    if (this->impl == NULL)
      throw silt::error::uninitialized();
    return this->impl->type();
  }

  silt::shape shape() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().shape();
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

  size_t size() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().size();
    });
  }

  void *data() {
    return select(this->type(), [self = this]<typename S>() {
      return (void *)self->as<S>().data();
    });
  }

  //
  // Shape Manipulation
  //

  void reshape (const int d0 = 1, const int d1 = 1, const int d2 = 1, const int d3 = 1) {
    select(this->type(), [&, self = this]<typename S>() {
      self->as<S>().reshape(d0, d1, d2, d3);
    });
  }

  void flatten() {
    select(this->type(), [self = this]<typename S>() {
      self->as<S>().flatten();
    });
  }

private:

  void clear() {
    if(this->impl != NULL)
      delete this->impl;
    this->impl = NULL;
  }

  //! Make a new Strict-Typed Tensor 
  static typedbase* make(const silt::dtype type, const silt::shape shape, const host_t host = CPU) {
    return select(type, [shape, host]<typename S>() -> typedbase* {
      return new silt::tensor_t<S>(shape, host);
    });
  }

  //! Clone the implementation pointer with new
  typedbase* clone() const {
    if(this->impl == NULL) 
      return NULL;
    return select(this->type(), [self = this]<typename S>() -> typedbase* {
      return new silt::tensor_t<S>(self->as<S>());
    });
  }

  typedbase* impl = NULL; //!< Polymorphic Implementation Pointer
};

}
