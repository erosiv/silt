#pragma once

#include <silt/silt.hpp>
#include <silt/core/view_t.hpp>
#include <silt/core/tensor.hpp>

namespace silt {

//! view is a tag-poylymorphic view_t wrapper type.
struct EXPORT_SHARED view {

  view() = default;

  //! Polymorphic Tensor Copy Constructor
  view(const silt::view& rhs): _owner{rhs._owner} {
    this->impl = rhs.clone();
  }

  //! Polymorphic Tensor Move Constructor
  view(silt::view&& rhs) noexcept: impl{rhs.impl}, _owner{std::move(rhs._owner)} {
    rhs.impl = nullptr;
  }

  //! Strict-Typed Tensor Copy Constructor.
  //! `owner`, if given, is kept alive for as long as this view is -- see
  //! the non-owning-view lifetime note on view_t in view_t.hpp. Defaults to
  //! nothing, preserving the non-owning behaviour for internal
  //! constructions that don't need it (e.g. views never handed to Python).
  template<typename T>
  view(const silt::view_t<T> &view, silt::tensor owner = silt::tensor()): _owner{std::move(owner)} {
    this->impl = new silt::view_t<T>(view);
  }

  //! Strict-Typed Tensor Move Constructor
  template<typename T>
  view(silt::view_t<T> &&ten, silt::tensor owner = silt::tensor()): _owner{std::move(owner)} {
    this->impl = new silt::view_t<T>(ten);
  }

  ~view() { this->clear(); }

  //! Copy Assignment Operator
  view& operator=(const silt::view& rhs) {
    this->clear();
    this->impl = rhs.clone();
    this->_owner = rhs._owner;
    return *this;
  }

  //! Move Assignment Operator
  view& operator=(silt::view &&rhs) noexcept {
    if (this == &rhs) return *this;
    this->clear();
    this->impl = rhs.impl;
    this->_owner = std::move(rhs._owner);
    rhs.impl = nullptr;
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
  
  inline silt::dtype type() const {
    if (this->impl == NULL) throw silt::error::uninitialized();
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

  // Shape Manipulation Methods

  void reshape(int d0, int d1, int d2, int d3) {
    select(this->type(), [&, self = this]<typename S>() {
      self->as<S>().reshape(d0, d1, d2, d3);
    });
  }

  void index(const int dim, const int offset, const int stride, const int extent) {
    select(this->type(), [&, self = this]<typename S>() {
      self->as<S>().index(dim, offset, stride, extent);
    });
  }

  void reset() {
    select(this->type(), [self = this]<typename S>() {
      self->as<S>().reset();
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

  typedbase* impl = NULL;       //!< Polymorphic Implementation Pointer
  //! Keeps a slice's source tensor alive by refcount (see the constructor
  //! above). A silt::tensor, not std::shared_ptr, for the same DLL-boundary
  //! reason tensor itself avoids std::shared_ptr (see its comment above).
  silt::tensor _owner;

};

}
