#pragma once

#include <silt/silt.hpp>
#include <silt/core/error.hpp>
#include <silt/core/memory.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/view_t.hpp>

namespace silt {

//! tensor_t<T> is a strict-typed, owning raw-data extent.
//!
//! tensor_t<T> is reference counting with correct move semantics,
//! making copying and moving tensor_t<T> values cheap and easy.
//!
//! tensor_t<T> data can be on the CPU or on the GPU.
//!
template<typename T>
struct tensor_t: typedbase {

  typedef T val_t;

  //
  // Construction and Assignment
  //

  //! Default Empty Constructor
  tensor_t() {
    this->_shape = shape();
    this->_data = NULL;
    this->_refs = NULL;
    this->_host = CPU;
  }

  //! Allocating Constructor
  tensor_t(const shape shape, const host_t host = CPU) {
    this->allocate(shape, host);
  }

  //! Non-Allocating Constructor
  tensor_t(T* data, const shape shape, const host_t host = CPU) {
    this->_data = data;
    this->_refs = NULL;
    this->_shape = shape;
    this->_host = host;
  }

  ~tensor_t() { this->deallocate(); }

  //! Copy Constructor (Reference Increment)
  tensor_t(const tensor_t<T>& other) {
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    if (this->_data != NULL) {
      if (this->_refs != NULL) {
        ++(*this->_refs);
      }
    }
  }

  //! Copy Assignment Operator (Reference Increment)
  tensor_t& operator=(const tensor_t<T>& other) {
    if (this == &other) return *this;
    this->deallocate();
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    if (this->_data != NULL) {
      if (this->_refs != NULL) {
        ++(*this->_refs);
      }
    }
    return *this;
  }

  //! Move Constructor (Reference Steal)
  tensor_t(tensor_t<T>&& other) {
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    other._data = NULL;
    other._refs = NULL;
  }

  //! Move Assginmment Operator (Reference Steal)
  tensor_t& operator=(tensor_t<T>&& other) {
    if (this == &other) return *this;
    this->deallocate();
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    other._data = NULL;
    other._refs = NULL;
    return *this;
  }

  //
  // Data Inspection
  //

  GPU_ENABLE inline silt::shape shape() const { return this->_shape; }
  GPU_ENABLE inline size_t elem() const { return this->_shape.elem(); }            //!< Number of Elements
  GPU_ENABLE inline size_t size() const { return this->elem() * sizeof(T); }       //!< Total Size in Bytes
  GPU_ENABLE inline size_t refs() const { return this->_refs ? *this->_refs : 0; } //!< Reference Count (0 if non-owning)
  GPU_ENABLE inline host_t host() const { return this->_host; }                    //!< Current Device (CPU / GPU)
  GPU_ENABLE inline const T* data() const { return this->_data; }                  //!< Raw Data Pointer (Const)
  GPU_ENABLE inline T* data() { return this->_data; }                              //!< Raw Data Pointer (Mutable)

  //! Type Enumerator Retrieval
  constexpr silt::dtype type() noexcept {
    return silt::typedesc<T>::type;
  }

  //! Const Subscript Operator (Flat)
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }

  //! Non-Const Subscript Operator (Float)
  GPU_ENABLE T& operator[](const size_t index) noexcept {
    return this->_data[index];
  }

  template<typename S>
  GPU_ENABLE view_t<S> view() noexcept {
    return view_t<S>(
        silt::shape(this->size() / sizeof(S)),
        this->host(),
        reinterpret_cast<S*>(this->data())
    );
  };

  template<typename S>
  GPU_ENABLE view_t<const S> view() const noexcept {
    return view_t<const S>(
        silt::shape(this->size() / sizeof(S)),
        this->host(),
        reinterpret_cast<const S*>(this->data())
    );
  };

  // Shape Manipulation

  void reshape(const int d0 = 1, const int d1 = 1, const int d2 = 1, const int d3 = 1) {
    this->_shape.reshape(d0, d1, d2, d3);
  }

  void flatten() {
    this->reshape(this->_shape.elem());
  }

  // Host Manipulation

  void to_cpu(); //!< In-Place Copy Data to the CPU
  void to_gpu(); //!< In-Place Copy Data to the GPU (if available)

  //! Return an independent copy of this tensor's data on `target`.
  //! Covers all four host pairings (CPU/GPU source x CPU/GPU target);
  //! transfer(host()) is a same-host deep copy.
  tensor_t<T> transfer(const host_t target) const;

  size_t* _refs = NULL; //!< Pointer to Reference Count
private:
  //! Device-Aware Allocation and De-Allocation
  void allocate(const silt::shape shape, const host_t host = CPU);
  void deallocate();

  silt::shape _shape; //!< Shape of Data
  host_t _host = CPU; //!< Currently Active Device
  T* _data = NULL;    //!< Raw Data Pointer (Device Agnostic)
};

//
// Member Function Implementations
//

template<typename T>
void silt::tensor_t<T>::allocate(const silt::shape shape, const host_t host) {

  if (shape.elem() == 0)
    throw std::invalid_argument("size must be greater than 0");
  this->_shape = shape;

  if (host == CPU) {
    this->_data = new T[shape.elem()];
  } else if (host == GPU) {
    this->_data = (T*)silt::device_alloc(this->size());
  } else {
    throw std::invalid_argument("device not recognized");
  }

  this->_host = host;
  this->_refs = new size_t(1);
}

template<typename T>
void silt::tensor_t<T>::deallocate() {

  if (this->_refs == NULL)
    return;

  if (*this->_refs == 0)
    return;

  (*this->_refs)--;
  if (*this->_refs > 0)
    return;

  delete this->_refs;
  this->_refs = NULL;

  if (this->_data != NULL) {
    if (this->_host == CPU) {
      delete[] this->_data;
      this->_data = NULL;
      this->_host = CPU;
    }

    if (this->_host == GPU) {
      silt::device_free(this->_data);
      this->_data = NULL;
      this->_host = CPU;
    }
  }
}

template<typename T>
tensor_t<T> silt::tensor_t<T>::transfer(const host_t target) const {

  tensor_t<T> out(this->_shape, target);

  copy_t kind;
  if (this->_host == CPU && target == CPU)
    kind = copy_t::HOST_TO_HOST;
  else if (this->_host == CPU && target == GPU)
    kind = copy_t::HOST_TO_DEVICE;
  else if (this->_host == GPU && target == CPU)
    kind = copy_t::DEVICE_TO_HOST;
  else
    kind = copy_t::DEVICE_TO_DEVICE;

  silt::device_copy(out.data(), this->data(), this->size(), kind);
  return out;
}

template<typename T>
void silt::tensor_t<T>::to_gpu() {

  if (this->_host == GPU)
    return;

  if (this->_data == NULL)
    return;

  if (this->elem() == 0)
    return;

  *this = this->transfer(GPU);
}

template<typename T>
void silt::tensor_t<T>::to_cpu() {

  if (this->_host == CPU)
    return;

  if (this->_data == NULL)
    return;

  if (this->elem() == 0)
    return;

  *this = this->transfer(CPU);
}

} // namespace silt
