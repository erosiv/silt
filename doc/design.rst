Design Details
===================

Hybrid Static Polymorphism with Concepts
----------------------------------------

The silt C++ API is designed to be completely strict-typed, for static compile-time guarantees. At the same time, it provides the option to have polymorphic type deduction where desired. The consequence is a blended structure that lets you decide where you need strict-typing, and where you want runtime deduction.

This is particularly useful in the context of python bindings, as data of various types can be easily constructed and passed into your kernels.

This is achieved using a simple tag-based polymorphic pattern, with a wrapper type ``tensor`` that points to a strict-typed ``tensor_t<T>``, and a templated lambda based type deduction that not only makes runtime deduction syntactically terse but also allows for concept based selection.

How it works
^^^^^^^^^^^^

We will use data-types as our polymorphic example, as it is implemented in silt. The basic runtime polymorphism setup involves a tag enumerator, a mapping struct, a base class and a strict-typed class.

1. The mapping struct maps between C++ strict types and the polymorphic tag enumerator
2. The base class has a virtual destructor and a constexpr virtual function that returns a tag enumerator.
3. The strict-typed class derives from the polymorphic base and overrides the constexpr virtual function.

.. code ::
  C++

  // Polymorphic Tag
  enum dtype {
    NONE,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64
  };

  // Mapping Struct
  template<typename T>
  struct typedesc {
    static constexpr dtype type = NONE;
  };

  // template specializations...
  template<>
  struct typedesc<int> {
    static constexpr const char* name = "int";
    static constexpr dtype type = INT;
    typedef int value_t;
  };

  // Polymorphic Base Type
  struct typedbase {
    virtual ~typedbase() {};
    constexpr virtual dtype type() noexcept {
      return {};
    }
  };

  // Strict-Typed Derived Class
  template<typename T> 
  struct tensor_t: typedbase {
    constexpr dtype type() noexcept {
      return typedesc<T>::type;
    }
  };

Note that the enumerator includes ``NONE`` so that we can detect invalid initializations.

In principle, this is all we need to create the fully polymorphic interface. We can now create the polymorphic wrapper class, which can be converted to and from the strict-typed class if the type is known statically:

.. code ::
  C++

  struct tensor {

    // Constructors for wrapping any tensor_t<T>...

    // Strict-Type Cast
    template<typename T>
    inline const tensor_t<T> &as() const noexcept {
      return static_cast<tensor_t<T> &>(*(this->impl));
    }

    template<typename T>
    inline tensor_t<T> &as() noexcept {
      return static_cast<tensor_t<T> &>(*(this->impl));
    }

    inline dtype type() const noexcept {
      return this->impl->type();
    }

    private:
      typedbase* impl = NULL; // Polymorphic Implementation Pointer

  }

If the strict-type is not known statically, we can use the overriden virtual tag function to select the type at runtime with a single switch statement. This is particularly convenient with templated lambdas:

.. code ::

  void runtime_poly_to_static(const tensor& tensor){
  
    select(tensor.type(), [tensor]<typename S>(){
      const auto tensor_t = tensor.as<S>();
      //... do something static ...
    });
  
  }

Note that the very definition of the lambda will instantiate all template paths specified by the select function, and that the entirety of the lambda is strict-typed. Only the one runtime `static_cast` will actually be executed and the static path selected based on the switch statement.

It is possible to restrict the path instantiation further using concepts by defining an additional concept that asks whether another concept generates a valid evaluatable expression ("matches_lambda"), and running an if constexpr on that concept. This is necessary because otherwise the compilation would fail. The consequence is that each enumerator has to duplicate this code once - it doesn't appear to be possible without that.

.. code ::

  template<typename T, typename F, typename... Args>
  concept matches_lambda = requires(F lambda, Args &&...args) {
    { lambda.template operator()<T>(std::forward<Args>(args)...) };
  };

  template<typename F, typename... Args>
  auto select(const dtype type, F lambda, Args &&...args) {

    switch (type) {
    case dtype::INT:
      if constexpr (matches_lambda<int, F, Args...>) {
        return lambda.template operator()<int>(std::forward<Args>(args)...);
      } else {
        throw type_op_error<int, F>(lambda);
      }
      break;
    case dtype::FLOAT32:
      if constexpr (matches_lambda<float, F, Args...>) {
        return lambda.template operator()<float>(std::forward<Args>(args)...);
      } else {
        throw:type_op_error<float, F>(lambda);
      }
      break;
    case dtype::FLOAT64:
      if constexpr (matches_lambda<double, F, Args...>) {
        return lambda.template operator()<double>(std::forward<Args>(args)...);
      } else {
        throw type_op_error<double, F>(lambda);
      }
      break;
    default:
      throw std::invalid_argument("type not supported");
    }

  }

Ultimately, this allows us to write strict-typed operations that only work for tensors that satisfy specific concepts, that are fully compiled with strict types, but call them using runtime polymorphic concept matching:

.. code ::
  C++

  template<std::floating_point T>
  void my_floating_point_operation(tensor_t<T>& tensor);

  // won't compile: some switch paths don't match concept
  void interface_func(tensor& tensor){
    select(tensor.type(), [tensor]<typename S>(){
      my_floating_point_operation(tensor.as<S>());
    });
  }

  // will compile! mismatched types throw runtime error.
  void interface_func(tensor& tensor){
    select(tensor.type(), [tensor]<std::floating_point S>(){
      my_floating_point_operation(tensor.as<S>());
    });
  }