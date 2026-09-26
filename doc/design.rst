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

Contiguous Storage, Strided Views
----------------------------------

``tensor_t<T>`` only ever holds a ``shape`` -- a dense, row-major extent with
no notion of stride. Its element access is a flat, unconditional index into
the owned buffer:

.. code ::
  C++

  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }

There is no way to construct a non-contiguous ``tensor_t<T>``. Strided access
-- the result of slicing along a dimension with an offset, stride or a
sub-extent -- lives entirely in a separate type, ``view_t<T>``, whose index
operator instead goes through ``slice::transform()``:

.. code ::
  C++

  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[this->_slice.transform(index)];
  }

So ``tensor_t<T>`` and ``view_t<T>`` are not two ways of writing the same
thing: ``tensor_t<T>`` is the type that *owns* memory and *guarantees*
contiguity, and ``view_t<T>`` is the type that can describe an arbitrary
strided sub-region but never owns anything. Slicing a tensor
(``tensor[...]`` in Python, or ``.view<T>()`` in C++) produces a
``view``/``view_t<T>``, never another ``tensor``/``tensor_t<T>`` -- there is
no operation in silt that takes a strided region and hands you back
something claiming to be densely packed.

This is also why numpy/torch interop copies rather than aliasing memory
(the "Note that these conversions currently copy the data" callout in
:doc:`usage` and the README): a numpy or torch array can be non-contiguous
(a transpose, a stride-2 slice, ...), and silt has no owning, contiguous-
guaranteed type that could wrap such a thing without either copying it into
one or exposing an owning type that lies about its own invariant. Wrapping
non-contiguous foreign memory as a *view* instead of a *tensor* would sidestep
that problem, and is roughly the shape zero-copy interop would need to take,
should it ever be implemented.

Explicit Host Placement
------------------------

silt never migrates data between CPU and GPU on your behalf. ``operator[]``
does not check ``host()`` and dispatch accordingly -- it unconditionally
dereferences ``_data``, exactly as shown above. Indexing a GPU tensor from
host code is therefore not a silt error, it is an illegal host dereference of
a device pointer, the same as it would be with a raw ``cudaMalloc``'d buffer.

The only host-crossing operations are the explicit ones: ``.cpu()``/``.gpu()``
(in-place, via ``tensor_t<T>::transfer()``, which also backs ``clone()`` --
all four host/device copy directions go through the one function) and the
numpy/torch conversion functions. This mirrors PyTorch's own
``.cpu()``/``.cuda()`` convention deliberately, for the same reason
``silt.synchronize()`` mirrors ``torch.cuda.synchronize()``: someone moving
between the two libraries should not have to learn a second mental model for
where data lives and when it moves.

Manual Reference Counting, Not ``shared_ptr``
-----------------------------------------------

``tensor_t<T>`` shares its buffer across copies via a hand-rolled
``size_t* _refs`` counter, incremented and decremented by hand in every copy
constructor, copy-assignment operator and destructor -- not
``std::shared_ptr``, which would do the same bookkeeping automatically and
with less code.

The reason is that ``tensor_t<T>`` and ``view_t<T>`` are passed *by value*
into ``__global__`` kernels:

.. code ::
  C++

  template<typename T, typename F>
  __global__ void uniop_inplace_gpu(tensor_t<T> lhs, F func);

nvcc has to be able to generate a device-side copy and, implicitly, a
device-side destructor call for every by-value kernel parameter.
``std::shared_ptr``'s destructor is not trivial (it's a host-only atomic
decrement into a host allocator) and cannot be instantiated for ``__device__``
code, so a ``tensor_t<T>`` holding one would fail to compile as a kernel
parameter at all. A plain ``size_t*`` with hand-written, host-only
increment/decrement logic sidesteps this entirely: the pointer itself
copies trivially by value (fine on the device, which never actually
dereferences it), and the refcounting logic that mutates what it points to
only ever runs on the host, in ordinary C++ constructors/destructors that
are never instantiated for ``__device__``.

This is also why the polymorphic, Python-facing ``silt::view`` holds a
``std::shared_ptr<void>`` to keep its source tensor alive, while ``view_t<T>``
does not: ``silt::view`` is a host-only type with no kernel-parameter
obligations, so it can afford a real ``shared_ptr``, while ``view_t<T>`` --
the type actually passed into kernels -- cannot.

The ``detail`` Namespace Convention
--------------------------------------

Functions and kernels that exist only to implement something else --
never called from outside their own translation unit or from Python --
live in a nested ``namespace detail { ... }``, rather than being named with
a leading double underscore (the codebase's older convention). This is a
correctness fix, not a style preference: `[lex.name]
<https://eel.is/c++draft/lex.name#3>`_ reserves any identifier containing a
double underscore to the implementation *at any scope*, not only at global
scope, so a name like ``op::__uniop_inplace_gpu`` was already undefined
behaviour to declare, namespace or not. Renaming to ``op::detail::
uniop_inplace_gpu`` fixes the reserved-identifier problem and, as a side
effect, gives every internal helper a consistent, greppable home:
``silt::detail`` and ``silt::op::detail`` should never appear in anything a
caller (C++ or Python) is meant to depend on, and the Doxygen configuration
(``EXCLUDE_SYMBOLS`` in ``doc/Doxyfile``) keeps them out of :doc:`api_cpp`
on that basis. A single leading underscore is *not* subject to the same
rule -- it is only reserved at global scope -- so a name like ``silt::_set``
inside a namespace is legal and was deliberately left alone rather than
renamed for its own sake.

If you add an internal helper, it belongs in ``detail`` inside whichever
namespace it supports (``silt::detail`` for core-layer helpers,
``silt::op::detail`` for kernel-layer helpers), not spelled with a
double underscore, and not left bare in the surrounding namespace either.

In-Place vs. Out-of-Place Naming
-----------------------------------

Every C++-bound operation is in-place: it writes into a tensor you already
own, and its Python binding carries a trailing underscore (``set_``, ``add_``,
``multiply_``, ``divide_``, ``mix_``, ``clamp_``) -- the same convention
PyTorch uses for its mutating methods. The out-of-place counterparts
(``silt.add``, ``silt.multiply``, ...) exist *only* in
``python/silt/__init__.py``, as plain Python functions that clone the first
argument and then call the in-place form on the clone:

.. code ::
  python

  def add(lhs, rhs):
      result = clone(lhs)
      add_(result, rhs)
      return result

There is no separate C++ kernel for "out-of-place add" and there is not meant
to be one: "out-of-place" is definitionally "allocate, then do the in-place
thing," so writing it in Python is not a shortcut being taken for now, it is
the whole implementation. A new operation added to silt should follow the
same split: write the in-place kernel and its C++/Python binding, and let the
out-of-place form (if you need one at all) be a two-line Python wrapper
alongside the existing ones, not a second kernel.
