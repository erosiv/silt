C++ API
=======

Generated from the ``//!`` doc-comments in ``source/silt`` via Doxygen and
Breathe, one section per header, grouped the same way the source tree already
is (``core/``, ``op/``) rather than by C++ namespace -- namespaces don't
otherwise carve up this library, so file/directory grouping is the natural
fit. Internal helpers (the ``detail`` namespaces, see :doc:`design`) are
excluded throughout.

Core
----

The tensor/view type hierarchy, shape and slice arithmetic, dtype dispatch,
and error types.

.. doxygenfile:: error.hpp

.. doxygenfile:: types.hpp

.. doxygenfile:: vector.hpp

.. doxygenfile:: shape.hpp

.. doxygenfile:: slice.hpp

.. doxygenfile:: view_t.hpp

.. doxygenfile:: view.hpp

.. doxygenfile:: tensor_t.hpp

.. doxygenfile:: tensor.hpp

.. doxygenfile:: memory.hpp

.. doxygenfile:: operation.hpp

Operations
----------

Tensor operations, dispatched through ``silt::op`` -- also the extension
point user code hooks into to add new operations (see :doc:`extending`).

.. doxygenfile:: common.hpp

.. doxygenfile:: gather.hpp

.. doxygenfile:: indexed.hpp

.. doxygenfile:: normal.hpp
