C++ API
=======

Generated from the ``//!`` doc-comments in ``source/silt`` via Doxygen and
Breathe. Internal helpers (the ``detail`` namespaces introduced to keep
implementation details off the public surface) are excluded -- see
:doc:`design` for the reasoning behind the types below, and :doc:`extending`
for how to add a new operation.

Tensor operations are dispatched through ``silt::op``, rendered below as a
nested namespace -- it is also the extension point user code hooks into to
add new operations (see :doc:`extending`).

.. doxygennamespace:: silt
   :members:
   :undoc-members:
