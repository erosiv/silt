Python API
===============

`silt` has two primary types: `shape` and `tensor`. Shape can have between 1 and 4 dimensions, while the tensor type supports multiple strict data-types and can live on the CPU or the GPU:

.. code ::
  python

  s = silt.shape(512, 512)                    # 2D Tensor Shape
  t = silt.tensor(silt.float32, s, silt.gpu)  # Strict-Typed GPU Tensor
  silt.set_(t, 0.0)                           # Set Data to Zeros (In-Place)


`silt` converts data to and from pytorch and numpy, and provides a simple device
upload / download interface. Note that these conversions currently **copy** the data:

.. code ::
  python

  t_numpy = silt.tensor.from_numpy(np.full((512, 512), 0.0, dtype=np.float32))                          # CPU Tensor
  t_torch = silt.tensor.from_torch(torch.full((512, 512), 0.0, dtype=torch.float32, device="cuda"))  # GPU Tensor

  t_numpy = t_numpy.gpu() # Move data to GPU
  t_torch = t_torch.cpu() # Move data to CPU

  t_numpy = t_numpy.torch() # Convert to pytorch
  t_torch = t_torch.numpy() # Convert to numpy

API Reference
-------------

Generated from the built extension module (docstrings come from nanobind's
``.def(...)`` bindings, or its own auto-generated signature when none is
given). ``:imported-members:`` is needed because ``python/silt/__init__.py``
re-exports the extension's contents via ``from .silt import *`` -- without
it, autodoc treats everything as "imported" rather than defined here and
skips it. Requires ``silt`` to be importable in whatever Python environment
runs ``sphinx-build`` (i.e. installed via ``pip install -e .`` first) --
unlike the C++ reference, this reads the real module, not source text.

.. automodule:: silt
   :members:
   :undoc-members:
   :imported-members:
   :show-inheritance:
