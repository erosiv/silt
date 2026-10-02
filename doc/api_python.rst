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

  t_numpy = t_numpy.to_gpu() # Move data to GPU
  t_torch = t_torch.to_cpu() # Move data to CPU

  t_numpy = t_numpy.torch() # Convert to pytorch
  t_torch = t_torch.numpy() # Convert to numpy

Views and Slicing
-----------------

Indexing a tensor with slices returns a non-owning ``view``. Views broadcast an operation over
a sub-region without copying, e.g. a single channel of an ``[x, y, c]`` tensor:

.. code ::
  python

  color = silt.zeros((512, 512, 3), silt.float32, silt.gpu)
  silt.set_(color[:, :, 0], 1.0)              # Paint the Red Channel

Index Sets, Gather and Scatter
------------------------------

An index set is a compact ``int64`` tensor of flat indices, typically living on the GPU. Selectors
build one from a region or from data values; the ``index_*`` set operations combine them, and
require sorted, unique operands (as the selectors produce; ``index_sort_unique`` establishes this
for any other set):

.. code ::
  python

  s = silt.shape(512, 512)
  a = silt.index_radius(s, [200.0, 256.0], 64.0)
  b = silt.index_box(s, [128.0, 128.0], [256.0, 256.0])
  idx = silt.index_difference(silt.index_union(a, b), silt.index_intersection(a, b))
  rest = silt.index_complement(idx, s)

``indexed_*`` operations and reductions act only on the selected cells, and ``gather`` / ``scatter_``
move them between a tensor and a compact dense tensor. An index set addresses the *logical*
index space of its operand, so slicing a view selects which elements it refers to:

.. code ::
  python

  silt.indexed_add_(color[:, :, 0], 0.5, idx)  # Modify Selected Cells of One Channel
  dense = silt.gather(color[:, :, 0], idx)     # Compact 1D Tensor, In Index Order
  silt.multiply_(dense, 2.0)                   # Any Dense Operation
  silt.scatter_(color[:, :, 0], dense, idx)    # Write Back (Duplicate Indices Race)

Out-of-range indices gather zero and are skipped by scatter and the indexed operations. Over an empty
index set, ``indexed_sum`` is 0 and a floating-point ``indexed_mean`` is NaN; ``indexed_min``,
``indexed_max``, ``indexed_argmin``, ``indexed_argmax`` and an integer ``indexed_mean`` raise.

Sorting and Histograms
----------------------

.. code ::
  python

  ordered = silt.sort(t)               # Sorted Copy (In-Place: silt.sort_(t))
  perm = silt.argsort(t)               # Stable Permutation, an Index Set
  counts = silt.histogram(t, 64)       # Equal-Width Bin Counts (int32) over [min, max]

``histogram`` bins over the closed interval ``[lo, hi]`` (pass ``lo`` / ``hi`` explicitly for a
view); NaNs and out-of-range values are not counted. Where NaNs sort is host-dependent.

Memory Tracking
---------------

silt counts the tensor memory it holds, for every library that links the same ``silt_lib``. The counters are
kept inside ``silt_lib`` and only reached through its exported functions:

.. code ::
  python

  m = silt.memory_usage()
  print(m.cpu_bytes, m.gpu_bytes)              # Live Bytes, per Host
  print(m.cpu_peak, m.gpu_peak)                # High-Water Marks
  silt.memory_reset_peak()                     # Restart the Peak Counters
  free, total = silt.device_memory_info()      # Driver View of the Whole GPU

Only allocations made through silt are counted (tensors, and the scratch buffers of silt's own operations). Temporary
buffers that thrust allocates internally, and memory allocated outside silt, are not; ``device_memory_info`` covers
everything. ``memory_usage().id`` identifies the ``silt_lib`` instance: if two libraries report different ids, each
has loaded its own copy of ``silt_lib`` and they keep separate counts.

API Reference
-------------

Generated from the type stubs (``python/silt/*.pyi``), which every build regenerates from the
compiled extension and the pure-Python layer in ``python/silt/__init__.py``. Functions ending in
``_`` operate in-place; their unsuffixed counterparts return a new tensor.

.. include:: _generated/python_reference.rst
