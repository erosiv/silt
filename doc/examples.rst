Examples
========

Short, self-contained snippets for each part of silt's API.
Tensors default to CPU; the sparse and RNG sections require CUDA.

Tensor Construction
--------------------

.. code::
  python

  import numpy as np
  import silt

  s = silt.shape(512, 512)                      # up to 4 dimensions
  t = silt.tensor(silt.float32, s)               # host defaults to CPU
  t = silt.tensor(silt.float32, s, silt.gpu)     # explicit host

  z = silt.zeros((256, 256))                     # convenience constructors
  o = silt.ones((256, 256), dtype=silt.float64)
  f = silt.full((256, 256), 3.5)
  e = silt.empty((256, 256))                     # uninitialized
  l = silt.like(t)                               # same shape/dtype/host as t

  a = silt.arange(10)                            # 1D, 0..9
  r = silt.linspace(0.0, 1.0, 100)               # 1D, 100 values in [0, 1]

  n = silt.tensor.from_numpy(np.zeros((64, 64), dtype=np.float32))
  # silt.tensor.from_torch(torch_tensor) works the same way

Device Placement & Interop
----------------------------

.. code::
  python

  t = silt.zeros((512, 512))       # CPU by default

  t.to_gpu()                       # move in place, no-op if already on GPU
  t.to_cpu()
  t.to(silt.gpu)                   # equivalent to to_gpu()

  c = t.copy_to()                  # independent copy, same host as t
  c_gpu = t.copy_to(silt.gpu)      # independent copy, explicit host

  arr = t.numpy()                  # CPU-only, always copies
  tens = t.torch()                 # either host, always copies

  silt.synchronize()               # block for outstanding GPU work; raises on error

Common Operations
-------------------

In-place ops carry a trailing underscore; each has a pure-Python out-of-place
counterpart with the same name minus the underscore.

.. code::
  python

  t = silt.zeros((4, 4))
  silt.set_(t, 1.0)
  silt.add_(t, 2.0)                # scalar
  silt.multiply_(t, 3.0)
  silt.divide_(t, 2.0)
  silt.clamp_(t, 0.0, 1.0)         # float32 only
  silt.minimum_(t, 0.5)
  silt.maximum_(t, 0.2)

  a = silt.ones((4, 4))
  b = silt.full((4, 4), 10.0)
  silt.add_(a, b)                  # tensor-tensor, same shape/dtype/host
  silt.mix_(a, b, 0.25)            # lerp(a, b, 0.25), written into a

  # out-of-place: input is left untouched, result is a new tensor
  result = silt.add(a, b)
  result = silt.multiply(a, 2.0)
  result = silt.mix(a, b, 0.5)
  result = silt.clamp(a, 0.0, 1.0)

Slicing & Views
-----------------

Indexing a tensor returns a ``silt.view`` -- a lightweight window into the
same buffer. A trailing comma is required for a 1D index: ``t[a:b:c]`` alone
passes a bare slice object, not a tuple.

.. code::
  python

  t = silt.zeros((10,))
  v = t[2:8:1,]                     # one slice spec per dimension
  silt.set_(v, 1.0)                 # writes through to t

  t2 = silt.zeros((8, 8))
  row = t2[3, 0:8:1]                 # mixed single index + range
  silt.set_(row, 5.0)

  print(v.slice.offset, v.slice.stride, v.slice.extent)

Reductions
------------

Whole-buffer reductions over dense tensors. ``min``/``max`` skip NaNs; the
others do not.

.. code::
  python

  t = silt.tensor.from_numpy(np.array([3.0, -2.0, 5.0], dtype=np.float32))

  silt.sum(t)
  silt.mean(t)
  silt.min(t)
  silt.max(t)
  silt.var(t)
  silt.std(t)
  silt.argmin(t)                   # flat index of the first minimum
  silt.argmax(t)

Sparse (Indexed) Operations
------------------------------

An index set is a 1D ``int64`` tensor of flat indices, built by compaction on
the GPU. Indexed ops read and write only the selected cells; everything else
is left untouched.

.. code::
  python

  s = silt.shape(64, 64)

  idx = silt.index_radius(s, [32.0, 32.0], 10.0)        # circular region
  idx = silt.index_box(s, [10.0, 10.0], [20.0, 20.0])   # axis-aligned box
  idx = silt.index_polygon(s, [(0, 0), (10, 0), (10, 10), (0, 10)])

  data = silt.zeros(s, host=silt.gpu)
  idx = silt.index_range(data, 0.2, 0.8)                # value-based, closed interval
  idx = silt.index_greater(data, 0.5)
  idx = silt.index_lesser(data, 0.5)
  idx = silt.index_match(data, 0.0)                     # exact value

  sl = silt.slice(64, 64)
  sl.index(0, 10, 1, 20)                                # rows 10..29
  idx = silt.index_slice(sl)                            # slice -> index set

.. code::
  python

  t = silt.zeros(s, host=silt.gpu)
  other = silt.full(s, 2.0, host=silt.gpu)

  silt.indexed_set(t, 1.0, idx)
  silt.indexed_add_(t, 1.0, idx)          # scalar
  silt.indexed_add_(t, other, idx)        # tensor-tensor
  silt.indexed_multiply_(t, 2.0, idx)
  silt.indexed_divide_(t, 2.0, idx)
  silt.indexed_mix_(t, other, idx, 0.5)

  silt.indexed_sum(t, idx)
  silt.indexed_mean(t, idx)
  silt.indexed_min(t, idx)
  silt.indexed_max(t, idx)
  silt.indexed_argmin(t, idx)             # flat index into t
  silt.indexed_argmax(t, idx)

Random Number Generation
---------------------------

RNG tensors are GPU-only (curand runs on device).

.. code::
  python

  rng = silt.tensor(silt.rng, silt.shape(4096), silt.gpu)
  silt.seed(rng, 42, 0)                       # seed, offset

  u = silt.sample_uniform(rng)                # [0, 1)
  u2 = silt.sample_uniform(rng, -1.0, 1.0)
  n = silt.sample_normal(rng)                 # mean 0, std 1
  n2 = silt.sample_normal(rng, 5.0, 2.0)

  rng2 = silt.rand(silt.shape(4096), seed_value=42)   # allocate + seed in one call
