Python API
===============

`silt` has two primary types: `shape` and `tensor`. Shape can have between 1 and 4 dimensions, while the tensor type supports multiple strict data-types and can live on the CPU or the GPU:

.. code ::
  python

  s = silt.shape(512, 512)                    # 2D Tensor Shape
  t = silt.tensor(s, silt.float32, silt.gpu)  # Strict-Typed GPU Tensor
  silt.set(t, 0.0)                            # Set Data to Zeros


`silt` supports no-copy data wrapping to and from pytorch or numpy, as well as a simple data uploading downloading interface:

.. code ::
  python

  t_numpy = silt.tensor.from_numpy(np.full((512, 512), 0.0, dtype=np.float32))                          # CPU Tensor
  t_torch = silt.tensor.from_torch(torch.full((512, 512), 0.0, dtype=torch.flota32, device=torch.cuda)) # GPU Tensor

  t_numpy = t_numpy.gpu() # Move data to GPU
  t_torch = t_torch.cpu() # Move data to CPU

  t_numpy = t_numpy.torch() # Convert to pytorch
  t_torch = t_torch.numpy() # Convert to numpy