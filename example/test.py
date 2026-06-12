#!/usr/bin/env python

import silt

def main():

  print(silt)

  print("Tensor Shape:")
  s = silt.shape(4, 4)
  print(s, s.elem, s.dim, s.ext)

  print("Tensor:")
  t = silt.tensor(silt.float32, s)
  t.gpu()
  silt.set(t, 0.5)
  print(t.cpu().numpy())

  print("Tensor View:")
  t.gpu()
  v = t[0, :]
  print(v)
  print(v.slice.offset)
  print(v.slice.stride)
  print(v.slice.extent)
  print(v.elem)

  silt.set(v, 0)
  print(t.cpu().numpy())

if __name__ == "__main__":
  main()