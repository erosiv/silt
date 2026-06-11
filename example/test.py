#!/usr/bin/env python

import silt

def main():

  print(silt)

  print("Tensor Shape:")
  s = silt.shape(4, 4)
  print(s, s.elem, s.dim, s.ext)

  print("Tensor:")
  t = silt.tensor(silt.float32, s)
  silt.set(t, 0.5)
  print(t.numpy())

  print("Tensor View:")
  v = t.view().reshape(4, 4)
  v.index(0, 0, 2, 2)
  v.index(1, 0, 2, 2)
  print(v.slice.elem)

  print(v.slice.transform(0))
  print(v.slice.transform(1))
  print(v.slice.transform(2))
  print(v.slice.transform(3))

if __name__ == "__main__":
  main()