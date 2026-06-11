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
  v.index(1, 2, 2, 4)

  silt.set(v, 0)
  print(t.numpy())

if __name__ == "__main__":
  main()