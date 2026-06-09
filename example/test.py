#!/usr/bin/env python

import silt

def main():

  print(silt)
  s = silt.shape(8, 8)
  t = silt.tensor(silt.float32, s)

  print("Tensor Shape:")
  print(s)
  print(s.ext)
  print(s.stride)
  print(s.offset)

  print("Tensor Data:")
  silt.set(t, 0.5)
  print(t.numpy())

if __name__ == "__main__":
  main()