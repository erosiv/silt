#!/usr/bin/env python

import silt

def main():

  print(silt)
  s = silt.shape(8, 8)
  print(s, s.elem)

  t = silt.tensor(silt.float32, s)

  print("Tensor Shape:")
  print(s)
  print(s.dim)
  print(s.ext)

  print("Tensor Data:")
  silt.set(t, 0.5)
  print(t.numpy())
  print(t.reshape(2, 4, 4, 2).numpy())

  print("Sliced Data:")

  v = t.view()
  print(v)
  print(v.slice)
  v.reshape(8, 8)
  print(v.slice)

#  u = s[1,4]
#  print(u)
#  print(u.dim)
#  print(u.ext)
#  print(u.stride)
#  print(u.offset)
#  print(u.extlim)
#
#  print(u.index(0))
#  print(u.index(1))
#  print(u.index(2))
#  print(u.index(3))
#  print(u.index(4))
#  print(u.index(5))
#  print(u.index(6))
#  print(u.index(7))

if __name__ == "__main__":
  main()