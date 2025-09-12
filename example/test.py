#!/usr/bin/env python

import silt

def main():

  print(silt)
  s = silt.shape(8, 8)
  t = silt.tensor(silt.float32, s)
  print(s)

if __name__ == "__main__":
  main()