#!/usr/bin/env python
import silt

# Initialize Random Number Generator State Tensor
rng = silt.tensor(silt.rng, silt.shape(1024), silt.gpu)
silt.seed(rng, 0, 0)

# Uniform Distribution Sample
sample = silt.sample_uniform(rng, 0, 1)
print(sample.cpu().numpy())

# Normal Sample Distribution
sample = silt.sample_normal(rng, 0, 1)
print(sample.cpu().numpy())