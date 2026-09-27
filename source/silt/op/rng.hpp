#pragma once

#include <silt/silt.hpp>
#include <silt/core/tensor.hpp>

namespace silt {

//! Seed a Random Number Generator Tensor
EXPORT_SHARED void seed(tensor_t<rng>& buf, const size_t seed, const size_t offset);

//! Generate Uniform Samples from a Random Number Generator Tensor
EXPORT_SHARED tensor_t<float> sample_uniform(tensor_t<rng>& buf);
EXPORT_SHARED tensor_t<float> sample_uniform(tensor_t<rng>& buf, const float min, const float max);

//! Generate Normal Distributed Samples from a Random Number Generator Tensor
EXPORT_SHARED tensor_t<float> sample_normal(tensor_t<rng>& buf);
EXPORT_SHARED tensor_t<float> sample_normal(tensor_t<rng>& buf, const float mean, const float std);

} // namespace silt
