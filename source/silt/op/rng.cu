#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/rng.hpp>

namespace silt {
namespace detail {

__global__ void seed_kernel(tensor_t<rng> buf, const size_t seed, const size_t offset) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

} // namespace detail

void seed(tensor_t<rng>& buf, const size_t seed, const size_t offset) {
  detail::seed_kernel<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

// Uniform Sampling

namespace detail {

__global__ void sample_uniform_kernel(tensor_t<rng> buf, tensor_t<float> sample, const float min, const float max) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= buf.elem()) return;
  sample[n] = min + curand_uniform(&buf[n]) * (max - min);
}

} // namespace detail

tensor_t<float> sample_uniform(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_uniform_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  gpuErrchk(cudaGetLastError());
  return sample;
}

tensor_t<float> sample_uniform(tensor_t<rng>& buf, const float min, const float max) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_uniform_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, min, max);
  gpuErrchk(cudaGetLastError());
  return sample;
}

// Normal Distribution Sampling

namespace detail {

__global__ void sample_normal_kernel(tensor_t<rng> buf, tensor_t<float> sample, const float mean, const float std) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= buf.elem()) return;
  sample[n] = mean + std * curand_normal(&buf[n]);
}

} // namespace detail

tensor_t<float> sample_normal(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_normal_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  gpuErrchk(cudaGetLastError());
  return sample;
}

tensor_t<float> sample_normal(tensor_t<rng>& buf, const float mean, const float std) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_normal_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, mean, std);
  gpuErrchk(cudaGetLastError());
  return sample;
}

} // end of namespace silt
