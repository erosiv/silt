#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>
#include <silt/op/histogram.hpp>

#include <algorithm>
#include <cmath>

namespace silt {

namespace detail {

//! Bin of `x` in [lo, hi] with `scale = bins / (hi - lo)`, or -1 if `x` is
//! outside the range or NaN. The last bin is closed.
GPU_ENABLE inline int histogram_bin(const double x, const double lo, const double hi, const double scale, const int bins) {
  if (!(x >= lo && x <= hi))
    return -1;
  const int bin = (int)((x - lo) * scale);
  return bin < bins ? bin : bins - 1;
}

//! Largest bin count privatized in shared memory per block.
constexpr int histogram_shared_bins = 8192;

// Grid-stride loop, with per-block private counts in shared memory when they
// fit -- contention then stays local to the block.
template<typename C>
__global__ void histogram_gpu(const C data, tensor_t<int> hist, const int bins, const double lo, const double hi, const double scale, const bool shared) {

  extern __shared__ int local[];
  int* counts = shared ? local : hist.data();

  if (shared) {
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
      local[b] = 0;
    __syncthreads();
  }

  const int64_t first = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t n = first; n < (int64_t)data.elem(); n += stride) {
    const int bin = histogram_bin((double)data[n], lo, hi, scale, bins);
    if (bin >= 0)
      atomicAdd(counts + bin, 1);
  }

  if (shared) {
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
      if (local[b] != 0)
        atomicAdd(hist.data() + b, local[b]);
  }
}

template<typename C>
__host__ void histogram_cpu(const C data, tensor_t<int> hist, const int bins, const double lo, const double hi, const double scale) {
  for (int64_t n = 0; n < (int64_t)data.elem(); ++n) {
    const int bin = histogram_bin((double)data[n], lo, hi, scale, bins);
    if (bin >= 0)
      ++hist[bin];
  }
}

template<typename C>
tensor_t<int> histogram_dispatch(const C data, const int bins, const typename C::val_t lo, const typename C::val_t hi) {

  if (bins < 1)
    throw std::invalid_argument("histogram requires at least one bin");
  if (!(std::isfinite((double)lo) && std::isfinite((double)hi) && (double)lo < (double)hi))
    throw std::invalid_argument("histogram requires a finite range with lo < hi");

  tensor_t<int> hist(silt::shape(bins), data.host());
  silt::set(hist, 0);

  const double scale = (double)bins / ((double)hi - (double)lo);

  if (data.elem() == 0)
    return hist;

  if (data.host() == silt::host_t::CPU) {
    histogram_cpu(data, hist, bins, (double)lo, (double)hi, scale);
  }

  else if (data.host() == silt::host_t::GPU) {
    const bool shared = bins <= histogram_shared_bins;
    const int64_t blocks = std::min<int64_t>(block(data.elem(), 512), 1024);
    const size_t bytes = shared ? (size_t)bins * sizeof(int) : 0;
    histogram_gpu<<<blocks, 512, bytes>>>(data, hist, bins, (double)lo, (double)hi, scale, shared);
    gpuErrchk(cudaGetLastError());
  }

  return hist;
}

} // namespace detail

template<typename T>
tensor_t<int> histogram(const tensor_t<T> data, const int bins, const T lo, const T hi) {
  return detail::histogram_dispatch(data, bins, lo, hi);
}

template<typename T>
tensor_t<int> histogram(const view_t<T> data, const int bins, const T lo, const T hi) {
  return detail::histogram_dispatch(data, bins, lo, hi);
}

template EXPORT_SHARED silt::tensor_t<int> silt::histogram<int>(const silt::tensor_t<int> data, const int bins, const int lo, const int hi);
template EXPORT_SHARED silt::tensor_t<int> silt::histogram<float>(const silt::tensor_t<float> data, const int bins, const float lo, const float hi);
template EXPORT_SHARED silt::tensor_t<int> silt::histogram<double>(const silt::tensor_t<double> data, const int bins, const double lo, const double hi);

template EXPORT_SHARED silt::tensor_t<int> silt::histogram<int>(const silt::view_t<int> data, const int bins, const int lo, const int hi);
template EXPORT_SHARED silt::tensor_t<int> silt::histogram<float>(const silt::view_t<float> data, const int bins, const float lo, const float hi);
template EXPORT_SHARED silt::tensor_t<int> silt::histogram<double>(const silt::view_t<double> data, const int bins, const double lo, const double hi);

} // end of namespace silt
