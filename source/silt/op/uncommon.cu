#include <silt/op/common.hpp>
#include <silt/op/gather.hpp>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <iostream>

namespace silt {

//
// Setting Kernels
//

template<typename T>
__global__ void _set(silt::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t i = start + n*step;
  if(i >= stop) return;
  lhs[i] = val;
}

template<typename T>
void set_impl(silt::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  if(stop > lhs.elem())
    throw silt::error::out_of_bounds(stop, lhs.elem());
  int thread = 1024;
  size_t elem = (stop - start + step - 1)/step;
  size_t block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(lhs, val, start, stop, step);
}

template EXPORT_SHARED void set_impl<int>   (silt::tensor_t<int> buffer,    const int val, size_t start, size_t stop, size_t step);
template EXPORT_SHARED void set_impl<float> (silt::tensor_t<float> buffer,  const float val, size_t start, size_t stop, size_t step);
template EXPORT_SHARED void set_impl<double>(silt::tensor_t<double> buffer, const double val, size_t start, size_t stop, size_t step);

//
// RNG Kernels
//

namespace detail {

__global__ void seed_kernel(tensor_t<rng> buf, const size_t seed, const size_t offset) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

}

void seed(tensor_t<rng>& buf, const size_t seed, const size_t offset){
  detail::seed_kernel<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  cudaDeviceSynchronize();
}

// Uniform Sampling

namespace detail {

__global__ void sample_uniform_kernel(tensor_t<rng> buf, tensor_t<float> sample, const float min, const float max) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  sample[n] = min + curand_uniform(&buf[n])*(max - min);
}

}

tensor_t<float> sample_uniform(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_uniform_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  return sample;
}

tensor_t<float> sample_uniform(tensor_t<rng>& buf, const float min, const float max) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_uniform_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, min, max);
  return sample;
}

// Normal Distribution Sampling

namespace detail {

__global__ void sample_normal_kernel(tensor_t<rng> buf, tensor_t<float> sample, const float mean, const float std) {
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  sample[n] = mean + std * curand_normal(&buf[n]);
}

}

tensor_t<float> sample_normal(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_normal_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  return sample;
}

tensor_t<float> sample_normal(tensor_t<rng>& buf, const float mean, const float std) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  detail::sample_normal_kernel<<<block(buf.elem(), 512), 512>>>(buf, sample, mean, std);
  return sample;
}

//
// Resizing Kernels
//

namespace detail {

template<typename T>
__global__ void resize_kernel(silt::tensor_t<T> lhs, const silt::tensor_t<T> rhs){

  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= lhs.elem()){
    return;
  }

  // Normalize Coordinates in Target Frame
  const shape out = silt::shape(lhs.shape()[1], lhs.shape()[0]);
  const ivec2 ipos = out.unflatten(n);
  const vec2 fpos = vec2(ipos)/vec2(out[0]-1, out[1]-1);
  
  // Unnormalize in Source Frame. Flat indices into the (potentially very
  // large) source buffer, so these must not be truncated to `int`.
  const shape in = silt::shape(rhs.shape()[1], rhs.shape()[0]);
  const vec2 npos = fpos * vec2(in[0]-1, in[1]-1);
  const int64_t i00 = in.flatten(npos + vec2(0, 0));
  const int64_t i01 = in.flatten(npos + vec2(0, 1));
  const int64_t i10 = in.flatten(npos + vec2(1, 0));
  const int64_t i11 = in.flatten(npos + vec2(1, 1));

  // Linear Interpolation w. Bounds Handling
  if(in.oob(npos)){
    lhs[n] = T(0);
  } else if(in.oob(npos + vec2(1, 1))){
    lhs[n] = rhs[i00]; 
  } else {
    T v00 = rhs[i00];
    T v01 = rhs[i01];
    T v10 = rhs[i10];
    T v11 = rhs[i11];
    lerp_t lerp(v00, v01, v10, v11, npos - glm::floor(npos));
    lhs[n] = lerp.val();
  }

}

}

template<typename T>
tensor_t<T> resize(const tensor_t<T> rhs, const shape shape){

  if(rhs.host() != silt::host_t::GPU){
    throw silt::error::mismatch_host(silt::host_t::GPU, rhs.host());
  }

  auto lhs = silt::tensor_t<T>(shape, silt::host_t::GPU);
  detail::resize_kernel<<<block(lhs.elem(), 1024), 1024>>>(lhs, rhs);
  return lhs;

}

template EXPORT_SHARED silt::tensor_t<int>    silt::resize<int>   (const silt::tensor_t<int> lhs,     const shape shape);
template EXPORT_SHARED silt::tensor_t<float>  silt::resize<float> (const silt::tensor_t<float> lhs,   const shape shape);
template EXPORT_SHARED silt::tensor_t<double> silt::resize<double>(const silt::tensor_t<double> lhs,  const shape shape);

//
// Tensor Re-Sampling Procedure
//! \todo add interpolation here.

// template<typename T, typename F>
namespace detail {

template<typename T, typename F>
__global__ void resample_kernel(view_t<T> target, const view_t<const T> source, F f){
  const size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if(n < target.elem()) {
    f(target, source, n);
  }
}

template<typename T, typename F>
void resample_launch(view_t<T> target, const view_t<const T> source, F func) {
  resample_kernel<<<block(target.elem(), 512), 512>>>(target, source, func);
}

__device__ bool isnanv(float val){
  return __isnanf(val);
}

__device__ bool isnanv(vec3 val){
  return __isnanf(val.x) || __isnanf(val.y) || __isnanf(val.z);
}

template<typename T>
__device__ lerp_t<T> gather(const silt::view_t<const T>& view, const silt::shape shape, const vec2 pos) {

  const ivec2 p00 = ivec2(pos) + ivec2(0, 0);
  const ivec2 p01 = ivec2(pos) + ivec2(0, 1);
  const ivec2 p10 = ivec2(pos) + ivec2(1, 0);
  const ivec2 p11 = ivec2(pos) + ivec2(1, 1);
  vec2 w = pos - glm::floor(pos);

  if(pos.x < 0) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.y < 0) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.x > shape[0] - 1) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.y > shape[1] - 1) return lerp_t<T>(T{CUDART_NAN_F});
  
  // Flat indices into the source view's (potentially very large) buffer,
  // so these must not be truncated to `int`.
  int64_t i00 = shape.flatten(p00);
  int64_t i01 = shape.flatten(p01);
  int64_t i10 = shape.flatten(p10);
  int64_t i11 = shape.flatten(p11);

  if(pos.x + 1 > shape[0] - 1){ w.x = 0; i10 = 0; i11 = 0; }
  if(pos.y + 1 > shape[1] - 1){ w.y = 0; i01 = 0; i11 = 0; }
  
  const T h00 = view[i00];
  const T h01 = view[i01];
  const T h10 = view[i10];
  const T h11 = view[i11];

  return lerp_t<T>{
    h00, h01,
    h10, h11,
    w
  };

}

template<typename T, typename S>
void resample_impl(
  tensor_t<T>& target,       //!< Target Buffer
  const tensor_t<T>& source, //!< Source Buffer
  const vec3 t_scale,       //!< Target World-Space Scale (incl. z)
  const vec3 s_scale,       //!< Source World-Space Scale (incl. z)
  const vec2 pdiff          //!< World-Space Positional Difference
){

  const silt::shape shape_t = silt::shape(target.shape()[1], target.shape()[0]);
  const silt::shape shape_s = silt::shape(source.shape()[1], source.shape()[0]);

  const view_t source_v = source.template view<S>();
  view_t target_v = target.template view<S>();

  resample_launch(target_v, source_v,
    [=] __device__ (view_t<S>& target, const view_t<const S> source, const size_t n){

      vec2 t_pos = shape_t.unflatten(n);
      t_pos.x = shape_t[0] - t_pos.x;
      
      t_pos = t_pos * vec2(t_scale.y, t_scale.x); //!< Target Position in World-Space
      vec2 s_pos = (t_pos - vec2(pdiff.y, pdiff.x)) / vec2(s_scale.y, s_scale.x);          //!< Source Position in Pixel-Space
      s_pos.x = shape_s[0] - s_pos.x;
      
      /*
      Gather Step...
      */

      lerp_t<S> lerp = gather(source, shape_s, s_pos);
      const S val = lerp.val();
      if(isnanv(val))
        return;
      target[n] = val;

  });

}

}

template<typename T>
void resample(
  tensor_t<T> target,       //!< Target Buffer
  const tensor_t<T> source, //!< Source Buffer
  const vec3 t_scale,       //!< Target World-Space Scale (incl. z)
  const vec3 s_scale,       //!< Source World-Space Scale (incl. z)
  const vec2 pdiff          //!< World-Space Positional Difference
) {

  // Validate Identical Shape
  if(target.shape()[2] != source.shape()[2]){
    throw silt::error::mismatch_size(target.shape()[2], source.shape()[2]);
  }

  // Note: These two scenarios should involve generic vector types instead.

  if(target.shape()[2] == 1) {
    detail::resample_impl<T, float>(target, source, t_scale, s_scale, pdiff);
  }

  if(target.shape()[2] == 3) {
    detail::resample_impl<T, vec3>(target, source, t_scale, s_scale, pdiff);
  }

}

// resample is only ever bound for float (see python/source/op.cpp); the
// int/double instantiations reinterpreted the buffer as float/vec3 regardless
// of T, which silently corrupted their data. Candidate for future removal.
template EXPORT_SHARED void silt::resample<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs,   const vec3 t_scale, const vec3 s_scale, const vec2 posdiff);

} // end of namespace silt
