#ifndef SILT_OP_COMMON_CU
#define SILT_OP_COMMON_CU
#define HAS_CUDA

#include <silt/op/common.hpp>
#include <silt/op/gather.hpp>
#include <silt/core/error.hpp>
#include <silt/core/operation.hpp>
#include <iostream>

namespace silt {

//
// Unary Operations
//

template<typename T>
void set(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template void silt::set<int>   (silt::tensor_t<int> lhs,     const int rhs);
template void silt::set<float> (silt::tensor_t<float> lhs,   const float rhs);
template void silt::set<double>(silt::tensor_t<double> lhs,  const double rhs);

template<typename T>
void add(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template void silt::add<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::add<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::add<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void multiply(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

template void silt::multiply<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::multiply<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::multiply<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void divide(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a / rhs;
  });
}

template void silt::divide<int>   (silt::tensor_t<int> buffer,    const int val);
template void silt::divide<float> (silt::tensor_t<float> buffer,  const float val);
template void silt::divide<double>(silt::tensor_t<double> buffer, const double val);

template<typename T>
void clamp(silt::tensor_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE (const T a){
    return glm::clamp(a, min, max);
  });
}

template void silt::clamp<int>   (silt::tensor_t<int> buffer,    const int min, const int max);
template void silt::clamp<float> (silt::tensor_t<float> buffer,  const float min, const float max);
template void silt::clamp<double>(silt::tensor_t<double> buffer, const double min, const double max);

//
// Binary Operations
//

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template void silt::set<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::set<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::set<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

template<typename T>
tensor_t<T> clone(const tensor_t<T> rhs) {
  tensor_t<T> lhs(rhs.shape(), silt::GPU);
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
  return lhs;
}

template tensor_t<int>    silt::clone<int>   (const silt::tensor_t<int> rhs);
template tensor_t<float>  silt::clone<float> (const silt::tensor_t<float> rhs);
template tensor_t<double> silt::clone<double>(const silt::tensor_t<double> rhs);

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template void silt::add<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::add<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::add<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
  });
}

template void silt::multiply<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::multiply<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::multiply<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);

template<typename T>
void divide(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a / b;
  });
}

template void silt::divide<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs);
template void silt::divide<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs);
template void silt::divide<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs);


template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w) {
  op::binop_inplace(lhs, rhs, [w] GPU_ENABLE (const T a, const T b){
    return (1.0f - w) * a + w * b;
  });
}

template void silt::mix<int>   (silt::tensor_t<int> buffer,     const silt::tensor_t<int> rhs, const float w);
template void silt::mix<float> (silt::tensor_t<float> buffer,   const silt::tensor_t<float> rhs, const float w);
template void silt::mix<double>(silt::tensor_t<double> buffer,  const silt::tensor_t<double> rhs, const float w);

//
// Setting Kernels
//

template<typename T>
__global__ void _set(silt::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = start + n*step;
  if(i >= stop) return;
  lhs[i] = val;
}

template<typename T>
void set_impl(silt::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  int thread = 1024;
  int elem = (stop - start + step - 1)/step;
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(lhs, val, start, stop, step);
}

template void set_impl<int>   (silt::tensor_t<int> buffer,    const int val, size_t start, size_t stop, size_t step);
template void set_impl<float> (silt::tensor_t<float> buffer,  const float val, size_t start, size_t stop, size_t step);
template void set_impl<double>(silt::tensor_t<double> buffer, const double val, size_t start, size_t stop, size_t step);

//
// RNG Kernels
//

__global__ void __seed(tensor_t<rng> buf, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

void seed(tensor_t<rng>& buf, const size_t seed, const size_t offset){
  __seed<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  cudaDeviceSynchronize();
}

// Uniform Sampling

__global__ void __sample_uniform(tensor_t<rng> buf, tensor_t<float> sample, const float min, const float max) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  sample[n] = min + curand_uniform(&buf[n])*(max - min);
}

tensor_t<float> sample_uniform(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  __sample_uniform<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  return sample;
}

tensor_t<float> sample_uniform(tensor_t<rng>& buf, const float min, const float max) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  __sample_uniform<<<block(buf.elem(), 512), 512>>>(buf, sample, min, max);
  return sample;
}

// Normal Distribution Sampling

__global__ void __sample_normal(tensor_t<rng> buf, tensor_t<float> sample, const float mean, const float std) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  sample[n] = mean + std * curand_normal(&buf[n]);
}

tensor_t<float> sample_normal(tensor_t<rng>& buf) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  __sample_normal<<<block(buf.elem(), 512), 512>>>(buf, sample, 0.0f, 1.0f);
  return sample;
}

tensor_t<float> sample_normal(tensor_t<rng>& buf, const float mean, const float std) {
  auto sample = tensor_t<float>(buf.shape(), silt::GPU);
  __sample_normal<<<block(buf.elem(), 512), 512>>>(buf, sample, mean, std);
  return sample;
}

//
// Resizing Kernels
//

template<typename T>
__global__ void __resize(silt::tensor_t<T> lhs, const silt::tensor_t<T> rhs){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= lhs.elem()){
    return;
  }

  // Normalize Coordinates in Target Frame
  const shape out = silt::shape(lhs.shape()[1], lhs.shape()[0]);
  const ivec2 ipos = out.unflatten(n);
  const vec2 fpos = vec2(ipos)/vec2(out[0]-1, out[1]-1);
  
  // Unnormalize in Source Frame
  const shape in = silt::shape(rhs.shape()[1], rhs.shape()[0]);
  const vec2 npos = fpos * vec2(in[0]-1, in[1]-1);
  const int i00 = in.flatten(npos + vec2(0, 0));
  const int i01 = in.flatten(npos + vec2(0, 1));
  const int i10 = in.flatten(npos + vec2(1, 0));
  const int i11 = in.flatten(npos + vec2(1, 1));

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

template<typename T>
tensor_t<T> resize(const tensor_t<T> rhs, const shape shape){

  if(rhs.host() != silt::host_t::GPU){
    throw silt::error::mismatch_host(silt::host_t::GPU, rhs.host());
  }

  auto lhs = silt::tensor_t<T>(shape, silt::host_t::GPU);
  __resize<<<block(lhs.elem(), 1024), 1024>>>(lhs, rhs);
  return lhs;

}

template silt::tensor_t<int>    silt::resize<int>   (const silt::tensor_t<int> lhs,     const shape shape);
template silt::tensor_t<float>  silt::resize<float> (const silt::tensor_t<float> lhs,   const shape shape);
template silt::tensor_t<double> silt::resize<double>(const silt::tensor_t<double> lhs,  const shape shape);

//
// Tensor Re-Sampling Procedure
//! \todo add interpolation here.

// template<typename T, typename F>
template<typename T, typename F>
__global__ void __resample(view_t<T> target, const const_view_t<T> source, F f){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < target.elem) {
    f(target, source, n);
  }
}

template<typename T, typename F>
void resample__(view_t<T> target, const const_view_t<T> source, F func) {
  __resample<<<block(target.elem, 512), 512>>>(target, source, func);
}

__device__ bool __isnanv(float val){
  return __isnanf(val);
}

__device__ bool __isnanv(vec3 val){
  return __isnanf(val.x) || __isnanf(val.y) || __isnanf(val.z);
}









template<typename T>
__device__ lerp_t<T> __gather(const silt::const_view_t<T>& view, const silt::shape shape, const vec2 pos) {

  const ivec2 p00 = ivec2(pos) + ivec2(0, 0);
  const ivec2 p01 = ivec2(pos) + ivec2(0, 1);
  const ivec2 p10 = ivec2(pos) + ivec2(1, 0);
  const ivec2 p11 = ivec2(pos) + ivec2(1, 1);
  vec2 w = pos - glm::floor(pos);

  if(pos.x < 0) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.y < 0) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.x > shape[0] - 1) return lerp_t<T>(T{CUDART_NAN_F});
  if(pos.y > shape[1] - 1) return lerp_t<T>(T{CUDART_NAN_F});
  
  int i00 = shape.flatten(p00);
  int i01 = shape.flatten(p01);
  int i10 = shape.flatten(p10);
  int i11 = shape.flatten(p11);

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
void __resample_impl(
  tensor_t<T>& target,       //!< Target Buffer
  const tensor_t<T>& source, //!< Source Buffer
  const vec3 t_scale,       //!< Target World-Space Scale (incl. z)
  const vec3 s_scale,       //!< Source World-Space Scale (incl. z)
  const vec2 pdiff          //!< World-Space Positional Difference
){

  const silt::shape shape_t = silt::shape(target.shape()[1], target.shape()[0]);
  const silt::shape shape_s = silt::shape(source.shape()[1], source.shape()[0]);

  const const_view_t source_v = source.template view<S>();
  view_t target_v = target.template view<S>();

  resample__(target_v, source_v,
    [=] __device__ (view_t<S>& target, const const_view_t<S> source, const unsigned int n){

      vec2 t_pos = shape_t.unflatten(n);
      t_pos.x = shape_t[0] - t_pos.x;
      
      t_pos = t_pos * vec2(t_scale.y, t_scale.x); //!< Target Position in World-Space
      vec2 s_pos = (t_pos - vec2(pdiff.y, pdiff.x)) / vec2(s_scale.y, s_scale.x);          //!< Source Position in Pixel-Space
      s_pos.x = shape_s[0] - s_pos.x;
      
      /*
      Gather Step...
      */

      lerp_t<S> lerp = __gather(source, shape_s, s_pos);
      const S val = lerp.val();
      if(__isnanv(val))
        return;
      target[n] = val;

  });

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
    __resample_impl<T, float>(target, source, t_scale, s_scale, pdiff);
  }

  if(target.shape()[2] == 3) {
    __resample_impl<T, vec3>(target, source, t_scale, s_scale, pdiff);
  }

}

template void silt::resample<int>   (silt::tensor_t<int> lhs,     const silt::tensor_t<int> rhs,     const vec3 t_scale, const vec3 s_scale, const vec2 posdiff);
template void silt::resample<float> (silt::tensor_t<float> lhs,   const silt::tensor_t<float> rhs,   const vec3 t_scale, const vec3 s_scale, const vec2 posdiff);
template void silt::resample<double>(silt::tensor_t<double> lhs,  const silt::tensor_t<double> rhs,  const vec3 t_scale, const vec3 s_scale, const vec2 posdiff);

} // end of namespace silt

#endif