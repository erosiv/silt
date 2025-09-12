#ifndef MYPROJECT_CU
#define MYPROJECT_CU

#include <iostream>
#include <my_cuda_lib/my_header.hpp>

namespace my_project {

template<typename T>
__global__ void __set(T* target, const T value, const size_t N){
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;
    target[n] = value;
}

template<typename T>
__global__ void __add(T* target, const T* source, const size_t N){
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;
    target[n] += source[n];
}

void my_func() {

  std::cout << "Hello CUDA!" << std::endl;
  const size_t N = 1024;

  float* a_device;
  float* b_device;
  cudaMalloc((void**)&a_device, sizeof(float)*N);
  cudaMalloc((void**)&b_device, sizeof(float)*N);

  __set<<<1, N>>>(a_device, 1.0f, N);
  __set<<<1, N>>>(b_device, 2.0f, N);
  __add<<<1, N>>>(a_device, b_device, N);

  float* r_host = (float*)malloc(sizeof(float) * N);
  cudaMemcpy(r_host, a_device, sizeof(float)*N, cudaMemcpyDeviceToHost);

  for(size_t n = 0; n < N; ++n){
      if(r_host[n] != 3.0){
          std::cout << "Invalid Add" << std::endl;
          return;
      }
  }

  cudaFree(a_device);
  cudaFree(b_device);
  free(r_host);

  std::cout << "Success" << std::endl;

}

}

#endif