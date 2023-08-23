#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>

#include "cuda_runtime.h"

#define CHECK_CUDA(call)                                                 \
  do                                                                     \
  {                                                                      \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess)                                          \
    {                                                                    \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

using namespace std;

struct Tensor
{
  int n = 0;
  int ndim = 0;
  int shape[4];
  float *buf = nullptr;
  float *gpu_buf = nullptr;
  Tensor(const vector<int> &shape_);
  Tensor(float *data, const vector<int> &shape_);

  ~Tensor();

  void load(const char *filename);
  void save(const char *filename);
  int get_elem();
  void reshape(const vector<int> &shape_);
  void free_gpu_buf()
  {
    // CHECK_CUDA(cudaFree(gpu_buf));
    CHECK_CUDA(cudaFreeAsync(gpu_buf, 0));
    // cudaFreeAsync
    gpu_buf = nullptr;
  }

  void print_shape()
  {
    std::cout << "========================" << std::endl;
    std::cout << "N: " << shape[0] << std::endl;
    std::cout << "C: " << shape[1] << std::endl;
    std::cout << "H: " << shape[2] << std::endl;
    std::cout << "W: " << shape[3] << std::endl;
    std::cout << "========================" << std::endl;
  }
};
