#include <cstring>
#include <stdlib.h>

#include "tensor.h"
#include "util.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

using namespace std;

Tensor::Tensor(const vector<int> &shape_) {
  reshape(shape_);
  CHECK_CUDA(cudaMalloc((void **)&gpu_buf, n * sizeof(float)));
}

Tensor::Tensor(float *data, const vector<int> &shape_) {
  reshape(shape_);

  CHECK_CUDA(cudaMalloc((void **)&gpu_buf, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(gpu_buf, data, n * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::~Tensor() {}

void Tensor::load(const char *filename) {
  size_t m;
  buf = (float *)read_binary(filename, &m);
  n = m;
  reshape({n});
}
void Tensor::save(const char *filename) { write_binary(buf, filename, n); }

int Tensor::get_elem() { return n; }

void Tensor::reshape(const vector<int> &shape_) {
  n = 1;
  ndim = shape_.size(); // ndim<=4
  for (int i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}
