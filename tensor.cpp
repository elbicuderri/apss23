#include <cstring>
#include <stdlib.h>

#include "tensor.h"
#include "util.h"
using namespace std;

Tensor::Tensor(const vector<int> &shape_) {
  reshape(shape_);
  buf = (float *)malloc(n * sizeof(float));
}

Tensor::Tensor(float *data, const vector<int> &shape_) {
  reshape(shape_);
  buf = (float *)malloc(n * sizeof(float));
  memcpy(buf, data, get_elem() * sizeof(float));
}

Tensor::~Tensor() { free(buf); }

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
