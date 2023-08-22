#pragma once
#include <stdio.h>
#include <vector>
using namespace std;

struct Tensor {
  int n = 0;
  int ndim = 0;
  int shape[4];
  float *buf = nullptr;
  Tensor(const vector<int> &shape_);
  Tensor(float *data, const vector<int> &shape_);

  ~Tensor();

  void load(const char *filename);
  void save(const char *filename);
  int get_elem();
  void reshape(const vector<int> &shape_);
};
