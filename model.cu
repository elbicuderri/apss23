#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <float.h>
#include <stdio.h>
#include <time.h>

#include "model.h"
#include "util.h"

#define THREADS_PER_BLOCK 256

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

extern int N;

// class BrainTumorModel(nn.Module):
//
//  def __init__(self):
//      super().__init__()
//      self.conv0 = nn.Sequential(
//          nn.Conv2d(1,128,kernel_size=3),
//          nn.InstanceNorm2d(128),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.conv1 = nn.Sequential(
//          nn.Conv2d(128,256,kernel_size=3),
//          nn.InstanceNorm2d(256),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.linear1 = nn.Linear(62,128)
//      self.linear2 = nn.Linear(128,64)
//      self.flat = nn.Flatten(1)
//      self.linear3 = nn.Linear(1015808,2)
//
//  def forward(self,x):
//      x = self.conv0(x)
//      x = self.conv1(x)
//      x = F.relu(self.linear1(x))
//      x = self.linear2(x)
//      x = self.flat(x)
//      x = self.linear3(x)
//
//      return x

static Tensor *conv0_weight, *conv0_bias, *conv1_weight, *conv1_bias,
    *linear1_weight, *linear1_bias, *linear2_weight, *linear2_bias,
    *linear3_weight, *linear3_bias, *instanceNorm2d0_weight,
    *instanceNorm2d0_bias, *instanceNorm2d1_weight, *instanceNorm2d1_bias;

static Tensor *input, *output, *c1, *i1, *m1, *c2, *i2, *m2, *l1, *l2;

static constexpr int max_batch_per_step = 256;

void initialize_model(const char *parameter_fname)
{
  size_t m; // 2345922
  float *buf = (float *)read_binary(parameter_fname, &m);
  conv0_weight = new Tensor(buf, {128, 1, 3, 3});
  buf += 1152;
  conv0_bias = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_weight = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_bias = new Tensor(buf, {128});
  buf += 128;
  conv1_weight = new Tensor(buf, {256, 128, 3, 3});
  buf += 294912;
  conv1_bias = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_weight = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_bias = new Tensor(buf, {256});
  buf += 256;
  linear1_weight = new Tensor(buf, {62, 128});
  buf += 7936;
  linear1_bias = new Tensor(buf, {128});
  buf += 128;
  linear2_weight = new Tensor(buf, {128, 64});
  buf += 8192;
  linear2_bias = new Tensor(buf, {64});
  buf += 64;
  linear3_weight = new Tensor(buf, {1015808, 2});
  buf += 2031616;
  linear3_bias = new Tensor(buf, {2});
  buf += 2;

  int batch = N;
  if (N > max_batch_per_step)
  {
    batch = 256;
  }

  input = new Tensor({batch, 1, 256, 256});

  c1 = new Tensor({batch, 128, 254, 254});

  i1 = new Tensor({batch, 128, 254, 254});

  m1 = new Tensor({batch, 128, 127, 127});

  c2 = new Tensor({batch, 256, 125, 125});

  i2 = new Tensor({batch, 256, 125, 125});

  m2 = new Tensor({batch, 256 * 62, 62});

  l1 = new Tensor({batch, 256 * 62, 128});

  l2 = new Tensor({batch, 1, 256 * 62 * 64});

  output = new Tensor({batch, 1, 2});
}

// Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// Size of in  = N * C_IN * H_IN * W_IN
// Size of out = N * C_OUT * (H_IN-K+1) * (W_IN-K+1)
// Weight : C_OUT * C_IN * K * K
// Bias : C_OUT
static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d_relu(Tensor *in_t, Tensor *out_t, int kH, int kW);

// InstanceNorm2D
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
// size of in  = N * C * H * W
// size of out = N * C * H * W
// weight : C
// bias : C
static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t);

// Linear
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// size of in  = N * H_IN
// size of out = N * H_OUT
// weight : H_OUT * H_IN
// bias : H_OUT
static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

static void linear_relu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                        Tensor *bias_t);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
// static void relu(Tensor *inout_t);

void model_forward(float *inputN, float *outputN)
{
  int steps = (N + max_batch_per_step - 1) / max_batch_per_step;
  int last_batch = ((N % max_batch_per_step) == 0) ? max_batch_per_step : N;

  for (int idx = 0; idx < steps; idx++)
  {

    int micro_batch = max_batch_per_step;
    if (idx == (steps - 1))
    {
      micro_batch = last_batch;
    }

    if (steps == 1)
    {
      CHECK_CUDA(cudaMemcpy(input->gpu_buf,
                            inputN,
                            micro_batch * 256 * 256 * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    else
    {
      CHECK_CUDA(cudaMemcpy(input->gpu_buf,
                            inputN + 256 * 256 * max_batch_per_step * idx,
                            micro_batch * 256 * 256 * sizeof(float),
                            cudaMemcpyHostToDevice));
    }

    conv2d(input, c1, conv0_weight, conv0_bias);

    instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

    maxpool2d_relu(i1, m1, 2, 2);

    conv2d(m1, c2, conv1_weight, conv1_bias);

    instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);

    maxpool2d_relu(i2, m2, 2, 2);

    linear_relu(m2, l1, linear1_weight, linear1_bias);

    linear(l1, l2, linear2_weight, linear2_bias);

    linear(l2, output, linear3_weight, linear3_bias);

    CHECK_CUDA(cudaMemcpy(outputN + 2 * max_batch_per_step * idx,
                          output->gpu_buf,
                          micro_batch * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
}

__global__ void conv2d_kernel(float *in, float *out, float *weight, float *bias,
                              int N, int C, int K, int H, int W, int kH, int kW)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int oH = H - kH + 1;
  int oW = W - kW + 1;

  // input (N, C, H, W)
  // weight (K, C, kH, kW)
  // bias (K)
  // output(N, K, oH, oW)

  // n * K * oH * oW + k * oH * oW + oh * oW + ow
  // ow: output w-index
  int ow = tid % oW;
  int idx = tid / oW; // n * K * oH + k * oH + oh

  // oh: output h-index
  int oh = idx % oH;
  idx = idx / oH; // n * K + k

  // k: output channel-index
  int k = idx % K;

  // n: output batch-index
  int n = idx / K;

  if (n >= N)
  {
    return;
  }

  float sum = bias[k];
  int in_idx, weight_idx;
  for (int c = 0; c < C; c++)
  {
    for (int kh = 0; kh < kH; kh++)
    {
      for (int kw = 0; kw < kW; kw++)
      {
        in_idx = n * C * H * W + c * H * W + (oh + kh) * W + (ow + kw);
        weight_idx = k * C * kH * kW + c * kH * kW + kh * kW + kw;
        sum += in[in_idx] * weight[weight_idx];
      }
    }
  }
  out[tid] = sum;

  // if (in_h_idx >= 0 && in_h_idx < H && in_w_idx >= 0 && in_w_idx < W)
  // {
  //   int in_idx = n * C * H * W + c * H * W + in_h_idx * W + in_w_idx;
  //   int weight_idx = k * C * kH * kW + c * kH * kW + kh * kW + kw;
  //   sum += in[in_idx] * weight[weight_idx];
  // }
}

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t)
{
  // input (N, C, H, W)
  // weight (K, C, kH, kW)
  // bias (K)
  // output(N, K, oH, oW)

  int N = in_t->shape[0];

  int K = weight_t->shape[0];
  int C = weight_t->shape[1];
  int kH = weight_t->shape[2];
  int kW = weight_t->shape[3];

  int H = in_t->shape[2];
  int W = in_t->shape[3];

  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  conv2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, N, C, K, H, W, kH, kW);
}

__global__ void instancenorm2d_kernel(const float *in, float *out,
                                      const float *weight, const float *bias,
                                      const float *mean, const float *var,
                                      int N, int C, int H, int W)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // int tid = n * C * H * W + c * H * W + h * W + w;
  // int w = tid % W;
  int idx = tid / W; // n * C * H + c * H + h
  // int h = idx % H;
  idx /= H; // n * C + c;
  int c = idx % C;
  int n = idx / C;

  if (n >= N)
  {
    return;
  }

  int mean_var_idx = n * C + c;
  float m = mean[mean_var_idx];
  float v = var[mean_var_idx];

  out[tid] = (((in[tid] - m) / sqrt(v + 1e-5)) * weight[c]) + bias[c];
}

__global__ void compute_mean_var_kernel(const float *in, float *mean_var,
                                        int N, int C, int H, int W)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // input (N, C, H, W)
  // mean (N, C)
  // var (N, C)

  // int tid = n * C + c;
  int n = tid / C;
  int c = tid % C;

  if (n >= N)
  {
    return;
  }

  float e = 0.0f;
  float v = 0.0f;

  int NC = N * C;

  // Caculate mean
  for (int h = 0; h < H; h++)
  {
    for (int w = 0; w < W; w++)
    {
      int in_idx = n * C * H * W + c * H * W + h * W + w;
      e += in[in_idx];
    }
  }
  e /= H * W;

  int mean_idx = n * C + c;
  mean_var[mean_idx] = e;

  // Caculate Variance
  for (int h = 0; h < H; h++)
  {
    for (int w = 0; w < W; w++)
    {
      int in_idx = n * C * H * W + c * H * W + h * W + w;
      v += (in[in_idx] - e) * (in[in_idx] - e);
    }
  }
  v /= H * W;
  int var_index = NC + mean_idx;
  mean_var[var_index] = v;
}

static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t)
{
  // input (N, C, H, W)
  // weight (C)
  // bias (C)
  // output(N, C, H, W)
  // mean (N, C)
  // var (N, C)

  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  const int N = in_t->shape[0]; //=out_t->shape[0];
  const int C = in_t->shape[1]; //=out_t->shape[1];
  const int H = in_t->shape[2]; //=out_t->shape[2];
  const int W = in_t->shape[3]; //=out_t->shape[3];

  float *mean_var_gpu;
  CHECK_CUDA(cudaMalloc((void **)&mean_var_gpu, 2 * N * C * sizeof(float)));

  dim3 gridDim_mv(((N * C) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim_mv(THREADS_PER_BLOCK);

  compute_mean_var_kernel<<<gridDim_mv, blockDim_mv>>>(in_gpu, mean_var_gpu, N, C, H, W);

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  float *mean_gpu = mean_var_gpu;
  float *var_gpu = mean_var_gpu + N * C;

  instancenorm2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, mean_gpu, var_gpu, N, C, H, W);

  CHECK_CUDA(cudaFree(mean_var_gpu));
}

__global__ void linear_kernel(float *in, float *out, float *weight, float *bias,
                              int B, int M, int N, int K, bool do_relu)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // input (B, M, K)
  // weight (N, K)
  // bias (N)
  // output(B, M, N)

  // n: output n-index
  int n = tid % N;
  int idx = tid / N;

  // m: output m-index
  int m = idx % M;

  // b: output batch-index
  int b = idx / M;

  if (b >= B)
  {
    return;
  }

  // float sum = bias[n];
  float sum = 0.0f;
  for (int k = 0; k < K; k++)
  {
    int in_idx = b * M * K + m * K + k;
    // int weight_idx = n * K + k;
    int weight_idx = n * K + k;
    sum += in[in_idx] * weight[weight_idx];
  }
  sum += bias[n];
  if (do_relu)
  {
    constexpr float zero = 0.0f;
    out[tid] = sum > zero ? sum : zero;
  }
  else
  {
    out[tid] = sum;
  }
}

static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t)
{
  int B = in_t->shape[0];
  int M = in_t->shape[1];

  int K = weight_t->shape[0];
  int N = weight_t->shape[1];

  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  linear_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, B, M, N, K, false);
}

static void linear_relu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                        Tensor *bias_t)
{
  int B = in_t->shape[0];
  int M = in_t->shape[1];

  int K = weight_t->shape[0];
  int N = weight_t->shape[1];

  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  linear_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, B, M, N, K, true);
}

__global__ void maxpool2d_relu_kernel(float *in, float *out, int N, int C, int H_IN, int W_IN, int kH, int kW)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int H_OUT = H_IN / kH;
  int W_OUT = W_IN / kW;

  // w_out: output w-index
  int w_out = tid % W_OUT;
  int idx = tid / W_OUT;

  // h_out: output h-index
  int h_out = idx % H_OUT;
  idx /= H_OUT;

  // c: output channel-index
  int c = idx % C;

  // n: output batch-index
  int n = idx / C;

  if (n > N)
  {
    return;
  }

  // float max_val = -FLT_MAX;
  float max_val = 0.0f;
  float in_val;

  int h_idx, w_idx, in_idx;
  int kh, kw;
  int nc_idx = n * C * H_IN * W_IN + c * H_IN * W_IN;

  for (kw = 0; kw < kW; kw++)
  {
    for (kh = 0; kh < kH; kh++)
    {
      h_idx = h_out * kH + kh;
      w_idx = w_out * kW + kw;

      in_idx = nc_idx + h_idx * W_IN + w_idx;
      in_val = in[in_idx];

      max_val = max(max_val, in_val);
    }
  }
  out[tid] = max_val;
}

static void maxpool2d_relu(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  int N = in_t->shape[0];
  int C = in_t->shape[1];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  maxpool2d_relu_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, N, C, H_IN, W_IN, kH, kW);
}

// __global__ void relu_kernel(float *x, int numel)
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= numel)
//     return;

//   const float in_val = x[tid];
//   const float val = in_val > 0.0f ? in_val : 0.0f;
//   x[tid] = val;
// }

// static void relu(Tensor *inout_t)
// {
//   const int N = inout_t->get_elem();
//   float *inout_gpu = inout_t->gpu_buf;

//   dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
//   dim3 blockDim(THREADS_PER_BLOCK);

//   relu_kernel<<<gridDim, blockDim>>>(inout_gpu, N);
// }

void finalize_model()
{
  delete (conv0_weight);
  delete (conv0_bias);
  delete (conv1_weight);
  delete (conv1_bias);
  delete (linear1_weight);
  delete (linear1_bias);
  delete (linear2_weight);
  delete (linear2_bias);
  delete (linear3_weight);
  delete (linear3_bias);
  delete (instanceNorm2d0_weight);
  delete (instanceNorm2d0_bias);
  delete (instanceNorm2d1_weight);
  delete (instanceNorm2d1_bias);
  delete (input);
  delete (output);
  delete (c1);
  delete (i1);
  delete (m1);
  delete (c2);
  delete (i2);
  delete (m2);
  delete (l1);
  delete (l2);
}
