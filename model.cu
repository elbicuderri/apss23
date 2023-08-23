#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <float.h>

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
  // linear1_weight = new Tensor(buf, {128, 62});
  buf += 7936;
  linear1_bias = new Tensor(buf, {128});
  buf += 128;
  linear2_weight = new Tensor(buf, {128, 64});
  // linear2_weight = new Tensor(buf, {64, 128});
  buf += 8192;
  linear2_bias = new Tensor(buf, {64});
  buf += 64;
  linear3_weight = new Tensor(buf, {1015808, 2});
  // linear3_weight = new Tensor(buf, {2, 1015808});
  buf += 2031616;
  linear3_bias = new Tensor(buf, {2});
  buf += 2;

  // input = new Tensor({1, 256, 256});
  input = new Tensor({N, 1, 256, 256});

  // c1 = new Tensor({128, 254, 254});
  c1 = new Tensor({N, 128, 254, 254});

  // i1 = new Tensor({128, 254, 254});
  i1 = new Tensor({N, 128, 254, 254});

  // m1 = new Tensor({128, 127, 127});
  m1 = new Tensor({N, 128, 127, 127});

  // c2 = new Tensor({256, 125, 125});
  c2 = new Tensor({N, 256, 125, 125});

  // i2 = new Tensor({256, 125, 125});
  i2 = new Tensor({N, 256, 125, 125});

  m2 = new Tensor({N, 256 * 62, 62});
  // m2 = new Tensor({N, 256, 62, 62});

  l1 = new Tensor({N, 256 * 62, 128});
  // l1 = new Tensor({N, 256, 62, 128});

  l2 = new Tensor({N, 1, 256 * 62 * 64});

  output = new Tensor({N, 1, 2});

  std::cout << "========================" << std::endl;
  std::cout << "initialize_model" << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "parameter_fname" << std::endl;
  std::cout << parameter_fname << std::endl;
  std::cout << "========================" << std::endl;

  CHECK_CUDA(cudaDeviceSynchronize());
}
// Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// Size of in  = N * C_IN * H_IN * W_IN
// Size of out = N * C_OUT * (H_IN-K+1) * (W_IN-K+1)
// Weight : C_OUT * C_IN * K * K
// Bias : C_OUT

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

static void conv2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t);

static void conv2d_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                       Tensor *bias_t);

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW);

static void maxpool2d_v2(Tensor *in_t, Tensor *out_t, int kH, int kW);

static void maxpool2d_gpu(Tensor *in_t, Tensor *out_t, int kH, int kW);

// InstanceNorm2D
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
// size of in  = N * C * H * W
// size of out = N * C * H * W
// weight : C
// bias : C
static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t);

static void instancenorm2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                              Tensor *bias_t);

// Linear
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// size of in  = N * H_IN
// size of out = N * H_OUT
// weight : H_OUT * H_IN
// bias : H_OUT
static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

static void linear_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t);

static void linear_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                       Tensor *bias_t);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(Tensor *inout_t);

static void relu_gpu(Tensor *inout_t);

void model_forward(float *inputN, float *outputN)
{
  printf("model_forward\n");
  printf("N: %d\n", N);

  // For test
  CHECK_CUDA(cudaMemcpy(input->buf,
                        inputN,
                        N * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToHost));

  // conv2d_v2(input, c1, conv0_weight, conv0_bias);
  conv2d_gpu(input, c1, conv0_weight, conv0_bias);

  instancenorm2d_v2(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

  // maxpool2d_v2(i1, m1, 2, 2);
  maxpool2d_gpu(i1, m1, 2, 2);

  // relu(m1);
  relu_gpu(m1);

  // conv2d_v2(m1, c2, conv1_weight, conv1_bias);
  conv2d_gpu(m1, c2, conv1_weight, conv1_bias);

  instancenorm2d_v2(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);

  // maxpool2d_v2(i2, m2, 2, 2);
  maxpool2d_gpu(i2, m2, 2, 2);

  // relu(m2);
  relu_gpu(m2);

  // linear_v2(m2, l1, linear1_weight, linear1_bias);
  linear_gpu(m2, l1, linear1_weight, linear1_bias);

  // relu(l1);
  relu_gpu(l1);

  // linear_v2(l1, l2, linear2_weight, linear2_bias);
  linear_gpu(l1, l2, linear2_weight, linear2_bias);

  // linear_v2(l2, output, linear3_weight, linear3_bias);
  linear_gpu(l2, output, linear3_weight, linear3_bias);

  memcpy(outputN, output->buf, N * 2 * sizeof(float));
}

// void model_forward(float *inputN, float *outputN)
// {
//   for (int idx = 0; idx < N; idx++)
//   {
//     // memcpy(input->buf, inputN + 256 * 256 * idx, 256 * 256 * sizeof(float));

//     // For test
//     CHECK_CUDA(cudaMemcpy(input->buf,
//                           inputN + 256 * 256 * idx,
//                           256 * 256 * sizeof(float),
//                           cudaMemcpyHostToHost));

//     conv2d(input, c1, conv0_weight, conv0_bias);
//     instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
//     maxpool2d(i1, m1, 2, 2);
//     relu(m1);
//     conv2d(m1, c2, conv1_weight, conv1_bias);
//     instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
//     maxpool2d(i2, m2, 2, 2);
//     relu(m2);

//     // m2->reshape({N, 256 * 62, 62});

//     // linear(m2, l1, linear1_weight, linear1_bias);

//     m2->reshape({N, 256 * 62, 62});
//     linear_v2(m2, l1, linear1_weight, linear1_bias);
//     // l1->reshape({256, 62, 128});

//     relu(l1);

//     // linear(l1, l2, linear2_weight, linear2_bias);

//     l1->reshape({N, 256 * 62, 128});
//     linear_v2(l1, l2, linear2_weight, linear2_bias);

//     // l2->reshape({1, 1015808});
//     // linear(l2, output, linear3_weight, linear3_bias);

//     l2->reshape({N, 1, 1015808});
//     linear_v2(l2, output, linear3_weight, linear3_bias);

//     memcpy(outputN + 2 * idx, output->buf, 2 * sizeof(float));
//   }
// }

__global__ void conv2d_kernel(float *in, float *out, float *weight, float *bias,
                              int N, int C, int K, int H, int W, int kH, int kW)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int oH = H - kH + 1;
  int oW = W - kW + 1;

  int output_numel = N * K * oH * oW;
  if (tid >= output_numel)
  {
    return;
  }

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

  float sum = bias[k];
  for (int c = 0; c < C; c++)
  {
    for (int kh = 0; kh < kH; kh++)
    {
      for (int kw = 0; kw < kW; kw++)
      {
        int in_h_idx = oh + kh;
        int in_w_idx = ow + kw;

        if (in_h_idx >= 0 && in_h_idx < H && in_w_idx >= 0 && in_w_idx < W)
        {
          int in_idx = n * C * H * W + c * H * W + in_h_idx * W + in_w_idx;
          int weight_idx = k * C * kH * kW + c * kH * kW + kh * kW + kw;
          sum += in[in_idx] * weight[weight_idx];
        }
      }
    }
  }
  // int out_idx = n * C_OUT * H_OUT * W_OUT + c_out * H_OUT * W_OUT + h_out * W_OUT + w_out;
  out[tid] = sum;
}

static void conv2d_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                       Tensor *bias_t)
{
  std::cout << "conv2d on GPU!!!" << std::endl;

  // input (N, C, H, W)
  // weight (K, C, kH, kW)
  // bias (K)
  // output(N, K, oH, oW)

  int K = weight_t->shape[0];
  int C = weight_t->shape[1];
  int kH = weight_t->shape[2];
  int kW = weight_t->shape[3];

  int H = in_t->shape[2];
  int W = in_t->shape[3];

  int oH = H - kH + 1;
  int oW = W - kW + 1;

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  float *in_cpu = in_t->buf;
  float *out_cpu = out_t->buf;
  float *weight_cpu = weight_t->buf;
  float *bias_cpu = bias_t->buf;

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  //////
  CHECK_CUDA(cudaMalloc(&weight_gpu, weight_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(weight_gpu, weight_cpu, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&bias_gpu, bias_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(bias_gpu, bias_cpu, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  conv2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, N, C, K, H, W, kH, kW);

  CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  in_t->free_gpu_buf();
  out_t->free_gpu_buf();
  weight_t->free_gpu_buf();
  bias_t->free_gpu_buf();
}

static void conv2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int kH = weight_t->shape[2];
  int kW = weight_t->shape[3];

  int C_IN = weight_t->shape[1];  //=in_t->shape[0];
  int C_OUT = weight_t->shape[0]; //=out_t->shape[0];

  int N = in_t->shape[0];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int H_OUT = H_IN - kH + 1; //=out_t->shape[1];
  int W_OUT = W_IN - kW + 1; //=out_t->shape[2];

  for (int n = 0; n < N; n++)
  {
    for (int c_out = 0; c_out < C_OUT; c_out++)
    {
      for (int h_out = 0; h_out < H_OUT; h_out++)
      {
        for (int w_out = 0; w_out < W_OUT; w_out++)
        {
          // int out_idx = c_out * H_OUT * W_OUT + h_out * W_OUT + w_out;
          // out[out_idx] = bias[c_out];
          float sum = bias[c_out];
          for (int c_in = 0; c_in < C_IN; c_in++)
          {
            for (int kh = 0; kh < kH; kh++)
            {
              for (int kw = 0; kw < kW; kw++)
              {
                int in_idx = n * C_IN * H_IN * W_IN + c_in * H_IN * W_IN + (h_out + kh) * W_IN + (w_out + kw);
                int weight_idx = c_out * C_IN * kH * kW + c_in * kH * kW + kh * kW + kw;
                sum += in[in_idx] * weight[weight_idx];
              }
            }
          }
          int out_idx = n * C_OUT * H_OUT * W_OUT + c_out * H_OUT * W_OUT + h_out * W_OUT + w_out;
          // int out_idx = c_out * H_OUT * W_OUT + h_out * W_OUT + w_out;
          out[out_idx] = sum;
        }
      }
    }
  }
}

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int K = weight_t->shape[2]; //=weight_t->shape[3];

  int C_IN = weight_t->shape[1];  //=in_t->shape[0];
  int C_OUT = weight_t->shape[0]; //=out_t->shape[0];

  int H_IN = in_t->shape[1];
  int W_IN = in_t->shape[2];
  int H_OUT = H_IN - K + 1; //=out_t->shape[1];
  int W_OUT = W_IN - K + 1; //=out_t->shape[2];

  for (int c_out = 0; c_out < C_OUT; c_out++)
  {
    for (int h_out = 0; h_out < H_OUT; h_out++)
    {
      for (int w_out = 0; w_out < W_OUT; w_out++)
      {
        out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] = bias[c_out];
        for (int c_in = 0; c_in < C_IN; c_in++)
        {
          for (int kh = 0; kh < K; kh++)
          {
            for (int kw = 0; kw < K; kw++)
            {
              out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] +=
                  in[c_in * H_IN * W_IN + (h_out + kh) * W_IN + (w_out + kw)] *
                  weight[c_out * C_IN * K * K + c_in * K * K + kh * K + kw];
            }
          }
        }
      }
    }
  }
}

static void instancenorm2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                              Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int N = in_t->shape[0]; //=out_t->shape[0];
  int C = in_t->shape[1]; //=out_t->shape[1];
  int H = in_t->shape[2]; //=out_t->shape[2];
  int W = in_t->shape[3]; //=out_t->shape[3];

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      float e = 0, v = 0;

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

      for (int h = 0; h < H; h++)
      {
        for (int w = 0; w < W; w++)
        {
          int out_idx = n * C * H * W + c * H * W + h * W + w;
          int in_idx = n * C * H * W + c * H * W + h * W + w;
          out[out_idx] = (in[in_idx] - e) / sqrt(v + 1e-5) * weight[c] + bias[c];
        }
      }
    }
  }
}

static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int C = in_t->shape[0]; //=out_t->shape[0];
  int H = in_t->shape[1]; //=out_t->shape[1];
  int W = in_t->shape[2]; //=out_t->shape[2];

  for (int c = 0; c < C; c++)
  {
    float e = 0, v = 0;

    // Caculate mean
    for (int h = 0; h < H; h++)
    {
      for (int w = 0; w < W; w++)
      {
        e += in[c * H * W + h * W + w];
      }
    }
    e /= H * W;

    // Caculate Variance
    for (int h = 0; h < H; h++)
    {
      for (int w = 0; w < W; w++)
      {
        v += (in[c * H * W + h * W + w] - e) * (in[c * H * W + h * W + w] - e);
      }
    }
    v /= H * W;

    for (int h = 0; h < H; h++)
    {
      for (int w = 0; w < W; w++)
      {
        out[c * H * W + h * W + w] =
            (in[c * H * W + h * W + w] - e) / sqrt(v + 1e-5) * weight[c] +
            bias[c];
      }
    }
  }
}

__global__ void linear_kernel(float *in, float *out, float *weight, float *bias, int B, int M, int N, int K)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int output_numel = B * M * N;
  if (tid >= output_numel)
  {
    return;
  }

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

  float sum = bias[n];
  for (int k = 0; k < K; k++)
  {
    int in_idx = b * M * K + m * K + k;
    // int weight_idx = n * K + k;
    int weight_idx = n * K + k;
    sum += in[in_idx] * weight[weight_idx];
  }
  out[tid] = sum;
}

static void linear_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                       Tensor *bias_t)
{
  std::cout << "linear on GPU!!!" << std::endl;

  // input (B, M, K)
  // weight (N, K)
  // bias (N)
  // output(B, M, N)

  int B = in_t->shape[0];
  int M = in_t->shape[1];

  int K = weight_t->shape[0];
  int N = weight_t->shape[1];

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();

  float *in_cpu = in_t->buf;
  float *out_cpu = out_t->buf;

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;

  CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  //////
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  float *weight_cpu = weight_t->buf;
  float *bias_cpu = bias_t->buf;

  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  CHECK_CUDA(cudaMalloc(&weight_gpu, weight_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(weight_gpu, weight_cpu, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&bias_gpu, bias_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(bias_gpu, bias_cpu, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  linear_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, B, M, N, K);

  CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  in_t->free_gpu_buf();
  out_t->free_gpu_buf();
  weight_t->free_gpu_buf();
  bias_t->free_gpu_buf();
}

static void linear_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  // (B, M, K) @ (K, N) -> (B, M, N)

  int B = in_t->shape[0];
  int M = in_t->shape[1];

  int K = weight_t->shape[0]; // in_t의 마지막 차원
  int N = weight_t->shape[1]; // out_t의 마지막 차원
  // int N = out_t->shape[2]; // out_t의 마지막 차원

  std::cout << "========================" << std::endl;
  std::cout << "out_t->shape[0]: " << out_t->shape[0] << std::endl;
  std::cout << "out_t->shape[1]: " << out_t->shape[1] << std::endl;
  std::cout << "out_t->shape[2]: " << out_t->shape[2] << std::endl;
  std::cout << "========================" << std::endl;

  for (int b = 0; b < B; b++)
  {
    for (int m = 0; m < M; m++)
    {
      for (int n = 0; n < N; n++)
      {
        float sum = bias[n];
        for (int k = 0; k < K; k++)
        {
          int in_idx = b * M * K + m * K + k;
          // int weight_idx = n * K + k;
          int weight_idx = n * K + k;
          sum += in[in_idx] * weight[weight_idx];
        }
        int out_idx = b * M * N + m * N + n;
        out[out_idx] = sum;
      }
    }
  }
  ///
}

static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int K = weight_t->shape[0]; // in_t의 마지막 차원
  int N = weight_t->shape[1]; // out_t의 마지막 차원

  int M = in_t->get_elem() / K; //=out_t->get_elem()/N

  for (int m = 0; m < M; m++)
  {
    for (int n = 0; n < N; n++)
    {
      float sum = bias[n];
      for (int k = 0; k < K; k++)
      {
        int in_idx = m * K + k;
        int weight_idx = n * K + k;
        sum += in[in_idx] * weight[weight_idx];
      }
      int out_idx = m * N + n;
      out[out_idx] = sum;
    }
  }
}

__global__ void maxpool2d_kernel(float *in, float *out, int N, int C, int H_IN, int W_IN, int kH, int kW)
{

  int H_OUT = H_IN / kH;
  int W_OUT = W_IN / kW;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int output_numel = N * C * H_OUT * W_OUT;
  if (tid >= output_numel)
  {
    return;
  }

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

  // output(n, c, h_out, w_out)

  // int first_in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + (h_out * kH) * W_IN + (w_out * kW);
  // float max_val = in[first_in_idx];

  float max_val = -FLT_MAX;
  for (int kh = 0; kh < kH; kh++)
  {
    for (int kw = 0; kw < kW; kw++)
    {
      int h_idx = h_out * kH + kh;
      int w_idx = w_out * kW + kw;
      if (h_idx >= 0 && h_idx < H_IN && w_idx >= 0 && w_idx < W_IN)
      {
        int in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
        float in_val = in[in_idx];

        if (in_val > max_val)
        {
          max_val = in_val;
        }
      }
    }
  }
  out[tid] = max_val;
}

static void maxpool2d_gpu(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  std::cout << "maxpool2d on GPU!!!" << std::endl;

  float *in_cpu = in_t->buf;
  float *out_cpu = out_t->buf;

  int N = in_t->shape[0];
  int C = in_t->shape[1];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;

  CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  maxpool2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, N, C, H_IN, W_IN, kH, kW);

  CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  in_t->free_gpu_buf();
  out_t->free_gpu_buf();
}

static void maxpool2d_v2(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  float *in = in_t->buf;
  float *out = out_t->buf;

  int N = in_t->shape[0];
  int C = in_t->shape[1];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int H_OUT = H_IN / kH; // =out_t->shape[2];
  int W_OUT = W_IN / kW; // =out_t->shape[3];

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h_out = 0; h_out < H_OUT; h_out++)
      {
        for (int w_out = 0; w_out < W_OUT; w_out++)
        {
          int first_in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + (h_out * kH) * W_IN + (w_out * kW);
          assert(first_in_idx < in_numel);

          float max_val = in[first_in_idx];
          for (int kh = 0; kh < kH; kh++)
          {
            for (int kw = 0; kw < kW; kw++)
            {
              int h_idx = h_out * kH + kh;
              int w_idx = w_out * kW + kw;
              if (h_idx < H_IN && w_idx < W_IN)
              {
                int in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
                assert(in_idx < in_numel);

                float in_val = in[in_idx];
                if (in_val > max_val)
                {
                  max_val = in_val;
                }
              }
            }
          }
          int out_idx = n * C * H_OUT * W_OUT + c * H_OUT * W_OUT + h_out * W_OUT + w_out;
          assert(out_idx < out_numel);

          out[out_idx] = max_val;
        }
      }
    }
  }
  ///
}

static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  float *in = in_t->buf;
  float *out = out_t->buf;

  int H_IN = in_t->shape[1];
  int W_IN = in_t->shape[2];
  int H_OUT = H_IN / kH; // =out_t->shape[1];
  int W_OUT = W_IN / kW; // =out_t->shape[2];

  int N = in_t->shape[0];

  for (int n = 0; n < N; n++)
  {
    for (int h_out = 0; h_out < H_OUT; h_out++)
    {
      for (int w_out = 0; w_out < W_OUT; w_out++)
      {
        float max_val = in[n * H_IN * W_IN + (h_out * kH) * H_IN + (w_out * kW)];
        for (int kh = 0; kh < kH; kh++)
        {
          for (int kw = 0; kw < kW; kw++)
          {
            int in_idx = n * H_IN * W_IN + (h_out * kH + kh) * H_IN + (w_out * kW + kw);
            float in_val = in[in_idx];
            if (in_val > max_val)
            {
              max_val = in_val;
            }
          }
        }
        int out_idx = n * H_OUT * W_OUT + h_out * W_OUT + w_out;
        out[out_idx] = max_val;
      }
    }
  }
}

__global__ void relu_kernel(float *inout, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  float val = inout[tid] > 0.0f ? inout[tid] : 0.0f;
  inout[tid] = val;
}

static void relu_gpu(Tensor *inout_t)
{
  printf("relu on GPU!!!\n");

  float *inout = inout_t->buf;
  int N = inout_t->get_elem();

  auto tmp_buf = inout_t->gpu_buf;

  CHECK_CUDA(cudaMalloc(&tmp_buf, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(tmp_buf, inout, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  relu_kernel<<<gridDim, blockDim>>>(tmp_buf, N);

  CHECK_CUDA(cudaMemcpy(inout, tmp_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

  inout_t->free_gpu_buf();
}

static void relu(Tensor *inout_t)
{
  float *inout = inout_t->buf;
  int N = inout_t->get_elem();
  for (int n = 0; n < N; n++)
  {
    inout[n] = fmaxf(inout[n], 0);
  }
}

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
