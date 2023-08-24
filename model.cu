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
  // input = new Tensor({N, 1, 256, 256}, /*malloc_on_host=*/true);

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

static void instancenorm2d_v3(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                              Tensor *bias_t);

static void instancenorm2d_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                               Tensor *bias_t);

static void instancenorm2d_gpu_cpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
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

  // int max_batch_per_step = 128;

  int steps = (N + max_batch_per_step - 1) / max_batch_per_step;
  int nn = N % max_batch_per_step;
  int last_batch = nn == 0 ? max_batch_per_step : nn;

  std::cout << "steps: " << steps << std::endl;
  std::cout << "max_batch_per_step: " << max_batch_per_step << std::endl;
  std::cout << "last_batch: " << last_batch << std::endl;

  for (int idx = 0; idx < steps; idx++)
  {
    // memcpy(input->buf, inputN + 256 * 256 * idx, 256 * 256 * sizeof(float));

    int micro_batch = max_batch_per_step;
    if (idx == (steps - 1))
    {
      micro_batch = last_batch;
    }

    std::cout << "========================" << std::endl;
    std::cout << "step idx: " << idx << std::endl;
    std::cout << "micro_batch: " << micro_batch << std::endl;

    // // For test
    // CHECK_CUDA(cudaMemcpy(input->gpu_buf,
    //                       inputN,
    //                       N * 256 * 256 * sizeof(float),
    //                       cudaMemcpyHostToDevice));

    // // For test
    // CHECK_CUDA(cudaMemcpy(input->gpu_buf + 256 * 256 * max_batch_per_step * idx,
    //                       inputN + 256 * 256 * max_batch_per_step * idx,
    //                       micro_batch * 256 * 256 * sizeof(float),
    //                       cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(input->gpu_buf + 256 * 256 * max_batch_per_step * idx,
                          inputN + 256 * 256 * max_batch_per_step * idx,
                          micro_batch * 256 * 256 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // conv2d_v2(input, c1, conv0_weight, conv0_bias);
    conv2d_gpu(input, c1, conv0_weight, conv0_bias);

    // instancenorm2d_v3(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
    // instancenorm2d_gpu(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
    instancenorm2d_gpu_cpu(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

    // maxpool2d_v2(i1, m1, 2, 2);
    maxpool2d_gpu(i1, m1, 2, 2);

    // relu(m1);
    relu_gpu(m1);

    // conv2d_v2(m1, c2, conv1_weight, conv1_bias);
    conv2d_gpu(m1, c2, conv1_weight, conv1_bias);

    // instancenorm2d_v2(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
    // instancenorm2d_gpu(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
    instancenorm2d_gpu_cpu(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);

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

    std::cout << "3rd linear start" << std::endl;
    // linear_v2(l2, output, linear3_weight, linear3_bias);
    linear_gpu(l2, output, linear3_weight, linear3_bias);
    std::cout << "3rd linear end" << std::endl;

    // memcpy(outputN, output->buf, N * 2 * sizeof(float));

    // CHECK_CUDA(cudaMemcpy(outputN + 2 * max_batch_per_step * idx,
    //                       output->buf + 2 * max_batch_per_step * idx,
    //                       micro_batch * 2 * sizeof(float),
    //                       cudaMemcpyHostToHost));

    std::cout << "cudaMemcpy output start" << std::endl;

    CHECK_CUDA(cudaMemcpy(outputN + 2 * max_batch_per_step * idx,
                          output->gpu_buf + 2 * max_batch_per_step * idx,
                          micro_batch * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // CHECK_CUDA(cudaMemcpy(outputN,
    //                       output->gpu_buf,
    //                       N * 2 * sizeof(float),
    //                       cudaMemcpyDeviceToHost));

    std::cout << "cudaMemcpy output end" << std::endl;
  }

  std::cout << "input output free start" << std::endl;

  input->free_cpu_buf();
  output->free_cpu_buf();

  std::cout << "input output free end" << std::endl;
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
  clock_t start_time2 = clock();

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

  // CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  // //////
  // CHECK_CUDA(cudaMalloc(&weight_gpu, weight_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(weight_gpu, weight_cpu, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&bias_gpu, bias_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(bias_gpu, bias_cpu, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  clock_t start_time = clock();

  conv2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, N, C, K, H, W, kH, kW);

  CHECK_CUDA(cudaDeviceSynchronize());

  // Code to be timed
  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  // in_t->free_gpu_buf();
  // out_t->free_gpu_buf();
  // weight_t->free_gpu_buf();
  // bias_t->free_gpu_buf();

  clock_t end_time2 = clock();

  double total_conv2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_conv2d_time: %.10f seconds\n", total_conv2d_time);
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

__global__ void instancenorm2d_kernel(const float *in, float *out, const float *weight, const float *bias,
                                      int N, int C, int H, int W)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int output_numel = N * C * H * W;
  // printf("tid: %d\n", tid);
  // printf("N: %d\n", N);
  // printf("C: %d\n", C);
  // printf("H: %d\n", H);
  // printf("W: %d\n", W);
  // printf("output_numel: %d\n", output_numel);

  if (tid >= output_numel)
  {
    // printf("tid >= output_numel\n");
    // printf("tid: %d\n", tid);
    // printf("output_numel: %d\n", output_numel);
    return;
  }

  // // /////////////////////////////////////////////////////
  // // input (N, C, H, W)
  // // weight (C)
  // // bias (C)
  // // output(N, C, H, W)

  // // n * C * H * W + c * H * W + oh * W + ow

  // // ow: output w-index
  // // int ow = tid % W;
  // int idx = tid / W; // n * C * H + c * H + oh

  // // oh: output h-index
  // int oh = idx % H;
  // idx = idx / oh; // n * C + c

  // // c: output channel-index
  // int c = idx % C;

  // // n: output batch-index
  // int n = idx / C;

  // float mean = 0.0f, var = 0.0f;

  // // float m = 0.0f, v = 0.0f;

  // // Caculate mean
  // for (int h = 0; h < H; h++)
  // {
  //   for (int w = 0; w < W; w++)
  //   {
  //     int idx1 = n * C * H * W + c * H * W + h * W + w;
  //     // if (idx1 >= output_numel) {
  //     //   printf("idx1 >= output_numel\n");
  //     //   printf("idx1: %d\n", idx1);
  //     //   printf("output_numel: %d\n", output_numel);
  //     // }
  //     if (idx1 < output_numel)
  //     {
  //       mean += in[idx1];
  //     }
  //     __syncthreads();
  //     // m += in[idx1];
  //   }
  //   __syncthreads();
  // }
  // mean /= H * W;
  // __syncthreads();
  // // __syncthreads();

  // // Caculate Variance
  // for (int hh = 0; hh < H; hh++)
  // {
  //   for (int ww = 0; ww < W; ww++)
  //   {
  //     int idx2 = n * C * H * W + c * H * W + hh * W + ww;
  //     // if (idx2 >= output_numel) {
  //     //   printf("idx2 >= output_numel\n");
  //     //   printf("idx2: %d\n", idx2);
  //     //   printf("output_numel: %d\n", output_numel);
  //     // }
  //     if (idx2 < output_numel)
  //     {
  //       var += (in[idx2] - mean) * (in[idx2] - mean);
  //     }
  //     __syncthreads();
  //   }
  //   __syncthreads();
  // }
  // var /= H * W;
  // __syncthreads();
  //
  // if (tid < output_numel)
  // {
  //   out[tid] = ((in[tid] - mean) / sqrt(var + 1e-5)) * weight[c] + bias[c];
  // }
  // // /////////////////////////////////////////////////////

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      float e = 0.0f, v = 0.0f;

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

static void instancenorm2d_gpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                               Tensor *bias_t)
{
  std::cout << "instancenorm2d on GPU!!!" << std::endl;

  clock_t start_time2 = clock();

  // input (N, C, H, W)
  // weight (C)
  // bias (C)
  // output(N, C, H, W)

  // std::cout << "========================" << std::endl;
  // std::cout << "in_t->shape[0]: " << in_t->shape[0] << std::endl;
  // std::cout << "in_t->shape[1]: " << in_t->shape[1] << std::endl;
  // std::cout << "in_t->shape[2]: " << in_t->shape[2] << std::endl;
  // std::cout << "in_t->shape[3]: " << in_t->shape[3] << std::endl;
  // std::cout << "========================" << std::endl;

  // std::cout << "========================" << std::endl;
  // std::cout << "out_t->shape[0]: " << out_t->shape[0] << std::endl;
  // std::cout << "out_t->shape[1]: " << out_t->shape[1] << std::endl;
  // std::cout << "out_t->shape[2]: " << out_t->shape[2] << std::endl;
  // std::cout << "out_t->shape[3]: " << out_t->shape[3] << std::endl;
  // std::cout << "========================" << std::endl;

  // std::cout << "========================" << std::endl;
  // std::cout << "weight_t->shape[0]: " << weight_t->shape[0] << std::endl;
  // std::cout << "bias_t->shape[0]: " << bias_t->shape[0] << std::endl;
  // std::cout << "========================" << std::endl;

  int N = in_t->shape[0];
  int C = in_t->shape[1];
  int H = in_t->shape[2];
  int W = in_t->shape[3];

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  // std::cout << "========================" << std::endl;
  // std::cout << "in_numel: " << in_numel << std::endl;
  // std::cout << "out_numel: " << out_numel << std::endl;
  // std::cout << "weight_numel: " << weight_numel << std::endl;
  // std::cout << "bias_numel: " << bias_numel << std::endl;
  // std::cout << "========================" << std::endl;

  float *in_cpu = in_t->buf;
  float *out_cpu = out_t->buf;
  float *weight_cpu = weight_t->buf;
  float *bias_cpu = bias_t->buf;

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  // CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  // //////
  // CHECK_CUDA(cudaMalloc(&weight_gpu, weight_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(weight_gpu, weight_cpu, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&bias_gpu, bias_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(bias_gpu, bias_cpu, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  clock_t start_time = clock();

  // instancenorm2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, N, C, H, W);

  instancenorm2d_kernel<<<1, 1>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, N, C, H, W);

  // Code to be timed
  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  // in_t->free_gpu_buf();
  // out_t->free_gpu_buf();
  // weight_t->free_gpu_buf();
  // bias_t->free_gpu_buf();

  clock_t end_time2 = clock();

  double total_instancenorm2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_instancenorm2d_time: %.10f seconds\n", total_instancenorm2d_time);
}

__global__ void instancenorm2d_kernel_v2(float *in, float *out,
                                         float *weight, float *bias,
                                         float *mean, float *var,
                                         int N, int C, int H, int W)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // constexpr int N = 1;
  // constexpr int C = 128;
  // constexpr int H = 254;
  // constexpr int W = 254;

  int output_numel = N * C * H * W;
  // int output_numel = 1 * 128 * 254 * 254;
  if (tid >= output_numel)
  {
    return;
  }

  // // int tid = n * C * H * W + c * H * W + h * W + w;
  // int idx = tid / 254; // n * C * H + c * H + h
  // if (tid >= 0 && tid < 1)
  //   printf("tid: %d, idx: %d\n", tid, idx);
  // if (tid >= 10047 && tid < 10048)
  //   printf("tid: %d, idx: %d\n", tid, idx);

  // idx = tid / 254; // n * C + c;
  // if (tid >= 0 && tid < 1)
  //   printf("tid: %d, idx: %d\n", tid, idx);
  // if (tid >= 10047 && tid < 10048)
  //   printf("tid: %d, idx: %d\n", tid, idx);
  // int c = idx % 128;
  // int n = idx / 128;

  // if (tid >= 10000 && tid < 10300)
  // {
  //   printf("tid: %d, n: %d, c: %d\n", tid, n, c);
  //   // printf("n: %d\n", n);
  //   // printf("c: %d\n", c);
  // }

  // if (tid == 0)
  // {
  //   for (int i = 0; i < 10; i++)
  //   {
  //     printf("i: %d\n", i);
  //     printf("mean[i]: %.4f\n", mean[i]);
  //     printf("var[i]: %.4f\n", var[i]);
  //   }
  // }

  // input (N, C, H, W)
  // weight (C)
  // bias (C)
  // output(N, C, H, W)
  // mean (N, C)
  // var (N, C)

  // int tid = n * C * H * W + c * H * W + h * W + w;
  int w = tid % W;
  int idx = tid / W; // n * C * H + c * H + h
  if (tid >= 10048 && tid < 10049)
  {
    // printf("tid: %d, N: %d, C: %d, H: %d, W: %d\n", tid, N, C, H, W);
    printf("11 -> tid: %d, idx: %d\n", tid, idx);
  }
  int h = idx % H;
  idx = idx / H; // n * C + c;
  if (tid >= 10048 && tid < 10049)
  {
    // printf("tid: %d, N: %d, C: %d, H: %d, W: %d\n", tid, N, C, H, W);
    printf("22 -> tid: %d, idx: %d\n", tid, idx);
  }
  int c = idx % C;
  int n = idx / C;

  // if (tid >= 0 && tid < 1)
  // {
  //   // printf("tid: %d, N: %d, C: %d, H: %d, W: %d\n", tid, N, C, H, W);
  //   printf("tid: %d, n: %d, c: %d, h: %d, w: %d\n", tid, n, c, h, w);
  // }

  if (tid >= 10048 && tid < 10049)
  {
    // printf("tid: %d, N: %d, C: %d, H: %d, W: %d\n", tid, N, C, H, W);
    printf("tid: %d, n: %d, c: %d, h: %d, w: %d\n", tid, n, c, h, w);
  }

  int mean_var_idx = n * C + c;
  // int mean_var_idx = c * N + n;
  // int mean_var_idx = n * 128 + c;
  // if (tid >= 0 && tid < 1)
  //   printf("tid: %d, mean_var_idx: %d, N: %d, C: %d, H: %d, W: %d\n", tid, mean_var_idx, N, C, H, W);
  // if (tid >= 10048 && tid < 10049)
  //   printf("tid: %d, mean_var_idx: %d, N: %d, C: %d, H: %d, W: %d\n", tid, mean_var_idx, N, C, H, W);

  // if (tid >= 0 && tid < 1)
  //   printf("tid: %d, n: %d, c: %d, mean[mean_var_idx]: %f, var[mean_var_idx]: %f, mean_var_idx: %d, N: %d, C: %d, H: %d, W: %d\n",
  //          tid, n, c, mean[mean_var_idx], var[mean_var_idx], mean_var_idx, N, C, H, W);
  // if (tid >= 10048 && tid < 10049)
  //   printf("tid: %d, n: %d, c: %d, mean[mean_var_idx]: %f, var[mean_var_idx]: %f, mean_var_idx: %d, N: %d, C: %d, H: %d, W: %d\n",
  //          tid, n, c, mean[mean_var_idx], var[mean_var_idx], mean_var_idx, N, C, H, W);

  // if (tid >= 0 && tid < 1)
  //   printf("tid: %d, n: %d, c: %d, mean[mean_var_idx]: %d, var[mean_var_idx]: %d, mean_var_idx: %d, N, %d, C: %d, H: %d, W: %d\n",
  //          tid, n, c, mean[mean_var_idx], var[mean_var_idx], mean_var_idx, N, C, H, W);

  // if (tid >= 10047 && tid < 10049)
  //   printf("tid: %d, n: %d, c: %d, mean[mean_var_idx]: %d, var[mean_var_idx]: %d, mean_var_idx: %d, N, %d, C: %d, H: %d, W: %d\n",
  //          tid, n, c, mean[mean_var_idx], var[mean_var_idx], mean_var_idx, N, C, H, W);

  // int mean_var_idx = n * C + c;
  float m = mean[mean_var_idx];
  float v = var[mean_var_idx];
  // __syncthreads();

  int wb_idx = c;
  // int wb_idx = n;

  if (tid < output_numel)
  {
    // out[tid] = (((in[tid] - m) / sqrt(v + 1e-5)) * weight[c]) + bias[c];
    out[tid] = (((in[tid] - m) / sqrt(v + 1e-5)) * weight[wb_idx]) + bias[wb_idx];
  }

  // if (tid >= 0 && tid < 1)
  //   // printf("tid: %d, out[tid]: %f, in[tid]: %f, N: %d, C: %d, H: %d, W: %d\n", tid, out[tid], in[tid], N, C, H, W);
  //   printf("tid: %d, out[tid]: %f, in[tid]: %f, n, %d, c: %d, mean: %f, var: %f, weight: %f, bias: %f\n", tid, out[tid], in[tid], n, wb_idx, m, v, weight[wb_idx], bias[wb_idx]);
  if (tid >= 10048 && tid < 10049)
    // printf("tid: %d, out[tid]: %f, in[tid]: %f, N: %d, C: %d, H: %d, W: %d\n", tid, out[tid], in[tid], N, C, H, W);
    printf("tid: %d, out[tid]: %f, in[tid]: %f, n, %d, c: %d, mean: %f, var: %f, weight: %f, bias: %f\n", tid, out[tid], in[tid], n, wb_idx, m, v, weight[wb_idx], bias[wb_idx]);

  // double m = (double)(mean[mean_var_idx]);
  // double v = (double)(var[mean_var_idx]);
  // double eps = 1e-5;

  // // out[tid] = (in[tid] - m) / sqrt(v + 1e-5) * weight[c] + bias[c];

  // if (tid < output_numel)
  // {
  //   out[tid] = (float)((((double)(in[tid]) - m)) / sqrt((double)(v + 1e-5)) * (double)weight[c] + (double)bias[c]);
  // }
}

static void instancenorm2d_gpu_cpu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                                   Tensor *bias_t)
{
  std::cout << "instancenorm2d on GPU and CPU!!!" << std::endl;

  clock_t start_time2 = clock();

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  in_t->buf = (float *)malloc(in_numel * sizeof(float));
  out_t->buf = (float *)malloc(out_numel * sizeof(float));
  weight_t->buf = (float *)malloc(weight_numel * sizeof(float));
  bias_t->buf = (float *)malloc(bias_numel * sizeof(float));

  CHECK_CUDA(
      cudaMemcpy(in_t->buf, in_t->gpu_buf, in_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(out_t->buf, out_t->gpu_buf, out_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(weight_t->buf, weight_t->gpu_buf, weight_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(bias_t->buf, bias_t->gpu_buf, bias_numel * sizeof(float), cudaMemcpyDeviceToHost));

  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight_cpu = weight_t->buf;
  float *bias_cpu = bias_t->buf;

  const int N = in_t->shape[0]; //=out_t->shape[0];
  const int C = in_t->shape[1]; //=out_t->shape[1];
  const int H = in_t->shape[2]; //=out_t->shape[2];
  const int W = in_t->shape[3]; //=out_t->shape[3];

  // input (N, C, H, W)
  // weight (C)
  // bias (C)
  // output(N, C, H, W)

  float *mean_cpu = (float *)malloc(N * C * sizeof(float));
  float *var_cpu = (float *)malloc(N * C * sizeof(float));

  clock_t start_time = clock();

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      float e = 0.0f, v = 0.0f;

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
      mean_cpu[n * C + c] = e;

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
      var_cpu[n * C + c] = v;

      // if (n == 0 && c == 0)
      // {
      //   std::cout << "===========================" << std::endl;
      //   std::cout << "n: " << n << std::endl;
      //   std::cout << "c: " << c << std::endl;
      //   std::cout << "mean: " << e << std::endl;
      //   std::cout << "var: " << v << std::endl;
      //   std::cout << "===========================" << std::endl;
      // }

      if (n == 0 && c == 39)
      {
        std::cout << "===========================" << std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << "c: " << c << std::endl;
        std::cout << "mean: " << e << std::endl;
        std::cout << "var: " << v << std::endl;
        std::cout << "===========================" << std::endl;
      }
    }
  }

  clock_t end_time3 = clock();

  double compute_mean_var_time = (double)(end_time3 - start_time) / CLOCKS_PER_SEC;
  printf("compute_mean_var_time: %.10f seconds\n", compute_mean_var_time);

  // for (int i = 0; i < 10; i++)
  // {
  //   printf("i: %d\n", i);
  //   printf("mean_cpu[i]: %.4f\n", mean_cpu[i]);
  //   printf("var_cpu[i]: %.4f\n", var_cpu[i]);
  //   // std::cout << "i: " << i << std::endl;
  //   // std::cout << "mean_cpu[i]: " << mean_cpu[i] << std::endl;
  //   // std::cout << "var_cpu[i]: " << var_cpu[i] << std::endl;
  // }

  float *mean_gpu;
  float *var_gpu;

  CHECK_CUDA(cudaMalloc((void **)&mean_gpu, N * C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&var_gpu, N * C * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(mean_gpu, mean_cpu, N * C * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(var_gpu, var_cpu, N * C * sizeof(float), cudaMemcpyHostToDevice));

  auto in_gpu = in_t->gpu_buf;
  auto out_gpu = out_t->gpu_buf;
  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  // dim3 gridDim(out_numel);
  dim3 blockDim(THREADS_PER_BLOCK);

  std::cout << "===========================" << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "C: " << C << std::endl;
  std::cout << "H: " << H << std::endl;
  std::cout << "W: " << W << std::endl;
  std::cout << "===========================" << std::endl;

  CHECK_CUDA(cudaDeviceSynchronize());

  // int *p_N = (int *)malloc(sizeof(int));
  // int *p_C = (int *)malloc(sizeof(int));
  // int *p_H = (int *)malloc(sizeof(int));
  // int *p_W = (int *)malloc(sizeof(int));

  // p_N[0] = N;
  // p_C[0] = C;
  // p_H[0] = H;
  // p_W[0] = W;

  // CHECK_CUDA(cudaMalloc((void **)&mean_gpu, sizeof(int)));
  // CHECK_CUDA(cudaMalloc((void **)&var_gpu, sizeof(int)));
  // CHECK_CUDA(cudaMalloc((void **)&mean_gpu, sizeof(int)));
  // CHECK_CUDA(cudaMalloc((void **)&var_gpu, sizeof(int)));

  // instancenorm2d_kernel_v2<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, mean_gpu, var_gpu);

  instancenorm2d_kernel_v2<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, mean_gpu, var_gpu, N, C, H, W);
  // instancenorm2d_kernel_v2<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, mean_gpu, var_gpu, p_N, p_C, p_H, p_W);

  CHECK_CUDA(cudaDeviceSynchronize());

  // free(mean_cpu);
  // free(var_cpu);

  // CHECK_CUDA(cudaFree(mean_gpu));
  // CHECK_CUDA(cudaFree(var_gpu));

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h = 0; h < H; h++)
      {
        for (int w = 0; w < W; w++)
        {
          int out_idx = n * C * H * W + c * H * W + h * W + w;
          int in_idx = n * C * H * W + c * H * W + h * W + w;
          int mean_var_idx = n * C + c;
          float mean = mean_cpu[mean_var_idx];
          float var = var_cpu[mean_var_idx];
          out[out_idx] = (in[in_idx] - mean) / sqrt(var + 1e-5) * weight_cpu[c] + bias_cpu[c];

          // if (out_idx == 0)
          // {
          //   std::cout << "===========================" << std::endl;
          //   std::cout << "out_idx: " << out_idx << std::endl;
          //   std::cout << "out[out_idx]: " << out[out_idx] << std::endl;
          //   std::cout << "in[in_idx]: " << in[in_idx] << std::endl;
          //   std::cout << "n: " << n << std::endl;
          //   std::cout << "c: " << c << std::endl;
          //   std::cout << "h: " << h << std::endl;
          //   std::cout << "w: " << w << std::endl;
          //   std::cout << "mean: " << mean << std::endl;
          //   std::cout << "var: " << var << std::endl;
          //   std::cout << "weight: " << weight_cpu[c] << std::endl;
          //   std::cout << "bias: " << bias_cpu[c] << std::endl;
          //   std::cout << "===========================" << std::endl;
          // }

          if (out_idx == 10048)
          {
            std::cout << "===========================" << std::endl;
            std::cout << "out_idx: " << out_idx << std::endl;
            std::cout << "out[out_idx]: " << out[out_idx] << std::endl;
            std::cout << "in[in_idx]: " << in[in_idx] << std::endl;
            std::cout << "n: " << n << std::endl;
            std::cout << "c: " << c << std::endl;
            std::cout << "h: " << h << std::endl;
            std::cout << "w: " << w << std::endl;
            std::cout << "mean: " << mean << std::endl;
            std::cout << "var: " << var << std::endl;
            std::cout << "weight: " << weight_cpu[c] << std::endl;
            std::cout << "bias: " << bias_cpu[c] << std::endl;
            std::cout << "===========================" << std::endl;
          }
        }
      }
    }
  }

  float *out_from_gpu = (float *)malloc(out_numel * sizeof(float));

  CHECK_CUDA(
      cudaMemcpy(out_from_gpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "out_numel: " << out_numel << std::endl;

  int wrong_n = 0;

  for (int ii = 0; ii < out_numel; ii++)
  {
    if (abs(out[ii] - out_from_gpu[ii]) > 0.1f)
    {
      wrong_n++;
    }
    // printf("ii: %d\n", ii);
    // printf("out[ii]: %.4f\n", out[ii]);
    // printf("out_from_gpu[ii]: %.4f\n", out_from_gpu[ii]);
  }

  std::cout << "wrong_n: " << wrong_n << std::endl;

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(
  //     cudaMemcpy(out_t->gpu_buf, out_t->buf, out_numel * sizeof(float), cudaMemcpyHostToDevice));
  // out_t->free_cpu_buf();

  clock_t end_time2 = clock();

  double total_instancenorm2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_instancenorm2d_time: %.10f seconds\n", total_instancenorm2d_time);
}

static void instancenorm2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                              Tensor *bias_t)
{
  std::cout << "instancenorm2d on CPU!!!" << std::endl;

  clock_t start_time2 = clock();

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  in_t->buf = (float *)malloc(in_numel * sizeof(float));
  out_t->buf = (float *)malloc(out_numel * sizeof(float));
  weight_t->buf = (float *)malloc(weight_numel * sizeof(float));
  bias_t->buf = (float *)malloc(bias_numel * sizeof(float));

  CHECK_CUDA(
      cudaMemcpy(in_t->buf, in_t->gpu_buf, in_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(out_t->buf, out_t->gpu_buf, out_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(weight_t->buf, weight_t->gpu_buf, weight_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(bias_t->buf, bias_t->gpu_buf, bias_numel * sizeof(float), cudaMemcpyDeviceToHost));

  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int N = in_t->shape[0]; //=out_t->shape[0];
  int C = in_t->shape[1]; //=out_t->shape[1];
  int H = in_t->shape[2]; //=out_t->shape[2];
  int W = in_t->shape[3]; //=out_t->shape[3];

  clock_t start_time = clock();

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

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  CHECK_CUDA(
      cudaMemcpy(out_t->gpu_buf, out_t->buf, out_numel * sizeof(float), cudaMemcpyHostToDevice));
  out_t->free_cpu_buf();

  clock_t end_time2 = clock();

  double total_instancenorm2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_instancenorm2d_time: %.10f seconds\n", total_instancenorm2d_time);
}

static void instancenorm2d_v3(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                              Tensor *bias_t)
{
  std::cout << "instancenorm2d on CPU!!!" << std::endl;

  clock_t start_time2 = clock();

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  in_t->buf = (float *)malloc(in_numel * sizeof(float));
  out_t->buf = (float *)malloc(out_numel * sizeof(float));
  weight_t->buf = (float *)malloc(weight_numel * sizeof(float));
  bias_t->buf = (float *)malloc(bias_numel * sizeof(float));

  CHECK_CUDA(
      cudaMemcpy(in_t->buf, in_t->gpu_buf, in_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(out_t->buf, out_t->gpu_buf, out_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(weight_t->buf, weight_t->gpu_buf, weight_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(bias_t->buf, bias_t->gpu_buf, bias_numel * sizeof(float), cudaMemcpyDeviceToHost));

  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int N = in_t->shape[0]; //=out_t->shape[0];
  int C = in_t->shape[1]; //=out_t->shape[1];
  int H = in_t->shape[2]; //=out_t->shape[2];
  int W = in_t->shape[3]; //=out_t->shape[3];

  // input (N, C, H, W)
  // weight (C)
  // bias (C)
  // output(N, C, H, W)

  float *mean_arr = (float *)malloc(N * C * sizeof(float));
  float *var_arr = (float *)malloc(N * C * sizeof(float));

  clock_t start_time = clock();

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      float e = 0.0f, v = 0.0f;

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
      mean_arr[n * C + c] = e;

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
      var_arr[n * C + c] = v;
    }
  }

  clock_t end_time3 = clock();

  double compute_mean_var_time = (double)(end_time3 - start_time) / CLOCKS_PER_SEC;
  printf("compute_mean_var_time: %.10f seconds\n", compute_mean_var_time);

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h = 0; h < H; h++)
      {
        for (int w = 0; w < W; w++)
        {
          int out_idx = n * C * H * W + c * H * W + h * W + w;
          int in_idx = n * C * H * W + c * H * W + h * W + w;
          int mean_var_idx = n * C + c;
          float mean = mean_arr[mean_var_idx];
          float var = var_arr[mean_var_idx];
          out[out_idx] = (in[in_idx] - mean) / sqrt(var + 1e-5) * weight[c] + bias[c];
        }
      }
    }
  }

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  CHECK_CUDA(
      cudaMemcpy(out_t->gpu_buf, out_t->buf, out_numel * sizeof(float), cudaMemcpyHostToDevice));
  out_t->free_cpu_buf();

  clock_t end_time2 = clock();

  double total_instancenorm2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_instancenorm2d_time: %.10f seconds\n", total_instancenorm2d_time);
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

  clock_t start_time2 = clock();

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

  // CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  //////
  int weight_numel = weight_t->get_elem();
  int bias_numel = bias_t->get_elem();

  float *weight_cpu = weight_t->buf;
  float *bias_cpu = bias_t->buf;

  auto weight_gpu = weight_t->gpu_buf;
  auto bias_gpu = bias_t->gpu_buf;

  // CHECK_CUDA(cudaMalloc(&weight_gpu, weight_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(weight_gpu, weight_cpu, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&bias_gpu, bias_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(bias_gpu, bias_cpu, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  clock_t start_time = clock();

  linear_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, weight_gpu, bias_gpu, B, M, N, K);

  CHECK_CUDA(cudaDeviceSynchronize());

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  // in_t->free_gpu_buf();
  // out_t->free_gpu_buf();
  // weight_t->free_gpu_buf();
  // bias_t->free_gpu_buf();

  clock_t end_time2 = clock();

  double total_linear_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_linear_time: %.10f seconds\n", total_linear_time);
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
  float in_val;
  // float max_val = -1000000000.0f;

  int h_idx, w_idx, in_idx;
  int kh, kw;
  int nc_idx = n * C * H_IN * W_IN + c * H_IN * W_IN;

  // for (kh = 0; kh < kH; kh++)
  // {
  //   for (kw = 0; kw < kW; kw++)
  //   {
  //     h_idx = h_out * kH + kh;
  //     w_idx = w_out * kW + kw;

  //     // if (h_idx >= 0 && h_idx < H_IN && w_idx >= 0 && w_idx < W_IN)
  //     // {
  //     //   int in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
  //     //   float in_val = in[in_idx];

  //     //   if (in_val > max_val)
  //     //   {
  //     //     max_val = in_val;
  //     //   }
  //     // }

  //     // in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
  //     in_idx = nc_idx + h_idx * W_IN + w_idx;
  //     // float in_val = in[in_idx];
  //     in_val = in[in_idx];

  //     max_val = max(max_val, in_val);

  //     // if (in_val > max_val)
  //     // {
  //     //   max_val = in_val;
  //     // }
  //   }
  // }
  // out[tid] = max_val;

  for (kw = 0; kw < kW; kw++)
  {
    for (kh = 0; kh < kH; kh++)
    {
      h_idx = h_out * kH + kh;
      w_idx = w_out * kW + kw;

      // if (h_idx >= 0 && h_idx < H_IN && w_idx >= 0 && w_idx < W_IN)
      // {
      //   int in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
      //   float in_val = in[in_idx];

      //   if (in_val > max_val)
      //   {
      //     max_val = in_val;
      //   }
      // }

      // in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + h_idx * W_IN + w_idx;
      in_idx = nc_idx + h_idx * W_IN + w_idx;
      // float in_val = in[in_idx];
      in_val = in[in_idx];

      max_val = max(max_val, in_val);

      // if (in_val > max_val)
      // {
      //   max_val = in_val;
      // }
    }
  }
  out[tid] = max_val;
}

static void maxpool2d_gpu(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  std::cout << "maxpool2d on GPU!!!" << std::endl;

  clock_t start_time2 = clock();

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

  // CHECK_CUDA(cudaMalloc(&in_gpu, in_numel * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(in_gpu, in_cpu, in_numel * sizeof(float), cudaMemcpyHostToDevice));

  // CHECK_CUDA(cudaMalloc(&out_gpu, out_numel * sizeof(float)));

  dim3 gridDim((out_numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  clock_t start_time = clock();

  maxpool2d_kernel<<<gridDim, blockDim>>>(in_gpu, out_gpu, N, C, H_IN, W_IN, kH, kW);

  CHECK_CUDA(cudaDeviceSynchronize());

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(cudaMemcpy(out_cpu, out_gpu, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  // in_t->free_gpu_buf();
  // out_t->free_gpu_buf();

  clock_t end_time2 = clock();

  double total_maxpool2d_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_maxpool2d_time: %.10f seconds\n", total_maxpool2d_time);
}

static void maxpool2d_v2(Tensor *in_t, Tensor *out_t, int kH, int kW)
{
  std::cout << "maxpool2d on CPU!!!" << std::endl;

  clock_t start_time2 = clock();

  int in_numel = in_t->get_elem();
  int out_numel = out_t->get_elem();

  // CHECK_CUDA(cudaMallocHost((void **)&in_t->buf, in_numel * sizeof(float)));
  // CHECK_CUDA(cudaMallocHost((void **)&out_t->buf, out_numel * sizeof(float)));

  in_t->buf = (float *)malloc(in_numel * sizeof(float));
  out_t->buf = (float *)malloc(out_numel * sizeof(float));

  CHECK_CUDA(
      cudaMemcpy(in_t->buf, in_t->gpu_buf, in_numel * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(out_t->buf, out_t->gpu_buf, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

  float *in = in_t->buf;
  float *out = out_t->buf;

  int N = in_t->shape[0];
  int C = in_t->shape[1];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int H_OUT = H_IN / kH; // =out_t->shape[2];
  int W_OUT = W_IN / kW; // =out_t->shape[3];

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h_out = 0; h_out < H_OUT; h_out++)
      {
        for (int w_out = 0; w_out < W_OUT; w_out++)
        {
          int first_in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + (h_out * kH) * W_IN + (w_out * kW);
          // assert(first_in_idx < in_numel);

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
                // assert(in_idx < in_numel);

                float in_val = in[in_idx];
                if (in_val > max_val)
                {
                  max_val = in_val;
                }
              }
            }
          }
          int out_idx = n * C * H_OUT * W_OUT + c * H_OUT * W_OUT + h_out * W_OUT + w_out;
          // assert(out_idx < out_numel);

          out[out_idx] = max_val;
        }
      }
    }
  }
  ///
  // CHECK_CUDA(
  //     cudaMemcpy(in_t->gpu_buf, in_t->buf, in_numel * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(out_t->gpu_buf, out_t->buf, out_numel * sizeof(float), cudaMemcpyHostToDevice));
  // in_t->free_cpu_buf();
  // out_t->free_cpu_buf();

  clock_t end_time2 = clock();

  double total_maxpool2d_cpu_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_maxpool2d_cpu_time: %.10f seconds\n", total_maxpool2d_cpu_time);
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

  clock_t start_time2 = clock();

  float *inout = inout_t->buf;
  int N = inout_t->get_elem();

  auto tmp_buf = inout_t->gpu_buf;

  // CHECK_CUDA(cudaMalloc(&tmp_buf, N * sizeof(float)));
  // CHECK_CUDA(cudaMemcpy(tmp_buf, inout, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 blockDim(THREADS_PER_BLOCK);

  clock_t start_time = clock();

  relu_kernel<<<gridDim, blockDim>>>(tmp_buf, N);

  CHECK_CUDA(cudaDeviceSynchronize());

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.10f seconds\n", elapsed_time);

  // CHECK_CUDA(cudaMemcpy(inout, tmp_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

  // inout_t->free_gpu_buf();

  clock_t end_time2 = clock();

  double total_relu_time = (double)(end_time2 - start_time2) / CLOCKS_PER_SEC;
  printf("total_relu_time: %.10f seconds\n", total_relu_time);
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
