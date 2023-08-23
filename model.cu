#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>

#include "model.h"
#include "util.h"

#define NUM_OF_THREADS 256

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
  // l2 = new Tensor({N, 256, 62, 64});

  output = new Tensor({N, 2});
  // output = new Tensor({N, 2});
  // output = new Tensor({N, 1, 1, 2});

  std::cout << "========================" << std::endl;
  std::cout << "initialize_model" << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "parameter_fname" << std::endl;
  std::cout << parameter_fname << std::endl;
  std::cout << "========================" << std::endl;

  // input = new Tensor({1, 256, 256});
  // output = new Tensor({2});
  // c1 = new Tensor({128, 254, 254});
  // i1 = new Tensor({128, 254, 254});
  // m1 = new Tensor({128, 127, 127});
  // c2 = new Tensor({256, 125, 125});
  // i2 = new Tensor({256, 125, 125});
  // m2 = new Tensor({256, 62, 62});

  // // m2 = new Tensor({N, 256 * 62, 62});

  // l1 = new Tensor({256, 62, 128});
  // // l1 = new Tensor({N, 256, 62, 128});
  // // l1 = new Tensor({N, 256 * 62, 128});

  // l2 = new Tensor({256, 62, 64});

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

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW);

static void maxpool2d_v2(Tensor *in_t, Tensor *out_t, int kH, int kW);

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

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(Tensor *inout_t);

void model_forward(float *inputN, float *outputN)
{
  printf("model_forward\n");
  printf("N: %d\n", N);
  // printf("N: %d\n", N);

  memcpy(input->buf, inputN, N * 256 * 256 * sizeof(float));

  // // For test
  // CHECK_CUDA(cudaMemcpy(input->buf,
  //                       inputN,
  //                       N * 256 * 256 * sizeof(float),
  //                       cudaMemcpyHostToHost));

  conv2d_v2(input, c1, conv0_weight, conv0_bias);

  instancenorm2d_v2(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

  maxpool2d_v2(i1, m1, 2, 2);

  relu(m1);

  conv2d_v2(m1, c2, conv1_weight, conv1_bias);

  instancenorm2d_v2(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);

  maxpool2d_v2(i2, m2, 2, 2);

  relu(m2);

  // m2 {N, 256, 62, 62}
  // l1 {N, 256, 62, 128}
  // m2->reshape({N, 256 * 62, 62});
  linear_v2(m2, l1, linear1_weight, linear1_bias);

  // m2->reshape({256, 62, 62});
  // linear(m2, l1, linear1_weight, linear1_bias);

  relu(l1);

  // l1 {N, 256, 62, 128}
  // l2 {N, 256, 62, 64}
  // l1->reshape({256, 62, 128});
  // linear_v2(l1, l2, linear2_weight, linear2_bias);
  linear_v2(l1, l2, linear2_weight, linear2_bias);

  // l2->reshape({N, 1, 1, 1015808});
  // l2->reshape({N, 1, 256 * 62 * 64});
  linear_v2(l2, output, linear3_weight, linear3_bias);
  // linear(l2, output, linear3_weight, linear3_bias);
  // output->reshape({N, 2});

  memcpy(outputN, output->buf, N * 2 * sizeof(float));
}

// void model_forward(float *inputN, float *outputN)
// {
//   printf("model_forward\n");
//   printf("N: %d\n", N);
//   // printf("N: %d\n", N);

//   for (int idx = 0; idx < N; idx++)
//   {
//     // memcpy(input->buf, inputN + 256 * 256 * idx, 256 * 256 * sizeof(float));

//     // For test
//     CHECK_CUDA(cudaMemcpy(input->buf,
//                           inputN + 256 * 256 * idx,
//                           256 * 256 * sizeof(float),
//                           cudaMemcpyHostToHost));

//     // std::cout << "========================" << std::endl;
//     // std::cout << "input shape" << std::endl;
//     // input->print_shape();
//     // std::cout << "========================" << std::endl;

//     // conv2d(input, c1, conv0_weight, conv0_bias);

//     input->reshape({N, 1, 256, 256});

//     // std::cout << "========================" << std::endl;
//     // std::cout << "input shape" << std::endl;
//     // input->print_shape();
//     // std::cout << "========================" << std::endl;

//     conv2d_v2(input, c1, conv0_weight, conv0_bias);

//     // std::cout << "========================" << std::endl;
//     // std::cout << "c1 shape" << std::endl;
//     // c1->print_shape();
//     // std::cout << "========================" << std::endl;

//     // c1->reshape({128, 254, 254});
//     // instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

//     instancenorm2d_v2(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);

//     // i1->reshape({128, 254, 254});

//     // maxpool2d(i1, m1, 2, 2);

//     maxpool2d_v2(i1, m1, 2, 2);
//     // m1->reshape({128, 127, 127});

//     relu(m1);

//     // conv2d(m1, c2, conv1_weight, conv1_bias);
//     conv2d_v2(m1, c2, conv1_weight, conv1_bias);

//     // instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
//     instancenorm2d_v2(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);

//     // maxpool2d(i2, m2, 2, 2);
//     maxpool2d_v2(i2, m2, 2, 2);

//     relu(m2);
//     // m2->reshape({256, 62, 62});

//     // m2 {N, 256, 62, 62}
//     // linear(m2, l1, linear1_weight, linear1_bias);
//     linear_v2(m2, l1, linear1_weight, linear1_bias);
//     // l1->reshape({256, 62, 128});

//     relu(l1);

//     // linear(l1, l2, linear2_weight, linear2_bias);
//     linear_v2(l1, l2, linear2_weight, linear2_bias);

//     l2->reshape({1, 1015808});
//     // linear(l2, output, linear3_weight, linear3_bias);
//     linear_v2(l2, output, linear3_weight, linear3_bias);

//     memcpy(outputN + 2 * idx, output->buf, 2 * sizeof(float));
//     // memcpy(outputN + 2 * idx, output->buf, N * 2 * sizeof(float));
//   }
// }

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

static void conv2d_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  // weight {128, 1, 3, 3} (K, C, kH, kW)

  // int out_channel = weight_t->shape[0];
  // int in_channel = weight_t->shape[1];

  int kH = weight_t->shape[2];
  int kW = weight_t->shape[3];

  // std::cout << "========================" << std::endl;
  // std::cout << "out_channel: " << out_channel << std::endl;
  // std::cout << "in_channel: " << in_channel << std::endl;
  // std::cout << "kH: " << kH << std::endl;
  // std::cout << "kW: " << kW << std::endl;
  // std::cout << "========================" << std::endl;

  // int kH = weight_t->shape[0]

  int C_IN = weight_t->shape[1];  //=in_t->shape[0];
  int C_OUT = weight_t->shape[0]; //=out_t->shape[0];

  int N = in_t->shape[0];

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];

  int H_OUT = H_IN - kH + 1; //=out_t->shape[1];
  int W_OUT = W_IN - kW + 1; //=out_t->shape[2];

  // std::cout << "========================" << std::endl;
  // std::cout << "N: " << N << std::endl;
  // std::cout << "C_IN: " << C_IN << std::endl;
  // std::cout << "C_OUT: " << C_OUT << std::endl;
  // std::cout << "H_IN: " << H_IN << std::endl;
  // std::cout << "W_IN: " << W_IN << std::endl;
  // std::cout << "H_OUT: " << H_OUT << std::endl;
  // std::cout << "W_OUT: " << W_OUT << std::endl;
  // std::cout << "========================" << std::endl;

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

  // int C = in_t->shape[0]; //=out_t->shape[0];
  // int H = in_t->shape[1]; //=out_t->shape[1];
  // int W = in_t->shape[2]; //=out_t->shape[2];

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

// __global__ void linear_kernel(float *output, float *input, float *weight, float *bias,
//                               int batch, int in_f, int out_f)
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int output_numel = batch * out_f;
//   if (tid >= output_numel)
//     return;

//   int k_idx = tid % out_f;
//   int n_idx = tid / out_f;

//   float sum = 0.0f;

//   for (int i = 0; i < in_f; i++)
//   {
//     int input_index = n_idx * in_f + i;
//     int weight_index = k_idx * in_f + i;
//     sum += input[input_index] * weight[weight_index];
//   }

//   sum += bias[k_idx];
//   output[tid] = sum;
// }

// static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
//                    Tensor *bias_t)
// {
//   float *cpu_in = in_t->buf;
//   // float *out = out_t->buf;
//   float *cpu_weight = weight_t->buf;
//   float *cpu_bias = bias_t->buf;

//   // input (N, C, H, W) -> (N, C * H * W)
//   // weight (K, C * H * W)
//   // bias (K)
//   // output (N, K)

//   // int batch = in_t->shape[0];
//   // int C = in_t->shape[1];
//   // int H = in_t->shape[2];
//   // int W = in_t->shape[3];

//   int batch = 1;
//   int C = in_t->shape[0];
//   int H = in_t->shape[1];
//   int W = in_t->shape[2];

//   std::cout << "========================" << std::endl;
//   std::cout << "N: " << batch << std::endl;
//   std::cout << "C: " << C << std::endl;
//   std::cout << "H: " << H << std::endl;
//   std::cout << "W: " << W << std::endl;
//   std::cout << "========================" << std::endl;

//   // int in_f = C * H * W;
//   int in_f = weight_t->shape[0];
//   int out_f = weight_t->shape[1];

//   std::cout << "========================" << std::endl;
//   std::cout << "weight_t->shape[0]: " << weight_t->shape[0] << std::endl;
//   std::cout << "weight_t->shape[1]: " << weight_t->shape[1] << std::endl;
//   std::cout << "========================" << std::endl;

//   int in_numel = in_t->get_elem();
//   int out_numel = out_t->get_elem();

//   int weight_numel = weight_t->get_elem();
//   int bias_numel = bias_t->get_elem();

//   std::cout << "========================" << std::endl;
//   // std::cout << "in_numel: " << in_numel << std::endl;
//   // std::cout << "out_numel: " << out_numel << std::endl;
//   // std::cout << "weight_numel: " << weight_numel << std::endl;
//   // std::cout << "bias_numel: " << bias_numel << std::endl;

//   std::cout << "========================" << std::endl;
//   std::cout << "in_t->shape[0]: " << in_t->shape[0] << std::endl;
//   std::cout << "in_t->shape[1]: " << in_t->shape[1] << std::endl;
//   std::cout << "========================" << std::endl;

//   std::cout << "in_f: " << in_f << std::endl;
//   std::cout << "out_f: " << out_f << std::endl;
//   std::cout << "========================" << std::endl;

//   std::cout << "========================" << std::endl;
//   std::cout << "out_t->shape[0]: " << out_t->shape[0] << std::endl;
//   std::cout << "out_t->shape[1]: " << out_t->shape[1] << std::endl;
//   std::cout << "========================" << std::endl;

//   auto gpu_in = in_t->gpu_buf;
//   CHECK_CUDA(cudaMalloc(&gpu_in, in_numel * sizeof(float)));
//   CHECK_CUDA(cudaMemcpy(gpu_in, cpu_in, in_numel * sizeof(float), cudaMemcpyHostToDevice));

//   auto gpu_weight = weight_t->gpu_buf;
//   CHECK_CUDA(cudaMalloc(&gpu_weight, weight_numel * sizeof(float)));
//   CHECK_CUDA(cudaMemcpy(gpu_weight, cpu_weight, weight_numel * sizeof(float), cudaMemcpyHostToDevice));

//   auto gpu_bias = bias_t->gpu_buf;
//   CHECK_CUDA(cudaMalloc(&gpu_bias, bias_numel * sizeof(float)));
//   CHECK_CUDA(cudaMemcpy(gpu_bias, cpu_bias, bias_numel * sizeof(float), cudaMemcpyHostToDevice));

//   auto gpu_out = out_t->gpu_buf;
//   CHECK_CUDA(cudaMalloc(&gpu_out, out_numel * sizeof(float)));
//   // CHECK_CUDA(cudaMemcpy(gpu_out, inout, out_numel * sizeof(float), cudaMemcpyHostToDevice));

//   dim3 gridDim((out_numel + NUM_OF_THREADS - 1) / NUM_OF_THREADS);
//   dim3 blockDim(NUM_OF_THREADS);

//   linear_kernel<<<gridDim, blockDim>>>(gpu_out, gpu_in, gpu_weight, gpu_bias, batch, in_f, out_f);

//   auto cpu_out = out_t->buf;
//   CHECK_CUDA(cudaMemcpy(cpu_out, gpu_out, out_numel * sizeof(float), cudaMemcpyDeviceToHost));

//   in_t->free_gpu_buf();
//   weight_t->free_gpu_buf();
//   bias_t->free_gpu_buf();
//   out_t->free_gpu_buf();
// }

// int N = batch;
// int H = input_height;
// int W = input_width;
// int C = in_channel;
// int K = out_channel;
// int L = C * H * W;

// for (int n = 0; n < N; n++) {
// 	for (int k = 0; k < K; k++) {
// 		float sum = 0.0f;
// 		for (int i = 0; i < L; i++) {
// 			int input_index = n * L + i;
// 			int weight_index = k * L + i;
// 			float s = input[input_index] * weight[weight_index];
// 			sum += s;
// 		}
// 		sum += bias[k];
// 		int output_index = n * K + k;
// 		output[output_index] = sum;

// 	}
// }

static void linear_v2(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                      Tensor *bias_t)
{
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int K = weight_t->shape[0]; // in_t의 마지막 차원
  int N = weight_t->shape[1]; // out_t의 마지막 차원

  // int N = in_t->get_elem() / H_IN; //=out_t->get_elem()/H_OUT

  int B = in_t->shape[0];
  // int S1 = in_t->shape[1];
  // int S2 = in_t->shape[2];
  int M = in_t->shape[1];
  // int K = in_t->shape[3];
  // assert(in_t->shape[3] == K);

  // int M = S1 * S2;

  std::cout << "========================" << std::endl;
  std::cout << "linear_v2" << std::endl;
  std::cout << "B: " << B << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "K: " << K << std::endl;

  // std::cout << "S1: " << S1 << std::endl;
  // std::cout << "S2: " << S2 << std::endl;

  std::cout << "========================" << std::endl;

  // (B, M, K) @ (K, N) -> (B, M, N)

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

  std::cout << "========================" << std::endl;
  std::cout << "linear" << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "N: " << N << std::endl;
  std::cout << "K: " << K << std::endl;
  std::cout << "========================" << std::endl;

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

  // std::cout << "========================" << std::endl;
  // std::cout << "N: " << N << std::endl;
  // std::cout << "C: " << C << std::endl;
  // std::cout << "H_IN: " << H_IN << std::endl;
  // std::cout << "W_IN: " << W_IN << std::endl;
  // std::cout << "H_OUT: " << H_OUT << std::endl;
  // std::cout << "W_OUT: " << W_OUT << std::endl;
  // std::cout << "in_numel: " << in_numel << std::endl;
  // std::cout << "out_numel: " << out_numel << std::endl;
  // std::cout << "========================" << std::endl;

  for (int n = 0; n < N; n++)
  {
    for (int c = 0; c < C; c++)
    {
      for (int h_out = 0; h_out < H_OUT; h_out++)
      {
        for (int w_out = 0; w_out < W_OUT; w_out++)
        {
          int first_in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + (h_out * kH) * H_IN + (w_out * kW);
          // std::cout << "first_in_idx: " << first_in_idx << std::endl;
          assert(first_in_idx < in_numel);
          float max_val = in[first_in_idx];
          for (int kh = 0; kh < kH; kh++)
          {
            for (int kw = 0; kw < kW; kw++)
            {
              int in_idx = n * C * H_IN * W_IN + c * H_IN * W_IN + (h_out * kH + kh) * H_IN + (w_out * kW + kw);
              // std::cout << "in_idx: " << in_idx << std::endl;
              assert(in_idx < in_numel);

              float in_val = in[in_idx];
              if (in_val > max_val)
              {
                max_val = in_val;
              }
            }
          }
          int out_idx = n * C * H_OUT * W_OUT + c * H_OUT * W_OUT + h_out * W_OUT + w_out;
          // std::cout << "out_idx: " << out_idx << std::endl;
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

// __global__ void relu_kernel(float *X, int N)
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= N)
//     return;

//   float val = X[tid] > 0.0f ? X[tid] : 0.0f;
//   X[tid] = val;
// }

// static void relu(Tensor *inout_t)
// {
//   // printf("relu on GPU!!!\n");

//   float *inout = inout_t->buf;
//   int N = inout_t->get_elem();

//   // float *tmp_buf = inout_t->gpu_buf;
//   auto tmp_buf = inout_t->gpu_buf;

//   CHECK_CUDA(cudaMalloc(&tmp_buf, N * sizeof(float)));
//   CHECK_CUDA(cudaMemcpy(tmp_buf, inout, N * sizeof(float), cudaMemcpyHostToDevice));

//   dim3 gridDim((N + NUM_OF_THREADS - 1) / NUM_OF_THREADS);
//   dim3 blockDim(NUM_OF_THREADS);

//   relu_kernel<<<gridDim, blockDim>>>(tmp_buf, N);

//   CHECK_CUDA(cudaMemcpy(inout, tmp_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

//   inout_t->free_gpu_buf();
// }

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
