#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model.h"
#include "util.h"

int N;
int S, W, V;

char parameter_fname[100] = "./data/weights.bin";
char input_fname[100] = "./data/bins/input1N.bin";
char answer_fname[100] = "./data/bins/output1N.bin";
char output_fname[100] = "./output.bin";

int main(int argc, char **argv) {
  parse_option(argc, argv);
  float *input, *output;

  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  // Initilization and Reading inputs must be done in this block.   //
  ////////////////////////////////////////////////////////////////////

  // Get input from binary file
  size_t sz;

  fprintf(stderr, " Reading input from: %s\n", input_fname);
  input = (float *)read_binary(input_fname, &sz);
  if (sz % (1 << 16) != 0) {
    fprintf(stderr, " Wrong input tensor shape: %ld\n", sz);
  }

  N = sz >> 16;

  // Define output Tensor
  output = (float *)malloc(sizeof(float) * N * 2);

  // Initalize model
  initialize_model(parameter_fname);

  // Warmup
  if (W) {
    fprintf(stderr, " Warming up... \n");

    for (int i = 0; i < W; i++) {
      model_forward(input, output);
    }
  }

  ////////////////////////////////////////////////////////////////////
  // COMMUNICATION & COMPUTATION                                    //
  // All communication and computation must be done in this block.  //
  // It is free to use any number of nodes and gpus.                //
  ////////////////////////////////////////////////////////////////////

  double st = 0.0, et = 0.0;
  fprintf(stderr, " Start...");

  st = get_time();

  model_forward(input, output);

  et = get_time();
  fprintf(stderr, "  DONE!\n");
  fprintf(stderr, " ---------------------------------------------\n");
  fprintf(stderr, " Elapsed time : %lf s\n", et - st);
  fprintf(stderr, " Throughput   : %lf img/sec\n", (double)N / (et - st));

  if (S) {
    fprintf(stderr, " Saving output to: %s\n", output_fname);
    write_binary(output, output_fname, N * 2);
  }

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////

  finalize_model();

  if (V) {
    size_t sz_ans;
    float *answer = (float *)read_binary(answer_fname, &sz_ans);

    int diff = -1;
    for (int i = 0; i < N * 2; i++) {
      if (abs(*(output + i) - *(answer + i)) > 1e-3) {
        diff = i;
        break;
      }
    }
    if (diff < 0)
      fprintf(stderr, " Validation success!\n");
    else
      fprintf(stderr,
              " Validation fail: First mistmatch on index %d(output[i]=%f , "
              "answer[i]=%f)\n",
              diff, *(output + diff), *(answer + diff));
  }

  return EXIT_SUCCESS;
}
