#include "util.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <unistd.h>
#include <vector>

using namespace std;

extern int N;        // defined in main.cpp
extern bool V, S, W; // defined in main.cpp
extern char input_fname[], output_fname[], answer_fname[], parameter_fname[];

void parse_option(int argc, char **argv) {
  int opt;
  while ((opt = getopt(argc, argv, "i:o:a:p:vswh")) != -1) {
    switch (opt) {
    case 'i':
      strcpy(input_fname, optarg);
      break;
    case 'o':
      strcpy(output_fname, optarg);
      break;
    case 'a':
      strcpy(answer_fname, optarg);
      break;
    case 'p':
      strcpy(parameter_fname, optarg);
      break;
    case 'v':
      V = true;
      break;
    case 's':
      S = true;
      break;
    case 'w':
      W = true;
      break;
    case 'h':
      print_help();
      exit(-1);
      break;
    default:
      print_help();
      exit(-1);
      break;
    }
  }

  fprintf(stderr, "\n Model : BrainTumorModel\n");
  fprintf(stderr, " =============================================\n");
  fprintf(stderr, " Warming up : %s\n", W ? "ON" : "OFF");
  fprintf(stderr, " Validation : %s\n", V ? "ON" : "OFF");
  fprintf(stderr, " Save output tensor : %s\n", S ? "ON" : "OFF");
  fprintf(stderr, " ---------------------------------------------\n");
}

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[ERROR] Cannot open file \'%s\'\n", filename);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stderr, "[ERROR] Cannot read file \'%s\'\n", filename);
    exit(-1);
  }
  fclose(f);

  if (size != NULL)
    *size = (size_t)(size_ / 4); // float
  return buf;
}

void write_binary(float *output, const char *filename, int size_) {
  fprintf(stderr, " Writing output ... ");
  FILE *output_fp = (FILE *)fopen(filename, "w");

  char *tmp = (char *)output;
  for (int i = 0; i < 4 * size_; i++) {
    fprintf(output_fp, "%c", tmp[i]);
  }
  fclose(output_fp);
  fprintf(stderr, "DONE!\n");
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_help() {
  fprintf(stderr,
          " Usage: ./main [-i pth] [-o pth] [-p pth] [-a pth] [-vswh]\n");
  fprintf(stderr, " Options:\n");
  fprintf(stderr, "  -i : input binary path (default: data/bins/input1N.bin)\n");
  fprintf(stderr, "  -o : output binary path (default: output.bin)\n");
  fprintf(stderr, "  -p : parameter binary path (default: data/weights.bin)\n");
  fprintf(stderr,
          "  -a : answer binary path (default: data/bins/output1N.bin)\n");
  fprintf(
      stderr,
      "  -v : enable validate. compare with answer binary (default: off)\n");
  fprintf(stderr, "  -s : save generated sentences (default: off)\n");
  fprintf(stderr, "  -w : enable warmup (default: off)\n");
  fprintf(stderr, "  -h : print this page.\n");
}
