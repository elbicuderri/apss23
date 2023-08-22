#pragma once

#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

void parse_option(int, char **);
void print_help();

void *read_binary(const char *filename, size_t *size);
void write_binary(float *output, const char *filename, int size_);

double get_time();
void check_validation(const char *);