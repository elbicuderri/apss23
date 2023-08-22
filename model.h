#pragma once

#include "tensor.h"

void initialize_model(const char *parameter_fname);
void model_forward(float *input, float *output);
void finalize_model();