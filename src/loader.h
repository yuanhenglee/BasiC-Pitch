#pragma once

#include "typedef.h"
#include "layer.h"
#include <vector>
#include <string>

void loadDefaultKernel(CQTKernelMat &kernel);

void loadDefaultLowPassFilter( Vectorf &filter_kernel);

void loadCNNModel(std::vector<Layer*> &layers, std::string model_name);

Vectorf getExampleAudio();