#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
using namespace std;
extern "C" {
#include "bmp.h"
}

#include "occa.hpp"

void runOCCA(int width, int height)
{
  std::string mode = "CUDA";
  int platformID = 0;
  int deviceID   = 2;

  occa::device device;
  occa::kernel render;
  occa::memory o_host_image;

  // set up compute device
  device.setup(mode, platformID, deviceID);

  // Multiply by 4 here, since we need red, green and blue for each pixel
  size_t buffer_size = sizeof(int) * width * height ;

  // host alloc array
  int *host_image = (int *) malloc(buffer_size);

  // device alloc array
  o_host_image = device.malloc(buffer_size);

  occa::kernelInfo info;
  
  // bake height and width into kernel through compiler variables
  info.addDefine("width", width);
  info.addDefine("height", height);

  // build kernel (with optional compiler defines from info)
  render = device.buildKernelFromSource("render.occa",
					"render",
					info);

  // set thread array
  occa::dim inner(32,32);

  // round up number of work-group
  occa::dim outer( (width+inner.x-1) / inner.x , (height+inner.y-1) / inner.y);
  int dims = 2;
  
  render.setWorkingDims(dims, inner, outer);
 
  // initialize timer
  occa::initTimer(device);

  // set tic for start time
  occa::tic("render");

  // start mandelbrot task
  render(o_host_image); // <<<<<<<<<<<<<

  // set tic for end time
  occa::toc("render", render);

  // copy result to host
  o_host_image.copyTo(host_image);

  // host sums up iterations (thus flops)
  long long int iterationCount = 0;
  for(int n=0;n<width*height;++n)
    iterationCount += host_image[n];

  long long int dataMoved = width*height*sizeof(int);
  
  printf("flop count  = %lld\n", iterationCount*10);
  printf("memory moved= %lld\n", dataMoved);

  // Now write the file (fix this - host_image is now an array of ints)
  write_bmp("output.bmp", width, height, (char*)host_image);
  render.free();

  // print out kernel timings
  occa::printTimer();

  // garbage collection
  o_host_image.free();
  device.free();
  delete [] host_image;
}

int main(int argc, char ** argv) {

  // build mandelbrot
  runOCCA(1*2048, 1*1024);

  return 0;
}
