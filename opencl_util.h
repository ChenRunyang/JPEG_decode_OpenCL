#ifndef __Util__
#define __Util__
#ifdef __APPLE__
#include<OpenCL/opencl.h>
#else
#include<CL/cl.h>
#endif
#include<iostream>
#include<fstream>
#include<math.h>
#include<algorithm>
#endif

using namespace std;

void init_OpenCL();
const char *CLerrorstring(cl_int err);
void contextCallback(const char *err, const void *private_info, size_t cb, void *usr_data);
void contextCallback(cl_int);
void to_rgb_gpu(int16_t* ycbcr,int16_t* rgb,unsigned int y,unsigned int x,unsigned int channels);
void idct_gpu(int16_t *image,int16_t *det_image_gpu,unsigned int blocks);