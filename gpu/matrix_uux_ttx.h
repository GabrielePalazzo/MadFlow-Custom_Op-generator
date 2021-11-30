
#ifndef MATRIX_H_
#define MATRIX_H_

#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include <cuComplex.h>
using namespace tensorflow;

template <typename Device, typename T>
struct MatrixFunctor {
  void operator()(const Device& d, const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_11, double* out_final, const int nevents, const OpKernelContext* context);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct MatrixFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_11, double* out_final, const int nevents, const OpKernelContext* context);
};
#endif

//#include <thrust/complex.h>

#define COMPLEX_TYPE complex128//thrust::complex<double>

#endif