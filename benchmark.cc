/* Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdio>
#include <cstdlib>
#include <string>

#include "test.h"


#define RUY_TEST_LHSSCALAR uint8_t
#define RUY_TEST_RHSSCALAR uint8_t
#define RUY_TEST_ACCUMSCALAR int32_t
#define RUY_TEST_DSTSCALAR uint8_t

namespace ruy {

using LhsScalar = RUY_TEST_LHSSCALAR;
using RhsScalar = RUY_TEST_RHSSCALAR;
using AccumScalar = RUY_TEST_ACCUMSCALAR;
using DstScalar = RUY_TEST_DSTSCALAR;

// TestSetType = TestSet<LhsScalar, RhsScalar>, BasicSpec<AccumScalar, DstScalar>>
using TestSetType =
    TestSet<LhsScalar, RhsScalar, BasicSpec<AccumScalar, DstScalar>>;

struct BenchmarkShape {
  int rows; // M
  int depth; // K
  int cols; // N
  int symm_lhs;
  int symm_rhs;
};

struct Shape {
  int rows;
  int depth;
  int cols;
  Shape(int _rows, int _depth, int _cols):
  rows(_rows), depth(_depth), cols(_cols)
  {
  }
};

template <typename TestSetType>
std::vector<std::unique_ptr<TestResult<DstScalar>>> BenchmarkRCC(
    const BenchmarkShape& shape) {
  TestSetType test_set;
  test_set.rows = shape.rows;
  test_set.depth = shape.depth;
  test_set.cols = shape.cols;
  test_set.lhs_order = Order::kRowMajor; // It is row major
  test_set.rhs_order = Order::kColMajor; // It is col major
  test_set.dst_order = Order::kColMajor; // It is col major
  test_set.layout_style = LayoutStyle::kPackedLinear; // What is kPackedLinear.
  test_set.benchmark = true; // The benchmark is true here.
  const int asymmetry_lhs = shape.symm_lhs ? 0 : 1; // If shape.symm_lhs is 0, then asymmetry_lhs is 0
  const int asymmetry_rhs = shape.symm_rhs ? 0 : 1;
  test_set.lhs_zero_point = SymmetricZeroPoint<LhsScalar>() + asymmetry_lhs;
  test_set.rhs_zero_point = SymmetricZeroPoint<RhsScalar>() + asymmetry_rhs;
  test_set.use_specified_zero_points = true; // In the benchmark, we use specified zero points by default.
  // Whether each channel uses a different quantization parameters.
  test_set.perchannel = GetBoolEnvVarOrFalse("PERCHANNEL"); // Define the per channel as environment variable.
  // Whether the left matrix is prepacked.
  test_set.benchmark_prepack_lhs = GetBoolEnvVarOrFalse("PREPACK_LHS");
  // Whether the right matrix is prepacked.
  test_set.benchmark_prepack_rhs = GetBoolEnvVarOrFalse("PREPACK_RHS");
  test_set.Run();
  return std::move(test_set.results);
}



void Benchmark() {

    // std::is_float_point checks whether T is a floating-point type.
    // If LhsScalar is float, then symm_lhs is true.
  const bool symm_lhs = std::is_floating_point<LhsScalar>::value ||
                        GetBoolEnvVarOrFalse("SYMM_LHS");
  const bool symm_rhs = std::is_floating_point<RhsScalar>::value ||
                        GetBoolEnvVarOrFalse("SYMM_RHS");

  const bool benchmark_cubic = GetBoolEnvVarOrFalse("RUY_BENCHMARK_CUBIC");
  std::vector<BenchmarkShape> shapes;

  // Often 8 is used for this multiplier, but to check teeny sizes one can
  // use 1.
  static constexpr int cubic_size_multiplier = 8;



  if (benchmark_cubic) {
#ifdef _WIN32
    _putenv_s("QUICK_BENCHMARK", "1");
#else
    setenv("QUICK_BENCHMARK", "1", 0);
#endif

    //td::map<>

    std::vector<Shape> mobilenetv1;

    mobilenetv1.push_back(Shape(112 * 112, 32, 3 * 3 * 3));
    mobilenetv1.push_back(Shape(56 * 56, 128, 64));
    mobilenetv1.push_back(Shape(56 * 56, 128, 128));
    mobilenetv1.push_back(Shape(28 * 28, 256, 128));
    mobilenetv1.push_back(Shape(28 * 28, 256, 256));
    mobilenetv1.push_back(Shape(14 * 14, 512, 256));
    mobilenetv1.push_back(Shape(14 * 14, 512, 512));
    mobilenetv1.push_back(Shape(7 * 7, 1024, 512));
    mobilenetv1.push_back(Shape(7 * 7, 1024, 1024));

    // Since the
    std::vector<Shape> vgg16;
    vgg16.push_back(Shape(224 * 224, 3 * 3 * 64, 64));
    vgg16.push_back(Shape(224 * 224, 3 * 3 * 64, 128));
    vgg16.push_back(Shape(112 * 112, 3 * 3 * 128, 128));
    vgg16.push_back(Shape(112 * 112, 3 * 3 * 128, 256));
    vgg16.push_back(Shape(56 * 56, 3 * 3 * 256, 256));
    vgg16.push_back(Shape(56 * 56, 3 * 3 * 256, 256));
    vgg16.push_back(Shape(56 * 56, 3 * 3 * 256, 512));
    vgg16.push_back(Shape(28 * 28, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(28 * 28, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(28 * 28, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(28 * 28, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(14 * 14, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(14 * 14, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(14 * 14, 3 * 3 * 512, 512));
    vgg16.push_back(Shape(14 * 14, 3 * 3 * 512, 512));

#if 0
    std::vector<int> sizes;
    for (int i = 2 * cubic_size_multiplier; i <= (512 * cubic_size_multiplier);
         i *= 2) {
      sizes.push_back(i);
      if (i < (512 * cubic_size_multiplier)) {
        sizes.push_back(i * 3 / 2);
      }
    }
#endif
    for (auto & layer : vgg16) {
      BenchmarkShape shape;
      shape.rows = layer.rows;
      shape.cols = layer.cols;
      shape.depth = layer.depth;
      shape.symm_lhs = symm_lhs;
      shape.symm_rhs = symm_rhs;
      shapes.push_back(shape);
    }
  } else {
    BenchmarkShape shape;
    shape.rows = GetIntEnvVarOrZero("ROWS");
    shape.cols = GetIntEnvVarOrZero("COLS");
    shape.depth = GetIntEnvVarOrZero("DEPTH");
    if (!shape.rows || !shape.depth || !shape.cols) {
      fprintf(stderr,
              "Please specify positive sizes with these env vars: ROWS, DEPTH, "
              "COLS.\n");
      exit(1);
    }
    shape.symm_lhs = symm_lhs;
    shape.symm_rhs = symm_rhs;
    shapes.push_back(shape);
  }

  for (int i = 0; i < shapes.size(); i++) {
    const auto& shape = shapes[i];
    const auto& results = BenchmarkRCC<TestSetType>(shape);
    if (i == 0) {
#if 0
      if (benchmark_cubic) {
        printf("size");
        for (const auto& result : results) {
          printf(",%s", PathName(*result).c_str());
        }
        printf("\n");
      } else {
        printf("path,shape,Gop/s\n");
      }
      fflush(stdout);
#endif
      if(benchmark_cubic){
          printf("Rows, Depth, Cols");

          for(const auto& result: results){
              printf(" ,%s", PathName(*result).c_str());
          }
          printf("\n");
      } else {
          printf("path,shape,Gop/s\n");
      }
    }

    if (benchmark_cubic) {

      printf("%d, %d, %d", shape.rows, shape.depth, shape.cols);
      for (const auto& result : results) {
        //printf(",%.4g", 2.0e-9 * shape.rows * shape.cols * shape.depth /
        //                    result->latency);
        printf(",%4g", 1.0e3 * result->latency);
        if (GetBoolEnvVarOrFalse("RUY_BENCHMARK_PMU")) {
          printf(",%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g",
                 result->l1_refill_rate, result->l2_refill_rate,
                 result->l3_refill_rate, result->l1tlb_refill_rate,
                 result->l2tlb_refill_rate, result->mispred_rate,
                 result->frontend_stall_rate, result->backend_stall_rate);
        }
      }
      printf("\n");
      fflush(stdout);
    } else {
      for (const auto& result : results) {
        printf(
            "%s,%dx%dx%d,%.4g", PathName(*result).c_str(), shape.rows,
            shape.depth, shape.cols,
            2.0e-9 * shape.rows * shape.cols * shape.depth / result->latency);
        if (GetBoolEnvVarOrFalse("RUY_BENCHMARK_PMU")) {
          printf(",%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g,%.3g",
                 result->l1_refill_rate, result->l2_refill_rate,
                 result->l3_refill_rate, result->l1tlb_refill_rate,
                 result->l2tlb_refill_rate, result->mispred_rate,
                 result->frontend_stall_rate, result->backend_stall_rate);
        }
        printf("\n");
      }
      fflush(stdout);
    }
  }
}

}  // namespace ruy

int main() { ruy::Benchmark(); }
