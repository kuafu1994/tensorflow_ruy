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
#include "im2col.h"


#define RUY_TEST_LHSSCALAR float
#define RUY_TEST_RHSSCALAR float
#define RUY_TEST_ACCUMSCALAR float
#define RUY_TEST_DSTSCALAR float

namespace ruy {

    static inline size_t compute_output_dimension(
            size_t padded_input_dimension,
            size_t kernel_dimension,
            size_t stride_dimension
    ) {
        return (padded_input_dimension - kernel_dimension) / stride_dimension + 1;
    }
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

// It should be OK now.
    struct ConvP {
        int input_height;
        int input_width;
        int kernel_height;
        int kernel_width;
        int stride;
        int input_channels;
        int output_channels;

        ConvP(int _ih, int _iw, int _kh, int _kw, int _stride, int _ic, int _oc):
                input_height(_ih), input_width(_iw), kernel_height(_kh), kernel_width(_kw),
                stride(_stride), input_channels(_ic), output_channels(_oc)
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
        // const int asymmetry_lhs = shape.symm_lhs ? 0 : 1; // If shape.symm_lhs is 0, then asymmetry_lhs is 0
        //const int asymmetry_rhs = shape.symm_rhs ? 0 : 1;
        const int asymmetry_lhs = 1;
        const int asymmetry_rhs = 1;
        test_set.lhs_zero_point = SymmetricZeroPoint<LhsScalar>() + asymmetry_lhs;
        test_set.rhs_zero_point = SymmetricZeroPoint<RhsScalar>() + asymmetry_rhs;
        test_set.use_specified_zero_points = true; // In the benchmark, we use specified zero points by default.
        // Whether each channel uses a different quantization parameters.
        test_set.perchannel = false;
        //test_set.perchannel = GetBoolEnvVarOrFalse("PERCHANNEL"); // Define the per channel as environment variable.
        // Whether the left matrix is prepacked.
        //test_set.benchmark_prepack_lhs = GetBoolEnvVarOrFalse("PREPACK_LHS");
        test_set.benchmark_prepack_lhs = false;
        // Whether the right matrix is prepacked.
        //test_set.benchmark_prepack_rhs = GetBoolEnvVarOrFalse("PREPACK_RHS");
        test_set.benchmark_prepack_rhs = true;
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

        // const bool benchmark_cubic = GetBoolEnvVarOrFalse("RUY_BENCHMARK_CUBIC");
        const bool benchmark_cubic = true;

        std::vector<ConvP> vgg16;
        std::vector<BenchmarkShape> shapes;



        // Often 8 is used for this multiplier, but to check teeny sizes one can
        // use 1.
        //static constexpr int cubic_size_multiplier = 8;



        if (benchmark_cubic) {
#ifdef _WIN32
            _putenv_s("QUICK_BENCHMARK", "1");
#else
            setenv("QUICK_BENCHMARK", "1", 0);
#endif

            //td::map<>

            vgg16.push_back(ConvP(28, 28, 5, 5, 2, 64, 192));

            vgg16.push_back(ConvP(14, 14, 3, 3, 1, 192, 384));

            vgg16.push_back(ConvP(14, 14, 3, 3, 1, 384, 256));

            vgg16.push_back(ConvP(224, 224, 7, 7, 2, 64, 64));

            vgg16.push_back(ConvP(112, 112,  3,  3, 1, 64, 128));


            vgg16.push_back(ConvP(112, 112,  3,  3, 1, 128, 128));

            vgg16.push_back(ConvP(56, 56,  3,  3, 1, 64, 64));

            vgg16.push_back(ConvP(32,  32,  3,  3, 1, 128, 256));

            vgg16.push_back(ConvP(28,  28,  3,  3, 1, 128, 128));

            vgg16.push_back(ConvP(16,  16,  3,  3, 1, 256, 512));

            vgg16.push_back(ConvP(14,  14,  3,  3, 1, 256, 256));

            vgg16.push_back(ConvP(8,  8,  3,  3, 1, 512, 1024));

            vgg16.push_back(ConvP(8,  8,  3,  3, 1, 512, 512));


            for (auto & layer : vgg16) {

                const int padding = layer.kernel_width / 2;
                const int output_height = compute_output_dimension(padding + layer.input_width + padding, layer.kernel_width, layer.stride);
                const int output_width = compute_output_dimension(padding + layer.input_height + padding, layer.kernel_height, layer.stride);
                const int buffer_depth = layer.kernel_width * layer.kernel_height * layer.input_channels;
                BenchmarkShape shape;
                shape.rows = output_height * output_width;
                shape.cols = layer.output_channels;
                shape.depth = buffer_depth;
                shape.symm_lhs = symm_lhs;
                shape.symm_rhs = symm_rhs;
                shapes.push_back(shape);
            }
        }

        for (int i = 0; i < shapes.size(); i++) {
            const auto& shape = shapes[i];

            const auto& layer = vgg16[i];

            const int padding = layer.kernel_width / 2;


            ConvParams conv_params;

            conv_params.stride_height = layer.stride;
            conv_params.stride_width = layer.stride;
            conv_params.padding_height = padding;
            conv_params.padding_width = padding;


            RuntimeShape input_shape;
            input_shape.depth = layer.input_channels;
            input_shape.height = layer.input_height;
            input_shape.width = layer.input_width;


            const int output_height = compute_output_dimension(padding + layer.input_width + padding, layer.kernel_width, layer.stride);
            const int output_width = compute_output_dimension(padding + layer.input_height + padding, layer.kernel_height, layer.stride);

            RuntimeShape buffer_shape;
            buffer_shape.depth = layer.kernel_height * layer.kernel_width * input_shape.depth;
            buffer_shape.width = output_width;
            buffer_shape.height = output_height;


            // benchmark imcol here

            double im2col_latency = benchmark_im2col(conv_params, layer.kernel_height, layer.kernel_width, input_shape, buffer_shape);

            const auto& results = BenchmarkRCC<TestSetType>(shape);
            if (i == 0) {
                if(benchmark_cubic){
                    printf("Rows, Depth, Cols, im2col, MM, sum");
                    printf("\n");
                } else {
                    printf("path,shape,Gop/s\n");
                }
            }

            if (benchmark_cubic) {

                printf("%d, %d, %d", shape.rows, shape.depth, shape.cols);
                //size_t items = 1 * 2 * shape.rows * shape.depth * shape.cols;
                for (const auto& result : results) {
                    //printf(",%4g,%.4g", 1.0e3 * result->latency, 2.0e-9 * shape.rows * shape.cols * shape.depth /
                    //                    result->latency);
                    printf(", %f, %f ,%f", 1.0e3 * im2col_latency,
                           1.0e3 * result->latency, 1.0e3*(im2col_latency + result->latency));

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

