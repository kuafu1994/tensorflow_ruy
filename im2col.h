//
// Created by PENGFEI ZHANG on 2020/3/30.
//

#ifndef RUY_IM2COL_H
#define RUY_IM2COL_H

#include <vector>
#include <random>
#include <stdint.h>

#include <time.h>

namespace ruy {
    typedef struct RuntimeShape {
        int depth;
        int width;
        int height;
    } RuntimeShape;

    typedef struct ConvParams {
        int stride_height;
        int stride_width;
        int padding_width;
        int padding_height;
    } ConvParams;


    int Offset(const RuntimeShape &shape, const int ih_start, const int iw_start, const int id_start) {
        const int width = shape.width;
        // const int height = shape.height;
        const int depth = shape.depth;

        return (ih_start * width + iw_start) * depth + id_start;
    }

    template<typename T>
    inline void ExtractPatchInputBufferColumn(
            const RuntimeShape &input_shape, int w, int h, int kheight, int kwidth,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int in_width, int in_height, int in_depth, int single_buffer_length, int buffer_id,
            const T *in_data, T *conv_buffer_data, int8_t zero_byte) {

        const int kwidth_times_indepth = kwidth * in_depth;
        const int inwidth_times_indepth = in_width * in_depth;
        const int ih_ungated_start = h * stride_height - pad_height;
        const int ih_ungated_end = (ih_ungated_start + kheight);
        const int ih_end = std::min(ih_ungated_end, in_height);

        // iw_ungated_start might be less than zero.
        const int iw_ungated_start = w * stride_width - pad_width;
        // iw_ungated_end might be larger than in_width.
        const int iw_ungated_end = (iw_ungated_start + kwidth);
        const int iw_end = std::min(iw_ungated_end, in_width);
        // If the patch is off the edge of the input image, skip writing those rows
        // and columns from the patch into the output array.
        const int h_offset = std::max(0, -ih_ungated_start);
        const int w_offset = std::max(0, -iw_ungated_start); // w_offset means the offset in the kwidth.
        const int ih_start = std::max(0, ih_ungated_start);
        const int iw_start = std::max(0, iw_ungated_start); // iw_start means the position in the input width.

        const int single_row_num =
                std::min(kwidth - w_offset, in_width - iw_start) * in_depth;

        const int output_row_offset = (buffer_id * single_buffer_length);
        int out_offset =
                output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
        int in_offset = Offset(input_shape, ih_start, iw_start, 0);

        // Express all of the calculations as padding around the input patch.
        const int top_padding = h_offset;
        const int bottom_padding = (ih_ungated_end - ih_end);
        const int left_padding = w_offset;
        const int right_padding = (iw_ungated_end - iw_end);
        assert(single_row_num ==
               ((kwidth - (left_padding + right_padding)) * in_depth));

        // Write out zeroes to the elements representing the top rows of the input
        // patch that are off the edge of the input image.
        if (top_padding > 0) {
            const int top_row_elements = (top_padding * kwidth * in_depth);
            memset(conv_buffer_data + output_row_offset, zero_byte,
                   (top_row_elements * sizeof(T)));
        }

        // If the patch is on the interior of the input image horizontally, just copy
        // over the rows sequentially, otherwise add zero padding at the start or end.
        //
        if ((left_padding == 0) && (right_padding == 0)) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                memcpy(conv_buffer_data + out_offset, in_data + in_offset,
                       single_row_num * sizeof(T));
                out_offset += kwidth_times_indepth; // kwidth_times_indepth.
                in_offset += inwidth_times_indepth;
            }
        } else {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                if (left_padding > 0) {
                    const int left_start = (out_offset - (left_padding * in_depth));
                    memset(conv_buffer_data + left_start, zero_byte,
                           (left_padding * in_depth * sizeof(T)));
                }
                memcpy(conv_buffer_data + out_offset, in_data + in_offset,
                       single_row_num * sizeof(T));
                if (right_padding > 0) {
                    const int right_start = (out_offset + single_row_num);
                    memset(conv_buffer_data + right_start, zero_byte,
                           (right_padding * in_depth * sizeof(T)));
                }
                out_offset += kwidth_times_indepth;
                in_offset += inwidth_times_indepth;
            }
        }

        // If the bottom of the patch falls off the input image, pad the values
        // representing those input rows with zeroes.
        if (bottom_padding > 0) {
            const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
            const int bottom_start =
                    output_row_offset +
                    ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
            memset(conv_buffer_data + bottom_start, zero_byte,
                   (bottom_row_elements * sizeof(T)));
        }

    }

    template<typename T>
    void Im2col(const ConvParams &params, int kheight, int kwidth, int8_t zero_byte,
                const RuntimeShape &input_shape, const T *input_data,
                const RuntimeShape &output_shape, T *output_data) {

        const int stride_width = params.stride_width;
        const int stride_height = params.stride_height;
        const int pad_width = params.padding_width;
        const int pad_height = params.padding_height;

        const int input_depth = input_shape.depth;
        const int input_width = input_shape.width;
        const int input_height = input_shape.height;

        const int output_depth = output_shape.depth;
        const int output_width = output_shape.width;
        const int output_height = output_shape.height;

        int buffer_id = 0;
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                ExtractPatchInputBufferColumn(
                        input_shape, w, h, kheight, kwidth, stride_width, stride_height,
                        pad_width, pad_height, input_width, input_height, input_depth,
                        output_depth, buffer_id, input_data, output_data, zero_byte);
                buffer_id++;
            }
        }

    }

    double benchmark_im2col(const ConvParams& conv_params, const int kheight, const int kwidth,
            const RuntimeShape& input_shape, const RuntimeShape& buffer_shape) {

        std::random_device rd;
        auto rng = std::mt19937(rd());
        auto activation_rng = std::bind(std::uniform_int_distribution<int8_t>(), rng);

        std::vector<int8_t> input(input_shape.height * input_shape.width * input_shape.depth);
        std::vector<int8_t> buffer(buffer_shape.height * buffer_shape.width * buffer_shape.depth);

        std::generate(input.begin(), input.end(), std::ref(activation_rng));

        const int repeats = 4;
        double elapsed_time = 0.0;

        for(int i = 0; i < repeats; i++) {

            TimePoint time_start = Now();
            Im2col(conv_params, kheight, kwidth, 1, input_shape, input.data(), buffer_shape, buffer.data());

            TimePoint time_end = Now();

            elapsed_time += ToFloatSeconds(time_end - time_start);
        }

        return elapsed_time / repeats;

    }
} // namespace ruy.

#endif //RUY_IM2COL_H
