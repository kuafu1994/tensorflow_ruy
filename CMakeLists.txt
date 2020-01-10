
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ruy)

include_directories(./gemmlowp/)
include_directories(/home/pfzhang/github/gemmlowp)

add_subdirectory(googletest)

set(SOURCE allocator.cc
        block_map.cc
        blocking_counter.cc
        context.cc
        detect_arm.cc
        detect_x86.cc
        have_built_path_for_avx2.cc
        have_built_path_for_avx512.cc
        kernel_arm32.cc
        kernel_arm64.cc
        kernel_avx2.cc
        kernel_avx512.cc
        pack_arm.cc
        pack_avx2.cc
        pack_avx512.cc
        pmu.cc
        prepacked_cache.cc
        thread_pool.cc
        trace.cc
        trmul.cc
        tune.cc
        wait.cc)

add_executable(benchmark_ruy benchmark.cc ${SOURCE})
target_link_libraries(benchmark_ruy pthread gtest gtest_main)
