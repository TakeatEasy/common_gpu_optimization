#include <cuda.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h>

#include "cuda_runtime.h"

const int ARR_SIZE = 32 * 1024 * 1024;

// This is baseline for scan write with cuda

__global__ void scan_and_write_part_sum_kernel(const int32_t* input, int32_t* part, int32_t* output, size_t n, size_t part_num) {
    
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // calculate for each block start and end index
        size_t part_begin = part_i * blockDim.x;
        size_t part_end = min((part_i + 1) * blockDim.x, n);
        if (threadIdx.x == 0) {
            // only thread 0 calculate the sum of each input
            int32_t acc = 0;
            for (size_t i = part_begin; i < part_end; i++) {
                acc += input[i];
                output[i] = acc;
            }
            part[part_i] = acc;
        }
    }
}

__global__ void scan_part_sum_kernel(int32_t* part, size_t part_num) {
    // calcute the total sum of input
    int32_t acc = 0;
    for (size_t i = 0; i < part_num; i++) {
        acc += part[i];
        part[i] = acc;
    }
}

__global__ void add_base_sum_kernel(int32_t* part, int32_t* output, size_t n, size_t part_num) {
    // add part sum for each index
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        if (part_i == 0) {
            continue;
        }
        int32_t index = part_i * blockDim.x + threadIdx.x;
        if (index < n) {
            output[index] += part[part_i - 1];
        }
    }
}


void scan_the_fan(const int32_t* input, int32_t* output, size_t n) {
    // for each block
    size_t part_size = 1024;
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);

    int32_t* part = new int[part_num];
    memset(part, 0, part_num);

    int32_t* d_input;
    int32_t* d_output;
    int32_t* d_part;

    
    cudaMalloc((void**)&d_input, ARR_SIZE * sizeof(int32_t));
    cudaMalloc((void**)&d_output, ARR_SIZE * sizeof(int32_t));
    cudaMalloc((void**)&d_part, part_num * sizeof(int32_t));
    
    cudaMemcpy(d_input, input, ARR_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, ARR_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_part, part, part_num * sizeof(int32_t), cudaMemcpyHostToDevice);

    scan_and_write_part_sum_kernel<<<block_num, part_size>>>(d_input, d_part, d_output, ARR_SIZE, part_num);

    scan_part_sum_kernel<<<1, 1>>>(d_part, part_num);

    add_base_sum_kernel<<<block_num, part_size>>>(d_part, d_output, ARR_SIZE, part_num);

    cudaMemcpy(output, d_output, ARR_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(part, d_part, part_num * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_part);
}

bool check_result(int* input, int* output, int size) {
    int acc = 0;
    for (int i = 0; i < size; i++) {
        acc += input[i];
        if (output[i] != acc) {
            return false;
        }
    }
    return true;
}

int main() {

    int* input = new int[ARR_SIZE];
    int* output = new int[ARR_SIZE];

    memset(output, 0, ARR_SIZE);

    for (int i = 0; i < ARR_SIZE; i++) {
        input[i] = i;
    }

    scan_the_fan(input, output, ARR_SIZE);

    for (int i = 0; i < 100; i++) {
        printf("output: %d\n", output[i]);
    }

    if (check_result(input, output, ARR_SIZE)) {
        printf("Right answer\n");
    } else {
        printf("Wrong answer\n");
    }

    delete[] input;
    delete[] output;
}