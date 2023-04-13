#include <cuda.h>
#include <time.h>
#include <iostream>
#include <bits/stdc++.h>

#include "cuda_runtime.h"

#define THREAD_PER_BLOCK 256

// this kernel reduce bank conflict
__global__ void reduce(float* in, float* out) {

    __shared__ float s_data[THREAD_PER_BLOCK];

    // load data from global to shared local memory
    int t_id = threadIdx.x;
    int global_idx = t_id + blockIdx.x * blockDim.x * 2;
    s_data[t_id] = in[global_idx] + in[global_idx + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t_id < s) {
            s_data[t_id] += s_data[t_id + s];
        }
        __syncthreads();
    }

    // update result for this block to global memory
    if (t_id == 0) {
        out[blockIdx.x] = s_data[0];
    }
}

bool check(float* out, float* res, int n) {
    for(int i = 0; i < n; i++){
        if (out[i] != res[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;

    float* a = (float*) malloc(sizeof(float) * N);
    float* d_a ;

    cudaMalloc((void**)&d_a, N * sizeof(float));

    int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;
    int block_dim = N / NUM_PER_BLOCK;
    float* out = (float*) malloc(block_dim * sizeof(float));
    float* d_out;

    cudaMalloc((void**)&d_out, block_dim * sizeof(float));

    float* res = (float*) malloc(block_dim * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
    }

    for (int i = 0; i < block_dim; i++) {
        float cur = 0;
        for (int j = 0; j < NUM_PER_BLOCK; j++) {
            cur += a[i * NUM_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_dim, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<<<Grid, Block>>>(d_a, d_out);

    cudaMemcpy(out, d_out, block_dim * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_dim)) {
        std::cout << "The answer is right" << std::endl;
    } else {
        std::cout << "The answer is wrong" << std::endl;
        // for(int i = 0; i < block_dim; i++){
        //     printf("%lf ",out[i]);
        // }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out);
}