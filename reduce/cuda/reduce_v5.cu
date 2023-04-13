#include <cuda.h>
#include <time.h>
#include <iostream>
#include <bits/stdc++.h>

#include "cuda_runtime.h"

#define THREAD_PER_BLOCK 256

template<int block_size>
__device__ void warpReduce(volatile float* cache, int t_id) {
    if (block_size >= 64) {
        cache[t_id] += cache[t_id + 32];
    }
    if (block_size >= 32) {
        cache[t_id] += cache[t_id + 16];
    }
    if (block_size >= 16) {
        cache[t_id] += cache[t_id + 8];
    }
    if (block_size >= 8) {
        cache[t_id] += cache[t_id + 4];
    }
    if (block_size >= 4) {
        cache[t_id] += cache[t_id + 2];
    }
    if (block_size >= 2) {
        cache[t_id] += cache[t_id + 1];
    }
}

// this kernel reduce bank conflict
template <unsigned int blockSize>
__global__ void reduce(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
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

    reduce<THREAD_PER_BLOCK><<<Grid, Block>>>(d_a, d_out);

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