#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

using namespace std;

#define BLOCK_SIZE 16


__global__ void kernel(float *A , float *B , float *C , int N){
    int row= blockIdx.y * blockDim.y + threadIdx.y;
    int col= blockIdx.y * blockDim.y + threadIdx.y;

    if( row<N && col<N){
        float sum=0.0f;
        for(int i=0;i<N;i++){
            sum+=A[row*N + i] * B[i* N + col];
        }
        C[row * N + col]=sum;
    }

}

void init_matrix(float *mat , int row , int col){
    for(int i=0;i<row * col ;i++){
        mat[i]=(float)rand()/RAND_MAX;
    }
}

void matrixmul(float *h_A , float *h_B ,float *h_C ,int N){

    nvtxRangePush("Matrix Multiplication");

    float *d_A,*d_B,*d_C;

    int size=N*N*sizeof(float);

    nvtxRangePush("Memory allocation");
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    nvtxRangePop();

    nvtxRangePush("Memory copy from H2D");
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    nvtxRangePop();


    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    nvtxRangePush("Kenerl Execution");
    kernel<<<numBlocks,threadsPerBlock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memoery copy from D2H");
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();
}
int main(){

    const int N=1024;
    float *h_A= new float[N*N];
    float *h_B= new float[N*N];
    float *h_C= new float[N*N];

    init_matrix(h_A,N,N);
    init_matrix(h_B,N,N);

    matrixmul(h_A,h_B,h_C,N);

    delete [] h_A;
    delete [] h_B;
    delete [] h_C;

    return 0;
}