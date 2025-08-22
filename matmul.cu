#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>


using namespace std;
//a*b=c
#define M 1024  // row of a and c
#define N 512   // column of a and row of b
#define P 2048  //column of b and c
#define BLOCK_SIZE 32

void matmul_cpu(float *A , float *B , float *C , int m , int n , int p){
    // m*n   *  n*p
    for(int i=0;i<m;i++){
        for(int j=0;j<p;j++){
            float sum=0.0f;
            for(int k=0;k<n;k++){
                sum+=A[i*n +k] * B[k*p + j];
            }
            C[i*p + j]=sum;
        }
    }
}


__global__ void matmul_gpu(float *A , float *B , float *C , int m , int n , int p){
    int row= blockIdx.y * blockDim.y + threadIdx.y; //see rows go down so thats why we use y
    int col= blockIdx.x * blockDim.x + threadIdx.x; //see cols go right so thats why we use x

    if( row<m && col<p){
        float sum=0.0f;
        for(int i=0;i<n;i++){
            sum+=A[row*n + i] * B[i* p + col];
        }
        C[row * p + col]=sum;
    }
}


void init_matrix(float *mat , int row , int col){
    for(int i=0;i<row * col ;i++){
        mat[i]=(float)rand()/RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;

    int size_A=M*N*sizeof(float);
    int size_B=N*P*sizeof(float);
    int size_C=M*P*sizeof(float);

    //allocate host memeory
    h_A=(float)malloc(size_A);
    h_B=(float)malloc(size_B);
    h_C=(float)malloc(size_C);

    //intialize matrices
    srand(time(NULL));
    init_matrix(h_A,M,N);
    init_matrix(h_B,N,P);

    //allocate device memeory
    cudaMalloc(&d_A,size_A);
    cudaMalloc(&d_B,size_B);
    cudaMalloc(&d_C,size_C);

    //copt data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    //launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    
}