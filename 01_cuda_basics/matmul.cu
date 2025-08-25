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
    h_A=(float*)malloc(size_A);
    h_B=(float*)malloc(size_B);
    h_C=(float*)malloc(size_C);

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
    dim3 gridDim((P + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //warm up runs
    printf("Performing warm-up runs...\n");
    for(int i=0;i<3;i++){
        matmul_cpu(h_A,h_B,h_C,M,N,P);
        matmul_gpu<<<gridDim,blockDim>>>(d_A,d_B,d_C,M,N,P);
        cudaDeviceSynchronize();
    }

    //cpu time 
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time=0.0;
    for(int i=0;i<20;i++){
        double start_time=get_time();
        matmul_cpu(h_A,h_B,h_C,M,N,P);
        double end_time=get_time();
        cpu_total_time+=end_time-start_time;
    }
    double cpu_avg_time=cpu_total_time/20.0;


    //gpu time 
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time=0.0;
    for(int i=0;i<20;i++){
        double start_time=get_time();
        matmul_gpu<<<gridDim,blockDim>>>(d_A,d_B,d_C,M,N,P);
        cudaDeviceSynchronize();
        double end_time=get_time();
        gpu_total_time+=end_time-start_time;
    }
    double gpu_avg_time=gpu_total_time/20.0;



    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;   
    
}