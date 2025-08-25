#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}


__global__ void  kernel1(float *A,int numOfElements){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<numOfElements){
        A[idx]=A[idx]+1.0f;
    }
}

__global__ void kernel2(float *A,int numOfElements){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<numOfElements){
        A[idx]=A[idx]+1.0f;
    }
}


void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}


int main(){
    int numOfElements=1000000;
    float *h_A, *h_B;
    float *d_A,*d_B;
    size_t size =numOfElements * sizeof(float);
    cudaStream_t stream1,stream2;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));

    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_A, size));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_B, size));

    int leastpriority,greatestpriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastpriority, &greatestpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1,cudaStreamNonBlocking,leastpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2,cudaStreamNonBlocking,greatestpriority));

    // Initialize data
    for (int i = 0; i < numOfElements; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 1.0; 
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

    kernel1<<<(numOfElements+255)/256,256,0,stream1>>>(d_A,numOfElements);
    kernel2<<<(numOfElements+255)/256,256,0,stream2>>>(d_B,numOfElements);

    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream1,myStreamCallback,NULL,0));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_A,d_A,size,cudaMemcpyDeviceToHost,stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_B,d_B,size,cudaMemcpyDeviceToHost,stream2));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));


    // Verify result for d_A and h_A (stream1)
    for (int i = 0; i < numOfElements; ++i) {
        float expected = 2.0f;   // Initial was 1.0f, kernel adds 1.0f once
        if (fabs(h_A[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d in h_A! Expected %f but got %f\n", i, expected, h_A[i]);
            exit(EXIT_FAILURE);
        }
    }

    // Verify result for d_B and h_B (stream2)
    for (int i = 0; i < numOfElements; ++i) {
        float expected = 2.0f;   // Initial was 1.0f, kernel adds 1.0f once
        if (fabs(h_B[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d in h_B! Expected %f but got %f\n", i, expected, h_B[i]);
            exit(EXIT_FAILURE);
        }
    }

printf("Test PASSED\n");


    // Clean up
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

    return 0;
}