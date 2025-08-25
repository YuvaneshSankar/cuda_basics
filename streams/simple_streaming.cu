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


__global__ void kernel(int *A , int *B , int *C, int numOfElements){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<numOfElements){
        C[idx]=A[idx]+B[idx];
    }   
}

int main(){
    int *h_A,*h_B,*h_C;
    int *d_A,*d_B,*d_C;

    int numOfElements=50000;

    size_t size=numOfElements*sizeof(int);
    cudaStream_t stream1,stream2;

    h_A=(int*)malloc(size);
    h_B=(int*)malloc(size);
    h_C=(int*)malloc(size);

    for(int i=0;i<numOfElements;i++){
        h_A[i]=rand()/RAND_MAX;
        h_B[i]=rand()/RAND_MAX;
    }

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));



    //now we are do both copies in seperate streams 
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A,h_A,size,cudaMemcpyHostToDevice,stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B,h_B,size,cudaMemcpyHostToDevice,stream2));


    //lets for now call the kernel function in stream1
    int threadsPerBlock = 256;
    int blocksPerGrid = (numOfElements + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_A,d_B,d_C,numOfElements);


    //lets copy back the result back to cpu using stream 1 only
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C,d_C,size,cudaMemcpyDeviceToHost,stream1));


    // Synchronize streams makes sure all tasks are done in both streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < numOfElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}