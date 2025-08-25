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
        A[idx]=A[idx]*2.0f;
    }
}

__global__ void kernel2(float *A,int numOfElements){
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<numOfElements){
        A[idx]=A[idx]+1.0f;
    }
}


//now this is callback function which tells the cpu the gpu tasks are done 
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}

int main(){
    float *h_A;
    float *d_A;
    int numOfElements=1000000;
    size_t size=numOfElements*sizeof(float);
    cudaStream_t stream1,stream2;
    cudaEvent_t event;


    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_A, size)); //this is pinned memeory for faster transfers

    // Initialize data
    for (int i = 0; i < numOfElements; ++i) {
        h_A[i] = static_cast<float>(i); //converts int to float explicitly
    }

    //now we are setting priorities 
    int leastpriority , greatestpriority;
    cudaDeviceGetStreamPriorityRange(&leastpriority, &greatestpriority);
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1,cudaStreamNonBlocking,leastpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2,cudaStreamNonBlocking,greatestpriority));

    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    kernel1<<<(numOfElements+255)/256,256,0,stream1>>>(d_A,numOfElements);


    //Now we are recording the event in stream1
    CHECK_CUDA_ERROR(cudaEventRecord(event,stream1));

    //stream2 should wait untill the stream1 tasks are over
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2,event,0));

    kernel2<<<(numOfElements+255)/256,256,0>>>(d_A,numOfElements);


    //this will just notify the cpu like all the gpu tasks are done
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2,myStreamCallback,NULL,0));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_A,d_A,size,cudaMemcpyDeviceToHost,stream2));


    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < numOfElements; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_A[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Clean up
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));

    return 0;

}