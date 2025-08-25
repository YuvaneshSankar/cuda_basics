#include <bits/stdc++.h>
#include <cuda_runtime.h>

struct Mutex{
    int *lock; 
};



//we have to initalize the struct mutex
__host__ void initmutex(Mutex *m){
    cudaMalloc((void**)&m->lock,sizeof(int)); //we are allocating memory to the lock value in gpu . so we get a mutex pointer so 
    //we allocate the memory for the mutex pointer pointing pointer which is the pointer of the lock
    //so the first pointer points the sturct Mutex in the memory and the second pointer points the lock variable inside that sturct mutex
    int initial=0;
    cudaMemcpy(m->lock,&initial,sizeof(int),cudaMemcpyHostToDevice);//here we have initialized a value for lock as 0 intitalize so 
    //we have to send that to the device memory which is the lock in m .
    
}

//aquire mutex or aquire the memory 
__device__ void lock(Mutex *m){
    while(atomicCAS(m->lock,0,1)!=0);
}

//release the mutex value after using like unlock it
__device__ void unlock(Mutex *m){
    atomicExch(m->lock,0);
}

//kernel function
__global__ void kernel(int *counter , Mutex *m){
    lock(m);
    int old=*counter;
    *counter=old+1;
    unlock(m);
}

int main(){
    Mutex m;
    
    int *d_counter;
    cudaMalloc(&d_counter,sizeof(int));

    int initial=0;
    cudaMemcpy(d_counter,&initial,sizeof(int),cudaMemcpyHostToDevice);

    kernel<<<1,1000>>>(d_counter,&m);

    int result;

    cudaMemcpy(&result,d_counter,sizeof(int),cudaMemcpyDeviceToHost);

    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;

}


