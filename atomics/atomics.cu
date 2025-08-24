#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define itr 1000

__device__ int tryAtomicAdd(int* address, int incremental_val){
    __shared__ int lock;

    int old;

    if(threadIdx.x==0) lock=0; //only the zeroth thread in everyblock should initilaize the lock value
    __syncthreads(); //wait till all the threads in a block reach here (waiting lobby)

    while(atomicCAS(&lock,0,1)!=0);//see it is not necessary to start with zeroth thread so we have wait till the lock values is zero so this just keeps
    //going untill we reach the zeroth thread of some block then we get the value of lock as 0 so here boths are equal the value at the address and the value
    //at compare so we reaplace the value of lock at the memory address to be 1 and we retunr the old value which is 1.

    old=*address; //get the old value from the address not this is our main atomic add function
    *address=old+incremental_val; //replace the value;

    __threadfence(); //notify to all the threads that there is a change in that memory location

    atomicExch(&lock,0); //unlock it as we are done with our task so we have free it now
    
    return old;

}


__global__ void kernel(int* address, int incremental_val){
    for(int i=0;i<itr;i++){
        tryAtomicAdd(address,incremental_val);
    }
}

int main(){
    int *d_A;
    int h_A=0;
    int incremental_val=1;

    //allocate memory in gpu
    cudaMalloc(&d_A,sizeof(int));

    //copy data from host to device
    cudaMemcpy(d_A,&h_A,sizeof(int),cudaMemcpyHostToDevice);

    //kernel function
    kernel<<<1,256>>>(d_A,incremental_val);
    //we did this only for 1 block

    //copy data from device to host
    cudaMemcpy(&h_A,d_A,sizeof(int),cudaMemcpyDeviceToHost);

    printf("The total incremented value of value h_A is %d\n",h_A);

    return 0;
}