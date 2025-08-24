#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define itr 1000

__device__ int tryAtomicAdd(int* address, int incremental_val){
    __shared__ int lock;

    int old;

    if(threadIdx.x==0) lock=0; //only the zeroth thread in everyblock should initilaize the lock value
    __syncthreads(); //wait till all the threads in a block reach here (waiting lobby)


    //spin lock
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

__global__ void kernel2(int* address, int incremental_val){
    for(int i=0;i<itr;i++){
        atomicAdd(address,incremental_val);
    }
}


int main(){
    int *d_A1;
    int *d_A2;
    int h_A1=0;
    int h_A2=0;

    int incremental_val=1;

    //allocate memory in gpu
    cudaMalloc(&d_A1,sizeof(int));

    //copy data from host to device
    cudaMemcpy(d_A1,&h_A1,sizeof(int),cudaMemcpyHostToDevice);

    //kernel function
    kernel<<<1,256>>>(d_A1,incremental_val);
    //we did this only for 1 block

    //copy data from device to host
    cudaMemcpy(&h_A1,d_A1,sizeof(int),cudaMemcpyDeviceToHost);

    printf("The total incremented value of value h_A1 is %d\n",h_A1);



    //lets check for the acutally atmociadd() done by gpu hardware itself
    cudaMalloc(&d_A2,sizeof(int));
    cudaMemcpy(d_A2,&h_A2,sizeof(int),cudaMemcpyHostToDevice);
    kernel2<<<1,256>>>(d_A2,incremental_val);
    cudaMemcpy(&h_A2,d_A2,sizeof(int),cudaMemcpyDeviceToHost);
    printf("The total incremented value of value h_A2 is %d\n",h_A2);

    cudaFree(d_A1);
    cudaFree(d_A2);

    return 0;
}



//what we got when we write custom atomicadd function ->254252 or something closer
//but what actually we should have got when we use atomicadd() actual function 256×1000×1=256,000
//CUDA launches all 256 threads together.They are grouped in 8 warps, and the GPU runs them mostly in parallel.
//Each thread runs 1000-iteration loop independently.
//this custom lock fails because it is not synchronized perfectly across all threads; it gets reinitialized mid-use, causing a few increments to be lost.
//atomicAdd() works because the GPU hardware does the locking internally perfectly, without needing the spin-lock.




