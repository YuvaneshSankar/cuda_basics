#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ kernel(void){
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int block_offset = block_id * blockIdx.x * blockIdx.y * blockIdx.z;
    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int id = block_offset + thread_offset ; 
}



int main(){
    const int b_x=2 , b_y=3 , b_z=4;
    const int t_x=4 , t_y=4 , t_z=4;

    int blocks_per_gird = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    dim3 blocksPerGrid(b_x,b_y,b_z); //also the grid dimension
    dim3 threadsPerBlock(t_x,t_y,t_z); //also the block dimension

    kernel<<<blocksPerGrid,threadsPerBlock>>>();

    cudaDeviceSynchronize(); // comands (kernel launches, memory copies, etc.) that were issued to the GPU are completed before the function returns control back to the CPU
}




