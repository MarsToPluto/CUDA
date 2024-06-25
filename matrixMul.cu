#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    // Allocate shared memory for tiles of matrices A and B
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    // Calculate row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the value of C to 0
    float value = 0;

    // Loop over the tiles of the matrices
    for (int m = 0; m < N / 32; ++m) {
        // Load elements of A and B into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (m * 32 + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(m * 32 + threadIdx.y) * N + col];
        __syncthreads();

        // Multiply the tiles together
        for (int k = 0; k < 32; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result back to the global memory
    C[row * N + col] = value;
}

void matrixMultiplication(float* h_A, float* h_B, float* h_C, int N) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block size and grid size
    dim3 blockSize(32, 32);
    dim3 gridSize(N / 32, N / 32);

    // Launch the CUDA kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1024; // Matrix size N x N
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform matrix multiplication
    matrixMultiplication(h_A, h_B, h_C, N);

    // Print a small part of the result matrix for verification
    for (int i = 0; i < min(10, N); ++i) {
        for (int j = 0; j < min(10, N); ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
