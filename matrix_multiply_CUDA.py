# run using colab
# in runtime, change the runtime type to t4 gpu
# pip install pycuda

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


# Define the input matrices with integer values
matrix_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
matrix_b = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32)
matrix_c = np.zeros_like(matrix_a, dtype=np.int32)\

# Matrix dimensions
M, N, K = matrix_a.shape[0], matrix_a.shape[1],
matrix_b.shape[1]

# CUDA kernel to perform matrix multiplication
cuda_code = f"""
__global__ void matrixMul(int *A, int *B, int *C, int M, int N,
int K) {{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int sum = 0;
if (row < M && col < K) {{
for (int i = 0; i < N; i++) {{
sum += A[row * N + i] * B[i * K + col];
}}
C[row * K + col] = sum;
}}
}}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel function
matrixMul = mod.get_function("matrixMul")

# Host and device matrices
device_matrix_a = cuda.mem_alloc(matrix_a.nbytes)
device_matrix_b = cuda.mem_alloc(matrix_b.nbytes)
device_matrix_c = cuda.mem_alloc(matrix_c.nbytes)

# Copy host data to device
cuda.memcpy_htod(device_matrix_a, matrix_a)
cuda.memcpy_htod(device_matrix_b, matrix_b)

# Define grid and block sizes
block_size = (16, 16, 1) # Specify the third dimension as 1 for 2D grid
grid_size = ((K + block_size[0] - 1) // block_size[0], (M +
block_size[1] - 1) // block_size[1])

# Execute the kernel for matrix multiplication
matrixMul(device_matrix_a, device_matrix_b, device_matrix_c,
np.int32(M), np.int32(N), np.int32(K),
block=block_size, grid=grid_size)

# Copy the result back to the host
cuda.memcpy_dtoh(matrix_c, device_matrix_c)

# Print the input matrices with integer values
print("Matrix A:")
print(matrix_a)
print("Matrix B:")
print(matrix_b)
# nice
# Print the resulting matrix with integer values
print("Resulting Matrix C:")
print(matrix_c)