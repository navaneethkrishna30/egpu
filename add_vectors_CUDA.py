# run using colab
# in runtime, change the runtime type to t4 gpu
# pip install pycuda

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


# Define the vector size (number of elements)
N = 100000

# CUDA kernel to add two vectors
cuda_code = """
__global__ void vectorAdd(float *a, float *b, float *c, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
c[idx] = a[idx] + b[idx];
}
}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel function
vectorAdd = mod.get_function("vectorAdd")

# Host vectors
h_a = np.arange(N, dtype=np.int32)
h_b = np.arange(N, dtype=np.int32) * 2
h_c = np.zeros_like(h_a)

# Device vectors
d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_c = cuda.mem_alloc(h_c.nbytes)

# Copy host data to device
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

# Define grid and block sizes
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Execute the kernel
vectorAdd(d_a, d_b, d_c, np.int32(N), block=(block_size, 1, 1),
grid=(grid_size, 1, 1))

# Copy the result back to the host
cuda.memcpy_dtoh(h_c, d_c)

# Print the initial and final vectors
print("Initial Vector (h_a):")
print(h_a)
print("\nInitial Vector (h_b):")
print(h_b)
print("\nFinal Vector (h_c):")
print(h_c)

# Clean up
d_a.free()
d_b.free()
d_c.free()
print("Vector addition completed.")