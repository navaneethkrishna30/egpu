# run using colab
# in runtime, change the runtime type to t4 gpu
# pip install pycuda

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


# Define the CUDA kernel code for squaring an array
cuda_code = """
__global__ void square_array(int *in, int *out, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
int val = in[idx];
out[idx] = val * val;
}
}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel function
square_array = mod.get_function("square_array")

# Input array
input_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int32)

# Define the array size
N = input_array.size

# Host and device arrays
host_array = input_array.copy()
device_array_in = cuda.mem_alloc(host_array.nbytes)
device_array_out = cuda.mem_alloc(host_array.nbytes)

# Copy host data to device
cuda.memcpy_htod(device_array_in, host_array)

# Define grid and block sizes
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Execute the kernel
square_array(device_array_in, device_array_out, np.int32(N),
block=(block_size, 1, 1), grid=(grid_size, 1, 1))

# Copy the result back to the host
cuda.memcpy_dtoh(host_array, device_array_out)

# Print the original and squared arrays
print("Original Array:", input_array)
print("Squared Array:", host_array)