# run using colab
# in runtime, change the runtime type to t4 gpu
# pip install pycuda

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


# Define the list of integer values
integer_values = [458, 237, 278, 430, 849, 25, 23, 894, 929,
954, 730, 470, 861, 595, 245, 921, 73, 520, 773, 797,
801, 614, 241, 402, 539, 988, 344, 944, 89,
876, 398, 165, 992, 515, 688, 180, 88, 505, 479, 320,
941, 639, 711, 584, 434, 605, 294, 12, 941,
190, 898, 651, 142, 257, 692, 226, 740, 650, 833,
370, 45, 310, 182, 724, 528, 130, 770, 510,
273, 594, 367, 650, 420, 593, 977, 458, 951, 195, 43,
694, 239, 790, 274, 891, 422, 105, 813, 458,
639, 726, 550, 984, 278, 0, 592, 507, 527, 469, 229,
791]

# Initialize the input data with the list of integer values
input_data = np.array(integer_values, dtype=np.int32)
output_data = np.zeros(1, dtype=np.int32)

# CUDA kernel to find the minimum value
cuda_code = """
__global__ void findMin(int *data, int *min, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
atomicMin(min, data[idx]);
}
}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel function
findMin = mod.get_function("findMin")

# Device arrays
device_input = cuda.mem_alloc(input_data.nbytes)
device_output = cuda.mem_alloc(output_data.nbytes)

# Copy host data to device
cuda.memcpy_htod(device_input, input_data)
cuda.memcpy_htod(device_output, output_data)

# Define grid and block sizes
block_size = 256
grid_size = (len(input_data) + block_size - 1) // block_size

# Initialize the minimum value to a large integer
output_data[0] = 2147483646 # Large positive integer (maximum value for int32)

# Execute the kernel to find the minimum value
findMin(device_input, device_output, np.int32(len(input_data)),
block=(block_size, 1, 1), grid=(grid_size, 1, 1))

# Copy the result back to the host
cuda.memcpy_dtoh(output_data, device_output)

# Print the input and minimum value
print("Input Data:")
print(input_data)
print("Minimum Value:", output_data[0]) # nice