import pyopencl as cl
import numpy as np


# Number of work items (threads)
num_work_items = 1000

# Number of samples for each work item
num_samples = 100000

# Initialize OpenCL context and command queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Create an OpenCL program from the kernel code
kernel_code = """
__kernel void matrix_multiply(__global int* A, __global int* B,
                              __global int* C, const int M, const int N, const int K) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int sum = 0;
    if (row < M && col < K) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
"""

program = cl.Program(ctx, kernel_code).build()

# Create a result buffer to store pi estimates
result = np.empty(num_work_items, dtype=np.float32)
buffer_result = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY,
result.nbytes)

# Execute the OpenCL kernel
program.estimate_pi(queue, (num_work_items,), None, buffer_result, np.uint32(num_samples))

# Copy the result back to the host
cl.enqueue_copy(queue, result, buffer_result).wait()

# Calculate the final estimate of pi
estimated_pi = np.mean(result)
print(f"Estimated value of π: {estimated_pi}")