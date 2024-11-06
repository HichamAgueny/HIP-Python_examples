# Compiling and Launching Device Kernels
import os
import math
import numpy as np
import ctypes
from hip import hip, hiprtc

# Error check routine
def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

# Generate random 1D-array
N = 10 #length
host_data = np.random.rand(N).astype(np.float32)
print()
print("--host data:", host_data)
print()

# Allocate device memory
num_bytes = N * np.float32().itemsize
device_data = hip_check(hip.hipMalloc(num_bytes))

# Copy data from host to device
hip_check(hip.hipMemcpy(device_data, host_data, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Read the kernel
#kernel_file_path = os.path.abspath(os.path.join(., 'kernel.hip'))
kernel_file_path = os.path.abspath('kernel.hip')
with open(kernel_file_path, 'r') as file:
    kernel_source = file.read()

# Create a program
prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_source.encode(), b"Kernel_test", 0, [], []))

# Extract the architecture name for compilation
props = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(props,0))
arch = props.gcnArchName

print(f"Compiling kernel for {arch}")
# Define the compile flags, including  macro preprocessor named CST and has the value cst
cflags = [b"--offload-arch="+arch]

# Compile the program
err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
print("--compilation:", err)
if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
    log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
    log = bytearray(log_size)
    hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
    raise RuntimeError(log.decode())

code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
code = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog, code))
module = hip_check(hip.hipModuleLoadData(code))
kernel = hip_check(hip.hipModuleGetFunction(module, b"Kernel_test"))

# Launching the device kernel
# Define grid & block dimensions
factor = 2.0 # parameter

block = hip.dim3(x=16, y=1, z=1)
grid = hip.dim3(math.ceil(N/block.x))
# Launch Kernel
hip_check(
    hip.hipModuleLaunchKernel(
        kernel,
        *grid,
        *block,
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=(
          ctypes.c_int(N),
          ctypes.c_float(factor),
          device_data,
            )
        )
    )

# Copy data from device to host
host_data_back = np.empty_like(host_data)
hip_check(hip.hipMemcpy(host_data_back, device_data, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Check the result
if np.allclose(host_data_back,factor*host_data):
    print("--Correct - the result is the same")
else:
    print("--Failed")

# Free device memory
hip_check(hip.hipFree(device_data))

print("--Completed :)")

