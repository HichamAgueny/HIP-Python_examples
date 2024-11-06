# Import some modules
import numpy as np
from hip import hip, hiprtc
from hip import hipblas


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
sum_np = np.sum(host_data)

# Allocate device memory
num_bytes = N * np.float32().itemsize
device_data = hip_check(hip.hipMalloc(num_bytes))

# Copy data from host to device
hip_check(hip.hipMemcpy(device_data, host_data, num_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

# Initiate hipblas
handle = hip_check(hipblas.hipblasCreate())

# 
num_bytes_r = np.dtype(np.float32).itemsize
result_d = hip_check(hip.hipMalloc(num_bytes_r))
# call hipblasSum
hip_check(hipblas.hipblasSasum(handle, N, device_data, 1, result_d))

# Destroy handle
hip_check(hipblas.hipblasDestroy(handle))

# copy the result (stored in result_d) back to host (store in result_h)
result_h = np.zeros(1, dtype=np.float32)
hip_check(hip.hipMemcpy(result_h, result_d, num_bytes_r, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Check the result
if np.allclose(result_h[0],sum_np):
    print()
    print("--Correct - the result is the same")
    print("--CPU: Sum using numpy:", sum_np)
    print("--GPU: Sum using hipblas:", result_h[0])
else:
    print("--Failed")

# Free device memory
hip_check(hip.hipFree(device_data))

print()
print("--Completed :)")
