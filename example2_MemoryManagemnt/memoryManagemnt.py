# Memory management
import numpy as np
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

# Copy data from device to host
host_data_back = np.empty_like(host_data)
hip_check(hip.hipMemcpy(host_data_back, device_data, num_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

# Check the result
if np.allclose(host_data_back,host_data):
    print("--Correct - the result is the same")
else:
    print("--Failed")

# Free device memory
hip_check(hip.hipFree(device_data))

print()
print("--Completed :)")
