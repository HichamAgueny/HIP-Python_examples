from hip import hip

print()
print("Device properties using the object hipDeviceProp_t")

#https://rocm.docs.amd.com/projects/hip-python/en/latest/python_api/hip.html#hip.hip.hipDeviceProp_t
props = hip.hipDeviceProp_t()
hip.hipGetDeviceProperties(props,0)

print(f"props.name = {props.name}")
print(f"props.gcnArchName = {props.gcnArchName}")
print(f"props.pciDeviceID = {props.pciDeviceID}")
print(f"props.totalGlobalMem = {props.totalGlobalMem}")
print(f"props.sharedMemPerBlock = {props.sharedMemPerBlock}")
print(f"props.l2CacheSize = {props.l2CacheSize}")

print()
print("Device attributes using hipDeviceAttribute_t")
#https://rocm.docs.amd.com/projects/hip-python/en/latest/python_api/hip.html#hip.hip.hipDeviceAttribute_t
device_id = 0
for attrib in (
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxBlockDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimX,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimY,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimZ,
   hip.hipDeviceAttribute_t.hipDeviceAttributeWarpSize,
   hip.hipDeviceAttribute_t.hipDeviceAttributeMaxThreadsPerBlock,
):
    result_attr = hip.hipDeviceGetAttribute(attrib,device_id)
    print(f"{attrib.name}: {result_attr[1]}")

print()
print("--Completed :)")
