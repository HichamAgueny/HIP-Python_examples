This directory contains 3 basic examples of how to use HIP Python interface:

**(i)**   To access the device properties (see `example1_DeviceProp/deviceProp.py`)

**(ii)**  To manage memory i.e. allocate/deallocate device memory and transfer data between host and device (see `example2_MemoryManagemnt/memoryManagemnt.py`)

**(iii)** To compile and launch device kernels (see `example3_DeviceKernels/deviceKernels.py`)
 
# To launch an interactive session:
```
salloc -A project_465001310 -t 00:30:00 -p dev-g -N 1 --gpus 1
```

# To install hip-python & numpy packages in a virtual environment 
## Load modules
```
module load LUMI/24.03 partition/G
module load cpeCray
module load cray-python/3.11.7
```
```
python -m venv MyVirtEnv_hip_pyt    
source MyVirtEnv_hip_pyt/bin/activate
python -m pip install --upgrade pip
python -m pip install -i https://test.pypi.org/simple/ hip-python
python -m pip install numpy
```

# To launch a python script in an interactive session
```
salloc -A project_465001310 -t 00:30:00 -p dev-g -N 1 --gpus 1

source MyVirtEnv_hip_pyt/bin/activate

srun python example1_DeviceProp/deviceProp.py
srun python example2_MemoryManagemnt/memoryManagemnt.py
srun python example3_DeviceKernels/deviceKernels.py
```

More examples can be found [here](https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html)
