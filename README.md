This directory contains 4 basic examples of how to use HIP Python interface:

**(i)**   To access the device properties (see `example1_DeviceProp/deviceProp.py`)

**(ii)**  To manage memory i.e. allocate/deallocate device memory and transfer data between host and device (see `example2_MemoryManagemnt/memoryManagemnt.py`)

**(iii)**  To run hipblasSasum (see `example3_HipBlas/hipblasSum.py`)

**(iv)** To compile and launch device kernels (see `example4_DeviceKernels/deviceKernels.py`)

# To run the examples above:
## Download examples
```
git clone https://github.com/HichamAgueny/HIP-Python_examples.git
```
## Launch an interactive session on the supercomputer [LUMI-G](https://docs.lumi-supercomputer.eu/):
```
salloc -A project_465001310 -t 00:30:00 -p dev-g -N 1 --gpus 1
```
## Load the LUMI software stack
```
module load LUMI/24.03 partition/G
module load cpeCray
module load cray-python/3.11.7
```
## Source the virtual env. where hip-python and numpy are installed
```
source /project/project_465001310/workshop_software/HIP-Python_examples/MyVirtEnv_hip_pyt/bin/activate
```
## Run examples
```
cd HIP-Python_examples
srun python example1_DeviceProp/deviceProp.py
srun python example2_MemoryManagemnt/memoryManagemnt.py
srun python example3_HipBlas/hipblasSum.py
srun python example4_DeviceKernels/deviceKernels.py
```
## To install hip-python & numpy packages in your virtual environment 
## Load the LUMI software stack
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
More examples can be found [here](https://github.com/ROCm/hip-python) and a detailed guide is available [here](https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html).
