# [NVIDIA Warp](https://developer.nvidia.com/warp-python)

## Purpose & Prerequisites

<!-- Document the library's core purpose and what's needed to use it. -->

## Installation & Basic Functionality

<!-- Verify installation and demonstrate basic functionality through scripts, playbooks, or demos in a scripts/ or examples/ subdirectory. -->

### Prerequisites
Warp can run on both CPU and GPU.

Hardware:
- CPU: x86-64 and ARMv8 on Windows and Linux. Apple Silicon (ARM64) required for macOS
- GPU (optional): CUDA-capable NVIDIA GPU
    - Warp packages built with CUDA Toolkit 12.x require NVIDIA driver 525 or newer
    - Warp packages built with CUDA Toolkit 13.x require NVIDIA driver 580 or newer

Software:
- Python 3.10+ (this example was developed on 3.10.12)
- Numpy

For other setups (e.g. Docker, Omniverse, nightly builds, Conda), please refer to the full [installation guide](https://nvidia.github.io/warp/stable/user_guide/installation.html)

### 1. Install Warp
```
cd warp
pip install -r requirements.txt
```

### 2. Verify installation
To check that Warp is installed correctly, from `/warp` do
```
cd examples
python3 gravity.py
```
This simulates 1M particles under gravitational attraction. Use the `--device` flag to set "cpu" or "cuda". For example, to run on gpu do
```
python3 gravity.py --device cuda
```
You should see something like the following in your terminal:
```
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"                                              
     "cuda:0"   : "NVIDIA L4" (22 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/ubuntu/.cache/warp/1.14.0
Module __main__ 3a8422f load on device 'cpu' took 4179.14 ms  (compiled)        # this is where the device cpu/cuda will be listed
[[  30.10494  -127.82458    86.85022 ]
 [ -92.41908   151.37328   105.72303 ]
 [ -67.45894   143.14563    14.270473]
 ...
 [  89.36835    -4.799613   10.386461]
 [ -20.704914   35.566715  109.66058 ]
 [   6.628495  -30.841984   65.58018 ]]
```

## Relevant Use Case

<!-- Identify a workflow within an industry (vertical) or segment (horizontal) where this library is highly relevant. Explain how it integrates into the end-to-end workflow. -->

## Helpful Links

- [Official Documentation](https://nvidia.github.io/warp/stable/)
- [GitHub Repository](https://github.com/nvidia/warp)
- [NVIDIA Developer Page](https://developer.nvidia.com/warp-python)

## Contributor
[Megan Tian](https://github.com/Megan-Tian)