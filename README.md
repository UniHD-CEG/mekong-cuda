# GPU Mekong

The main objective of (GPU) Mekong is to provide a simplified path to scale out
the execution of GPU programs from one GPU to almost any number, independent of
whether the GPUs are located within one host or distributed at the cloud or
cluster level. Unlike existing solutions, this work proposes to maintain the
GPU’s native programming model, which relies on a bulk-synchronous,
thread-collective execution; that is, no hybrid solutions like OpenCL/CUDA
programs combined with message passing are required. As a result, we can
maintain the simplicity and efficiency of GPU computing in the scale-out case,
together with a high productivity and performance.

Homepage: [www.gpumekong.org](www.gpumekong.org)

Note: this is work in progress

# Installation

GPU Mekong is an external project to LLVM/clang, so for the most part follow
the [Getting Started Guide](https://llvm.org/docs/GettingStarted.html), except
for these changes:

1. before running `cmake`, clone mekong into your `llvm/tools` directory, just
   like clang, e.g.:
  - `$ cd where-you-want-llvm-to-live`
  - `$ cd llvm/tools`
  - `$ git clone github.com/unihd-ceg/mekong-cuda`
2. apply the patch `llvm-enable-mekong.patch`:
  - `$ cd where-you-want-llvm-to-live`
  - `$ cd llvm`
  - `$ patch -p1 < tools/mekong/llvm-enable-mekong.diff`

Now you should be able to compile and install llvm as usual.

# Usage

Mekong adds a new tool to the bin directory, the compiler driver "mekc".
It supports a subset of the gpucc arguments and orchestrates the pipeline.
CUDA linking directives are automatically added, except for the linker
search path.

So in order to compiler an application `myapp.cu` to the binary `myapp`,
use the following command:

`$ mekc myapp.cu -o myapp -L /usr/local/cuda/lib64`

For details about gpucc CUDA compilation, consult
[Compiling CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html).

Two new compilation steps are added after preprocessing but
before compiling: "polyhedral analysis" and "rewriting".
Analog to the `-c` switch, there is a new switch `-A` that terminates
compilation after analysis and a switch `-R` that terminates compilation
after rewriting the source code.

# Application configuration

Applications compiled with Mekong generally do not require configuration.
However, there are some environment variables that influence the behavior
of the mekong system, and therefore the application, at runtime.

- `CUDA_VISIBLE_DEVICES`. Although not directly used by Mekong directly, this variable
  determines the set of visible (= utilized) GPUs. Valid values are comma separated lists
  of GPU indices starting at `0`. E.g. to only enable the first four available GPUs, use
  `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- `MELOGLEVEL`. Set to numbers `0` to `8` to enable logging. Currently only
  log levels `3` through `7` are in use. Higher numbers correspond to more
  detailed, verbose logging.
- `MEDEBUG`. Can be set to `notransfers` to disable all transfers for buffer
  synchronization or `nopatterns` to disable memory access patterns and
  transfers. These options are used for timing analysis.
