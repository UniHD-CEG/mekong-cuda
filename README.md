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

Mekong adds two new commands in the `bin` directory: `mekongcc` and `mekongrw`.
`mekongcc` is a thin clang++ wrapper that loads the Mekong LLVM Module and
transforms some of the arguments it is given.
`mekongrw` is a lua based rewriter for CUDA code.

In order to compile an application `myapp.cu` into a binary 'myapp' for
multiple GPUs, follow these steps:

1. `$ mekongcc -mekong-pre -mekong-model=model.yaml -o /dev/null myapp.cu`
2. `$ mekongrw -info=model.yaml myapp.cu > tmp.cu`
3. `$ mekongcc -mekong -mekong-model=model.yaml -O3 -o myapp tmp.cu \` \
   `-L /usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread`

For details about the long list of linker flags for the last step, consult
[Compiling CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html).
