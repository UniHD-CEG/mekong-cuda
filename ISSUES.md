# Issues

In contrast to many optimizations based on polyhedral compilation, whether to
partitioning a kernel for execution on multiple GPUs is a binary decision.
The write pattern of a kernel into global memory must be proven to be an injective
mapping from thread IDs to array elements.
The model of read patterns allow more graduation, more accurate read patterns
result in communication that is closer to the optimum but even completely random
read accesses to global memory allow the kernel to be partitioned in many cases.

The following conditions are required for an application to compile successful:

- kernels only used in the translation unit they are defined in
- all translation units containing cuda calls translated with mekcc

The following conditions are required for a kernel to be partioned:

- injective-write accesses
- bounded read-accesses
- all `__device__` function inlinable
