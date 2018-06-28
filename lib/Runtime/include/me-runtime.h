#ifndef __ME_RUNTIME_TYPES
#define __ME_RUNTIME_TYPES

#include <cuda.h>

typedef struct __subgrid_kernel_t {
    int64_t zmin, zmax, ymin, ymax, xmin, xmax;
} __subgrid_kernel_t;
typedef struct __subgrid_full_t {
    int64_t zmin, zmax, ymin, ymax, xmin, xmax, zdim, ydim, xdim;
} __subgrid_full_t;
typedef union __subgrid_t {
  __subgrid_kernel_t kernel;
  __subgrid_full_t full;
} __subgrid_t;

/** Expected values in grid[] arguments:
 * { boffmin_z, boffmax_z, boffmin_y, boffmax_y, boffmin_x, boffmax_x,
 *   bidmin_z, bidmax_z, bidmin_y, bidmax_y, bidmin_x, bidmax_x,
 *   bdim_z, bdim_y, bdim_x}
 */

typedef void(*__me_cbfn_t)(int64_t lower, int64_t upper, void *user);
typedef void(*__me_itfn_t)(int64_t grid[], int64_t param[], __me_cbfn_t, void*);

extern "C" {
// CUDA wrappers
cudaError_t __meGetDeviceCount(int *);
cudaError_t __meDeviceSynchronize(void);
cudaError_t __meMalloc(void** devPtr, size_t size);
cudaError_t __meFree(void* devPtr);
cudaError_t __meMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t __meMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

// Mekong custom functions
int __me_num_gpus(void);
void __me_sync(void);
void* __me_nth_array(void* buf, int i);

// both functions require full subgrid
void __me_buffer_sync(void* buf, int forGPU, __me_itfn_t iterator,
    int elementSize, __subgrid_t* partitition, int64_t* params);
void __me_buffer_update(void* buf, int forGPU, __me_itfn_t iterator,
    int elementSize, __subgrid_t* partitition, int64_t* params);
// "dumb" versions ignoring memory access patterns
void __me_buffer_sync_all(void* buf, int forGPU);
void __me_buffer_update_all(void* buf, int forGPU);

cudaError_t __me_buffer_gather(void* dst, const void* src, size_t count);
cudaError_t __me_buffer_broadcast(void* dst, const void* src, size_t count);

// partitioning schemes
void __me_partition_linear_x(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid);
void __me_partition_linear_y(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid);
void __me_partition_linear_z(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid);
}

// C++ convenience interface
template<class T>
static __inline__ __host__ cudaError_t __meMalloc(T **devPtr, size_t size) {
  return ::__meMalloc((void**)(void*)devPtr, size);
}

#endif
