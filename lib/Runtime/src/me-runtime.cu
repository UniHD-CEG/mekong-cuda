// vim: ts=2 sts=2 sw=2 et ai
#include "cuda.h"
#include "cuda_runtime_api.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "me-runtime.h"
#include "me-runtime-private.h"


// format: ":: copying  start-offset .. end-offset, base_src (tag_src) -> base_dst (tag_dest)
#define PRITRANSFER ":: copying %" PRId64 " .. %" PRId64 ", %p (%d) -> %p (%d)\n"
#define PRITRANSFERHOST ":: copying %" PRId64 " .. %" PRId64 ", %p (%d) -> %p (N/A)\n"

// format: ":: tagging  start-offset .. end-offset, virtualbuf = base (tag)
#define PRITAGGING  ":: tagging %" PRId64 " .. %" PRId64 ", %p = %p (%d)\n"

// format: ":: alloc size -> buf (tag)
#define PRIALLOC  ":: alloc %lu -> %p (%d)\n"

static MeState meState;

/*******************************************************************************
 * PARTITIONING SCHEMES
 *******************************************************************************/

void __me_partition_linear_x(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid) {
  int xChunkSize = origGrid.x / nParts;
  part->kernel.zmin = 0;
  part->kernel.zmax = origGrid.z;
  part->kernel.ymin = 0;
  part->kernel.ymax = origGrid.y;
  if (partIdx == nParts-1) {
    part->kernel.xmin = partIdx * xChunkSize;
    part->kernel.xmax = origGrid.x;
  } else {
    part->kernel.xmin = partIdx * xChunkSize;
    part->kernel.xmax = (partIdx+1) * xChunkSize;
  }
}

void __me_partition_linear_y(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid) {
  int yChunkSize = origGrid.y / nParts;
  part->kernel.zmin = 0;
  part->kernel.zmax = origGrid.z;
  if (partIdx == nParts-1) {
    part->kernel.ymin = partIdx * yChunkSize;
    part->kernel.ymax = origGrid.y;
  } else {
    part->kernel.ymin = partIdx * yChunkSize;
    part->kernel.ymax = (partIdx+1) * yChunkSize;
  }
  part->kernel.xmin = 0;
  part->kernel.xmax = origGrid.x;
}

void __me_partition_linear_z(__subgrid_t* part, int partIdx, int nParts, dim3 origGrid) {
  int zChunkSize = origGrid.x / nParts;
  if (partIdx == nParts-1) {
    part->kernel.zmin = partIdx * zChunkSize;
    part->kernel.zmax = origGrid.z;
  } else {
    part->kernel.zmin = partIdx * zChunkSize;
    part->kernel.zmax = (partIdx+1) * zChunkSize;
  }
  part->kernel.ymin = 0;
  part->kernel.ymax = origGrid.y;
  part->kernel.xmin = 0;
  part->kernel.xmax = origGrid.x;
}

/*******************************************************************************
 * MEKONG CUSTOM FUNCTIONALITY
 *******************************************************************************/

void __me_initialize() {
  if (meState.initialized) return;
  assert(cudaGetDeviceCount(&meState.numGPUs) == cudaSuccess);
  if (getenv("MELOGLEVEL")) {
    meState.log_level = atoi(getenv("MELOGLEVEL"));
  } else {
    meState.log_level = DEFAULTLOGLEVEL;
  }
  meState.initialized = true;
}

int __me_num_gpus() {
  __me_initialize();
  return meState.numGPUs;
}

void __me_sync() {
  const int n = __me_num_gpus();
  for (int i = 0; i < n; ++i) {
    assert(cudaSetDevice(i) == cudaSuccess && "unable to set device");
    assert(cudaDeviceSynchronize() == cudaSuccess && "unable to synchronize device");
  }
}

void* __me_nth_array(void* buf, int i) {
  VirtualBuffer *vb = (VirtualBuffer*)buf;
  return vb->getInstance(i+1);
}

/** Auxiliary structs for the buffer synchronization
 */
typedef struct __me_sync_info {
  VirtualBuffer *vb;
  MemTracker<int> *mt;
  int64_t start;
  int64_t end;
  int lastSrc;
  int dest;
  int elementSize;
  bool initialized;
} __me_sync_info;

static void __me_sync_flush(__me_sync_info *info) {
  int64_t start =  info->start;
  int64_t end =  info->end;
  int from = info->lastSrc;
  int to = info->dest;
  VirtualBuffer *vb = info->vb;

  assert(from != 0 && "invalid transfer source");
  assert(to != 0 && "invalid transfer destination");
  if (from == to) return;

  char *srcBufBase = (char*)vb->getInstance(from);
  char *dstBufBase = (char*)vb->getInstance(to);
  char *srcBuf = srcBufBase + start;
  char *dstBuf = dstBufBase + start;
  size_t count = end - start;

  MELOG(4, PRITRANSFER, start, end, srcBufBase, from, dstBufBase, to);

  cudaError_t result;
  if (from < 0 && to < 0) {
    cudaAssert(cudaMemcpyAsync(dstBuf, srcBuf, count, cudaMemcpyHostToHost));
  } else if (from < 0 && to > 0) {
    cudaAssert(cudaMemcpyAsync(dstBuf, srcBuf, count, cudaMemcpyHostToDevice));
  } else if (from > 0 && to < 0) {
    cudaAssert(cudaMemcpyAsync(dstBuf, srcBuf, count, cudaMemcpyDeviceToHost));
  } else if (from > 0 && to > 0) {
    cudaAssert(cudaMemcpyAsync(dstBuf, srcBuf, count, cudaMemcpyDeviceToDevice));
  } else {
    unreachable("this should not have happened");
  }
}

static void __me_sync_cb(int64_t lower, int64_t upper, void* user) {
  __me_sync_info *info = (__me_sync_info*)user;

  MELOG(5, ":: interval %" PRId64 " .. %" PRId64 "\n", lower, upper);
  MemTracker<int>::Chunk chunk;
  int64_t lowerReal = lower * info->elementSize;
  int64_t upperReal = upper * info->elementSize;
  while (info->mt->queryRange2(lowerReal, upperReal, &chunk)) {
    int src = chunk.tag;
    int64_t start = chunk.start;
    int64_t end = chunk.end;
    //int elementSize = info->elementSize;
    MELOG(6, ":: chunk: %" PRId64 " .. %" PRId64 " @ %d\n", start, end, src);

    // initialized info at first chunk (dummy zero length interval)
    if (!info->initialized) {
      info->lastSrc = src;
      info->start = start;
      info->end = start;
      info->initialized = true;
    }

    // flush if 1) source of last chunk differs or 2) last chunk does not coalesce
    bool needsFlush = (src != info->lastSrc) || (start != info->end);
    if (needsFlush) {
      __me_sync_flush(info);
      // reset info to start at current chunk
      info->start = start;
      info->end = end;
      info->lastSrc = src;
    } else {
      // append current chunk to info
      info->end = end;
    }
  }
}

/** Buffer synchronize according to the memory accessed by the gpu "forGPU"
 * given some partition, a set of parameters and a kernel.
 * 
 * Used in preparation for the read accesses of a GPU.
 * @param buf          Virtual buffer that needs synchronization
 * @param forGPU       GPU index that requires data (>= 0)
 * @param iterator     Chunk iterator for this read access
 * @param elementSize  Size in *bytes* of the elements in the buffer, as
 *                     used by the kernel. Size may differ between kernels, e.g.
 *                     one kernel uses the buffer as a float-array, another as
 *                     a byte-array.
 * @param subgrid      The subgrid of the kernel the buffer is synced for.
 *                     Required by the iterator.
 * @param params       Copy of the kernel arguments that are used as parameters
 *                     to accesses to the buffer. Required by the iterator.
 */
void __me_buffer_sync(void* buf, int forGPU, __me_itfn_t iterator,
    int elementSize, __subgrid_t* subgrid, int64_t* params) {
  VirtualBuffer *vb = (VirtualBuffer*)buf;
  MemTracker<int> &mt = vb->getTracker();
  __subgrid_full_t *full = &subgrid->full;
  int64_t grid[15] = { full->zmin * full->zdim, full->zmax * full->zdim,
    full->ymin * full->ydim, full->ymax * full->ydim,
    full->xmin * full->xdim, full->xmax * full->xdim,
    full->zmin, full->zmax, full->ymin, full->ymax, full->xmin, full->xmax,
    full->zdim, full->ydim, full->xdim };
  MELOG(3, ":: buffer sync to device %d, elSize: %d\n", forGPU+1, elementSize);
  MELOG(7, ":: gridDim: %" PRId64 ", %" PRId64 "; %" PRId64 ", %" PRId64
      "; %" PRId64 ",%" PRId64 "  blockDim: %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
      grid[6], grid[7], grid[8], grid[9], grid[10], grid[11],
      grid[12], grid[13], grid[14]);
  __me_sync_info info = {vb, &mt, 0, 0, 0, forGPU+1, elementSize, false};
  iterator(grid, params, __me_sync_cb, &info);
  // if there was at least one chunk (indicated by lastSrc being updated), flush last chunk
  if (info.lastSrc != 0) {
    __me_sync_flush(&info);
  }
  //assert(cudaStreamSynchronize(0) == cudaSuccess);
}

/** Synchronize whole buffer for a given GPU, ignore memory access patterns.
 * As opposed to most of the other primitives, undefined chunks are not an
 * error. Since there is no information about the array size supplied by a
 * memory access pattern or the user, we have simply copy all valid chunks and
 * ignore invalid ones, thus automatically staying inside the limits of the array.
 * 
 * Used as preparatory step for execution of unsplittable kernels.
 * @param buf     Virtual buffer to synchronize
 * @param forGPU  GPU that all data is gathered to (>= 0)
 */
void __me_buffer_sync_all(void* buf, int forGPU) {
  VirtualBuffer *vb = (VirtualBuffer*)buf;
  MemTracker<int> &mt = vb->getTracker();

  assert(forGPU >= 0 && "invalid target GPU");
  MELOG(3, ":: buffer sync all to device %d\n", forGPU+1);

  MemTracker<int>::Chunk chunk;
  int64_t count = INT64_MAX;

  int dstTag = forGPU+1;
  void *dstBase = vb->getInstance(dstTag);

  while (mt.queryRange2(0, count, &chunk)) {
    MELOG(6, ":: chunk %" PRId64 " .. %" PRId64 " @ %d\n", chunk.start, chunk.end, chunk.tag);
    if (chunk.tag == 0) {
      continue;
    }
    assert(chunk.tag > 0 && "sync all only supported for gpu");
    void *srcBase = vb->getInstance(chunk.tag);
    char* chunkDst = (char*)dstBase + chunk.start;
    char* chunkSrc = (char*)srcBase + chunk.start;
    size_t chunkSize = chunk.end - chunk.start;
    MELOG(4, PRITRANSFER, chunk.start, chunk.end, srcBase, chunk.tag, dstBase, dstTag);
    cudaAssert(cudaMemcpyAsync(chunkDst, chunkSrc, chunkSize, cudaMemcpyDeviceToDevice));
  }
  assert(cudaStreamSynchronize(0) == cudaSuccess);
}

/** Auxiliary structs for the buffer update
 */
typedef struct __me_update_info {
  VirtualBuffer *vb;
  MemTracker<int> *mt;
  int tag;
  int elementSize;
  int64_t start;
  int64_t end;
} __me_update_info;

static void __me_update_flush(__me_update_info *info) {
  int64_t lowerReal = info->start * info->elementSize;
  int64_t upperReal = info->end * info->elementSize;
  MELOG(4, ":: tagging %p, %" PRId64 " .. %" PRId64 " = %d\n", info->vb, lowerReal, upperReal, info->tag);
  info->mt->update(lowerReal, upperReal, info->tag);
}

static void __me_update_cb(int64_t lower, int64_t upper, void* user) {
  __me_update_info *info = (__me_update_info*)user;

  // if first range, initialize chunk
  if (info->start == -1) {
    info->start = lower;
    info->end = upper;
  } else {
    if (info->end < lower) {
      // if chunk is not contiguous with current one, flush and reinitialize
      __me_update_flush(info);
      info->start = lower;
      info->end = upper;
    } else {
      // otherwise just append
      info->end = upper;
    }
  }
}

/** Buffer update according to the memory accessed by the gpu "forGPU"
 * given some partition, a set of parameters and a kernel.
 *
 * Being used to account for write accesses of a GPU in a kernel.
 *
 * @param buf          Buffer to synchronize
 * @param forGPU       Tag to use for updated chunks
 * @param iterator     Iterator for this write access.
 * @param elementSize  Size of elements in *bytes* in the buffer. May differ between kernels.
 * @param subgrid      Kernel grid partition to update for.
 * @param params       Copy of kernel parameters that act as isl arguments.
 */
void __me_buffer_update(void* buf, int forGPU, __me_itfn_t iterator,
    int elementSize, __subgrid_t* subgrid, int64_t* params) {
  VirtualBuffer *vb = (VirtualBuffer*)buf;
  MemTracker<int> &mt = vb->getTracker();

  MELOG(3, ":: buffer update for device %d\n", forGPU+1);

  __subgrid_full_t *full = &subgrid->full;
  int64_t grid[15] = { full->zmin * full->zdim, full->zmax * full->zdim,
    full->ymin * full->ydim, full->ymax * full->ydim,
    full->xmin * full->xdim, full->xmax * full->xdim,
    full->zmin, full->zmax, full->ymin, full->ymax, full->xmin, full->xmax,
    full->zdim, full->ydim, full->xdim };
  __me_update_info info = { vb, &mt, forGPU+1, elementSize, -1, -1 };
  iterator(grid, params, __me_update_cb, &info);
  // flush last iteration
  __me_update_flush(&info);
}

/** Update whole buffer, ignore memory access patterns. First identifiestotal
 * size of buffer by iterating through all elements. Then issues one update
 * over full range.
 *
 * Used as worst case scenario to account for writes of opaque kernel.
 *
 * @param buf     Buffer to update.
 * @param forGPU  GPU to set tag for.
 */
void __me_buffer_update_all(void* buf, int forGPU) {
  VirtualBuffer *vb = (VirtualBuffer*)buf;
  MemTracker<int> &mt = vb->getTracker();

  assert(forGPU >= 0 && "invalid owner GPU");
  MELOG(3, ":: buffer update all to device %d\n", forGPU+1);

  MemTracker<int>::Chunk chunk;
  int64_t count = INT64_MAX;
  int64_t lower = INT64_MAX;
  int64_t upper = -INT64_MAX;

  while (mt.queryRange2(0, count, &chunk)) {
    MELOG(6, ":: chunk %" PRId64 " .. %" PRId64 " @ %d\n", chunk.start, chunk.end, chunk.tag);
    if (chunk.tag == 0) {
      continue;
    }
    if (chunk.start < lower) lower = chunk.start;
    if (chunk.end > upper) upper = chunk.end;
  }
  MELOG(4, ":: tagging %p, %" PRId64 " .. %" PRId64 " = %d\n", vb, lower, upper, forGPU+1);
  mt.update(lower, upper, forGPU+1);
}

/** "Gather" buffer pieces from potentially all devices to the host.
 * This immediately initiates transfers, unless the data is already located
 * in the host buffer.
 * Conditions:
 *  - dst is a host buffer
 *  - src is a virtual buffer
 */
cudaError_t __me_buffer_gather(void* dstBase, const void* src, size_t count) {
  VirtualBuffer *vb = (VirtualBuffer*)src;
  MemTracker<int> &mt = vb->getTracker();

  MELOG(3, ":: buffer gather to %p\n", dstBase);

  MemTracker<int>::Chunk chunk;
  cudaError_t res;
  while (mt.queryRange2(0, count, &chunk)) {
    MELOG(6, ":: chunk %" PRId64 " .. %" PRId64 " @%d\n", chunk.start, chunk.end, chunk.tag);
    assert(chunk.tag != 0 && "requested range contains undefined chunks");
    assert(chunk.tag > 0 && "gather only supported from gpu for now");
    void *srcBase = vb->getInstance(chunk.tag);
    char* chunkDst = (char*)dstBase + chunk.start;
    char* chunkSrc = (char*)srcBase + chunk.start;
    size_t chunkSize = chunk.end - chunk.start;

    MELOG(4, PRITRANSFERHOST, chunk.start, chunk.end, srcBase, chunk.tag, dstBase);

    //MELOG(4, ":: transfer %p : % 4" PRId64 " .. % 4" PRId64 ", %p -> %p\n",
    //    vb, chunk.start, chunk.end, devInst, dst);
    res = cudaMemcpyAsync(chunkDst, chunkSrc, chunkSize, cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) {
      break;
    }
  }
  
  return res;
}

/** "Broadcast" buffer from host to all devices.
 * This operation does not initiate any actual transfers. The only side effect
 * is a buffer update that signals that all data is located on the host.
 *
 * Conditions:
 *  - dst is a virtual buffer
 *  - src is a host buffer
 */
cudaError_t __me_buffer_broadcast(void* dst, const void* src, size_t count) {
  VirtualBuffer *vb = (VirtualBuffer*)dst;
  MemTracker<int> &mt = vb->getTracker();

  MELOG(3, ":: buffer broadcast from host buffer %p\n", src);

  int ref = vb->findOrInsertHostReference((void*)src); // :'(
  mt.update(0, count, ref);
  MELOG(4, PRITAGGING, (int64_t)0, (int64_t)count, vb, src, ref);
  return cudaSuccess;
}

/*******************************************************************************
 * MEKONG CUDA WRAPPERS
 *******************************************************************************/

cudaError_t __meGetDeviceCount(int *count) {
  *count = 1;
  return cudaSuccess;
}

cudaError_t __meDeviceSynchronize() {
  const int n = __me_num_gpus();
  for (int i = 0; i < n; ++i) {
    cudaError_t res = cudaSetDevice(i);
    if (res != cudaSuccess) return res;
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) return res;
  }
  return cudaSuccess;
}

cudaError_t __meMalloc(void** devPtr, size_t size) {
  const int n = __me_num_gpus();
  VirtualBuffer *vb = new VirtualBuffer(n);

  MELOG(3, ":: meMalloc, vb %p, size %lu\n", vb, size);

  for (int i = 0; i < n; ++i) {
    assert(cudaSetDevice(i) == cudaSuccess && "unable to set device");
    void* dst;
    assert(cudaMalloc(&dst, size) == cudaSuccess && "unable to allocate buffer on device");
    vb->setDevInstance(i+1, dst); // devices: 1 .. inf
    MELOG(4, PRIALLOC, size, dst, i+1);
  }
  assert(cudaSetDevice(0) == cudaSuccess && "unable to reset device to first");
  *devPtr = (void*)vb;
  return cudaSuccess;
}

cudaError_t __meFree(void* devPtr) {
  const int n = __me_num_gpus();
  VirtualBuffer *vb = (VirtualBuffer*)devPtr;

  MELOG(3, ":: meFree for  %p\n", devPtr);

  for (int i = 0; i < n; ++i) {
    assert(cudaSetDevice(i) == cudaSuccess && "unable to set device");
    void* dst = vb->getInstance(i+1);
    assert(cudaFree(dst) == cudaSuccess && "unable to free buffer");
  }
  delete vb;
  assert(cudaSetDevice(0) == cudaSuccess && "unable to reset device to first");
  return cudaSuccess;
}

cudaError_t __meMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  cudaError_t result = __meMemcpyAsync(dst, src, count, kind);
  cudaAssert(cudaStreamSynchronize(0));
  return result;
}

cudaError_t __meMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  cudaError_t result;
  switch (kind) {
  case cudaMemcpyHostToDevice:
    result = __me_buffer_broadcast(dst, src, count);
    break;
  case cudaMemcpyHostToHost:
    result = cudaMemcpy(dst, src, count, kind);
    break;
  case cudaMemcpyDeviceToDevice:
    assert(false && "Device to device copy not supported");
  case cudaMemcpyDeviceToHost:
    result = __me_buffer_gather(dst, src, count);
    break;
  case cudaMemcpyDefault:
    assert(false && "Unified Virtual Address Space not supported");
  default:
    assert(false && "invalid cudaMemcpyKind");
  }
  return result;
}
