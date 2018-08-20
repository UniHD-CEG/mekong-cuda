#ifndef __ME_RUNTIME_PRIVATE
#define __ME_RUNTIME_PRIVATE

#include <stdio.h>

#include "VirtualBuffer.h"
#include <vector>
#include <utility>

#undef assert
#define assert(cond) do { if (!(cond)) { printf("%s\n", #cond); abort(); } } while (0)

#define cudaAssert(cond) do { if ((cond) != cudaSuccess) { printf("%s == %s\n", #cond, cudaGetErrorString(cudaGetLastError())); abort(); } } while (0)

#define unreachable(msg) do { printf("%s\n", msg); abort(); } while (0)

#ifdef NOLOG
  #define MELOG(lvl, ...) do{;}while(0)
#else
  #define MELOG(lvl, ...) do{ if (lvl <= meState.log_level) { printf("%*s", 2*lvl, ""); printf(__VA_ARGS__); } }while(0)
#endif

/* Provides a distribution of a buffer of a given size across some number of GPUs.
 * @param  buffer_size  size of the buffer
 * @param  gpus         number of gpus to distribute across partition
 * @param  start        return next chunk that starts here or later
 * @return a tuple containing chunk start, chunk size, and target gpu
 */
typedef std::tuple<int64_t,int64_t,int>(*__me_pattern_fn)(size_t buffer_size, int gpus, size_t start);

#define DEFAULTLOGLEVEL 0

/* available log levels:
 * 0 CRITICAL
 * 1 ERROR
 * 2 WARN
 * 3 OPERATION
 * 4 TRANSFER
 * 5 INTERVAL
 * 6 CHUNK
 * 7 DEBUG
 */

typedef struct {
  const void* reference;
  void* shadow;
  size_t size;
} ShadowCopy;

enum DistributeMode {
  DISTRIBUTE_INVALID,
  DISTRIBUTE_DEFER_SAFE,
  DISTRIBUTE_DEFER_UNSAFE,
  DISTRIBUTE_LINEAR,
};

typedef struct {
  bool initialized;
  int numGPUs;
  std::vector<ShadowCopy> shadows;

  int log_level;
  enum DistributeMode dist_mode;
} MeState;

#endif
