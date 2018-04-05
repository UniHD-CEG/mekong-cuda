#ifndef __ME_RUNTIME_PRIVATE
#define __ME_RUNTIME_PRIVATE

#include <stdio.h>
#include "VirtualBuffer.h"

#undef assert
#define assert(cond) do { if (!(cond)) { printf("%s\n", #cond); abort(); } } while (0)

#define cudaAssert(cond) do { if ((cond) != cudaSuccess) { printf("%s == %s\n", #cond, cudaGetErrorString(cudaGetLastError())); abort(); } } while (0)

#define unreachable(msg) do { printf("%s\n", msg); abort(); } while (0)

#ifdef NOLOG
  #define MELOG(lvl, ...) do{;}while(0)
#else
  #define MELOG(lvl, ...) do{ if (lvl < meState.log_level) printf(__VA_ARGS__); }while(0)
#endif

#define DEFAULTLOGLEVEL 0

typedef struct {
  bool initialized;
  int numGPUs;
  int log_level;
} MeState;

#endif
