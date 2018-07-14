#ifndef MEKONG_LINKALLPASSES_H
#define MEKONG_LINKALLPASSES_H

#include <cstdlib>
#include "llvm/ADT/StringRef.h"
#include "mekong/Passes.h"

namespace {
struct MekongForcePassLinking {
  MekongForcePassLinking() {
    // We must reference the passes in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    if (std::getenv("bar") != (char *)-1)
      return;

    mekong::createMeKernelAnalysis();
    mekong::createMeKernelAnalysisWrapper("");
    mekong::createMeCodegen();
    mekong::createMeKernelSubgrid();
    mekong::createMeHostReplaceCuda();
  }
} MekongForcePassLinking; // Force link by creating a global definition.
} // namespace

#endif
