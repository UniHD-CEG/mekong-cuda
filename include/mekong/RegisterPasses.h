#ifndef MEKONG_REGISTER_PASSES_H
#define MEKONG_REGISTER_PASSES_H

#include "llvm/IR/LegacyPassManager.h"
#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

namespace llvm {
namespace legacy {
class PassManagerBase;
} // namespace legacy
} // namespace llvm

namespace mekong {
void initializeMekongPasses(llvm::PassRegistry &Registry);
void registerMekongPasses(llvm::legacy::PassManagerBase &PM);
} // namespace polly
#endif
