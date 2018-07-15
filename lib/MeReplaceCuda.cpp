#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "me-host-replace-cuda"

namespace llvm {

struct MeHostReplaceCuda : public ModulePass {
  static char ID;
  MeHostReplaceCuda() : ModulePass(ID) {}

  bool substituteWrapperByName(Module &M, const StringRef &fnName, const StringRef &wrapperName) {
    Function *F = M.getFunction(fnName);
    if (!F) {
      return false;
    }
    Constant *Wrapper = M.getOrInsertFunction(wrapperName, F->getFunctionType());
    F->replaceAllUsesWith(Wrapper);
    return true;
  };
  
  bool runOnModule(Module &M) override {
    if (M.getTargetTriple() == "nvptx64-nvidia-cuda" ||
        M.getTargetTriple() == "nvptx-nvidia-cuda")
      return false;

    bool changed = false;
    changed |= substituteWrapperByName(M, "cudaMalloc",            "__meMalloc");
    changed |= substituteWrapperByName(M, "cudaFree",              "__meFree");
    changed |= substituteWrapperByName(M, "cudaMemcpy",            "__meMemcpy");
    changed |= substituteWrapperByName(M, "cudaMemcpyAsync",       "__meMemcpyAsync");
    changed |= substituteWrapperByName(M, "cudaGetDeviceCount",    "__meGetDeviceCount");
    changed |= substituteWrapperByName(M, "cudaDeviceSynchronize", "__meDeviceSynchronize");
    return changed;
  }
};

char MeHostReplaceCuda::ID = 0;

}
//
// Pass registration

using namespace llvm;

Pass *mekong::createMeHostReplaceCuda() {
  return new MeHostReplaceCuda();
}

INITIALIZE_PASS_BEGIN(MeHostReplaceCuda, "me-host-substitute",
                      "Mekong substitute CUDA calls in host code", false, false);
INITIALIZE_PASS_END(MeHostReplaceCuda, "me-host-substitute",
                      "Mekong substitute CUDA calls in host code", false, false);
