#include "mekong/RegisterPasses.h"

#include "mekong/LinkAllPasses.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace mekong;

static cl::opt<StringRef>
    MekongModel("mekong-model", cl::desc("Mekong model"),
        cl::init(""));

static cl::opt<bool>
    MekongPreEnabled("mekong-pre", cl::desc("Enable mekong pre-processing"),
        cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
    MekongEnabled("mekong", cl::desc("Enable mekong"),
        cl::init(false), cl::ZeroOrMore);

namespace mekong {

void registerMekongPasses(llvm::legacy::PassManagerBase &PM) {
  if (MekongPreEnabled) {
    PM.add(mekong::createMeKernelAnalysisWrapper(MekongModel));
  }
}

void initializeMekongPasses(llvm::PassRegistry &Registry) {
  //initializePolyhedralValueInfoWrapperPassPass(Registry);
  //initializePolyhedralValueTransformerWrapperPassPass(Registry);
  //initializePolyhedralAccessInfoWrapperPassPass(Registry);
  //initializePolyhedralDependenceInfoWrapperPassPass(Registry);
  
  //initializeMeKernelAnalysisPass(Registry);
  initializeMeKernelAnalysisWrapperPass(Registry);
  initializeMeCodegenPass(Registry);
  initializeMeKernelSubgridPass(Registry);
}

}
