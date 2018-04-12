#include "mekong/RegisterPasses.h"

#include "mekong/LinkAllPasses.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace mekong;

static cl::opt<std::string>
    MekongModel("mekong-model", cl::desc("Mekong model"),
        cl::init(""), cl::ZeroOrMore);

static cl::opt<bool>
    MekongPreEnabled("mekong-pre", cl::desc("Enable mekong pre-processing"),
        cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
    MekongEnabled("mekong", cl::desc("Enable mekong"),
        cl::init(false), cl::ZeroOrMore);

namespace mekong {

// These passes should profit from all standard optimization
static void registerEarlyMekongPasses(const llvm::PassManagerBuilder &Builder,
    llvm::legacy::PassManagerBase &PM) {
  if (MekongEnabled) {
    PM.add(mekong::createMeCodegen(MekongModel));
    PM.add(mekong::createMeKernelSubgrid());
  }
  if (MekongPreEnabled) {
  }
}

// These passes should work on optimized (and canonicalized) IR
static void registerLateMekongPasses(const llvm::PassManagerBuilder &Builder,
    llvm::legacy::PassManagerBase &PM) {
  if (MekongPreEnabled) {
    PM.add(mekong::createMeKernelAnalysisWrapper(MekongModel));
  }
  if (MekongEnabled) {
  }
}

static llvm::RegisterStandardPasses RegisterEarlyMekong(
    llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
    registerEarlyMekongPasses);

static llvm::RegisterStandardPasses RegisterLateMekong(
    llvm::PassManagerBuilder::EP_OptimizerLast,
    registerLateMekongPasses);

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
