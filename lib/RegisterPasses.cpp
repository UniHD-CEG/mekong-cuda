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
  if (MekongEnabled) {
    PM.add(mekong::createMeCodegen(MekongModel));
    PM.add(mekong::createMeKernelSubgrid());
  }
}

static void
registerMekongPassesEarly(const llvm::PassManagerBuilder &Builder,
    llvm::legacy::PassManagerBase &PM) {
  registerMekongPasses(PM);
}

static llvm::RegisterStandardPasses RegisterMekongEarly(
    //llvm::PassManagerBuilder::EP_ModuleOptimizerEarly,
    llvm::PassManagerBuilder::EP_VectorizerStart,
    registerMekongPassesEarly);

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
