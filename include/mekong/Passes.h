#ifndef MEKONG_LLVM_ANALYSIS_PASSES_H
#define MEKONG_LLVM_ANALYSIS_PASSES_H


namespace llvm {
  class FunctionPass;
  class ImmutablePass;
  class LoopPass;
  class ModulePass;
  class Pass;
  class PassInfo;
  class StringRef;

  FunctionPass *createPolyhedralValueInfoWrapperPass();
  FunctionPass *createPolyhedralAccessInfoWrapperPass();
  FunctionPass *createPolyhedralDependenceInfoWrapperPass();

}

namespace mekong {
  llvm::Pass *createMeKernelAnalysis();
  llvm::Pass *createMeKernelAnalysisWrapper(llvm::StringRef File);
  llvm::Pass *createMeCodegen();
  llvm::Pass *createMeCodegen(llvm::StringRef File);
  llvm::Pass *createMeKernelSubgrid();
}

#endif
