#ifndef MEKONG_INITIALIZEPASSES_H
#define MEKONG_INITIALIZEPASSES_H

namespace llvm {
class PassRegistry;

void initializePolyhedralValueInfoWrapperPassPass(PassRegistry&);
void initializePolyhedralValueTransformerWrapperPassPass(PassRegistry&);
void initializePolyhedralAccessInfoWrapperPassPass(PassRegistry&);
void initializePolyhedralDependenceInfoWrapperPassPass(PassRegistry&);

void initializeMeKernelAnalysisWrapperPass(PassRegistry&);
void initializeMeKernelAnalysisPass(PassRegistry&);
void initializeMeCodegenPass(PassRegistry&);
void initializeMeKernelSubgridPass(PassRegistry&);

} // namespace llvm


#endif
