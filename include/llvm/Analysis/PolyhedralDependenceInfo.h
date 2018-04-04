//===--- PolyhedralDependenceInfo.h -- Polyhedral dep. analysis -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLYHEDRAL_DEPENDENCE_INFO_H
#define POLYHEDRAL_DEPENDENCE_INFO_H

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/PValue.h"

namespace llvm {

class Loop;
class LoopInfo;
class PolyhedralValueInfo;
class PolyhedralAccessInfo;

/// Representation of a polyhedral dependence.
class PDEP {
public:

  enum DependenceKind {
    DK_WAR,
    DK_RAW,
    DK_WAW,
  };

  /// Print this polyhedral representation to @p OS.
  void print(raw_ostream &OS) const;

  /// Dump this polyhedral representation to the dbgs() stream.
  void dump() const;

private:

  bool IsExact;

  DependenceKind DepKind;
};

class PolyhedralDependenceInfo {

  /// The PolyhedralAccessInfo used to get access information.
  PolyhedralAccessInfo &PAI;

  LoopInfo &LI;

public:
  /// Constructor
  PolyhedralDependenceInfo(PolyhedralAccessInfo &PAI, LoopInfo &LI);

  ~PolyhedralDependenceInfo();

  bool isVectorizableLoop(llvm::Loop &L);

  /// Clear all cached information.
  void releaseMemory();

  /// Print some statistics to @p OS.
  void print(raw_ostream &OS) const;
  void dump() const;
};

/// Wrapper pass for PolyhedralDependenceInfo on a per-function basis.
class PolyhedralDependenceInfoWrapperPass : public FunctionPass {
  PolyhedralDependenceInfo *PDI;
  Function *F;

public:
  static char ID;
  PolyhedralDependenceInfoWrapperPass() : FunctionPass(ID) {}

  /// Return the PolyhedralDependenceInfo object for the current function.
  PolyhedralDependenceInfo &getPolyhedralDependenceInfo() {
    assert(PDI);
    return *PDI;
  }

  /// @name Pass interface
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
  virtual void releaseMemory() override;
  virtual bool runOnFunction(Function &F) override;
  //@}

  virtual void print(raw_ostream &OS, const Module *) const override;
  void dump() const;
};

class PolyhedralDependenceInfoAnalysis
    : public AnalysisInfoMixin<PolyhedralDependenceInfoAnalysis> {
  friend AnalysisInfoMixin<PolyhedralDependenceInfoAnalysis>;
  static AnalysisKey Key;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef PolyhedralDependenceInfo Result;

  /// \brief Run the analysis pass over a function and produce BFI.
  Result run(Function &F, FunctionAnalysisManager &AM);
};

raw_ostream &operator<<(raw_ostream &OS, PDEP::DependenceKind Kind);
raw_ostream &operator<<(raw_ostream &OS, const PDEP *PD);
raw_ostream &operator<<(raw_ostream &OS, const PDEP &PD);

} // namespace llvm
#endif
