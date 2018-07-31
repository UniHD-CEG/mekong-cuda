//===--- PolyhedralAccessInfo.h -- Polyhedral access analysis ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLYHEDRAL_ACCESS_INFO_H
#define POLYHEDRAL_ACCESS_INFO_H

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/PValue.h"
#include "llvm/Analysis/PolyhedralUtils.h"

#include <set>

namespace llvm {

class Loop;
class LoopInfo;
class PEXP;
class PolyhedralValueInfo;
class PolyhedralExpressionBuilder;

/// Representation of a polyhedral access.
class PACC {
public:

  enum AccessKind {
    AK_READ,
    AK_WRITE,
    AK_READ_WRITE, ///< Used for calls and intrinsics that have multiple effects.
  };

  PACC(Value &Pointer, PEXP &PE, PVId Id, AccessKind AccKind);
  ~PACC();

  const PEXP *getPEXP() const { return &PE; }

  PVId getId() const { return Id; }

  bool isRead() const { return AccKind == AK_READ; }
  bool isWrite() const { return AccKind == AK_WRITE; }

  Value *getBasePointer() const { return BasePointer; }
  Value *getPointer() const { return Pointer; }

  /// Print this polyhedral representation to @p OS.
  void print(raw_ostream &OS) const;

  /// Dump this polyhedral representation to the dbgs() stream.
  void dump() const;

private:

  Value *BasePointer;
  Value *Pointer;

  PVId Id;

  ///
  /// Note: The ownership of PE is with this PACC!
  PEXP &PE;

  AccessKind AccKind;
};

/// Summary of PACCs in a certain code region (e.g., a Function).
///
/// While PACCs are access descriptions in isolation, PACCSummary's allow to
/// look at multiple accesses at once and in a larger scope.
class PACCSummary {
public:

  struct ArrayInfo;
  using ArrayInfoMapTy = DenseMap<Value *, ArrayInfo *>;
  using KnownAccessesTy = SmallVector<const PACC *, 4>;
  using UnknownInstructionsTy = SmallVector<Instruction *, 4>;

  enum SummaryScopeKind {
    SSK_EXTERNAL, ///< Accesses accessible outside the code region.
    SSK_INTERNAL, ///< Accesses accessible inside the code region.
    SSK_COMPLETE  ///< Accesses accessible outside and inside the code region.
  };

  struct ArrayInfo {
    /// The unit in which accesses to this pointer are measured (in bytes).
    ///
    /// If we have only accesses with the same type (or at least type size) this
    /// value will be the type size, e.g., 4 if we have float and i32 accesses
    /// to this array. However, if there are accesses that use differently sized
    /// types, e.g., i64 and i32, this will be the greatest common divisor (gcd)
    /// of all used access sizes in the code region.
    uint64_t ElementSize;

    /// The dimension sizes in case this array has a multidimensional view.
    SmallVector<const PEXP *, 4> DimensionSizes;

    DenseMap<const PACC *, PVMap> AccessMultiDimMap;

    PVMap MayReadMap;

    PVMap MustReadMap;

    PVMap MayWriteMap;

    PVMap MustWriteMap;

    void collectParameters(std::set<PVId> &ParameterSet) const;
    void print(raw_ostream &OS) const;
  };

  const ArrayInfo *getArrayInfoForPointer(const Value *Pointer) const {
    return ArrayInfoMap.lookup(Pointer);
  };

  /// Iterator interface for array infos.
  ///
  ///{
  using const_iterator = ArrayInfoMapTy::const_iterator;
  const_iterator begin() const { return ArrayInfoMap.begin(); }
  const_iterator end() const { return ArrayInfoMap.end(); }
  ///}

  size_t getNumReads() const { return KnownReads.size(); }
  size_t getNumWrites() const { return KnownWrites.size(); }
  size_t getNumUnknownReads() const { return UnknownReads.size(); }
  size_t getNumUnknownWrites() const { return UnknownWrites.size(); }

  /// Iterator interface for known read/write instructions.
  ///
  ///{
  using const_inst_iterator = KnownAccessesTy::const_iterator;
  const_inst_iterator reads_begin() const {
    return KnownReads.begin();
  }
  const_inst_iterator reads_end() const { return KnownReads.end(); }
  const_inst_iterator writes_begin() const {
    return KnownWrites.begin();
  }

  const_inst_iterator writes_end() const { return KnownWrites.end(); }
  ///}

  /// Iterator interface for unknown read/write instructions.
  ///
  ///{
  using const_unknown_inst_iterator = UnknownInstructionsTy::const_iterator;
  const_unknown_inst_iterator unknown_reads_begin() const {
    return UnknownReads.begin();
  }
  const_unknown_inst_iterator unknown_reads_end() const { return UnknownReads.end(); }
  const_unknown_inst_iterator unknown_writes_begin() const {
    return UnknownWrites.begin();
  }
  const_unknown_inst_iterator unknown_writes_end() const { return UnknownWrites.end(); }
  ///}

  void print(raw_ostream &OS, PolyhedralValueInfo *PVI = nullptr) const;
  void dump(PolyhedralValueInfo *PVI = nullptr) const;

  bool contains(Instruction *I) const { return Contains(I); }

  void rewrite(PVRewriter<PVMap> &Rewriter);

  ~PACCSummary();

private:

  using ContainsFuncTy = std::function<bool(Instruction *)>;
  PACCSummary(SummaryScopeKind Kind, const ContainsFuncTy &Contains, Loop *Scope = nullptr);

  using PEXPVectorTy = SmallVector<const PEXP *, 8>;
  using InstSetTy = SmallVector<Instruction *, 8>;

  struct MultiDimensionalViewInfo {
    SmallVector<const PEXP *, 4> &DimensionSizes;

    DenseMap<Instruction *, std::pair<unsigned, const PEXP *>> DimensionInstsMap;

    MultiDimensionalViewInfo(SmallVector<const PEXP *, 4> &DimensionSizes)
        : DimensionSizes(DimensionSizes) {}
  };

  const PEXP *findMultidimensionalViewSize(
      PolyhedralValueInfo &PI, ArrayRef<const PEXP *> PEXPs,
      DenseSet<std::pair<Instruction *, const PEXP *>>
          &InstsAndRemainders);

  void findMultidimensionalView(PolyhedralValueInfo &PI,
                                MultiDimensionalViewInfo &MDVI,
                                ArrayRef<const PACC *> PACCs);

  void finalize(PolyhedralValueInfo &PI, ArrayRef<const PACC *> PACCs,
                const DataLayout &DL);

  SummaryScopeKind Kind;
  const ContainsFuncTy &Contains;

  KnownAccessesTy KnownReads;
  KnownAccessesTy KnownWrites;

  UnknownInstructionsTy UnknownReads;
  UnknownInstructionsTy UnknownWrites;

  /// Map containing the information about all accessed arrays in the region.
  ArrayInfoMapTy ArrayInfoMap;

  DenseMap<Value *, SmallVector<const PACC *, 8>> PACCMap;

  Loop *Scope;

  friend class PolyhedralAccessInfo;
};

class PolyhedralAccessInfo {

  /// The PolyhedralValueInfo used to get value information.
  PolyhedralValueInfo &PI;

  LoopInfo &LI;

  PolyhedralExpressionBuilder &PEBuilder;

  DenseMap<Instruction *, PACC *> AccessMap;

  const PACC *getAsAccess(Instruction &Inst, Value &Pointer, bool IsWrite, Loop *Scope = nullptr);

public:
  /// Constructor
  PolyhedralAccessInfo(PolyhedralValueInfo &PI, LoopInfo &LI);

  ~PolyhedralAccessInfo();

  /// Clear all cached information.
  void releaseMemory();

  PolyhedralValueInfo &getPolyhedralValueInfo() { return PI; }

  const PACC *getAsAccess(LoadInst *LI, Loop *Scope = nullptr);
  const PACC *getAsAccess(StoreInst *SI, Loop *Scope = nullptr);
  const PACC *getAsAccess(Instruction *Inst, Loop *Scope = nullptr);

  /// Return an access summary for the blocks in @p Blocks.
  PACCSummary *getAccessSummary(ArrayRef<BasicBlock *> Blocks,
                                PACCSummary::SummaryScopeKind Kind,
                                Loop *Scope = nullptr);

  /// Return an access summary for the function @p F.
  PACCSummary *getAccessSummary(Function &F,
                                PACCSummary::SummaryScopeKind Kind);

  void extractComputations(Function &F);
  void detectKnownComputations(Function &F);

  /// Return true if @p PA represents a value that is fixed for one function
  /// invocation.
  bool hasFunctionScope(const PACC *PE) const;

  /// Return the unknown ids referenced by @p PE  in @p Values.
  void getParameters(const PACC *PE, SmallVectorImpl<PVId> &Values) const;

  /// Return the unknown values referenced by @p PE  in @p Values.
  void getParameters(const PACC *PE, SmallVectorImpl<Value *> &Values) const;

  /// Print some statistics to @p OS.
  void print(raw_ostream &OS) const;
};

/// Wrapper pass for PolyhedralAccessInfo on a per-function basis.
class PolyhedralAccessInfoWrapperPass : public FunctionPass {
  PolyhedralAccessInfo *PAI;
  Function *F;

public:
  static char ID;
  PolyhedralAccessInfoWrapperPass() : FunctionPass(ID) {}

  /// Return the PolyhedralAccessInfo object for the current function.
  PolyhedralAccessInfo &getPolyhedralAccessInfo() {
    assert(PAI);
    return *PAI;
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

class PolyhedralAccessInfoAnalysis
    : public AnalysisInfoMixin<PolyhedralAccessInfoAnalysis> {
  friend AnalysisInfoMixin<PolyhedralAccessInfoAnalysis>;
  static AnalysisKey Key;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef PolyhedralAccessInfo Result;

  /// \brief Run the analysis pass over a function and produce BFI.
  Result run(Function &F, FunctionAnalysisManager &AM);
};

raw_ostream &operator<<(raw_ostream &OS, PACC::AccessKind Kind);
raw_ostream &operator<<(raw_ostream &OS, const PACC *PA);
raw_ostream &operator<<(raw_ostream &OS, const PACC &PA);

} // namespace llvm
#endif
