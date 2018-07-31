//===--- PolyhedralExpressionBuilder.h -- Builder for PEXPs -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLYHEDRAL_EXPRESSION_BUILDER_H
#define POLYHEDRAL_EXPRESSION_BUILDER_H

#include "llvm/Analysis/PolyhedralValueInfo.h"
#include "llvm/IR/InstVisitor.h"

#include <map>

namespace llvm {

/// A cache that maps values/basic blocks and scopes to the computed polyhedral
/// representation (PEXP). The results for maximal scopes (nullptr) can be used
/// in an inter-procedural setting.
class PolyhedralValueInfoCache final {

  /// Mapping from scoped basic blocks to their domain expressed as a PEXP.
  using DomainMapKey = std::tuple<BasicBlock *, Loop *>;
  std::map<DomainMapKey, PEXP *> DomainMap;

  /// Mapping from scoped values to their polyhedral representation.
  using ValueMapKey = std::tuple<Value *, Loop *>;
  std::map<ValueMapKey, PEXP *> ValueMap;

  /// Mapping from scoped loops to their backedge taken count.
  using LoopMapKey = std::pair<const Loop *, Loop *>;
  DenseMap<LoopMapKey, PEXP *> LoopMap;

  /// Mapping from parameter values to their unique id.
  DenseMap<Value *, PVId> ParameterMap;

  /// Return or create and cache a PEXP for @p BB in @p Scope.
  PEXP *getOrCreateDomain(BasicBlock &BB, Loop *Scope) {
    auto *&PE = DomainMap[{&BB, Scope}];
    if (!PE)
      PE = new PEXP(&BB, Scope);

    // Verify the internal state
    assert(PE == lookup(BB, Scope));
    return PE;
  }

  /// Return or create and cache a PEXP for @p V in @p Scope.
  PEXP *getOrCreatePEXP(Value &V, Loop *Scope) {
    auto *&PE = ValueMap[{&V, Scope}];
    if (!PE)
      PE = new PEXP(&V, Scope);

    // Verify the internal state
    assert(PE == lookup(V, Scope));
    return PE;
  }

  /// Create or return a PEXP for the backedge taken count of @p L in @p Scope.
  PEXP *getOrCreateBackedgeTakenCount(const Loop &L, Loop *Scope) {
    auto *&PE = LoopMap[{&L, Scope}];
    if (!PE)
      PE = new PEXP(L.getHeader(), Scope);

    // Verify the internal state
    assert(PE == lookup(L, Scope));
    return PE;
  }

  std::string getParameterNameForValue(Value &V);

  /// Return the unique parameter id for @p V.
  PVId getParameterId(Value &V, const PVCtx &Ctx);

  friend class PolyhedralExpressionBuilder;

public:
  ~PolyhedralValueInfoCache();

  /// Return the cached polyhedral representation of @p V in @p Scope, if any.
  PEXP *lookup(Value &V, Loop *Scope) { return ValueMap[{&V, Scope}]; }

  /// Return the cached polyhedral representation of @p BB in @p Scope, if any.
  PEXP *lookup(BasicBlock &BB, Loop *Scope) { return DomainMap[{&BB, Scope}]; }

  /// Return the cached backedge taken count of @p L in @p Scope, if any.
  PEXP *lookup(const Loop &L, Loop *Scope) {
    return LoopMap.lookup({&L, Scope});
  }

  /// Forget the value for @p BB in @p Scope. Returns true if there was one.
  bool forget(BasicBlock &BB, Loop *Scope) {
    return DomainMap.erase({&BB, Scope});
  }

  /// Forget the value for @p V in @p Scope. Returns true if there was one.
  bool forget(Value &V, Loop *Scope) {
    return ValueMap.erase({&V, Scope});
  }

  /// Iterators for polyhedral representation of values.
  ///{
  using iterator = decltype(ValueMap)::iterator;
  using const_iterator = decltype(ValueMap)::const_iterator;

  iterator begin() { return ValueMap.begin(); }
  iterator end() { return ValueMap.end(); }
  const_iterator begin() const { return ValueMap.begin(); }
  const_iterator end() const { return ValueMap.end(); }

  iterator_range<iterator> values() { return make_range(begin(), end()); }
  iterator_range<const_iterator> values() const {
    return make_range(begin(), end());
  }
  ///}

  /// Iterators for polyhedral domains of basic block.
  ///{
  using domain_iterator = decltype(DomainMap)::iterator;
  using const_domain_iterator = decltype(DomainMap)::const_iterator;

  domain_iterator domain_begin() { return DomainMap.begin(); }
  domain_iterator domain_end() { return DomainMap.end(); }
  const_domain_iterator domain_begin() const { return DomainMap.begin(); }
  const_domain_iterator domain_end() const { return DomainMap.end(); }

  iterator_range<domain_iterator> domains() {
    return make_range(domain_begin(), domain_end());
  }
  iterator_range<const_domain_iterator> domains() const {
    return make_range(domain_begin(), domain_end());
  }
  ///}
};


class PolyhedralExpressionBuilder
    : public InstVisitor<PolyhedralExpressionBuilder, PEXP *> {

  Loop *Scope;
  Loop *UseLocation;

  PolyhedralValueInfo &PI;
  PolyhedralValueInfoCache PIC;

  PEXP *visit(Constant &I);
  PEXP *visit(ConstantInt &I);
  PEXP *visitParameter(Value &V);
  PEXP *createParameter(PEXP *PE);

  PVAff getZero(const PVAff &RefPWA) { return PVAff(RefPWA.getDomain(), 0); }

  PVAff getOne(const PVAff &RefPWA) { return PVAff(RefPWA.getDomain(), 1); }

  PEXP *getOrCreatePEXP(Value &V) {
    PEXP *PE = PIC.getOrCreatePEXP(V, Scope);
    assert(PE && PIC.lookup(V, Scope));
    assert(PE->getScope() == Scope && PE->getValue() == &V);

    // Initialize the PEXP for outer scopes as well.
    auto *I = dyn_cast<Instruction>(&V);
    Loop *OuterScope = Scope ? Scope->getParentLoop() : nullptr;
    while (!I || (OuterScope && !OuterScope->contains(I))) {
      PEXP *OuterPE = PIC.getOrCreatePEXP(V, OuterScope);

      // If we found an initialized value we will not find an uninitialized one
      // later.
      if (OuterPE->isInitialized())
        break;

      // Assign the PEXP for the outer scope.
      *OuterPE = *PE;

      // Widen the scope if possible, otherwise we are done.
      if (!OuterScope)
        break;

      OuterScope = OuterScope->getParentLoop();
    }

    return PE;
  }

  PEXP *getOrCreateDomain(BasicBlock &BB) {
    PEXP *PE = PIC.getOrCreateDomain(BB, Scope);
    assert(PE && PIC.lookup(BB, Scope));
    assert(PE->getScope() == Scope && PE->getValue() == &BB);

    // Initialize the PEXP for this block for other scopes as well.
    Loop *OuterScope = Scope ? Scope->getParentLoop() : nullptr;
    while (OuterScope && OuterScope->contains(&BB)) {
      PEXP *OuterPE = PIC.getOrCreateDomain(BB, OuterScope);

      // If we found an initialized value we will not find an uninitialized one
      // later.
      if (OuterPE->isInitialized())
        break;

      // Assign the PEXP for the outer scope.
      *OuterPE = *PE;

      // Widen the scope if possible, otherwise we are done.
      if (!OuterScope)
        break;

      OuterScope = OuterScope->getParentLoop();
    }

    return PE;
  }

  /// Combine the invalid and known domain of @p Other into @p PE.
  ///
  /// @param PE    The polyhedral representation to combine into.
  /// @param Other The polyhedral representation to combine.
  ///
  /// @return True, if the result is affine, false otherwise.
  bool combine(PEXP *PE, const PEXP *Other);

  /// Combine the polyhedral representation @p PE restricted to the domain
  /// @p Domain into this one using @p Combinator as a combinator function.
  ///
  /// @param PE         The polyhedral representation to combine into.
  /// @param Other      The polyhedral representation to combine.
  /// @param Combinator The combinator function.
  /// @param Domain     The domain for which @p PEXP will be combined.
  ///
  /// @return True, if the result is affine, false otherwise.
  bool combine(PEXP *PE, const PEXP *Other, PVAff::CombinatorFn Combinator,
                const PVSet *Domain = nullptr);
  bool combine(PEXP *PE, const PEXP *Other, PVAff::IslCombinatorFn Combinator,
                const PVSet *Domain = nullptr);

  PVSet createParameterRanges(const PVSet &S, const DataLayout &DL);

  unsigned getRelativeLoopDepth(Loop *L);
  unsigned getRelativeLoopDepth(BasicBlock *BB);

  Loop *getLoopForPE(const PEXP *PE);

  template <typename PVTy>
  bool  adjustDomainDimensions(PVTy &Obj, const PEXP *OldPE, const PEXP *NewPE,
                              bool LastIt = false);
  template <typename PVTy>
  bool adjustDomainDimensions(PVTy &Obj, Loop *OldL, Loop *NewL,
                              bool LastIt = false);

public:
  PolyhedralExpressionBuilder(PolyhedralValueInfo &PI)
      : Scope(nullptr), PI(PI) {}

  PolyhedralValueInfoCache& getPolyhedralValueInfoCache() { return PIC; }
  const PolyhedralValueInfoCache &getPolyhedralValueInfoCache() const {
    return PIC;
  }

  /// Assign the combination of @p LHS and @p RHS using @p Combinator to @p PE.
  ///
  /// @return True, if the result is affine, false otherwise.
  bool assign(PEXP *PE, const PEXP *LHS, const PEXP *RHS,
               PVAff::CombinatorFn Combinator);
  bool assign(PEXP *PE, const PEXP *LHS, const PEXP *RHS,
               PVAff::IslCombinatorFn Combinator);

  PVSet buildNotEqualDomain(const PEXP *VI, ArrayRef<Constant *> CIs);
  PVSet buildEqualDomain(const PEXP *VI, Constant &CI);

  void setScope(Loop *NewScope) { Scope = NewScope; }

  PEXP *getDomain(BasicBlock &BB);

  PEXP *getBackedgeTakenCount(const Loop &L);

  PVId getParameterId(Value &V) { return PIC.getParameterId(V, PI.getCtx()); }

  PEXP *getTerminatorPEXP(BasicBlock &BB);

  bool getDomainOnEdge(PVSet &DomainOnEdge, const PEXP &PredDomPE,
                       BasicBlock &BB);
  bool getEdgeCondition(PVSet &EdgeCondition, BasicBlock &PredBB,
                        BasicBlock &BB);

  PEXP *visitOperand(Value &Op, Instruction &I);
  PEXP *visit(Value &V);
  PEXP *visit(Instruction &I);

  PEXP *visitBinaryOperator(BinaryOperator &I);
  PEXP *visitCallInst(CallInst &I);
  PEXP *visitCastInst(CastInst &I);
  PEXP *visitFCmpInst(FCmpInst &I);
  PEXP *visitGetElementPtrInst(GetElementPtrInst &I);
  PEXP *visitICmpInst(ICmpInst &I);
  PEXP *visitInvokeInst(InvokeInst &I);
  PEXP *visitLoadInst(LoadInst &I);
  PEXP *visitSelectInst(SelectInst &I);
  PEXP *visitConditionalPHINode(PHINode &I);
  PEXP *visitPHINode(PHINode &I);
  PEXP *visitAllocaInst(AllocaInst &I);

  PEXP *visitInstruction(Instruction &I);
};


} // namespace llvm
#endif
