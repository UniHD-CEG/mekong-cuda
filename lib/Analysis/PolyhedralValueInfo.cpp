//===-- PolyhedralValueInfo.cpp  - Polyhedral value analysis ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

#include "llvm/Analysis/PolyhedralValueInfo.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PolyhedralExpressionBuilder.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/ast.h"
#include "isl/ast_build.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "polyhedral-value-info"

static cl::opt<bool> PVIDisable("pvi-disable", cl::init(false), cl::Hidden,
                                cl::desc("Disable PVI."));

raw_ostream &llvm::operator<<(raw_ostream &OS, PEXP::ExpressionKind Kind) {
  switch (Kind) {
  case PEXP::EK_NONE:
    return OS << "NONE";
  case PEXP::EK_INTEGER:
    return OS << "INTEGER";
  case PEXP::EK_DOMAIN:
    return OS << "DOMAIN";
  case PEXP::EK_UNKNOWN_VALUE:
    return OS << "UNKNOWN";
  case PEXP::EK_NON_AFFINE:
    return OS << "NON AFFINE";
  default:
    llvm_unreachable("Unknown polyhedral expression kind");
  }
}

PEXP *PEXP::setDomain(const PVSet &Domain, bool Overwrite) {
  assert((!PWA || Overwrite) && "PWA already initialized");
  if (Domain.isComplex()) {
    return invalidate();
  }

  setKind(PEXP::EK_DOMAIN);
  PWA = PVAff(Domain, 1);
  PWA.dropUnusedParameters();
  if (!InvalidDomain)
    InvalidDomain = PVSet::empty(PWA);

  if (!KnownDomain)
    KnownDomain = PVSet::universe(PWA);
  else
    PWA.simplify(KnownDomain);

  // Sanity check
  assert(KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());

  return this;
}

void PEXP::print(raw_ostream &OS) const {
  OS << PWA << " [" << (getValue() ? getValue()->getName() : "<none>") << "] ["
     << getKind()
     << "] [Scope: " << (getScope() ? getScope()->getName() : "<max>") << "]";
  if (!InvalidDomain.isEmpty())
    OS << " [ID: " << InvalidDomain << "]";
  if (!KnownDomain.isUniverse())
    OS << " [KD: " << KnownDomain << "]";
}
void PEXP::dump() const { print(dbgs()); }

raw_ostream &llvm::operator<<(raw_ostream &OS, const PEXP *PE) {
  if (PE)
    OS << *PE;
  else
    OS << "<null>";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PEXP &PE) {
  PE.print(OS);
  return OS;
}

PEXP &PEXP::operator=(const PEXP &PE) {
  Kind = PE.Kind;
  PWA = PE.getPWA();
  InvalidDomain = PE.getInvalidDomain();
  KnownDomain = PE.getKnownDomain();

  // Sanity check
  assert(!KnownDomain || KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(!InvalidDomain || InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());

  return *this;
}

PEXP &PEXP::operator=(PEXP &&PE) {
  std::swap(Kind, PE.Kind);
  std::swap(PWA, PE.PWA);
  std::swap(InvalidDomain, PE.InvalidDomain);
  std::swap(KnownDomain, PE.KnownDomain);

  // Sanity check
  assert(!KnownDomain ||
         KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(!InvalidDomain ||
         InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());

  return *this;
}

void PEXP::addInvalidDomain(const PVSet &ID) {
  InvalidDomain.unify(ID);
  if (InvalidDomain.isUniverse()) {
    invalidate();
  }
  if (InvalidDomain.isComplex()) {
    invalidate();
  }

  // Sanity check
  assert(!KnownDomain || !PWA ||
         KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  //assert(!InvalidDomain || !PWA ||
         //InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  //assert(!KnownDomain || !InvalidDomain ||
         //KnownDomain.getNumInputDimensions() ==
             //InvalidDomain.getNumInputDimensions());
}

void PEXP::addKnownDomain(const PVSet &KD) {
  KnownDomain.intersect(KD);
  if (KnownDomain.isComplex()) {
    KnownDomain = PVSet::universe(KnownDomain);
  }
  PWA.simplify(KnownDomain);
  InvalidDomain.simplify(KD);

  // Sanity check
  assert(!KnownDomain || !PWA ||
         KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(!InvalidDomain || !PWA ||
         InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(!KnownDomain || !InvalidDomain ||
         KnownDomain.getNumInputDimensions() ==
             InvalidDomain.getNumInputDimensions());
}

PEXP *PEXP::invalidate() {
  Kind = PEXP::EK_NON_AFFINE;
  PWA = PVAff();
  return this;
}

void PEXP::adjustInvalidAndKnownDomain() {
  auto *ITy = cast<IntegerType>(getValue()->getType());
  unsigned BitWidth = ITy->getBitWidth();
  assert(BitWidth > 0 && BitWidth <= 64);
  int64_t LowerBound = -1 * (1 << (BitWidth - 1));
  int64_t UpperBound = (1 << (BitWidth - 1)) - 1;

  PVAff LowerPWA(getDomain(), LowerBound);
  PVAff UpperPWA(getDomain(), UpperBound);

  auto *OVBinOp = cast<OverflowingBinaryOperator>(getValue());
  bool HasNSW = OVBinOp->hasNoSignedWrap();

  const PVAff &PWA = getPWA();
  if (HasNSW) {
    PVSet BoundedDomain = PWA.getGreaterEqualDomain(LowerPWA).intersect(
        PWA.getLessEqualDomain(UpperPWA));

    KnownDomain.intersect(BoundedDomain);
  } else {
    PVSet BoundedDomain = LowerPWA.getGreaterEqualDomain(PWA).unify(
        UpperPWA.getLessEqualDomain(PWA));

    InvalidDomain.unify(BoundedDomain);
  }

  // Sanity check
  assert(!KnownDomain || KnownDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
  assert(!InvalidDomain || InvalidDomain.getNumInputDimensions() == PWA.getNumInputDimensions());
}

// ------------------------------------------------------------------------- //

PolyhedralValueInfo::PolyhedralValueInfo(PVCtx Ctx, LoopInfo &LI)
    : Ctx(Ctx), LI(LI), PEBuilder(new PolyhedralExpressionBuilder(*this)) {
}

PolyhedralValueInfo::~PolyhedralValueInfo() { delete PEBuilder; }

const PEXP *PolyhedralValueInfo::getPEXP(Value *V, Loop *Scope, bool Strict,
                                         bool NoAlias) const {
  PEBuilder->setScope(Scope);
  PEXP *PE = PEBuilder->visit(*V);
  if (Strict && !hasScope(PE, Scope, Strict, NoAlias))
    return nullptr;
  return PE;
}

const PEXP *PolyhedralValueInfo::getDomainFor(BasicBlock *BB, Loop *Scope,
                                              bool Strict, bool NoAlias) const {
  PEBuilder->setScope(Scope);
  PEXP *PE = PEBuilder->getDomain(*BB);
  if (Strict && !hasScope(PE, Scope, Strict, NoAlias))
    return nullptr;
  return PE;
}

const PEXP *PolyhedralValueInfo::getBackedgeTakenCount(const Loop &L,
                                                       Loop *Scope, bool Strict,
                                                       bool NoAlias) const {
  if (PVIDisable)
    return nullptr;
  PEBuilder->setScope(Scope);
  PEXP *PE = PEBuilder->getBackedgeTakenCount(L);
  if (Strict && !hasScope(PE, Scope, Strict, NoAlias))
    return nullptr;
  return PE;
}

PVId PolyhedralValueInfo::getParameterId(Value &V) const {
  return PEBuilder->getParameterId(V);
}

bool PolyhedralValueInfo::isUnknown(const PEXP *PE) const {
  return PE->Kind == PEXP::EK_UNKNOWN_VALUE;
}

bool PolyhedralValueInfo::isInteger(const PEXP *PE) const {
  return PE->Kind == PEXP::EK_INTEGER;
}

bool PolyhedralValueInfo::isConstant(const PEXP *PE) const {
  return isInteger(PE) && PE->PWA.getNumPieces() == 1;
}

bool PolyhedralValueInfo::isAffine(const PEXP *PE) const {
  return PE->Kind != PEXP::EK_NON_AFFINE && PE->isInitialized();
}

bool PolyhedralValueInfo::isNonAffine(const PEXP *PE) const {
  return PE->Kind == PEXP::EK_NON_AFFINE;
}

bool PolyhedralValueInfo::isVaryingInScope(Instruction &I,
                                           const Region &RegionScope,
                                           bool Strict, bool NoAlias) const {
  if (!RegionScope.contains(&I))
    return false;
  if (Strict)
    return true;
  if (I.mayReadFromMemory()) {
    if (!NoAlias)
      return true;
    if (!isa<LoadInst>(I))
      return true;
    Value *Ptr = cast<LoadInst>(&I)->getPointerOperand();
    if (!isa<Instruction>(Ptr))
      return false;
    return isVaryingInScope(*cast<Instruction>(Ptr), RegionScope, Strict, NoAlias);
  }

  Loop *L = nullptr;
  if (auto *PHI = dyn_cast<PHINode>(&I)) {
    L = LI.isLoopHeader(PHI->getParent()) ? LI.getLoopFor(PHI->getParent()) : L;
    if (L)
      return RegionScope.contains(L);
  }

  for (Value *Op : I.operands())
    if (auto *OpI = dyn_cast<Instruction>(Op)) {
      if (L && L->contains(OpI))
        continue;
      if (isVaryingInScope(*OpI, RegionScope, Strict, NoAlias))
        return true;
    }
  return false;
}

bool PolyhedralValueInfo::isVaryingInScope(Instruction &I, Loop *Scope,
                                           bool Strict, bool NoAlias) const {
  if (Scope && !Scope->contains(&I))
    return false;
  if (Strict)
    return true;
  Loop *L = LI.getLoopFor(I.getParent());
  if (L == Scope)
    return false;
  if (I.mayReadFromMemory()) {
    if (!NoAlias)
      return true;
    if (!isa<LoadInst>(I))
      return true;
    Value *Ptr = cast<LoadInst>(&I)->getPointerOperand();
    if (!isa<Instruction>(Ptr))
      return false;
    return isVaryingInScope(*cast<Instruction>(Ptr), Scope, Strict, NoAlias);
  }

  if (auto *PHI = dyn_cast<PHINode>(&I)) {
    if (Scope && PHI->getParent() == Scope->getHeader())
      return false;
    L = LI.isLoopHeader(PHI->getParent()) ? LI.getLoopFor(PHI->getParent()) : L;
    if (L && (!Scope || Scope->contains(L)))
      return true;
  }

  for (Value *Op : I.operands())
    if (auto *OpI = dyn_cast<Instruction>(Op)) {
      if (L && L->contains(OpI))
        continue;
      if (isVaryingInScope(*OpI, Scope, Strict, NoAlias))
        return true;
    }
  return false;
}

bool PolyhedralValueInfo::hasScope(Value &V, const Region &RegionScope,
                                   bool Strict, bool NoAlias) const {
  auto *I = dyn_cast<Instruction>(&V);
  if (!I || !isVaryingInScope(*I, RegionScope, Strict, NoAlias))
    return true;

  return false;
}

bool PolyhedralValueInfo::hasScope(const PEXP *PE, const Region &RegionScope,
                                   bool Strict, bool NoAlias) const {

  SmallVector<Value *, 4> Values;
  getParameters(PE, Values);
  for (Value *V : Values)
    if (!hasScope(*V, RegionScope, Strict, NoAlias))
      return false;
  return true;
}

bool PolyhedralValueInfo::hasScope(Value &V, Loop *Scope,
                                   bool Strict, bool NoAlias) const {
  auto *I = dyn_cast<Instruction>(&V);
  if (!I || !isVaryingInScope(*I, Scope, Strict, NoAlias))
    return true;

  return false;
}

bool PolyhedralValueInfo::hasScope(const PEXP *PE, Loop *Scope,
                                   bool Strict, bool NoAlias) const {

  SmallVector<Value *, 4> Values;
  getParameters(PE, Values);
  for (Value *V : Values)
    if (!hasScope(*V, Scope, Strict, NoAlias))
      return false;
  return true;
}

unsigned PolyhedralValueInfo::getNumPieces(const PEXP *PE) const {
  return PE->getPWA().getNumPieces();
}

bool PolyhedralValueInfo::isAlwaysValid(const PEXP *PE) const {
  return PE->getInvalidDomain().isEmpty();
}

bool PolyhedralValueInfo::mayBeInfinite(Loop &L) const {
  if (PVIDisable)
    return true;
  const PEXP *HeaderBBPE = getDomainFor(L.getHeader());
  if (!isAffine(HeaderBBPE))
    return true;

  assert(HeaderBBPE->getDomain().isBounded());

  const PVSet &InvDom = HeaderBBPE->getInvalidDomain();
  return !InvDom.isBounded();
}

void PolyhedralValueInfo::getParameters(const PEXP *PE,
                                        SmallVectorImpl<PVId> &Values,
                                        bool Recursive) const {
  unsigned u = Values.size();
  const PVAff &PWA = PE->getPWA();
  PWA.getParameters(Values);
  if (!Recursive)
    return;

  unsigned e = Values.size();
  assert(u <= e);
  Loop *Scope = PE->getScope();
  for (; u < e; u++) {
    Instruction *I = dyn_cast<Instruction>(Values[u].getPayloadAs<Value *>());
    if (!I || (Scope && !Scope->contains(I)))
      continue;
    if (I == PE->getValue())
      continue;
    // TODO PHIS
    if (isa<PHINode>(I) && LI.isLoopHeader(I->getParent()))
      continue;
    getParameters(getPEXP(I, Scope, false), Values);
  }
}

void PolyhedralValueInfo::getParameters(const PEXP *PE,
                                        SmallVectorImpl<Value *> &Values,
                                        bool Recursive) const {
  unsigned u = Values.size();
  const PVAff &PWA = PE->getPWA();
  PWA.getParameters(Values);
  if (!Recursive)
    return;

  unsigned e = Values.size();
  assert(u <= e);
  Loop *Scope = PE->getScope();
  for (; u < e; u++) {
    Instruction *I = dyn_cast<Instruction>(Values[u]);
    if (!I || (Scope && !Scope->contains(I)))
      continue;
    if (I == PE->getValue())
      continue;
    // TODO PHIS
    if (isa<PHINode>(I) && LI.isLoopHeader(I->getParent()))
      continue;
    getParameters(getPEXP(I, Scope, false), Values);
  }
}

bool PolyhedralValueInfo::isKnownToHold(Value *LHS, Value *RHS,
                                        ICmpInst::Predicate Pred,
                                        Instruction *IP, Loop *Scope) {
  const PEXP *LHSPE = getPEXP(LHS, Scope);
  if (isNonAffine(LHSPE))
    return false;

  const PEXP *RHSPE = getPEXP(RHS, Scope);
  if (isNonAffine(RHSPE))
    return false;

  const PEXP *IPDomPE = IP ? getDomainFor(IP->getParent(), Scope) : nullptr;
  if (IP && (isNonAffine(IPDomPE) || !IPDomPE->getInvalidDomain().isEmpty()))
    return false;

  PVSet LHSInvDom = LHSPE->getInvalidDomain();
  PVSet RHSInvDom = RHSPE->getInvalidDomain();
  if (IPDomPE) {
    LHSInvDom.intersect(IPDomPE->getDomain());
    RHSInvDom.intersect(IPDomPE->getDomain());
  }

  if (!LHSInvDom.isEmpty() || !RHSInvDom.isEmpty())
    return false;

  PVAff LHSAff = LHSPE->getPWA();
  PVAff RHSAff = RHSPE->getPWA();

  if (IPDomPE) {
    LHSAff.intersectDomain(IPDomPE->getDomain());
    RHSAff.intersectDomain(IPDomPE->getDomain());
  }

  auto FalseDomain = PVAff::buildConditionSet(
      ICmpInst::getInversePredicate(Pred), LHSAff, RHSAff);
  return FalseDomain.isEmpty();
}

void PolyhedralValueInfo::print(raw_ostream &OS) const {
  auto &PVIC = PEBuilder->getPolyhedralValueInfoCache();
  for (auto &It : PVIC.domains()) {
    Loop *L = It.first.second;
    OS << "V: " << It.first.first->getName() << " in "
       << (L ? L->getName() : "<max>") << ":\n\t" << It.second << "\n";
  }
  for (auto &It : PVIC) {
    Loop *L = It.first.second;
    OS << "V: " << *It.first.first << " in " << (L ? L->getName() : "<max>")
       << ":\n\t" << It.second << "\n";
  }


  PVSet DomS, DomT;
  for (BasicBlock &BB : *(*LI.begin())->getHeader()->getParent()) {
    if (BB.getName() == "S")
      DomS = getDomainFor(&BB)->getDomain();
    if (BB.getName() == "T")
      DomT = getDomainFor(&BB)->getDomain();
  }

  OS<< "\n\nDomS: " << DomS << "\nDomT: " << DomT <<"\n";
  DomT = DomT.dropLastInputDims(1);
  OS<< "DomT: " << DomT <<"\n";
  PVSet R = DomS;
  R.subtract(DomT);
  OS << "R: " << R << " DomS: " << DomS << "\n";
  R.dropUnusedParameters();
  DomS.dropUnusedParameters();
  R = R.simplify(DomS);
  OS << "R: "<<R<<"\n";

  auto *ASTB = isl_ast_build_from_context(isl_set_params(DomS.getObj()));
  auto *RExp = isl_ast_build_expr_from_set(ASTB, isl_set_params(R.getObj()));
  isl_ast_expr_dump(RExp);
}

// ------------------------------------------------------------------------- //

char PolyhedralValueInfoWrapperPass::ID = 0;

void PolyhedralValueInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
}

void PolyhedralValueInfoWrapperPass::releaseMemory() {
  //F = nullptr;
  //delete PI;
  //PI = nullptr;
}

bool PolyhedralValueInfoWrapperPass::runOnFunction(Function &F) {

  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  delete PI;
  PI = new PolyhedralValueInfo(Ctx, LI);

  this->F = &F;

  return false;
}

void PolyhedralValueInfoWrapperPass::print(raw_ostream &OS,
                                           const Module *) const {
  PI->print(OS);

  if (!F)
    return;

  PolyhedralValueInfoWrapperPass &PIWP =
      *const_cast<PolyhedralValueInfoWrapperPass *>(this);
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  SmallVector<Loop *, 8> LastLoops;
  for (auto &BB : *PIWP.F) {
    LastLoops.clear();
    Loop *Scope = LI.getLoopFor(&BB);
    do {
      PIWP.PI->getDomainFor(&BB, Scope);
      for (auto &Inst : BB)
        if (!Inst.getType()->isVoidTy())
          PIWP.PI->getPEXP(&Inst, Scope);
      if (!Scope)
        break;
      LastLoops.push_back(Scope);
      Scope = Scope->getParentLoop();
      for (Loop *L : LastLoops)
        PIWP.PI->getBackedgeTakenCount(*L, Scope);
    } while (true);
  }

  for (Loop *L : LI.getLoopsInPreorder()) {
    Loop *Scope = L->getParentLoop();
    do {
      OS << "Scope: " << (Scope ? Scope->getName() : "<none>") << "\n";
      const PEXP *PE = PIWP.PI->getBackedgeTakenCount(*L, Scope);
      OS << "back edge taken count of " << L->getName() << "\n";
      OS << "\t => " << PE << "\n";
      if (!Scope)
        break;
      Scope = Scope->getParentLoop();
    } while (true);
  }

  for (auto &BB : *PIWP.F) {
    Loop *Scope, *L;
    Scope = L = LI.getLoopFor(&BB);

    do {
      const PEXP *PE = PIWP.PI->getDomainFor(&BB, Scope);
      OS << "Domain of " << BB.getName() << ":\n";
      OS << "\t => " << PE << "\n";
      for (auto &Inst : BB) {
        if (Inst.getType()->isVoidTy()) {
          OS << "\tValue of " << Inst << ":\n";
          OS << "\t\t => void type!\n";
          continue;
        }
        const PEXP *PE = PIWP.PI->getPEXP(&Inst, Scope);
        OS << "\tValue of " << Inst << ":\n";
        OS << "\t\t => " << PE << "\n";
        SmallVector<Value *, 4> Values;
        PIWP.PI->getParameters(PE, Values);
        if (Values.empty())
          continue;
        OS << "\t\t\tParams:\n";
        for (Value *Val : Values)
          OS << "\t\t\t - " << *Val << "\n";
      }

      if (!Scope)
        break;
      Scope = Scope->getParentLoop();
    } while (true);
  }
}

FunctionPass *llvm::createPolyhedralValueInfoWrapperPass() {
  return new PolyhedralValueInfoWrapperPass();
}

void PolyhedralValueInfoWrapperPass::dump() const {
  return print(dbgs(), nullptr);
}

AnalysisKey PolyhedralValueInfoAnalysis::Key;

PolyhedralValueInfo
PolyhedralValueInfoAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  return PolyhedralValueInfo(Ctx, LI);
}

INITIALIZE_PASS_BEGIN(PolyhedralValueInfoWrapperPass, "polyhedral-value-info",
                      "Polyhedral value analysis", false, true);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_END(PolyhedralValueInfoWrapperPass, "polyhedral-value-info",
                    "Polyhedral value analysis", false, true)
