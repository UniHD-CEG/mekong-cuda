//===-- PolyhedralExpressionBuilder.cpp  - Builder for PEXPs ----*- C++ -*-===//
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

#include "llvm/Analysis/PolyhedralExpressionBuilder.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/PolyhedralValueInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "polyhedral-expression-builder"

STATISTIC(NUM_PARAMETERS, "Number of parameters created");
STATISTIC(NUM_DOMAINS, "Number of domains created");
STATISTIC(NUM_EXPRESSIONS, "Number of expressions created");
STATISTIC(COMPLEX_DOMAIN, "Number of domains to complex");

// ------------------------------------------------------------------------- //

PolyhedralValueInfoCache::~PolyhedralValueInfoCache() {
  DeleteContainerSeconds(LoopMap);
  DeleteContainerSeconds(ValueMap);
  DeleteContainerSeconds(DomainMap);
  ParameterMap.clear();
}

std::string PolyhedralValueInfoCache::getParameterNameForValue(Value &V) {
  if (V.hasName())
    return V.getName().str();
  return "p" + std::to_string(ParameterMap.size());
}

PVId PolyhedralValueInfoCache::getParameterId(Value &V, const PVCtx &Ctx) {
  PVId &Id = ParameterMap[&V];
  if (Id)
    return Id;

  std::string ParameterName = getParameterNameForValue(V);
  ParameterName = PVBase::getIslCompatibleName("", ParameterName, "");
  Id = PVId(Ctx, ParameterName, &V);

  return Id;
}

// ------------------------------------------------------------------------- //


bool PolyhedralExpressionBuilder::combine(PEXP *PE, const PEXP *Other) {
  PE->Kind = std::max(PE->Kind, Other->Kind);
  if (PE->Kind == PEXP::EK_NON_AFFINE) {
    PE->invalidate();
    return false;
  }

  PVSet OtherID = Other->getInvalidDomain();
  if (OtherID) {
    if (!adjustDomainDimensions(OtherID, Other, PE)) {
      PE->invalidate();
      return false;
    }
    PE->addInvalidDomain(OtherID);
  }

  PVSet OtherKD = Other->getKnownDomain();
  if (OtherKD) {
    if (!adjustDomainDimensions(OtherKD, Other, PE)) {
      PE->invalidate();
      return false;
    }
    PE->addKnownDomain(OtherKD);
  }

  return PE->getKind() != PEXP::EK_NON_AFFINE;
}

bool PolyhedralExpressionBuilder::combine(PEXP *PE, const PEXP *Other,
                                          PVAff::IslCombinatorFn Combinator,
                                          const PVSet *Domain) {
  return combine(PE, Other, PVAff::getCombinatorFn(Combinator), Domain);
}

bool PolyhedralExpressionBuilder::combine(PEXP *PE, const PEXP *Other,
                                          PVAff::CombinatorFn Combinator,
                                          const PVSet *Domain) {
  assert(PE->isInitialized() && Other && Other->isInitialized() && Combinator &&
         "Can only combine initialized polyhedral expressions");

  if (!combine(PE, Other))
    return false;

  PVAff OtherPWA = Other->getPWA();
  if (Domain)
    OtherPWA.intersectDomain(*Domain);

  PE->PWA = PE->PWA ? Combinator(PE->PWA, OtherPWA) : OtherPWA;

  return true;
}

bool PolyhedralExpressionBuilder::assign(PEXP *PE, const PEXP *LHSPE,
                                          const PEXP *RHSPE,
                                          PVAff::IslCombinatorFn Combinator) {
  return assign(PE, LHSPE, RHSPE, PVAff::getCombinatorFn(Combinator));
}

bool PolyhedralExpressionBuilder::assign(PEXP *PE, const PEXP *LHSPE,
                                          const PEXP *RHSPE,
                                          PVAff::CombinatorFn Combinator) {
  assert(!PE->isInitialized() &&
         "Cannot assign to an initialized polyhedral expression");
  assert(LHSPE && LHSPE->isInitialized() && RHSPE && RHSPE->isInitialized() &&
         "Cannot assign from an uninitialized polyhedral expression");

  if (!combine(PE, LHSPE))
    return false;
  if (!combine(PE, RHSPE))
    return false;

  auto &PWA0 = LHSPE->getPWA();
  auto &PWA1 = RHSPE->getPWA();
  PE->PWA = Combinator(PWA0, PWA1);

  // Sanity test.
  unsigned LoopDims = getRelativeLoopDepth(getLoopForPE(PE));
  unsigned NumDims = PE->PWA.getNumInputDimensions();
  assert(LoopDims >= NumDims);

  PE->PWA.addInputDims(LoopDims - NumDims);

  return PE->getKind() != PEXP::EK_NON_AFFINE;
}

PEXP *PolyhedralExpressionBuilder::getBackedgeTakenCount(const Loop &L) {
  assert(&L != Scope);

  PEXP *PE = PIC.getOrCreateBackedgeTakenCount(L, Scope);
  if (PE->isInitialized())
    return PE;

  BasicBlock *HeaderBB = L.getHeader();

  const PEXP *HeaderBBPE = getDomain(*HeaderBB);

  // TODO: Allow (and skip) non-affine latch domains for under-approximations,
  // thus a minimal trip count.
  if (!PI.isAffine(HeaderBBPE)) {
    return PE->invalidate();
  }

  assert(HeaderBBPE->getPWA().getNumInputDimensions() > 0);

  // TODO: Allow latch domains that do not have the proper scope for
  // both under and over-approximations, thus a minimal and maximal trip
  // count.
  if (!PI.hasScope(HeaderBBPE, Scope, false)) {
    return PE->invalidate();
  }

  PVSet HeaderBBDom = HeaderBBPE->getDomain();
  if (!HeaderBBDom.isBounded()) {
    return PE->invalidate();
  }

  combine(PE, HeaderBBPE);

  if (HeaderBBDom.isEmpty())
    PE->PWA = PVAff(PVSet::universe(HeaderBBDom), 0);
  else
    PE->PWA = PVAff::getBackEdgeTakenCountFromDomain(HeaderBBDom);

  PE->PWA.simplify(HeaderBBDom);
  PE->setKind(PE->PWA.isInteger() ? PEXP::EK_INTEGER : PEXP::EK_UNKNOWN_VALUE);

  return PE;
}

PVSet PolyhedralExpressionBuilder::createParameterRanges(const PVSet &S,
                                                         const DataLayout &DL) {
  PVSet ParameterRanges = PVSet::universe(S);

  SmallVector<PVId, 4> Parameters;
  S.getParameters(Parameters);
  for (const PVId &PId : Parameters) {
    auto *Parameter = PId.getPayloadAs<Value *>();

    // Treat i1 types as unsigned even if the rest is assumed to be signed.
    if (Parameter->getType()->isIntegerTy(1)) {
      ParameterRanges.intersect(PVSet::createParameterRange(PId, 0, 1));
      continue;
    }

    // Do not add constaints involving big constants (> 2^7).
    unsigned TypeWidth = DL.getTypeSizeInBits(Parameter->getType());
    if (TypeWidth > 8 || TypeWidth < 2)
      continue;

    int ExpVal = ((int)1) << (TypeWidth - 1);
    ParameterRanges.intersect(
        PVSet::createParameterRange(PId, -ExpVal, ExpVal - 1));
  }

  return ParameterRanges;
}

bool PolyhedralExpressionBuilder::getEdgeCondition(PVSet &EdgeCondition,
                                                   BasicBlock &PredBB,
                                                   BasicBlock &BB) {
  unsigned PredLD = getRelativeLoopDepth(&PredBB);
  EdgeCondition = PVSet::universe(PI.getCtx());
  EdgeCondition.addInputDims(PredLD);

  auto &TI = *PredBB.getTerminator();
  if (TI.getNumSuccessors() == 1) {
    assert(&BB == TI.getSuccessor(0));
    return true;
  }

  auto *TermPE = getTerminatorPEXP(PredBB);
  if (!TermPE || PI.isNonAffine(TermPE)) {
    return false;
  }

  auto *Int64Ty = Type::getInt64Ty(TI.getContext());
  if (isa<BranchInst>(TI)) {
    EdgeCondition = PVSet::empty(EdgeCondition);
    if (TI.getSuccessor(0) == &BB)
      EdgeCondition.unify(
          buildEqualDomain(TermPE, *ConstantInt::get(Int64Ty, 1)));
    if (TI.getSuccessor(1) == &BB)
      EdgeCondition.unify(
          buildEqualDomain(TermPE, *ConstantInt::get(Int64Ty, 0)));
    return true;
  }

  if (auto *SI = dyn_cast<SwitchInst>(&TI)) {
    bool IsDefaultBlock = (SI->getDefaultDest() == &BB);
    SmallVector<Constant *, 8> OtherCaseValues;
    for (auto &Case : SI->cases()) {
      if (Case.getCaseSuccessor() == &BB)
        EdgeCondition.unify(buildEqualDomain(TermPE, *Case.getCaseValue()));
      else if (IsDefaultBlock)
        OtherCaseValues.push_back(Case.getCaseValue());
    }

    if (IsDefaultBlock && !OtherCaseValues.empty())
      EdgeCondition.unify(buildNotEqualDomain(TermPE, OtherCaseValues));

    return true;
  }

  llvm_unreachable("Unknown terminator!");
  return false;
}

PEXP *PolyhedralExpressionBuilder::getDomain(BasicBlock &BB) {
  PEXP *PE = getOrCreateDomain(BB);
  if (PE->isInitialized())
    return PE;

  if (&BB.getParent()->getEntryBlock() == &BB) {
    NUM_DOMAINS++;
    return PE->setDomain(PVSet::universe(PI.getCtx()));
  }

  auto *L = PI.LI.getLoopFor(&BB);
  bool IsLoopHeader = L && L->getHeader() == &BB;

  if (Scope && (!Scope->contains(&BB) || (Scope == L && IsLoopHeader))) {
    NUM_DOMAINS++;
    return PE->setDomain(PVSet::universe(PI.getCtx()));
  }

  if (L) {
    if (!IsLoopHeader) {
      PEXP *HeaderPE = getDomain(*L->getHeader());
      if (PI.isNonAffine(HeaderPE))
        return PE->invalidate();
    //} else if (auto *PL = L->getParentLoop()) {
      //getDomain(*PL->getHeader());
    }

    // After the recursion we have to update PE.
    //PE = getOrCreateDomain(BB);
  }

  // If we created the domain of the loop header we did produce partial
  // results for other blocks in the loop. While we could update these results
  // we will simply forget them for now and recreate them if necessary.
  auto ForgetDomainsInLoop = [&](Loop &L) {
    if (!IsLoopHeader)
      return;
    for (auto *BB : L.blocks())
      if (BB != L.getHeader())
        PIC.forget(*BB, Scope);
  };

  unsigned LD = getRelativeLoopDepth(&BB);
  PE->setDomain(PVSet::empty(PI.getCtx(), LD), true);

  for (auto *PredBB : predecessors(&BB)) {
    if (IsLoopHeader && L->contains(PredBB)) {
      continue;
    }

    PEXP *PredDomPE = getDomain(*PredBB);
    assert(PredDomPE);

    PVSet DomainOnEdge;
    if (!getDomainOnEdge(DomainOnEdge, *PredDomPE, BB)) {
      ForgetDomainsInLoop(*L);
      return PE->invalidate();
    }

    // Sanity checks
    assert(PredDomPE->isInitialized() &&
           PredDomPE->getKind() != PEXP::EK_NON_AFFINE &&
           !PredDomPE->getPWA().isComplex());

    if (!combine(PE, PredDomPE)) {
      ForgetDomainsInLoop(*L);
      return PE->invalidate();
    }

    PVAff PredDomPWA = PredDomPE->getPWA();
    PredDomPWA.intersectDomain(DomainOnEdge);
    if (!adjustDomainDimensions(PredDomPWA, PredDomPE, PE)) {
      ForgetDomainsInLoop(*L);
      return PE->invalidate();
    }
    PE->PWA.union_max(PredDomPWA);

    // Sanity check
    assert(PE->PWA.getNumInputDimensions() == LD);

    if (!PE->getPWA().isComplex())
      continue;

    COMPLEX_DOMAIN++;
    ForgetDomainsInLoop(*L);
    return PE->invalidate();
  }

  //PVSet ParameterRanges =
      //createParameterRanges(Domain, BB.getModule()->getDataLayout());
  //Domain.simplifyParameters(ParameterRanges);

  if (!IsLoopHeader) {
    NUM_DOMAINS++;
    return PE;
  }

//#if 0
  PVSet Domain = PE->getDomain();
  PVSet NonExitDom = Domain.setInputLowerBound(LD - 1, 0);
  PE->setDomain(NonExitDom, true);
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (auto *ExitingBB : ExitingBlocks) {
    const PEXP *ExitingBBDomainPE =
        ExitingBB == &BB ? PE : getDomain(*ExitingBB);
    assert(ExitingBBDomainPE);

    if (PI.isNonAffine(ExitingBBDomainPE)) {
      ForgetDomainsInLoop(*L);
      return PE->invalidate();
    }

    PVSet ExitingBBDom = ExitingBBDomainPE->getDomain();
    PVSet ExitCond;
    for (auto *SuccBB : successors(ExitingBB))
      if (!L->contains(SuccBB)) {
        PVSet DomainOnEdge;
        if (!getDomainOnEdge(DomainOnEdge, *ExitingBBDomainPE, *SuccBB) ||
            DomainOnEdge.isComplex()) {
          ForgetDomainsInLoop(*L);
          return PE->invalidate();
        }
        //combine(PE, DomainOnEdge);
        ExitCond.unify(DomainOnEdge);
      }
    ExitingBBDom.intersect(ExitCond);
    ExitingBBDom.dropDimsFrom(LD);
    if (ExitingBBDom.getNumInputDimensions() >= LD)
      ExitingBBDom.getNextIterations(LD - 1);
    NonExitDom.subtract(ExitingBBDom);

    if (NonExitDom.isComplex()) {
      ForgetDomainsInLoop(*L);
      return PE->invalidate();
    }
  }

  Domain.fixInputDim(LD - 1, 0);
  Domain.unify(NonExitDom);
//#endif

  //PVSet UnboundedDomain, Domain;
  //Domain = PE->getDomain();
  //Domain.restrictToBoundedPart(LD - 1, &UnboundedDomain);
  //PE->addInvalidDomain(UnboundedDomain);

  //if (Domain.isEmpty())
    //PE->invalidate();
  //else
    PE->setDomain(Domain, true);

  ForgetDomainsInLoop(*L);

  NUM_DOMAINS++;
  PE->PWA.dropUnusedParameters();
  PE->KnownDomain.dropUnusedParameters();
  PE->InvalidDomain.dropUnusedParameters();
  return PE;
}

bool PolyhedralExpressionBuilder::getDomainOnEdge(PVSet &DomainOnEdge,
                                                  const PEXP &PredDomPE,
                                                  BasicBlock &BB) {
  if (PredDomPE.getKind() == PEXP::EK_NON_AFFINE)
    return false;

  assert(PredDomPE.getKind() == PEXP::EK_DOMAIN);
  auto &PredBB = *cast<BasicBlock>(PredDomPE.getValue());

  PVSet EdgeCondition;
  if (!getEdgeCondition(EdgeCondition, PredBB, BB))
    return false;

  PVSet PredDomain = PredDomPE.getDomain();

  DomainOnEdge = PVSet::intersect(PredDomain, EdgeCondition);
  DomainOnEdge.dropUnusedParameters();
  return true;
}

PEXP *PolyhedralExpressionBuilder::visitOperand(Value &Op, Instruction &I) {
  PEXP *PE = visit(Op);
  return PE;

  Instruction *OpI = dyn_cast<Instruction>(&Op);
  Loop *OpL = OpI ? PI.LI.getLoopFor(OpI->getParent()) : nullptr;
  adjustDomainDimensions(PE->PWA, OpL, PI.LI.getLoopFor(I.getParent()), true);
  return PE;

  if (!OpI) {
    return PE;
  }

  Loop *OpIL = PI.LI.getLoopFor(OpI->getParent());
  unsigned NumDims = PE->getPWA().getNumInputDimensions();
  unsigned NumLeftLoops = 0;
  while (OpIL && NumDims && !OpIL->contains(&I)) {
    NumLeftLoops++;
    NumDims--;
    OpIL = OpIL->getParentLoop();
  }

  if (NumLeftLoops) {
    PEXP *OpIDomPE = getDomain(*OpI->getParent());
    if (PI.isNonAffine(OpIDomPE))
      PE->getPWA().dropLastInputDims(NumLeftLoops);
    else {
      PVSet OpIDom = OpIDomPE->getDomain();
      OpIDom.maxInLastInputDims(NumLeftLoops);
      PE->getPWA().intersectDomain(OpIDom);
      PE->getPWA().dropLastInputDims(NumLeftLoops);
    }
  }

  return PE;
}

PEXP *PolyhedralExpressionBuilder::visit(Value &V) {

  PEXP *PE = PIC.lookup(V, Scope);
  if (PE && PE->isInitialized())
    return PE;

  if (!V.getType()->isIntegerTy() && !V.getType()->isPointerTy() &&
      !isa<SelectInst>(V))
    PE = visitParameter(V);
  else if (auto *I = dyn_cast<Instruction>(&V))
    PE = visit(*I);
  else if (auto *CI = dyn_cast<ConstantInt>(&V))
    PE = visit(*CI);
  else if (auto *C = dyn_cast<Constant>(&V))
    PE = visit(*C);
  else
    PE = visitParameter(V);

  assert(PE && PE->isInitialized());

  if (PE->getPWA().isComplex())
    PE->invalidate();

  if (PI.isAffine(PE))
    NUM_EXPRESSIONS++;

  PE->PWA.dropUnusedParameters();
  PE->KnownDomain.dropUnusedParameters();
  PE->InvalidDomain.dropUnusedParameters();
  return PE;
}

PEXP *PolyhedralExpressionBuilder::visit(Constant &I) {
  if (I.isNullValue())
    return visit(*ConstantInt::get(Type::getInt64Ty(I.getContext()), 0));
  return visitParameter(I);
}

PEXP *PolyhedralExpressionBuilder::visit(ConstantInt &I) {
  auto *PE = getOrCreatePEXP(I);
  PE->PWA = PVAff(PI.getCtx(), I.getSExtValue());
  PE->setKind(PEXP::EK_INTEGER);

  return PE;
}

PEXP *PolyhedralExpressionBuilder::createParameter(PEXP *PE) {
  PE->PWA = PVAff(PI.getParameterId(*PE->getValue()));
  PE->setKind(PEXP::EK_UNKNOWN_VALUE);

  NUM_PARAMETERS++;
  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitParameter(Value &V) {
  auto *PE = getOrCreatePEXP(V);
  return createParameter(PE);
}

unsigned PolyhedralExpressionBuilder::getRelativeLoopDepth(Loop *L) {
  if (!L)
    return 0;
  if (!Scope)
    return L->getLoopDepth();
  if (!Scope->contains(L))
    return 0;
  return L->getLoopDepth() - Scope->getLoopDepth();
}

unsigned PolyhedralExpressionBuilder::getRelativeLoopDepth(BasicBlock *BB) {
  Loop *L = PI.LI.getLoopFor(BB);
  return getRelativeLoopDepth(L);
}

Loop *PolyhedralExpressionBuilder::getLoopForPE(const PEXP *PE) {
  Value *V = PE->getValue();
  if (auto *I = dyn_cast<Instruction>(V))
    return PI.LI.getLoopFor(I->getParent());
  if (auto *BB = dyn_cast<BasicBlock>(V))
    return PI.LI.getLoopFor(BB);
  return nullptr;
}

template <typename PVTy>
bool PolyhedralExpressionBuilder::adjustDomainDimensions(PVTy &Obj,
                                                         const PEXP *OldPE,
                                                         const PEXP *NewPE,
                                                         bool LastIt) {
  return adjustDomainDimensions(Obj, getLoopForPE(OldPE), getLoopForPE(NewPE),
                                LastIt);
}

/// Adjust the dimensions of @p Dom that was constructed for @p OldL
///        to be compatible to domains constructed for loop @p NewL.
///
/// This function assumes @p NewL and @p OldL are equal or there is a CFG
/// edge from @p OldL to @p NewL.
template<typename PVTy>
bool PolyhedralExpressionBuilder::adjustDomainDimensions(PVTy &Obj, Loop *OldL,
                                                         Loop *NewL,
                                                         bool LastIt) {
  // If the loops are the same there is nothing to do.
  if (NewL == OldL)
    return true;

  unsigned OldDepth = getRelativeLoopDepth(OldL);
  unsigned NewDepth = getRelativeLoopDepth(NewL);
  if (OldDepth == NewDepth && OldDepth == 0)
    return true;

  // Sanity check
  assert(Obj.getNumInputDimensions() == OldDepth);

  // Distinguish three cases:
  //   1) The depth is the same but the loops are not.
  //      => One loop was left one was entered.
  //   2) The depth increased from OldL to NewL.
  //      => Loops were entered, none was left.
  //   3) The depth decreased from OldL to NewL.
  //      => Loops were left were difference of the depths defines how many.
  if (OldDepth == NewDepth) {
    assert(OldL->getParentLoop() == NewL->getParentLoop());
    if (LastIt)
      Obj.maxInLastInputDims(1);
    if (!Obj)
      return false;
    Obj.dropLastInputDims(1);
    Obj.addInputDims(1);
  } else if (OldDepth < NewDepth) {
    // TODO: one loop could be left
    Obj.addInputDims(NewDepth - OldDepth);
  } else {
    assert(OldDepth > NewDepth);
    unsigned DepthDiff = OldDepth - NewDepth;
    if (LastIt)
      Obj.maxInLastInputDims(DepthDiff);
    if (!Obj)
      return false;
    Obj.dropLastInputDims(DepthDiff);
  }

  return true;
}

PEXP *PolyhedralExpressionBuilder::visit(Instruction &I) {
  assert(I.getType()->isIntegerTy() || I.getType()->isPointerTy() ||
         isa<SelectInst>(I));

  auto *PE = InstVisitor::visit(I);

  unsigned RelLD = getRelativeLoopDepth(I.getParent());
  unsigned NumDims = PE->PWA.getNumInputDimensions();
  // TODO FIXME:
  if (NumDims > RelLD) {
    return PE->invalidate();
  }
  assert(NumDims <= RelLD);
  PE->PWA.addInputDims(RelLD - NumDims);
  assert(PE->PWA.getNumInputDimensions() == RelLD);

  return PE;
}

PEXP *PolyhedralExpressionBuilder::getTerminatorPEXP(BasicBlock &BB) {
  auto *Term = BB.getTerminator();

  switch (Term->getOpcode()) {
  case Instruction::Br: {
    auto *BI = cast<BranchInst>(Term);
    if (BI->isUnconditional())
      return nullptr;
    else
      return visitOperand(*BI->getCondition(), *Term);
  }
  case Instruction::Switch:
    return visitOperand(*cast<SwitchInst>(Term)->getCondition(), *Term);
  case Instruction::Ret:
  case Instruction::Unreachable:
    return nullptr;
  case Instruction::IndirectBr:
    /// @TODO This can be over-approximated
    return nullptr;
  case Instruction::Invoke:
  case Instruction::Resume:
  case Instruction::CleanupRet:
  case Instruction::CatchRet:
  case Instruction::CatchSwitch:
    return nullptr;
  default:
    return nullptr;
  }

  llvm_unreachable("unknown terminator");
  return nullptr;
}

// ------------------------------------------------------------------------- //

PEXP *PolyhedralExpressionBuilder::visitICmpInst(ICmpInst &I) {

  auto *LPE = visitOperand(*I.getOperand(0), I);
  if (PI.isNonAffine(LPE))
    return visitParameter(I);
  auto *RPE = visitOperand(*I.getOperand(1), I);
  if (PI.isNonAffine(RPE))
    return visitParameter(I);

  auto Pred = I.getPredicate();
  auto IPred = I.getInversePredicate();
  auto TrueDomain = PVAff::buildConditionSet(Pred, LPE->PWA, RPE->PWA);
  if (TrueDomain.isComplex()) {
    COMPLEX_DOMAIN++;
    return visitParameter(I);
  }

  auto FalseDomain = PVAff::buildConditionSet(IPred, LPE->PWA, RPE->PWA);
  if (FalseDomain.isComplex()) {
    COMPLEX_DOMAIN++;
    return visitParameter(I);
  }

  auto *PE = getOrCreatePEXP(I);
  PE->PWA = PVAff(FalseDomain, 0);
  PE->PWA.union_add(PVAff(TrueDomain, 1));
  combine(PE, LPE);
  combine(PE, RPE);
  PE->Kind = PEXP::EK_INTEGER;
  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitFCmpInst(FCmpInst &I) {
  return visitParameter(I);
}

PEXP *PolyhedralExpressionBuilder::visitLoadInst(LoadInst &I) {
  return visitParameter(I);
}

PEXP *
PolyhedralExpressionBuilder::visitGetElementPtrInst(GetElementPtrInst &I) {
  auto &DL = I.getModule()->getDataLayout();

  auto *PtrPE = visitOperand(*I.getPointerOperand(), I);
  if (PI.isNonAffine(PtrPE))
    return visitParameter(I);

  auto *PE = getOrCreatePEXP(I);
  *PE = *PtrPE;
  if (!adjustDomainDimensions(PE->PWA, PtrPE, PE, true))
    return visitParameter(I);

  auto *Ty = I.getPointerOperandType();
  for (auto &Op : make_range(I.idx_begin(), I.idx_end())) {
    auto *PEOp = visitOperand(*Op, I);
    if (PI.isNonAffine(PEOp))
      return visitParameter(I);

    if (Ty->isStructTy()) {
      if (!PI.isConstant(PEOp)) {
        return visitParameter(I);
      }
      if (auto *ConstOp = dyn_cast<ConstantInt>(Op)) {
        auto StructElementNo = ConstOp->getZExtValue();
        assert(StructElementNo < Ty->getStructNumElements());
        const StructLayout *Layout = DL.getStructLayout(cast<StructType>(Ty));
        auto ElementOffset = Layout->getElementOffset(StructElementNo);
        PVAff Offset(PE->PWA, ElementOffset);
        PE->PWA.add(Offset);
        Ty = Ty->getStructElementType(StructElementNo);
        continue;
      }
      assert(0 && "TODO Constant OpPE but not constant Op!\n");
    }

    uint64_t Size = 0;
    if (Ty->isPointerTy()) {
      Ty = Ty->getPointerElementType();
      Size = DL.getTypeAllocSize(Ty);
    } else if (Ty->isArrayTy()) {
      Ty = Ty->getArrayElementType();
      Size = DL.getTypeAllocSize(Ty);
    } else {
      return visitParameter(I);
    }

    combine(PE, PEOp);

    PVAff ScaledPWA(PEOp->getDomain(), Size);
    ScaledPWA.multiply(PEOp->getPWA());
    PE->PWA.add(ScaledPWA);
  }

  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitCastInst(CastInst &I) {
  PEXP *PE = getOrCreatePEXP(I);
  *PE = *visitOperand(*I.getOperand(0), I);

  switch (I.getOpcode()) {
  case Instruction::Trunc: {
    // Handle changed values.
    unsigned TypeWidth =
        I.getModule()->getDataLayout().getTypeSizeInBits(I.getType());
    if (TypeWidth > 64 || TypeWidth < 2)
      return visitParameter(I);
    int64_t ExpVal = ((int64_t)1) << (TypeWidth - 1);
    const PVAff &PWA = PE->getPWA();
    PVSet UpperLimitSet = PWA.getGreaterEqualDomain(PVAff(PWA, ExpVal));
    PVSet LowerLimitSet = PWA.getLessThanDomain(PVAff(PWA, -ExpVal));
    PE->addInvalidDomain(UpperLimitSet);
    PE->addInvalidDomain(LowerLimitSet);
    break;
  }
  case Instruction::ZExt:
    // Handle negative values.
    PE->addInvalidDomain(
        PE->getPWA().getLessThanDomain(PVAff(PE->getPWA(), 0)));
    break;
  case Instruction::SExt:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    // No-op
    break;
  default:
    llvm_unreachable("Unhandled cast operation!\n");
  }

  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitSelectInst(SelectInst &I) {
  auto *CondPE = visitOperand(*I.getCondition(), I);
  if (PI.isNonAffine(CondPE))
    return visitParameter(I);

  auto *OpTrue = visitOperand(*I.getTrueValue(), I);
  auto CondZero = CondPE->getPWA().zeroSet();

  auto *OpFalse = visitOperand(*I.getFalseValue(), I);
  auto CondNonZero = CondPE->getPWA().nonZeroSet();

  auto *PE = getOrCreatePEXP(I);
  if (!PI.isNonAffine(OpTrue))
    combine(PE, OpTrue);

  if (!PI.isNonAffine(OpFalse))
    combine(PE, OpFalse);

  if (PI.isNonAffine(OpTrue)) {
    PE->InvalidDomain.unify(CondNonZero);
    PE->PWA = OpFalse->getPWA();
    PE->setKind(OpFalse->getKind());
    return PE;
  }

  if (PI.isNonAffine(OpFalse)) {
    PE->InvalidDomain.unify(CondZero);
    PE->PWA = OpTrue->getPWA();
    PE->setKind(OpTrue->getKind());
    return PE;
  }

  PE->PWA = PVAff::createSelect(CondPE->getPWA(), OpTrue->getPWA(),
                                OpFalse->getPWA());
  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitInvokeInst(InvokeInst &I) {
  // TODO: Interprocedural support
  return visitParameter(I);
}

PEXP *PolyhedralExpressionBuilder::visitCallInst(CallInst &I) {
  // TODO: Interprocedural support
  return visitParameter(I);
}

PEXP *PolyhedralExpressionBuilder::visitConditionalPHINode(PHINode &I) {
  bool InvalidOtherwise = false;

  auto *PE = getOrCreatePEXP(I);

  BasicBlock &BB = *I.getParent();
  unsigned RelLD = getRelativeLoopDepth(&BB);

  for (unsigned u = 0, e = I.getNumIncomingValues(); u < e; u++) {

    auto *PredOpPE = visit(*I.getIncomingValue(u));
    assert(PredOpPE);

    if (PI.isNonAffine(PredOpPE)) {
      InvalidOtherwise = true;
      continue;
    }

    PEXP *PredDomPE = getDomain(*I.getIncomingBlock(u));
    assert(PredDomPE);

    PVSet DomainOnEdge;
    if (!getDomainOnEdge(DomainOnEdge, *PredDomPE, BB)) {
      InvalidOtherwise = true;
      continue;
    }

    if (!combine(PE, PredOpPE) || !combine(PE, PredDomPE)) {
      InvalidOtherwise = true;
      continue;
    }

    PVAff PredOpPWA = PredOpPE->getPWA();
    PredOpPWA.intersectDomain(DomainOnEdge);

    if (PredOpPWA.getDomain().isEmpty())
      continue;

    if (!adjustDomainDimensions(PredOpPWA, PredDomPE, PE, true))
      return visitParameter(I);
    PE->PWA.union_add(PredOpPWA);

    // Sanity check
    assert(PE->getPWA().getNumInputDimensions() == RelLD);

    if (PE->getPWA().isComplex()) {
      COMPLEX_DOMAIN++;
      return visitParameter(I);
    }
  }

  if (!PE->isInitialized() || PE->getDomain().isEmpty()) {
    // No valid predecessor found.
    return visitParameter(I);
  }

  assert(PE->PWA.getNumInputDimensions() == RelLD);

  if (InvalidOtherwise) {
    assert(PE->PWA);
    PVSet Dom = PE->getDomain();

    // TODO: check complexity.
    PE->addInvalidDomain(Dom.complement());
  }

  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitPHINode(PHINode &I) {
  auto &BB = *I.getParent();
  Loop *L = PI.LI.getLoopFor(&BB);
  bool IsLoopHeader = L && L->getHeader() == &BB;

  if (!IsLoopHeader)
    return visitConditionalPHINode(I);

  unsigned NumLatches = L->getNumBackEdges();
  if (NumLatches > 1) {
    return visitParameter(I);
  }

  PEXP *ParamPE = PIC.getOrCreatePEXP(I, L);
  assert(ParamPE);

  PVId Id = PI.getParameterId(I);
  if (!ParamPE->isInitialized()) {
    ParamPE->PWA = PVAff(Id);
    ParamPE->Kind = PEXP::EK_UNKNOWN_VALUE;
  }

  PEXP *PE = getOrCreatePEXP(I);

  if (Scope == L || (Scope && !Scope->contains(&I))) {
    *PE = *ParamPE;
    return PE;
  }

  if (!PE || PE->isInitialized()) {
    if (PE)
      PE->dump();
  }
  assert(PE && !PE->isInitialized());
  PE->PWA = PVAff(Id);
  PE->setKind(PEXP::EK_UNKNOWN_VALUE);

  unsigned LoopDim = getRelativeLoopDepth(L);

  auto OldScope = Scope;
  setScope(L);

  PVAff BackEdgeOp;

  PVSet NegationSet;
  bool OtherPHIs = false;
  SmallVector<std::pair<PVId, PVAff>, 4> PHIInfos;
  for (unsigned u = 0, e = I.getNumIncomingValues(); u != e; u++) {
    auto *OpBB = I.getIncomingBlock(u);
    if (!L->contains(OpBB))
      continue;

    auto *OpVal = I.getIncomingValue(u);
    auto *OpPE = visit(*OpVal);
    PVAff OpAff = OpPE->getPWA();

    bool SelfRecurrent = false;
    SmallVector<PVId, 4> Parameters;
    OpAff.getParameters(Parameters);
    for (const PVId &ParameterId : Parameters) {
      Value *ParameterV = ParameterId.getPayloadAs<Value *>();
      if (!isa<Instruction>(ParameterV)) {
        //OpAff.dropParameter(ParameterId);
        continue;
      }
      Instruction *ParameterI = cast<Instruction>(ParameterV);
      if (!L->contains(ParameterI)) {
        //OpAff.dropParameter(ParameterId);
        continue;
      }
      if (ParameterV == &I) {
        SelfRecurrent = true;
        continue;
      }
      if (isa<PHINode>(ParameterI) &&
          ParameterI->getParent() == I.getParent()) {
        const PEXP *ParameterPE = PI.getPEXP(ParameterI, OldScope);
        if (!PI.isAffine(ParameterPE)) {
          setScope(OldScope);
          return visitParameter(I);
        }
        if (!PI.hasScope(ParameterPE, L, true)) {
          setScope(OldScope);
          return visitParameter(I);
        }
        PVAff ParameterCoeff = OpAff.getParameterCoeff(ParameterId);
        assert(ParameterCoeff.isInteger());
        OtherPHIs = true;
        PHIInfos.push_back(
            {ParameterId, ParameterCoeff.multiply(ParameterPE->getPWA())});
        continue;
      }
      setScope(OldScope);
      return visitParameter(I);
    }

    if (SelfRecurrent) {
      for (auto &PHIInfo : PHIInfos) {
        if (PHIInfo.second.involvesInput(LoopDim - 1)) {
          setScope(OldScope);
          return visitParameter(I);
        }
      }

      OpAff = OpAff.perPiecePHIEvolution(Id, LoopDim - 1, NegationSet);
      if (!OpAff) {
        setScope(L->getParentLoop());
        return visitParameter(I);
      }
    }

#if 0
    if (!OpAff.isConstant()) {
      if (!PI.hasScope(*OpVal, L, true)) {
        // TODO: This is too strict.
        setScope(OldScope);
        return visitParameter(I);
      }

      ConstantStride = false;
      if (OldScope != Scope) {
        setScope(OldScope);
        OpPE = visit(*OpVal);
        setScope(L);
      }
    }
#endif

    //OpAff = OpPE->getPWA();
    if (NumLatches > 1 || OtherPHIs) {
      PEXP *OpBBDomPE = getDomain(*OpBB);
      assert(OpBBDomPE);

      PVSet EdgeDom;
      if (!getDomainOnEdge(EdgeDom, *OpBBDomPE, BB)) {
        setScope(OldScope);
        return visitParameter(I);
      }

      EdgeDom.setInputLowerBound(LoopDim - 1, 1);
      OpAff.intersectDomain(EdgeDom);
    }

    BackEdgeOp.union_add(OpAff);
  }

#if 0
  if (ConstantStride) {
    BackEdgeOp = BackEdgeOp.perPiecePHIEvolution(Id, LoopDim - 1);
    if (!BackEdgeOp) {
      setScope(OldScope);
      return visitParameter(I);
    }
  }
#endif

  for (auto &PHIInfo : PHIInfos) {
    BackEdgeOp.dropParameter(PHIInfo.first);
    BackEdgeOp.add(PHIInfo.second.moveOneIteration(LoopDim - 1));
  }

  PE->PWA = PVAff();
  PE->PWA.union_add(BackEdgeOp);
  PE->PWA.dropParameter(Id);
  setScope(OldScope);

  for (unsigned u = 0, e = I.getNumIncomingValues(); u != e; u++) {
    auto *OpBB = I.getIncomingBlock(u);
    if (L->contains(OpBB))
      continue;

    auto *OpVal = I.getIncomingValue(u);
    auto *OpPE = visit(*OpVal);
    if (PI.isNonAffine(OpPE)) {
      return visitParameter(I);
    }

    PVAff OpAff = OpPE->getPWA();
    assert(e > NumLatches);
    if (e - NumLatches > 1 || OtherPHIs) {
      PEXP *OpBBDomPE = getDomain(*OpBB);
      assert(OpBBDomPE);

      PVSet EdgeDom;
      if (!getDomainOnEdge(EdgeDom, *OpBBDomPE, BB)) {
        return visitParameter(I);
      }

      EdgeDom.fixInputDim(LoopDim - 1, 0);
      OpAff.intersectDomain(EdgeDom);
    }

    PE->PWA.union_add(OpAff);
    combine(PE, OpPE);
  }

  if (NegationSet) {
    PVAff NegationAff(PE->PWA.getDomain(), 1);
    NegationAff.union_add(PVAff(NegationSet, -2));
    PE->PWA.multiply(NegationAff);
  }

  PE->Kind = PEXP::EK_UNKNOWN_VALUE;

  return PE;
}

PEXP *PolyhedralExpressionBuilder::visitBinaryOperator(BinaryOperator &I) {

  Value *Op0 = I.getOperand(0);
  auto *PEOp0 = visitOperand(*Op0, I);
  if (PI.isNonAffine(PEOp0))
    return visitParameter(I);

  Value *Op1 = I.getOperand(1);
  auto *PEOp1 = visitOperand(*Op1, I);
  if (PI.isNonAffine(PEOp1))
    return visitParameter(I);

  auto *PE = getOrCreatePEXP(I);
  switch (I.getOpcode()) {
  case Instruction::Add:
    if (assign(PE, PEOp0, PEOp1, PVAff::createAdd))
      PE->adjustInvalidAndKnownDomain();
    return PE;
  case Instruction::Sub:
    if (assign(PE, PEOp0, PEOp1, PVAff::createSub))
      PE->adjustInvalidAndKnownDomain();
    return PE;

  case Instruction::Mul:
    if (PEOp0->Kind != PEXP::EK_INTEGER && PEOp1->Kind != PEXP::EK_INTEGER)
      return visitParameter(I);
    if (assign(PE, PEOp0, PEOp1, PVAff::createMultiply))
      PE->adjustInvalidAndKnownDomain();
    return PE;

  case Instruction::SRem:
// TODO: This is not yet compatible with the PHI handling which assumes
//       monotonicity!
#if 0
    if (PEOp1->Kind == PEXP::EK_INTEGER) {
      auto NZ = PEOp1->PWA.nonZeroSet();
      PEOp1->PWA.intersectDomain(NZ);
      return assign(PE, PEOp0, PEOp1, isl_pw_aff_tdiv_r);
    }
#endif
    return visitParameter(I);
  case Instruction::SDiv:
    if (PEOp1->Kind == PEXP::EK_INTEGER) {
      auto NZ = PEOp1->PWA.nonZeroSet();
      PEOp1->PWA.intersectDomain(NZ);
      assign(PE, PEOp0, PEOp1, PVAff::createSDiv);
      return PE;
    }
    return visitParameter(I);
  case Instruction::Shl:
    if (PEOp1->Kind == PEXP::EK_INTEGER) {
      assign(PE, PEOp0, PEOp1, PVAff::createShiftLeft);
      return PE;
    }
    return visitParameter(I);
  case Instruction::UDiv:
  case Instruction::AShr:
  case Instruction::LShr:
  case Instruction::URem:
    // TODO
    return visitParameter(I);

  // Bit operations
  case Instruction::And:
    if (I.getType()->isIntegerTy(1)) {
      if (assign(PE, PEOp0, PEOp1, PVAff::createAdd)) {
        PE->PWA.floordiv(2);
        PE->PWA.select(getOne(PE->PWA), getZero(PE->PWA));
      }
      return PE;
    }
    return visitParameter(I);
  case Instruction::Or:
    if (I.getType()->isIntegerTy(1)) {
      if (assign(PE, PEOp0, PEOp1, PVAff::createAdd))
        PE->PWA.select(getOne(PE->PWA), getZero(PE->PWA));
      return PE;
    }
    return visitParameter(I);
  case Instruction::Xor:
    if (I.getType()->isIntegerTy(1)) {
      auto OneBitXOR = [this](const PVAff &PWA0, const PVAff &PWA1) {
        auto OnePWA = getOne(PWA0);
        auto ZeroPWA = getZero(PWA0);
        auto Mul = PVAff::createMultiply(PWA0, PWA1);
        auto MulZero = Mul.zeroSet();
        auto Add = PVAff::createAdd(PWA0, PWA1);
        auto AddNonZero = Add.nonZeroSet();
        auto TrueSet = MulZero.intersect(AddNonZero);
        return PVAff(TrueSet).select(OnePWA, ZeroPWA);
      };
      assign(PE, PEOp0, PEOp1, OneBitXOR);
      return PE;
    }
    if (auto *COp0 = dyn_cast<ConstantInt>(Op0))
      if (COp0->isMinusOne()) {
        if (assign(PE, PEOp0, PEOp1, PVAff::createMultiply))
          combine(PE, PEOp0, PVAff::createAdd);
      }
    if (auto *COp1 = dyn_cast<ConstantInt>(Op1))
      if (COp1->isMinusOne()) {
        if (assign(PE, PEOp0, PEOp1, PVAff::createMultiply))
          combine(PE, PEOp1, PVAff::createAdd);
      }

    // TODO
    return visitParameter(I);
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::BinaryOpsEnd:
    break;
  }

  llvm_unreachable("Invalid Binary Operation");
}

PEXP *PolyhedralExpressionBuilder::visitAllocaInst(AllocaInst &I) {
  return visitParameter(I);
}

PEXP *PolyhedralExpressionBuilder::visitInstruction(Instruction &I) {
  assert(!I.getType()->isVoidTy());
  return visitParameter(I);
}

PVSet PolyhedralExpressionBuilder::buildNotEqualDomain(
    const PEXP *PE, ArrayRef<Constant *> CIs) {
  assert(PE->Kind != PEXP::EK_NON_AFFINE && PE->PWA);

  PVSet NotEqualDomain;
  for (auto *CI : CIs) {
    auto *CPE = visit(*static_cast<Value *>(CI));
    assert(CPE->Kind == PEXP::EK_INTEGER && CPE->PWA && !CPE->InvalidDomain);

    NotEqualDomain.intersect(
        PVAff::buildConditionSet(ICmpInst::ICMP_NE, PE->PWA, CPE->PWA));
  }

  return NotEqualDomain;
}

PVSet PolyhedralExpressionBuilder::buildEqualDomain(const PEXP *PE,
                                                    Constant &CI) {
  assert(PE->Kind != PEXP::EK_NON_AFFINE && PE->PWA);

  auto *CPE = visit(static_cast<Value &>(CI));
  assert(CPE->Kind == PEXP::EK_INTEGER && CPE->PWA && !CPE->InvalidDomain);

  return PVAff::buildConditionSet(ICmpInst::ICMP_EQ, PE->PWA, CPE->PWA);
}
