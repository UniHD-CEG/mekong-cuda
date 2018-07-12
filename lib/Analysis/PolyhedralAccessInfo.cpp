//===----- PolyhedralExpressionBuilder.cpp  - TODO --------------*- C++ -*-===//
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

#include "llvm/Analysis/PolyhedralAccessInfo.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PolyhedralExpressionBuilder.h"
#include "llvm/Analysis/PolyhedralValueInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "polyhedral-access-info"

raw_ostream &llvm::operator<<(raw_ostream &OS, PACC::AccessKind Kind) {
  switch (Kind) {
  case PACC::AK_READ:
    return OS << "READ";
  case PACC::AK_WRITE:
    return OS << "WRITE";
  case PACC::AK_READ_WRITE:
    return OS << "READ & WRITE";
  default:
    llvm_unreachable("Unknown polyhedral access kind");
  }
}

PACC::PACC(Value &Pointer, PEXP &PE, PVId Id, AccessKind AccKind)
    : BasePointer(Id.getPayloadAs<Value *>()), Pointer(&Pointer), Id(Id),
      PE(PE), AccKind(AccKind) {
  PE.getPWA().dropUnusedParameters();
}

PACC::~PACC() { delete &PE; }

void PACC::print(raw_ostream &OS) const {
  OS << "PACC [" << *PE.getValue() << "] @ [" << Pointer << "]\n"
     << Id << "[" << PE << "] (" << AccKind << ")\n";
}

void PACC::dump() const { print(dbgs()); }

raw_ostream &llvm::operator<<(raw_ostream &OS, const PACC *PA) {
  if (PA)
    OS << *PA;
  else
    OS << "<null>";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PACC &PA) {
  PA.print(OS);
  return OS;
}

// ------------------------------------------------------------------------- //

PACCSummary::PACCSummary(SummaryScopeKind Kind, const ContainsFuncTy &Contains,
                         Loop *Scope)
    : Kind(Kind), Contains(Contains), Scope(Scope) {}

PACCSummary::~PACCSummary() { DeleteContainerSeconds(ArrayInfoMap); }

static uint64_t getElementSize(Value *Pointer, const DataLayout &DL) {
  Type *PointerTy = Pointer->getType();
  assert(PointerTy->isPointerTy());

  return DL.getTypeStoreSize(PointerTy->getPointerElementType());
}

const PEXP *PACCSummary::findMultidimensionalViewSize(
    PolyhedralValueInfo &PI, ArrayRef<const PEXP *> PEXPs,
    DenseSet<std::pair<Instruction *, const PEXP *>>
        &InstsAndRemainders) {

  if (PEXPs.empty())
    return nullptr;

  SmallPtrSet<Value *, 4> DomainParameterSet;
  DenseMap<Value *, SmallVector<const PEXP*, 4>> ExprParameterMap;
  SmallVector<Value *, 4> ParameterVector;
  for (const PEXP *PE : PEXPs) {
    ParameterVector.clear();
    PE->getPWA().getParameters(ParameterVector);
    for (Value *Parameter : ParameterVector)
      ExprParameterMap[Parameter].push_back(PE);
    ParameterVector.clear();

    PE->getDomain().getParameters(ParameterVector);
    DomainParameterSet.insert(ParameterVector.begin(), ParameterVector.end());
  }

  for (Value *V : DomainParameterSet)
    ExprParameterMap.erase(V);

  DenseMap<const PEXP *, SmallVector<std::pair<Instruction *, const PEXP *>, 4>>
      PotentialSizes;

  for (auto &It : ExprParameterMap) {
    Value *V = It.first;
    auto *I = dyn_cast<Instruction>(V);
    if (!I) {
      continue;
    }

    if (I->getOpcode() != Instruction::Mul) {
      continue;
    }

    Value *Op0 = I->getOperand(0);
    Value *Op1 = I->getOperand(1);

    const PEXP *OpPE0 = PI.getPEXP(Op0, Scope);
    const PEXP *OpPE1 = PI.getPEXP(Op1, Scope);

    if (PI.isUnknown(OpPE0) && OpPE0->getPWA().getNumInputDimensions() == 0)
      PotentialSizes[OpPE0].push_back({I, OpPE1});
    if (PI.isUnknown(OpPE1) && OpPE1->getPWA().getNumInputDimensions() == 0)
      PotentialSizes[OpPE1].push_back({I, OpPE0});
  }

  // TODO Look for loop bounds, etc.
  for (const PEXP *PE : PEXPs) {

  }

  if (PotentialSizes.empty()) {
    return nullptr;
  }

  const PEXP *PotentialSize = nullptr;
  if (PotentialSizes.size()  == 1)
    PotentialSize = PotentialSizes.begin()->first;
  else {
    for (auto &It : PotentialSizes) {
      ParameterVector.clear();
      PI.getParameters(It.first, ParameterVector);
      if (ParameterVector.empty() ||
          !std::all_of(ParameterVector.begin(), ParameterVector.end(),
                       [](Value *P) { return isa<Argument>(P); }))
        continue;
      if (!PotentialSize) {
        PotentialSize = It.first;
        continue;
      }

      if (!PotentialSize->getPWA().isEqual(It.first->getPWA()))
        return nullptr;
      PotentialSizes[PotentialSize].append(It.second.begin(), It.second.end());
    }
  }

  if (!PotentialSize) {
    return nullptr;
  }

  for (auto &It : PotentialSizes[PotentialSize])
    InstsAndRemainders.insert(It);
  return PotentialSize;
}

void PACCSummary::findMultidimensionalView(PolyhedralValueInfo &PI,
                                           MultiDimensionalViewInfo &MDVI,
                                           ArrayRef<const PACC *> PACCs) {
  if (PACCs.empty())
    return;

  SmallVector<const PEXP *, 8> PEXPs;
  PEXPs.reserve(PACCs.size());
  for (auto *PA : PACCs)
    PEXPs.push_back(PA->getPEXP());

  DenseSet<std::pair<Instruction *, const PEXP *>> InstsAndRemainders;
  while (1) {
    unsigned IARSize = InstsAndRemainders.size();
    const PEXP *DimSize =
        findMultidimensionalViewSize(PI, PEXPs, InstsAndRemainders);
    if (!DimSize)
      return;

    PEXPs.clear();

    MDVI.DimensionSizes.push_back(DimSize);
    unsigned CurDim = MDVI.DimensionSizes.size();
    for (const auto &InstAndRemainder : InstsAndRemainders) {
      auto &DimInfo = MDVI.DimensionInstsMap[InstAndRemainder.first];
      PEXPs.push_back(InstAndRemainder.second);
      if (!DimInfo.second)
        DimInfo = std::make_pair(CurDim, InstAndRemainder.second);
      else {
        assert(DimInfo.second == InstAndRemainder.second);
        DimInfo.first = std::max(DimInfo.first, CurDim);
      }
    }
    InstsAndRemainders.clear();
  }
}

void PACCSummary::finalize(PolyhedralValueInfo &PI,
                           ArrayRef<const PACC *> PACCs, const DataLayout &DL) {
  assert(ArrayInfoMap.empty());

  for (const PACC *PA : PACCs) {
    Value *BasePointer = PA->getBasePointer();

    if (Kind == SSK_EXTERNAL && isa<AllocaInst>(BasePointer))
      continue;

    PACCMap[BasePointer].push_back(PA);
  }

  for (auto It : PACCMap) {
    Value *BasePointer = It.first;
    auto &PACCVector = It.second;

    ArrayInfo *&AI = ArrayInfoMap[BasePointer];
    AI = new ArrayInfo();

    AI->ElementSize = getElementSize(PACCVector[0]->getPointer(), DL);
    for (const PACC *PA : PACCVector) {
      assert(PA);
      assert(PA->getPEXP());
      assert(PA->getPEXP()->getPWA());
      // TODO: Also take the constant PA offset into account!
      AI->ElementSize = GreatestCommonDivisor64(
          AI->ElementSize, getElementSize(PA->getPointer(), DL));
    }

    MultiDimensionalViewInfo MDVI(AI->DimensionSizes);
    findMultidimensionalView(PI, MDVI, PACCVector);

    for (const PACC *PA : PACCVector) {
      PVAff PWA = PA->getPEXP()->getPWA();
      PWA.floordiv(AI->ElementSize);

      SmallVector<PVAff, 4> DimPWAs;

      SmallVector<SmallVector<std::pair<Instruction *, const PEXP *>, 4>, 4>
          Dimensions;
      Dimensions.resize(MDVI.DimensionSizes.size());
      for (const auto &It : MDVI.DimensionInstsMap) {
        assert(It.second.first > 0);
        unsigned Dim = It.second.first - 1;
        auto &DimInfo = Dimensions[Dim];
        //assert(DimInfo.first == nullptr && DimInfo.second == nullptr);
        //DimInfo.first = It.first;
        //DimInfo.second = It.second.second;
        DimInfo.push_back({It.first, It.second.second});
      }

      DimPWAs.resize(Dimensions.size() + 1);

      int LastDim = Dimensions.size();
      assert(!DimPWAs[LastDim]);
      DimPWAs[LastDim] = PWA;

      for (int Dim = 0; Dim < LastDim; Dim++) {
        auto &DimInfo = Dimensions[Dim];
        assert(DimInfo.size());

        PVAff &LastPWA = DimPWAs[LastDim - Dim];

        for (auto &It : DimInfo) {
          PVId PId = PI.getParameterId(*It.first);
          PVAff Coeff = LastPWA.getParameterCoeff(PId);
          assert(!Coeff || Coeff.isConstant());
          if (!Coeff || Coeff.isEqual(PVAff(Coeff, 0)))
            continue;

          PVAff &DimPWA = DimPWAs[LastDim - Dim - 1];
          assert(!DimPWA || DimPWA.isEqual(It.second->getPWA()));

          DimPWA = It.second->getPWA();
          DimPWA.multiply(Coeff);

          const PVAff &Size = AI->DimensionSizes[Dim]->getPWA();
          PVAff SizeFactor = LastPWA.extractFactor(Size);
          if (SizeFactor) {
            DimPWA.add(SizeFactor);
            LastPWA.sub(SizeFactor.multiply(Size));
          }
          LastPWA.sub(Coeff.multiply(PVAff(PId)));
        }
      }

      PVMap Map(DimPWAs, PA->getId());
      Map.dropUnusedParameters();

      SmallVector<PVId, 4> Parameters;
      Map.getParameters(Parameters);

      bool IsMayAccess = false;
      for (const PVId &PId : Parameters) {
        auto *ParamValue = PId.getPayloadAs<Value *>();
        if (PI.hasScope(*ParamValue, /* Scope */ nullptr, false))
          continue;

        // TODO approximate.

        IsMayAccess = true;
        Map.eliminateParameter(PId);
      }

      AI->AccessMultiDimMap[PA] = Map;

      if (PA->isWrite()) {
        if (IsMayAccess)
          AI->MayWriteMap.union_add(Map);
        else
          AI->MustWriteMap.union_add(Map);
      } else {
        if (IsMayAccess)
          AI->MayReadMap.union_add(Map);
        else
          AI->MustReadMap.union_add(Map);
      }
    }
  }
}

void PACCSummary::rewrite(PVRewriter<PVMap> &Rewriter) {
  for (auto AIt : *this) {
    ArrayInfo *AI = AIt.second;
    Rewriter.rewrite(AI->MayWriteMap);
    Rewriter.rewrite(AI->MustWriteMap);
    Rewriter.rewrite(AI->MayReadMap);
    Rewriter.rewrite(AI->MustReadMap);
  }
}

void PACCSummary::print(raw_ostream &OS, PolyhedralValueInfo *PVI) const {
  OS << "\nPACC summary\n";

  if (getNumUnknownReads())
    OS << "\tUnknown reads:\n";
  else
    OS << "\tUnknown reads: None\n";
  for (auto It = unknown_reads_begin(), End = unknown_reads_end(); It != End;
       It++)
    OS << "\t - " << **It << "\n";

  if (getNumUnknownWrites())
    OS << "\tUnknown writes:\n";
  else
    OS << "\tUnknown writes: None\n";
  for (auto It = unknown_writes_begin(), End = unknown_writes_end(); It != End;
       It++)
    OS << "\t - " << **It << "\n";

  std::set<PVId> ParameterSet;
  SmallVector<PVId, 8> ParameterVector;
  OS << "Array infos:\n";
  for (auto AIt : *this) {
    Value *BasePointer = AIt.first;
    OS << "\tBase pointer: " << (BasePointer ? BasePointer->getName() : "<n/a>")
       << "\n";
    AIt.second->collectParameters(ParameterSet);
    AIt.second->print(OS);
  }

  OS << "Referenced parameters:\n";
  if (PVI) {
    std::set<PVId> ParameterWorklist(ParameterSet);
    while (!ParameterWorklist.empty()) {
      const PVId &ParameterId = *ParameterWorklist.begin();
      ParameterWorklist.erase(ParameterId);
      if (!ParameterId.getPayload())
        continue;
      Value *Parameter = ParameterId.getPayloadAs<Value *>();
      auto *ParameterInst = dyn_cast<Instruction>(Parameter);
      if (!ParameterInst)
        continue;
      for (Value *ParameterOperand : ParameterInst->operands()) {
        const PEXP *ParameterPE = PVI->getPEXP(ParameterOperand, Scope);
        ParameterVector.clear();
        PVI->getParameters(ParameterPE, ParameterVector);
        for (const PVId NewParameterId : ParameterVector) {
          if (!ParameterSet.insert(NewParameterId).second)
            continue;
          ParameterWorklist.insert(NewParameterId);
        }
      }
    }
  }
  for (const PVId &ParameterId : ParameterSet) {
    if (!ParameterId.getPayload()) {
      OS << "\t\t - " << ParameterId.str() << " (P)\n";
      continue;
    }

    Value *Parameter = ParameterId.getPayloadAs<Value *>();
    if (auto *ArgumentParameter = dyn_cast<Argument>(Parameter)) {
      OS << "\t\t - " << ParameterId.str() << " (A)("
         << ArgumentParameter->getArgNo() << "):  " << *Parameter << "\n";
    } else if (isa<Instruction>(Parameter)) {
      OS << "\t\t - " << ParameterId.str() << " (I):" << *Parameter << "\n";
    } else if (isa<Function>(Parameter)) {
      OS << "\t\t - " << ParameterId.str() << " (F)\n";
    } else {
      OS << "\t\t - " << ParameterId.str() << " (U): " << *Parameter << "\n";
    }
  }
}

void PACCSummary::ArrayInfo::print(raw_ostream &OS) const {
  if (!DimensionSizes.empty()) {
    OS << "\t\tDimension sizes:\n";
    for (const PEXP *DimSizePE : DimensionSizes)
      OS << "\t\t- " << DimSizePE->getPWA().str() << "\n";
  }
  if (MayReadMap)
    OS << "\t\tMayRead: " << MayReadMap << "\n";
  if (MustReadMap)
    OS << "\t\tMustRead: " << MustReadMap << "\n";
  if (MayWriteMap)
    OS << "\t\tMayWrite: " << MayWriteMap << "\n";
  if (MustWriteMap)
    OS << "\t\tMustWrite: " << MustWriteMap << "\n";
  OS << "\n";
}

void PACCSummary::ArrayInfo::collectParameters(std::set<PVId> &ParameterSet) const {
  SmallVector<PVId, 8> ParameterVector;
  MustWriteMap.getParameters(ParameterVector);
  ParameterSet.insert(ParameterVector.begin(), ParameterVector.end());
  ParameterVector.clear();
  MustReadMap.getParameters(ParameterVector);
  ParameterSet.insert(ParameterVector.begin(), ParameterVector.end());
  ParameterVector.clear();
  MayWriteMap.getParameters(ParameterVector);
  ParameterSet.insert(ParameterVector.begin(), ParameterVector.end());
  ParameterVector.clear();
  MayReadMap.getParameters(ParameterVector);
  ParameterSet.insert(ParameterVector.begin(), ParameterVector.end());
}

void PACCSummary::dump(PolyhedralValueInfo *PVI) const { return print(dbgs(), PVI); }

// ------------------------------------------------------------------------- //

PolyhedralAccessInfo::PolyhedralAccessInfo(PolyhedralValueInfo &PI,
                                           LoopInfo &LI)
    : PI(PI), LI(LI), PEBuilder(PI.getPolyhedralExpressionBuilder()) {}

PolyhedralAccessInfo::~PolyhedralAccessInfo() { releaseMemory(); }

void PolyhedralAccessInfo::releaseMemory() {
  DeleteContainerSeconds(AccessMap);
}

const PACC *PolyhedralAccessInfo::getAsAccess(Instruction &Inst, Value &Pointer,
                                              bool IsWrite, Loop *Scope) {
  PACC *&AccessPA = AccessMap[&Inst];
  if (AccessPA)
    return AccessPA;

  const PEXP *PointerPE = PI.getPEXP(&Pointer, Scope);

  SmallVector<Value *, 4> Parameters;
  PI.getParameters(PointerPE, Parameters);

  Value *PtrVal = nullptr;
  for (Value *Parameter : Parameters) {
    if (!Parameter->getType()->isPointerTy())
      continue;
    assert(!PtrVal && "Found multiple pointer values!");
    PtrVal = Parameter;
  }
  assert(PtrVal && "Did not find pointer value!");

  PVId PtrValId = PI.getParameterId(*PtrVal);

  const PEXP *PtrValPE = PI.getPEXP(PtrVal, Scope);

  PEXP *AccessPE = new PEXP(&Inst, nullptr);
  PEBuilder.assign(AccessPE, PointerPE, PtrValPE, PVAff::createSub);

  PACC::AccessKind AccKind = IsWrite ? PACC::AK_WRITE : PACC::AK_READ;

  const PEXP *Domain = PI.getDomainFor(Inst.getParent(), Scope);
  if (PI.isAffine(Domain))
    AccessPE->getPWA().intersectDomain(Domain->getDomain());

  AccessPA = new PACC(Pointer, *AccessPE, PtrValId, AccKind);
  return AccessPA;
}

const PACC *PolyhedralAccessInfo::getAsAccess(StoreInst *SI, Loop *Scope) {
  return getAsAccess(*SI, *SI->getPointerOperand(), true, Scope);
}

const PACC *PolyhedralAccessInfo::getAsAccess(LoadInst *LI, Loop *Scope) {
  return getAsAccess(*LI, *LI->getPointerOperand(), false, Scope);
}

const PACC *PolyhedralAccessInfo::getAsAccess(Instruction *Inst, Loop *Scope) {
  if (auto *LI = dyn_cast<LoadInst>(Inst))
    return getAsAccess(LI, Scope);
  if (auto *SI = dyn_cast<StoreInst>(Inst))
    return getAsAccess(SI, Scope);
  return nullptr;
}

PACCSummary *
PolyhedralAccessInfo::getAccessSummary(Function &F,
                                       PACCSummary::SummaryScopeKind Kind) {
  SmallVector<BasicBlock *, 32> Blocks;
  Blocks.reserve(F.size());
  for (BasicBlock &BB : F)
    Blocks.push_back(&BB);
  return getAccessSummary(Blocks, Kind);
}

PACCSummary *
PolyhedralAccessInfo::getAccessSummary(ArrayRef<BasicBlock *> Blocks,
                                       PACCSummary::SummaryScopeKind Kind,
                                       Loop *Scope) {

  PACCSummary::ContainsFuncTy ContainsFn = [=](Instruction *I) {
    return !Scope ||
           (Scope->contains(I) &&
            !(isa<PHINode>(I) && I->getParent() == Scope->getHeader()));
  };
  PACCSummary *PS = new PACCSummary(Kind, ContainsFn, Scope);
  SmallVector<const PACC *, 32> PACCs;

  for (auto &BB : Blocks) {
    for (auto &Inst : *BB) {

      const PACC *PA = getAsAccess(&Inst, Scope);
      if (PA)
        PACCs.push_back(PA);

      if (Inst.mayReadFromMemory())
        PA ? PS->KnownReads.push_back(PA) : PS->UnknownReads.push_back(&Inst);
      if (Inst.mayWriteToMemory())
        PA ? PS->KnownWrites.push_back(PA) : PS->UnknownWrites.push_back(&Inst);
    }
  }

  PS->finalize(PI, PACCs, Blocks.front()->getModule()->getDataLayout());
  return PS;
}

bool PolyhedralAccessInfo::hasFunctionScope(const PACC *PA) const {
  return PI.hasFunctionScope(PA->getPEXP(), false);
}

void PolyhedralAccessInfo::getParameters(const PACC *PA,
                                         SmallVectorImpl<PVId> &Values) const {
  PI.getParameters(PA->getPEXP(), Values);
}

void PolyhedralAccessInfo::getParameters(
    const PACC *PA, SmallVectorImpl<Value *> &Values) const {
  PI.getParameters(PA->getPEXP(), Values);
}

struct Expr {
  SmallVector<Expr *, 4> OperandExpressions;
  SmallDenseSet<unsigned> Opcodes;

  bool Commutative;

  enum KindTy {
    EK_ARGUMENT,
    EK_CONSTANT,
    EK_VALUE,
    EK_INSTRUCTION,
    EK_PHI,
    EK_RECURRENCE,
    EK_LOAD,
  } Kind;

  Value *Val;
  SmallPtrSet<Value *, 4> PossibleMatches;

  Expr(KindTy Kind, Value *Val) : Kind(Kind), Val(Val) {}

  Expr(KindTy Kind, ArrayRef<Expr *> OperandExpressions, ArrayRef<unsigned> Opcodes,
       bool Commutative)
      : Kind(Kind), OperandExpressions(OperandExpressions.begin(), OperandExpressions.end()),
        Commutative(Commutative), Val(nullptr) {
    assert(Kind != EK_ARGUMENT || (Opcodes.empty() && OperandExpressions.empty()));
    assert(Kind != EK_CONSTANT || (Opcodes.empty() && OperandExpressions.empty()));
    for (unsigned Opcode : Opcodes)
      this->Opcodes.insert(Opcode);
  };

  ~Expr() { DeleteContainerPointers(OperandExpressions); }

  void print(raw_ostream &OS) const {
    switch (Kind) {
    case EK_ARGUMENT:
      assert(Val);
      OS << "[A] " << Val->getName();
      break;
    case EK_VALUE:
      assert(Val);
      OS << "[V] " << Val->getName();
      break;
    case EK_CONSTANT:
      assert(Val);
      OS << "[C] " << *Val;
      break;
    case EK_LOAD:
      assert(Val);
      OS << "[L] " << Val->getName();
      break;
    case EK_PHI:
      assert(Val);
      OS << "[P] " << Val->getName();
      break;
    case EK_RECURRENCE:
      assert(OperandExpressions.size() == 0 || OperandExpressions.size() == 2);
      if (OperandExpressions.size() == 0) {
        OS << "[" << Val->getName() << "]";
        break;
      }
      OS << "rec[" << Val->getName() << "](";
      OperandExpressions[0]->print(OS);
      OS << ", ";
      OperandExpressions[1]->print(OS);
      OS << ")";
      break;
    case EK_INSTRUCTION: {
      std::string OpcodeStr = "";
      switch (*Opcodes.begin()) {
      case Instruction::Add:
      case Instruction::FAdd:
        OpcodeStr = "+";
        break;
      case Instruction::Mul:
      case Instruction::FMul:
        OpcodeStr = "*";
        break;
      case Instruction::SDiv:
      case Instruction::UDiv:
      case Instruction::FDiv:
        OpcodeStr = "/";
        break;
      case Instruction::SRem:
      case Instruction::URem:
      case Instruction::FRem:
        OpcodeStr = "%";
        break;
      default:
        OpcodeStr = "@";
      }
      assert(!OperandExpressions.empty());
      OS << "(";
      OperandExpressions[0]->print(OS);
      for (unsigned u = 1, e = OperandExpressions.size(); u < e; u++) {
        OS << " " << OpcodeStr << " ";
        OperandExpressions[u]->print(OS);
      }
      OS << ")";
    }
    }
  }
  void dump() const { print(dbgs()); }

  bool matches(Value *V) {
    if (Val && V == Val) {
      assert(Kind == EK_VALUE || Kind == EK_INSTRUCTION || Kind == EK_ARGUMENT);
      PossibleMatches.insert(Val);
      return true;
    }
    if (Instruction *I = dyn_cast<Instruction>(V))
      return matches(I);
    if (!OperandExpressions.empty() || !Opcodes.empty())
      return false;
    if (Kind == EK_CONSTANT && isa<Constant>(V)) {
      PossibleMatches.insert(V);
      return true;
    }
    if (Kind == EK_ARGUMENT && isa<Argument>(V)) {
      PossibleMatches.insert(V);
      return true;
    }
    if (!Val && Kind == EK_VALUE) {
      PossibleMatches.insert(V);
      return true;
    }
    return false;
  }

  static void collectOperands(Value *CurV, unsigned OpcodeI,
                              SmallVectorImpl<Value *> &OperandsI) {
    Instruction *CurI = dyn_cast<Instruction>(CurV);
    if (!CurI || CurI->getOpcode() != OpcodeI) {
      OperandsI.push_back(CurV);
      return;
    }
    for (auto &CurOperand : CurI->operands())
      return collectOperands(CurOperand, OpcodeI, OperandsI);
  }

  bool matches(Instruction *I) {
    if (Kind == EK_CONSTANT || Kind == EK_ARGUMENT)
      return false;
    if (Val && I == Val) {
      assert(Kind == EK_VALUE || Kind == EK_INSTRUCTION);
      PossibleMatches.insert(I);
      return true;
    }
    if (!Val && OperandExpressions.empty() && Opcodes.empty() &&
        (Kind == EK_VALUE || Kind == EK_INSTRUCTION)) {
      PossibleMatches.insert(I);
      return true;
    }

    unsigned NumRequiredOperands = OperandExpressions.size();
    if (I->getNumOperands() > NumRequiredOperands)
      return false;

    unsigned OpcodeI = I->getOpcode();
    if (!Opcodes.count(OpcodeI))
      return false;

    using OperandsVecTy = SmallVector<Value*, 4>;
    OperandsVecTy OperandsI;

    for (auto &Operand : I->operands())
      collectOperands(Operand, OpcodeI, OperandsI);

    if (OperandsI.size() < NumRequiredOperands)
      return false;

    SmallVector<SmallDenseSet<unsigned, 4>, 4> OperandMatches;
    OperandMatches.resize(NumRequiredOperands);
    for (unsigned OpIdx = 0; OpIdx < NumRequiredOperands; OpIdx++) {
      Expr *OperandExpression = OperandExpressions[OpIdx];
      bool MatchesFirst = OperandExpression->matches(OperandsI[OpIdx]);
      if (!Commutative && !MatchesFirst)
        return false;
      if (MatchesFirst)
        OperandMatches[OpIdx].insert(OpIdx);
      if (!Commutative)
        continue;
      for (unsigned OpIdx2 = 0; OpIdx2 < NumRequiredOperands; OpIdx2++) {
        if (OpIdx2 != OpIdx && OperandExpression->matches(OperandsI[OpIdx2]))
          OperandMatches[OpIdx2].insert(OpIdx2);
      }
    }

    if (!Commutative) {
      PossibleMatches.insert(I);
      return true;
    }

    // TODO
    for (unsigned OpIdx = 0; OpIdx < NumRequiredOperands; OpIdx++)
      if (!OperandMatches[OpIdx].count(OpIdx)) {
        return false;
      }

    PossibleMatches.insert(I);
    return true;
  }
};

void PolyhedralAccessInfo::detectKnownComputations(Function &F) {
}

void PolyhedralAccessInfo::extractComputations(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (!isa<StoreInst>(I))
        continue;
      StoreInst *SI = cast<StoreInst>(&I);

      SmallVector<LoadInst *, 8> Loads;
      SmallVector<PHINode *, 8> PHIs;
      DenseMap<Value *, Expr *> Value2ExprMap;
      Value *ExprV = SI->getValueOperand();

      std::function<Expr *(Value *)> ExprExtractor = [&](Value *V) {
        if (Expr *E = Value2ExprMap.lookup(V))
          return E;

        if (isa<Constant>(V)) {
          Expr *E = new Expr(Expr::EK_CONSTANT, V);
          Value2ExprMap[V] = E;
          return E;
        }

        Instruction *I = dyn_cast<Instruction>(V);
        if (!I) {
          Expr *E = new Expr(Expr::EK_VALUE, V);
          Value2ExprMap[V] = E;
          return E;
        }

        if (PHINode *PHI = dyn_cast<PHINode>(I)) {
          if (PHI->getNumOperands() == 1)
            return ExprExtractor(PHI->getOperand(0));
          if (PHI->getNumOperands() == 2 && LI.isLoopHeader(PHI->getParent())) {
            unsigned LoopIdx = LI.getLoopFor(PHI->getParent())
                                       ->contains(PHI->getIncomingBlock(0))
                                   ? 0
                                   : 1;
            Expr *StartE = ExprExtractor(PHI->getOperand(1 - LoopIdx));
            Value2ExprMap[PHI] = new Expr(Expr::EK_RECURRENCE, PHI);
            Expr *LoopE = ExprExtractor(PHI->getOperand(LoopIdx));
            Expr *E = new Expr(Expr::EK_RECURRENCE, {StartE, LoopE}, {}, false);
            E->Val = PHI;
            Value2ExprMap[PHI] = E;
            PHIs.push_back(PHI);
            return E;
          }

          Expr *E = new Expr(Expr::EK_PHI, I);
          Value2ExprMap[V] = E;
          return E;
        }

        if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
          Loads.push_back(LI);
          Expr *E = new Expr(Expr::EK_LOAD, I);
          Value2ExprMap[V] = E;
          return E;
        }

        SmallVector<unsigned, 4> Opcodes;
        switch (I->getOpcode()) {
        case Instruction::Add:
        case Instruction::FAdd:
          Opcodes.push_back(Instruction::Add);
          Opcodes.push_back(Instruction::FAdd);
          break;
        case Instruction::Mul:
        case Instruction::FMul:
          Opcodes.push_back(Instruction::Mul);
          Opcodes.push_back(Instruction::FMul);
          break;
        default:
          Opcodes.push_back(I->getOpcode());
        }

        SmallVector<Expr *, 4> OperandExpressions;
        for (auto &Operand : I->operands()) {
          Expr *OperandExpr = ExprExtractor(Operand);
          if (std::all_of(Opcodes.begin(), Opcodes.end(), [&](unsigned Opcode) {
                return OperandExpr->Opcodes.count(Opcode);
              }))
            OperandExpressions.append(OperandExpr->OperandExpressions.begin(),
                                      OperandExpr->OperandExpressions.end());
          else
            OperandExpressions.push_back(OperandExpr);
        }

        Expr *E = new Expr(Expr::EK_INSTRUCTION, OperandExpressions, Opcodes,
                           I->isCommutative());
        Value2ExprMap[V] = E;
        return E;
      };

      Expr *E = ExprExtractor(ExprV);

      SmallVector<std::pair<PHINode *, const PEXP *>, 32> Recurrences;
      SmallVector<const PACC *, 32> PACCs;
      PACCs.push_back(getAsAccess(SI));
      size_t MaxLoadName = 4;
      bool ValidLoads = true;
      for (LoadInst *LI : Loads) {
        if (!LI->hasName())
          LI->setName("L");
        MaxLoadName = std::max(MaxLoadName, LI->getName().size());
        const PACC *LIPA = getAsAccess(LI);
        if (!LIPA) {
          ValidLoads = false;
          break;
        }
        PACCs.push_back(LIPA);
      }
      bool ValidRecurrences = true;
      for (PHINode *PHI : PHIs) {
        if (!PHI->hasName())
          PHI->setName("P");
        const PEXP *PHIDomainPE = PI.getDomainFor(PHI->getParent());
        if (!PHIDomainPE || !PI.isAffine(PHIDomainPE)) {
          ValidRecurrences = false;
          break;
        }
        Recurrences.push_back({PHI, PHIDomainPE});
      }

      if (!ValidLoads || !ValidRecurrences) {
        //delete E;
        continue;
      }

      PACCSummary *PS =
          new PACCSummary(PACCSummary::SSK_COMPLETE,
                          [](Instruction *) { return true; }, nullptr);
      PS->finalize(PI, PACCs, DL);

      for (auto &ArrayInfoIt : PS->ArrayInfoMap) {
        for (auto &MultiDimAccessIt : ArrayInfoIt.second->AccessMultiDimMap) {
          Value *AccessInst = MultiDimAccessIt.first->getPEXP()->getValue();
        }
      }
      //delete E;
      delete PS;
    }
  }
}

void PolyhedralAccessInfo::print(raw_ostream &OS) const {}

// ------------------------------------------------------------------------- //

char PolyhedralAccessInfoWrapperPass::ID = 0;

void PolyhedralAccessInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<PolyhedralValueInfoWrapperPass>();
  AU.setPreservesAll();
}

void PolyhedralAccessInfoWrapperPass::releaseMemory() {
  delete PAI;

  F = nullptr;
  PAI = nullptr;
}

bool PolyhedralAccessInfoWrapperPass::runOnFunction(Function &F) {

  this->F = &F;

  PAI = new PolyhedralAccessInfo(
      getAnalysis<PolyhedralValueInfoWrapperPass>().getPolyhedralValueInfo(),
      getAnalysis<LoopInfoWrapperPass>().getLoopInfo());

  PAI->detectKnownComputations(F);

  PAI->extractComputations(F);

  return false;
}

void PolyhedralAccessInfoWrapperPass::print(raw_ostream &OS,
                                            const Module *) const {
  PACCSummary *PS = PAI->getAccessSummary(*F, PACCSummary::SSK_COMPLETE);
  NVVMRewriter<PVMap, /* UseGlobalIdx */ false> CudaRewriter;
  PS->rewrite(CudaRewriter);
  PS->print(OS, &PAI->getPolyhedralValueInfo());

}

FunctionPass *llvm::createPolyhedralAccessInfoWrapperPass() {
  return new PolyhedralAccessInfoWrapperPass();
}

void PolyhedralAccessInfoWrapperPass::dump() const {
  return print(dbgs(), nullptr);
}

AnalysisKey PolyhedralAccessInfoAnalysis::Key;

PolyhedralAccessInfo
PolyhedralAccessInfoAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  auto &PI = AM.getResult<PolyhedralValueInfoAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  return PolyhedralAccessInfo(PI, LI);
}

INITIALIZE_PASS_BEGIN(PolyhedralAccessInfoWrapperPass, "polyhedral-access-info",
                      "Polyhedral value analysis", false, true);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(PolyhedralValueInfoWrapperPass);
INITIALIZE_PASS_END(PolyhedralAccessInfoWrapperPass, "polyhedral-access-info",
                    "Polyhedral value analysis", false, true)
