//===--- PolyhedralDependenceInfo.cpp -- Polyhedral dependence analysis ---===//
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

#include "llvm/Analysis/PolyhedralDependenceInfo.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PolyhedralValueInfo.h"
#include "llvm/Analysis/PolyhedralAccessInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "polyhedral-dependence-info"

raw_ostream &llvm::operator<<(raw_ostream &OS, PDEP::DependenceKind Kind) {
  return OS;
}

void PDEP::print(raw_ostream &OS) const {
}

void PDEP::dump() const { print(dbgs()); }

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDEP *PD) {
  if (PD)
    return OS << *PD;
  return OS << "<null>";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDEP &PD) {
  PD.print(OS);
  return OS;
}

// ------------------------------------------------------------------------- //

PolyhedralDependenceInfo::PolyhedralDependenceInfo(PolyhedralAccessInfo &PAI, LoopInfo &LI)
    : PAI(PAI), LI(LI) {}

PolyhedralDependenceInfo::~PolyhedralDependenceInfo() { releaseMemory(); }

bool PolyhedralDependenceInfo::isVectorizableLoop(Loop &L) {
  errs() << "CHECK L:";
  L.dump();

  if (L.begin() != L.end()) {
    DEBUG(dbgs() << "PDI: Loop is not innermost. Not vectorizable!\n");
    return false;
  }

  SmallVector<BasicBlock *, 32> Blocks;
  Blocks.append(L.block_begin(), L.block_end());
  const PACCSummary *PS =
      PAI.getAccessSummary(Blocks, PACCSummary::SSK_COMPLETE, &L);
  PS->dump();

  if (PS->getNumUnknownReads() || PS->getNumUnknownWrites())
    return false;

  for (const auto &ArrayInfoMapIt : *PS) {
    const PACCSummary::ArrayInfo *AI = ArrayInfoMapIt.getSecond();
    PVMap WriteMap = AI->MustWriteMap;
    WriteMap.union_add(AI->MayWriteMap);
    if (!WriteMap)
      continue;

    PVMap ReadMap = AI->MustReadMap;
    ReadMap.union_add(AI->MayReadMap);
    if (!ReadMap)
      continue;

    PVId OffsetId(WriteMap, "Offset");
    PVMap OffsetMap(OffsetId, PVId(), WriteMap.getOutputId());
    WriteMap.addToOutputDimension(OffsetMap, 0);
    PVMap &ReadWriteMap = WriteMap.intersect(ReadMap);
    if (ReadWriteMap.isEmpty())
      continue;

    PVSet ConflictSet = ReadWriteMap.getParameterSet();
    ConflictSet.dropUnusedParameters();
    errs() << ConflictSet <<"\n";
    DEBUG(dbgs() << "Conflicting accesses for " << *ArrayInfoMapIt.getFirst()
                 << "\n\t => " << ReadWriteMap << "\n");
    if (!ConflictSet.hasLowerBoundForParam(OffsetId))
      return false;

    ConflictSet.minForParam(OffsetId);
    errs() << ConflictSet <<"\n";

    return false;
  }

  return false;
}


void PolyhedralDependenceInfo::releaseMemory() {
}

void PolyhedralDependenceInfo::print(raw_ostream &OS) const {
  auto Loops = LI.getLoopsInPreorder();
  for (Loop *L : Loops) {
    bool Vec =
        const_cast<PolyhedralDependenceInfo *>(this)->isVectorizableLoop(*L);
    errs() << "L: " << L->getName() << " Vec: " << Vec << "\n";
  }
}

void PolyhedralDependenceInfo::dump() const { return print(dbgs()); }

// ------------------------------------------------------------------------- //

char PolyhedralDependenceInfoWrapperPass::ID = 0;

void PolyhedralDependenceInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<PolyhedralAccessInfoWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesAll();
}

void PolyhedralDependenceInfoWrapperPass::releaseMemory() {
  delete PDI;

  F = nullptr;
  PDI = nullptr;
}

bool PolyhedralDependenceInfoWrapperPass::runOnFunction(Function &F) {

  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &PAI =
      getAnalysis<PolyhedralAccessInfoWrapperPass>().getPolyhedralAccessInfo();
  PDI = new PolyhedralDependenceInfo(PAI, LI);

  this->F = &F;
  return false;
}

void PolyhedralDependenceInfoWrapperPass::print(raw_ostream &OS,
                                                const Module *M) const {
  M->dump();
  PDI->print(OS);
}

FunctionPass *llvm::createPolyhedralDependenceInfoWrapperPass() {
  return new PolyhedralDependenceInfoWrapperPass();
}

void PolyhedralDependenceInfoWrapperPass::dump() const {
  return print(dbgs(), nullptr);
}

AnalysisKey PolyhedralDependenceInfoAnalysis::Key;

PolyhedralDependenceInfo
PolyhedralDependenceInfoAnalysis::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &PAI = AM.getResult<PolyhedralAccessInfoAnalysis>(F);
  return PolyhedralDependenceInfo(PAI, LI);
}

INITIALIZE_PASS_BEGIN(PolyhedralDependenceInfoWrapperPass,
                      "polyhedral-dependence-info",
                      "Polyhedral dependence analysis", false, true);
INITIALIZE_PASS_DEPENDENCY(PolyhedralAccessInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_END(PolyhedralDependenceInfoWrapperPass,
                    "polyhedral-dependence-info",
                    "Polyhedral dependence analysis", false, true)
