#include "mekong/MeModel.h"
#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Debug.h"

#include <map>

#define DEBUG_TYPE "me-analyze-buffers"


namespace llvm {

struct HostBuffer {
  CallInst Creation;
  SmallVector<CallInst*, 4> SourceUses;
  SmallVector<CallInst*, 4> KernelUses;
  SmallVector<Instruction*, 4> MayWrites;
  SmallVector<Instruction*, 4> MustWrites;
};

bool isCudaMallocInstance(Function *F) {
  if (F->getName() == "cudaMalloc") return true;

  auto &BBs = F->getBasicBlockList();
  if (BBs.size() != 1)
    return false;

  BasicBlock &BB = BBs.front();
  if (BB.size() != 3)
    return false;

  Instruction *Second = BB.getFirstNonPHI()->getNextNode();
  CallInst *MaybeMalloc = dyn_cast_or_null<CallInst>(Second);
  if (!MaybeMalloc)
    return false;

  return isCudaMallocInstance(MaybeMalloc->getCalledFunction());
}

// 1 = HostToDevice
// 2 = DeviceToHost
bool isCudaMemcpy(Instruction *I, int kind) {
  CallInst *Call = dyn_cast_or_null<CallInst>(I);
  if (!Call)
    return false;

  Function *Callee = Call->getCalledFunction();
  if (!Callee)
    return false;

  if (Callee->getName() != "cudaMemcpy")
    return false;

  ConstantInt *Kind = dyn_cast_or_null<ConstantInt>(Call->getOperand(3));
  if (!Kind)
    return false;

  if (!Kind->equalsInt(kind))
    return false;

  return true;
}

bool possiblyModifies(Instruction *I, Value* Pointer) {
  StoreInst *Store = dyn_cast<StoreInst>(I);
  if (Store && Store->getPointerOperand()->stripPointerCasts() == Pointer) {
    return true;
  }

  CallInst *Call = dyn_cast<CallInst>(I);
  if (Call) {
    
    // if function does not read or write any to any of it's 
    // pointer arguments, we don't care
    Function *Fn = Call->getCalledFunction();
    //if (Fn->hasAttribute(Attribute::ReadNone) || Fn->hasAttribute(Attribute::ReadOnly))
    //  return false;

    // otherwise check if the function uses our pointer, and if it does,
    // whether it is a readnone/readonly access

    for (int i = 0; i < (int)Call->getNumArgOperands(); ++i) {
      Value* Op = Call->getOperand(i);

      // If argument is not our pointer, we don't care
      if (Op->stripPointerCasts() != Pointer)
        continue;

      // If our pointer is not read or written, we don't care
      if (Fn->hasParamAttribute(i, Attribute::ReadNone) ||
          Fn->hasParamAttribute(i, Attribute::ReadOnly))
        continue;

      // None of our 
      return true;
    }
  }
  return false;
}

struct MeAnalyzeBuffers : public ModulePass {
  static char ID;
  MeAnalyzeBuffers() : ModulePass(ID) {}

  // A Memcpy introduces potential WAR dependencies on the pointer used
  // in the memcpy after the memcpy instruction
  using Memcpy = std::pair<Value*, Instruction*>;

  SmallVector<HostBuffer, 4> HostBuffers;
  SmallVector<Memcpy, 8> Memcpies;

  bool runOnModule(Module &M) override {
    HostBuffers.clear();
    Memcpies.clear();

    SmallPtrSet<Function*, 8> mallocAliases;

    // collect cuda malloc template instances + cudaMalloc
    for (Function &F : M) {
      if (isCudaMallocInstance(&F)) {
        mallocAliases.insert(&F);
      }
    }

    for (Function &F : M) {
      //if (mallocAliases.find(&F) != mallocAliases.end()) {
      //  continue;
      //}

      // collect potentially difficult memcpies
      for (auto &I : instructions(F)) {
        if (isCudaMemcpy(&I, 1)) {
          Value* Buffer = I.getOperand(1)->stripPointerCasts();
          Memcpies.push_back(std::make_pair(Buffer, &I));
        }
      }

      // collect writes to memcpied buffers
      for (auto &I : instructions(F)) {
      }

      // collect kernel launches + arguments
    }

    return false;
  }

  void print(raw_ostream &OS, const Module *M) const override {
    OS << "Memcpies Host to Device:\n";
    for (auto &memcpy : Memcpies) {
      OS << "  " << *memcpy.first << " @ " << *memcpy.second << "\n";
    }

    //OS << "dummy\n";
  }

  void releaseMemory() override {
    HostBuffers.clear();
  }

  void dump() const {
    print(dbgs(), nullptr);
  }

  void getAnalysisUsage(AnalysisUsage &Info) const override {
  }
};

char MeAnalyzeBuffers::ID = 0;

}

// Pass registration

using namespace llvm;

static llvm::RegisterPass<MeAnalyzeBuffers>
  X("me-analyze-buffers", "Analyze Host Buffer Usage", true, true);
