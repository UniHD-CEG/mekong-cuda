/*******************************************************************************
 * Cuda Kernel Subgrid Transformations
 *
 * This pass creates a new instance of all host-callable CUDA Kernels named
 * "__" + original + "_subgrid" that has an additional argument of the following
 * type:
 *
 * struct SubgridSpec {
 *   int64_t zmin, zmax, ymin, ymax, xmin, xmax;
 * }
 *
 * These minima and maxima correspond to the minimum and maximum block ID of
 * the subgrid the should execute on.
 *
 * In the clone of the original kernel, occurences of special registers are
 * replaces according to the following rules:
 *
 * 1. blockIdx.{w} -> {w}min + blockIdx.{w},  w \in {z, y, x}
 * 2. gridDim.{w} -> {w}max,  w \in {z, y, x}
 *
 * Rule 1 is recursive and recursive application must be avoided.
 *
 * Cloned kernel allows executing an arbitrary rectangular subsection of the
 * cuda thread grid by carefully choosing the launch grid and the subgrid
 * specification.
 *
 */

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
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Debug.h"
#include <cxxabi.h>

#include <map>

#define DEBUG_TYPE "me-kernel-subgrid"


namespace llvm {

static std::string demangle(std::string name) {
    // demangle according to clang mangling rules
    int status = -1;
    std::unique_ptr<char, void(*)(void*)> res { abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free };
    std::string result = (status == 0) ? res.get() : std::string(name);
    // drop parameter list
    return result.substr(0, result.find("("));
}

struct SubgridSpec {
  int64_t zmin, zmax, ymin, ymax, xmin, xmax;
};

// Specialize TypeBuilder so that it supports our type
template<bool xcompile> class TypeBuilder<SubgridSpec, xcompile> {
public:
  static StructType *get(LLVMContext &C) {
    return StructType::get(
        TypeBuilder<int64_t, xcompile>::get(C),
        TypeBuilder<int64_t, xcompile>::get(C),
        TypeBuilder<int64_t, xcompile>::get(C),
        TypeBuilder<int64_t, xcompile>::get(C),
        TypeBuilder<int64_t, xcompile>::get(C),
        TypeBuilder<int64_t, xcompile>::get(C));
  }
};

/** List all host-callable CUDA kernels
 * @param  M  Module to search in
 * @return  List of all host callable CUDA Kernels
 */
SmallVector<Function*, 4> getKernels(const Module *M) {
  const auto* NamedMD = M->getNamedMetadata("nvvm.annotations");
  if (NamedMD == nullptr) { // no kernel info at all (not nvvm)
    return {};
  }

  SmallVector<Function*, 4> result;
  for (const auto* OP : NamedMD->operands()) {
    assert(OP->getNumOperands() > 0 && "nvvm.annotations Metadata needs at least one argument");
    const auto* OPFirstOP = OP->getOperand(0).get();
    if (OPFirstOP == nullptr) continue;
    const auto* ValueMD = dyn_cast<ValueAsMetadata>(OPFirstOP);
    if (ValueMD == nullptr) continue;
    if (auto* OPFirstOPF = dyn_cast<Function>(ValueMD->getValue())) {
      result.push_back(OPFirstOPF);
    }
  }
  return result;
}

struct MeKernelSubgrid : public ModulePass {
  static char ID;
  MeKernelSubgrid() : ModulePass(ID) {}

  typedef TypeBuilder<SubgridSpec, false> SubgridSpecT;
  Type *subgridSpecT;

  SmallVector<std::pair<std::string, int>, 4> updatedKernels;

  Value *loadOrExtract(IRBuilder<> &IRB, Value* StructVal, unsigned int Idx, const Twine &Name) {
    if (StructVal->getType()->isStructTy()) {
      return IRB.CreateExtractValue(StructVal, Idx, Name);
    } else if (StructVal->getType()->isPointerTy()) { 
      Value* GEP = IRB.CreateStructGEP(nullptr, StructVal, Idx);
      return IRB.CreateLoad(GEP, Name);
    } else {
      report_fatal_error("loadOrExtract requires struct or pointer to struct");
    }
    return nullptr;
  }

  /** Clone function OF but add an additional parameter of type SubgridSpec at
   * the end.
   *
   * @param  OF  function to clone
   * @return The cloned function
   */
  Function *cloneFunctionWithSubgridSpec(Function *OF, const Twine &Name) {
    // Piece together new type as: <old>(<old> ... , SubgridSpec)
    auto *OT = OF->getFunctionType();
    auto *RType = OT->getReturnType();
    SmallVector<Type*, 6> ParamTypes(OT->params().begin(), OT->params().end());
    ParamTypes.push_back(subgridSpecT->getPointerTo());

    FunctionType *NewFType = FunctionType::get(RType, ParamTypes, false);

    // reuse function if already found, otherwise create
    Module *M = OF->getParent();
    Function *F = M->getFunction(Name.getSingleStringRef());
    if (F != nullptr) {
      F->deleteBody();
    } else {
      F = cast<Function>(M->getOrInsertFunction(Name.getSingleStringRef(), NewFType));
    } 


    ValueToValueMapTy VM;
    auto *OArgsIt = OF->arg_begin();
    auto *ArgsIt = F->arg_begin();
    while (OArgsIt != OF->arg_end()) {
      VM.insert(std::make_pair(OArgsIt, ArgsIt));
      OArgsIt++;
      ArgsIt++;
    }

    SmallVector<ReturnInst*,4> _;
    CloneFunctionInto(F, OF, VM, true, _);

    Argument *SpecArg = (F->arg_end()-1);
    SpecArg->addAttr(Attribute::ByVal);
    SpecArg->addAttr(Attribute::NoCapture);

    return F;
  }

  /** Apply transformations to CUDA special registers according to:
   * 1. blockIdx.{w} -> {w}min + blockIdx.{w}  ,  w \in {z, y, x}
   * 2. gridDim.{w} -> {w}max  ,  w \in {z, y, x}
   *
   * Two phased algorithm. The first phase collects all intrinsic instructions
   * in the function. The second phase iterates through the found intrinsics
   * and applys the transformation rules if applicable. The separation avoids
   * recursive application of rule 1.
   *
   * @param  F  The function the rules should be applied to
   * @return Number of applications
   */
  int applyGridTransformations(Function *F) {
    int applications = 0;
    SmallVector<IntrinsicInst*, 4> candidates;

    for (auto &I : instructions(F)) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        candidates.push_back(II);
      }
    }

    IRBuilder<> IRB(F->getEntryBlock().getFirstNonPHIOrDbg());

    // Convenience look up map that allows combining instances of the same rule
    Argument* SpecArg = (F->arg_end()-1);
    std::map<Intrinsic::ID, unsigned int> Indices = {
      {Intrinsic::nvvm_read_ptx_sreg_ctaid_z, 0},
      {Intrinsic::nvvm_read_ptx_sreg_ctaid_y, 2},
      {Intrinsic::nvvm_read_ptx_sreg_ctaid_x, 4},
      {Intrinsic::nvvm_read_ptx_sreg_nctaid_z, 1},
      {Intrinsic::nvvm_read_ptx_sreg_nctaid_y, 3},
      {Intrinsic::nvvm_read_ptx_sreg_nctaid_x, 5}
    };
    // Pretty variable names for debugging
    const char *Names[] = {"subgrid.zmin", "subgrid.zmax", "subgrid.ymin",
      "subgrid.ymax", "subgrid.xmin", "subgrid.xmax"};
    const char *Dims[] = {"z", "z", "y", "y", "x", "x"};

    for (auto *I : candidates) {
      Intrinsic::ID ID = I->getIntrinsicID();

      // Rule 1. : blockIdx.{w} -> {w}min + blockIdx.{w}  ,  w \in {z, y, x}
      if (ID == Intrinsic::nvvm_read_ptx_sreg_ctaid_z ||
          ID == Intrinsic::nvvm_read_ptx_sreg_ctaid_y ||
          ID == Intrinsic::nvvm_read_ptx_sreg_ctaid_x) {
        Instruction* Clone = I->clone();
        Clone->insertAfter(I);
        IRB.SetInsertPoint(Clone->getNextNode());
        // upcast
        Value* Cast = IRB.CreateIntCast(Clone, IRB.getInt64Ty(), true);
        unsigned int Idx = Indices[ID];
        Value* Offset = loadOrExtract(IRB, SpecArg, Idx, Names[Idx]);
        Value* New = IRB.CreateAdd(Offset, Cast, StringRef("subgridIdx.") + Dims[Idx]);

        // Cast down for compatibility
        Value* NewCast = IRB.CreateIntCast(New, I->getType(), false);
        I->replaceAllUsesWith(NewCast);
        I->setName(I->getName()+".replaced");
        applications += 1;
        continue;
      }

      // Rule 2. : gridDim.{w} -> {w}max  ,  w \in {z, y, x}
      if (ID == Intrinsic::nvvm_read_ptx_sreg_nctaid_z ||
          ID == Intrinsic::nvvm_read_ptx_sreg_nctaid_y ||
          ID == Intrinsic::nvvm_read_ptx_sreg_nctaid_x) {
        IRB.SetInsertPoint(I->getNextNode());
        unsigned int Idx = Indices[ID];
        Value* New = loadOrExtract(IRB, SpecArg, Idx, StringRef("subgridDim.") + Dims[Idx]);

        // Cast down for compatibility
        Value* NewCast = IRB.CreateIntCast(New, I->getType(), false);
        I->replaceAllUsesWith(NewCast);
        I->setName(I->getName()+".replaced");
        applications += 1;
        continue;
      }
    }

    return applications;
  }

  bool runOnModule(Module &M) override {
    updatedKernels.clear();
    if (M.getTargetTriple() != "nvptx64-nvidia-cuda" &&
        M.getTargetTriple() != "nvptx-nvidia-cuda") {
      return false;
    }

    subgridSpecT = SubgridSpecT::get(M.getContext());

    for (auto *K : getKernels(&M)) {
      const auto Name = K->getName();
      // skip already split kernels
      if (Name.startswith("__") && Name.endswith("_subgrid")) {
        continue;
      }
      const auto Demangled = demangle(Name);
      const auto NewName = "__" + Demangled + "_subgrid";
      auto *cloned = cloneFunctionWithSubgridSpec(K, NewName);
      int replacements = applyGridTransformations(cloned);
      updatedKernels.push_back(std::make_pair(NewName, replacements));
    }
    return true;
  }

  void print(raw_ostream &OS, const Module *M) const override {
    OS << "Partitioned Kernels:\n";
    for (auto& UK : updatedKernels) {
      OS << "  " << UK.first << " (" << UK.second << " intrinsic substitutions)\n";
    }
  }

  void releaseMemory() override {
    updatedKernels.clear();
  }

  void dump() const {
    print(dbgs(), nullptr);
  }

  void getAnalysisUsage(AnalysisUsage &Info) const override {
  }
};

char MeKernelSubgrid::ID = 0;

}

// Pass registration

using namespace llvm;

//static RegisterPass<MeKernelSubgrid> X("me-kernel-subgrid", "Mekong CUDA Kernel Subgrid Transformation", false, false);
Pass *mekong::createMeKernelSubgrid() {
  return new MeKernelSubgrid();
}

INITIALIZE_PASS_BEGIN(MeKernelSubgrid, "me-kernel-subgrid",
                      "Mekong CUDA Kernel Subgrid Transformation", false, false);
INITIALIZE_PASS_END(MeKernelSubgrid, "me-kernel-subgrid",
                      "Mekong CUDA Kernel Subgrid Transformation", false, false)
