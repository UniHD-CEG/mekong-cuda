//==========--- MeKernelAnalysis.cpp -- Mekong Kernel Analysis ---*- C++ -*-============//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mekong/MeModel.h"
#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"

#include "llvm/Analysis/PolyhedralAccessInfo.h"
#include "llvm/Analysis/PolyhedralValueInfo.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"

#include <memory>
#include <string>
#include <cxxabi.h>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>

using namespace llvm;

#define DEBUG_TYPE "me-kernel-analysis"

namespace llvm {

static std::string demangle(std::string name) {
    // demangle according to clang mangling rules
    int status = -1;
    std::unique_ptr<char, void(*)(void*)> res { abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free };
    std::string result = (status == 0) ? res.get() : std::string(name);
    // drop parameter list
    return result.substr(0, result.find("("));
}

using ArrayInfo = llvm::PACCSummary::ArrayInfo;

/** Check if function is a kernel
 * @param  F  Function to check
 * @return true if F is a kernel, false otherwise
 */
bool isKernel(const Function &F) {
  const auto* M = F.getParent();
  const auto* NamedMD = M->getNamedMetadata("nvvm.annotations");
  if (NamedMD == nullptr) { // no kernel info at all (not nvvm)
    return false;
  }
  for (const auto* OP : NamedMD->operands()) {
    assert(OP->getNumOperands() > 0 && "nvvm.annotations Metadata needs at least one argument");
    const auto* OPFirstOP = OP->getOperand(0).get();
    if (OPFirstOP == nullptr) {
      continue;
    }
    const auto* ValueMD = dyn_cast<ValueAsMetadata>(OPFirstOP);
    if (ValueMD == nullptr) {
      continue;
    }
    const auto* OPFirstOPF = dyn_cast<Function>(ValueMD->getValue());
    if (OPFirstOPF == &F) {
      return true;
    }
  }
  return false;
}


/** Retrieves an annotation for a Value that matches the given prefix.  The
 * annotation must contain the words in the prefix in order and separated by
 * one or more whitespace characters. The value returned does not contain the
 * prefix.
 */
StringRef getPrefixedGlobalAnnotation(const GlobalValue* V, ArrayRef<StringRef> Prefix) {
  const auto *M = V->getParent();

  // Taken and cleaned up from:
  // https://gist.github.com/serge-sans-paille/c57d0791c7a9dbfc76e5c2f794e650b4

  // the first operand holds the metadata
  GlobalVariable* GA = M->getGlobalVariable("llvm.global.annotations");
  if (!GA) return "";

  for (Value *AOp : GA->operands()) {
    // all metadata are stored in an array of struct of metadata
    ConstantArray *CA = dyn_cast<ConstantArray>(AOp);
    if (!CA) continue;

    // so iterate over the operands
    for (Value *CAOp : CA->operands()) {
      // get the struct, which holds a pointer to the annotated function
      // as first field, and the annotation as second field
      ConstantStruct *CS = dyn_cast<ConstantStruct>(CAOp);
      if (!CS) continue;

      if (CS->getNumOperands() < 2) continue;

      GlobalValue* AnnotatedValue = cast<GlobalValue>(CS->getOperand(0)->getOperand(0));
      if (AnnotatedValue != V) continue;

      // the second field is a pointer to a global constant Array that holds the string
      GlobalVariable *GAnn =dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0));
      if (!GAnn) continue;

      ConstantDataArray *A = dyn_cast<ConstantDataArray>(GAnn->getOperand(0));
      if (!A) continue;

      // we have the annotation! Check it's an epona annotation and process
      StringRef AS = A->getAsString().trim('\0');

      bool matches = true;
      for (auto &Word: Prefix) {
        AS = AS.ltrim();
        if (AS.startswith(Word)) {
          AS = AS.drop_front(Word.size());
        } else {
          matches = false;
          continue;
        }
      }
      if (!matches) continue;

      return AS;
    }
  }
  return "";
}

/** Try to match the filter with the function. Multiple expressions are
 * combined using OR. Expressions:
 * - "intriple:<arch>" true if "<arch>" is contained in the target triple.
 * - "iskernel:yes|no" true if the function is a kernel (or not)
 * - "<word>" true if the function name contains "<word>"
 *
 * @param  F      Function to match agains
 * @param  filter Query string
 * @return True if query matches F
 */
bool filterMatches(const Function &F, StringRef filter) {
  auto isWs = [](char c) -> bool { return c == ' ' || c == '\t' || c == '\f' || c == '\n'; };
  auto isNotWs = [](char c) -> bool { return !(c == ' ' || c == '\t' || c == '\f' || c == '\n'); };

  filter = filter.drop_while(isWs);
  if (filter.empty()) return true;

  while (!filter.empty()) {
    // ltrim
    filter = filter.drop_while(isWs);

    // take word and delete from query
    StringRef word = filter.take_while(isNotWs);
    filter = filter.drop_while(isNotWs);

    if (word.consume_front("intriple:")) {
      const auto triple = F.getParent()->getTargetTriple();
      if (triple.find(word) != StringRef::npos) {
        return true;
      }
    } else if (word.consume_front("iskernel:")) {
      bool IsKernel = isKernel(F);
      bool negate = !(word.compare_lower("yes") || word.compare_lower("true") || word.compare_lower("1"));
      return IsKernel ^ negate;
    } else if (F.getName().find(word) != StringRef::npos) {
      return true;
    }
  }

  return false;
}


/** Rename unwieldy cuda intrinsic names to something more readable. Also acts as another layer
 * of separation.
 */
__isl_give isl_map* renameCudaIntrinsics(__isl_take isl_map* map, isl_dim_type dimType) {
  const Twine pairs[][2] = {
    {"nvvm_block_offset", "boff"},
    {"nvvm_ctaid", "bid"},
    {"nvvm_tid", "tid"},
  };
  const Twine suffixes[] = {"_x", "_y", "_z"};

  for (size_t i = 0; i < sizeof(pairs)/sizeof(*pairs); ++i) {
    for (size_t j = 0; j < sizeof(suffixes)/sizeof(*suffixes); ++j) {
      std::string src = pairs[i][0].concat(suffixes[j]).str();
      // is this cuda "intrinsic" used as a dimension name?
      int idx = isl_map_find_dim_by_name(map, dimType, src.c_str());
      if (idx > -1) {
        // if so, replace
        std::string dst = pairs[i][1].concat(suffixes[j]).str();
        map = isl_map_set_dim_name(map, dimType, idx, dst.c_str());
      }
    }
  }

  return map;
}


/** Force input dims to specific format as used by our analysis. If a dimension
 * name is found in the parameters, move from there, otherwise create a new
 * dimension. Names are 
 */
__isl_give isl_map* fixInputDims(__isl_take isl_map* map) {
  const char* names[] = {
    "boff_z", "boff_y", "boff_x",
    "bid_z", "bid_y", "bid_x",
    "tid_z", "tid_y", "tid_x",
  };
  const int numNames = (int)sizeof(names)/sizeof(*names);

  for (size_t i = 0; i < numNames; ++i) {
    const char* name = names[i];
    int idx = -1;
    // try to find this dimension in input dims and reorder if necessary
    idx = isl_map_find_dim_by_name(map, isl_dim_in, name);
    if (idx > -1) {
      if (idx != (int)i) {
        // TODO: FIXME
        // can't move within same dimension type, so we move it to parameters
        // first, then back to input dims at correct position. This is a
        // terrible hack.
        map = isl_map_move_dims(map, isl_dim_param, 0, isl_dim_in, idx, 1);
        map = isl_map_move_dims(map, isl_dim_in, i, isl_dim_param, 0, 1);
      }
      continue;
    }

    // if not found in input dims, try to find in parameter dims
    idx = isl_map_find_dim_by_name(map, isl_dim_param, name);
    if (idx > -1) {
      // if name found in parameters, move to input
      map = isl_map_move_dims(map, isl_dim_in, i, isl_dim_param, idx, 1);
    } else {
      // if not found, insert new at right position and rename
      map = isl_map_insert_dims(map, isl_dim_in, i, 1);
      map = isl_map_set_dim_name(map, isl_dim_in, i, name);
    }
  }
  // project out all additional dimensions
  int numDims = isl_map_dim(map, isl_dim_in);
  map = isl_map_project_out(map, isl_dim_in, numNames, numDims - numNames);
  assert(isl_map_dim(map, isl_dim_in) == numNames);
  return map;
}

/** Intersect domain that collapsed thread blocks to a singular thread for injectivity
 * analysis.
 */
__isl_give isl_map* intersectCollapsedBlockDomain(__isl_take isl_map* M) {
  const char* domain =
  "[] -> { \
    [boff_z, boff_y, boff_x, bid_z, bid_y, bid_x, tid_z, tid_y, tid_x] : \
    tid_z = 0 and tid_y = 0 and tid_x = 0 \
  }";
  isl_ctx *Ctx = isl_map_get_ctx(M);
  isl_set *S = isl_set_read_from_str(Ctx, domain);
  M = isl_map_intersect_domain(M, S);
  M = isl_map_project_out(M, isl_dim_in, 6, 3);
  return M;
}

/** Intersect domain that collapsed thread blocks to a singular thread for injectivity
 * analysis.
 */
__isl_give isl_map* intersectBlockDomain(__isl_take isl_map* M) {
  const char* domain =
  "[bdim_z, bdim_y, bdim_x] -> { \
    [boff_z, boff_y, boff_x, bid_z, bid_y, bid_x, tid_z, tid_y, tid_x] : \
      0 <= tid_z < bdim_z and \
      0 <= tid_y < bdim_y and \
      0 <= tid_x < bdim_x \
  }";
  isl_ctx *Ctx = isl_map_get_ctx(M);
  isl_set *S = isl_set_read_from_str(Ctx, domain);
  M = isl_map_intersect_domain(M, S);
  M = isl_map_project_out(M, isl_dim_in, 6, 3);
  return M;
}

/** Apply domain that models the GPU execution grid and project out (constrained)
 * local thread ids to make thread blocks atomic.
 */
__isl_give isl_map* intersectGridDomain(__isl_take isl_map* M) {
  const char* domain =
  "[boffmin_z, boffmax_z, boffmin_y, boffmax_y, boffmin_x, boffmax_x, \
    bidmin_z, bidmax_z, bidmin_y, bidmax_y, bidmin_x, bidmax_x, \
    bdim_z, bdim_y, bdim_x] -> { \
      [boff_z, boff_y, boff_x, bid_z, bid_y, bid_x, tid_z, tid_y, tid_x] : \
      boffmin_z <= boff_z < boffmax_z and \
      boffmin_y <= boff_y < boffmax_z and \
      boffmin_x <= boff_x < boffmax_z and \
      bidmin_z <= bid_z < bidmax_z and \
      bidmin_y <= bid_y < bidmax_z and \
      bidmin_x <= bid_x < bidmax_z and \
      0 <= tid_z < bdim_z and \
      0 <= tid_y < bdim_y and \
      0 <= tid_x < bdim_x \
    }";
  isl_ctx *Ctx = isl_map_get_ctx(M);
  isl_set *S = isl_set_read_from_str(Ctx, domain);
  M = isl_map_intersect_domain(M, S);
  M = isl_map_project_out(M, isl_dim_in, 4, 2);
  return M;
}

/** Cull input dimensions that are not used in the calculation of an output dimsions
 */
__isl_give isl_map* cullInputDims(__isl_take isl_map* M) {
  isl_map* M2 = isl_map_copy(M);
  int numOutDims = isl_map_dim(M, isl_dim_out);
  int numInputDims = isl_map_dim(M, isl_dim_in);

  // drop all constraints not involving output dims to identify "unused" input dims
  M2 = isl_map_drop_constraints_not_involving_dims(M2, isl_dim_out, 0, numOutDims);

  // iterate + delete in reverse, so indices don't change
  for (int dim = numInputDims-1; dim >= 0; --dim) {
    if (!isl_map_involves_dims(M2, isl_dim_in, dim, 1)) {
      M = isl_map_project_out(M, isl_dim_in, dim, 1);
    }
  }
  isl_map_free(M2);
  return M;
}

/** Checks for write "injectivity". Our "injectivity" differs from regular injectivity
 * in that unused dimensions are projected out.
 * Expects: M to have 9 input dimensions: boff.$w, bid.$w, tid.$w
 */ 
bool isInjectiveEnough(__isl_take isl_map* M) {
  M = intersectCollapsedBlockDomain(M);
  M = cullInputDims(M);
  bool result = isl_map_is_injective(M) && (isl_map_dim(M, isl_dim_in) > 0);
  isl_map_free(M);
  return result;
}

/** Intersect range and rename tuple ID of set if necessary.
 */
__isl_give isl_map* intersectRange(__isl_take isl_map* M, __isl_take isl_set* S) {
  const char* tupleName = isl_map_get_tuple_name(M, isl_dim_out);
  S = isl_set_set_tuple_name(S, tupleName);
  M = isl_map_intersect_range(M, S);
  return M;
}


std::string typeToString(const Type* t) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  t->print(rso);
  return rso.str();
}

struct MeKernelAnalysis : public FunctionPass {
    static char ID;
    static AnalysisKey Key;


    MeKernelAnalysis() : FunctionPass(ID) {
      kernel = nullptr;
    }

    mekong::Kernel* kernel;

    /** Run analysis on each kernel prospect
     * @param  F  Reference to kernel we want to inspect
     * @return modified?
     */
    bool runOnFunction(Function &F) override {
      const DataLayout &DL = F.getParent()->getDataLayout();

      kernel = new mekong::Kernel();

      kernel->name = demangle(F.getName());
      kernel->mangled_name = (F.getName()).str();
      kernel->partitioned_name = "__" + kernel->name + "_subgrid";

      StringRef PartitioningSuggestion = getPrefixedGlobalAnnotation(&F, {"me-partitioning"}).trim();
      if (PartitioningSuggestion != "") {
	kernel->partitioning = PartitioningSuggestion;
      } else {
	kernel->partitioning = "linear:x";
      }

      // arguments that are parameters to isl maps
      SmallSet<const Value*,4> Parameters;

      auto& PAI = getAnalysis<PolyhedralAccessInfoWrapperPass>().getPolyhedralAccessInfo();
      PACCSummary *PS = PAI.getAccessSummary(F, PACCSummary::SSK_COMPLETE);
      NVVMRewriter<PVMap, false> CudaRewriter;
      PS->rewrite(CudaRewriter);

      // collect infos about arguments
      for (auto &arg : F.args()) {
        kernel->arguments.push_back(mekong::Argument());
        mekong::Argument& kernelArg = kernel->arguments.back();

        Type* t = arg.getType();
        kernelArg.name = arg.getName();
        kernelArg.typeName = typeToString(t);
        kernelArg.isParameter = false; // we don't know yet, initialize as false

        kernelArg.isPointer = t->isPointerTy();
        if (!kernelArg.isPointer) {
          kernelArg.bitsize = DL.getTypeStoreSizeInBits(t);
          kernelArg.elementBitsize = DL.getTypeStoreSizeInBits(t);

          // rest of analysis only applies to pointers, so exit early
          continue;
        } else {
          Type* elT = dyn_cast<PointerType>(t)->getElementType();
          kernelArg.bitsize = DL.getPointerTypeSizeInBits(t);
          kernelArg.elementBitsize = DL.getTypeStoreSizeInBits(elT);
        }


        kernelArg.isReadInjective = false;
        kernelArg.isWriteInjective = false;

        const ArrayInfo* arrayInfo = PS->getArrayInfoForPointer(&arg);
        if (arrayInfo == nullptr) {
          continue;
        }

        auto readMap = PVMap(arrayInfo->MustReadMap).union_add(arrayInfo->MayReadMap);
        auto writeMap = PVMap(arrayInfo->MustWriteMap).union_add(arrayInfo->MayWriteMap);

        size_t NumParams = readMap.getNumParameters();
        for (size_t i = 0; i < NumParams; ++i) {
          Parameters.insert(readMap.getParameter(i).getPayloadAs<Value *>());
        }
        NumParams = writeMap.getNumParameters();
        for (size_t i = 0; i < NumParams; ++i) {
          Parameters.insert(writeMap.getParameter(i).getPayloadAs<Value *>());
        }

        isl_ctx *ctx = isl_ctx_alloc();

        StringRef BoundsAnnotation = getPrefixedGlobalAnnotation(&F, {"me-bounds", arg.getName()});
        isl_set *ArrayBounds = nullptr;
        if (BoundsAnnotation != "") {
          ArrayBounds = isl_set_read_from_str(ctx, BoundsAnnotation.str().c_str());
        }

        if (readMap) {
          isl_map *M = nullptr;
          StringRef ForcedMap = getPrefixedGlobalAnnotation(&F, {"me-readmap", arg.getName()}).trim();
          if (ForcedMap != "") {
            M = isl_map_read_from_str(ctx, ForcedMap.str().c_str());
          } else {
            M = isl_map_read_from_str(ctx, readMap.str().c_str());
            M = renameCudaIntrinsics(M, isl_dim_param);
          }
          M = fixInputDims(M);
          kernelArg.isReadInjective = isInjectiveEnough(isl_map_copy(M));
          M = intersectBlockDomain(M);
          M = cullInputDims(M);
          if (ArrayBounds != nullptr) {
            M = intersectRange(M, isl_set_copy(ArrayBounds));
          }
          kernelArg.readMap = isl_map_to_str(M);
          isl_map_free(M);
        }

        if (writeMap) {
          isl_map *M = nullptr;
          StringRef ForcedMap = getPrefixedGlobalAnnotation(&F, {"me-writemap", arg.getName()}).trim();
          if (ForcedMap != "") {
            M = isl_map_read_from_str(ctx, ForcedMap.str().c_str());
          } else {
            M = isl_map_read_from_str(ctx, writeMap.str().c_str());
            M = renameCudaIntrinsics(M, isl_dim_param);
          }
          M = fixInputDims(M);
          kernelArg.isWriteInjective = isInjectiveEnough(isl_map_copy(M));
          M = intersectBlockDomain(M);
          M = cullInputDims(M);
          if (ArrayBounds != nullptr) {
            M = intersectRange(M, isl_set_copy(ArrayBounds));
          }
          kernelArg.writeMap = isl_map_to_str(M);
          isl_map_free(M);
        }
        if (ArrayBounds) {
          isl_set_free(ArrayBounds);
        }
        isl_ctx_free(ctx);

        for (const auto *pexp : arrayInfo->DimensionSizes) {
          kernelArg.dimsizes.push_back(pexp->getPWA().str());
        }
      }
      // check if argument is a parameter to an isl map
      for (auto &arg : F.args()) {
        mekong::Argument& kernelArg = kernel->arguments[arg.getArgNo()];
        if (Parameters.count(&arg) > 0) {
          kernelArg.isParameter = true;
        }
      }
      return false;
    }

    const mekong::Kernel *getInfo() const {
      return kernel;
    }

    void print(raw_ostream &OS, const Module *) const override {
      OS << "mangled_name: " << kernel->mangled_name << "\n";
      OS << "partitioned_name: " << kernel->partitioned_name << "\n";
      OS << "partitioning: " << kernel->partitioning << "\n";
      for (const auto &a : kernel->arguments) {
        OS << "# Argument : " << a.name << "\n";
        OS << "  name: '" << a.name << "'\n";
        OS << "  isPointer: " << a.isPointer << "\n";
        OS << "  isParameter: " << a.isParameter << "\n";
        OS << "  bitsize: " << a.bitsize << "\n";
        OS << "  typeName: '" << a.typeName << "'\n";
        if (a.isPointer) {
          OS << "  elementBitSize: " << a.elementBitsize << "\n";
          OS << "  readMap:  '" << a.readMap << "'\n";
          OS << "  writeMap: '" << a.writeMap << "'\n";
          OS << "  isReadInjective " << a.isReadInjective << "\n";
          OS << "  isWriteInjective " << a.isWriteInjective << "\n";
          OS << "  ## Dimension sizes (" << a.dimsizes.size() << ")\n";
          for (const auto &dimsize : a.dimsizes) {
            OS << "    '" << dimsize << "'\n";
          }
        }
      }
    }

    void dump() const {
      print(dbgs(), nullptr);  
    }

    void releaseMemory() override {
      if (kernel != nullptr) {
        delete kernel;
        kernel = nullptr;
      }
    }

    /** Communicates pass dependencies to LLVM.
     * We require Polyhedral Access Info.
     */
    void getAnalysisUsage(AnalysisUsage &Info) const override {
        Info.addRequired<PolyhedralAccessInfoWrapperPass>();
        Info.setPreservesAll();
    }
};

char MeKernelAnalysis::ID = 0;
AnalysisKey MeKernelAnalysis::Key;
// ------------------------------------------------------------------------- //

struct MeKernelAnalysisWrapper : public ModulePass {
  static char ID;
  static AnalysisKey Key;

  MeKernelAnalysisWrapper() : ModulePass(ID) {
    modelFile = "";
  }
  MeKernelAnalysisWrapper(StringRef File) : ModulePass(ID) {
     modelFile = File;
  }

  mekong::App *app;
  std::string modelFile;

  bool runOnModule(Module &M) override {
    app = new mekong::App();

    if (M.getTargetTriple() != "nvptx64-nvidia-cuda" &&
        M.getTargetTriple() != "nvptx-nvidia-cuda")
      return false;

    for (auto &F : M) {
      if (F.isDeclaration() || !filterMatches(F, "iskernel:yes")) {
        continue;
      }
      auto &PKA = getAnalysis<MeKernelAnalysis>(F);
      const auto* kernel = PKA.getInfo();
      app->kernels.push_back(*kernel);
      PKA.releaseMemory();
    }

    if (modelFile != "") {
      serialize(mekong::ModelFile);
    }
    if (mekong::ModelFile != "") {
      serialize(mekong::ModelFile);
    }
    return false;
  }

  void serialize(StringRef Outfile) {
    std::error_code EC;
    sys::fs::OpenFlags Flags = sys::fs::F_RW | sys::fs::F_Text;
    raw_fd_ostream OS(Outfile, EC, Flags);
    app->serialize(OS);
    OS.close();
  }

  void print(raw_ostream &OS, const Module *) const override {
    app->serialize(OS);
  }

  void dump() const {
    print(dbgs(), nullptr);
  }

  void releaseMemory() override {
    delete app;
  }

  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.addRequired<MeKernelAnalysis>();
    Info.setPreservesAll();
  }
};


char MeKernelAnalysisWrapper::ID = 0;
AnalysisKey MeKernelAnalysisWrapper::Key;

}

// Pass registration

//static RegisterPass<MeKernelAnalysis> X1("me-kernel-analysis", "Mekong Kernel Analysis", true, true);
Pass *mekong::createMeKernelAnalysis() {
  return new MeKernelAnalysis();
}

INITIALIZE_PASS_BEGIN(MeKernelAnalysis, "me-kernel-analysis",
                      "Mekong Kernel Analysis", true, true);
INITIALIZE_PASS_DEPENDENCY(PolyhedralAccessInfoWrapperPass);
INITIALIZE_PASS_END(MeKernelAnalysis, "me-kernel-analysis",
                      "Mekong Kernel Analysis", true, true)

//static RegisterPass<MeKernelAnalysisWrapper> X2("me-analysis", "Mekong Application Analysis", true, true);
Pass *mekong::createMeKernelAnalysisWrapper(StringRef file) {
  return new MeKernelAnalysisWrapper(file);
}

INITIALIZE_PASS_BEGIN(MeKernelAnalysisWrapper, "me-analysis",
                      "Mekong Application Analysis", true, true);
INITIALIZE_PASS_DEPENDENCY(MeKernelAnalysis);
INITIALIZE_PASS_END(MeKernelAnalysisWrapper, "me-analysis",
                      "Mekong Application Analysis", true, true)
