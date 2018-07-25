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
#include <isl/aff.h>

using namespace llvm;

#define DEBUG_TYPE "me-kernel-analysis"

namespace llvm {

enum Partitioning {
  PART_NONE = 0,
  PART_LINEAR_X,
  PART_LINEAR_Y,
  PART_LINEAR_Z,
};

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
std::pair<bool, std::string> getPrefixedGlobalAnnotation(const GlobalValue* V, ArrayRef<StringRef> Prefix) {
  const auto *M = V->getParent();

  // Taken and cleaned up from:
  // https://gist.github.com/serge-sans-paille/c57d0791c7a9dbfc76e5c2f794e650b4

  // the first operand holds the metadata
  GlobalVariable* GA = M->getGlobalVariable("llvm.global.annotations");
  if (!GA) return std::make_pair(false, "");

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

      return std::make_pair(true, AS.trim());
    }
  }
  return std::make_pair(false, "");
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
    {"nvvm_nctaid", "gdim"},
    {"nvvm_block_offset", "boff"},
    {"nvvm_ctaid", "bid"},
    {"nvvm_ntid", "bdim"},
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

/** Ensure that all parameters are either cuda intrinsics or function arguments
 */
bool checkPVMapParameters(PVMap &M) {
  SmallVector<PVId, 16> parameters;
  M.getParameters(parameters);

  // We check for valid parameters and exit early if the parameter is one.
  // Reaching the end of a loop iterations then means that parameter is
  // not a valid parameter for us.
  for (auto parameter : parameters) {
    auto name = parameter.str();
    auto *value = parameter.getPayloadAs<Value*>();
    // check if "cuda intrinsic" as understood by NVVMRewriter
    if (name.find("nvvm_") == 0) {
      continue;
    }
    // check if value is literal function argument
    if (isa<Argument>(value)) {
      continue;
    }
    return false;
  }
  return true;
}

/** Eliminate "invalid" parameters.
 * Parameters are considered invalid if they are none of:
 * a) NVVM intrinsics (including pseudo intrinsics like block offset)
 * b) Function arguments
 */
void cleanupPVMapParameters(PVMap *M) {
  SmallVector<PVId, 16> parameters;
  M->getParameters(parameters);

  // We check for valid parameters and exit early if the parameter is one.
  // Reaching the end of a loop iterations then means that parameter is
  // not a valid parameter for us.
  for (auto parameter : parameters) {
    auto name = parameter.str();
    auto *value = parameter.getPayloadAs<Value*>();
    // check if "cuda intrinsic" as understood by NVVMRewriter
    if (name.find("nvvm_") == 0) {
      continue;
    }
    // check if value is literal function argument
    if (isa<Argument>(value)) {
      continue;
    }
    // not a valid parameter, eliminate
    M->eliminateParameter(parameter);
  }
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

/** Checks if accesses our "injective". We collapse blocks into a single thread
 * and project out unused dimensions.
 * The grid is always three-dimensional, but not every application utilizes
 * all three dimensions. These singleton dimensions are usually not used
 * in any index calculations and therefore projected out.
 * Expects preprocessed map.
 */ 
bool isInjectiveEnough(__isl_take isl_map* M) {
  // Reduce blocks to one thread in each direction.
  // There is probably a better criterion to decide whether accesses on thread block
  // level are injective.
  const char* domain = "[bdim_z, bdim_y, bdim_x] -> { : bdim_z = 1 and bdim_y = 1 and bdim_x = 1 }";
  isl_ctx *Ctx = isl_map_get_ctx(M);
  isl_set *S = isl_set_read_from_str(Ctx, domain);
  M = isl_map_intersect_params(M, S);
  bool result = isl_map_is_injective(M) && (isl_map_dim(M, isl_dim_in) > 0);
  isl_map_free(M);
  return result;
}

/** Check if the map range has a lower and upper bound.
 */
bool isMapRangeBounded(__isl_take isl_map* M) {
  isl_set *R = isl_map_range(M);
  bool bounded = true;
  int numDims = isl_set_dim(R, isl_dim_out);
  for (int i = 0; i < numDims; ++i) {
    bounded &= isl_set_dim_has_lower_bound(R, isl_dim_out, i) == isl_bool_true;
    bounded &= isl_set_dim_has_upper_bound(R, isl_dim_out, i) == isl_bool_true;
  }
  isl_set_free(R);
  return bounded;
}

/** Suggest partitioning for single write access.
 * Input map must be canonical.
 */
Partitioning guessPartitioning(__isl_keep isl_map* M) {
  int numDims = isl_map_dim(M, isl_dim_in);
  for (int i = 0; i < numDims; ++i) {
    const StringRef dimName(isl_map_get_dim_name(M, isl_dim_in, i));
    if (dimName == "boff_z" || dimName == "bid_z" || dimName == "tid_z")
      return PART_LINEAR_Z;
    else if (dimName == "boff_y" || dimName == "bid_y" || dimName == "tid_y")
      return PART_LINEAR_Y;
    else if (dimName == "boff_x" || dimName == "bid_x" || dimName == "tid_x")
      return PART_LINEAR_X;
  }
  return PART_LINEAR_X;
}

/** Combine partitioning suggestions of multiple arrays into single suggestion
 * for entire kernel.
 * Strategy: choose highest suggested dimension, default to x.
 */
Partitioning combinePartitioningSuggestions(const ArrayRef<Partitioning> suggestions) {
  Partitioning suggestion = PART_LINEAR_X;
  for (const auto part : suggestions) {
    if (part > suggestion) {
      suggestion = part;
    }
  }
  return suggestion;
}

std::string partitioningToString(Partitioning part) {
  switch (part) {
    case PART_NONE:
      return "none";
    case PART_LINEAR_X:
      return "linear:x";
    case PART_LINEAR_Y:
      return "linear:y";
    case PART_LINEAR_Z:
      return "linear:z";
  }
  return "unknown";
}

std::string typeToString(const Type* t) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  t->print(rso);
  return rso.str();
}

/** collect parameters from an access map that are arguments to the function
 * containing the access.
 */
void collectMapParameters(__isl_keep isl_map *M, SmallSet<const Value*,4> *Parameters, Function *F) {
  int numParams = isl_map_dim(M, isl_dim_param);

  for (int i = 0; i < numParams; ++i) {
    const char* name = isl_map_get_dim_name(M, isl_dim_param, i);
    for (auto const &arg : F->args()) {
      if (arg.getName() == name) {
        Parameters->insert(&arg);
        break;
      }
    }
  }
}

struct MeKernelAnalysis : public FunctionPass {
    static char ID;
    static AnalysisKey Key;

    mekong::Kernel* kernel;
    PACCSummary *PS;

    MeKernelAnalysis() : FunctionPass(ID) {
      kernel = nullptr;
      PS = nullptr;
    }

    PACCSummary* getAccessSummary(Function &F) {
      if (PS == nullptr) {
          auto& PAI = getAnalysis<PolyhedralAccessInfoWrapperPass>().getPolyhedralAccessInfo();
          PS = PAI.getAccessSummary(F, PACCSummary::SSK_COMPLETE);
          NVVMRewriter<PVMap, false> CudaRewriter;
          PS->rewrite(CudaRewriter);
      }
      return PS;
    }

    struct IslArrayInfo {
      IslArrayInfo()
      : ctx(nullptr), readMap(nullptr), writeMap(nullptr)
      {}
      ~IslArrayInfo() {
        if (readMap) isl_map_free(readMap);
        if (writeMap) isl_map_free(writeMap);
        for (auto *dim: dims) isl_pw_multi_aff_free(dim);
        if (ctx) isl_ctx_free(ctx);
      }
      isl_ctx *ctx;
      isl_map *readMap;
      isl_map *writeMap;
      SmallVector<isl_pw_multi_aff*, 4> dims;
    };

    enum MapType {
      READMAP,
      WRITEMAP,
    };

    /** Collect cleaned up info array usage.
     */
    IslArrayInfo *getArrayInfo(Argument &arg) {
      Function *F = arg.getParent();

      IslArrayInfo *result = new IslArrayInfo;

      isl_ctx *ctx = isl_ctx_alloc();
      result->ctx = ctx;

      isl_set *bounds = nullptr;

      auto BoundsAnnotation = getPrefixedGlobalAnnotation(F, {"me-bounds", arg.getName()});
      if (BoundsAnnotation.first) {
        bounds = isl_set_read_from_str(ctx, BoundsAnnotation.second.c_str());
      }

      result->readMap = getMapForArgument(arg, READMAP, {"me-readmap", arg.getName()}, ctx);
      result->writeMap = getMapForArgument(arg, WRITEMAP, {"me-writemap", arg.getName()}, ctx);
      if (bounds != nullptr) {
        if (result->readMap) {
          const char* tupleName = isl_map_get_tuple_name(result->readMap, isl_dim_out);
          isl_set *boundsCopy = isl_set_set_tuple_name(isl_set_copy(bounds), tupleName);
          result->readMap = isl_map_intersect_range(result->readMap, boundsCopy);
        }
        if (result->writeMap) {
          const char* tupleName = isl_map_get_tuple_name(result->writeMap, isl_dim_out);
          isl_set *boundsCopy = isl_set_set_tuple_name(isl_set_copy(bounds), tupleName);
          result->writeMap = isl_map_intersect_range(result->writeMap, boundsCopy);
        }
        isl_set_free(bounds);
      }

      auto DimsAnnotation = getPrefixedGlobalAnnotation(F, {"me-dims", arg.getName()});
      if (DimsAnnotation.first) {
        StringRef dims = DimsAnnotation.second;
        dims = dims.trim();
        auto isSep = [](char ch) -> bool { return ch == '|'; };
        while (dims != "") {
          StringRef dimStr = dims.take_until(isSep);
          dims = dims.substr(dimStr.size()+1).trim();
          isl_pw_multi_aff *dim = isl_pw_multi_aff_read_from_str(ctx, dimStr.str().c_str());
          if (dim == nullptr) {
            report_fatal_error((std::string("")) + "invalid dimension size: " + dimStr);
          }
          result->dims.push_back(dim);
        }
      } else {
        const ArrayInfo* arrayInfo = getAccessSummary(*F)->getArrayInfoForPointer(&arg);
        if (arrayInfo != nullptr) {
          for (const auto *pexp : arrayInfo->DimensionSizes) {
            isl_pw_multi_aff *dim = isl_pw_multi_aff_read_from_str(ctx, pexp->getPWA().str().c_str());
            result->dims.push_back(dim);
          }
        }
      }
      return result;
    }

    /** return access (read or write) as ISL map for the given arguments.
     * If provided via annotation, use the annotation, otherwise use polyhedral analysis.
     */
    __isl_give isl_map *getMapForArgument(Argument& arg, MapType type, ArrayRef<StringRef> Prefix, isl_ctx *ctx) {
      Function *F = arg.getParent();

      isl_map *M = nullptr;

      auto ForcedMap = getPrefixedGlobalAnnotation(F, Prefix);
      if (ForcedMap.first) { // Use map from annotation
        M = isl_map_read_from_str(ctx, ForcedMap.second.c_str());

      } else {
        const ArrayInfo* arrayInfo = getAccessSummary(*F)->getArrayInfoForPointer(&arg);
        if (arrayInfo == nullptr) {
          return nullptr;
        }
        PVMap map;
        if (type == READMAP) {
          map = PVMap(arrayInfo->MustReadMap).union_add(arrayInfo->MayReadMap);
        } else if (type == WRITEMAP) {
          map = PVMap(arrayInfo->MustWriteMap).union_add(arrayInfo->MayWriteMap);
        } else {
          llvm_unreachable("invalid map type");
        }
        if (!map) {
          return nullptr;
        }

        cleanupPVMapParameters(&map);
        M = isl_map_read_from_str(ctx, map.str().c_str());
        M = renameCudaIntrinsics(M, isl_dim_param);
      }

      M = fixInputDims(M);
      M = intersectBlockDomain(M);
      M = cullInputDims(M);
      return M;
    }

    mekong::Kernel* dontAnalyzeKernel(Function &F) {
      const DataLayout &DL = F.getParent()->getDataLayout();
      mekong::Kernel *kernel = new mekong::Kernel();
      kernel->name = demangle(F.getName());
      kernel->mangled_name = (F.getName()).str();
      kernel->partitioning = "none";
      for (auto &arg : F.args()) {
        kernel->arguments.push_back(mekong::Argument());
        mekong::Argument &kernelArg = kernel->arguments.back();
        Type* t = arg.getType();
        kernelArg.name = arg.getName();
        kernelArg.typeName = typeToString(t);
        kernelArg.isParameter = false;
        kernelArg.isPointer = t->isPointerTy();
        if (!kernelArg.isPointer) {
          kernelArg.bitsize = DL.getTypeStoreSizeInBits(t);
          kernelArg.elementBitsize = DL.getTypeStoreSizeInBits(t);
        } else {
          Type* elT = dyn_cast<PointerType>(t)->getElementType();
          kernelArg.bitsize = DL.getPointerTypeSizeInBits(t);
          kernelArg.elementBitsize = DL.getTypeStoreSizeInBits(elT);
        }
      }
      return kernel;
    }

    mekong::Kernel* analyzeKernel(Function &F) {
      const DataLayout &DL = F.getParent()->getDataLayout();
      mekong::Kernel *kernel = new mekong::Kernel();
      kernel->name = demangle(F.getName());
      kernel->mangled_name = (F.getName()).str();
      kernel->partitioned_name = "__" + kernel->name + "_subgrid";

      SmallVector<Partitioning,4> suggestions;

      // arguments that are parameters to isl maps
      SmallSet<const Value*,4> Parameters;

      bool splittable = true;

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

        IslArrayInfo *arrayInfo = getArrayInfo(arg);

        isl_map *M;

        M = arrayInfo->readMap;
        if (M) {
          collectMapParameters(M, &Parameters, &F);
          kernelArg.isReadInjective = isInjectiveEnough(isl_map_copy(M));
          kernelArg.isReadBounded = isMapRangeBounded(isl_map_copy(M));
          kernelArg.readMap = isl_map_to_str(M);
        }

        M = arrayInfo->writeMap;
        if (M) {
          collectMapParameters(M, &Parameters, &F);
          kernelArg.isWriteInjective = isInjectiveEnough(isl_map_copy(M));
          kernelArg.isWriteBounded = isMapRangeBounded(isl_map_copy(M));
          kernelArg.writeMap = isl_map_to_str(M);

          suggestions.push_back(guessPartitioning(M));
          splittable = splittable & kernelArg.isWriteInjective;
        }

        for (auto *pwaff : arrayInfo->dims) {
          kernelArg.dimsizes.push_back(isl_pw_multi_aff_to_str(pwaff));
        }
      }

      auto PartitioningSuggestion = getPrefixedGlobalAnnotation(&F, {"me-partitioning"});
      if (PartitioningSuggestion.first) {
        kernel->partitioning = PartitioningSuggestion.second;
      } else {
        if (splittable) {
          Partitioning suggestion = combinePartitioningSuggestions(suggestions);
          kernel->partitioning = partitioningToString(suggestion);
        } else {
          kernel->partitioning = "none";
        }
      }

      // check if argument is a parameter to an isl map
      for (auto &arg : F.args()) {
        mekong::Argument& kernelArg = kernel->arguments[arg.getArgNo()];
        if (Parameters.count(&arg) > 0) {
          kernelArg.isParameter = true;
        }
      }
      return kernel;
    }


    /** Run analysis on each kernel prospect
     * @param  F  Reference to kernel we want to inspect
     * @return modified?
     */
    bool runOnFunction(Function &F) override {
      // check for analysis bypass
      auto forceIgnore = getPrefixedGlobalAnnotation(&F, {"me-ignore"});
      if (forceIgnore.first) {
        kernel = dontAnalyzeKernel(F);
      } else {
        kernel = analyzeKernel(F);
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
      PS = nullptr;
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
    modelFile = mekong::ModelFile;
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
      serialize(modelFile);
    }

    return false;
  }

  void serialize(StringRef Outfile) {
    if (Outfile == "-") {
      app->serialize(outs());
    } else {
      std::error_code EC;
      sys::fs::OpenFlags Flags = sys::fs::F_Text;
      raw_fd_ostream OS(Outfile, EC, Flags);
      app->serialize(OS);
      OS.close();
    }

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
