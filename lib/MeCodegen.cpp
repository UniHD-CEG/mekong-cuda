/******************************************************************************
 *
 * Host code generation for Mekong using ISL objects. The code generated here
 * differs from the usual use cases of ISL code generation. Instead of
 * transforming loops and the statements contained in them by reordering etc.
 * we generate completely new code based on a descrition of the memory accesses
 * of a GPU kernel.
 *
 * The polyhedral analysis is performed in a compile path different from the
 * current one and ISL objects have are serialized for the transport between
 * compilation paths. As a consequence, ID payloads, usually holding user
 * defined data, are lost.
 *
 * For each memory access, a set of functions is created, some of which are
 * only intended as auxiliary functions to the interface and have internal
 * linkage and an "always inline" attribute. All functions belonging to a
 * memory access share a common prefix. The following listing defines the
 * suffixes used for these functions (auxiliary functions are marked by an
 * asterisk):
 *
 * "<prefix>"
 *  prototype: void(int64_t grid[], int64_t param[], callback*, void*)
 * "<prefix>_payload"
 *  prototype: void(int64_t fix[], int64_t grid[], int64_t param[], callback*, void*)
 * "<prefix>_dim" (not yet)
 *
 * User Callback:
 *  void(int64_t linearizedLower, int64_t linearizedUpper, void* user)
 *
 *
 * <prefix> = "__<kernel name>_<array index>_<read|write>"
 * Functions starting with "__" puts them in the reserved name space and avoids
 * name collisions with user named functions (or at least allows to blame the
 * user instead).
 */
#include "mekong/IslBuilder.h"
#include "mekong/MeModel.h"
#include "mekong/Passes.h"
#include "mekong/InitializePasses.h"

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/aff.h>
#include <isl/union_map.h>
#include <isl/ast.h>
#include <isl/options.h>

#include <map>

using namespace llvm;

#define DEBUG_TYPE "me-codegen"

namespace llvm {

cl::opt<bool> BuildDispatcher("me-codegen-dispatch",
    cl::desc("Build dispatcher for mekong iterators (name __me_dispatch)"),
    cl::Hidden);

enum ParamType {
  NONE,
  GRID,
  FIX,
  PARAM,
};
struct ParamLoc {
  ParamType type;
  int pos;
};
raw_ostream& operator<<(raw_ostream& OS, const ParamLoc& loc) {
  const char* PType = "NONE";
  if (loc.type == GRID) PType = "GRID";
  if (loc.type == PARAM) PType = "PARAM";
  OS << "ParamLoc{" << PType << ", " << loc.pos << "}";
  return OS;
}

ParamLoc resolveParam(const StringRef name, const mekong::Kernel &kernel) {
  if (name == "boffmin_z") return {GRID, 0};
  if (name == "boffmax_z") return {GRID, 1};
  if (name == "boffmin_y") return {GRID, 2};
  if (name == "boffmax_y") return {GRID, 3};
  if (name == "boffmin_x") return {GRID, 4};
  if (name == "boffmax_x") return {GRID, 5};
  if (name == "bidmin_z") return {GRID, 6};
  if (name == "bidmax_z") return {GRID, 7};
  if (name == "bidmin_y") return {GRID, 8};
  if (name == "bidmax_y") return {GRID, 9};
  if (name == "bidmin_x") return {GRID, 10};
  if (name == "bidmax_x") return {GRID, 11};
  if (name == "bdim_z") return {GRID, 12};
  if (name == "bdim_y") return {GRID, 13};
  if (name == "bdim_x") return {GRID, 14};
  if (name == "fix_0") return {FIX, 0}; // number of these is arbitrary, for
  if (name == "fix_1") return {FIX, 1}; // now let's assume that arrays don't
  if (name == "fix_2") return {FIX, 2}; // have more than 6 dimensions
  if (name == "fix_3") return {FIX, 3};
  if (name == "fix_4") return {FIX, 4};
  if (name == "fix_5") return {FIX, 5};
  if (name == "fix_6") return {FIX, 6};
  int pos = -1;
  for (auto &arg : kernel.arguments) {
    if (arg.isParameter) {
      ++pos;
      if (arg.name == name) {
        return {PARAM, pos};
      }
    }
  }
  return {NONE, -1};
}

ParamLoc resolveParam(__isl_keep isl_id* param, const mekong::Kernel &kernel) {
  const char* ParamName = isl_id_get_name(param);
  return resolveParam(ParamName, kernel);
}

std::string getPrefix(StringRef kernelName, int argumentIndex, StringRef mode) {
  assert((mode == "read" || mode == "write") && "Only read and write modes supported");
  std::string buf;
  raw_string_ostream ss(buf);
  ss << "__" << kernelName << "_" << argumentIndex << "_" << mode;
  return buf;
}

struct IteratorDebugInfo {
  Function *iterator;
  std::vector<std::string> paramNames;
};

struct MeCodegen : public ModulePass {
  static char ID;
  static AnalysisKey Key;
  MeCodegen() : ModulePass(ID) {
    modelFile = mekong::ModelFile;
  }
  MeCodegen(StringRef File) : ModulePass(ID) {
    modelFile = File;
  }

  std::string modelFile;
  mekong::App app;
  std::vector<std::pair<std::string, std::string>> builds_in_c;
  std::vector<IteratorDebugInfo> iteratorDebugInfo;

  // User callback, supplied externally
  //   void(int64_t linearizedLower, int64_t linearizedUpper, void* user)
  // Payload, internal use only
  //   void(int64_t fix[], int64_t grid[], int64_t param[], UserCBFType*, void*)
  // Iterator:
  //   void(int64_t grid[], int64_t param[], UserCBFType*, void*)
  //
  // We have to use typedefs here because of templates being the main abstraction.
  // in order to use them in the application later, you have to call the static
  // function get(LLVMContext&) in the types defined here.
  typedef TypeBuilder<void(int64_t,int64_t,void*), false> UserCBFType;
  typedef TypeBuilder<void(int64_t[], int64_t[], int64_t[], void(int64_t,int64_t,void*), void*), false> PayloadFType;
  typedef TypeBuilder<void(int64_t[], int64_t[], void(int64_t,int64_t,void*), void*), false> IteratorFType;
  typedef TypeBuilder<void(int,int64_t[], int64_t[], void(int64_t,int64_t,void*), void*), false> DispatcherFType;

  Value* LoadNth(IRBuilder<> *IRB, Value* Ptr, unsigned Idx, const Twine &Name = "") {
    Value* gep = IRB->CreateConstGEP1_32(Ptr, Idx);
    Value* load = IRB->CreateLoad(gep, Name);
    return load;
  }

  void addBuild(Twine name, char* build) {
    size_t len = strlen(build);
    if (build[len-1] == '\n') build[len-1] = '\0';
    builds_in_c.push_back(std::make_pair(name.str(), build));
    free(build);
  }

  /** Generate code to iterate set upper/lower
   * prototype of returned function:
   * (int64_t grid[], int64_t param[], CBF*, void*) -> void
   *
   * expected payload prototype:
   *
   */
  Function* generateIterator(const Twine &Name, Function* Payload, __isl_keep
      isl_set* Set, mekong::Kernel &K, Module &M) {
    LLVMContext &C = M.getContext();

    // reuse function if already found
    Function *F = M.getFunction(Name.getSingleStringRef());
    if (F == nullptr) {
      F = Function::Create(IteratorFType::get(C), Function::InternalLinkage, Name, &M);
    }

    auto *EntryBB = BasicBlock::Create(C, "entry", F);
    IRBuilder<> IRB(EntryBB);
    std::map<isl_id*,Value*> IdMap;

    auto *Arg = F->arg_begin();
    Value* GridPtr = Arg++;
    Value* ParamPtr = Arg++;
    Value* CallbackPtr = Arg++;
    Value* UserPtr = Arg++;
    assert(Arg == F->arg_end());
    GridPtr->setName("grid");
    ParamPtr->setName("param");
    CallbackPtr->setName("callback");
    UserPtr->setName("user");

    int NParams = isl_set_dim(Set, isl_dim_param);
    // create loads
    for (int i = 0; i < NParams; ++i) {
      isl_id* DimId = isl_set_get_dim_id(Set, isl_dim_param, i);
      const char* DimName = isl_id_get_name(DimId);
      ParamLoc Loc = resolveParam(DimName, K);
      Value* Src = nullptr;
      switch (Loc.type) {
      case GRID:
        Src = GridPtr; break;
      case PARAM:
        Src = ParamPtr; break;
      default:
        llvm_unreachable("Invalid Parameter Type");
      }
      Value* Val = LoadNth(&IRB, Src, Loc.pos, DimName);
      IdMap.insert(std::make_pair(DimId, Val));
    }

    IslBuilder ISLB;
    isl_set *SetParams = isl_set_params(isl_set_copy(Set));
    isl_ast_build *Build = isl_ast_build_from_context(SetParams);
    isl_map *SchedulePre = isl_set_identity(isl_set_copy(Set));
    isl_union_map *Schedule = isl_union_map_from_map(SchedulePre);
    
    isl_ast_node* Node = isl_ast_build_node_from_schedule_map(Build, Schedule);

    ISLB.setIRBuilder(&IRB).setTarget(Payload).setIDMap(&IdMap)
      .setTupleType(IslBuilder::TupleType::Array)
      .setExtraArgs({ GridPtr, ParamPtr, CallbackPtr, UserPtr });
    ISLB.buildNode(Node);
    addBuild(Name + " : schedule", isl_ast_node_to_C_str(Node));

    isl_ast_node_free(Node);
    isl_ast_build_free(Build);

    IRB.CreateRetVoid();

    for (auto &it : IdMap) {
      isl_id_free(it.first);
    }

    return F;
  }


  /* Generates payload for iterator above.
   *
   */
  Function* generatePayload(const Twine &Name, __isl_keep isl_pw_aff *lower, __isl_keep isl_pw_aff *upper,
      ArrayRef<__isl_keep isl_pw_aff*> dimSizes, mekong::Kernel &K, Module &M) {
    LLVMContext &C = M.getContext();

    auto *F = Function::Create(PayloadFType::get(C), Function::InternalLinkage, Name, &M);

    auto *EntryBB = BasicBlock::Create(C, "entry", F);

    auto NDimSizes = (int)dimSizes.size();

    auto *Arg = F->arg_begin();
    Value* FixPtr = Arg++;
    Value* GridPtr = Arg++;
    Value* ParamPtr = Arg++;
    Value* Callback = Arg++;
    Value* AuxPtr = Arg++;
    assert(Arg == F->arg_end());
    FixPtr->setName("fix");
    GridPtr->setName("grid");
    ParamPtr->setName("param");
    Callback->setName("user_cb");
    AuxPtr->setName("user_aux");

    // compile list of all parameters
    isl_set *SetParams = isl_pw_aff_params(isl_pw_aff_copy(lower));
    SetParams = isl_set_union(SetParams, isl_pw_aff_params(isl_pw_aff_copy(upper)));
    for (int i = 0; i < NDimSizes; ++i) {
      SetParams = isl_set_union(SetParams, isl_pw_aff_params(isl_pw_aff_copy(dimSizes[i])));
    }

    IRBuilder<> IRB(EntryBB);
    std::map<isl_id*,Value*> IdMap;

    // load all parameters
    int NParams = isl_set_dim(SetParams, isl_dim_param);
    // create loads
    for (int i = 0; i < NParams; ++i) {
      isl_id* DimId = isl_set_get_dim_id(SetParams, isl_dim_param, i);
      const char* DimName = isl_id_get_name(DimId);
      ParamLoc Loc = resolveParam(DimName, K);
      Value* Src = nullptr;
      switch (Loc.type) {
      case GRID:
        Src = GridPtr; break;
      case PARAM:
        Src = ParamPtr; break;
      case FIX:
        Src = FixPtr; break;
      default:
        llvm_unreachable("Invalid Parameter Type");
      }
      Value* Val = LoadNth(&IRB, Src, Loc.pos, DimName);
      IdMap.insert(std::make_pair(DimId, Val));
    }

    IslBuilder ISLB;
    ISLB.setIRBuilder(&IRB).setIDMap(&IdMap).setTupleType(IslBuilder::TupleType::Array);
    isl_ast_build *Build = isl_ast_build_from_context(SetParams);

    isl_ast_expr* expr = nullptr;

    // generate code for lower bound (inclusive)
    expr = isl_ast_build_expr_from_pw_aff(Build, isl_pw_aff_copy(lower));
    Value* lowerVal = ISLB.buildExpr(expr);
    addBuild(Name + " : lower bound", isl_ast_expr_to_C_str(expr));
    lowerVal->setName("lower");
    isl_ast_expr_free(expr);

    // generate code for upper bound (we add +1 so it's exclusive)
    expr = isl_ast_build_expr_from_pw_aff(Build, isl_pw_aff_copy(upper));
    Value* upperVal = ISLB.buildExpr(expr);
    addBuild(Name + " : upper bound", isl_ast_expr_to_C_str(expr));
    upperVal = IRB.CreateAdd(upperVal, IRB.getInt64(1));
    upperVal->setName("upper");
    isl_ast_expr_free(expr);

    // generate code for dimension sizes
    SmallVector<Value*, 4> dimSizeVals;
    // this is a hack, spaces don't match but we don't care, there's useful results
    isl_ctx* Ctx = isl_ast_build_get_ctx(Build);
    int s = isl_options_get_on_error(Ctx);
    isl_options_set_on_error(Ctx, ISL_ON_ERROR_WARN);
    for (int i = 0; i < NDimSizes; ++i) {
      expr = isl_ast_build_expr_from_pw_aff(Build, isl_pw_aff_copy(dimSizes[i]));
      assert(expr != nullptr && "unable to generate code for dimension size");
      dimSizeVals.push_back(ISLB.buildExpr(expr));
      addBuild(Name + " : dim size", isl_ast_expr_to_C_str(expr));
      dimSizeVals.back()->setName("dimSize");
      isl_ast_expr_free(expr);
    }
    isl_options_set_on_error(Ctx, s);

    isl_ast_build_free(Build);

    // compute linearized offset of current row
    // all dimension sizes:
    //  s_1 .. s_(n-1)
    // all indices:
    //  i_1 .. i_(n-1) .. i_n
    // offsets:
    //  o_0 = 0
    //  o_k = (i_k + o_(k-1)) * s_k
    // we compute o_(n-1) with n = dimensionality of original map
    Value *Offset = IRB.getInt64(0);
    for (int i = 0; i < NDimSizes; ++i) {
      Value* gep = IRB.CreateConstGEP1_32(FixPtr, i);
      Value* Oi = IRB.CreateLoad(gep);
      Value* Si = dimSizeVals[i];
      Value* tmp = IRB.CreateAdd(Oi, Offset);
      Offset = IRB.CreateMul(tmp, Si);
    }

    Value *LinearizedLower = IRB.CreateAdd(Offset, lowerVal);
    Value *LinearizedUpper = IRB.CreateAdd(Offset, upperVal);
    LinearizedLower->setName("linLower");
    LinearizedUpper->setName("linUpper");

    IRB.CreateCall(UserCBFType::get(C), Callback, {LinearizedLower, LinearizedUpper, AuxPtr});

    IRB.CreateRetVoid();

    for (auto &it : IdMap) {
      isl_id_free(it.first);
    }

    return F;
  }

  /** struct containing a set of isl objects fully specifying a mekong memory access
   */
  typedef struct {
    isl_set *outerLoops;
    isl_pw_aff *lowerBound;
    isl_pw_aff *upperBound;
    SmallVector<isl_pw_aff*, 4> dimSizes;
  } cg_access_info;

  void cg_access_info_release(cg_access_info* info) {
    if (info->outerLoops != nullptr) isl_set_free(info->outerLoops);
    if (info->lowerBound != nullptr) isl_pw_aff_free(info->lowerBound);
    if (info->upperBound != nullptr) isl_pw_aff_free(info->upperBound);
    for (auto * dimSize : info->dimSizes) {
      isl_pw_aff_free(dimSize);
    }
  }
  void cg_access_info_free(cg_access_info* info) {
    cg_access_info_release(info);
    delete info;
  }

  /** Half a hack, half on purpose. There is no definite answer to where
   * canonicalization should occur. In order to keep the application model
   * readable it is split between analysis and code gen. Builtins are renamed
   * during analysis, domain intersection etc. is done during code gen.
   */
  __isl_give isl_map *cg_access_info_canonicalize_map(__isl_take isl_map *M) {
    const char* domain_str = "\
      [boffmin_z, boffmax_z, boffmin_y, boffmax_y, boffmin_x, boffmax_x, \
       bidmin_z, bidmax_z, bidmin_y, bidmax_y, bidmin_x, bidmax_x, \
       bdim_z, bdim_y, bdim_x] -> { \
         [boff_z, boff_y, boff_x, bid_z, bid_y, bid_x, tid_z, tid_y, tid_x] : \
           bidmin_z <= boffmin_z and bidmax_z <= boffmax_z and \
           bidmin_y <= boffmin_y and bidmax_y <= boffmax_y and \
           bidmin_x <= boffmin_x and bidmax_x <= boffmax_x and \
           0 <= boffmin_z <= boff_z < boffmax_z and 0 <= bidmin_z <= bid_z < bidmax_z and \
           0 <= boffmin_y <= boff_y < boffmax_y and 0 <= bidmin_y <= bid_y < bidmax_y and \
           0 <= boffmin_x <= boff_x < boffmax_x and 0 <= bidmin_x <= bid_x < bidmax_x and \
           0 <= tid_z < bdim_z and 0 <= tid_y < bdim_y and 0 <= tid_x < bdim_x\
       }";
    const char *in_dims[] = {"boff_z", "boff_y", "boff_x",
      "bid_z", "bid_y", "bid_x", "tid_z", "tid_y", "tid_x"};

    isl_ctx *Ctx = isl_map_get_ctx(M);

    // sort + fill in dimensions
    for (int i = 0; (size_t)i <  sizeof(in_dims)/sizeof(const char**); ++i) {
      int idx = isl_map_find_dim_by_name(M, isl_dim_in, in_dims[i]);
      if (idx > -1) {
        M = isl_map_move_dims(M, isl_dim_in, i, isl_dim_in, idx, 1);
      } else {
        M = isl_map_insert_dims(M, isl_dim_in, i, 1);
        M = isl_map_set_dim_name(M, isl_dim_in, i, in_dims[i]);
      }
    }
    assert(isl_map_dim(M, isl_dim_in) == 9 && "expected exactly 9 in dimensions");

    // apply domain
    isl_set *D = isl_set_read_from_str(Ctx, domain_str);
    assert(D != nullptr && "whoops, domain description buggy");
    M = isl_map_intersect_domain(M, D);
    assert(M != nullptr && "unable to apply domain");

    // project out thread ids
    M = isl_map_project_out(M, isl_dim_in, 6, 3);

    return M;
  }

  /** Computes all expressions required for the domain iterator of an access map.
   *
   */
  void cg_access_info_process_map(__isl_keep cg_access_info *info, __isl_take isl_map* M) {
    isl_set *S = isl_map_range(M);
    int numDims = isl_set_dim(S, isl_dim_out);

    ////////////////////////////////////////////////
    // outer loops simply require projection out last dimension
    isl_set *outer = isl_set_copy(S);
    outer = isl_set_project_out(outer, isl_dim_out, numDims-1, 1);

    ////////////////////////////////////////////////
    // bounds require modified set with extra dimensions
    int paramOffset = isl_set_dim(S, isl_dim_param);

    isl_set *boundsSet = isl_set_copy(S);
    boundsSet = isl_set_add_dims(boundsSet, isl_dim_param, numDims-1);

    for (int i = 0; i < numDims-1; ++i) {
      char *dimName = nullptr;
      asprintf(&dimName, "fix_%d", i); // reuses/reallocates buffer if != nullptr
      boundsSet = isl_set_set_dim_name(boundsSet, isl_dim_param, paramOffset+i, dimName);
      free(dimName);
      boundsSet = isl_set_equate(boundsSet, isl_dim_out, i, isl_dim_param, paramOffset + i);
    }
    boundsSet = isl_set_remove_redundancies(boundsSet);

    isl_pw_aff *lower = isl_set_dim_min(isl_set_copy(boundsSet), numDims-1);
    assert(lower != nullptr && "no lower bound found, check access map");
    isl_pw_aff *upper = isl_set_dim_max(isl_set_copy(boundsSet), numDims-1);
    assert(upper != nullptr && "no upper bound found, check access map");

    isl_set_free(boundsSet);
    isl_set_free(S);

    info->outerLoops = outer;
    info->lowerBound = lower;
    info->upperBound = upper;
  }

  /** Constructs new access info object from a string of the following format:
   * <isl map>[:<isl pw aff> ...]
   */
  __isl_give cg_access_info *cg_access_info_from_str(StringRef Str, isl_ctx* Ctx) {
    cg_access_info *info = new cg_access_info;

    SmallVector<StringRef, 4> Pieces;
    Str.split(Pieces, "#");

    // Read map from first piece of data
    isl_map* Map = isl_map_read_from_str(Ctx, Pieces[0].str().c_str());
    Map = cg_access_info_canonicalize_map(Map);
    cg_access_info_process_map(info, Map);

    // Read dim sizes from remaining pieces
    for (int i = 1; i < (int)Pieces.size(); ++i) {
      info->dimSizes.push_back(isl_pw_aff_read_from_str(Ctx, Pieces[i].str().c_str()));
    }
    return info;
  }

  /** Constructs new access info object from kernel info:
   */
  __isl_give cg_access_info *cg_access_info_from_model(
      mekong::Argument &argument, StringRef what, isl_ctx* Ctx) {
    assert(argument.isPointer == true && "access info requires pointer argument");
    cg_access_info *info = new cg_access_info;

    // Read map from first piece of data
    isl_map* Map = nullptr;
    assert((what == "read" || what == "write") && "really?");
    if (what == "read") {
      Map = isl_map_read_from_str(Ctx, argument.readMap.c_str());
      assert(Map != nullptr && "unable to read access map");
    } else if (what == "write") {
      Map = isl_map_read_from_str(Ctx, argument.writeMap.c_str());
      assert(Map != nullptr && "unable to read access map");
    }
    Map = cg_access_info_canonicalize_map(Map);
    cg_access_info_process_map(info, Map);

    // Read dim sizes from remaining pieces
    for (auto &dimsize : argument.dimsizes) {
      auto* PWA = isl_pw_aff_read_from_str(Ctx, dimsize.c_str());
      assert(PWA != nullptr && "unable to read dimension size");
      info->dimSizes.push_back(PWA);
    }

    return info;
  }

  Function* buildDispatch(const Twine& Name, ArrayRef<Function*> Iterators, Module &M) {
    LLVMContext &C = M.getContext();

    // reuse function if already found
    Function *F = M.getFunction(Name.getSingleStringRef());
    if (F == nullptr) {
      F = Function::Create(DispatcherFType::get(C), Function::InternalLinkage, Name, &M);
    }

    auto *abortFType = TypeBuilder<void(void), false>::get(C);
    auto *abortFun = M.getOrInsertFunction("abort", abortFType);

    auto *Arg = F->arg_begin();
    Value* SelectArg = Arg++;
    Value* GridPtr = Arg++;
    Value* ParamPtr = Arg++;
    Value* Callback = Arg++;
    Value* AuxPtr = Arg++;
    assert(Arg == F->arg_end());
    
    auto *EntryBB = BasicBlock::Create(C, "entry", F);
    auto *DefaultBB = BasicBlock::Create(C, "default", F);
    IRBuilder<> IRB(EntryBB);
    auto *Switch = IRB.CreateSwitch(SelectArg, DefaultBB, Iterators.size());
    for (unsigned int i = 0; i < Iterators.size(); ++i) {
      auto *BB = BasicBlock::Create(C, "case", F);
      BB->moveBefore(DefaultBB);
      IRB.SetInsertPoint(BB);
      IRB.CreateCall(Iterators[i], {GridPtr, ParamPtr, Callback, AuxPtr});
      IRB.CreateRetVoid();
      Switch->addCase(IRB.getInt32(i), BB);
    }
    IRB.SetInsertPoint(DefaultBB);
    IRB.CreateCall(abortFun, {});
    IRB.CreateRetVoid();
    return F;
  }

  /** Entry point of this pass
   */
  bool runOnModule(Module &M) override {
    if (M.getTargetTriple() == "nvptx64-nvidia-cuda" ||
        M.getTargetTriple() == "nvptx-nvidia-cuda")
      return false;

    if (modelFile == "")
      return false;

    deserialize(modelFile);

    SmallVector<Function*,8> generatedIterators;

    for (auto &kernel : app.kernels) {
      // skip if not partitionable
      if (kernel.partitioning == "none") {
        continue;
      }

      int argIdx = -1;
      for (auto &argument : kernel.arguments) {
        argIdx += 1;
        if (!argument.isPointer) continue;
        if (argument.readMap != "") {
          std::string Prefix = getPrefix(kernel.name, argIdx, "read");

          isl_ctx* Ctx = isl_ctx_alloc();
          isl_options_set_on_error(Ctx, ISL_ON_ERROR_ABORT);
          auto *info = cg_access_info_from_model(argument, "read", Ctx);

          Function *Payload = generatePayload(Prefix + "_payload", info->lowerBound,
              info->upperBound, info->dimSizes, kernel, M);
          Payload->addFnAttr(Attribute::AlwaysInline);
          Function *Iterator = generateIterator(Prefix, Payload, info->outerLoops, kernel, M);
          generatedIterators.push_back(Iterator);

          cg_access_info_free(info);
          isl_ctx_free(Ctx);
        }
        if (argument.writeMap != "") {
          std::string Prefix = getPrefix(kernel.name, argIdx, "write");

          isl_ctx* Ctx = isl_ctx_alloc();
          isl_options_set_on_error(Ctx, ISL_ON_ERROR_ABORT);
          auto *info = cg_access_info_from_model(argument, "write", Ctx);

          Function *Payload = generatePayload(Prefix + "_payload", info->lowerBound,
              info->upperBound, info->dimSizes, kernel, M);
          Payload->addFnAttr(Attribute::AlwaysInline);
          Function *Iterator = generateIterator(Prefix, Payload, info->outerLoops, kernel, M);
          generatedIterators.push_back(Iterator);

          cg_access_info_free(info);
          isl_ctx_free(Ctx);
        }
      }
    }

    if (BuildDispatcher) {
      buildDispatch("__me_dispatch", generatedIterators, M);
    }

    return true;
  }

  std::string readWholeFile(StringRef Infile) {
    auto res = MemoryBuffer::getFileAsStream(Infile);
    assert(res && "error reading file");
    return std::string((*res)->getBufferStart(), (*res)->getBufferSize());
  }

  void deserialize(StringRef Infile) {
    app.kernels.clear();
    auto res = MemoryBuffer::getFileAsStream(Infile);
    if (res) {
      app.deserialize(*(res.get()));
    } else {
      report_fatal_error("unable to read file " + Infile);
    }
  }

  void print(raw_ostream &OS, const Module *M) const override {
    OS << "------------------------------------------------\n";
    OS << "generated functions:\n";
    for (auto &kernel : app.kernels) {
      int argIdx = -1;
      for (auto &argument : kernel.arguments) {
        argIdx += 1;
        if (!argument.isPointer) continue;
        if (argument.readMap != "") {
          std::string Prefix = getPrefix(kernel.name, argIdx, "read");
          OS << Prefix << "_payload" << "\n";
          OS << Prefix << "\n";
        }
        if (argument.writeMap != "") {
          std::string Prefix = getPrefix(kernel.name, argIdx, "write");
          OS << Prefix << "_payload" << "\n";
          OS << Prefix << "\n";
        }
      }
    }

    OS << "------------------------------------------------\n";
    OS << "\n";
    OS << "generated isl expressions/nodes (C equivalent):\n";
    OS << "------------------------------------------------\n";
    for (auto &p : builds_in_c) {
      OS << p.first << ":\n" << p.second << "\n";
      OS << "------------------------------------------------\n";
    }
    OS << "\n";
    OS << "------------------------------------------------\n";
    OS << "iterator debug info\n";
    OS << "\n";
    for (auto &info : iteratorDebugInfo) {
      OS << "- " << info.iterator->getName() << ":\n";
      for (auto &name : info.paramNames) {
        OS << "    " << name << "\n";
      }
      OS << "\n";
    }
  }

  void releaseMemory() override {
    app.kernels.clear();
    builds_in_c.clear();
    iteratorDebugInfo.clear();
  }

  void dump() const {
    print(dbgs(), nullptr);
  }

  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.setPreservesAll();
  }
};

char MeCodegen::ID = 0;

}

// Pass registration

Pass *mekong::createMeCodegen() {
  return new MeCodegen();
}

Pass *mekong::createMeCodegen(StringRef File) {
  return new MeCodegen(File);
}

INITIALIZE_PASS_BEGIN(MeCodegen, "me-codegen",
                      "Mekong Iterator Host Code Generation", false, false);
INITIALIZE_PASS_END(MeCodegen, "me-codegen",
                      "Mekong Iterator Host Code Generation", false, false)
