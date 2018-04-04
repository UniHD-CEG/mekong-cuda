#include "mekong/IslBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"

#include "isl/ctx.h"
#include "isl/ast_build.h"


namespace llvm {

IslBuilder::IslBuilder()
: IRB(nullptr), Target(nullptr), IDLookup(nullptr), ExtraArgs(None), TType(TupleType::Values)
{}

IslBuilder::IslBuilder(IRBuilder<>* IRB, IDMap* IDToValueMap)
: IRB(IRB), Target(nullptr), IDLookup(IDToValueMap), ExtraArgs(None), TType(TupleType::Values)
{}

///////////////////////////////////////////////////////////////////////////////
// FLUENT INTERFACE
///////////////////////////////////////////////////////////////////////////////

IslBuilder& IslBuilder::setIRBuilder(IRBuilder<>* IRB) {
  this->IRB = IRB;
  return *this;
}

IslBuilder& IslBuilder::setIDMap(IDMap* IDToValMap) {
  this->IDLookup = IDToValMap;
  return *this;
}

IslBuilder& IslBuilder::setTarget(Function* F) {
  this->Target = F;
  return *this;
}

IslBuilder& IslBuilder::setExtraArgs(ArrayRef<Value*> ExtraArgs) {
  this->ExtraArgs = ExtraArgs;
  return *this;
}

IslBuilder& IslBuilder::setTupleType(TupleType TT) {
  this->TType = TT;
  return *this;
}

///////////////////////////////////////////////////////////////////////////////
// EXPRESSIONS
///////////////////////////////////////////////////////////////////////////////

Value* IslBuilder::buildExpr(__isl_keep isl_ast_expr* expr) {
  return createAnyExpr(expr);
}
Value* IslBuilder::createAnyExpr(__isl_keep isl_ast_expr* expr) {
  switch (isl_ast_expr_get_type(expr)) {
  case isl_ast_expr_id:
    return fetchId(isl_ast_expr_get_id(expr));
  case isl_ast_expr_int:
    return createInt(isl_ast_expr_get_val(expr));
  case isl_ast_expr_op:
    return createOp(isl_ast_expr_get_op_type(expr), expr);
  case isl_ast_expr_error:
    llvm_unreachable("isl_ast_expr_error encountered");
  }

  return nullptr;
}

Value* IslBuilder::fetchId(__isl_take isl_id* id) {
  auto result = IDLookup->find(id);
  assert(result != IDLookup->end() && "ID not found");
  isl_id_free(id);
  return result->second;
}

Value* IslBuilder::createInt(__isl_take isl_val *val) {
  int64_t i64val = isl_val_get_d(val);
  isl_val_free(val);
  //return ConstantInt::get(IRB->getInt64Ty(), i64val);
  return IRB->getInt64(i64val);
}

Value* IslBuilder::createOp(isl_ast_op_type type, __isl_keep isl_ast_expr* expr) {
  switch (type) {
  case isl_ast_op_minus:
    return createUnaryOp(type, isl_ast_expr_get_op_arg(expr, 0));
  case isl_ast_op_min:
  case isl_ast_op_max:
    return createMinMax(type, isl_ast_expr_get_op_arg(expr, 0), isl_ast_expr_get_op_arg(expr, 1));
  case isl_ast_op_and:
  case isl_ast_op_or:
  case isl_ast_op_add:
  case isl_ast_op_sub:
  case isl_ast_op_mul:
    return createBinaryOp(type, isl_ast_expr_get_op_arg(expr, 0), isl_ast_expr_get_op_arg(expr, 1));
  case isl_ast_op_div:
  case isl_ast_op_fdiv_q:
  case isl_ast_op_pdiv_q:
  case isl_ast_op_pdiv_r:
  case isl_ast_op_zdiv_r:
    return createDivision(type, isl_ast_expr_get_op_arg(expr, 0), isl_ast_expr_get_op_arg(expr, 1));
  case isl_ast_op_eq:
  case isl_ast_op_lt:
  case isl_ast_op_le:
  case isl_ast_op_gt:
  case isl_ast_op_ge:
    return createComparison(type, isl_ast_expr_get_op_arg(expr, 0), isl_ast_expr_get_op_arg(expr, 1));
  case isl_ast_op_select:
    return createSelect(isl_ast_expr_get_op_arg(expr, 0), isl_ast_expr_get_op_arg(expr, 1),
        isl_ast_expr_get_op_arg(expr, 2));
  case isl_ast_op_cond:
    llvm_unreachable("isl_ast_op_cond not supported yet");
  default:
    llvm_unreachable("unsupported operator type");
  }
  return nullptr;
}

Value* IslBuilder::createUnaryOp(isl_ast_op_type type, __isl_take isl_ast_expr* first) {
  Value* operand = createAnyExpr(first);
  isl_ast_expr_free(first);
  switch (type) {
  case isl_ast_op_minus:
    return IRB->CreateNeg(operand);
  default:
    llvm_unreachable("invalid unary operator");
  }
  return nullptr;
}

Value* IslBuilder::createMinMax(isl_ast_op_type type,
    __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second) {

  Value* lhs = createAnyExpr(first);
  Value* rhs = createAnyExpr(second);
  isl_ast_expr_free(first);
  isl_ast_expr_free(second);

  Value* lhsIsLess = IRB->CreateICmpSLT(lhs, rhs);

  switch (type) {
  case isl_ast_op_min:
    return IRB->CreateSelect(lhsIsLess, lhs, rhs);
  case isl_ast_op_max:
    return IRB->CreateSelect(lhsIsLess, rhs, lhs);
  default:
    llvm_unreachable("invalid operator (expected min or max)");
  }

  return nullptr;
}

Value* IslBuilder::createBinaryOp(isl_ast_op_type type,
    __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second) {
  Value* lhs = createAnyExpr(first);
  Value* rhs = createAnyExpr(second);
  isl_ast_expr_free(first);
  isl_ast_expr_free(second);

  Instruction::BinaryOps opcode;
  switch (type) {
  case isl_ast_op_and:
    opcode = Instruction::And; break;
  case isl_ast_op_or:
    opcode = Instruction::Or; break;
  case isl_ast_op_add:
    opcode = Instruction::Add; break;
  case isl_ast_op_sub:
    opcode = Instruction::Sub; break;
  case isl_ast_op_mul:
    opcode = Instruction::Mul; break;
  default:
    llvm_unreachable("invalid binary operator");
  }
  return IRB->CreateBinOp(opcode, lhs, rhs);
}

Value* IslBuilder::createDivision(isl_ast_op_type type,
    __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second) {
  Value* lhs = createAnyExpr(first);
  Value* rhs = createAnyExpr(second);
  isl_ast_expr_free(first);
  isl_ast_expr_free(second);


  switch (type) {
  case isl_ast_op_div:
    return IRB->CreateExactSDiv(lhs, rhs);
  case isl_ast_op_fdiv_q: // floor div, rounding towards negative infinity
    return IRB->CreateUDiv(lhs, rhs);
  case isl_ast_op_pdiv_q: // positiv div
    return IRB->CreateUDiv(lhs, rhs);
  case isl_ast_op_pdiv_r: // positiv div, remainder
    return IRB->CreateURem(lhs, rhs);
  case isl_ast_op_zdiv_r: // "zero if the remainder is zero"
    return IRB->CreateSRem(lhs, rhs);
  default:
    break;
  }
  llvm_unreachable("invalid division operator");
  return nullptr;
}

Value* IslBuilder::createComparison(isl_ast_op_type type,
    __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second) {
  Value* lhs = createAnyExpr(first);
  Value* rhs = createAnyExpr(second);
  isl_ast_expr_free(first);
  isl_ast_expr_free(second);

  switch (type) {
  case isl_ast_op_eq:
    return IRB->CreateICmpEQ(lhs, rhs);
  case isl_ast_op_lt:
    return IRB->CreateICmpSLT(lhs, rhs);
  case isl_ast_op_le:
    return IRB->CreateICmpSLE(lhs, rhs);
  case isl_ast_op_gt:
    return IRB->CreateICmpSGT(lhs, rhs);
  case isl_ast_op_ge:
    return IRB->CreateICmpSGE(lhs, rhs);
  default:
    llvm_unreachable("invalid unary operator");
  }
  return nullptr;
}

Value* IslBuilder::createSelect(__isl_take isl_ast_expr* cond,
    __isl_take isl_ast_expr* iftrue, __isl_take isl_ast_expr* iffalse) {
  Value* valCond = createAnyExpr(cond);
  Value* valTrue = createAnyExpr(iftrue);
  Value* valFalse = createAnyExpr(iffalse);
  isl_ast_expr_free(cond);
  isl_ast_expr_free(iftrue);
  isl_ast_expr_free(iffalse);
  return IRB->CreateSelect(valCond, valTrue, valFalse);
}

///////////////////////////////////////////////////////////////////////////////
// NODES
///////////////////////////////////////////////////////////////////////////////

void IslBuilder::buildNode(__isl_keep isl_ast_node* node) {
  createAnyNode(node);
}

void IslBuilder::createAnyNode(__isl_keep isl_ast_node* node) {
  switch (isl_ast_node_get_type(node)) {
  case isl_ast_node_block:
    createBlock(node);
    return;
  case isl_ast_node_if:
    createIf(node);
    return;
  case isl_ast_node_for:
    createFor(node);
    return;
  case isl_ast_node_mark:
    llvm_unreachable("unsupported ast node type: mark");
    return;
  case isl_ast_node_user:
    createUser(node);
    return;
  case isl_ast_node_error:
    outs() << "top level is error\n";
    break;
  }

  //llvm_unreachable("invalid isl_ast_node_type");
}

void IslBuilder::createBlock(__isl_keep isl_ast_node* node) {
  assert(isl_ast_node_get_type(node) == isl_ast_node_block);

  auto foreachFn = [](__isl_take isl_ast_node* child, void* _this) -> isl_stat {
    ((IslBuilder*)_this)->createAnyNode(child);
    isl_ast_node_free(child);
    return isl_stat_ok;
  };

  isl_ast_node_list* children = isl_ast_node_block_get_children(node);
  isl_ast_node_list_foreach(children, foreachFn, (void*)this);
  isl_ast_node_list_free(children);
}

void IslBuilder::createIf(__isl_keep isl_ast_node* node) {
  assert(isl_ast_node_get_type(node) == isl_ast_node_if);
  LLVMContext &C = IRB->getContext();

  Function* F = IRB->GetInsertBlock()->getParent();

  isl_ast_expr* condExpr = isl_ast_node_if_get_cond(node);
  Value* cond = buildExpr(condExpr);
  isl_ast_expr_free(condExpr);


  BasicBlock* ThenBB = BasicBlock::Create(C, "if_then", F);
  BasicBlock* MergeBB = BasicBlock::Create(C, "if_cont", F);
  BasicBlock* ElseBB = nullptr;

  if (isl_ast_node_if_has_else(node)) {
    ElseBB = BasicBlock::Create(C, "if_else", F);
    ElseBB->moveBefore(ThenBB);
  } else {
    ElseBB = MergeBB;
  }

  IRB->CreateCondBr(cond, ThenBB, ElseBB);

  IRB->SetInsertPoint(ThenBB);
  isl_ast_node* thenNode = isl_ast_node_if_get_then(node);
  createAnyNode(thenNode);
  isl_ast_node_free(thenNode);
  IRB->CreateBr(MergeBB);

  if (isl_ast_node_if_has_else(node)) {
    IRB->SetInsertPoint(ElseBB);
    isl_ast_node* elseNode = isl_ast_node_if_get_else(node);
    createAnyNode(elseNode);
    isl_ast_node_free(elseNode);
    IRB->CreateBr(MergeBB);
  }

  IRB->SetInsertPoint(MergeBB);
}

void IslBuilder::createFor(__isl_keep isl_ast_node* node) {
  assert(isl_ast_node_get_type(node) == isl_ast_node_for);
  LLVMContext &C = IRB->getContext();

  Function* F = IRB->GetInsertBlock()->getParent();

  isl_ast_expr* iteratorExpr = isl_ast_node_for_get_iterator(node);
  assert(isl_ast_expr_get_type(iteratorExpr) == isl_ast_expr_id);
  isl_id* iterator = isl_ast_expr_get_id(iteratorExpr);
  isl_ast_expr_free(iteratorExpr);

  BasicBlock* originBB = IRB->GetInsertBlock();

  BasicBlock* CondBB = BasicBlock::Create(C, "for_cond", F);
  BasicBlock* BodyBB = BasicBlock::Create(C, "for_body", F);
  BasicBlock* IncBB = BasicBlock::Create(C, "for_inc", F);
  BasicBlock* ContBB = BasicBlock::Create(C, "for_cont", F);

  // build initial value
  isl_ast_expr* initExpr = isl_ast_node_for_get_init(node);

  Value* init = buildExpr(initExpr);
  isl_ast_expr_free(initExpr);
  if (!init->hasName()) {
    init->setName("for_init");
  }

  // always jump to CondBB
  IRB->CreateBr(CondBB);

  // start condition setup
  IRB->SetInsertPoint(CondBB);
  // condPHI -> iterator
  PHINode* condPHI = IRB->CreatePHI(IRB->getInt64Ty(), 2); // create PHI with two incoming -> init + inc
  condPHI->setName(isl_id_get_name(iterator));
  // init value
  condPHI->addIncoming(init, originBB);

  // isl avoids renames iterators on collisions but let's keep this in
  assert(IDLookup->find(iterator) == IDLookup->end() && "trying to shadow existing id");
  IDLookup->insert(std::make_pair(iterator, condPHI));

  // build condition with iterator instance pointing to PHI node
  isl_ast_expr* condExpr = isl_ast_node_for_get_cond(node);
  Value* cond = buildExpr(condExpr);
  isl_ast_expr_free(condExpr);
  // jump to body or not
  IRB->CreateCondBr(cond, BodyBB, ContBB);

  // start body setup
  IRB->SetInsertPoint(BodyBB);
  // build loop body
  isl_ast_node* bodyNode = isl_ast_node_for_get_body(node);
  createAnyNode(bodyNode);
  isl_ast_node_free(bodyNode);
  // jump to inc
  IRB->CreateBr(IncBB);


  // start inc setup
  IRB->SetInsertPoint(IncBB);
  // build inc body
  isl_ast_expr* incExpr = isl_ast_node_for_get_inc(node);
  Value* increment = buildExpr(incExpr);
  Value* incremented = IRB->CreateAdd(condPHI, increment);
  isl_ast_expr_free(incExpr);
  // update condPHI and jump to cond
  condPHI->addIncoming(incremented, IncBB);
  IRB->CreateBr(CondBB);

  IRB->SetInsertPoint(ContBB);

  IDLookup->erase(iterator);
  isl_id_free(iterator);
}

void IslBuilder::createUser(__isl_keep isl_ast_node* node) {
  assert(isl_ast_node_get_type(node) == isl_ast_node_user);

  isl_ast_expr* callExpr = isl_ast_node_user_get_expr(node);
  assert(isl_ast_expr_get_type(callExpr) == isl_ast_expr_op);
  assert(isl_ast_expr_get_op_type(callExpr) == isl_ast_op_call);

  int numArgs = isl_ast_expr_get_op_n_arg(callExpr);
  Function* F = nullptr;

  if (Target != nullptr) {
    F = Target;
  } else {
    Module* M = IRB->GetInsertBlock()->getParent()->getParent();

    isl_ast_expr* FnExpr = isl_ast_expr_get_op_arg(callExpr, 0);
    assert(isl_ast_expr_get_type(FnExpr) == isl_ast_expr_id);

    isl_id* FnId = isl_ast_expr_get_id(FnExpr);
    F = M->getFunction(isl_id_get_name(FnId));
    assert(F != nullptr && "Target function not found");
    isl_id_free(FnId);
    isl_ast_expr_free(FnExpr);
  }

  SmallVector<Value*, 8> arguments;

  if (TType == TupleType::Values)  {
    assert(numArgs == (int)F->arg_size() - (int)ExtraArgs.size() + 1 && "Target function has wrong number of arguments");
    for (int i = 1; i < numArgs; ++i) {
      isl_ast_expr* op = isl_ast_expr_get_op_arg(callExpr, i);
      Value* opval = buildExpr(op);
      arguments.push_back(opval);
      isl_ast_expr_free(op);
    }
  } else if (TType == TupleType::Array) {
    assert(F->arg_size() == 1 + ExtraArgs.size() && "Target function has wrong number of arguments");
    Type* i64T = IRB->getInt64Ty();
    Value *Alloc = IRB->CreateAlloca(i64T, IRB->getInt32(numArgs-1), "fix");
    for (int i = 1; i < numArgs; ++i) {
      isl_ast_expr* op = isl_ast_expr_get_op_arg(callExpr, i);
      Value* opval = buildExpr(op);
      isl_ast_expr_free(op);
      Value* Target = IRB->CreateConstGEP1_32(Alloc, i-1);
      IRB->CreateStore(opval, Target);
    }
    arguments.push_back(Alloc);
  }

  for (auto *arg : ExtraArgs) {
    arguments.push_back(arg);
  }

  isl_ast_expr_free(callExpr);

  IRB->CreateCall(F, arguments);
}

} // end namespace
