#ifndef __ISL_BUILDER_H
#define __ISL_BUILDER_H

#include <map>
#include "llvm/IR/IRBuilder.h"
#include "isl/ast_build.h"

namespace llvm {


class IslBuilder {
public:
  using IDMap = std::map<isl_id*, Value*>;
  enum class TupleType {
    Array, // pass index vector of point as vector
    Values // pass index vector of point as scalar values
  };

  IslBuilder();
  IslBuilder(IRBuilder<>* IRB, IDMap* IDToValueMap);

  IslBuilder& setIRBuilder(IRBuilder<>* IRB);
  IslBuilder& setIDMap(IDMap* IRB);
  IslBuilder& setTarget(Function* F);
  IslBuilder& setExtraArgs(ArrayRef<Value*> ExtraArgs);
  IslBuilder& setTupleType(TupleType TT);

  Value* buildExpr(__isl_keep isl_ast_expr* expr);
  void buildNode(__isl_keep isl_ast_node* node);

private:
  void createAnyNode(__isl_keep isl_ast_node* node);
  void createBlock(__isl_keep isl_ast_node* node);
  void createIf(__isl_keep isl_ast_node* node);
  void createFor(__isl_keep isl_ast_node* node);
  void createUser(__isl_keep isl_ast_node* node);

  Value* fetchId(__isl_take isl_id* id);

  Value* createAnyExpr(__isl_keep isl_ast_expr* expr);
  Value* createInt(__isl_take isl_val *val);
  Value* createOp(isl_ast_op_type type, __isl_keep isl_ast_expr* expr);
  Value* createUnaryOp(isl_ast_op_type type, __isl_take isl_ast_expr* first);
  Value* createMinMax(isl_ast_op_type type, __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second);
  Value* createBinaryOp(isl_ast_op_type type, __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second);
  Value* createDivision(isl_ast_op_type type, __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second);
  Value* createComparison(isl_ast_op_type type, __isl_take isl_ast_expr* first, __isl_take isl_ast_expr* second);
  Value* createSelect(__isl_take isl_ast_expr* cond, __isl_take isl_ast_expr* iftrue,
      __isl_take isl_ast_expr* iffalse);

  IRBuilder<>* IRB;
  Function* Target;
  IDMap* IDLookup;
  ArrayRef<Value*> ExtraArgs;
  TupleType TType;
};

} // end namespace

#endif
