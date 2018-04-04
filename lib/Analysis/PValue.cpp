//===- PValue.cpp ------ isl C++ interface wrapper ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TODO Add comments to this implementation!
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PValue.h"

#include "isl/aff.h"
#include "isl/id.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/constraint.h"
#include "isl/options.h"

#define DEBUG_TYPE "pvalue"

using namespace llvm;

static int DOMAIN_N_BASIC_SEK_TRESHOLD = 8;
static int PWA_N_PIECE_TRESHOLD = 4;


static void replace(std::string &str, const std::string &find,
                    const std::string &replace) {
  size_t pos = 0;
  while ((pos = str.find(find, pos)) != std::string::npos) {
    str.replace(pos, find.length(), replace);
    pos += replace.length();
  }
}

static void makeIslCompatible(std::string &str) {
  replace(str, ".", "_");
  replace(str, "\"", "_");
  replace(str, " ", "__");
  replace(str, "=>", "TO");
}

/// Create a map to map from a given iteration to a subsequent iteration.
///
/// This map maps from SetSpace -> SetSpace where the dimensions @p Dim
/// is incremented by one and all other dimensions are equal, e.g.,
///             [i0, i1, i2, i3] -> [i0, i1, i2 + 1, i3]
///
/// if @p Dim is 2 and @p SetSpace has 4 dimensions.
static __isl_give isl_map *
createNextIterationMap(__isl_take isl_space *SetSpace, unsigned Dim, bool All) {
  auto *MapSpace = isl_space_map_from_set(SetSpace);
  auto *NextIterationMap = isl_map_universe(isl_space_copy(MapSpace));
  for (unsigned u = 0; u < isl_map_dim(NextIterationMap, isl_dim_in); u++)
    if (u != Dim)
      NextIterationMap =
          isl_map_equate(NextIterationMap, isl_dim_in, u, isl_dim_out, u);
  isl_constraint *C;
  if (All) {
    C = isl_constraint_alloc_inequality(isl_local_space_from_space(MapSpace));
    C = isl_constraint_set_constant_si(C, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, Dim, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, Dim, 1);
  } else {
    C = isl_constraint_alloc_equality(isl_local_space_from_space(MapSpace));
    C = isl_constraint_set_constant_si(C, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, Dim, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, Dim, -1);
  }
  NextIterationMap = isl_map_add_constraint(NextIterationMap, C);
  return NextIterationMap;
}

std::string PVBase::getIslCompatibleName(const std::string &Prefix,
                                         const std::string &Middle,
                                         const std::string &Suffix) {
  std::string S = Prefix + Middle + Suffix;
  makeIslCompatible(S);
  return S;
}

std::string PVBase::str() const {
  llvm_unreachable("PVBase::str() should be overloaded!");
}

/* -------------------- PVCtx ------------------------- */

PVCtx::PVCtx() : PVCtx(isl_ctx_alloc()) {
  isl_options_set_on_error(getIslCtx(), ISL_ON_ERROR_ABORT);
}

isl_space *PVCtx::getSpace() const {
  return isl_space_set_alloc(getIslCtx(), 0, 0);
}

std::string PVCtx::str() const {
  return "PVCTX";
}

/* -------------------- PVId ------------------------- */

PVId::PVId(const PVBase &Base, const std::string &Name, void *Payload)
    : Obj(isl_id_alloc(Base.getIslCtx(), Name.c_str(), Payload)) {}
PVId::PVId(const PVId &Other) : Obj(isl_id_copy(Other.Obj)) {}
PVId::PVId(PVId &&Other) : Obj(Other.Obj) { Other.Obj = nullptr; }

PVId::~PVId() { isl_id_free(Obj); }

PVId &PVId::operator=(const PVId &Other) {
  if (this != &Other) {
    isl_id_free(Obj);
    Obj = isl_id_copy(Other.Obj);
  }
  return *this;
}

PVId &PVId::operator=(PVId &&Other) {
  if (this != &Other) {
    Obj = isl_id_free(Obj);
    std::swap(Obj, Other.Obj);
  }
  return *this;
}

PVId::operator isl_id *() const { return isl_id_copy(Obj); }

void *PVId::getPayload() const {
  return isl_id_get_user(Obj);
}

isl_ctx *PVId::getIslCtx() const { return isl_id_get_ctx(Obj); }

isl_space *PVId::getSpace() const {
  return isl_space_set_alloc(getIslCtx(), 0, 0);
}

std::string PVId::str() const {
  const char *cstr = isl_id_get_name(Obj);
  if (!cstr)
    return "null";
  std::string Result(cstr);
  return Result;
}

bool PVId::operator<(const PVId &Other) const {
  return isl_id_get_hash(Obj) < isl_id_get_hash(Other);
}

template <>
bool PVLess<PVId>::operator()(const PVId &lhs, const PVId &rhs) const {
  return isl_id_get_hash(lhs) < isl_id_get_hash(rhs);
}

/* -------------------- PVSet ------------------------ */

PVSet::PVSet(isl_set *S) : Obj(S) {}
PVSet::PVSet(const PVSet &Other) : Obj(isl_set_copy(Other.Obj)) {}

PVSet PVSet::createParameterRange(const PVId &Id, int LowerBound,
                                  int UpperBound) {
  PVSet ParamRange = universe(Id);
  ParamRange.Obj = isl_set_add_dims(ParamRange.Obj, isl_dim_param, 1);
  ParamRange.Obj = isl_set_set_dim_id(ParamRange.Obj, isl_dim_param, 0, Id);
  //errs() << "LB: " << LowerBound << " UB: " << UpperBound << "\n";
  //errs() << ParamRange << "\n";
  ParamRange.Obj =
      isl_set_lower_bound_si(ParamRange.Obj, isl_dim_param, 0, LowerBound);
  //errs() << ParamRange << "\n";
  ParamRange.Obj =
      isl_set_upper_bound_si(ParamRange.Obj, isl_dim_param, 0, UpperBound);
  //errs() << ParamRange << "\n";
  return ParamRange;
}

PVSet::~PVSet() { isl_set_free(Obj); }

PVSet &PVSet::operator=(const PVSet &Other) {
  if (this != &Other) {
    isl_set_free(Obj);
    Obj = isl_set_copy(Other.Obj);
  }
  return *this;
}

PVSet &PVSet::operator=(PVSet &&Other) {
  if (this != &Other) {
    Obj = isl_set_free(Obj);
    std::swap(Obj, Other.Obj);
  }
  return *this;
}

PVSet::operator isl_set *() const { return isl_set_copy(Obj); }

isl_set * PVSet::getObj() const { return *this; }

isl_ctx *PVSet::getIslCtx() const { return isl_set_get_ctx(Obj); }

isl_space *PVSet::getSpace() const { return isl_set_get_space(Obj); }

size_t PVSet::getNumInputDimensions() const { return isl_set_n_dim(Obj); }

size_t PVSet::getNumParameters() const { return isl_set_n_param(Obj); }

bool PVSet::isEmpty() const { return isl_set_is_empty(Obj); }
bool PVSet::isUniverse() const {
  isl_set *C = isl_set_complement(isl_set_copy(Obj));
  bool U = isl_set_is_empty(C);
  isl_set_free(C);
  return U;
}

bool PVSet::isBounded() const { return isl_set_is_bounded(Obj); }

bool PVSet::isComplex() const {
  bool Complex = isl_set_n_basic_set(Obj) > DOMAIN_N_BASIC_SEK_TRESHOLD;
  return Complex;
}

PVSet PVSet::getParameterSet() const {
  return isl_set_params(isl_set_copy(Obj));
}

int PVSet::getParameterPosition(const PVId &Id) const {
  int Pos = isl_set_find_dim_by_id(Obj, isl_dim_param, Id);
  return Pos;
}

bool PVSet::hasLowerBoundForParam(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  if (Pos < 0)
    return false;
  return isl_set_dim_has_any_lower_bound(Obj, isl_dim_param, Pos);
}

PVSet &PVSet::minForParam(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  assert(Pos >= 0);
  assert(isl_set_is_params(Obj));
  Obj = isl_set_move_dims(Obj, isl_dim_set, 0, isl_dim_param, Pos, 1);
  Obj = isl_set_lexmin(Obj);
  return *this;
}

void PVSet::dropUnusedParameters() {
  size_t NumParams = getNumParameters();
  for (size_t i = 0; i < NumParams; i++) {
    if (isl_set_involves_dims(Obj, isl_dim_param, i, 1))
      continue;

    Obj = isl_set_project_out(Obj, isl_dim_param, i, 1);
    i--;
    NumParams--;
  }
}

PVSet &PVSet::simplify(const PVSet &S) {
  Obj = isl_set_gist(Obj, S);
  return *this;
}

PVSet &PVSet::simplifyParameters(const PVSet &S) {
  Obj = isl_set_gist_params(Obj, S);
  return *this;
}

PVSet &PVSet::complement() {
  Obj = isl_set_complement(Obj);
  Obj = isl_set_coalesce(Obj);
  return *this;
}

static void unifySetDimensions(isl_set *&S0, isl_set *&S1) {
  int DimDiff = isl_set_n_dim(S0) - isl_set_n_dim(S1);
  if (DimDiff > 0)
    S1 = isl_set_add_dims(S1, isl_dim_set, DimDiff);
  else if (DimDiff < 0)
    S0 = isl_set_add_dims(S0, isl_dim_set, -1 * DimDiff);
  // S0 = isl_set_align_params(S0, isl_set_get_space(S1));
  // S1 = isl_set_align_params(S1, isl_set_get_space(S0));
}

PVSet &PVSet::unify(const PVSet &S) {
  if (!Obj)
    Obj = isl_set_copy(S.Obj);
  else if (S.Obj) {
    isl_set *SObj = S;
    unifySetDimensions(Obj, SObj);
    Obj = isl_set_union(Obj, SObj);
    Obj = isl_set_coalesce(Obj);
  }
  return *this;
}

PVSet &PVSet::neg() {
  if (Obj)
    Obj = isl_set_neg(Obj);
  return *this;
}

PVSet &PVSet::add(const PVSet &S) {
  if (!Obj || !S.Obj)
    return *this;
  isl_set *SObj = S;
  unifySetDimensions(Obj, SObj);
  Obj = isl_set_sum(Obj, SObj);
  return *this;
}

PVSet &PVSet::sub(const PVSet &S) {
  PVSet SCopy = S;
  return add(SCopy.neg());
}

PVSet &PVSet::subtract(const PVSet &S) {
  if (!Obj)
    Obj = isl_set_copy(S.Obj);
  else if (S.Obj) {
    isl_set *SObj = S;
    unifySetDimensions(Obj, SObj);
    Obj = isl_set_subtract(Obj, SObj);
    Obj = isl_set_coalesce(Obj);
  }
  return *this;
}

PVSet &PVSet::intersect(const PVSet &S) {
  if (!Obj)
    Obj = isl_set_copy(S.Obj);
  else if (S.Obj) {
    isl_set *SObj = S;
    unifySetDimensions(Obj, SObj);
    Obj = isl_set_intersect(Obj, SObj);
    Obj = isl_set_coalesce(Obj);
  }
  return *this;
}

/// Add @p BSet to the set @p User if @p BSet is bounded.
static isl_stat collectBoundedParts(__isl_take isl_basic_set *BSet,
                                    void *User) {
  isl_set **BoundedParts = static_cast<isl_set **>(User);
  if (isl_basic_set_is_bounded(BSet))
    *BoundedParts = isl_set_union(*BoundedParts, isl_set_from_basic_set(BSet));
  else
    isl_basic_set_free(BSet);
  return isl_stat_ok;
}

static __isl_give isl_set *collectBoundedParts(__isl_take isl_set *S) {
  isl_set *BoundedParts = isl_set_empty(isl_set_get_space(S));
  isl_set_foreach_basic_set(S, collectBoundedParts, &BoundedParts);
  isl_set_free(S);
  return BoundedParts;
}

/// Compute the (un)bounded parts of @p S wrt. to dimension @p Dim.
///
/// @returns A separation of @p S into first an unbounded then a bounded subset,
///          both with regards to the dimension @p Dim.
static std::pair<__isl_give isl_set *, __isl_give isl_set *>
partitionSetParts(__isl_take isl_set *S, unsigned Dim) {
  for (unsigned u = 0, e = isl_set_n_dim(S); u < e; u++)
    S = isl_set_lower_bound_si(S, isl_dim_set, u, 0);

  unsigned NumDimsS = isl_set_n_dim(S);
  isl_set *OnlyDimS = isl_set_copy(S);

  // Remove dimensions that are greater than Dim as they are not interesting.
  assert(NumDimsS >= Dim + 1);
  OnlyDimS =
      isl_set_project_out(OnlyDimS, isl_dim_set, Dim + 1, NumDimsS - Dim - 1);

  // Create artificial parametric upper bounds for dimensions smaller than Dim
  // as we are not interested in them.
  OnlyDimS = isl_set_insert_dims(OnlyDimS, isl_dim_param, 0, Dim);
  for (unsigned u = 0; u < Dim; u++) {
    isl_constraint *C = isl_inequality_alloc(
        isl_local_space_from_space(isl_set_get_space(OnlyDimS)));
    C = isl_constraint_set_coefficient_si(C, isl_dim_param, u, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_set, u, -1);
    OnlyDimS = isl_set_add_constraint(OnlyDimS, C);
  }

  // Collect all bounded parts of OnlyDimS.
  isl_set *BoundedParts = collectBoundedParts(OnlyDimS);

  // Create the dimensions greater than Dim again.
  BoundedParts = isl_set_insert_dims(BoundedParts, isl_dim_set, Dim + 1,
                                     NumDimsS - Dim - 1);

  // Remove the artificial upper bound parameters again.
  BoundedParts = isl_set_remove_dims(BoundedParts, isl_dim_param, 0, Dim);

  isl_set *UnboundedParts = isl_set_subtract(S, isl_set_copy(BoundedParts));
  return std::make_pair(UnboundedParts, BoundedParts);
}

PVSet &PVSet::restrictToBoundedPart(unsigned Dim, PVSet *UnboundedPart) {
  auto Parts = partitionSetParts(Obj, Dim);
  Obj = isl_set_coalesce(Parts.second);
  if (UnboundedPart)
    *UnboundedPart = PVSet(isl_set_coalesce(Parts.first));
  else
    isl_set_free(Parts.first);
  return *this;
}

PVSet PVSet::empty(const PVBase &Base) {
  auto *Space = Base.getSpace();
  if (isl_space_is_map(Space))
    Space = isl_space_domain(Space);
  return isl_set_empty(Space);
}

PVSet PVSet::empty(const PVBase &Base, unsigned Dims) {
  auto *Space = isl_space_set_alloc(Base.getIslCtx(), 0, Dims);
  return isl_set_empty(Space);
}

PVSet PVSet::universe(const PVBase &Base) {
  auto *Space = Base.getSpace();
  if (isl_space_is_map(Space))
    Space = isl_space_domain(Space);
  return isl_set_add_dims(isl_set_universe(Space), isl_dim_set, 0);
}

PVSet PVSet::unify(const PVSet &S0, const PVSet &S1) {
  return PVSet(S0).unify(S1);
}

PVSet PVSet::intersect(const PVSet &S0, const PVSet &S1) {
  return PVSet(S0).intersect(S1);
}

PVSet &PVSet::maxInLastInputDims(unsigned Dims) {
  if (Dims == 0)
    return *this;
  unsigned NumDims = getNumInputDimensions();
  assert(NumDims >= Dims);
  Obj = isl_set_project_out(Obj, isl_dim_set, 0, NumDims - Dims);
  if (!isl_set_is_bounded(Obj)) {
    Obj = isl_set_free(Obj);
    return *this;
  }
  Obj = isl_set_lexmax(Obj);
  Obj = isl_set_insert_dims(Obj, isl_dim_set, 0, NumDims - Dims);
  return *this;
}

PVSet &PVSet::fixInputDim(unsigned Dim, int64_t Value) {
  auto Dims = getNumInputDimensions();
  if (Dims <= Dim)
    Obj = isl_set_add_dims(Obj, isl_dim_set, Dim - Dims + 1);
  Obj = isl_set_fix_si(Obj, isl_dim_set, Dim, Value);
  Obj = isl_set_coalesce(Obj);
  return *this;
}

PVSet &PVSet::equateInputDim(unsigned Dim, const PVId &Id) {
  int Pos = isl_set_find_dim_by_id(Obj, isl_dim_param, Id);
  assert(Pos >= 0);
  Obj = isl_set_equate(Obj, isl_dim_param, Pos, isl_dim_set, Dim);
  return *this;
}

PVSet &PVSet::addInputDims(unsigned Dims) {
  Obj = isl_set_add_dims(Obj, isl_dim_set, Dims);
  return *this;
}

PVSet &PVSet::dropFirstInputDims(unsigned Dims) {
  Obj = isl_set_project_out(Obj, isl_dim_set, 0, Dims);
  return *this;
}

PVSet &PVSet::dropLastInputDims(unsigned Dims) {
  Obj = isl_set_project_out(Obj, isl_dim_set, getNumInputDimensions() - Dims, Dims);
  Obj = isl_set_coalesce(Obj);
  return *this;
}

PVSet &PVSet::dropDimsFrom(unsigned Dim) {
  if (Dim < getNumInputDimensions())
    Obj = isl_set_project_out(Obj, isl_dim_set, Dim, getNumInputDimensions() - Dim);
  return *this;
}

PVSet &PVSet::setInputLowerBound(unsigned Dim, int64_t Value) {
  auto Dims = getNumInputDimensions();
  if (Dims <= Dim)
    Obj = isl_set_add_dims(Obj, isl_dim_set, Dim - Dims + 1);
  Obj = isl_set_lower_bound_si(Obj, isl_dim_set, Dim, Value);
  Obj = isl_set_coalesce(Obj);
  return *this;
}

PVSet &PVSet::preimage(const PVAff &PWA) {
  Obj = isl_set_preimage_pw_multi_aff(Obj, isl_pw_multi_aff_from_pw_aff(PWA));
  dropUnusedParameters();
  return *this;
}

PVSet &PVSet::projectParameter(const PVId &Id) {
  int Pos = isl_set_find_dim_by_id(Obj, isl_dim_param, Id);
  assert(Pos >= 0);
  Obj = isl_set_project_out(Obj, isl_dim_param, Pos, 1);
  return *this;
}

PVSet &PVSet::getNextIteration(unsigned Dim) {
  isl_map *NIMap = createNextIterationMap(getSpace(), Dim, false);
  Obj = isl_set_apply(Obj, NIMap);
  return *this;
}

PVSet &PVSet::getNextIterations(unsigned Dim) {
  isl_map *NIMap = createNextIterationMap(getSpace(), Dim, true);
  Obj = isl_set_apply(Obj, NIMap);
  return *this;
}

PVId PVSet::getParameter(unsigned No) const {
  return PVId(isl_set_get_dim_id(Obj, isl_dim_param, No));
}

void PVSet::getParameters(SmallVectorImpl<PVId> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++) {
    if (isl_set_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i));
  }
}

void PVSet::getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++)
    if (isl_set_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i).getPayloadAs<Value *>());
}

std::string PVSet::str() const {
  char *cstr = isl_set_to_str(Obj);
  if (!cstr)
    return "null";
  std::string Result(cstr);
  ::free(cstr);
  return Result;
}

/* -------------------- PVMap ----------------------- */
static void adjustDimensionsPlain(isl_map *&Map0, isl_map *&Map1) {
  int DimIn0 = isl_map_dim(Map0, isl_dim_in);
  int DimIn1 = isl_map_dim(Map1, isl_dim_in);
  if (DimIn0 == DimIn1)
    return;

  auto *&Map = (DimIn0 > DimIn1) ? Map1 : Map0;
  Map = isl_map_add_dims(Map, isl_dim_in, std::abs(DimIn0 - DimIn1));

  assert(isl_map_dim(Map0, isl_dim_in) == isl_map_dim(Map1, isl_dim_in));
  assert(isl_map_dim(Map0, isl_dim_out) == isl_map_dim(Map1, isl_dim_out));
}


PVMap::PVMap(isl_map *M) : Obj(M) {}
PVMap::PVMap(const PVMap &Other) : Obj(isl_map_copy(Other.Obj)) {}

PVMap::PVMap(const PVAff &Coeff, const PVId &Id, const PVBase &Base) {
  PVAff Aff(Coeff, Id, Base);
  Obj = isl_map_from_pw_aff(Aff);
}

PVMap::PVMap(ArrayRef<PVAff> Affs, const PVId &Id) {
  if (Affs.empty())
    return;

  const PVAff &Aff = Affs.front();
  isl_space *Space = isl_space_alloc(Aff.getIslCtx(), 0,
                                     Aff.getNumInputDimensions(), Affs.size());
  isl_pw_multi_aff *MPWA = isl_pw_multi_aff_zero(Space);
  for (unsigned i = 0; i < Affs.size(); i++) {
    MPWA = isl_pw_multi_aff_set_pw_aff(MPWA, i, Affs[i]);
  }

  Obj = isl_map_from_pw_multi_aff(MPWA);
  assert(!isl_map_has_tuple_id(Obj, isl_dim_out));
  Obj = isl_map_coalesce(Obj);

  Obj = isl_map_set_tuple_id(Obj, isl_dim_out, Id);
  assert(isl_map_has_tuple_id(Obj, isl_dim_out));
}

PVMap::PVMap(const PVAff &Aff, long Factor) {
  isl_val *FactorVal = isl_val_int_from_si(getIslCtx(), Factor);
  Obj = isl_map_from_pw_aff(isl_pw_aff_scale_val(Aff, FactorVal));
}

PVMap::PVMap(const PVId &ValId, const PVId &InputId, const PVId &OutputId) {
  Obj = isl_map_from_pw_aff(PVAff(ValId));
  if (InputId)
    Obj = isl_map_set_tuple_id(Obj, isl_dim_in, InputId);
  if (OutputId)
    Obj = isl_map_set_tuple_id(Obj, isl_dim_out, OutputId);
}

PVMap::~PVMap() { isl_map_free(Obj); }

PVMap &PVMap::operator=(const PVMap &Other) {
  if (this != &Other) {
    isl_map_free(Obj);
    Obj = isl_map_copy(Other.Obj);
  }
  return *this;
}

PVMap &PVMap::operator=(PVMap &&Other) {
  if (this != &Other) {
    Obj = isl_map_free(Obj);
    std::swap(Obj, Other.Obj);
  }
  return *this;
}
PVMap::operator isl_map *() const { return isl_map_copy(Obj); }

isl_ctx *PVMap::getIslCtx() const { return isl_map_get_ctx(Obj); }

isl_space *PVMap::getSpace() const { return isl_map_get_space(Obj); }

bool PVMap::isEmpty() const { return Obj && isl_map_is_empty(Obj); }

bool PVMap::isEqual(const PVMap &Map) const {
  return Obj && Map && isl_map_is_equal(Obj, Map);
}

size_t PVMap::getNumInputDimensions() const { return isl_map_dim(Obj, isl_dim_in); }

size_t PVMap::getNumOutputDimensions() const { return isl_map_dim(Obj, isl_dim_out); }

size_t PVMap::getNumParameters() const { return isl_map_dim(Obj, isl_dim_param); }

PVMap::CombinatorFn PVMap::getCombinatorFn(PVMap::IslCombinatorFn Fn) {
  return [&](const PVMap &PV0, const PVMap &PV1) -> PVMap {
    if (!PV0)
      return PV1;
    if (!PV1)
      return PV0;
    isl_map *PWAff0 = PV0;
    isl_map *PWAff1 = PV1;
    adjustDimensionsPlain(PWAff0, PWAff1);
    return PVMap(isl_map_coalesce(Fn(PWAff0, PWAff1)));
  };
}

PVMap &PVMap::intersect(const PVMap &PM) {
  Obj = getCombinatorFn(isl_map_intersect)(Obj, PM);
  return *this;
}

PVMap &PVMap::intersectDomain(const PVSet &Dom) {
  auto DomDim = Dom.getNumInputDimensions();
  auto MapDim = getNumInputDimensions();
  if (DomDim > MapDim)
    Obj = isl_map_add_dims(Obj, isl_dim_in, DomDim - MapDim);

  isl_set *Set = Dom;
  if (DomDim < MapDim)
    Set = isl_set_add_dims(Set, isl_dim_set, MapDim - DomDim);
  Obj = isl_map_intersect_domain(Obj, Set);
  return *this;
}

PVMap &PVMap::union_add(const PVMap &PM) {
  Obj = getCombinatorFn(isl_map_union)(Obj, PM);
  return *this;
}

int PVMap::getParameterPosition(const PVId &Id) const {
  int Pos = isl_map_find_dim_by_id(Obj, isl_dim_param, Id);
  return Pos;
}

void PVMap::eliminateParameter(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  assert(Pos >= 0);
  return eliminateParameter(Pos);
}

void PVMap::eliminateParameter(unsigned Pos) {
  Obj = isl_map_project_out(Obj, isl_dim_param, Pos, 1);
}

PVSet PVMap::getParameterSet() const {
  return isl_map_params(isl_map_copy(Obj));
}

void PVMap::equateParameters(unsigned Pos0, unsigned Pos1) {
  assert(Pos0 < getNumParameters() && Pos1 < getNumParameters());
  Obj = isl_map_equate(Obj, isl_dim_param, Pos0, isl_dim_param, Pos1);
}

void PVMap::equateParameters(const PVId &Id0, const PVId &Id1) {
  int Pos0 = getParameterPosition(Id0);
  int Pos1 = getParameterPosition(Id1);
  if (Pos0 < 0 || Pos1 < 0)
    return;
  return equateParameters(Pos0, Pos1);
}

PVId PVMap::getParameter(unsigned No) const {
  return PVId(isl_map_get_dim_id(Obj, isl_dim_param, No));
}

PVMap &PVMap::setParameter(unsigned No, const PVId &Id) {
  Obj = isl_map_set_dim_id(Obj, isl_dim_param, No, Id);
  return *this;
}

PVId PVMap::getInputId() const {
  return isl_map_get_tuple_id(Obj, isl_dim_in);
}

PVId PVMap::getOutputId() const {
  return isl_map_get_tuple_id(Obj, isl_dim_out);
}

void PVMap::getParameters(SmallVectorImpl<PVId> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++) {
    if (isl_map_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i));
  }
}

void PVMap::getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++)
    if (isl_map_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i).getPayloadAs<Value *>());
}

void PVMap::simplify(const PVAff &Aff) {
}

PVMap &PVMap::floordiv(int64_t V) {
  isl_val *Val = isl_val_int_from_si(getIslCtx(), V);
  Obj = isl_map_floordiv_val(Obj, Val);
  return *this;
}

PVMap &PVMap::addToOutputDimension(const PVMap &Other, unsigned Dim) {
  assert(Other.getNumOutputDimensions() == getNumOutputDimensions());
  assert(Other.getNumInputDimensions() == getNumInputDimensions());

  isl_map *OtherMap = Other;
  OtherMap = isl_map_align_params(OtherMap, getSpace());
  Obj = isl_map_align_params(Obj, isl_map_get_space(OtherMap));
  Obj = isl_map_sum(Obj, OtherMap);
  return *this;
}

void PVMap::dropUnusedParameters() {
  size_t NumParams = getNumParameters();
  for (size_t i = 0; i < NumParams; i++) {
    if (isl_map_involves_dims(Obj, isl_dim_param, i, 1))
      continue;

    Obj = isl_map_project_out(Obj, isl_dim_param, i, 1);
    i--;
    NumParams--;
  }
}

PVAff PVMap::getPVAffForDim(unsigned Dim) {
  isl_pw_multi_aff *PWMA = isl_pw_multi_aff_from_map(isl_map_copy(Obj));
  isl_pw_aff *PWA = isl_pw_multi_aff_get_pw_aff(PWMA, Dim);
  isl_pw_multi_aff_free(PWMA);
  return PWA;
}

std::string PVMap::str() const {
  char *cstr = isl_map_to_str(Obj);
  if (!cstr)
    return "null";
  std::string Result(cstr);
  ::free(cstr);
  return Result;
}

/* -------------------- PVAff ----------------------- */

static void adjustDimensionsPlain(isl_pw_aff *&PWA0, isl_pw_aff *&PWA1) {
  int DimIn0 = isl_pw_aff_dim(PWA0, isl_dim_in);
  int DimIn1 = isl_pw_aff_dim(PWA1, isl_dim_in);
  if (DimIn0 == DimIn1)
    return;

  auto *&PWA = (DimIn0 > DimIn1) ? PWA1 : PWA0;
  PWA = isl_pw_aff_add_dims(PWA, isl_dim_in, std::abs(DimIn0 - DimIn1));

  assert(isl_pw_aff_dim(PWA0, isl_dim_in) == isl_pw_aff_dim(PWA1, isl_dim_in));
  assert(isl_pw_aff_dim(PWA0, isl_dim_out) ==
         isl_pw_aff_dim(PWA1, isl_dim_out));
}

PVAff::PVAff(const PVId &Id) {
  auto *Space = isl_space_set_alloc(Id.getIslCtx(), 1, 0);
  Space = isl_space_set_dim_id(Space, isl_dim_param, 0, Id);

  auto *LS = isl_local_space_from_space(Space);
  Obj = isl_pw_aff_var_on_domain(LS, isl_dim_param, 0);
}

PVAff::PVAff(const PVBase &Base, int64_t ConstVal)
    : PVAff(PVSet::universe(Base), ConstVal) {}

PVAff::PVAff(const PVSet &S) {
  isl_set *Set = S;
  Set = isl_set_detect_equalities(Set);
  Set = isl_set_remove_redundancies(Set);
  Set = isl_set_coalesce(Set);
  Obj = isl_set_indicator_function(Set);
}

PVAff::PVAff(const PVSet &S, int64_t ConstVal) {
  isl_val *V = isl_val_int_from_si(S.getIslCtx(), ConstVal);
  isl_set *Set = S;
  Set = isl_set_detect_equalities(Set);
  Set = isl_set_remove_redundancies(Set);
  Set = isl_set_coalesce(Set);
  Obj = isl_pw_aff_val_on_domain(Set, V);
}

PVAff::PVAff(const PVBase &Base, unsigned CoeffPos, int64_t CoeffVal,
             const PVId &Id) {
  auto *Space = isl_space_set_alloc(Base.getIslCtx(), 1, CoeffPos + 1);
  auto *LSpace = isl_local_space_from_space(Space);
  auto *Aff = isl_aff_zero_on_domain(LSpace);
  Aff = isl_aff_set_coefficient_si(Aff, isl_dim_param, 0, CoeffVal);
  Aff = isl_aff_set_dim_id(Aff, isl_dim_param, 0, Id);
  Obj = isl_pw_aff_from_aff(Aff);
}

PVAff::PVAff(const PVAff &Other) : Obj(isl_pw_aff_copy(Other.Obj)) {}

PVAff::PVAff(const PVAff &Coeff, const PVId &Id, const PVBase &Base)
    : PVAff(Base, 0, 1, Id) {
  assert(Coeff.isInteger());
  multiply(Coeff);
}

PVAff::~PVAff() { isl_pw_aff_free(Obj); }

PVAff &PVAff::operator=(const PVAff &Other) {
  if (this != &Other) {
    isl_pw_aff_free(Obj);
    Obj = isl_pw_aff_copy(Other.Obj);
  }
  return *this;
}

PVAff &PVAff::operator=(PVAff &&Other) {
  if (this != &Other) {
    Obj = isl_pw_aff_free(Obj);
    std::swap(Obj, Other.Obj);
  }
  return *this;
}

bool PVAff::operator==(const PVAff &Other) {
  if (this == &Other)
    return true;
  if (!(*this) || !Other)
    return false;
  return isl_pw_aff_is_equal(Obj, Other.Obj);
}

PVAff::operator isl_pw_aff *() const { return isl_pw_aff_copy(Obj); }

isl_ctx *PVAff::getIslCtx() const { return isl_pw_aff_get_ctx(Obj); }

isl_space *PVAff::getSpace() const { return isl_pw_aff_get_space(Obj); }

size_t PVAff::getNumInputDimensions() const { return isl_pw_aff_dim(Obj, isl_dim_in); }

size_t PVAff::getNumOutputDimensions() const { return isl_pw_aff_dim(Obj, isl_dim_out); }

size_t PVAff::getNumParameters() const { return isl_pw_aff_dim(Obj, isl_dim_param); }

size_t PVAff::getNumPieces() const { return isl_pw_aff_n_piece(Obj); }

PVAff PVAff::extractFactor(const PVAff &Aff) const {
  SmallVector<PVId, 4> Parameters;
  Aff.getParameters(Parameters);
  assert(Parameters.size() == 1 && "TODO deal with more or less parameters");

  return const_cast<PVAff *>(this)->getParameterCoeff(Parameters.front());
}

int PVAff::getFactor(const PVAff &Aff) const {
  // TODO: This is just a test implementation and needs to be replaced!
  PVAff OtherAff = Aff;
  OtherAff.intersectDomain(getDomain());
  if (isl_pw_aff_is_equal(Obj, OtherAff))
    return 1;
  for (int i = 2; i < 10; i++) {
    isl_pw_aff *ScaledAffPWA =
        isl_pw_aff_scale_val(OtherAff, isl_val_int_from_si(getIslCtx(), i));
    bool Equal = isl_pw_aff_is_equal(Obj, ScaledAffPWA);
    isl_pw_aff_free(ScaledAffPWA);
    if (Equal)
      return i;
  }
  llvm_unreachable("TODO");
  return -1;
}

PVId PVAff::getParameter(unsigned No) const {
  return PVId(isl_pw_aff_get_dim_id(Obj, isl_dim_param, No));
}

PVAff &PVAff::setParameter(unsigned No, const PVId &Id) {
  Obj = isl_pw_aff_set_dim_id(Obj, isl_dim_param, No, Id);
  return *this;
}

void PVAff::getParameters(SmallVectorImpl<PVId> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++)
    if (isl_pw_aff_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i));
}

void PVAff::getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const {
  size_t NumParams = getNumParameters();
  Parameters.reserve(Parameters.size() + NumParams);
  for (size_t i = 0; i < NumParams; i++)
    if (isl_pw_aff_involves_dims(Obj, isl_dim_param, i, 1))
      Parameters.push_back(getParameter(i).getPayloadAs<Value *>());
}

int PVAff::getParameterPosition(const PVId &Id) const {
  auto *Params = isl_pw_aff_params(isl_pw_aff_copy(Obj));
  int Pos = isl_set_find_dim_by_id(Params, isl_dim_param, Id);
  isl_set_free(Params);
  return Pos;
}

void PVAff::eliminateParameter(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  assert(Pos >= 0);
  return eliminateParameter(Pos);
}

void PVAff::eliminateParameter(unsigned Pos) {
  Obj = isl_pw_aff_drop_dims(Obj, isl_dim_param, Pos, 1);
}

void PVAff::equateParameters(unsigned Pos0, unsigned Pos1) {
  assert(Pos0 < getNumParameters() && Pos1 < getNumParameters());
  Obj = isl_pw_aff_intersect_params(Obj, isl_set_equate(isl_pw_aff_params(Obj),
                                                        isl_dim_param, Pos0,
                                                        isl_dim_param, Pos1));
}

void PVAff::equateParameters(const PVId &Id0, const PVId &Id1) {
  int Pos0 = getParameterPosition(Id0);
  int Pos1 = getParameterPosition(Id1);
  if (Pos0 < 0 || Pos1 < 0)
    return;
  return equateParameters(Pos0, Pos1);
}

bool PVAff::involvesId(const PVId &Id) const {
  return getParameterPosition(Id) >= 0;
}

bool PVAff::involvesInput(unsigned Dim) const {
  assert(Dim <= getNumInputDimensions());
  return isl_pw_aff_involves_dims(Obj, isl_dim_in, Dim, 1);
}

void PVAff::addInputDims(unsigned Dims) {
  Obj = isl_pw_aff_add_dims(Obj, isl_dim_in, Dims);
}

void PVAff::insertInputDims(unsigned Pos, unsigned Dims) {
  Obj = isl_pw_aff_insert_dims(Obj, isl_dim_in, Pos, Dims);
}

void PVAff::dropInputDim(unsigned Dim) {
  Obj = isl_pw_aff_drop_dims(Obj, isl_dim_in, Dim, 1);
}

struct DropInfo {
  isl_pw_aff *NewObj;
  unsigned NumDims;
  unsigned Dims;
};

static isl_stat dropLastInputDimsHelper(isl_set *Dom, isl_aff *Aff, void *User) {
  DropInfo *DI = static_cast<DropInfo *>(User);
  Dom = isl_set_project_out(Dom, isl_dim_set, DI->NumDims - DI->Dims, DI->Dims);
  Aff = isl_aff_drop_dims(Aff, isl_dim_in, DI->NumDims - DI->Dims, DI->Dims);
  isl_pw_aff *PWA = isl_pw_aff_from_aff(Aff);
  PWA = isl_pw_aff_intersect_domain(PWA, Dom);
  DI->NewObj = isl_pw_aff_union_max(DI->NewObj, PWA);
  return isl_stat_ok;
}

void PVAff::dropLastInputDims(unsigned Dims) {
  isl_pw_aff *NewObj = isl_pw_aff_empty(isl_space_drop_dims(getSpace(), isl_dim_in, 0, Dims));
  unsigned NumDims = getNumInputDimensions();
  DropInfo DI = {NewObj, NumDims, Dims};
  isl_pw_aff_foreach_piece(Obj, dropLastInputDimsHelper, &DI);
  isl_pw_aff_free(Obj);
  Obj = isl_pw_aff_coalesce(DI.NewObj);
}

void PVAff::dropParameter(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  if (Pos >= 0)
    Obj = isl_pw_aff_drop_dims(Obj, isl_dim_param, Pos, 1);
}

void PVAff::dropUnusedParameters() {
  size_t NumParams = getNumParameters();
  for (size_t i = 0; i < NumParams; i++) {
    if (isl_pw_aff_involves_dims(Obj, isl_dim_param, i, 1))
      continue;

    Obj = isl_pw_aff_drop_dims(Obj, isl_dim_param, i, 1);
    i--;
    NumParams--;
  }
}

bool PVAff::isComplex() const {
  bool Complex = isl_pw_aff_n_piece(Obj) > PWA_N_PIECE_TRESHOLD;
  if (!Complex) {
    isl_set *Dom = getDomain();
    Complex = isl_set_n_basic_set(Dom) > DOMAIN_N_BASIC_SEK_TRESHOLD;
    isl_set_free(Dom);
  }
  return Complex;
}

bool PVAff::isInteger() const {
  return isl_pw_aff_n_piece(Obj) == 1 && isConstant();
}

bool PVAff::isConstant() const {
  return isl_pw_aff_is_cst(Obj);
}

bool PVAff::isEqual(const PVAff &Aff) const {
  isl_pw_aff *AffPWA = isl_pw_aff_add_dims(
      Aff, isl_dim_in, getNumInputDimensions() - Aff.getNumInputDimensions());
  return isl_pw_aff_is_equal(Obj, AffPWA);
}

PVSet PVAff::getEqualDomain(const PVAff &Aff) const {
  isl_pw_aff *AffPWA = isl_pw_aff_add_dims(
      Aff, isl_dim_in, getNumInputDimensions() - Aff.getNumInputDimensions());
  return isl_pw_aff_eq_set(isl_pw_aff_copy(Obj), AffPWA);
}

PVSet PVAff::getLessThanDomain(const PVAff &Aff) const {
  isl_pw_aff *AffPWA = isl_pw_aff_add_dims(
      Aff, isl_dim_in, getNumInputDimensions() - Aff.getNumInputDimensions());
  return isl_pw_aff_lt_set(isl_pw_aff_copy(Obj), AffPWA);
}

PVSet PVAff::getLessEqualDomain(const PVAff &Aff) const {
  isl_pw_aff *AffPWA = isl_pw_aff_add_dims(
      Aff, isl_dim_in, getNumInputDimensions() - Aff.getNumInputDimensions());
  return isl_pw_aff_le_set(isl_pw_aff_copy(Obj), AffPWA);
}
PVSet PVAff::getGreaterEqualDomain(const PVAff &Aff) const {
  isl_pw_aff *AffPWA = isl_pw_aff_add_dims(
      Aff, isl_dim_in, getNumInputDimensions() - Aff.getNumInputDimensions());
  return isl_pw_aff_ge_set(isl_pw_aff_copy(Obj), AffPWA);
}

PVSet PVAff::getDomain() const {
  return isl_pw_aff_domain(isl_pw_aff_copy(Obj));
}

PVAff &PVAff::fixParamDim(unsigned Dim, int64_t Value) {
  auto *Dom = isl_pw_aff_domain(isl_pw_aff_copy(Obj));
  Dom = isl_set_fix_si(Dom, isl_dim_param, Dim, Value);
  Obj = isl_pw_aff_intersect_domain(Obj, Dom);
  return *this;
}

PVAff &PVAff::fixInputDim(unsigned Dim, int64_t Value) {
  auto *Dom = isl_pw_aff_domain(isl_pw_aff_copy(Obj));
  Dom = isl_set_fix_si(Dom, isl_dim_set, Dim, Value);
  Obj = isl_pw_aff_intersect_domain(Obj, Dom);
  return *this;
}

PVAff &PVAff::equateInputDim(unsigned Dim, const PVId &Id) {
  int Pos = getParameterPosition(Id);
  if (Pos < 0) {
    Pos = getNumParameters();
    Obj = isl_pw_aff_add_dims(Obj, isl_dim_param, 1);
    Obj = isl_pw_aff_set_dim_id(Obj, isl_dim_param, Pos, Id);
  }
  assert(Pos >= 0);
  PVSet Dom = getDomain();
  Dom.equateInputDim(Dim, Id);
  intersectDomain(Dom);
  return *this;
}

PVAff &PVAff::setInputLowerBound(unsigned Dim,
                                 int64_t Value) {
  auto *Dom = isl_pw_aff_domain(isl_pw_aff_copy(Obj));
  Dom = isl_set_lower_bound_si(Dom, isl_dim_set, Dim, Value);
  Obj = isl_pw_aff_intersect_domain(Obj, Dom);
  return *this;
}

PVAff &PVAff::setInputId(const PVId &Id) {
  Obj = isl_pw_aff_set_tuple_id(Obj, isl_dim_in, Id);
  return *this;
}

PVAff &PVAff::setOutputId(const PVId &Id) {
  Obj = isl_pw_aff_set_tuple_id(Obj, isl_dim_out, Id);
  return *this;
}

PVAff &PVAff::intersectDomain(const PVSet &Dom) {
  auto DomDim = Dom.getNumInputDimensions();
  auto PWADim = getNumInputDimensions();
  if (DomDim > PWADim)
    Obj = isl_pw_aff_add_dims(Obj, isl_dim_in, DomDim - PWADim);

  isl_set *Set = Dom;
  if (DomDim < PWADim)
    Set = isl_set_add_dims(Set, isl_dim_set, PWADim - DomDim);
  Obj = isl_pw_aff_intersect_domain(Obj, Set);
  Obj = isl_pw_aff_coalesce(Obj);
  return *this;
}

PVAff &PVAff::maxInLastInputDims(unsigned Dims) {
  intersectDomain(getDomain().maxInLastInputDims(Dims));
  return *this;
}

PVAff &PVAff::simplify(const PVSet &S) {
  if (!Obj)
    return *this;

  isl_set *Set;
  int DimDiff = S.getNumInputDimensions() - getNumInputDimensions();
  if (DimDiff > 0) {
    unsigned Dim = S.getNumInputDimensions() - DimDiff;
    Set = S;
    Set = isl_set_project_out(Set, isl_dim_set, Dim, DimDiff);
  } else if (DimDiff < 0)
    Set = isl_set_add_dims(S, isl_dim_set, -DimDiff);
  else
    Set = S;

  isl_set *Dom = isl_pw_aff_domain(isl_pw_aff_copy(Obj));
  isl_set *InvDom = isl_set_subtract(Dom, Set);
  isl_set *InvCtx = isl_set_params(InvDom);
  isl_set *OkCtx = isl_set_complement(InvCtx);
  if (isl_set_is_empty(OkCtx)) {
    isl_set_free(OkCtx);
    return *this;
  }
  //S.intersect(isl_set_copy(OkCtx));
  OkCtx = isl_set_add_dims(OkCtx, isl_dim_set, getNumInputDimensions());
  Obj = isl_pw_aff_gist(Obj, OkCtx);
  Obj = isl_pw_aff_coalesce(Obj);
  dropUnusedParameters();
  return *this;
}

PVAff &PVAff::floordiv(int64_t V) {
  isl_val *Val = isl_val_int_from_si(getIslCtx(), V);
  Obj = isl_pw_aff_div(Obj, isl_pw_aff_val_on_domain(getDomain(), Val));
  return *this;
}

PVAff &PVAff::add(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_add)(Obj, PV);
  return *this;
}

PVAff &PVAff::sub(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_sub)(Obj, PV);
  return *this;
}

PVAff &PVAff::multiply(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_mul)(Obj, PV);
  return *this;
}

PVAff &PVAff::union_add(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_union_add)(Obj, PV);
  return *this;
}

PVAff &PVAff::union_min(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_union_min)(Obj, PV);
  return *this;
}

PVAff &PVAff::union_max(const PVAff &PV) {
  Obj = getCombinatorFn(isl_pw_aff_union_max)(Obj, PV);
  return *this;
}

PVAff &PVAff::select(const PVAff &PV0, const PVAff &PV1) {
  isl_pw_aff *PV0Obj = PV0;
  isl_pw_aff *PV1Obj = PV1;
  adjustDimensionsPlain(Obj, PV0Obj);
  adjustDimensionsPlain(PV0Obj, PV1Obj);
  adjustDimensionsPlain(Obj, PV1Obj);
  Obj = isl_pw_aff_cond(Obj, PV0Obj, PV1Obj);
  return *this;
}

PVSet PVAff::zeroSet() const {
  return isl_pw_aff_zero_set(isl_pw_aff_copy(Obj));
}

PVSet PVAff::nonZeroSet() const {
  return isl_pw_aff_non_zero_set(isl_pw_aff_copy(Obj));
}

struct ParameterInfo {
  PVAff Coeff;
  int Pos;
};

static isl_stat getParameterAff(isl_set *Domain, isl_aff *Aff, void *User) {
  auto *PI = static_cast<ParameterInfo*>(User);
  isl_val *CoeffVal = isl_aff_get_coefficient_val(Aff, isl_dim_param, PI->Pos);
  isl_aff *CoeffAff = isl_aff_zero_on_domain(isl_aff_get_domain_local_space(Aff));
  CoeffAff = isl_aff_set_constant_val(CoeffAff, CoeffVal);
  isl_pw_aff *CoeffPWA = isl_pw_aff_from_aff(CoeffAff);
  CoeffPWA = isl_pw_aff_intersect_domain(CoeffPWA, Domain);
  PI->Coeff.union_add(CoeffPWA);
  isl_aff_free(Aff);
  return isl_stat_ok;
}

PVAff PVAff::getParameterCoeff(const PVId &Id) {
  int Pos = getParameterPosition(Id);
  if (Pos < 0)
    return PVAff(Id, 0);

  ParameterInfo PI = {PVAff(Id, 0), Pos};
  isl_stat Success = isl_pw_aff_foreach_piece(Obj, getParameterAff, &PI);
  (void) Success;
  assert(Success == isl_stat_ok);

  return PI.Coeff;
}

struct EvolutionInfo {
  PVAff PWA;
  int LD;
  int Pos;
  long Val;
  PVSet &NegationSet;
};

static isl_stat adjustBackedgeVal(isl_set *D, isl_aff *Aff, void *User) {
  PVSet Domain(D);
  auto *EI = static_cast<EvolutionInfo*>(User);
  auto *ConstantVal = isl_aff_get_constant_val(Aff);
  if (isl_val_get_den_si(ConstantVal) != 1) {
    isl_val_free(ConstantVal);
    isl_aff_free(Aff);
    return isl_stat_error;
  }
  auto *PosVal = isl_aff_get_coefficient_val(Aff, isl_dim_param, EI->Pos);
  if (isl_val_is_zero(PosVal)) {
    isl_val_free(ConstantVal);
    EI->PWA.union_add(PVAff(isl_pw_aff_from_aff(Aff)).intersectDomain(Domain));
    return isl_stat_ok;
  }
  if (!isl_val_is_one(PosVal) && !isl_val_is_negone(PosVal)) {
    isl_val_free(ConstantVal);
    isl_aff_free(Aff);
    return isl_stat_error;
  }
  //long ValL = isl_val_get_num_si(ConstantVal);
  //if (EI->ConstantVal != LONG_MAX && EI->ConstantVal != ValL) {
    //isl_val_free(ConstantVal);
    //isl_aff_free(Aff);
    //return isl_stat_error;
  //}
  //EI->ConstantVal = ValL;

  Aff = isl_aff_drop_dims(Aff, isl_dim_param, EI->Pos, 1);
  assert(isl_aff_dim(Aff, isl_dim_in) < EI->LD + 1);
  Aff = isl_aff_add_dims(Aff, isl_dim_in,
                         EI->LD + 1 - isl_aff_dim(Aff, isl_dim_in));
  assert(Domain.getNumInputDimensions() < EI->LD + 1);
  Domain.addInputDims(EI->LD + 1 - Domain.getNumInputDimensions());
  Domain.projectParameter(Domain.getParameter(EI->Pos));
  if (isl_val_is_one(PosVal)) {
    // Aff = isl_aff_set_constant_val(Aff, isl_val_neg(isl_val_copy(ConstantVal)));
    Aff = isl_aff_set_constant_si(Aff, 0);
    Aff = isl_aff_add_coefficient_val(Aff, isl_dim_in, EI->LD, ConstantVal);
    PVAff Increment = isl_pw_aff_from_aff(Aff);
    Increment.intersectDomain(Domain);
    EI->PWA.union_add(Increment);
  } else {
    isl_val_free(ConstantVal);
    assert(isl_val_is_negone(PosVal));
    auto *LSpace = isl_aff_get_domain_local_space(Aff);
    auto *ModAff = isl_aff_zero_on_domain(LSpace);
    ModAff = isl_aff_set_coefficient_si(ModAff, isl_dim_in, EI->LD, 1);
    ModAff = isl_aff_mod_val(ModAff,
                              isl_val_int_from_si(isl_aff_get_ctx(ModAff), 2));
    PVSet EvenSet(isl_set_from_basic_set(isl_aff_zero_basic_set(ModAff)));
    EvenSet.intersect(Domain);
    PVSet OddSet(Domain);
    OddSet.subtract(EvenSet);
    EI->NegationSet.unify(OddSet);
    PVAff OddAff = isl_pw_aff_from_aff(isl_aff_neg(Aff));
    OddAff.intersectDomain(OddSet);
    PVAff EvenAff(EvenSet, 0);
    EI->PWA.union_add(EvenAff);
    EI->PWA.union_add(OddAff);
  }

  return isl_stat_ok;
}

PVAff PVAff::perPiecePHIEvolution(const PVId &Id, int LD, PVSet &NegationSet) const {
  int Pos = getParameterPosition(Id);
  if (Pos < 0)
    return *this;

  EvolutionInfo EI = {PVAff(), LD, Pos, LONG_MAX, NegationSet};
  isl_stat Success = isl_pw_aff_foreach_piece(Obj, adjustBackedgeVal, &EI);
  if (Success != isl_stat_ok)
    return PVAff();

  return EI.PWA;
}

struct IterationMoveInfo {
  PVAff PWA;
  unsigned Dim;
};

static isl_stat movePieceOneIteration(isl_set *D, isl_aff *Aff, void *User) {
  auto *IMI = static_cast<IterationMoveInfo *>(User);
  PVSet Domain(D);
  Domain.getNextIteration(IMI->Dim);
  IMI->PWA.union_add(PVAff(isl_pw_aff_from_aff(Aff)).intersectDomain(Domain));
  return isl_stat_ok;
}

PVAff PVAff::moveOneIteration(unsigned Dim) {
  IterationMoveInfo IMI = {PVAff(), Dim};
  isl_stat Success = isl_pw_aff_foreach_piece(Obj, movePieceOneIteration, &IMI);
  assert(Success == isl_stat_ok);
  return IMI.PWA;
}

PVAff PVAff::getExpPWA(const PVAff &PWA) {
  assert(isl_pw_aff_is_cst(PWA));

  auto *ExpPWA = isl_pw_aff_empty(isl_pw_aff_get_space(PWA));
  auto ExpPiece = [](isl_set *Dom, isl_aff *Aff, void *User) {
    auto **ExpPWA = static_cast<isl_pw_aff **>(User);
    assert(isl_aff_is_cst(Aff));

    auto *Val = isl_aff_get_constant_val(Aff);
    isl_aff_free(Aff);

    Val = isl_val_2exp(Val);
    auto *ExpPiecePWA = isl_pw_aff_val_on_domain(Dom, Val);
    *ExpPWA = isl_pw_aff_union_add(*ExpPWA, ExpPiecePWA);
    return isl_stat_ok;
  };

  auto Success = isl_pw_aff_foreach_piece(PWA, ExpPiece, &ExpPWA);
  (void)Success;
  assert(Success == isl_stat_ok);

  ExpPWA = isl_pw_aff_add_dims(ExpPWA, isl_dim_in, 0);
  return ExpPWA;
}

PVAff PVAff::createAdd(const PVAff &LHS, const PVAff &RHS) {
  return PVAff(LHS).add(RHS);
}

PVAff PVAff::createUnionAdd(const PVAff &LHS, const PVAff &RHS) {
  return getCombinatorFn(isl_pw_aff_union_add)(LHS, RHS);
}

PVAff PVAff::createSub(const PVAff &LHS, const PVAff &RHS) {
  return PVAff(LHS).sub(RHS);
}

PVAff PVAff::createMultiply(const PVAff &LHS, const PVAff &RHS) {
  return PVAff(LHS).multiply(RHS);
}

PVAff PVAff::createSDiv(const PVAff &LHS, const PVAff &RHS) {
  return getCombinatorFn(isl_pw_aff_tdiv_q)(LHS, RHS);
}

PVAff PVAff::createShiftLeft(const PVAff &LHS, const PVAff &RHS) {
  auto ExpRHS = getExpPWA(RHS);
  return ExpRHS.multiply(LHS);
}

PVAff PVAff::createSelect(const PVAff &CondPV, const PVAff &TruePV,
                    const PVAff &FalsePV) {
  return PVAff(CondPV).select(TruePV, FalsePV);
}

PVAff::CombinatorFn PVAff::getCombinatorFn(PVAff::IslCombinatorFn Fn) {
  return [&](const PVAff &PV0, const PVAff &PV1) -> PVAff {
    if (!PV0)
      return PV1;
    if (!PV1)
      return PV0;
    isl_pw_aff *PWAff0 = PV0;
    isl_pw_aff *PWAff1 = PV1;
    adjustDimensionsPlain(PWAff0, PWAff1);
    return PVAff(isl_pw_aff_coalesce(Fn(PWAff0, PWAff1)));
  };
}

/// Create the conditions under which @p L @p Pred @p R is true.
PVSet PVAff::buildConditionSet(ICmpInst::Predicate Pred, const PVAff &PV0,
                               const PVAff &PV1) {
  isl_pw_aff *L = PV0;
  isl_pw_aff *R = PV1;
  adjustDimensionsPlain(L, R);

  switch (Pred) {
  case ICmpInst::ICMP_EQ:
    return isl_pw_aff_eq_set(L, R);
  case ICmpInst::ICMP_NE:
    return isl_pw_aff_ne_set(L, R);
  case ICmpInst::ICMP_SLT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_SLE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_SGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_SGE:
    return isl_pw_aff_ge_set(L, R);
  case ICmpInst::ICMP_ULT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_UGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_ULE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_UGE:
    return isl_pw_aff_ge_set(L, R);
  default:
    llvm_unreachable("Non integer predicate not supported");
  }
}

PVAff PVAff::getBackEdgeTakenCountFromDomain(const PVSet &Dom) {
  assert(Dom.getNumInputDimensions() > 0);
  isl_map *M = isl_map_from_domain(Dom);
  M = isl_map_move_dims(M, isl_dim_out, 0, isl_dim_in,
                        Dom.getNumInputDimensions() - 1, 1);
  isl_pw_aff *MaxPWA = isl_pw_multi_aff_get_pw_aff(
      isl_map_lexmax_pw_multi_aff(isl_map_copy(M)), 0);
  isl_pw_aff *MinPWA =
      isl_pw_multi_aff_get_pw_aff(isl_map_lexmin_pw_multi_aff(M), 0);
  assert(isl_set_is_equal(isl_pw_aff_domain(isl_pw_aff_copy(MinPWA)),
                          isl_pw_aff_zero_set(isl_pw_aff_copy(MinPWA))));
  isl_pw_aff_free(MinPWA);
  return PVAff(MaxPWA);
}

PVMap &PVMap::preimage(const PVAff &PWA, bool Range) {
  if (Range)
    return preimageRange(PWA);
  return preimageDomain(PWA);
}

PVMap &PVMap::preimageDomain(const PVAff &PWA) {
  isl_pw_multi_aff *PWMA =  isl_pw_multi_aff_from_pw_aff(PWA);
  PWMA = isl_pw_multi_aff_set_tuple_id(PWMA, isl_dim_in, getInputId());
  Obj = isl_map_preimage_domain_pw_multi_aff(Obj, PWMA);
  dropUnusedParameters();
  return *this;
}

PVMap &PVMap::preimageRange(const PVAff &PWA) {
  isl_pw_multi_aff *PWMA =  isl_pw_multi_aff_from_pw_aff(PWA);
  PWMA = isl_pw_multi_aff_set_tuple_id(PWMA, isl_dim_out, getOutputId());
  Obj = isl_map_preimage_range_pw_multi_aff(Obj, PWMA);
  dropUnusedParameters();
  return *this;
}

std::string PVAff::str() const {
  char *cstr = isl_pw_aff_to_str(Obj);
  if (!cstr)
    return "null";
  std::string Result(cstr);
  ::free(cstr);
  return Result;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &OS, const PVBase &PV) {
  OS << PV.str();
  return OS;
}
