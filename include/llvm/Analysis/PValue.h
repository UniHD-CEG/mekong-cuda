//===- PValue.h -------- isl C++ interface and wrapper --------*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// High-level interface for the genration and manipulation of polyhedral values
// (PV) with an isl backend. Abstractions for sets (isl_set -> PVSet), maps
// (isl_map -> PVMap) and piece-wise affine functions (isl_pw_aff -> PVAff)
// exist. Memory managment and space adjustments happen automatically.
//
// Parts of the code and ideas have been ported from the Polly [0] project.
//
// [0] http://polly.llvm.org/
//
// NOTE: This is a work in progress and therefor not a stable interface.
//
// TODO: Comments and descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PVALUE_H
#define LLVM_SUPPORT_PVALUE_H

#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

struct isl_id;
struct isl_ctx;
struct isl_set;
struct isl_map;
struct isl_space;
struct isl_pw_aff;

namespace llvm {
class PVAff;

template <typename PVType> struct PVLess {
  bool operator()(const PVType &lhs, const PVType &rhs) const;
};

class PVBase {
  friend class PVAff;
  friend class PVMap;
  friend class PVCtx;
  friend class PVSet;
  friend class PVId;

  virtual isl_ctx *getIslCtx() const = 0;
  virtual isl_space *getSpace() const = 0;

public:
  virtual ~PVBase() {};
  virtual std::string str() const;

  virtual bool isComplex() const { return false; }

  virtual operator bool() const = 0;

  /// Return @p Prefix + @p Middle + @p Suffix but Isl compatible.
  static std::string getIslCompatibleName(const std::string &Prefix,
                                          const std::string &Middle,
                                          const std::string &Suffix);
};

class PVCtx : public PVBase {
  isl_ctx *Obj;

public:
  PVCtx();
  PVCtx(isl_ctx *Ctx) : Obj(Ctx) {}
  PVCtx(const PVCtx &Other) : Obj(Other.Obj) {}

  isl_ctx *getIslCtx() const { return Obj; }
  isl_space *getSpace() const;

  std::string str() const;
  operator bool() const { return Obj != nullptr; }
};


class PVId : public PVBase {
  friend class PVAff;
  friend class PVMap;
  friend class PVSet;

  isl_id *Obj;

  operator isl_id *() const;

public:
  PVId() : Obj(nullptr) {}
  PVId(isl_id *Obj) : Obj(Obj) {}
  PVId(const PVId &Other);
  PVId(PVId &&Other);
  PVId(const PVBase &Base, const std::string &Name, void *Payload = nullptr);

  ~PVId();

  PVId &operator=(const PVId &Other);
  PVId &operator=(PVId &&Other);

  void *getPayload() const;
  template <typename T> T getPayloadAs() const {
    return static_cast<T>(getPayload());
  }

  isl_ctx *getIslCtx() const;
  isl_space *getSpace() const;
  std::string str() const;

  operator bool() const { return Obj != nullptr; }

  bool operator<(const PVId &Other) const;

  friend struct PVLess<PVId>;
};

class PVSet : public PVBase {
  friend class PVAff;
  friend class PVMap;

  isl_set *Obj;

  operator isl_set *() const;

protected:

  isl_ctx *getIslCtx() const;
  isl_space *getSpace() const;

public:

  PVSet() : Obj(nullptr) {}
  PVSet(const PVSet &Other);
  PVSet(PVSet &&Other) : Obj(Other.Obj) { Other.Obj = nullptr; }

  static PVSet createParameterRange(const PVId &Id, int LowerBound,
                                    int UpperBound);

  PVSet(isl_set *S);

  ~PVSet();

  PVSet &operator=(const PVSet &Other);
  PVSet &operator=(PVSet &&Other);
  operator bool() const { return Obj != nullptr; }

  isl_set * getObj() const;

  /// Return the number of dimensions of this set.
  size_t getNumInputDimensions() const;

  /// Return the number of parameters of this set.
  size_t getNumParameters() const;

  bool isEmpty() const;
  bool isUniverse() const;
  bool isBounded() const;

  bool isComplex() const;

  int getParameterPosition(const PVId &Id) const;

  bool hasLowerBoundForParam(const PVId &Id);
  PVSet &minForParam(const PVId &Id);

  void dropUnusedParameters();

  PVSet getParameterSet() const;

  PVSet &simplify(const PVSet &S);
  PVSet &simplifyParameters(const PVSet &S);

  PVSet &neg();
  PVSet &add(const PVSet &S);
  PVSet &sub(const PVSet &S);

  /// Unify this set inplace with @p S.
  PVSet &unify(const PVSet &S);

  /// Subtract ...
  PVSet &subtract(const PVSet &S);

  /// Intersect this set inplace with @p S.
  PVSet &intersect(const PVSet &S);

  PVSet &restrictToBoundedPart(unsigned Dim, PVSet *UnboundedPart = nullptr);

  ///
  PVSet &complement();

  PVSet &maxInLastInputDims(unsigned Dims);

  PVSet &addInputDims(unsigned Dims);
  PVSet &dropFirstInputDims(unsigned Dims);
  PVSet &dropLastInputDims(unsigned Dims);
  PVSet &dropDimsFrom(unsigned FirstDim);
  PVSet &fixInputDim(unsigned Dim, int64_t Value);
  PVSet &equateInputDim(unsigned Dim, const PVId &Id);
  PVSet &setInputLowerBound(unsigned Dim, int64_t Value);

  PVSet &preimage(const PVAff &PWA);
  PVSet &projectParameter(const PVId &Id);

  PVSet &getNextIteration(unsigned Dim);
  PVSet &getNextIterations(unsigned Dim);

  PVId getParameter(unsigned No) const;
  void getParameters(SmallVectorImpl<PVId> &Parameters) const;
  void getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const;

  static PVSet empty(const PVBase &Base);
  static PVSet empty(const PVBase &Base, unsigned Dims);
  static PVSet universe(const PVBase &Base);

  static PVSet unify(const PVSet &S0, const PVSet &S1);
  static PVSet intersect(const PVSet &S0, const PVSet &S1);

  std::string str() const;
};


class PVMap : public PVBase {
  friend class PVAff;
  friend class PVSet;

  isl_map *Obj;

  operator isl_map *() const;

protected:
  isl_ctx *getIslCtx() const;
  isl_space *getSpace() const;

public:

  PVMap() : Obj(nullptr) {}
  PVMap(const PVMap &Other);
  PVMap(PVMap &&Other) : Obj(Other.Obj) { Other.Obj = nullptr; }
  PVMap(ArrayRef<PVAff> Affs, const PVId &Id);
  PVMap(const PVAff &Coeff, const PVId &Id, const PVBase &Base);
  PVMap(const PVAff &Aff, long Factor);
  PVMap(const PVId &ValId, const PVId &InputId = PVId(),
        const PVId &OutputId = PVId());

  PVMap(isl_map *M);

  ~PVMap();

  PVMap &operator=(const PVMap &Other);
  PVMap &operator=(PVMap &&Other);
  operator bool() const { return Obj != nullptr; }

  bool isEmpty() const;
  bool isEqual(const PVMap &Map) const;

  /// Return the number of input dimensions of this function.
  size_t getNumInputDimensions() const;

  /// Return the number of output dimensions of this function.
  size_t getNumOutputDimensions() const;

  /// Return the number of parameters of this function.
  size_t getNumParameters() const;

  /// Use the intersection of the domain with the set @p S as new domain.
  ///
  /// @param Dom The set to intersect the domain with.
  ///
  /// @returns A reference to this object (*this).
  PVMap &intersectDomain(const PVSet &Dom);

  PVMap &intersect(const PVMap &Other);

  PVMap &addToOutputDimension(const PVMap &Other, unsigned Dim);

  int getParameterPosition(const PVId &Id) const;
  void eliminateParameter(unsigned Pos);
  void eliminateParameter(const PVId &Id);

  PVSet getParameterSet() const;

  void equateParameters(unsigned Pos0, unsigned Pos1);
  void equateParameters(const PVId &Id0, const PVId &Id1);

  PVId getParameter(unsigned No) const;
  PVMap &setParameter(unsigned No, const PVId &Id);

  PVId getInputId() const;
  PVId getOutputId() const;

  void dropUnusedParameters();

  PVAff getPVAffForDim(unsigned Dim);

  void getParameters(SmallVectorImpl<PVId> &Parameters) const;
  void getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const;

  void simplify(const PVAff &Aff);

  using IslCombinatorFn = std::function<isl_map *(isl_map *, isl_map *)>;
  using CombinatorFn = std::function<PVMap (const PVMap &, const PVMap &)>;
  static CombinatorFn getCombinatorFn(IslCombinatorFn Fn);

  PVMap &neg();
  PVMap &add(const PVMap &S);
  PVMap &sub(const PVMap &S);

  PVMap &union_add(const PVMap &PM);
  PVMap &floordiv(int64_t V);

  PVMap &preimage(const PVAff &PWA, bool Range = true);
  PVMap &preimageDomain(const PVAff &PWA);
  PVMap &preimageRange(const PVAff &PWA);

  std::string str() const;
};

class PVAff : public PVBase {
  friend class PVMap;
  friend class PVSet;

  isl_pw_aff *Obj;

  operator isl_pw_aff *() const;

protected:
  isl_ctx *getIslCtx() const;
  isl_space *getSpace() const;

public:

  /// Constructors
  ///{
  PVAff() : Obj(nullptr) {}
  PVAff(isl_pw_aff *Obj) : Obj(Obj) {}
  PVAff(const PVAff &Other);
  PVAff(PVAff &&Other) : Obj(Other.Obj) { Other.Obj = nullptr; }
  PVAff(const PVId &Id);
  PVAff(const PVBase &Base, int64_t ConstVal);
  PVAff(const PVSet &S);
  PVAff(const PVSet &S, int64_t ConstVal);
  PVAff(const PVBase &Base, unsigned CoeffPos, int64_t CoeffVal, const PVId &Id);
  PVAff(const PVAff &Coeff, const PVId &Id, const PVBase &Base);
  ///}

  ~PVAff();

  bool operator==(const PVAff &Other);
  bool operator!=(const PVAff &Other) { return !(*this == Other); }

  PVAff &operator=(const PVAff &Other);
  PVAff &operator=(PVAff &&Other);
  operator bool() const { return Obj != nullptr; }

  /// Return the number of input dimensions of this function.
  size_t getNumInputDimensions() const;

  /// Return the number of output dimensions of this function.
  size_t getNumOutputDimensions() const;

  /// Return the number of parameters of this function.
  size_t getNumParameters() const;

  /// Return the number of pieces
  size_t getNumPieces() const;

  bool isComplex() const;
  bool isInteger() const;
  bool isConstant() const;
  bool isEqual(const PVAff &Aff) const;

  int getParameterPosition(const PVId &Id) const;
  void eliminateParameter(unsigned Pos);
  void eliminateParameter(const PVId &Id);

  bool involvesId(const PVId &Id) const;
  bool involvesInput(unsigned Dim) const;

  void addInputDims(unsigned Dims);
  void insertInputDims(unsigned Pos, unsigned Dims);
  void dropInputDim(unsigned Dim);
  void dropLastInputDims(unsigned Dims);
  void dropParameter(const PVId &Id);
  void dropUnusedParameters();

  PVId getParameter(unsigned No) const;
  PVAff &setParameter(unsigned No, const PVId &Id);

  void getParameters(SmallVectorImpl<PVId> &Parameters) const;
  void getParameters(SmallVectorImpl<llvm::Value *> &Parameters) const;

  void equateParameters(unsigned Pos0, unsigned Pos1);
  void equateParameters(const PVId &Id0, const PVId &Id1);

  PVAff extractFactor(const PVAff &Aff) const;
  int getFactor(const PVAff &Aff) const;

  PVAff &add(const PVAff &PV);
  PVAff &sub(const PVAff &PV);
  PVAff &multiply(const PVAff &PV);
  PVAff &union_add(const PVAff &PV);
  PVAff &union_min(const PVAff &PV);
  PVAff &union_max(const PVAff &PV);

  PVAff &select(const PVAff &PV0, const PVAff &PV1);

  PVAff &fixParamDim(unsigned Dim, int64_t Value);
  PVAff &fixInputDim(unsigned Dim, int64_t Value);
  PVAff &equateInputDim(unsigned Dim, const PVId &Id);
  PVAff &setInputLowerBound(unsigned Dim, int64_t Value);

  PVAff &setInputId(const PVId &Id);
  PVAff &setOutputId(const PVId &Id);

  PVAff &floordiv(int64_t V);

  PVAff &maxInLastInputDims(unsigned Dims);

  PVSet zeroSet() const;
  PVSet nonZeroSet() const;

  PVAff getParameterCoeff(const PVId &Id);
  PVAff perPiecePHIEvolution(const PVId &Id, int LD,
                             PVSet &NegationSet) const;
  PVAff moveOneIteration(unsigned Dim);

  /// Use the intersection of the domain with the set @p S as new domain.
  ///
  /// @param S The set to intersect the domain with.
  ///
  /// @returns A reference to this object (*this).
  PVAff &intersectDomain(const PVSet &S);

  PVAff &simplify(const PVSet &S);
  PVAff &simplifyParameters(const PVSet &S);

  PVSet getEqualDomain(const PVAff &Aff) const;
  PVSet getLessThanDomain(const PVAff &Aff) const;
  PVSet getLessEqualDomain(const PVAff &Aff) const;
  PVSet getGreaterEqualDomain(const PVAff &Aff) const;
  PVSet getDomain() const;

  static PVAff getExpPWA(const PVAff &PWA);
  static PVAff createAdd(const PVAff &LHS, const PVAff &RHS);
  static PVAff createUnionAdd(const PVAff &LHS, const PVAff &RHS);
  static PVAff createSub(const PVAff &LHS, const PVAff &RHS);
  static PVAff createMultiply(const PVAff &LHS, const PVAff &RHS);
  static PVAff createSDiv(const PVAff &LHS, const PVAff &RHS);
  static PVAff createShiftLeft(const PVAff &LHS, const PVAff &RHS);
  static PVAff createSelect(const PVAff &CondPV, const PVAff &TruePV,
                            const PVAff &FalsePV);

  /// Create the conditions under which @p L @p Pred @p R is true.
  static PVSet buildConditionSet(ICmpInst::Predicate Pred, const PVAff &PV0,
                                 const PVAff &PV1);

  static PVAff getBackEdgeTakenCountFromDomain(const PVSet &Dom);

  /// The function type that is needed to combine two polyhedral
  /// representations.
  using IslCombinatorFn = std::function<isl_pw_aff *(isl_pw_aff *, isl_pw_aff *)>;
  using CombinatorFn = std::function<PVAff (const PVAff &, const PVAff &)>;
  static CombinatorFn getCombinatorFn(IslCombinatorFn Fn);

  std::string str() const;
};

template<typename PVType>
struct PVRewriter {
  virtual PVType rewrite(const PVType &Obj) {
    PVType Copy(Obj);
    rewrite(Copy);
    return Copy;
  };
  virtual void rewrite(PVType &Obj) {};
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const PVBase &PV);

} // end namespace llvm

#endif // LLVM_SUPPORT_PVALUE_H
