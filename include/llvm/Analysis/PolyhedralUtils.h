//===--- PolyhedralUtils.h --- Polyhedral Helper Classes --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef POLYHEDRAL_UTILS_H
#define POLYHEDRAL_UTILS_H

#include "llvm/Analysis/PValue.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

template<typename PVType, bool UseGlobalIdx = false>
struct NVVMRewriter : public PVRewriter<PVType> {
  enum NVVMDim {
    NVVMDIM_X = 0,
    NVVMDIM_Y = 1,
    NVVMDIM_Z = 2,
    NVVMDIM_W = 3,
    NVVMDIM_NONE = 4,
  };

  static constexpr unsigned NumNVVMDims = 4;
  NVVMDim NVVMDims[NumNVVMDims] = {NVVMDIM_X, NVVMDIM_Y, NVVMDIM_Z, NVVMDIM_W};
  std::string NVVMDimNames[NumNVVMDims] = {"x", "y", "z", "w"};

  bool isIntrinsic(Value *V, Intrinsic::ID IntrId) {
    auto *Intr = dyn_cast<IntrinsicInst>(V);
    return Intr && Intr->getIntrinsicID() == IntrId;
  }

  NVVMDim getBlockOffsetDim(Value *V) {
    auto *Inst = dyn_cast<Instruction>(V);
    if (!Inst)
      return NVVMDIM_NONE;

    if (Inst->getOpcode() != Instruction::Mul)
      return NVVMDIM_NONE;

    Value *Op0 = Inst->getOperand(0);
    Value *Op1 = Inst->getOperand(1);

    std::pair<Intrinsic::ID, Intrinsic::ID> IdPairs[] = {
        {Intrinsic::nvvm_read_ptx_sreg_ntid_x,
         Intrinsic::nvvm_read_ptx_sreg_ctaid_x},
        {Intrinsic::nvvm_read_ptx_sreg_ntid_y,
         Intrinsic::nvvm_read_ptx_sreg_ctaid_y},
        {Intrinsic::nvvm_read_ptx_sreg_ntid_z,
         Intrinsic::nvvm_read_ptx_sreg_ctaid_z},
        {Intrinsic::nvvm_read_ptx_sreg_ntid_w,
         Intrinsic::nvvm_read_ptx_sreg_ctaid_w}};

    for (unsigned d = 0; d < NumNVVMDims; d++) {
      auto IdPair = IdPairs[d];
      if ((isIntrinsic(Op0, IdPair.first) && isIntrinsic(Op1, IdPair.second)) ||
          (isIntrinsic(Op1, IdPair.first) && isIntrinsic(Op0, IdPair.second)))
        return NVVMDims[d];
    }

    return NVVMDIM_NONE;
  }

  std::string getCudaIntrinsicName(Value *V) {
    auto *Intr = dyn_cast<IntrinsicInst>(V);
    if (!Intr)
      return "";
    switch (Intr->getIntrinsicID()) {
      case Intrinsic::nvvm_read_ptx_sreg_tid_x: return "nvvm_tid_x";
      case Intrinsic::nvvm_read_ptx_sreg_tid_y: return "nvvm_tid_y";
      case Intrinsic::nvvm_read_ptx_sreg_tid_z: return "nvvm_tid_z";
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_x: return "nvvm_ctaid_x";
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_y: return "nvvm_ctaid_y";
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_z: return "nvvm_ctaid_z";
      case Intrinsic::nvvm_read_ptx_sreg_ntid_x: return "nvvm_ntid_x";
      case Intrinsic::nvvm_read_ptx_sreg_ntid_y: return "nvvm_ntid_y";
      case Intrinsic::nvvm_read_ptx_sreg_ntid_z: return "nvvm_ntid_z";
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_x: return "nvvm_nctaid_x";
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_y: return "nvvm_nctaid_y";
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_z: return "nvvm_nctaid_z";
    }
    return "";
  }

  virtual void rewrite(PVType &Obj) override {
    SmallVector<PVId, 4> ThreadIdCallsPerDim[NumNVVMDims];

    Intrinsic::ID ThreadIdIntrinsicIds[] = {
        Intrinsic::nvvm_read_ptx_sreg_tid_x,
        Intrinsic::nvvm_read_ptx_sreg_tid_y,
        Intrinsic::nvvm_read_ptx_sreg_tid_z,
        Intrinsic::nvvm_read_ptx_sreg_tid_w};

    for (unsigned d = 0, e = Obj.getNumParameters(); d < e; d++) {
      const PVId &Id = Obj.getParameter(d);
      auto *IdValue = Id.getPayloadAs<Value *>();
      for (unsigned u = 0; u < NumNVVMDims; u++) {
        Intrinsic::ID ThreadIdIntrinsicId = ThreadIdIntrinsicIds[u];
        if (!isIntrinsic(IdValue, ThreadIdIntrinsicId))
          continue;
        ThreadIdCallsPerDim[u].push_back(Id);
        break;
      }
    }

    for (const auto &ThreadIdCalls : ThreadIdCallsPerDim) {
      while (ThreadIdCalls.size() > 1) {
        Obj.equateParameters(ThreadIdCalls[0], ThreadIdCalls[1]);
        Obj.eliminateParameter(ThreadIdCalls[1]);
      }
    }

    PVId BlockOffset[NumNVVMDims];
    for (unsigned d = 0, e = Obj.getNumParameters(); d < e; d++) {
      const PVId &Id = Obj.getParameter(d);
      auto *IdValue = Id.getPayloadAs<Value *>();

      NVVMDim Dim =getBlockOffsetDim(IdValue);
      if (Dim >= NumNVVMDims)
        continue;

      assert(!BlockOffset[Dim] && "TODO: Handle multiple block "
                                              "offsets in the same "
                                              "dimension!\n");
      BlockOffset[Dim] =
          PVId(Id, "nvvm_block_offset_" + NVVMDimNames[Dim], IdValue);
      Obj.setParameter(d, BlockOffset[Dim]);
    }

    // must run after blockOffset resolution
    for (unsigned d = 0, e = Obj.getNumParameters(); d < e; ++d) {
      const PVId &Id = Obj.getParameter(d);
      Value *IdValue = Id.getPayloadAs<Value *>();
      std::string name = getCudaIntrinsicName(IdValue);
      if (name != "") {
        Obj.setParameter(d, PVId(Id, name, IdValue));
      }
    }

    if (!UseGlobalIdx)
      return;

    SmallVector<PVId, 4> ThreadIds;
    for (const auto &ThreadIdCalls : ThreadIdCallsPerDim)
      ThreadIds.push_back(ThreadIdCalls.empty() ? PVId() : ThreadIdCalls[0]);
    rewriteGlobalIdx(Obj, BlockOffset, ThreadIds);
  }

  void rewriteGlobalIdx(PVSet &Set, ArrayRef<PVId> BlockOffset,
                        ArrayRef<PVId> ThreadIds) {
    // TODO
  }

  void rewriteGlobalIdx(PVMap &Map, ArrayRef<PVId> BlockOffset,
                        ArrayRef<PVId> ThreadIds) {
    SmallVector<PVAff, 4> Affs;
    for (unsigned d = 0, e = Map.getNumOutputDimensions(); d < e ;d++) {
      Affs.push_back(Map.getPVAffForDim(d));
      rewriteGlobalIdx(Affs.back(), BlockOffset, ThreadIds);
    }
    Map = PVMap(Affs, Map.getOutputId());
  }

  void rewriteGlobalIdx(PVAff &Aff, ArrayRef<PVId> BlockOffset,
                        ArrayRef<PVId> ThreadIds) {
    for (unsigned d = 0; d < NumNVVMDims; d++) {
      if (!BlockOffset[d] || !ThreadIds[d])
        continue;

      const PVId &ThreadId = ThreadIds[d];
      PVId GlobalIdx =
          PVId(ThreadId, "nvvm_global_id_" + NVVMDimNames[d], nullptr);
      PVAff ThreadIdCoeff = Aff.getParameterCoeff(ThreadId);
      assert(ThreadIdCoeff.isInteger());
      PVAff BlockOffsetIdCoeff = Aff.getParameterCoeff(BlockOffset[d]);
      assert(BlockOffsetIdCoeff.isInteger());
      PVAff MinIdCoeff = ThreadIdCoeff;
      MinIdCoeff.union_min(BlockOffsetIdCoeff);
      assert(MinIdCoeff.isInteger());

      Aff = Aff.sub({MinIdCoeff, ThreadId, Aff});
      Aff = Aff.sub({MinIdCoeff, BlockOffset[d], Aff});

      Aff = Aff.add({MinIdCoeff, GlobalIdx, Aff});
    }

  }

private:
};


} // namespace llvm
#endif
