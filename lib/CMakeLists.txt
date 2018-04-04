set(LLVM_NO_RTTI 1)

add_subdirectory(Analysis/isl)

#add_llvm_library(Mekong OBJECT
#  Mekong.cpp
#  IslBuilder.cpp
#  MeKernelAnalysis.cpp
#  MeCodegen.cpp
#  MeKernelSubgrid.cpp
#  )
#
#if (LLVM_LINK_LLVM_DYLIB)
#  target_link_libraries(Mekong PUBLIC LLVM)
#else ()
#  target_link_libraries(Mekong PUBLIC
#    LLVMSupport
#    LLVMCore
#    LLVMScalarOpts
#    LLVMInstCombine
#    LLVMTransformUtils
#    LLVMAnalysis
#    LLVMipo
#    LLVMisl
#    LLVMMC
#    LLVMPasses
#    LLVMLinker
#    LLVMIRReader
#    # The libraries below are required for darwin: http://PR26392
#    LLVMBitReader
#    LLVMMCParser
#    LLVMObject
#    LLVMProfileData
#    LLVMTarget
#    LLVMVectorize
#    )
#endif()

add_llvm_loadable_module(LLVMMekong
  Mekong.cpp


  Analysis/PolyhedralAccessInfo.cpp
  Analysis/PolyhedralDependenceInfo.cpp
  Analysis/PolyhedralExpressionBuilder.cpp
  Analysis/PolyhedralUtils.cpp
  Analysis/PolyhedralValueInfo.cpp
  Analysis/PValue.cpp

  IslBuilder.cpp

  RegisterPasses.cpp
  MeModel.cpp
  MeKernelAnalysis.cpp
  MeCodegen.cpp
  MeKernelSubgrid.cpp
  )
set_target_properties(LLVMMekong
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
target_link_libraries(LLVMMekong LLVMisl)
