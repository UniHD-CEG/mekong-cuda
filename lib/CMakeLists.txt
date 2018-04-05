set(LLVM_NO_RTTI 1)

add_subdirectory(Analysis/isl)

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

if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(LLVMMekong PUBLIC LLVM)
else ()
  target_link_libraries(LLVMMekong PUBLIC
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMisl
    LLVMMC
    LLVMPasses
    LLVMLinker
    LLVMIRReader
    # The libraries below are required for darwin: http://PR26392
    LLVMBitReader
    LLVMMCParser
    LLVMObject
    LLVMProfileData
    LLVMTarget
    LLVMVectorize
    )
endif()

find_package(CUDAh)

add_library(mekongrt STATIC
  Runtime/src/me-runtime.cpp
  )
target_include_directories(mekongrt PUBLIC
  Runtime/include
  ${CUDAH_INCLUDE_DIRS}
  )
install(TARGETS mekongrt
  DESTINATION lib)

file(COPY Runtime/include/me-runtime.h
  DESTINATION ${LLVM_BINARY_DIR}/include/mekong)
install(FILES ${LLVM_BINARY_DIR}/include/mekong/me-runtime.h
  DESTINATION include/mekong)
