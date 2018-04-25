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

add_custom_command(OUTPUT "${LLVM_BINARY_DIR}/lib/libmekongrt.bc"
  COMMAND bin/clang++
    -std=c++11 -O3
    "-I${CMAKE_CURRENT_SOURCE_DIR}/Runtime/include"
    --cuda-host-only
    -c -emit-llvm -flto
    -o lib/libmekongrt.bc
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime/src/me-runtime.cu"
  
  WORKING_DIRECTORY "${LLVM_BINARY_DIR}"
  DEPENDS Runtime/src/me-runtime.cu clang
  VERBATIM
  )
add_custom_target(libmekongrt ALL DEPENDS
  "${LLVM_BINARY_DIR}/lib/libmekongrt.bc"
  )
install(FILES ${LLVM_BINARY_DIR}/lib/libmekongrt.bc
  DESTINATION lib)

file(COPY Runtime/include/me-runtime.h
  DESTINATION ${LLVM_BINARY_DIR}/include/mekong)
install(FILES ${LLVM_BINARY_DIR}/include/mekong/me-runtime.h
  DESTINATION include/mekong)
