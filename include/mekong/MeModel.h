//=========--- MeModel.h -- Mekong Application Model ---*- C++ -*-==========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef __MEKONG_APP_MODEL
#define __MEKONG_APP_MODEL

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

#include <string>

namespace mekong {
struct Argument {
  std::string name;
  bool isPointer;
  bool isParameter; // is this argument a parameter to a read/write map?
  int bitsize;
  std::string typeName;

  int elementBitsize;
  std::string readMap;
  std::string writeMap;
  bool isReadInjective;
  bool isWriteInjective;
  llvm::SmallVector<std::string, 4> dimsizes;
};

struct Kernel {
  std::string name;
  std::string mangled_name;
  std::string partitioned_name;
  std::string partitioning;

  llvm::SmallVector<Argument,4> arguments;
};

struct App {
  llvm::SmallVector<Kernel,4> kernels;

  void serialize(llvm::raw_ostream &OS);
  bool deserialize(llvm::MemoryBuffer &MB);
};

extern llvm::cl::opt<std::string> ModelFile;
//extern llvm::cl::opt<bool> EnableAnalysis;
//extern llvm::cl::opt<bool> EnableTransform;

}

#endif
