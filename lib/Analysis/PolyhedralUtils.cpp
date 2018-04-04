//===--- PolyhedralUtils.cpp --- Polyhedral Utility Classes & Functions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PolyhedralUtils.h"

#include "llvm/Analysis/PolyhedralValueInfo.h"
#include "llvm/Analysis/PolyhedralAccessInfo.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "polyhedral-utils"

