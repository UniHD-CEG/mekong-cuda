#include "mekong/MeModel.h"
#include "llvm/Support/YAMLTraits.h"

#include <stdint.h>

using llvm::yaml::SequenceTraits;
using llvm::yaml::MappingTraits;
using llvm::yaml::IO;

using mekong::App;
using mekong::Kernel;
using mekong::Argument;

template <>
struct llvm::yaml::MappingTraits<App> {
  static void mapping(IO &io, App &info) {
    io.mapRequired("kernels", info.kernels);
  }
};

template <>
struct llvm::yaml::SequenceTraits<llvm::SmallVector<Kernel, 4>> {
  static size_t size(IO &io, llvm::SmallVector<Kernel, 4> &seq) { return seq.size(); };
  static Kernel &element(IO &io, llvm::SmallVector<Kernel, 4> &seq, size_t idx) {
    while (idx >= seq.size()) { seq.push_back(Kernel()); }
    return seq[idx];
  }
};

template <>
struct llvm::yaml::MappingTraits<Kernel> {
  static void mapping(IO &io, Kernel &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("mangled-name", info.mangled_name);
    io.mapRequired("partitioned-name", info.partitioned_name);
    io.mapOptional("partitioning", info.partitioning);
    io.mapOptional("arguments", info.arguments);
  }
};

template <>
struct llvm::yaml::SequenceTraits<llvm::SmallVector<Argument, 4>> {
  static size_t size(IO &io, llvm::SmallVector<Argument, 4> &seq) { return seq.size(); };
  static Argument &element(IO &io, llvm::SmallVector<Argument, 4> &seq, size_t idx) {
    while (idx >= seq.size()) { seq.push_back(Argument()); }
    return seq[idx];
  }
};

template <>
struct llvm::yaml::MappingTraits<Argument> {
  static void mapping(IO &io, Argument &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("is-pointer", info.isPointer);
    io.mapRequired("is-parameter", info.isParameter);
    io.mapRequired("bitsize", info.bitsize);
    io.mapRequired("type-name", info.typeName);

    io.mapOptional("element-bitsize", info.elementBitsize);
    io.mapOptional("read-map", info.readMap);
    io.mapOptional("is-read-injective", info.isReadInjective);
    io.mapOptional("is-read-bounded", info.isReadBounded);
    io.mapOptional("write-map", info.writeMap);
    io.mapOptional("is-write-injective", info.isWriteInjective);
    io.mapOptional("is-write-bounded", info.isWriteBounded);
    io.mapOptional("dim-sizes", info.dimsizes);
  }
};

namespace mekong {

using namespace llvm;

cl::opt<std::string> ModelFile("me-model",
    cl::desc("Mekong application model"),
    cl::value_desc("filename"),
    cl::init(""));

//cl::opt<bool> EnableAnalysis("me-enable-analysis",
//    cl::desc("Enable Mekong Application Analysis"));
//cl::opt<bool> EnableTransform("me-enable-transform",
//    cl::desc("Enable Mekong Application Transformation"));

using llvm::yaml::Output;
using llvm::yaml::Input;

void App::serialize(raw_ostream &OS) {
  Output yout(OS);
  yout << *this;
}

bool App::deserialize(MemoryBuffer &MB) {
  Input yin(MB);
  this->kernels.clear();
  yin >> *this;
  return !yin.error();
}

}
