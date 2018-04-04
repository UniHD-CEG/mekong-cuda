#include "mekong/RegisterPasses.h"
#include "mekong/LinkAllPasses.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace {

/// Initialize Polly passes when library is loaded.
///
/// We use the constructor of a statically declared object to initialize the
/// different Polly passes right after the Polly library is loaded. This ensures
/// that the Polly passes are available e.g. in the 'opt' tool.
class StaticInitializer {
public:
  StaticInitializer() {
    llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
    mekong::initializeMekongPasses(Registry);
  }
};
static StaticInitializer InitializeEverything;
} // end of anonymous namespace.
