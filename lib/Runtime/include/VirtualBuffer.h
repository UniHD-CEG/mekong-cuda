// vim: ts=2 sts=2 sw=2 et ai
#ifndef __ME_RUNTIME_VIRTUAL_BUFFER
#define __ME_RUNTIME_VIRTUAL_BUFFER

#include "btree_memtracker.h"
#include <vector>

/*******************************************************************************
 * Virtual Buffer Class that replaces allocations from cudaMalloc.
 * Each virtual buffer represents multiple GPU buffers and can have references
 * to multiple host buffers.
 * Instead of directly holding any data, it consists of a tracker component that
 * has a list of intervals and buffer references that hold the most up to date
 * data of that interval.
 * Since the virtual buffer typically owns the GPU buffers but not host buffers
 * the following numbering scheme is used to distinguish the buffers:
 *
 * -inf .. -1 -> host buffers
 * 0          -> undefined
 * 1 .. inf   -> device buffers (buffer i belongs to GPU i-1)
 */
class VirtualBuffer {
public:
  VirtualBuffer(int numDevices)
    : devInstances(numDevices), hostInstances(0), tracker(0) {}

  void setDevInstance(int i, void* ptr) {
    assert(i >= 1 && i <= devInstances.size());
    devInstances[i-1] = ptr;
  }

  int findDevReference(void* ptr) {
    size_t num = devInstances.size();
    for (int i = 0; i < num; ++i) {
      if (devInstances[i] == ptr) {
        return -(i+1);
      }
    }
    return 0;
  }

  int findHostReference(void* ptr) {
    size_t num = hostInstances.size();
    for (int i = 0; i < num; ++i) {
      if (hostInstances[i] == ptr) {
        return -(i+1);
      }
    }
    return 0;
  }

  int findOrInsertHostReference(void *ptr) {
    int idx = findHostReference(ptr);
    if (idx != 0) return idx;
    hostInstances.push_back(ptr);
    return -(hostInstances.size());
  }

  void* getInstance(int i) {
    if (i > 0) {
      assert(i <= devInstances.size() && "invalid device instance");
      return devInstances[i-1];
    } else if (i < 0) {
      assert(-i <= hostInstances.size() && "invalid host reference");
      return hostInstances[-i-1];
    } else return nullptr;
  }

  const MemTracker<int>& getTracker() const {
    return tracker;
  }

  MemTracker<int>& getTracker() {
    return tracker;
  }
private:
  std::vector<void*> devInstances;
  std::vector<void*> hostInstances;
  MemTracker<int> tracker;
};

#endif
