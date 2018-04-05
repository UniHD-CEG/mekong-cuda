#ifndef __MEM_TRACKER_H
#define __MEM_TRACKER_H

#include "btree_map.h"
#include <iostream>
#include <functional>
#include <stdint.h>

// Memory efficient non-overlapping range tracker
// Function:
// 1. "Assign value V to range [x0, x1)"
// 2. "Which value was assigned to the point x?"
// 3. Sub ranges + assigned values for range [x0, x1)
//
// Features:
// - coalesce adjacent ranges
// - Data store is Google cpp-btree map (btree)
// - fast lookup: O(log n)
// - fast-ish insert: O(k * log n)
template<class T>
class MemTracker {
public:
  typedef typename btree::btree_map<int64_t, T> map_type;
  typedef typename btree::btree_map<int64_t, T>::iterator it_type;
  /** Creates new instance
   * @param noneVal  Value to return on queries that do not hit a range
   */
  MemTracker(T noneVal)
  : noneVal(noneVal)
  {}

  /** Updates given range with new value
   * @param lower  Lower bound (inclusive) of range
   * @param upper  Upper bound (exclusive) of range
   * @param val    Value to assign to range
   */
  void update(int64_t lower, int64_t upper, T val) {
    it_type it = ranges.lower_bound(lower);

    // store value of predecessor range
    T lowerVal = noneVal;
    if (it != ranges.end()) {
      lowerVal = it->second;
    }

    // delete, all bounds we find in our range
    it_type start = it;
    while (it != ranges.end() && it->first <= upper) {
      it = ranges.erase(it);
    }
    
    // we need a new upper bound when
    // 1. we reached the end, OR
    // 2. the next upper bound has a different value
    if (it == ranges.end() || it->second != val) {
      ranges.insert(std::make_pair(upper, val));
    }

    // restore old lower bound, if new value differs
    if (lowerVal != val) {
      ranges.insert(std::make_pair(lower, lowerVal));
    }
  }


  /** Get value of the range containing point
   * @param point  The point to query for
   *
   * @return  Value of the range containing point or noneVal
   */
  T query(int64_t point) const {
    it_type it = ranges.upper_bound(point);
    if (it == ranges.end()) {
      return noneVal;
    } else {
      return it->second;
    }
  }

  /** Enumerate all ranges contained in [start..end) and their tags
   * @param start   Lower bound (inclusive) of range
   * @param end     Upper bound (exclusive) of range
   * @param cb      Callback void(int64_t, int64_t, T), called once for each segment
   */
  typedef std::function<void(int64_t, int64_t, T)> callback;
  void queryRange(int64_t start, int64_t end, callback cb) const {
    it_type it = ranges.upper_bound(start);
    int64_t last_low = start;
    while (it != ranges.end() && last_low < end) {
      int64_t safeEnd = (it->first < end ? it->first : end);
      cb(last_low, safeEnd, it->second);
      last_low = safeEnd;
      it++;
    }
    if (last_low < end) {
      cb(last_low, end, noneVal);
    }
  }

  struct Chunk {
  public:
    Chunk()
    : start(-1), end(-1), initialized(false)
    {}

    int64_t start; // contains start of chunk (inclusive)
    int64_t end;   // contains end of chunk (exclusive)
    T       tag;   // tag of chunk
  private:
    friend bool MemTracker<T>::queryRange2(int64_t, int64_t, Chunk*);
    it_type it;
    bool    initialized;
  };

  /** Enumerate all ranges contained in [start..end) and their tags.
   * Instead of requiring a callback, the "Chunk" structs start/end/tag
   * values are updated.
   * @param start   Lower bound (inclusive) of range
   * @param end     Upper bound (exclusive) of range
   * @param chunk   Struct containing start end of a chunk
   *
   * @return        Chunk has valid data?
   */
  bool queryRange2(int64_t start, int64_t end, Chunk *chunk) {
    if (!chunk->initialized) {
      chunk->it = ranges.upper_bound(start);
      chunk->end = start;
      chunk->initialized = true;
    } else if (chunk->end >= end) {
      return false;
    }
    if (chunk->it != ranges.end()) {
      chunk->start = chunk->end;
      chunk->end = (chunk->it->first < end ? chunk->it->first : end);
      chunk->tag = chunk->it->second;
      chunk->it++;
    } else {
      chunk->start = chunk->end;
      chunk->end = end;
      chunk->tag = noneVal;
    }
    return true;
  }

  /** Get the size of the range map (mostly for debugging purposes)
   * @return  The number of elements in the range map
   */
  size_t size() {
    return ranges.size();
  }

  /** Get empty value
   * @return  The value assigned to points outside any range
   */
  T none() {
    return noneVal;
  }

private:
  T noneVal;
  map_type ranges;
};

#endif
