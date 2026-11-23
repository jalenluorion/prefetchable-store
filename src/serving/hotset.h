// Saved hot set for the serving worker (tools/dmserve.cpp --hotset FILE): at shutdown (the front end's quit, also its
// idle release) the ids of the experts in the cache are written to FILE, most valuable first; at start they are read
// back into the cache in the background, in batches, while no request waits (a request stops the refill between two
// batches). Only which records are in memory changes, never what is computed.
//
// File: "DMHS" u32 1 | i32 n_layers | i32 n_experts | i32 n | i32 ids[n]   (layer * n_experts + expert)
#pragma once
#include <atomic>
#include <string>
#include <vector>

#include "dmeng/lm.h"

namespace dm {

class HotSet {
 public:
  HotSet(LM& lm, std::string path, int batch = 64) : lm_(lm), path_(std::move(path)), batch_(batch) {}
  bool load();          // read the file (false: none / another model's)
  bool pending() const { return next_ < ids_.size() && !stopped_.load(); }
  // main thread, idle: refill one batch; returns the worker's "H {json}" line when the refill ends (done or stopped), else ""
  std::string step(bool request_waiting);
  void stop() { stopped_.store(true); }
  bool save() const;    // at shutdown
  size_t size() const { return ids_.size(); }

 private:
  LM& lm_;
  std::string path_;
  int batch_;
  std::vector<int32_t> ids_;
  size_t next_ = 0;
  double ms_ = 0;
  std::atomic<bool> stopped_{false};
  bool reported_ = false;
};

}  // namespace dm
