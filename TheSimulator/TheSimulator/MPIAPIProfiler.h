#pragma once

#include <cstdint>
#include <string>
namespace DesmarMpiApiProfiler {
void Configure(bool enable, const std::string& outDir);

bool Enabled();
void StartWindow(const std::string& scope = "Unknown");
void StopAndDump();
void RecordCall(int funcId, uint64_t durationNs, uint64_t bytes);

bool ShouldCollect();
void RegisterThreadLabel(const std::string& label);

// Per-thread override: when enabled, this thread will NOT be profiled.
// This is used to exclude long-lived blocking collectives (e.g. learner doorbell MPI_Bcast)
// so StopAndDump() can't deadlock waiting for inflight calls to finish.
void SetIgnoreThisThread(bool ignore);
bool IsIgnoreThisThread();

} // namespace DesmarMpiApiProfiler


