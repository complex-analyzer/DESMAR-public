#include "MPIAPIProfiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using u64 = uint64_t;

static inline u64 now_steady_ns() {
    return (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

enum FuncId : int {
    F_MPI_Init = 0,
    F_MPI_Init_thread,
    F_MPI_Finalize,

    // P2P
    F_MPI_Send,
    F_MPI_Isend,
    F_MPI_Recv,
    F_MPI_Irecv,
    F_MPI_Wait,
    F_MPI_Waitall,
    F_MPI_Test,
    F_MPI_Testall,
    F_MPI_Probe,
    F_MPI_Iprobe,

    // Collectives
    F_MPI_Barrier,
    F_MPI_Bcast,
    F_MPI_Allreduce,
    F_MPI_Reduce,

    // RMA
    F_MPI_Put,
    F_MPI_Get,
    F_MPI_Accumulate,
    F_MPI_Win_flush,
    F_MPI_Win_sync,
    F_MPI_Win_lock,
    F_MPI_Win_unlock,
    F_MPI_Win_lock_all,
    F_MPI_Win_unlock_all,
    F_MPI_Win_flush_all,

    // Group helpers
    F_MPI_Group_translate_ranks,

    F_kCount,
};

static inline const char* func_name(int id) {
    switch (id) {
        case F_MPI_Init: return "MPI_Init";
        case F_MPI_Init_thread: return "MPI_Init_thread";
        case F_MPI_Finalize: return "MPI_Finalize";
        case F_MPI_Send: return "MPI_Send";
        case F_MPI_Isend: return "MPI_Isend";
        case F_MPI_Recv: return "MPI_Recv";
        case F_MPI_Irecv: return "MPI_Irecv";
        case F_MPI_Wait: return "MPI_Wait";
        case F_MPI_Waitall: return "MPI_Waitall";
        case F_MPI_Test: return "MPI_Test";
        case F_MPI_Testall: return "MPI_Testall";
        case F_MPI_Probe: return "MPI_Probe";
        case F_MPI_Iprobe: return "MPI_Iprobe";
        case F_MPI_Barrier: return "MPI_Barrier";
        case F_MPI_Bcast: return "MPI_Bcast";
        case F_MPI_Allreduce: return "MPI_Allreduce";
        case F_MPI_Reduce: return "MPI_Reduce";
        case F_MPI_Put: return "MPI_Put";
        case F_MPI_Get: return "MPI_Get";
        case F_MPI_Accumulate: return "MPI_Accumulate";
        case F_MPI_Win_flush: return "MPI_Win_flush";
        case F_MPI_Win_sync: return "MPI_Win_sync";
        case F_MPI_Win_lock: return "MPI_Win_lock";
        case F_MPI_Win_unlock: return "MPI_Win_unlock";
        case F_MPI_Win_lock_all: return "MPI_Win_lock_all";
        case F_MPI_Win_unlock_all: return "MPI_Win_unlock_all";
        case F_MPI_Win_flush_all: return "MPI_Win_flush_all";
        case F_MPI_Group_translate_ranks: return "MPI_Group_translate_ranks";
        default: return "MPI_Unknown";
    }
}

static inline const char* group_name(int id) {
    switch (id) {
        case F_MPI_Send:
        case F_MPI_Isend:
        case F_MPI_Recv:
        case F_MPI_Irecv:
        case F_MPI_Wait:
        case F_MPI_Waitall:
        case F_MPI_Test:
        case F_MPI_Testall:
        case F_MPI_Probe:
        case F_MPI_Iprobe:
            return "P2P";
        case F_MPI_Barrier:
        case F_MPI_Bcast:
        case F_MPI_Allreduce:
        case F_MPI_Reduce:
            return "COLL";
        case F_MPI_Put:
        case F_MPI_Get:
        case F_MPI_Accumulate:
        case F_MPI_Win_flush:
        case F_MPI_Win_sync:
        case F_MPI_Win_lock:
        case F_MPI_Win_unlock:
        case F_MPI_Win_lock_all:
        case F_MPI_Win_unlock_all:
        case F_MPI_Win_flush_all:
            return "RMA";
        case F_MPI_Group_translate_ranks:
            return "GROUP";
        default:
            return "OTHER";
    }
}

struct Stat {
    u64 calls{0};
    u64 total_ns{0};
    u64 min_ns{~(u64)0};
    u64 max_ns{0};
};

static inline void add_stat(Stat& s, u64 dt) {
    s.calls += 1;
    s.total_ns += dt;
    s.min_ns = std::min(s.min_ns, dt);
    s.max_ns = std::max(s.max_ns, dt);
}

struct ThreadState {
    u64 epoch_seen{0};
    uint64_t tid_hash{0};
    std::string label;
    Stat stats[(size_t)F_kCount]{};
};

static std::mutex g_threads_mu;
static std::vector<ThreadState*> g_threads;

static std::atomic<bool> g_enabled{false};
static std::atomic<bool> g_active{false};
static std::atomic<u64> g_epoch{0};
static std::atomic<u64> g_inflight{0};

// Thread-local escape hatch: exclude selected threads from profiling.
// Rationale: some threads may block in long-lived collectives (e.g. learner doorbell MPI_Bcast),
// and StopAndDump() must not deadlock waiting for such inflight calls to finish.
static thread_local bool g_ignore_this_thread = false;

static std::mutex g_cfg_mu;
static std::string g_outDir = ".";
static std::string g_scope = "Unknown";

static std::atomic<u64> g_wall_start_ns{0};
static std::atomic<u64> g_wall_end_ns{0};

static ThreadState* tls_state() {
    thread_local ThreadState* st = nullptr;
    if (!st) {
        st = new ThreadState();
        st->tid_hash = (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
        std::lock_guard<std::mutex> lk(g_threads_mu);
        g_threads.push_back(st);
    }
    return st;
}

static void dump_csv_locked() {
    std::string outDir;
    std::string scope;
    {
        std::lock_guard<std::mutex> lk(g_cfg_mu);
        outDir = g_outDir;
        scope = g_scope;
    }

    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);

    const u64 wall_s = g_wall_start_ns.load(std::memory_order_acquire);
    const u64 wall_e = g_wall_end_ns.load(std::memory_order_acquire);
    const u64 wall_ms = (wall_e > wall_s) ? ((wall_e - wall_s) / 1'000'000ull) : 0;

    Stat merged[(size_t)F_kCount]{};
    for (size_t i = 0; i < (size_t)F_kCount; ++i) {
        merged[i].min_ns = ~(u64)0;
    }

    const u64 e = g_epoch.load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lk(g_threads_mu);
        for (ThreadState* ts : g_threads) {
            if (!ts) continue;
            if (ts->epoch_seen != e) continue;
            for (size_t i = 0; i < (size_t)F_kCount; ++i) {
                const Stat& s = ts->stats[i];
                if (s.calls == 0) continue;
                Stat& d = merged[i];
                d.calls += s.calls;
                d.total_ns += s.total_ns;
                d.min_ns = std::min(d.min_ns, s.min_ns);
                d.max_ns = std::max(d.max_ns, s.max_ns);
            }
        }
    }

    int rank = -1;
    if (const char* r = std::getenv("OMPI_COMM_WORLD_RANK"); r && *r) rank = std::atoi(r);
    else if (const char* r2 = std::getenv("PMI_RANK"); r2 && *r2) rank = std::atoi(r2);
    else if (const char* r3 = std::getenv("MV2_COMM_WORLD_RANK"); r3 && *r3) rank = std::atoi(r3);
    else if (const char* r4 = std::getenv("DESMAR_MPI_RANK"); r4 && *r4) rank = std::atoi(r4);

    if (rank < 0) rank = 0;

    const std::string path = (std::filesystem::path(outDir) / ("mpi_api_profile_rank" + std::to_string(rank) + ".csv")).string();
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    ofs << "rank,wall_start_ns,wall_end_ns,wall_ms,scope,group,func,calls,total_ns,avg_ns,min_ns,max_ns\n";
    for (int i = 0; i < F_kCount; ++i) {
        const Stat& s = merged[(size_t)i];
        if (s.calls == 0) continue;
        const u64 avg = s.total_ns / s.calls;
        const u64 min_ns = (s.min_ns == ~(u64)0) ? 0 : s.min_ns;
        ofs << rank << ","
            << wall_s << "," << wall_e << "," << wall_ms << ","
            << scope << ","
            << group_name(i) << ","
            << func_name(i) << ","
            << s.calls << ","
            << s.total_ns << ","
            << avg << ","
            << min_ns << ","
            << s.max_ns
            << "\n";
    }

    const std::string pathT = (std::filesystem::path(outDir) / ("mpi_api_profile_rank" + std::to_string(rank) + ".threads.csv")).string();
    std::ofstream ofst(pathT);
    if (!ofst.is_open()) return;
    ofst << "rank,tid_hash,tid_label,wall_start_ns,wall_end_ns,wall_ms,scope,group,func,calls,total_ns,avg_ns,min_ns,max_ns\n";
    {
        std::lock_guard<std::mutex> lk(g_threads_mu);
        for (ThreadState* ts : g_threads) {
            if (!ts) continue;
            if (ts->epoch_seen != e) continue;
            const uint64_t tid = ts->tid_hash;
            const std::string& tlabel = ts->label;
            for (int i = 0; i < F_kCount; ++i) {
                const Stat& s = ts->stats[(size_t)i];
                if (s.calls == 0) continue;
                const u64 avg = s.total_ns / s.calls;
                const u64 min_ns = (s.min_ns == ~(u64)0) ? 0 : s.min_ns;
                ofst << rank << ","
                     << tid << ","
                     << tlabel << ","
                     << wall_s << "," << wall_e << "," << wall_ms << ","
                     << scope << ","
                     << group_name(i) << ","
                     << func_name(i) << ","
                     << s.calls << ","
                     << s.total_ns << ","
                     << avg << ","
                     << min_ns << ","
                     << s.max_ns
                     << "\n";
            }
        }
    }
}

} // namespace

namespace DesmarMpiApiProfiler {

void Configure(bool enable, const std::string& outDir) {
    g_enabled.store(enable, std::memory_order_release);
    if (!enable) return;
    std::lock_guard<std::mutex> lk(g_cfg_mu);
    if (!outDir.empty()) g_outDir = outDir;
}

bool Enabled() {
    return g_enabled.load(std::memory_order_acquire);
}

bool ShouldCollect() {
    return Enabled() && g_active.load(std::memory_order_acquire) && !g_ignore_this_thread;
}

void SetIgnoreThisThread(bool ignore) {
    g_ignore_this_thread = ignore;
}

bool IsIgnoreThisThread() {
    return g_ignore_this_thread;
}

void StartWindow(const std::string& scope) {
    if (!Enabled()) return;
    {
        std::lock_guard<std::mutex> lk(g_cfg_mu);
        g_scope = scope.empty() ? "Unknown" : scope;
    }
    const u64 e = g_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    (void)e;
    g_wall_start_ns.store(now_steady_ns(), std::memory_order_release);
    g_wall_end_ns.store(0, std::memory_order_release);
    g_active.store(true, std::memory_order_release);
}

void StopAndDump() {
    if (!Enabled()) return;
    g_active.store(false, std::memory_order_release);
    while (g_inflight.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
    g_wall_end_ns.store(now_steady_ns(), std::memory_order_release);
    dump_csv_locked();
}

void RecordCall(int funcId, uint64_t durationNs, uint64_t /*bytes*/) {
    if (!ShouldCollect()) return;
    if (funcId < 0 || funcId >= F_kCount) return;

    ThreadState* ts = tls_state();
    const u64 e = g_epoch.load(std::memory_order_acquire);
    if (ts->epoch_seen != e) {
        ts->epoch_seen = e;
        for (size_t i = 0; i < (size_t)F_kCount; ++i) {
            ts->stats[i] = Stat{};
        }
    }
    add_stat(ts->stats[(size_t)funcId], durationNs);
}

void RegisterThreadLabel(const std::string& label) {
    if (!Enabled()) return;
    ThreadState* ts = tls_state();
    if (!ts) return;
    // Best-effort: store label for offline attribution. No locks needed because only the thread itself writes.
    if (!label.empty()) {
        ts->label = label;
    }
}

} // namespace DesmarMpiApiProfiler

extern "C" void __desmar_mpi_profiler_inflight_begin() { g_inflight.fetch_add(1, std::memory_order_acq_rel); }
extern "C" void __desmar_mpi_profiler_inflight_end() { g_inflight.fetch_sub(1, std::memory_order_acq_rel); }
