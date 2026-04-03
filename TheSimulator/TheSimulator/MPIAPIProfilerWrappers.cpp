#include "MPIAPIProfiler.h"

#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>


extern "C" void __desmar_mpi_profiler_inflight_begin();
extern "C" void __desmar_mpi_profiler_inflight_end();

namespace {
using u64 = uint64_t;

static inline u64 now_ns() {
    return (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

static inline u64 datatype_nbytes(int count, MPI_Datatype dt) {
    if (count <= 0) return 0;
#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
    MPI_Count sz = 0;
    if (PMPI_Type_size_x(dt, &sz) != MPI_SUCCESS) return 0;
    return (u64)sz * (u64)count;
#else
    int sz = 0;
    if (PMPI_Type_size(dt, &sz) != MPI_SUCCESS) return 0;
    if (sz <= 0) return 0;
    return (u64)sz * (u64)count;
#endif
}

// funcId must be aligned with enum FuncId in MPIAPIProfiler.cpp
enum FuncId : int {
    F_MPI_Init = 0,
    F_MPI_Init_thread,
    F_MPI_Finalize,

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

    F_MPI_Barrier,
    F_MPI_Bcast,
    F_MPI_Allreduce,
    F_MPI_Reduce,

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

    F_MPI_Group_translate_ranks,
};

template <typename F>
static inline int timed(int funcId, u64 bytes, F&& f) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return f();
    }
    __desmar_mpi_profiler_inflight_begin();
    const u64 t0 = now_ns();
    int rc = f();
    const u64 t1 = now_ns();
    DesmarMpiApiProfiler::RecordCall(funcId, (t1 >= t0) ? (t1 - t0) : 0, bytes);
    __desmar_mpi_profiler_inflight_end();
    return rc;
}

} // namespace

extern "C" {

int MPI_Init(int* argc, char*** argv) {
    return timed(F_MPI_Init, 0, [&] { return PMPI_Init(argc, argv); });
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
    int rc = timed(F_MPI_Init_thread, 0, [&] { return PMPI_Init_thread(argc, argv, required, provided); });
    // Record rank information (for dump file name fallback, does not depend on MPI_Comm_rank again)
    int r = 0;
    if (PMPI_Comm_rank(MPI_COMM_WORLD, &r) == MPI_SUCCESS) {
        // Only used for profiler output file name fallback, does not affect simulation logic
        char buf[32] = {0};
        std::snprintf(buf, sizeof(buf), "%d", r);
        (void)setenv("DESMAR_MPI_RANK", buf, 1);
    }
    return rc;
}

int MPI_Finalize(void) {
    // Before finalize, if still in active window, dump is triggered by upper layer stop;
    // Here we still record the API time of finalize for academic statistics completeness.
    return timed(F_MPI_Finalize, 0, [&] { return PMPI_Finalize(); });
}

// --- P2P ---
int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Send(buf, count, datatype, dest, tag, comm);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Send, bytes, [&] { return PMPI_Send(buf, count, datatype, dest, tag, comm); });
}

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Isend, bytes, [&] { return PMPI_Isend(buf, count, datatype, dest, tag, comm, request); });
}

int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Recv, bytes, [&] { return PMPI_Recv(buf, count, datatype, source, tag, comm, status); });
}

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Irecv, bytes, [&] { return PMPI_Irecv(buf, count, datatype, source, tag, comm, request); });
}

int MPI_Wait(MPI_Request* request, MPI_Status* status) {
    return timed(F_MPI_Wait, 0, [&] { return PMPI_Wait(request, status); });
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]) {
    return timed(F_MPI_Waitall, 0, [&] { return PMPI_Waitall(count, array_of_requests, array_of_statuses); });
}

int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status) {
    return timed(F_MPI_Test, 0, [&] { return PMPI_Test(request, flag, status); });
}

int MPI_Testall(int count, MPI_Request array_of_requests[], int* flag, MPI_Status array_of_statuses[]) {
    return timed(F_MPI_Testall, 0, [&] { return PMPI_Testall(count, array_of_requests, flag, array_of_statuses); });
}

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status* status) {
    return timed(F_MPI_Probe, 0, [&] { return PMPI_Probe(source, tag, comm, status); });
}

int MPI_Iprobe(int source, int tag, MPI_Comm comm, int* flag, MPI_Status* status) {
    return timed(F_MPI_Iprobe, 0, [&] { return PMPI_Iprobe(source, tag, comm, flag, status); });
}

// --- Collectives ---
int MPI_Barrier(MPI_Comm comm) {
    return timed(F_MPI_Barrier, 0, [&] { return PMPI_Barrier(comm); });
}

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Bcast(buffer, count, datatype, root, comm);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Bcast, bytes, [&] { return PMPI_Bcast(buffer, count, datatype, root, comm); });
}

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Allreduce, bytes, [&] { return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm); });
}

int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    }
    const u64 bytes = datatype_nbytes(count, datatype);
    return timed(F_MPI_Reduce, bytes, [&] { return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm); });
}

// --- RMA ---
int MPI_Put(const void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
            MPI_Win win) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
    }
    const u64 bytes = datatype_nbytes(origin_count, origin_datatype);
    return timed(F_MPI_Put, bytes, [&] { return PMPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win); });
}

int MPI_Get(void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
            MPI_Win win) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win);
    }
    const u64 bytes = datatype_nbytes(origin_count, origin_datatype);
    return timed(F_MPI_Get, bytes, [&] { return PMPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win); });
}

int MPI_Accumulate(const void* origin_addr, int origin_count, MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype,
                   MPI_Op op, MPI_Win win) {
    if (!DesmarMpiApiProfiler::ShouldCollect()) {
        return PMPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win);
    }
    const u64 bytes = datatype_nbytes(origin_count, origin_datatype);
    return timed(F_MPI_Accumulate, bytes, [&] { return PMPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win); });
}

int MPI_Win_flush(int rank, MPI_Win win) {
    return timed(F_MPI_Win_flush, 0, [&] { return PMPI_Win_flush(rank, win); });
}

int MPI_Win_flush_all(MPI_Win win) {
    return timed(F_MPI_Win_flush_all, 0, [&] { return PMPI_Win_flush_all(win); });
}

int MPI_Win_sync(MPI_Win win) {
    return timed(F_MPI_Win_sync, 0, [&] { return PMPI_Win_sync(win); });
}

int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win) {
    return timed(F_MPI_Win_lock, 0, [&] { return PMPI_Win_lock(lock_type, rank, assert, win); });
}

int MPI_Win_unlock(int rank, MPI_Win win) {
    return timed(F_MPI_Win_unlock, 0, [&] { return PMPI_Win_unlock(rank, win); });
}

int MPI_Win_lock_all(int assert, MPI_Win win) {
    return timed(F_MPI_Win_lock_all, 0, [&] { return PMPI_Win_lock_all(assert, win); });
}

int MPI_Win_unlock_all(MPI_Win win) {
    return timed(F_MPI_Win_unlock_all, 0, [&] { return PMPI_Win_unlock_all(win); });
}

// --- Group helpers ---
int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[],
                              MPI_Group group2, int ranks2[]) {
    return timed(F_MPI_Group_translate_ranks, 0,
                 [&] { return PMPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2); });
}

} // extern "C"


