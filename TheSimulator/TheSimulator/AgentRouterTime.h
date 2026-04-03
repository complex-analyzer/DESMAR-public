#pragma once

#include "Timestamp.h"

// Forward declaration to avoid pulling MPI headers (AgentRankRouter.h includes mpi.h).
class AgentRankRouter;

// Read router LVT without including AgentRankRouter.h in agent code.
Timestamp getRouterLVTNoMPI(const AgentRankRouter* router);


