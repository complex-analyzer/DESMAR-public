#include "AgentRouterTime.h"

#include "AgentRankRouter.h"

Timestamp getRouterLVTNoMPI(const AgentRankRouter* router) {
    return router ? router->getLVT() : static_cast<Timestamp>(0);
}


