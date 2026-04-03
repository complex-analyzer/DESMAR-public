#include <iostream>
#include <thread>
#include <filesystem>
#include <sstream>
#include <chrono>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>

#include "Simulation.h"
#include "SimulationException.h"
#include "ParameterStorage.h"

#include "pugi/pugixml.hpp"
#include "dimcli/cli.h"
namespace fs = std::filesystem;

static bool silent = false;
void trace(const std::string& msg);
void traceLine(const std::string& msg);
void etrace(const std::string& msg);
void etraceLine(const std::string& msg);

void runSimulations(std::pair<unsigned int, unsigned int> runIndexRange, pugi::xml_node configurationNode, const ParameterStorage& parameterBase);

std::string getIPv6Address() {
    try {
        int sock = socket(AF_INET6, SOCK_DGRAM, 0);
        if (sock == -1) {
            return "::1";
        }
        
        struct sockaddr_in6 addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin6_family = AF_INET6;
        addr.sin6_port = htons(80);
        inet_pton(AF_INET6, "2001:4860:4860::8888", &addr.sin6_addr);
        
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
            close(sock);
            return "::1";
        }
        
        struct sockaddr_in6 localAddr;
        socklen_t addrLen = sizeof(localAddr);
        if (getsockname(sock, (struct sockaddr*)&localAddr, &addrLen) == -1) {
            close(sock);
            return "::1";
        }
        
        char ipStr[INET6_ADDRSTRLEN];
        if (inet_ntop(AF_INET6, &localAddr.sin6_addr, ipStr, INET6_ADDRSTRLEN) == NULL) {
            close(sock);
            return "::1";
        }
        
        close(sock);
        return std::string(ipStr);
    } catch (...) {
        return "::1";
    }
}

int main(int argc, char* argv[]) {
	auto __overall_start = std::chrono::steady_clock::now();
	
	// handle the command line argument parsing
	Dim::Cli cli;
	auto& simulationFile = cli.opt<std::string>("f file [file]", "Simulation.xml").desc("the simulation file to be used");

	auto& silencio = cli.opt<bool>("s silent", false).desc("supresses all verbose trace output, error traces remain enabled");
	auto& runId = cli.opt<std::string>("r run-id", "0").desc("Unique identifier for this simulation run");
	auto& threadCount = cli.opt<unsigned int>("t threads", 1).desc("The maximum number of threads to use for evaluating different runs");
	auto& simParameters = cli.optVec<std::string>("[params]").desc("Parameters to be passed to the simulation configuration & the simulation itself");
	

	
	if (!cli.parse(std::cerr, argc, argv)) {
		return cli.exitCode();
	}
	silent = *silencio;

	if (*threadCount == 0) {
		etraceLine("Error: can not run the simulation on 0 threads, specify 1 to use the main thread");
		return 1;
	}



	ParameterStorage parameterBase;
	parameterBase.set("runId", *runId);
	parameterBase.set("file", *simulationFile);
	for (const std::string& simParamPair : *simParameters) {
		auto pos = simParamPair.find('=');
		if (pos == std::string::npos) {
			etraceLine("Error: invalid param pair '" + simParamPair + "', expected 'name=value'");
			return 1;
		}

		const std::string name(simParamPair, 0, pos);
		const std::string value(simParamPair, pos+1, std::string::npos);
		if (name.empty()) {
			etraceLine("Error: invalid param pair '" + simParamPair + "', param name can not be empty");
			return 1;
		}

		parameterBase.set(name, value);
	}

	// say hello world, if not in silent mode
	traceLine("ExchangeSimulator v2.0");

	// parse the simulation configuration file
	pugi::xml_document doc;
	auto parse_result = doc.load_file(simulationFile->c_str());
	if (!parse_result) {
		etraceLine(" - error: could not parse the file '" + (*simulationFile) + "'");
		return 1;
	}
	traceLine(" - '" + *simulationFile + "' loaded successfully");

	// find the root node
	auto node = doc.child("Simulation");
	if (node.empty()) {
		etraceLine(" - error: when parsing '" + *simulationFile + "' - the 'Simulation' element was not found");
		return 1;
	}

	std::vector<std::pair<unsigned int, unsigned int>> loads;
	loads.push_back(std::make_pair(0, 1));
	
	try {
		try {
			traceLine(" - starting the simulations");

			auto __sim_wall_start = std::chrono::steady_clock::now();
			if (loads.size() == 1) {
				runSimulations(loads.front(), node, parameterBase);
			} else {
				std::vector<std::unique_ptr<std::thread>> threads;
				for (const auto& load : loads) {
					threads.push_back(std::make_unique<std::thread>(runSimulations, load, node, parameterBase));
				}

				for (const auto& threadptr : threads) {
					if (threadptr->joinable()) {
						threadptr->join();
					}
				}
			}
		
			auto __sim_wall_end = std::chrono::steady_clock::now();
			auto __sim_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__sim_wall_end - __sim_wall_start).count();
			std::cout << "[Perf] Single-process simulation wall time: "
			          << __sim_wall_ms << " ms (" << (__sim_wall_ms / 1000.0) << " s)" << std::endl;
			traceLine(" - all simulations finished, exiting");
		} catch (const SimulationException& ex) {
			std::cout << ex.what() << std::endl;
		}
	} catch (const std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}

	auto __overall_end = std::chrono::steady_clock::now();
	auto __overall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__overall_end - __overall_start).count();
	std::cout << "[Perf] Program total wall time: "
	          << __overall_ms << " ms (" << (__overall_ms / 1000.0) << " s)" << std::endl;

	return 0;
}



void runSimulations(std::pair<unsigned int, unsigned int> runIndexRange, pugi::xml_node configurationNode, const ParameterStorage& parameterBase) {
	for(unsigned int runIndex = runIndexRange.first; runIndex < runIndexRange.second; ++runIndex) {
		ParameterStorage* parameters = new ParameterStorage(parameterBase);
		parameters->set("runIndex", std::to_string(runIndex));
		Simulation* simulation = new Simulation(parameters);
		simulation->configure(configurationNode, "");
		
		simulation->simulate();
		
		delete simulation;
		delete parameters;
	}
}

void trace(const std::string& msg) {
	if (silent) {
		return;
	}
	std::cout << msg;
}
void traceLine(const std::string& msg) {
	if (silent) {
		return;
	}
	std::cout << msg << std::endl;
}
void etrace(const std::string& msg) {
	std::cerr << msg;
}
void etraceLine(const std::string& msg) {
	std::cerr << msg << std::endl;
}
