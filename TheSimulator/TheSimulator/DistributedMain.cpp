#include "DistributedSimulation.h"
#include "AgentRankRouter.h"
#include "CrossAgentRankRouter.h"
#include "CppAgentBatch.h"
#include "ParameterStorage.h"
#include "ProxySimulation.h"
#include "MPIAPIProfiler.h"
#include <mpi.h>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <cmath>
#if defined(__linux__)
#include <sched.h>
#endif
#include <pugixml.hpp>
#include "DateTimeConverter.h"
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstdint>
#include <iomanip>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "nlohmann/json.hpp"
#include "mtkahypar.h"
#include "FundamentalValueModel.h"

namespace {
    static constexpr int COMPONENT_COMM_CREATE_TAG = 1201;
    static constexpr int CPP_ONLY_COMM_CREATE_TAG = 1001;

    static inline bool isLeapYear(int y) {
        return ((y % 4) == 0 && (y % 100) != 0) || ((y % 400) == 0);
    }
    static inline int daysInMonth(int y, int m) {
        static const int mdays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
        if (m == 2) return isLeapYear(y) ? 29 : 28;
        return mdays[m - 1];
    }
    static inline bool parseYYYYMMDD(const std::string& s, int& y, int& m, int& d) {
        if (s.size() != 8) return false;
        for (char c : s) if (c < '0' || c > '9') return false;
        y = std::stoi(s.substr(0, 4));
        m = std::stoi(s.substr(4, 2));
        d = std::stoi(s.substr(6, 2));
        if (y < 1970 || m < 1 || m > 12 || d < 1 || d > daysInMonth(y, m)) return false;
        return true;
    }
    static inline std::string formatYYYYMMDD(int y, int m, int d) {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << y
            << std::setw(2) << std::setfill('0') << m
            << std::setw(2) << std::setfill('0') << d;
        return oss.str();
    }
    // 0=Sunday, 1=Monday, ... 6=Saturday (Sakamoto)
    static inline int weekdayYYYYMMDD(int y, int m, int d) {
        static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        if (m < 3) y -= 1;
        return (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;
    }
    static inline void addOneDay(int& y, int& m, int& d) {
        d += 1;
        int dim = daysInMonth(y, m);
        if (d > dim) {
            d = 1;
            m += 1;
            if (m > 12) {
                m = 1;
                y += 1;
            }
        }
    }
    static inline void subOneDay(int& y, int& m, int& d) {
        d -= 1;
        if (d < 1) {
            m -= 1;
            if (m < 1) {
                m = 12;
                y -= 1;
            }
            d = daysInMonth(y, m);
        }
    }
    static inline std::string nextTradingDateSkipWeekend(const std::string& curYYYYMMDD) {
        int y=0,m=0,d=0;
        if (!parseYYYYMMDD(curYYYYMMDD, y, m, d)) return curYYYYMMDD;
        do {
            addOneDay(y, m, d);
            int wd = weekdayYYYYMMDD(y, m, d);
            if (wd != 0 && wd != 6) break; // skip Sun/Sat
        } while (true);
        return formatYYYYMMDD(y, m, d);
    }
    static inline std::string prevTradingDateSkipWeekend(const std::string& curYYYYMMDD) {
        int y=0,m=0,d=0;
        if (!parseYYYYMMDD(curYYYYMMDD, y, m, d)) return curYYYYMMDD;
        do {
            subOneDay(y, m, d);
            int wd = weekdayYYYYMMDD(y, m, d);
            if (wd != 0 && wd != 6) break; // skip Sun/Sat
        } while (true);
        return formatYYYYMMDD(y, m, d);
    }

    // Trading-day index (0-based) from startDate -> date, skipping weekends.
    // If date == startDate => 0.
    // NOTE: intended for labeling/logging in process-level restart mode, so "epoch index" remains stable across runs.
    static inline int tradingDayIndexFromStart(const std::string& startYYYYMMDD, const std::string& dateYYYYMMDD) {
        if (startYYYYMMDD.empty() || dateYYYYMMDD.empty()) return 0;
        if (startYYYYMMDD == dateYYYYMMDD) return 0;
        int guard = 0;
        int idx = 0;
        std::string cur = startYYYYMMDD;
        while (cur != dateYYYYMMDD && guard++ < 10000) {
            cur = nextTradingDateSkipWeekend(cur);
            idx++;
        }
        if (cur != dateYYYYMMDD) return 0; // fallback if something odd
        return idx;
    }
    static inline uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }
    static inline uint64_t fnv1a64(const std::string& s) {
        uint64_t h = 14695981039346656037ull;
        for (unsigned char c : s) {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ull;
        }
        return h;
    }
    // Day seed should depend only on (master seed, date) so that it remains stable across
    // both in-process multi-epoch runs and process-level "hard restart" runs.
    static inline uint64_t deriveDaySeed(uint64_t master, const std::string& yyyymmdd) {
        uint64_t d = 0;
        try { d = static_cast<uint64_t>(std::stoull(yyyymmdd)); } catch (...) { d = 0; }
        uint64_t x = master ^ (d * 0xD6E8FEB86659FD93ull);
        return splitmix64(x);
    }
    static inline uint64_t deriveAssetDaySeed(uint64_t daySeed, const std::string& assetTag) {
        // Deterministic per-asset seed derived from the global day seed + asset identifier.
        // This keeps per-asset randomness different while remaining reproducible.
        return splitmix64(daySeed ^ (fnv1a64(assetTag) * 0x9E3779B97F4A7C15ull));
    }
    static inline int envInt(const char* name, int defVal) {
        const char* v = std::getenv(name);
        if (!v || !*v) return defVal;
        try { return std::stoi(v); } catch (...) { return defVal; }
    }
    // Strict bool env: only accepts "true" or "false" (case-insensitive).
    // Any other value is a hard error to prevent accidental experiment misconfiguration.
    static inline bool envBoolStrict(const char* name, bool defVal) {
        const char* v = std::getenv(name);
        if (!v || !*v) return defVal;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (s == "true") return true;
        if (s == "false") return false;
        throw std::runtime_error(std::string("Environment variable ") + name +
                                 " must be 'true' or 'false' (got '" + s + "')");
    }
    static inline uint64_t envU64(const char* name, uint64_t defVal) {
        const char* v = std::getenv(name);
        if (!v || !*v) return defVal;
        try { return static_cast<uint64_t>(std::stoull(v)); } catch (...) { return defVal; }
    }
    static inline double envDoubleInRange(const char* name, double defVal, double lo, double hi) {
        const char* v = std::getenv(name);
        if (!v || !*v) return defVal;
        double x = defVal;
        try { x = std::stod(v); } catch (...) {
            throw std::runtime_error(std::string("Environment variable ") + name +
                                     " must be a number (got '" + std::string(v) + "')");
        }
        if (!std::isfinite(x) || x < lo || x > hi) {
            throw std::runtime_error(std::string("Environment variable ") + name +
                                     " must be in [" + std::to_string(lo) + "," + std::to_string(hi) +
                                     "] (got '" + std::to_string(x) + "')");
        }
        return x;
    }

    // ===== Rank->Node mapping (OpenMPI-friendly) =====
    // We treat each MPI shared-memory domain (MPI_COMM_TYPE_SHARED) as a "node".
    // Returns a dense node id in [0, numNodes) for every MPI_COMM_WORLD rank.
    // Determinism: node ids are assigned by sorting each node leader's world rank.
    static inline std::vector<int> buildWorldRankToNodeIdMap(MPI_Comm world, int worldRank, int worldSize) {
        std::vector<int> rankToNode;
        rankToNode.assign(std::max(0, worldSize), 0);

        MPI_Comm nodeComm = MPI_COMM_NULL;
        MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);

        int localRank = 0;
        if (nodeComm != MPI_COMM_NULL) MPI_Comm_rank(nodeComm, &localRank);

        int nodeLeaderWorldRank = -1;
        if (localRank == 0) nodeLeaderWorldRank = worldRank;
        if (nodeComm != MPI_COMM_NULL) {
            MPI_Bcast(&nodeLeaderWorldRank, 1, MPI_INT, 0, nodeComm);
        }

        std::vector<int> leadersByRank;
        if (worldRank == 0) leadersByRank.assign(std::max(0, worldSize), -1);
        MPI_Gather(&nodeLeaderWorldRank, 1, MPI_INT,
                   (worldRank == 0 ? leadersByRank.data() : nullptr), 1, MPI_INT,
                   0, world);

        if (worldRank == 0) {
            std::vector<int> uniqueLeaders = leadersByRank;
            std::sort(uniqueLeaders.begin(), uniqueLeaders.end());
            uniqueLeaders.erase(std::unique(uniqueLeaders.begin(), uniqueLeaders.end()), uniqueLeaders.end());
            std::unordered_map<int,int> leaderToNodeId;
            leaderToNodeId.reserve(uniqueLeaders.size());
            for (size_t i = 0; i < uniqueLeaders.size(); ++i) leaderToNodeId[uniqueLeaders[i]] = static_cast<int>(i);
            for (int r = 0; r < worldSize; ++r) {
                auto it = leaderToNodeId.find(leadersByRank[r]);
                rankToNode[r] = (it == leaderToNodeId.end()) ? 0 : it->second;
            }
        }

        if (!rankToNode.empty() && worldSize > 0) {
            MPI_Bcast(rankToNode.data(), worldSize, MPI_INT, 0, world);
        }
        if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
        return rankToNode;
    }

    // Truncate/overwrite launcher-redirected stdout/stderr at the beginning of each epoch.
    // This keeps log files bounded in HPC runs (only the current epoch remains).
    static inline void overwriteStdoutStderrPerEpoch(
        const std::filesystem::path& logRoot,
        const std::string& assetOrCross,
        int globalRank
    ) {
        try {
            std::filesystem::path outdir = logRoot / assetOrCross / std::to_string(globalRank);
            std::error_code ec;
            std::filesystem::create_directories(outdir, ec);
            std::filesystem::path outPath = outdir / "stdout";
            std::filesystem::path errPath = outdir / "stderr";

            // Flush current buffers before truncating.
            std::cout.flush();
            std::cerr.flush();
            std::fflush(stdout);
            std::fflush(stderr);

            FILE* outFile = ::freopen(outPath.string().c_str(), "w", stdout);
            FILE* errFile = ::freopen(errPath.string().c_str(), "w", stderr);
            if (!outFile || !errFile) {
                return;
            }

            // Clear C++ stream states in case freopen changed underlying fds.
            std::cout.clear();
            std::cerr.clear();
        } catch (...) {
            // Best-effort; never break simulation due to logging.
        }
    }
    static inline bool endsWithSlash(const std::string& s) {
        if (s.empty()) return false;
        char c = s.back();
        return (c == '/' || c == '\\');
    }

    // Read the last TRADE price from OrderActionLogAgent csv:
    // Header: Timestamp,SourceAgent,ActionType,Direction,OrderID,Volume,Price,AdditionalInfo
    static inline bool readLastTradePriceFromOrderLogCsv(const std::string& path, double& outPrice) {
        outPrice = 0.0;
        std::ifstream in(path);
        if (!in.is_open()) return false;
        std::string line;
        bool first = true;
        double last = 0.0;
        bool found = false;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (first) { first = false; continue; } // header
            // Cheap filter
            if (line.find(",TRADE,") == std::string::npos) continue;
            // Parse first 7 columns (up to Price).
            // Timestamp,SourceAgent,ActionType,Direction,OrderID,Volume,Price,...
            size_t pos = 0;
            auto nextField = [&](std::string& out) -> bool {
                size_t comma = line.find(',', pos);
                if (comma == std::string::npos) { out = line.substr(pos); pos = line.size(); return true; }
                out = line.substr(pos, comma - pos);
                pos = comma + 1;
                return true;
            };
            std::string f0,f1,f2,f3,f4,f5,f6;
            nextField(f0); nextField(f1); nextField(f2); nextField(f3); nextField(f4); nextField(f5); nextField(f6);
            if (f2 != "TRADE") continue;
            try {
                double px = std::stod(f6);
                if (px > 0.0 && std::isfinite(px)) { last = px; found = true; }
            } catch (...) {
            }
        }
        if (!found) return false;
        outPrice = last;
        return true;
    }

    // Update SetupAgent's bid/ask based on previous trading day's last TRADE price.
    // tickCents: e.g. 1 => 0.01
    // spreadTicks: e.g. 2 => +/- 2 ticks around mid
    static inline bool updateSetupAgentPricesFromPrevClose(
        pugi::xml_node rootNode,
        const std::string& asset,
        const std::string& prevDateYYYYMMDD,
        int tickCents = 1,
        int spreadTicks = 2
    ) {
        if (asset.empty() || prevDateYYYYMMDD.empty()) return false;
        // Find OrderActionLogAgent output directory (prefer explicit outputFile attr).
        std::string outDir;
        if (auto core = rootNode.child("CoreRank")) {
            if (auto logAgent = core.child("OrderActionLogAgent")) {
                auto att = logAgent.attribute("outputFile");
                if (!att.empty()) outDir = att.as_string();
            }
        }
        std::string csvPath;
        if (!outDir.empty() && endsWithSlash(outDir)) {
            csvPath = outDir + asset + "_ordertracker_" + prevDateYYYYMMDD + ".csv";
        } else if (!outDir.empty()) {
            // If outputFile is a file path, use it directly (cannot infer previous day suffix reliably).
            // Fallback: keep default behavior by not changing SetupAgent.
            return false;
        } else {
            // Default path in OrderActionLogAgent when outputFile not set:
            // data/<date>_result/<exchange>_ordertracker_<date>.csv
            csvPath = "data/" + prevDateYYYYMMDD + "_result/" + asset + "_ordertracker_" + prevDateYYYYMMDD + ".csv";
        }

        double lastTradePx = 0.0;
        if (!readLastTradePriceFromOrderLogCsv(csvPath, lastTradePx)) {
            return false;
        }

        // Convert to cents (Money(0, cents)). CSV price is in "yuan" (double).
        const int midCents = static_cast<int>(std::llround(lastTradePx * 100.0));
        const int delta = std::max(0, spreadTicks) * std::max(1, tickCents);
        const int bidCents = midCents - delta;
        const int askCents = midCents + delta;

        // Update SetupAgent attributes in the in-memory XML before Simulation::configure().
        if (auto core = rootNode.child("CoreRank")) {
            if (auto setup = core.child("SetupAgent")) {
                if (!setup.attribute("bidPrice").empty()) setup.attribute("bidPrice").set_value(bidCents);
                else setup.append_attribute("bidPrice").set_value(bidCents);
                if (!setup.attribute("askPrice").empty()) setup.attribute("askPrice").set_value(askCents);
                else setup.append_attribute("askPrice").set_value(askCents);
                return true;
            }
        }
        return false;
    }

    // ===== DataFactory =====
    struct OhlcvRow {
        long long start_ts{0};
        double open{0.0};
        double high{0.0};
        double low{0.0};
        double close{0.0};
        unsigned long long volume{0};
    };

    static inline bool endsWith(const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    }

    static inline std::filesystem::path dataFactoryRootDir() {
        return std::filesystem::path("data") / "agent_outputs" / "DataFactoryAgent";
    }
    static inline std::filesystem::path dataFactoryMergedDir() {
        return dataFactoryRootDir() / "_merged";
    }

    static inline bool parseOhlcvFilenameKey(
        const std::string& filename,
        const std::string& yyyymmdd,
        std::string& outKey /* output filename in merged dir */
    ) {
        // Expect: <minutes>m_<asset>_<yyyymmdd>.csv
        const std::string suffix = std::string("_") + yyyymmdd + ".csv";
        if (!endsWith(filename, suffix)) return false;
        std::string left = filename.substr(0, filename.size() - suffix.size()); // <minutes>m_<asset>
        size_t pos = left.find("m_");
        if (pos == std::string::npos) return false;
        if (pos == 0) return false;
        // minutes digits are [0,pos)
        for (size_t i = 0; i < pos; ++i) {
            if (left[i] < '0' || left[i] > '9') return false;
        }
        // require at least one char after "m_"
        if (pos + 2 >= left.size()) return false;
        outKey = filename; // keep identical output filename
        return true;
    }

    static inline bool parseLobFilenameKey(
        const std::string& filename,
        const std::string& yyyymmdd,
        std::string& outKey /* output filename in merged dir */
    ) {
        // Expect: lob_<multiple>x_<asset>_<yyyymmdd>.csv
        const std::string suffix = std::string("_") + yyyymmdd + ".csv";
        if (!endsWith(filename, suffix)) return false;
        if (filename.rfind("lob_", 0) != 0) return false;
        outKey = filename;
        return true;
    }

    static inline void mergeDedupDataFactoryToMergedDir(const std::string& yyyymmdd) {
        namespace fs = std::filesystem;
        fs::path root = dataFactoryRootDir();
        fs::path merged = dataFactoryMergedDir();
        std::error_code ec;
        fs::create_directories(merged, ec);

        if (!fs::exists(root) || !fs::is_directory(root)) {
            std::cout << "[DataFactory][Merge] root missing: " << root.string() << " (skip)" << std::endl;
            return;
        }

        // ===== OHLCV =====
        // key(filename) -> (start_ts -> row)
        std::unordered_map<std::string, std::map<long long, OhlcvRow>> grouped;

        // ===== LOB =====
        // key(filename) -> { fieldCountMax, rows(ts -> fields...) }
        struct LobGrouped {
            size_t fieldCountMax{0}; // number of fields after "ts"
            // Store the CSV tail ("field1,field2,...") as a single string to avoid huge numbers of small allocations.
            std::map<long long, std::string> rows; // ts -> csv tail
        };
        std::unordered_map<std::string, LobGrouped> lobGrouped;

        size_t scannedFiles = 0;
        size_t usedFiles = 0;

        for (const auto& sub : fs::directory_iterator(root)) {
            if (!sub.is_directory()) continue;
            const std::string dirName = sub.path().filename().string();
            if (dirName == "_merged") continue;

            for (const auto& f : fs::directory_iterator(sub.path())) {
                if (!f.is_regular_file()) continue;
                const std::string fname = f.path().filename().string();
                scannedFiles += 1;

                // OHLCV
                std::string keyO;
                if (parseOhlcvFilenameKey(fname, yyyymmdd, keyO)) {
                    std::ifstream ifs(f.path());
                    if (!ifs.is_open()) continue;
                    usedFiles += 1;

                    std::string line;
                    bool first = true;
                    while (std::getline(ifs, line)) {
                        if (line.empty()) continue;
                        if (first) {
                            first = false;
                            // header
                            if (line.find("start_ts") != std::string::npos) continue;
                        }
                        // start_ts,open,high,low,close,volume
                        std::istringstream ss(line);
                        std::string t0,t1,t2,t3,t4,t5;
                        if (!std::getline(ss, t0, ',')) continue;
                        if (!std::getline(ss, t1, ',')) continue;
                        if (!std::getline(ss, t2, ',')) continue;
                        if (!std::getline(ss, t3, ',')) continue;
                        if (!std::getline(ss, t4, ',')) continue;
                        if (!std::getline(ss, t5, ',')) continue;
                        try {
                            OhlcvRow r;
                            r.start_ts = std::stoll(t0);
                            r.open = std::stod(t1);
                            r.high = std::stod(t2);
                            r.low = std::stod(t3);
                            r.close = std::stod(t4);
                            r.volume = static_cast<unsigned long long>(std::stoull(t5));
                            auto& mp = grouped[keyO];
                            if (mp.find(r.start_ts) == mp.end()) {
                                mp.emplace(r.start_ts, r);
                            }
                        } catch (...) {
                            continue;
                        }
                    }
                    continue;
                }

                // LOB
                std::string keyL;
                if (parseLobFilenameKey(fname, yyyymmdd, keyL)) {
                    std::ifstream ifs(f.path());
                    if (!ifs.is_open()) continue;
                    usedFiles += 1;

                    std::string line;
                    bool first = true;
                    size_t fileFieldCount = 0;
                    while (std::getline(ifs, line)) {
                        if (line.empty()) continue;
                        if (first) {
                            first = false;
                            // header: ts,<4*depth columns>
                            // count columns by commas
                            size_t cols = 1;
                            for (char c : line) if (c == ',') cols++;
                            if (cols >= 1) {
                                fileFieldCount = (cols >= 1) ? (cols - 1) : 0;
                            }
                            // If first line isn't header, we'll still parse data below; keep fileFieldCount as 0 for now.
                            if (line.find("ts") != std::string::npos) {
                                // header line, proceed to next
                                continue;
                            }
                            // fallthrough: treat it as data line
                        }

                        // data row: ts,field1,field2,...
                        size_t commaPos = line.find(',');
                        if (commaPos == std::string::npos) continue;
                        long long ts = 0;
                        try { ts = std::stoll(line.substr(0, commaPos)); } catch (...) { continue; }
                        std::string tail = line.substr(commaPos + 1); // fields-only CSV tail
                        // field count for this row (exclude ts)
                        size_t rowFieldCount = tail.empty() ? 0 : (static_cast<size_t>(std::count(tail.begin(), tail.end(), ',')) + 1);

                        auto& g = lobGrouped[keyL];
                        // Determine max field count
                        if (fileFieldCount == 0) fileFieldCount = rowFieldCount;
                        g.fieldCountMax = std::max(g.fieldCountMax, std::max(fileFieldCount, rowFieldCount));
                        if (g.rows.find(ts) == g.rows.end()) {
                            g.rows.emplace(ts, std::move(tail));
                        }
                    }
                    continue;
                }
            }
        }

        size_t outFilesOhlcv = 0;
        size_t outRowsOhlcv = 0;
        for (auto& kv : grouped) {
            const std::string& outName = kv.first;
            const auto& rows = kv.second;
            if (rows.empty()) continue;
            fs::path outPath = merged / outName;
            // overwrite
            std::ofstream ofs(outPath, std::ios::out | std::ios::trunc);
            if (!ofs.is_open()) continue;
            ofs << "start_ts,open,high,low,close,volume\n";
            for (const auto& it : rows) {
                const auto& r = it.second;
                ofs << r.start_ts << ","
                    << std::fixed << std::setprecision(8) << r.open << ","
                    << r.high << ","
                    << r.low << ","
                    << r.close << ","
                    << r.volume << "\n";
                outRowsOhlcv += 1;
            }
            outFilesOhlcv += 1;
        }

        size_t outFilesLob = 0;
        size_t outRowsLob = 0;
        auto write_first_n_fields = [](std::ostream& os, const std::string& s, size_t nFields) {
            // Write at most nFields comma-separated fields from s.
            // Caller controls leading comma.
            if (nFields == 0 || s.empty()) return;
            size_t fieldsWritten = 0;
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                if (c == ',') {
                    fieldsWritten++;
                    if (fieldsWritten >= nFields) return;
                }
                os.put(c);
            }
        };
        for (auto& kv : lobGrouped) {
            const std::string& outName = kv.first;
            auto& g = kv.second;
            if (g.rows.empty()) continue;
            if (g.fieldCountMax == 0) continue;
            if (g.fieldCountMax % 4 != 0) {
                std::cout << "[DataFactory][Merge][LOB][Warn] invalid fieldCountMax=" << g.fieldCountMax
                          << " (not multiple of 4), skip file=" << outName << std::endl;
                continue;
            }
            const size_t depth = g.fieldCountMax / 4;
            fs::path outPath = merged / outName;
            std::ofstream ofs(outPath, std::ios::out | std::ios::trunc);
            if (!ofs.is_open()) continue;
            ofs << "ts";
            for (size_t i = 1; i <= depth; ++i) {
                ofs << ",bid_price_" << i << ",bid_vol_" << i << ",ask_price_" << i << ",ask_vol_" << i;
            }
            ofs << "\n";
            for (auto& it : g.rows) {
                long long ts = it.first;
                const std::string& tail = it.second;
                ofs << ts;
                if (!tail.empty()) {
                    ofs << ",";
                    size_t tailFields = static_cast<size_t>(std::count(tail.begin(), tail.end(), ',')) + 1;
                    if (tailFields <= g.fieldCountMax) {
                        ofs << tail;
                        for (size_t i = tailFields; i < g.fieldCountMax; ++i) ofs << ",0";
                    } else {
                        // Too many columns: clamp to header width.
                        write_first_n_fields(ofs, tail, g.fieldCountMax);
                    }
                } else {
                    for (size_t i = 0; i < g.fieldCountMax; ++i) ofs << ",0";
                }
                ofs << "\n";
                outRowsLob += 1;
            }
            outFilesLob += 1;
        }

        std::cout << "[DataFactory][Merge] date=" << yyyymmdd
                  << " scannedFiles=" << scannedFiles
                  << " usedFiles=" << usedFiles
                  << " outFilesOhlcv=" << outFilesOhlcv
                  << " outRowsOhlcv=" << outRowsOhlcv
                  << " outFilesLob=" << outFilesLob
                  << " outRowsLob=" << outRowsLob
                  << " mergedDir=" << merged.string()
                  << std::endl;

        // After merging: clean up all DataFactoryAgent name directories, only keep _merged (to prevent data from growing indefinitely)
        // Note: this function is called by rank0 after the epoch end barrier, theoretically it will not conflict with writing.
        size_t deletedDirs = 0;
        size_t deletedEntries = 0;
        for (const auto& sub : fs::directory_iterator(root)) {
            if (!sub.is_directory()) continue;
            const std::string dirName = sub.path().filename().string();
            if (dirName == "_merged") continue;
            std::error_code ec_rm;
            // remove_all 返回删除的文件/目录数量（best-effort）
            uintmax_t n = fs::remove_all(sub.path(), ec_rm);
            if (ec_rm) {
                std::cout << "[DataFactory][Merge][Cleanup][Warn] failed to delete dir="
                          << sub.path().string() << " err=" << ec_rm.message() << std::endl;
                continue;
            }
            deletedDirs += 1;
            deletedEntries += static_cast<size_t>(n);
        }
        if (deletedDirs > 0 || deletedEntries > 0) {
            std::cout << "[DataFactory][Merge][Cleanup] deletedDirs=" << deletedDirs
                      << " deletedEntries=" << deletedEntries
                      << " root=" << root.string()
                      << std::endl;
        }
    }

    // ===== Cross-epoch topology coordination (basket -> METIS -> assignment) =====
    static inline std::filesystem::path crossTopologyDir() {
        return std::filesystem::path("data") / "agent_outputs" / "Topology";
    }

    // ===== Experiment archiver (epoch snapshots) =====
    // Copy all communication_logs folders under distributed log root into:
    //   data/agent_outputs/ExperimentArchives/<epochDate>/<rank>/communication_logs/...
    static inline void archiveCommunicationLogsForEpoch(
        const std::filesystem::path& logRoot,
        const std::string& epochDate
    ) {
        namespace fs = std::filesystem;
        if (epochDate.empty()) return;
        if (!fs::exists(logRoot)) return;

        fs::path destBase = fs::path("data") / "agent_outputs" / "ExperimentArchives" / epochDate;
        std::error_code ec;
        fs::create_directories(destBase, ec);

        size_t copiedDirs = 0;
        size_t skippedDirs = 0;
        size_t failedDirs = 0;

        for (auto it = fs::recursive_directory_iterator(logRoot, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (!it->is_directory()) continue;
            const fs::path p = it->path();
            if (p.filename() != "communication_logs") continue;

            const fs::path rankDir = p.parent_path();
            const std::string rankName = rankDir.filename().string(); // expected to be numeric rank id
            if (rankName.empty()) continue;

            fs::path dest = destBase / rankName / "communication_logs";
            if (fs::exists(dest)) {
                // Avoid mixing different runs with the same epochDate; keep first snapshot.
                skippedDirs++;
                continue;
            }
            fs::create_directories(dest, ec);
            if (ec) { failedDirs++; ec.clear(); continue; }

            // Copy directory contents (not the directory itself).
            bool ok = true;
            for (auto it2 = fs::directory_iterator(p, fs::directory_options::skip_permission_denied, ec);
                 it2 != fs::directory_iterator(); it2.increment(ec)) {
                if (ec) { ok = false; ec.clear(); continue; }
                const fs::path srcChild = it2->path();
                const fs::path dstChild = dest / srcChild.filename();
                std::error_code ec2;
                fs::copy(srcChild, dstChild,
                         fs::copy_options::recursive | fs::copy_options::overwrite_existing,
                         ec2);
                if (ec2) ok = false;
            }
            if (ok) copiedDirs++; else failedDirs++;
        }

        std::cout << "[Archive][CommunicationLogs] date=" << epochDate
                  << " out=" << destBase.string()
                  << " copied=" << copiedDirs
                  << " skipped_existing=" << skippedDirs
                  << " failed=" << failedDirs
                  << std::endl;
    }
    static inline std::filesystem::path assignmentFilePath(const std::string& yyyymmdd) {
        return crossTopologyDir() / (std::string("assignment_") + yyyymmdd + ".json");
    }
    static inline std::filesystem::path basketDirForDate(const std::string& yyyymmdd) {
        return std::filesystem::path("data") / "agent_outputs" / "BasketFinal" / yyyymmdd;
    }
    static inline std::filesystem::path basketRankFile(const std::string& yyyymmdd, int rank) {
        return basketDirForDate(yyyymmdd) / (std::string("rank") + std::to_string(rank) + ".jsonl");
    }

    static inline bool parseMultiKernelTargets(const pugi::xml_node& rootNode,
                                               std::unordered_map<std::string,int>& outAsset2Kernel,
                                               std::vector<std::string>& outAssets) {
        outAsset2Kernel.clear();
        outAssets.clear();
        auto mk = rootNode.child("MultiKernel");
        if (!mk) return false;
        auto attr = mk.attribute("targets");
        if (attr.empty()) return false;
        std::string s = attr.as_string();
        size_t start = 0;
        while (start < s.size()) {
            size_t sep = s.find(';', start);
            std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
            size_t colon = item.find(':');
            if (colon != std::string::npos) {
                try {
                    int kr = std::stoi(item.substr(0, colon));
                    std::string asset = item.substr(colon+1);
                    if (!asset.empty()) {
                        outAsset2Kernel[asset] = kr;
                    }
                } catch (...) {
                }
            }
            if (sep==std::string::npos) break;
            start = sep + 1;
        }
        outAssets.reserve(outAsset2Kernel.size());
        for (const auto& kv : outAsset2Kernel) outAssets.push_back(kv.first);
        std::sort(outAssets.begin(), outAssets.end());
        return !outAssets.empty();
    }

    static inline std::vector<int> parseCsvInts(const std::string& s) {
        std::vector<int> out;
        if (s.empty()) return out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            try {
                out.push_back(std::stoi(tok));
            } catch (...) {
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    static inline std::string formatIntList(const std::vector<int>& values) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < values.size(); ++i) {
            oss << values[i];
            if (i + 1 < values.size()) oss << ",";
        }
        oss << "}";
        return oss.str();
    }

    struct MultiKernelCommunicationTopology {
        std::vector<int> kernelRanks;
        std::unordered_map<int, std::vector<int>> agentRanksByKernel;
        std::unordered_map<int, std::vector<int>> crossRanksByKernel;
        std::unordered_map<int, std::vector<int>> crossAgentSenders;
        std::unordered_map<int, size_t> remoteWindowBytesByKernel;
        std::unordered_map<int, int> kernelComponentByKernel;
    };

    struct ComponentCommunicationView {
        int componentId{-1};
        std::vector<int> kernelRanks;
        std::vector<int> agentRanks;
        std::vector<int> crossRanks;
        std::vector<int> members;
        std::unordered_map<int, std::vector<int>> agentRanksByKernel;
        std::unordered_map<int, std::vector<int>> crossRanksByKernel;
        std::unordered_map<int, std::vector<int>> crossAgentSenders;
        std::unordered_map<int, size_t> remoteWindowBytesByKernel;
    };

    static inline MultiKernelCommunicationTopology buildMultiKernelCommunicationTopology(
            const pugi::xml_node& rootNode,
            const std::vector<int>& discoveredKernelRanks) {
        MultiKernelCommunicationTopology topo;
        topo.kernelRanks = discoveredKernelRanks;
        std::sort(topo.kernelRanks.begin(), topo.kernelRanks.end());
        topo.kernelRanks.erase(std::unique(topo.kernelRanks.begin(), topo.kernelRanks.end()), topo.kernelRanks.end());

        if (auto mk = rootNode.child("MultiKernel")) {
            for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                int kr = kn.attribute("rank").as_int(-1);
                if (kr < 0) continue;
                topo.kernelRanks.push_back(kr);
                auto agents = parseCsvInts(kn.attribute("agentRanks").as_string());
                if (!agents.empty()) topo.agentRanksByKernel[kr] = std::move(agents);
                auto cross = parseCsvInts(kn.attribute("crossAgentRanks").as_string());
                if (!cross.empty()) {
                    topo.crossRanksByKernel[kr] = cross;
                    for (int cr : cross) {
                        topo.crossAgentSenders[cr].push_back(kr);
                    }
                }
                size_t rmb = kn.attribute("kernelWindowMB").as_uint(0);
                if (rmb > 0) {
                    topo.remoteWindowBytesByKernel[kr] = rmb * 1024ull * 1024ull;
                }
            }
        }

        if (!rootNode.child("MultiKernel")) {
            int simRank = 0;
            if (auto mpiCfg = rootNode.child("MPIConfiguration")) {
                simRank = mpiCfg.child("SimulationRank").text().as_int(0);
                auto agents = parseCsvInts(mpiCfg.child("AgentRanks").text().as_string());
                if (!agents.empty()) {
                    topo.agentRanksByKernel[simRank] = std::move(agents);
                }
                auto cross = parseCsvInts(mpiCfg.child("CrossAgentRanks").text().as_string());
                if (!cross.empty()) {
                    topo.crossRanksByKernel[simRank] = cross;
                    for (int cr : cross) {
                        topo.crossAgentSenders[cr].push_back(simRank);
                    }
                }
            }
            topo.kernelRanks.push_back(simRank);
        }

        std::sort(topo.kernelRanks.begin(), topo.kernelRanks.end());
        topo.kernelRanks.erase(std::unique(topo.kernelRanks.begin(), topo.kernelRanks.end()), topo.kernelRanks.end());
        for (auto& kv : topo.crossAgentSenders) {
            auto& senders = kv.second;
            std::sort(senders.begin(), senders.end());
            senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
        }

        if (topo.kernelRanks.empty()) {
            return topo;
        }

        std::unordered_map<int, int> kernelIndex;
        kernelIndex.reserve(topo.kernelRanks.size());
        for (size_t i = 0; i < topo.kernelRanks.size(); ++i) {
            kernelIndex[topo.kernelRanks[i]] = static_cast<int>(i);
        }
        std::vector<int> parent(topo.kernelRanks.size());
        std::iota(parent.begin(), parent.end(), 0);
        std::function<int(int)> findRoot = [&](int x) -> int {
            if (parent[x] == x) return x;
            parent[x] = findRoot(parent[x]);
            return parent[x];
        };
        auto unite = [&](int a, int b) {
            int ra = findRoot(a);
            int rb = findRoot(b);
            if (ra != rb) parent[rb] = ra;
        };
        for (const auto& kv : topo.crossAgentSenders) {
            const auto& kernels = kv.second;
            for (size_t i = 1; i < kernels.size(); ++i) {
                auto ita = kernelIndex.find(kernels[0]);
                auto itb = kernelIndex.find(kernels[i]);
                if (ita != kernelIndex.end() && itb != kernelIndex.end()) {
                    unite(ita->second, itb->second);
                }
            }
        }

        std::unordered_map<int, int> rootToDense;
        int nextDense = 0;
        for (int kr : topo.kernelRanks) {
            int idx = kernelIndex[kr];
            int root = findRoot(idx);
            auto [it, inserted] = rootToDense.emplace(root, nextDense);
            if (inserted) nextDense++;
            topo.kernelComponentByKernel[kr] = it->second;
        }
        return topo;
    }

    static inline ComponentCommunicationView buildComponentCommunicationView(
            const MultiKernelCommunicationTopology& topo,
            int rank,
            int simRankGlobal,
            bool isKernel,
            bool isCross,
            bool baselineFullMesh) {
        ComponentCommunicationView view;
        if (topo.kernelRanks.empty()) return view;

        if (baselineFullMesh) {
            view.componentId = 0;
            view.kernelRanks = topo.kernelRanks;
        } else if (isCross) {
            auto it = topo.crossAgentSenders.find(rank);
            if (it != topo.crossAgentSenders.end() && !it->second.empty()) {
                auto compIt = topo.kernelComponentByKernel.find(it->second.front());
                if (compIt != topo.kernelComponentByKernel.end()) {
                    view.componentId = compIt->second;
                }
            }
            if (view.componentId < 0) {
                auto fallback = topo.kernelComponentByKernel.find(simRankGlobal);
                if (fallback != topo.kernelComponentByKernel.end()) {
                    view.componentId = fallback->second;
                    std::cout << "[TOPO][COMPONENT][WARN] rank=" << rank
                              << " role=cross fallback_to_sim_kernel=" << simRankGlobal
                              << " reason=no_cross_sender_mapping"
                              << std::endl;
                }
            }
        } else {
            auto compIt = topo.kernelComponentByKernel.find(simRankGlobal);
            if (compIt != topo.kernelComponentByKernel.end()) {
                view.componentId = compIt->second;
            }
        }

        if (view.componentId < 0) {
            view.componentId = 0;
            if (!baselineFullMesh) {
                std::cout << "[TOPO][COMPONENT][WARN] rank=" << rank
                          << " fallback_component=0 reason=missing_component_mapping"
                          << std::endl;
            }
        }

        for (int kr : topo.kernelRanks) {
            auto compIt = topo.kernelComponentByKernel.find(kr);
            if (!baselineFullMesh && (compIt == topo.kernelComponentByKernel.end() || compIt->second != view.componentId)) {
                continue;
            }
            view.kernelRanks.push_back(kr);
            auto itAgents = topo.agentRanksByKernel.find(kr);
            if (itAgents != topo.agentRanksByKernel.end()) {
                view.agentRanksByKernel[kr] = itAgents->second;
                view.agentRanks.insert(view.agentRanks.end(), itAgents->second.begin(), itAgents->second.end());
            }
            auto itCross = topo.crossRanksByKernel.find(kr);
            if (itCross != topo.crossRanksByKernel.end()) {
                view.crossRanksByKernel[kr] = itCross->second;
                view.crossRanks.insert(view.crossRanks.end(), itCross->second.begin(), itCross->second.end());
            }
            auto itWin = topo.remoteWindowBytesByKernel.find(kr);
            if (itWin != topo.remoteWindowBytesByKernel.end()) {
                view.remoteWindowBytesByKernel[kr] = itWin->second;
            }
        }

        std::sort(view.kernelRanks.begin(), view.kernelRanks.end());
        view.kernelRanks.erase(std::unique(view.kernelRanks.begin(), view.kernelRanks.end()), view.kernelRanks.end());
        std::sort(view.agentRanks.begin(), view.agentRanks.end());
        view.agentRanks.erase(std::unique(view.agentRanks.begin(), view.agentRanks.end()), view.agentRanks.end());
        std::sort(view.crossRanks.begin(), view.crossRanks.end());
        view.crossRanks.erase(std::unique(view.crossRanks.begin(), view.crossRanks.end()), view.crossRanks.end());

        for (int cr : view.crossRanks) {
            auto it = topo.crossAgentSenders.find(cr);
            if (it == topo.crossAgentSenders.end()) continue;
            std::vector<int> senders;
            for (int kr : it->second) {
                auto compIt = topo.kernelComponentByKernel.find(kr);
                if (baselineFullMesh || (compIt != topo.kernelComponentByKernel.end() && compIt->second == view.componentId)) {
                    senders.push_back(kr);
                }
            }
            std::sort(senders.begin(), senders.end());
            senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
            if (!senders.empty()) {
                view.crossAgentSenders[cr] = std::move(senders);
            }
        }

        view.members = view.kernelRanks;
        view.members.insert(view.members.end(), view.agentRanks.begin(), view.agentRanks.end());
        view.members.insert(view.members.end(), view.crossRanks.begin(), view.crossRanks.end());
        if (isKernel || !isCross) {
            view.members.push_back(rank);
        } else if (std::find(view.crossRanks.begin(), view.crossRanks.end(), rank) == view.crossRanks.end()) {
            view.members.push_back(rank);
            view.crossRanks.push_back(rank);
            std::sort(view.crossRanks.begin(), view.crossRanks.end());
            view.crossRanks.erase(std::unique(view.crossRanks.begin(), view.crossRanks.end()), view.crossRanks.end());
        }
        std::sort(view.members.begin(), view.members.end());
        view.members.erase(std::unique(view.members.begin(), view.members.end()), view.members.end());
        return view;
    }

    static inline MPI_Comm createGroupCommunicatorForMembers(const std::vector<int>& members, int rank) {
        if (members.empty()) return MPI_COMM_NULL;
        if (std::find(members.begin(), members.end(), rank) == members.end()) return MPI_COMM_NULL;
        MPI_Group worldGroup = MPI_GROUP_NULL;
        MPI_Group subGroup = MPI_GROUP_NULL;
        MPI_Comm newComm = MPI_COMM_NULL;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
        MPI_Group_incl(worldGroup, static_cast<int>(members.size()), members.data(), &subGroup);
        std::cout << "[COMPONENT_COMM][CREATE_ENTER] rank=" << rank
                  << " members=" << formatIntList(members)
                  << " tag=" << COMPONENT_COMM_CREATE_TAG << std::endl;
        MPI_Comm_create_group(MPI_COMM_WORLD, subGroup, COMPONENT_COMM_CREATE_TAG, &newComm);
        std::cout << "[COMPONENT_COMM][CREATE_EXIT] rank=" << rank
                  << " commNull=" << (newComm == MPI_COMM_NULL ? "true" : "false")
                  << " tag=" << COMPONENT_COMM_CREATE_TAG << std::endl;
        if (subGroup != MPI_GROUP_NULL) MPI_Group_free(&subGroup);
        if (worldGroup != MPI_GROUP_NULL) MPI_Group_free(&worldGroup);
        return newComm;
    }

    static inline nlohmann::json buildInitialAssignmentFromConfig(const pugi::xml_node& rootNode,
                                                                  const std::string& yyyymmdd) {
        nlohmann::json j;
        j["date"] = yyyymmdd;
        j["agents"] = nlohmann::json::object();

        // Cross ranks only: follow the existing XML range semantics.
        for (auto cr : rootNode.children("CrossAgentRank")) {
            int rank = cr.attribute("rank").as_int(-1);
            if (rank < 0) continue;
            for (auto n : cr.children("CrossBehavioralSPTAgents")) {
                int count = n.attribute("count").as_int(0);
                int startIndex = n.attribute("startIndex").as_int(1);
                for (int i = 0; i < count; ++i) {
                    int id = startIndex + i;
                    std::string name = std::string("CppCrossBehavioralSPTAgent_") + std::to_string(id);
                    j["agents"][name] = rank;
                }
            }
            for (auto n : cr.children("CrossRLAgents")) {
                int count = n.attribute("count").as_int(0);
                int startIndex = n.attribute("startIndex").as_int(1);
                for (int i = 0; i < count; ++i) {
                    int id = startIndex + i;
                    std::string name = std::string("CppCrossRLAgent_") + std::to_string(id);
                    j["agents"][name] = rank;
                }
            }
            for (auto n : cr.children("DataFactoryAgents")) {
                int count = n.attribute("count").as_int(0);
                int startIndex = n.attribute("startIndex").as_int(1);
                for (int i = 0; i < count; ++i) {
                    int id = startIndex + i;
                    std::string name = std::string("CppDataFactoryAgent_") + std::to_string(id);
                    j["agents"][name] = rank;
                }
            }
        }
        return j;
    }

    static inline bool writeJsonAtomic(const std::filesystem::path& path, const nlohmann::json& j) {
        try {
            std::filesystem::create_directories(path.parent_path());
            std::filesystem::path tmp = path;
            tmp += ".tmp";
            {
                std::ofstream ofs(tmp, std::ios::out | std::ios::trunc);
                if (!ofs.is_open()) return false;
                ofs << j.dump(2) << "\n";
            }
            std::error_code ec;
            std::filesystem::rename(tmp, path, ec);
            if (ec) {
                // fallback: overwrite
                std::ofstream ofs(path, std::ios::out | std::ios::trunc);
                if (!ofs.is_open()) return false;
                ofs << j.dump(2) << "\n";
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static inline nlohmann::json readJsonFile(const std::filesystem::path& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) return nlohmann::json();
        std::stringstream ss; ss << ifs.rdbuf();
        auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded()) return nlohmann::json();
        return j;
    }

    static inline nlohmann::json readAssignmentFileOrThrow(const std::string& yyyymmdd) {
        const auto p = assignmentFilePath(yyyymmdd);
        auto j = readJsonFile(p);
        if (!j.is_object() || !j.contains("agents") || !j["agents"].is_object()) {
            throw std::runtime_error("[Topology][Assignment][FATAL] invalid or missing assignment file for date=" +
                                     yyyymmdd + " path=" + p.string());
        }
        return j;
    }

    static inline std::vector<std::string> agentsAssignedToRank(const nlohmann::json& assignment,
                                                                const std::string& agentTypePrefix,
                                                                int myRank) {
        std::vector<std::string> out;
        if (!assignment.is_object()) return out;
        if (!assignment.contains("agents") || !assignment["agents"].is_object()) return out;
        for (auto it = assignment["agents"].begin(); it != assignment["agents"].end(); ++it) {
            if (!it.value().is_number_integer()) continue;
            if (it.value().get<int>() != myRank) continue;
            const std::string name = it.key();
            if (name.rfind(agentTypePrefix, 0) == 0) out.push_back(name);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    static inline bool readOhlcvLastNBars(const std::filesystem::path& path,
                                         size_t maxBars,
                                         std::vector<OhlcvRow>& out) {
        out.clear();
        std::ifstream ifs(path);
        if (!ifs.is_open()) return false;
        std::string line;
        bool first = true;
        std::deque<OhlcvRow> ring;
        ring.clear();
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            if (first) {
                first = false;
                if (line.find("start_ts") != std::string::npos) continue;
            }
            std::istringstream ss(line);
            std::string t0,t1,t2,t3,t4,t5;
            if (!std::getline(ss, t0, ',')) continue;
            if (!std::getline(ss, t1, ',')) continue;
            if (!std::getline(ss, t2, ',')) continue;
            if (!std::getline(ss, t3, ',')) continue;
            if (!std::getline(ss, t4, ',')) continue;
            if (!std::getline(ss, t5, ',')) continue;
            try {
                OhlcvRow r;
                r.start_ts = std::stoll(t0);
                r.open = std::stod(t1);
                r.high = std::stod(t2);
                r.low = std::stod(t3);
                r.close = std::stod(t4);
                r.volume = static_cast<unsigned long long>(std::stoull(t5));
                ring.push_back(r);
                if (maxBars > 0 && ring.size() > maxBars) ring.pop_front();
            } catch (...) {
            }
        }
        if (ring.empty()) return false;
        out.assign(ring.begin(), ring.end());
        return true;
    }

    static inline bool computeCloseReturnStats(const std::vector<OhlcvRow>& bars,
                                               size_t windowW,
                                               size_t horizonH,
                                               double& outMu,
                                               double& outSigma) {
        outMu = 0.0;
        outSigma = 0.0;
        if (bars.empty()) return false;
        horizonH = std::max<size_t>(1, horizonH);
        windowW = std::max<size_t>(horizonH + 3, windowW);
        if (bars.size() < windowW) return false;
        const size_t start = bars.size() - windowW;
        std::vector<double> r;
        r.reserve(windowW);
        for (size_t i = start + horizonH; i < bars.size(); ++i) {
            double c0 = bars[i - horizonH].close;
            double c1 = bars[i].close;
            if (!(c0 > 0.0) || !(c1 > 0.0)) continue;
            double lr = std::log(c1 / c0);
            if (!std::isfinite(lr)) continue;
            r.push_back(lr);
        }
        if (r.size() < 3) return false;
        double mean = 0.0;
        for (double x : r) mean += x;
        mean /= static_cast<double>(r.size());
        double var = 0.0;
        for (double x : r) {
            double d = x - mean;
            var += d * d;
        }
        var /= static_cast<double>(r.size() - 1);
        outMu = mean;
        outSigma = std::sqrt(std::max(0.0, var));
        return std::isfinite(outMu) && std::isfinite(outSigma);
    }

    static inline double normal_pdf_local(double x, double mu, double sigma) {
        const double z = (x - mu) / sigma;
        const double inv = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
        return inv * std::exp(-0.5 * z * z);
    }

    static inline double spt_value(double x, double alpha_gain, double beta_loss, double lambda_loss) {
        if (x >= 0.0) return std::pow(x, alpha_gain);
        return -lambda_loss * std::pow(-x, beta_loss);
    }

    static inline double spt_weight(double p, double gamma_weight) {
        p = std::max(0.0, std::min(1.0, p));
        const double g = std::max(1e-6, gamma_weight);
        const double a = std::pow(p, g);
        const double b = std::pow(1.0 - p, g);
        const double denom = std::pow(a + b, 1.0 / g);
        if (!(denom > 0.0)) return p;
        return a / denom;
    }

    static inline double computeSPTV(double mu, double sigma,
                                     int grid_points,
                                     double n_sigma,
                                     double sigma_floor,
                                     double alpha_gain,
                                     double beta_loss,
                                     double lambda_loss,
                                     double gamma_weight) {
        sigma = std::max(sigma, sigma_floor);
        const int K = std::max(11, grid_points);
        const double lo = mu - n_sigma * sigma;
        const double hi = mu + n_sigma * sigma;
        const double step = (hi - lo) / static_cast<double>(K - 1);
        std::vector<double> x(K), p(K);
        double sum_pdf = 0.0;
        for (int i = 0; i < K; ++i) {
            double xi = lo + static_cast<double>(i) * step;
            x[i] = xi;
            double pdf = normal_pdf_local(xi, mu, sigma);
            if (!std::isfinite(pdf) || pdf < 0.0) pdf = 0.0;
            p[i] = pdf;
            sum_pdf += pdf;
        }
        if (!(sum_pdf > 0.0)) {
            return spt_value(mu, alpha_gain, beta_loss, lambda_loss);
        }
        for (int i = 0; i < K; ++i) p[i] /= sum_pdf;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < K; ++i) {
            double wi = spt_weight(p[i], gamma_weight);
            num += wi * spt_value(x[i], alpha_gain, beta_loss, lambda_loss);
            den += wi;
        }
        if (!(den > 0.0)) return 0.0;
        return num / den;
    }

    static inline void evaluateBasketsForCrossRankAndWriteJsonl(
        int myRank,
        const std::string& epochDate,
        const std::vector<std::string>& allAssets,
        const nlohmann::json& assignment,
        size_t basketCapN,
        size_t assetEvalN,
        uint64_t daySeed,
        uint64_t epochIdx
    ) {
        // Only handle SPT agents for now.
        auto myAgents = agentsAssignedToRank(assignment, "CppCrossBehavioralSPTAgent_", myRank);
        if (myAgents.empty()) return;

        std::filesystem::create_directories(basketDirForDate(epochDate));
        std::ofstream ofs(basketRankFile(epochDate, myRank), std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "[Basket][Eval][Error] rank=" << myRank
                      << " cannot open output " << basketRankFile(epochDate, myRank).string() << std::endl;
            return;
        }

        const std::filesystem::path mergedDir = dataFactoryMergedDir();

        const size_t totalAssets = allAssets.size();
        size_t effectiveEvalN = assetEvalN;
        if (effectiveEvalN == 0 || effectiveEvalN >= totalAssets) effectiveEvalN = totalAssets;

        auto buildEvalAssetsForAgent = [&](const std::string& agentName, uint64_t& outEvalSeed) -> std::vector<std::string> {
            outEvalSeed = 0;
            if (totalAssets == 0 || effectiveEvalN == 0) return {};

            // Make the seed explicitly "per-agent": derived from the daySeed + agentName.
            const uint64_t agentSeed = splitmix64(daySeed ^ (fnv1a64(agentName) * 0x9E3779B97F4A7C15ull));

            // Then derive the evaluation seed from agentSeed + epochIdx (+ effectiveEvalN to make budget changes explicit).
            outEvalSeed = splitmix64(agentSeed ^
                                     (epochIdx * 0xD1B54A32D192ED03ull) ^
                                     (static_cast<uint64_t>(effectiveEvalN) * 0x94d049bb133111ebull));

            if (effectiveEvalN >= totalAssets) {
                return allAssets; // deterministic order
            }

            std::vector<size_t> idx(totalAssets);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937_64 rng(outEvalSeed);
            std::shuffle(idx.begin(), idx.end(), rng);

            std::vector<std::string> evalAssets;
            evalAssets.reserve(effectiveEvalN);
            for (size_t i = 0; i < effectiveEvalN; ++i) evalAssets.push_back(allAssets[idx[i]]);
            // Keep stable readable order in logs.
            std::sort(evalAssets.begin(), evalAssets.end());
            evalAssets.erase(std::unique(evalAssets.begin(), evalAssets.end()), evalAssets.end());
            return evalAssets;
        };

        for (const auto& agentName : myAgents) {
            uint64_t evalSeed = 0;
            std::vector<std::string> evalAssets = buildEvalAssetsForAgent(agentName, evalSeed);

            // Load checkpoint
            const std::filesystem::path ckpt = std::filesystem::path("data") / "agent_outputs" / "SPTAgents" / (agentName + ".json");
            auto j = readJsonFile(ckpt);
            if (!j.is_object()) {
                // Still emit a row with empty basket so rank0 can detect missing state.
                nlohmann::json out;
                out["date"] = epochDate;
                out["agent"] = agentName;
                out["basket_cap"] = basketCapN;
                out["asset_total"] = totalAssets;
                out["asset_eval_n_cfg"] = assetEvalN;
                out["asset_eval_n_effective"] = effectiveEvalN;
                out["eval_mode"] = (effectiveEvalN >= totalAssets) ? "all" : "sample";
                out["eval_seed"] = evalSeed;
                out["eval_assets"] = evalAssets;
                out["final_basket"] = nlohmann::json::array();
                out["locked_assets"] = nlohmann::json::array();
                out["error"] = "missing_checkpoint";
                ofs << out.dump() << "\n";
                continue;
            }

            std::map<std::string,int> holdings;
            if (j.contains("holdings") && j["holdings"].is_object()) {
                for (auto it = j["holdings"].begin(); it != j["holdings"].end(); ++it) {
                    if (it.value().is_number_integer()) holdings[it.key()] = it.value().get<int>();
                }
            }

            // Params needed for scoring
            size_t W = 60;
            size_t H = 10;
            double alpha_gain = 0.88, beta_loss = 0.88, lambda_loss = 2.25, gamma_weight = 0.61;
            int grid_points = 101;
            double n_sigma = 3.0;
            double sigma_floor = 1e-6;
            if (j.contains("params") && j["params"].is_object()) {
                auto& p = j["params"];
                if (p.contains("ohlcvHistoryWindowBars") && p["ohlcvHistoryWindowBars"].is_number_integer()) W = static_cast<size_t>(std::max(3, p["ohlcvHistoryWindowBars"].get<int>()));
                if (p.contains("returnHorizonBars") && p["returnHorizonBars"].is_number_integer()) H = static_cast<size_t>(std::max(1, p["returnHorizonBars"].get<int>()));
                if (p.contains("spt") && p["spt"].is_object()) {
                    auto& s = p["spt"];
                    if (s.contains("alphaGain") && s["alphaGain"].is_number()) alpha_gain = s["alphaGain"].get<double>();
                    if (s.contains("betaLoss") && s["betaLoss"].is_number()) beta_loss = s["betaLoss"].get<double>();
                    if (s.contains("lambdaLoss") && s["lambdaLoss"].is_number()) lambda_loss = s["lambdaLoss"].get<double>();
                    if (s.contains("gammaWeight") && s["gammaWeight"].is_number()) gamma_weight = s["gammaWeight"].get<double>();
                }
                if (p.contains("discretization") && p["discretization"].is_object()) {
                    auto& d = p["discretization"];
                    if (d.contains("gridPoints") && d["gridPoints"].is_number_integer()) grid_points = std::max(11, d["gridPoints"].get<int>());
                    if (d.contains("nSigma") && d["nSigma"].is_number()) n_sigma = std::max(0.5, d["nSigma"].get<double>());
                    if (d.contains("sigmaFloor") && d["sigmaFloor"].is_number()) sigma_floor = std::max(1e-12, d["sigmaFloor"].get<double>());
                }
            }

            // Locked assets: position > 0, excluding cash
            std::vector<std::string> locked;
            for (const auto& kv : holdings) {
                if (kv.first == "cash") continue;
                if (kv.second > 0) locked.push_back(kv.first);
            }
            std::sort(locked.begin(), locked.end());
            locked.erase(std::unique(locked.begin(), locked.end()), locked.end());

            // Previous end-of-day basket: if NO holdings, keep it as base.
            std::vector<std::string> prevBasket;
            if (locked.empty() && j.contains("end_of_day_basket") && j["end_of_day_basket"].is_array()) {
                for (auto& v : j["end_of_day_basket"]) {
                    if (v.is_string()) prevBasket.push_back(v.get<std::string>());
                }
                std::sort(prevBasket.begin(), prevBasket.end());
                prevBasket.erase(std::unique(prevBasket.begin(), prevBasket.end()), prevBasket.end());
            }

            // Score sampled assets by SPTV (historical only) using merged OHLCV for epochDate.

            struct Scored { std::string asset; double sptv; };
            std::vector<Scored> scored;
            scored.reserve(evalAssets.size());
            for (const auto& asset : evalAssets) {
                std::filesystem::path f = mergedDir / (std::string("1m_") + asset + "_" + epochDate + ".csv");
                std::vector<OhlcvRow> bars;
                if (!std::filesystem::exists(f)) continue;
                if (!readOhlcvLastNBars(f, std::max<size_t>(W + H + 10, W + 50), bars)) continue;
                double mu = 0.0, sig = 0.0;
                if (!computeCloseReturnStats(bars, W, H, mu, sig)) continue;
                // Convert log-return distribution to simple-return x distribution like agent does (k=1 here).
                const double s2 = sig * sig;
                const double exp_mean = std::exp(mu + 0.5 * s2);
                const double exp_var = (std::exp(s2) - 1.0) * std::exp(2.0 * mu + s2);
                const double mu_x = exp_mean - 1.0;
                const double sigma_x = std::sqrt(std::max(0.0, exp_var));
                double sptv = computeSPTV(mu_x, sigma_x, grid_points, n_sigma, sigma_floor, alpha_gain, beta_loss, lambda_loss, gamma_weight);
                if (!std::isfinite(sptv)) continue;
                scored.push_back({asset, sptv});
            }
            std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b){
                if (a.sptv != b.sptv) return a.sptv > b.sptv;
                return a.asset < b.asset;
            });

            // Build final basket with cap N: must include locked, then fill with top positive SPTV.
            std::vector<std::string> basket = locked.empty() ? prevBasket : locked;
            if (basket.size() < basketCapN) {
                for (const auto& s : scored) {
                    if (s.sptv <= 0.0) break;
                    if (std::find(basket.begin(), basket.end(), s.asset) != basket.end()) continue;
                    basket.push_back(s.asset);
                    if (basket.size() >= basketCapN) break;
                }
            }

            // Persist basket into checkpoint (so next epoch agent can load)
            j["basket_assets"] = basket;
            j["basket_cap"] = basketCapN;
            j["end_of_day_basket"] = basket;
            {
                std::ofstream ck(ckpt, std::ios::out | std::ios::trunc);
                if (ck.is_open()) ck << j.dump(2) << "\n";
            }

            nlohmann::json out;
            out["date"] = epochDate;
            out["agent"] = agentName;
            out["basket_cap"] = basketCapN;
            out["asset_total"] = totalAssets;
            out["asset_eval_n_cfg"] = assetEvalN;
            out["asset_eval_n_effective"] = effectiveEvalN;
            out["eval_mode"] = (effectiveEvalN >= totalAssets) ? "all" : "sample";
            out["eval_seed"] = evalSeed;
            out["eval_assets"] = evalAssets;
            out["final_basket"] = basket;
            out["locked_assets"] = locked;
            ofs << out.dump() << "\n";
        }
    }

    struct GroupInfo {
        std::vector<int> kernels;              // sorted unique kernel ids
        size_t weight{0};                      // number of agents in this group
        std::vector<std::string> agents;       // member agent names
    };

    static inline std::string kernelSigKey(const std::vector<int>& ks) {
        std::ostringstream oss;
        for (size_t i = 0; i < ks.size(); ++i) {
            if (i) oss << "|";
            oss << ks[i];
        }
        return oss.str();
    }

    static inline bool loadAllBasketsForDate(
        const std::string& epochDate,
        const std::unordered_map<std::string,int>& asset2kernel,
        std::unordered_map<std::string, std::vector<std::string>>& outAgentBasket,
        std::vector<GroupInfo>& outGroups
    ) {
        outAgentBasket.clear();
        outGroups.clear();
        namespace fs = std::filesystem;
        fs::path dir = basketDirForDate(epochDate);
        if (!fs::exists(dir)) return false;

        std::unordered_map<std::string, GroupInfo> groupsByKey;

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension().string() != ".jsonl") continue;
            std::ifstream ifs(entry.path());
            if (!ifs.is_open()) continue;
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;
                auto j = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
                if (j.is_discarded() || !j.is_object()) continue;
                if (!j.contains("agent") || !j["agent"].is_string()) continue;
                if (!j.contains("final_basket") || !j["final_basket"].is_array()) continue;
                std::string agent = j["agent"].get<std::string>();
                std::vector<std::string> basket;
                for (auto& v : j["final_basket"]) {
                    if (v.is_string()) basket.push_back(v.get<std::string>());
                }
                outAgentBasket[agent] = basket;

                std::vector<int> kernels;
                kernels.reserve(basket.size());
                for (const auto& a : basket) {
                    auto it = asset2kernel.find(a);
                    if (it != asset2kernel.end()) kernels.push_back(it->second);
                }
                std::sort(kernels.begin(), kernels.end());
                kernels.erase(std::unique(kernels.begin(), kernels.end()), kernels.end());
                std::string key = kernelSigKey(kernels);
                auto& g = groupsByKey[key];
                g.kernels = kernels;
                g.weight += 1;
                g.agents.push_back(agent);
            }
        }

        outGroups.reserve(groupsByKey.size());
        for (auto& kv : groupsByKey) {
            auto& g = kv.second;
            std::sort(g.agents.begin(), g.agents.end());
            outGroups.push_back(std::move(g));
        }
        // deterministic order
        std::sort(outGroups.begin(), outGroups.end(), [](const GroupInfo& a, const GroupInfo& b){
            if (a.kernels != b.kernels) return a.kernels < b.kernels;
            return a.weight > b.weight;
        });
        return !outGroups.empty();
    }

    static inline void ensureMtKaHyParInitialized() {
        static bool mtkahypar_initialized = false;
        if (mtkahypar_initialized) return;
        // Default to 8 threads, but cap to the number of CPUs available to this process (rank0 affinity).
        size_t threads = 8;
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            const int n = CPU_COUNT(&set);
            if (n > 0) threads = std::min<size_t>(threads, static_cast<size_t>(n));
        }
#endif
        if (threads == 0) threads = 1;
        std::cout << "[Topology][MtKaHyPar] initialize threads=" << threads << std::endl;
        mt_kahypar_initialize(/*num_threads=*/threads, /*interleaved_allocations=*/false);
        mtkahypar_initialized = true;
    }

    static inline double loadWakeupIntervalSecondsFromCheckpointOrThrow(const std::string& agentName) {
        namespace fs = std::filesystem;
        fs::path p = fs::path("data") / "agent_outputs" / "SPTAgents" / (agentName + ".json");
        auto j = readJsonFile(p);
        if (!j.is_object()) {
            throw std::runtime_error("[Topology][MtKaHyPar] failed to read checkpoint for agent=" + agentName + " path=" + p.string());
        }
        if (!j.contains("params") || !j["params"].is_object()) {
            throw std::runtime_error("[Topology][MtKaHyPar] checkpoint missing params for agent=" + agentName);
        }
        auto& params = j["params"];
        if (!params.contains("wakeupIntervalSeconds") || !params["wakeupIntervalSeconds"].is_number()) {
            throw std::runtime_error("[Topology][MtKaHyPar] checkpoint missing params.wakeupIntervalSeconds for agent=" + agentName);
        }
        const double v = params["wakeupIntervalSeconds"].get<double>();
        if (!(v > 0.0) || !std::isfinite(v)) {
            throw std::runtime_error("[Topology][MtKaHyPar] invalid wakeupIntervalSeconds for agent=" + agentName);
        }
        return v;
    }

    // Optional: capture partition quality metrics (km1/imbalance/weights) for offline analysis.
    // This avoids relying on stdout, which may be truncated per-epoch in HPC runs.
    static inline std::vector<int> mtKaHyParPartitionAgents(
        const std::unordered_map<std::string,int>& asset2kernel,
        const std::unordered_map<std::string, std::vector<std::string>>& agentBasket,
        int nparts,
        double epsilon,
        nlohmann::json* outMetrics /*nullable*/ = nullptr
    ) {
        if (nparts <= 0 || agentBasket.empty()) return {};
        if (nparts == 1) return std::vector<int>(agentBasket.size(), 0);

        // Hypergraph model (agent-level):
        // - Hypernodes: agents
        // - Hyperedges: kernels
        // Objective: KM1 / connectivity (minimize #blocks per kernel)
        // Balance: node weights ~ per-day compute+comm cost proxy derived from wakeup interval + basket size.

        // Deterministic agent order -> hypernode id.
        std::vector<std::string> agents;
        agents.reserve(agentBasket.size());
        for (const auto& kv : agentBasket) agents.push_back(kv.first);
        std::sort(agents.begin(), agents.end());

        std::vector<std::vector<int>> agentKernels(agents.size());
        std::vector<int> allKernels;
        allKernels.reserve(agents.size() * 5);
        for (size_t i = 0; i < agents.size(); ++i) {
            const auto itB = agentBasket.find(agents[i]);
            if (itB == agentBasket.end()) continue;
            std::vector<int> ks;
            ks.reserve(itB->second.size());
            for (const auto& a : itB->second) {
                auto it = asset2kernel.find(a);
                if (it != asset2kernel.end()) ks.push_back(it->second);
            }
            std::sort(ks.begin(), ks.end());
            ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
            agentKernels[i] = std::move(ks);
            for (int k : agentKernels[i]) allKernels.push_back(k);
        }
        std::sort(allKernels.begin(), allKernels.end());
        allKernels.erase(std::unique(allKernels.begin(), allKernels.end()), allKernels.end());

        const mt_kahypar_hypernode_id_t numVertices = static_cast<mt_kahypar_hypernode_id_t>(agents.size());
        const mt_kahypar_hyperedge_id_t numHyperedges = static_cast<mt_kahypar_hyperedge_id_t>(allKernels.size());
        if (numHyperedges == 0) {
            throw std::runtime_error("[Topology][MtKaHyPar] hypergraph has 0 hyperedges (no kernels in baskets). Refusing to continue.");
        }

        std::unordered_map<int, size_t> kernelToEdge;
        kernelToEdge.reserve(allKernels.size());
        for (size_t i = 0; i < allKernels.size(); ++i) kernelToEdge[allKernels[i]] = i;

        std::vector<std::vector<mt_kahypar_hypernode_id_t>> edgePins(allKernels.size());
        for (size_t v = 0; v < agents.size(); ++v) {
            for (int k : agentKernels[v]) {
                auto it = kernelToEdge.find(k);
                if (it != kernelToEdge.end()) edgePins[it->second].push_back(static_cast<mt_kahypar_hypernode_id_t>(v));
            }
        }

        // Build CSR-like incidence arrays expected by mt_kahypar_create_hypergraph.
        std::vector<size_t> hyperedge_indices(allKernels.size() + 1, 0);
        size_t totalPins = 0;
        for (size_t e = 0; e < edgePins.size(); ++e) {
            auto& pins = edgePins[e];
            std::sort(pins.begin(), pins.end());
            pins.erase(std::unique(pins.begin(), pins.end()), pins.end());
            totalPins += pins.size();
        }
        std::vector<mt_kahypar_hyperedge_id_t> hyperedges;
        hyperedges.reserve(totalPins);
        size_t off = 0;
        for (size_t e = 0; e < edgePins.size(); ++e) {
            hyperedge_indices[e] = off;
            for (auto v : edgePins[e]) {
                hyperedges.push_back(static_cast<mt_kahypar_hyperedge_id_t>(v));
                off++;
            }
        }
        hyperedge_indices[edgePins.size()] = off;

        // Node weight proxy:
        //   weight ~ (1 + |basket_kernels|) * (1000 / wakeupIntervalSeconds)
        // This captures: higher wakeup rate -> more work; more kernels -> more comm per wakeup.
        const double scale = 1000.0;
        std::vector<mt_kahypar_hypernode_weight_t> vertex_weights(agents.size(), 1);
        // Hyperedge (kernel) weight proxy:
        //   edge_weight(k) ~ sum_{agent: k in basket(agent)} (1000 / wakeupIntervalSeconds(agent))
        // Rationale: if a kernel is "hot" (touched frequently by many agents), then cutting it
        // across many parts is more expensive; weighting makes KM1 prioritize keeping hot kernels
        // within fewer blocks/nodes/ranks.
        std::vector<mt_kahypar_hyperedge_weight_t> hyperedge_weights(allKernels.size(), 1);
        double minIntv = std::numeric_limits<double>::infinity();
        double maxIntv = 0.0;
        double sumIntv = 0.0;
        size_t cntIntv = 0;
        int wMin = std::numeric_limits<int>::max();
        int wMax = 0;
        long long wSum = 0;
        long long ewMin = std::numeric_limits<long long>::max();
        long long ewMax = 0;
        long long ewSum = 0;
        for (size_t i = 0; i < agents.size(); ++i) {
            const double interval = loadWakeupIntervalSecondsFromCheckpointOrThrow(agents[i]);
            minIntv = std::min(minIntv, interval);
            maxIntv = std::max(maxIntv, interval);
            sumIntv += interval;
            cntIntv++;
            const int bk = (int)std::max<size_t>(1, agentKernels[i].size());
            const double wd = (1.0 + (double)bk) * (scale / interval);
            long long wi = (long long)std::llround(wd);
            if (wi < 1) wi = 1;
            const long long cap = (long long)std::numeric_limits<mt_kahypar_hypernode_weight_t>::max();
            if (wi > cap) wi = cap;
            vertex_weights[i] = (mt_kahypar_hypernode_weight_t)wi;
            wMin = std::min(wMin, (int)wi);
            wMax = std::max(wMax, (int)wi);
            wSum += wi;

            // Add this agent's wakeup rate contribution to each kernel it touches.
            // Contribution uses the same scale (1000) as vertex weights for consistency.
            const double rd = (scale / interval); // wakeup_rate ~= 1/interval
            long long ri = (long long)std::llround(rd);
            if (ri < 1) ri = 1;
            const long long ecap = (long long)std::numeric_limits<mt_kahypar_hyperedge_weight_t>::max();
            if (ri > ecap) ri = ecap;
            for (int k : agentKernels[i]) {
                auto itE = kernelToEdge.find(k);
                if (itE == kernelToEdge.end()) continue;
                const size_t e = itE->second;
                long long cur = (long long)hyperedge_weights[e];
                long long nxt = cur + ri;
                if (nxt > ecap) nxt = ecap;
                hyperedge_weights[e] = (mt_kahypar_hyperedge_weight_t)nxt;
            }
        }
        // Edge weight stats
        for (size_t e = 0; e < hyperedge_weights.size(); ++e) {
            long long v = (long long)hyperedge_weights[e];
            ewMin = std::min(ewMin, v);
            ewMax = std::max(ewMax, v);
            ewSum += v;
        }

        ensureMtKaHyParInitialized();

        mt_kahypar_error_t error{nullptr, 0, SUCCESS};
        mt_kahypar_context_t* ctx = mt_kahypar_context_from_preset(DETERMINISTIC);
        // Balance constraint: smaller epsilon => stricter balance (less load imbalance allowed).
        // Expected in [0,1]; validated by envDoubleInRange() when configured.
        mt_kahypar_set_partitioning_parameters(ctx, static_cast<mt_kahypar_partition_id_t>(nparts), epsilon, KM1);

        (void)mt_kahypar_set_context_parameter(ctx, VERBOSE, "0", &error);
        if (error.status != SUCCESS) {
            mt_kahypar_free_error_content(&error);
            error = {nullptr, 0, SUCCESS};
        }

        std::cout << "[Topology][MtKaHyPar] weights"
                  << " agents=" << agents.size()
                  << " kernels=" << allKernels.size()
                  << " wakeupIntervalSeconds(min/avg/max)="
                  << minIntv << "/"
                  << (cntIntv ? (sumIntv / (double)cntIntv) : 0.0) << "/"
                  << maxIntv
                  << " w(min/avg/max)="
                  << wMin << "/"
                  << (agents.empty() ? 0.0 : ((double)wSum / (double)agents.size())) << "/"
                  << wMax
                  << std::endl;

        auto hg = mt_kahypar_create_hypergraph(
            ctx,
            numVertices,
            numHyperedges,
            hyperedge_indices.data(),
            hyperedges.data(),
            /*hyperedge_weights=*/hyperedge_weights.data(),
            vertex_weights.data(),
            &error
        );
        if (error.status != SUCCESS || hg.type == NULLPTR_HYPERGRAPH || hg.hypergraph == nullptr) {
            std::string msg = "[Topology][MtKaHyPar] create_hypergraph failed status=" + std::to_string((int)error.status);
            if (error.msg) msg += std::string(" msg=") + error.msg;
            mt_kahypar_free_error_content(&error);
            mt_kahypar_free_context(ctx);
            throw std::runtime_error(msg);
        }

        auto phg = mt_kahypar_partition(hg, ctx, &error);
        if (error.status != SUCCESS || phg.partitioned_hg == nullptr) {
            std::string msg = "[Topology][MtKaHyPar] partition failed status=" + std::to_string((int)error.status)
                              + " vertices=" + std::to_string((unsigned long long)numVertices)
                              + " hyperedges=" + std::to_string((unsigned long long)numHyperedges)
                              + " k=" + std::to_string(nparts);
            if (error.msg) msg += std::string(" msg=") + error.msg;
            mt_kahypar_free_error_content(&error);
            mt_kahypar_free_hypergraph(hg);
            mt_kahypar_free_context(ctx);
            throw std::runtime_error(msg);
        }

        std::vector<mt_kahypar_partition_id_t> partIds(agents.size(), 0);
        mt_kahypar_get_partition(phg, partIds.data());

        const auto km1 = mt_kahypar_km1(phg);
        const auto imb = mt_kahypar_imbalance(phg, ctx);
        std::cout << "[Topology][MtKaHyPar] ok"
                  << " vertices=" << agents.size()
                  << " hyperedges=" << allKernels.size()
                  << " k=" << nparts
                  << " epsilon=" << epsilon
                  << " km1=" << km1
                  << " imbalance=" << imb
                  << std::endl;

        if (outMetrics) {
            nlohmann::json j;
            j["vertices"] = agents.size();
            j["hyperedges"] = allKernels.size();
            j["total_pins"] = totalPins;
            j["k"] = nparts;
            j["epsilon"] = epsilon;
            j["km1"] = km1;
            j["imbalance"] = imb;
            j["wakeupIntervalSeconds_min"] = std::isfinite(minIntv) ? minIntv : 0.0;
            j["wakeupIntervalSeconds_avg"] = (cntIntv ? (sumIntv / (double)cntIntv) : 0.0);
            j["wakeupIntervalSeconds_max"] = std::isfinite(maxIntv) ? maxIntv : 0.0;
            j["weight_min"] = (wMin == std::numeric_limits<int>::max()) ? 0 : wMin;
            j["weight_avg"] = agents.empty() ? 0.0 : ((double)wSum / (double)agents.size());
            j["weight_max"] = wMax;
            j["edge_weight_min"] = (ewMin == std::numeric_limits<long long>::max()) ? 0 : ewMin;
            j["edge_weight_avg"] = hyperedge_weights.empty() ? 0.0 : ((double)ewSum / (double)hyperedge_weights.size());
            j["edge_weight_max"] = ewMax;
            *outMetrics = std::move(j);
        }

        std::vector<int> out(agents.size(), 0);
        for (size_t i = 0; i < partIds.size(); ++i) {
            int p = (int)partIds[i];
            if (p < 0) p = 0;
            if (p >= nparts) p = p % nparts;
            out[i] = p;
        }

        mt_kahypar_free_partitioned_hypergraph(phg);
        mt_kahypar_free_hypergraph(hg);
        mt_kahypar_free_context(ctx);
        return out;
    }

    static inline std::vector<std::string> sortedAgentNamesFromBasket(
        const std::unordered_map<std::string, std::vector<std::string>>& agentBasket
    ) {
        std::vector<std::string> agents;
        agents.reserve(agentBasket.size());
        for (const auto& kv : agentBasket) agents.push_back(kv.first);
        std::sort(agents.begin(), agents.end());
        return agents;
    }

    static inline int nodeIdForRankSafe(const std::vector<int>& worldRankToNodeId, int rank) {
        if (rank >= 0 && rank < (int)worldRankToNodeId.size()) return worldRankToNodeId[(size_t)rank];
        return 0;
    }

    static inline std::unordered_map<int, std::vector<int>> buildCrossRanksByNode(
        const std::vector<int>& crossRanksAllSortedUnique,
        const std::vector<int>& worldRankToNodeId
    ) {
        std::unordered_map<int, std::vector<int>> out;
        out.reserve(16);
        for (int cr : crossRanksAllSortedUnique) {
            int nid = nodeIdForRankSafe(worldRankToNodeId, cr);
            out[nid].push_back(cr);
        }
        for (auto& kv : out) {
            auto& lst = kv.second;
            std::sort(lst.begin(), lst.end());
            lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
        }
        return out;
    }

    // Two-level partitioning:
    // 1) agents -> nodes (only nodes that have >=1 cross rank)
    // 2) within each node, agents -> cross ranks on that node
    //
    // Single-node environment naturally degenerates: the node-level partition has k=1.
    static inline std::unordered_map<std::string,int> hierarchicalPartitionAgentsToCrossRanks(
        const std::unordered_map<std::string,int>& asset2kernel,
        const std::unordered_map<std::string, std::vector<std::string>>& agentBasket,
        const std::vector<int>& crossRanksAllSortedUnique,
        const std::vector<int>& worldRankToNodeId
    ) {
        std::unordered_map<std::string,int> out;
        out.reserve(agentBasket.size());
        if (agentBasket.empty() || crossRanksAllSortedUnique.empty()) return out;

        // Nodes that have cross ranks.
        auto crossByNode = buildCrossRanksByNode(crossRanksAllSortedUnique, worldRankToNodeId);
        std::vector<int> nodes;
        nodes.reserve(crossByNode.size());
        for (const auto& kv : crossByNode) {
            if (!kv.second.empty()) nodes.push_back(kv.first);
        }
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        if (nodes.empty()) return out;

        const double epsNode = envDoubleInRange("TOPO_EPSILON_NODE", 0.05, 0.0, 1.0);
        const double epsIntra = envDoubleInRange("TOPO_EPSILON_INTRA", 0.05, 0.0, 1.0);

        // Level-1: assign each agent to a node id (index -> actual nodeId via nodes[]).
        const int nodeParts = (int)nodes.size();
        auto nodePartIds = mtKaHyParPartitionAgents(asset2kernel, agentBasket, std::max(1, nodeParts), epsNode);
        auto agentsAll = sortedAgentNamesFromBasket(agentBasket); // must match mtKaHyParPartitionAgents ordering
        if (nodePartIds.size() != agentsAll.size()) {
            throw std::runtime_error("[Topology][MtKaHyPar] hierarchical: nodePartIds size mismatch");
        }
        std::unordered_map<std::string,int> agentToNode;
        agentToNode.reserve(agentsAll.size());
        for (size_t i = 0; i < agentsAll.size(); ++i) {
            int p = nodePartIds[i];
            if (p < 0) p = 0;
            if (p >= nodeParts) p = p % nodeParts;
            agentToNode[agentsAll[i]] = nodes[(size_t)p];
        }

        // Group baskets by node.
        std::unordered_map<int, std::unordered_map<std::string, std::vector<std::string>>> basketByNode;
        basketByNode.reserve(nodes.size());
        for (const auto& kv : agentBasket) {
            const std::string& agent = kv.first;
            int nid = nodes.front();
            auto itN = agentToNode.find(agent);
            if (itN != agentToNode.end()) nid = itN->second;
            basketByNode[nid][agent] = kv.second;
        }

        // Level-2: within each node, partition to cross ranks on that node.
        for (int nid : nodes) {
            auto itB = basketByNode.find(nid);
            if (itB == basketByNode.end() || itB->second.empty()) continue;
            auto itC = crossByNode.find(nid);
            if (itC == crossByNode.end() || itC->second.empty()) {
                // Should not happen (nodes derived from crossByNode), but guard.
                const int fallback = crossRanksAllSortedUnique.front();
                for (const auto& kv : itB->second) out[kv.first] = fallback;
                continue;
            }
            const auto& crossRanksOnNode = itC->second;
            if (crossRanksOnNode.size() == 1) {
                const int only = crossRanksOnNode.front();
                for (const auto& kv : itB->second) out[kv.first] = only;
                continue;
            }

            auto parts = mtKaHyParPartitionAgents(asset2kernel, itB->second, (int)crossRanksOnNode.size(), epsIntra);
            auto agents = sortedAgentNamesFromBasket(itB->second); // must match mtKaHyParPartitionAgents ordering
            if (parts.size() != agents.size()) {
                throw std::runtime_error("[Topology][MtKaHyPar] hierarchical: rankParts size mismatch");
            }
            for (size_t i = 0; i < agents.size(); ++i) {
                int p = parts[i];
                if (p < 0) p = 0;
                if (p >= (int)crossRanksOnNode.size()) p = p % (int)crossRanksOnNode.size();
                out[agents[i]] = crossRanksOnNode[(size_t)p];
            }
        }

        return out;
    }

    // Same as hierarchicalPartitionAgentsToCrossRanks(), but also exposes intermediate results
    // for debugging/validation:
    // - outAgentToNode: agent -> node id (MPI shared-memory domain)
    // - outCrossByNode: node id -> cross ranks on that node
    // - outNodes: sorted node ids with >=1 cross rank
    static inline std::unordered_map<std::string,int> hierarchicalPartitionAgentsToCrossRanksWithTrace(
        const std::unordered_map<std::string,int>& asset2kernel,
        const std::unordered_map<std::string, std::vector<std::string>>& agentBasket,
        const std::vector<int>& crossRanksAllSortedUnique,
        const std::vector<int>& worldRankToNodeId,
        std::unordered_map<std::string,int>& outAgentToNode,
        std::unordered_map<int, std::vector<int>>& outCrossByNode,
        std::vector<int>& outNodes,
        std::vector<nlohmann::json>* outPartitionMetrics /*nullable*/ = nullptr
    ) {
        outAgentToNode.clear();
        outCrossByNode.clear();
        outNodes.clear();

        std::unordered_map<std::string,int> out;
        out.reserve(agentBasket.size());
        if (agentBasket.empty() || crossRanksAllSortedUnique.empty()) return out;

        outCrossByNode = buildCrossRanksByNode(crossRanksAllSortedUnique, worldRankToNodeId);
        outNodes.reserve(outCrossByNode.size());
        for (const auto& kv : outCrossByNode) {
            if (!kv.second.empty()) outNodes.push_back(kv.first);
        }
        std::sort(outNodes.begin(), outNodes.end());
        outNodes.erase(std::unique(outNodes.begin(), outNodes.end()), outNodes.end());
        if (outNodes.empty()) return out;

        const double epsNode = envDoubleInRange("TOPO_EPSILON_NODE", 0.05, 0.0, 1.0);
        const double epsIntra = envDoubleInRange("TOPO_EPSILON_INTRA", 0.05, 0.0, 1.0);

        const int nodeParts = (int)outNodes.size();
        nlohmann::json lvl1;
        auto nodePartIds = mtKaHyParPartitionAgents(
            asset2kernel,
            agentBasket,
            std::max(1, nodeParts),
            epsNode,
            outPartitionMetrics ? &lvl1 : nullptr
        );
        if (outPartitionMetrics) {
            lvl1["level"] = "agents_to_nodes";
            lvl1["epsilon"] = epsNode;
            lvl1["node_parts"] = nodeParts;
            lvl1["nodes"] = outNodes;
            outPartitionMetrics->push_back(std::move(lvl1));
        }
        auto agentsAll = sortedAgentNamesFromBasket(agentBasket); // must match mtKaHyParPartitionAgents ordering
        if (nodePartIds.size() != agentsAll.size()) {
            throw std::runtime_error("[Topology][MtKaHyPar] hierarchical(trace): nodePartIds size mismatch");
        }
        outAgentToNode.reserve(agentsAll.size());
        for (size_t i = 0; i < agentsAll.size(); ++i) {
            int p = nodePartIds[i];
            if (p < 0) p = 0;
            if (p >= nodeParts) p = p % nodeParts;
            outAgentToNode[agentsAll[i]] = outNodes[(size_t)p];
        }

        std::unordered_map<int, std::unordered_map<std::string, std::vector<std::string>>> basketByNode;
        basketByNode.reserve(outNodes.size());
        for (const auto& kv : agentBasket) {
            const std::string& agent = kv.first;
            int nid = outNodes.front();
            auto itN = outAgentToNode.find(agent);
            if (itN != outAgentToNode.end()) nid = itN->second;
            basketByNode[nid][agent] = kv.second;
        }

        for (int nid : outNodes) {
            auto itB = basketByNode.find(nid);
            if (itB == basketByNode.end() || itB->second.empty()) continue;
            auto itC = outCrossByNode.find(nid);
            if (itC == outCrossByNode.end() || itC->second.empty()) {
                const int fallback = crossRanksAllSortedUnique.front();
                for (const auto& kv : itB->second) out[kv.first] = fallback;
                continue;
            }
            const auto& crossRanksOnNode = itC->second;
            if (crossRanksOnNode.size() == 1) {
                const int only = crossRanksOnNode.front();
                for (const auto& kv : itB->second) out[kv.first] = only;
                continue;
            }
            nlohmann::json lvl2;
            auto parts = mtKaHyParPartitionAgents(asset2kernel, itB->second, (int)crossRanksOnNode.size(), epsIntra, outPartitionMetrics ? &lvl2 : nullptr);
            if (outPartitionMetrics) {
                lvl2["level"] = "agents_to_cross_within_node";
                lvl2["epsilon"] = epsIntra;
                lvl2["node_id"] = nid;
                lvl2["cross_ranks_on_node"] = crossRanksOnNode;
                outPartitionMetrics->push_back(std::move(lvl2));
            }
            auto agents = sortedAgentNamesFromBasket(itB->second);
            if (parts.size() != agents.size()) {
                throw std::runtime_error("[Topology][MtKaHyPar] hierarchical(trace): rankParts size mismatch");
            }
            for (size_t i = 0; i < agents.size(); ++i) {
                int p = parts[i];
                if (p < 0) p = 0;
                if (p >= (int)crossRanksOnNode.size()) p = p % (int)crossRanksOnNode.size();
                out[agents[i]] = crossRanksOnNode[(size_t)p];
            }
        }

        return out;
    }

    static inline void buildAndWriteNextAssignmentViaMtKaHyPar(
        const std::string& epochDate,
        const std::string& nextDate,
        const pugi::xml_node& rootNode,
        const std::vector<int>& crossRanksAll,
        const std::vector<int>& worldRankToNodeId
    ) {
        std::unordered_map<std::string,int> asset2kernel;
        std::vector<std::string> assets;
        (void)parseMultiKernelTargets(rootNode, asset2kernel, assets);

        std::unordered_map<std::string, std::vector<std::string>> agentBasket;
        std::vector<GroupInfo> groups;
        if (!loadAllBasketsForDate(epochDate, asset2kernel, agentBasket, groups)) {
            std::cerr << "[Topology][MtKaHyPar] no basket data for date=" << epochDate << std::endl;
            return;
        }

        // Map part -> actual cross rank id (sorted)
        std::vector<int> crossRanks = crossRanksAll;
        std::sort(crossRanks.begin(), crossRanks.end());
        crossRanks.erase(std::unique(crossRanks.begin(), crossRanks.end()), crossRanks.end());
        if (crossRanks.empty()) {
            std::cerr << "[Topology][MtKaHyPar] no cross ranks configured; skip assignment update" << std::endl;
            return;
        }

        // Two-level mapping agent -> cross rank:
        // - level1: agents -> nodes
        // - level2: nodes -> cross ranks on that node
        std::unordered_map<std::string,int> agentToNode;
        std::unordered_map<int, std::vector<int>> crossByNode;
        std::vector<int> nodesWithCross;
        std::vector<nlohmann::json> partMetrics;
        auto agentToCross = hierarchicalPartitionAgentsToCrossRanksWithTrace(
            asset2kernel, agentBasket, crossRanks, worldRankToNodeId,
            agentToNode, crossByNode, nodesWithCross,
            &partMetrics
        );

        // Start from previous day's assignment (carry forward non-SPT agents like DataFactory/RL).
        nlohmann::json assign = readAssignmentFileOrThrow(epochDate);
        assign["date"] = nextDate;

        // Only assign SPT agents here (others keep their original ranks from current assignment file).
        // Deterministic agent order must match mtKaHyParPartitionAgents().
        std::vector<std::string> sptAgents;
        sptAgents.reserve(agentBasket.size());
        for (const auto& kv : agentBasket) sptAgents.push_back(kv.first);
        std::sort(sptAgents.begin(), sptAgents.end());
        for (size_t i = 0; i < sptAgents.size(); ++i) {
            const std::string& agent = sptAgents[i];
            int dstRank = -1;
            auto it = agentToCross.find(agent);
            if (it != agentToCross.end()) dstRank = it->second;
            if (dstRank < 0) {
                // Safety fallback: deterministic pick from all cross ranks.
                uint64_t h = splitmix64(fnv1a64(agent));
                dstRank = crossRanks[(size_t)(h % (uint64_t)crossRanks.size())];
            }
            assign["agents"][sptAgents[i]] = dstRank;
        }

        (void)writeJsonAtomic(assignmentFilePath(nextDate), assign);

        // Build kernel -> crossRanks mapping for next day (used to update MultiKernel crossAgentRanks).
        nlohmann::json kernelMap;
        kernelMap["date"] = nextDate;
        kernelMap["kernel_to_cross_ranks"] = nlohmann::json::object();
        std::unordered_map<int, std::set<int>> k2r;
        for (const auto& kv : agentBasket) {
            const std::string& agent = kv.first;
            if (!assign["agents"].contains(agent)) continue;
            if (!assign["agents"][agent].is_number_integer()) continue;
            int cr = assign["agents"][agent].get<int>();
            for (const auto& asset : kv.second) {
                auto itK = asset2kernel.find(asset);
                if (itK == asset2kernel.end()) continue;
                k2r[itK->second].insert(cr);
            }
        }

        // Keep uncovered kernels disconnected: serialize an explicit empty list instead of
        // forcing a fallback cross-rank connection.
        std::set<int> allKernelRanks;
        for (const auto& kv : asset2kernel) allKernelRanks.insert(kv.second);
        nlohmann::json uncoveredKernels = nlohmann::json::array();
        for (int k : allKernelRanks) {
            std::vector<int> lst;
            auto it = k2r.find(k);
            if (it != k2r.end()) {
                lst.assign(it->second.begin(), it->second.end());
                std::sort(lst.begin(), lst.end());
            } else {
                uncoveredKernels.push_back(k);
            }
            kernelMap["kernel_to_cross_ranks"][std::to_string(k)] = lst;
        }
        (void)writeJsonAtomic(crossTopologyDir() / (std::string("kernel_cross_") + nextDate + ".json"), kernelMap);

        {
            nlohmann::json j;
            j["date"] = nextDate;
            j["epochDate"] = epochDate;
            j["metrics"] = partMetrics;
            (void)writeJsonAtomic(crossTopologyDir() / (std::string("mtkahypar_metrics_") + nextDate + ".json"), j);
        }
        // Trace file to make hierarchical optimization observable.
        // This lets you verify the node-level partition and the per-node rank assignment.
        {
            nlohmann::json j;
            j["date"] = nextDate;
            j["world_rank_to_node_id"] = worldRankToNodeId;
            j["node_to_cross_ranks"] = nlohmann::json::object();
            for (const auto& kv : crossByNode) {
                j["node_to_cross_ranks"][std::to_string(kv.first)] = kv.second;
            }
            j["agents"] = nlohmann::json::object();
            for (const auto& agent : sptAgents) {
                nlohmann::json row;
                auto itN = agentToNode.find(agent);
                auto itC = agentToCross.find(agent);
                row["node"] = (itN == agentToNode.end()) ? 0 : itN->second;
                row["cross_rank"] = (itC == agentToCross.end()) ? -1 : itC->second;
                j["agents"][agent] = row;
            }
            j["uncovered_kernels"] = uncoveredKernels;
            j["uncovered_kernels_count"] = (int)uncoveredKernels.size();
            (void)writeJsonAtomic(crossTopologyDir() / (std::string("hierarchy_") + nextDate + ".json"), j);
        }

        std::cout << "[Topology][MtKaHyPar] epochDate=" << epochDate
                  << " nextDate=" << nextDate
                  << " agents=" << agentBasket.size()
                  << " groups(sig)=" << groups.size()
                  << " crossRanks=" << crossRanks.size()
                  << " out=" << assignmentFilePath(nextDate).string()
                  << std::endl;
    }

    // Baseline variant: update *connectivity* topology from baskets, but keep assignment fixed
    // (no migration / no load balancing).
    static inline void buildAndWriteNextKernelCrossMapNoMigration(
        const std::string& epochDate,
        const std::string& nextDate,
        const pugi::xml_node& rootNode,
        const std::vector<int>& crossRanksAll,
        const std::vector<int>& worldRankToNodeId
    ) {
        (void)worldRankToNodeId;
        std::unordered_map<std::string,int> asset2kernel;
        std::vector<std::string> assets;
        (void)parseMultiKernelTargets(rootNode, asset2kernel, assets);

        std::unordered_map<std::string, std::vector<std::string>> agentBasket;
        std::vector<GroupInfo> groups;
        if (!loadAllBasketsForDate(epochDate, asset2kernel, agentBasket, groups)) {
            std::cerr << "[Topology][BaselineTopo] no basket data for date=" << epochDate << std::endl;
            return;
        }

        // Cross rank list (deterministic).
        std::vector<int> crossRanks = crossRanksAll;
        std::sort(crossRanks.begin(), crossRanks.end());
        crossRanks.erase(std::unique(crossRanks.begin(), crossRanks.end()), crossRanks.end());
        if (crossRanks.empty()) {
            std::cerr << "[Topology][BaselineTopo] no cross ranks configured; skip kernel_cross update" << std::endl;
            return;
        }

        // Load current assignment (epochDate). In baseline this is usually the static XML mapping.
        nlohmann::json assign = readAssignmentFileOrThrow(epochDate);
        // Carry forward assignment unchanged (only update date field) so offline evaluation remains consistent.
        assign["date"] = nextDate;
        (void)writeJsonAtomic(assignmentFilePath(nextDate), assign);

        // Build kernel -> crossRanks mapping for next day based on baskets and fixed assignment.
        nlohmann::json kernelMap;
        kernelMap["date"] = nextDate;
        kernelMap["kernel_to_cross_ranks"] = nlohmann::json::object();
        std::unordered_map<int, std::set<int>> k2r;

        for (const auto& kv : agentBasket) {
            const std::string& agent = kv.first;
            if (!assign["agents"].contains(agent)) continue;
            if (!assign["agents"][agent].is_number_integer()) continue;
            int cr = assign["agents"][agent].get<int>();
            for (const auto& asset : kv.second) {
                auto itK = asset2kernel.find(asset);
                if (itK == asset2kernel.end()) continue;
                k2r[itK->second].insert(cr);
            }
        }

        // Keep uncovered kernels disconnected: serialize an explicit empty list instead of
        // forcing a fallback cross-rank connection.
        std::set<int> allKernelRanks;
        for (const auto& kv : asset2kernel) allKernelRanks.insert(kv.second);
        nlohmann::json uncoveredKernels = nlohmann::json::array();
        for (int k : allKernelRanks) {
            std::vector<int> lst;
            auto it = k2r.find(k);
            if (it != k2r.end()) {
                lst.assign(it->second.begin(), it->second.end());
                std::sort(lst.begin(), lst.end());
            } else {
                uncoveredKernels.push_back(k);
            }
            kernelMap["kernel_to_cross_ranks"][std::to_string(k)] = lst;
        }
        kernelMap["uncovered_kernels"] = uncoveredKernels;
        kernelMap["uncovered_kernels_count"] = (int)uncoveredKernels.size();

        (void)writeJsonAtomic(crossTopologyDir() / (std::string("kernel_cross_") + nextDate + ".json"), kernelMap);

        std::cout << "[Topology][BaselineTopo] epochDate=" << epochDate
                  << " nextDate=" << nextDate
                  << " agents=" << agentBasket.size()
                  << " groups(sig)=" << groups.size()
                  << " crossRanks=" << crossRanks.size()
                  << " out=" << (crossTopologyDir() / (std::string("kernel_cross_") + nextDate + ".json")).string()
                  << std::endl;
    }

    static inline std::unordered_map<int, std::vector<int>> readKernelCrossMapForDate(const std::string& yyyymmdd) {
        std::unordered_map<int, std::vector<int>> out;
        auto p = crossTopologyDir() / (std::string("kernel_cross_") + yyyymmdd + ".json");
        auto j = readJsonFile(p);
        if (!j.is_object()) return out;
        if (!j.contains("kernel_to_cross_ranks") || !j["kernel_to_cross_ranks"].is_object()) return out;
        for (auto it = j["kernel_to_cross_ranks"].begin(); it != j["kernel_to_cross_ranks"].end(); ++it) {
            int k = -1;
            try { k = std::stoi(it.key()); } catch (...) { continue; }
            if (!it.value().is_array()) continue;
            std::vector<int> lst;
            for (auto& v : it.value()) if (v.is_number_integer()) lst.push_back(v.get<int>());
            std::sort(lst.begin(), lst.end());
            lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
            out[k] = std::move(lst);
        }
        return out;
    }

    static inline void applyKernelCrossMapToMultiKernel(pugi::xml_node rootNode,
                                                        const std::unordered_map<int, std::vector<int>>& k2r) {
        auto mk = rootNode.child("MultiKernel");
        if (!mk) return;
        std::vector<int> missing;
        for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
            int kr = kn.attribute("rank").as_int(-1);
            if (kr < 0) continue;
            if (k2r.find(kr) == k2r.end()) missing.push_back(kr);
        }
        if (!missing.empty()) {
            std::ostringstream oss;
            oss << "[Topology][KernelCross][FATAL] incomplete kernel_cross map; missing kernels={";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i) oss << ",";
                oss << missing[i];
            }
            oss << "}";
            throw std::runtime_error(oss.str());
        }
        for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
            int kr = kn.attribute("rank").as_int(-1);
            if (kr < 0) continue;
            auto it = k2r.find(kr);
            std::string csv;
            if (it != k2r.end()) {
                for (size_t i = 0; i < it->second.size(); ++i) {
                    if (i) csv += ",";
                    csv += std::to_string(it->second[i]);
                }
            } else {
                csv.clear();
            }
            if (!kn.attribute("crossAgentRanks").empty()) {
                kn.attribute("crossAgentRanks").set_value(csv.c_str());
            } else {
                kn.append_attribute("crossAgentRanks").set_value(csv.c_str());
            }
        }
    }

    // ===== FundamentalModel checkpoint helpers (epoch resets) =====
    static inline Timestamp computeEpochStartNsFromRoot(const pugi::xml_node& rootNode) {
        Timestamp startNs = 0;
        try {
            pugi::xml_attribute att;
            std::string date = rootNode.attribute("date").as_string();
            if (!(att = rootNode.attribute("start")).empty()) {
                std::string startStr = att.as_string();
                if (!date.empty()) startNs = DateTimeConverter::dateTimeToNs(date, startStr);
                else if (std::all_of(startStr.begin(), startStr.end(), ::isdigit)) startNs = DateTimeConverter::marketTimeToNs(std::stoull(startStr));
                else if (startStr.find(':') != std::string::npos) startNs = DateTimeConverter::timeStringToNs(startStr);
                else startNs = static_cast<Timestamp>(att.as_ullong());
            }
        } catch (...) {
            startNs = 0;
        }
        return startNs;
    }

    static inline FundamentalValueModel::Config fundamentalCfgFromRoot(pugi::xml_node rootNode) {
        FundamentalValueModel::Config cfg;
        cfg.enabled = false;
        if (auto ga = rootNode.child("GlobalAgentConfig")) {
            if (auto rbarNode = ga.child("GlobalRBar"); rbarNode && rbarNode.attribute("value")) {
                cfg.r_bar = rbarNode.attribute("value").as_double(cfg.r_bar);
            }
            if (auto seedNode = ga.child("GlobalSeed"); seedNode && seedNode.attribute("value")) {
                cfg.seed = static_cast<uint64_t>(seedNode.attribute("value").as_ullong(cfg.seed));
            }
            if (auto fm = ga.child("FundamentalValueModel")) {
                cfg.enabled = fm.attribute("enabled").as_bool(false);
                double dtSec = fm.attribute("dtSeconds").as_double(60.0);
                if (!(dtSec > 0.0) || !std::isfinite(dtSec)) dtSec = 60.0;
                cfg.dt_ns = static_cast<uint64_t>(dtSec * 1e9);
                cfg.kappa = fm.attribute("kappa").as_double(0.0);
                cfg.sigma_s = fm.attribute("sigmaS").as_double(0.0);
                cfg.shockClampPct = fm.attribute("shockClampPct").as_double(0.05);
                cfg.checkpointEnabled = fm.attribute("checkpointEnabled").as_bool(false);
                if (!fm.attribute("checkpointDir").empty()) cfg.checkpointDir = fm.attribute("checkpointDir").as_string();
            }
        }
        return cfg;
    }

    // MPMD multi-asset support:
    // - Each kernel segment uses its own XML (asset-specific) and therefore has the correct fundamental config for that asset.
    // - Cross ranks are started with first_cfg and MUST NOT use that XML's fundamental config for all assets.
    //   Instead, we collect per-asset fundamental configs from all kernel ranks via epochComm.
    //
    // Returns a map asset -> cfg for all kernels in the current job.
    static inline std::unordered_map<std::string, FundamentalValueModel::Config>
    allgatherFundamentalCfgsFromKernels(MPI_Comm epochComm,
                                        bool isKernel,
                                        const std::string& assetNameForLogs,
                                        pugi::xml_node rootNode) {
        std::unordered_map<std::string, FundamentalValueModel::Config> out;
        if (epochComm == MPI_COMM_NULL) return out;
        int commRank = 0, commSize = 0;
        MPI_Comm_rank(epochComm, &commRank);
        MPI_Comm_size(epochComm, &commSize);
        if (commSize <= 0) return out;

        std::string local;
        if (isKernel) {
            FundamentalValueModel::Config cfg = fundamentalCfgFromRoot(rootNode);
            nlohmann::json j;
            j["asset"] = assetNameForLogs;
            j["cfg"] = {
                {"enabled", cfg.enabled},
                {"dtNs", static_cast<unsigned long long>(cfg.dt_ns)},
                {"rBar", cfg.r_bar},
                {"kappa", cfg.kappa},
                {"sigmaS", cfg.sigma_s},
                {"shockClampPct", cfg.shockClampPct},
                {"seed", static_cast<unsigned long long>(cfg.seed)},
                {"checkpointEnabled", cfg.checkpointEnabled},
                {"checkpointDir", cfg.checkpointDir}
            };
            local = j.dump();
        }

        int myLen = static_cast<int>(local.size());
        std::vector<int> lens(static_cast<size_t>(commSize), 0);
        MPI_Allgather(&myLen, 1, MPI_INT, lens.data(), 1, MPI_INT, epochComm);

        std::vector<int> displs(static_cast<size_t>(commSize), 0);
        int total = 0;
        for (int i = 0; i < commSize; ++i) {
            displs[(size_t)i] = total;
            total += std::max(0, lens[(size_t)i]);
        }
        if (total <= 0) return out;

        std::vector<char> recv(static_cast<size_t>(total));
        MPI_Allgatherv(local.data(), myLen, MPI_CHAR,
                       recv.data(), lens.data(), displs.data(), MPI_CHAR,
                       epochComm);

        for (int i = 0; i < commSize; ++i) {
            int len = lens[(size_t)i];
            if (len <= 0) continue;
            int off = displs[(size_t)i];
            if (off < 0 || off + len > total) continue;
            std::string s(recv.data() + off, recv.data() + off + len);
            auto j = nlohmann::json::parse(s, nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.is_object()) continue;
            if (!j.contains("asset") || !j["asset"].is_string()) continue;
            if (!j.contains("cfg") || !j["cfg"].is_object()) continue;
            std::string asset = j["asset"].get<std::string>();
            if (asset.empty()) continue;
            auto& c = j["cfg"];
            FundamentalValueModel::Config cfg;
            cfg.enabled = c.contains("enabled") ? c["enabled"].get<bool>() : false;
            if (c.contains("dtNs") && c["dtNs"].is_number_integer()) cfg.dt_ns = static_cast<uint64_t>(c["dtNs"].get<long long>());
            if (c.contains("rBar") && c["rBar"].is_number()) cfg.r_bar = c["rBar"].get<double>();
            if (c.contains("kappa") && c["kappa"].is_number()) cfg.kappa = c["kappa"].get<double>();
            if (c.contains("sigmaS") && c["sigmaS"].is_number()) cfg.sigma_s = c["sigmaS"].get<double>();
            if (c.contains("shockClampPct") && c["shockClampPct"].is_number()) cfg.shockClampPct = c["shockClampPct"].get<double>();
            if (c.contains("seed") && c["seed"].is_number_integer()) cfg.seed = static_cast<uint64_t>(c["seed"].get<long long>());
            cfg.checkpointEnabled = c.contains("checkpointEnabled") ? c["checkpointEnabled"].get<bool>() : false;
            if (c.contains("checkpointDir") && c["checkpointDir"].is_string()) cfg.checkpointDir = c["checkpointDir"].get<std::string>();

            if (out.find(asset) == out.end()) {
                out.emplace(asset, cfg);
            }
        }

        return out;
    }

    static inline void logFundamentalCfgOneLine(const char* tag,
                                                int rank,
                                                const char* role,
                                                const std::string& asset,
                                                const FundamentalValueModel::Config& cfg) {
        const double dtSeconds = static_cast<double>(cfg.dt_ns) / 1e9;
        std::cout << "[FundamentalCfg]"
                  << " tag=" << (tag ? tag : "")
                  << " rank=" << rank
                  << " role=" << (role ? role : "")
                  << " asset=" << asset
                  << " enabled=" << (cfg.enabled ? "true" : "false")
                  << " rBar=" << cfg.r_bar
                  << " kappa=" << cfg.kappa
                  << " sigmaS=" << cfg.sigma_s
                  << " dtSeconds=" << dtSeconds
                  << " shockClampPct=" << cfg.shockClampPct
                  << " seed=" << cfg.seed
                  << " ckpt=" << (cfg.checkpointEnabled ? "true" : "false")
                  << " ckptDir=" << cfg.checkpointDir
                  << std::endl;
    }

    static inline void configureAndLoadFundamentalModelForEpoch(pugi::xml_node rootNode,
                                                                const std::vector<std::string>& assetsToLoad) {
        auto cfg = fundamentalCfgFromRoot(rootNode);
        FundamentalValueModel::instance().configure(cfg);
        if (!cfg.enabled || !cfg.checkpointEnabled) return;
        Timestamp startNs = computeEpochStartNsFromRoot(rootNode);
        for (const auto& asset : assetsToLoad) {
            if (asset.empty()) continue;
            if (asset == "CROSS") continue;
            FundamentalValueModel::instance().loadCheckpointForAsset(asset, startNs);
        }
    }

    static inline void saveFundamentalModelForEpoch(pugi::xml_node rootNode,
                                                    const std::string& assetNameForLogs,
                                                    Timestamp epochCloseNs) {
        auto cfg = FundamentalValueModel::instance().config();
        if (!cfg.enabled || !cfg.checkpointEnabled) return;
        Timestamp startNs = computeEpochStartNsFromRoot(rootNode);
        if (!assetNameForLogs.empty() && assetNameForLogs != "CROSS") {
            FundamentalValueModel::instance().saveCheckpointForAsset(assetNameForLogs, startNs, epochCloseNs);
        }
    }
}

int main(int argc, char* argv[]) { 
    // Select MPI threading requirement by env:
    // - DESMAR_MPI_MODE=multiple: request MPI_THREAD_MULTIPLE (legacy)
    // - DESMAR_MPI_MODE=proxy:    request MPI_THREAD_SERIALIZED (single MPI thread in-process)
    auto envLower = [](const char* name, const char* defVal) -> std::string {
        const char* v = std::getenv(name);
        std::string s = (v && *v) ? std::string(v) : std::string(defVal ? defVal : "");
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    std::string mpiMode = envLower("DESMAR_MPI_MODE", "multiple");
    int required = MPI_THREAD_MULTIPLE;
    if (mpiMode == "proxy" || mpiMode == "single" || mpiMode == "single_thread" || mpiMode == "singlethread") {
        required = MPI_THREAD_SERIALIZED;
    }
    int provided = 0;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
        std::cerr << "[FATAL][MPI] MPI_Init_thread: required=" << required << " provided=" << provided
                  << " (DESMAR_MPI_MODE=" << mpiMode << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "[MPI][Init] rank=" << rank
              << " DESMAR_MPI_MODE=" << mpiMode
              << " required=" << required
              << " provided=" << provided
              << std::endl;

    // Default label for the main thread (helps attribute max-thread bottlenecks).
    DesmarMpiApiProfiler::RegisterThreadLabel("main");
    
    int maxRank = (size > 0) ? (size - 1) : 0;
    std::cout << "Ranks 0-" << maxRank << std::endl;
    std::cout << "Process rank " << rank << " of " << maxRank << " started" << std::endl;
    // NOTE (MPMD):
    // Learner ranks may run a different executable (Python learner_mpi.py) and must NOT be forced to
    // participate in C++ startup collectives. Therefore we avoid MPI_COMM_WORLD collectives here.
    //
    // Fallback: assume single-node placement (all ranks -> node 0). This is sufficient for correctness,
    // but may reduce the accuracy of node-aware constraints in multi-node partitioning experiments.
    std::vector<int> worldRankToNodeId(std::max(0, size), 0);
    auto __kernel_proc_start = std::chrono::steady_clock::now();
    
    try {
        std::string configFile = "Simulator_configs/config_templates/SimulationKernel_configs_template.xml";
        if (argc > 1) {
            configFile = argv[1];
        }
        
        pugi::xml_document doc;
        if (!doc.load_file(configFile.c_str())) {
            std::cerr << "Failed to load config file: " << configFile << std::endl;
            MPI_Finalize();
            return 1;
        }
        
        auto rootNode = doc.child("Simulation");
        if (rootNode.empty()) {
            std::cerr << "Invalid config file format" << std::endl;
            MPI_Finalize();
            return 1;
        }
        namespace fs = std::filesystem;
        const char* envLogRoot = std::getenv("DESMAR_LOG_ROOT");
        fs::path logRoot = envLogRoot && *envLogRoot ? fs::path(envLogRoot) : fs::path("distributed_logs");

        auto getExchangeNameFromConfig = [&]() -> std::string {
            std::string name;
            if (auto core = rootNode.child("CoreRank")) {
                if (auto ex = core.child("ExchangeAgent")) {
                    auto a = ex.attribute("name");
                    if (!a.empty()) name = a.as_string();
                }
            }
            if (name.empty()) {
                for (auto ex = rootNode.child("ExchangeAgent"); ex; ex = ex.next_sibling("ExchangeAgent")) {
                    auto a = ex.attribute("name");
                    if (!a.empty()) { name = a.as_string(); break; }
                }
            }
            return name;
        };
        std::string assetNameForLogs = getExchangeNameFromConfig();
        if (assetNameForLogs.empty()) assetNameForLogs = "UNKNOWN_ASSET";

        // Multi-day (epoch) loop controls:
        // - DESMAR_EPOCHS: number of trading days (default 1)
        // - DESMAR_MASTER_SEED: master seed for per-day seed derivation (default: GlobalSeed from XML)
        const int epochCount = std::max(1, envInt("DESMAR_EPOCHS", 1));

        std::string baseDate = rootNode.attribute("date").as_string();
        if (baseDate.empty()) baseDate = "20230101";
        {
            int yy=0, mm=0, dd=0;
            if (!parseYYYYMMDD(baseDate, yy, mm, dd)) {
                std::cerr << "[Epoch] invalid base date in XML: '" << baseDate << "', fallback to 20230101" << std::endl;
                baseDate = "20230101";
            }
        }

        // Experiment-level start date (stable across process-level restarts).
        // We keep it in XML as baseDate; scripts set it once in the temp workdir.
        // This allows us to derive a correct global epochIdx even when each day runs in a fresh process.
        std::string experimentBaseDate = rootNode.attribute("baseDate").as_string();
        if (experimentBaseDate.empty()) experimentBaseDate = baseDate;
        {
            int yy=0, mm=0, dd=0;
            if (!parseYYYYMMDD(experimentBaseDate, yy, mm, dd)) {
                experimentBaseDate = baseDate;
            }
        }

        uint64_t globalSeedFromXml = 1;
        if (auto ga = rootNode.child("GlobalAgentConfig")) {
            if (auto gs = ga.child("GlobalSeed")) {
                auto a = gs.attribute("value");
                if (!a.empty()) globalSeedFromXml = a.as_ullong(1ull);
            }
        }
        const uint64_t masterSeed = envU64("DESMAR_MASTER_SEED", globalSeedFromXml);
        // Cross-agent basket cap (max number of assets in next-day tradable basket, excluding locked holdings).
        // Default matches previous hard-coded behavior (5). Clamp to >=1.
        const int basketCapN_int = std::max(1, envInt("DESMAR_BASKET_CAP", 5));
        const size_t basketCapN = static_cast<size_t>(basketCapN_int);
        // Per-epoch asset evaluation budget (0 => evaluate all assets; >total => clamped later).
        const int assetEvalN_int = std::max(0, envInt("DESMAR_ASSET_EVAL_N", 0));
        const size_t assetEvalN = static_cast<size_t>(assetEvalN_int);
        auto rankLogDirFor = [&](int r) -> std::string {
            fs::path p = logRoot;
            if (!assetNameForLogs.empty()) p /= assetNameForLogs;
            p /= std::to_string(r);
            std::error_code ec;
            fs::create_directories(p, ec);
            return p.string();
        };
        
        int simRankCfg = 0;
        std::vector<int> agentRanksCfg;
        std::vector<int> crossAgentRanksCfg;
        std::vector<int> learnerRanksCfg;
        {
            auto mpiCfg = rootNode.child("MPIConfiguration");
            if (mpiCfg) {
                simRankCfg = mpiCfg.child("SimulationRank").text().as_int(0);
                std::string agentsStr = mpiCfg.child("AgentRanks").text().as_string();
                if (!agentsStr.empty()) {
                    std::stringstream ss(agentsStr);
                    std::string item;
                    while (std::getline(ss, item, ',')) {
                        if (!item.empty()) agentRanksCfg.push_back(std::stoi(item));
                    }
                }
                std::string crossStr = mpiCfg.child("CrossAgentRanks").text().as_string();
                if (!crossStr.empty()) {
                    std::stringstream ss2(crossStr);
                    std::string it2;
                    while (std::getline(ss2, it2, ',')) {
                        if (!it2.empty()) crossAgentRanksCfg.push_back(std::stoi(it2));
                    }
                }
                std::string learnStr = mpiCfg.child("LearningRanks").text().as_string();
                if (!learnStr.empty()) {
                    std::stringstream ss3(learnStr);
                    std::string it3;
                    while (std::getline(ss3, it3, ',')) {
                        if (!it3.empty()) learnerRanksCfg.push_back(std::stoi(it3));
                    }
                }
            }
        }

        int simRankGlobal = simRankCfg;
        std::vector<int> agentRanksGlobal = agentRanksCfg;
        bool isKernel = (rank == simRankGlobal);
        bool isAgent = false;
        bool isLearner = false;
        if (!isKernel) {
            for (int ar : agentRanksGlobal) { if (ar == rank) { isAgent = true; break; } }
            if (!isAgent) {
                for (int cr : crossAgentRanksCfg) { if (cr == rank) { isAgent = true; break; } }
            }
            if (!isAgent) {
                for (int lr : learnerRanksCfg) { if (lr == rank) { isLearner = true; break; } }
            }
        }
        {
            std::cout << "[DistributedMain] rank=" << rank
                      << " cfgFile=" << configFile
                      << " simRankGlobal=" << simRankGlobal
                      << " isKernel=" << (isKernel?"true":"false")
                      << " isAgent=" << (isAgent?"true":"false")
                      << " isLearner=" << (isLearner?"true":"false")
                      << " agentRanksGlobal={";
            for (size_t i=0;i<agentRanksGlobal.size();++i){ std::cout << agentRanksGlobal[i] << (i+1<agentRanksGlobal.size()? ",":""); }
            std::cout << "}" << std::endl;
        }

        // ===== C++-only communicator (exclude Python learner ranks) =====
        //
        // MPMD note:
        // Learner ranks run a Python executable and do NOT participate in arbitrary C++ collectives.
        // We therefore build a communicator that excludes learner ranks, and run C++ collectives on it.
        MPI_Comm commCppOnly = MPI_COMM_NULL;
        bool amCpp = true;
        {
            std::vector<int> learnerRanksEnv;
            if (const char* envL = std::getenv("DESMAR_LEARNER_RANKS")) {
                std::stringstream ss(envL); std::string t; while (std::getline(ss, t, ',')) { if (!t.empty()) { try { learnerRanksEnv.push_back(std::stoi(t)); } catch(...){} } }
            }
            std::vector<int> learnerSet = learnerRanksEnv.empty() ? learnerRanksCfg : learnerRanksEnv;
            std::sort(learnerSet.begin(), learnerSet.end()); learnerSet.erase(std::unique(learnerSet.begin(), learnerSet.end()), learnerSet.end());
            amCpp = (std::find(learnerSet.begin(), learnerSet.end(), rank) == learnerSet.end());
            if (amCpp) {
                MPI_Group worldGroup; MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
                std::vector<int> incl; incl.reserve(size);
                for (int r = 0; r < size; ++r) {
                    if (!std::binary_search(learnerSet.begin(), learnerSet.end(), r)) incl.push_back(r);
                }
                MPI_Group cppGroup; MPI_Group_incl(worldGroup, (int)incl.size(), incl.data(), &cppGroup);
                MPI_Comm newComm = MPI_COMM_NULL;
                std::cout << "[CPP_ONLY_COMM][CREATE_ENTER] rank=" << rank
                          << " members=" << formatIntList(incl)
                          << " tag=" << CPP_ONLY_COMM_CREATE_TAG << std::endl;
                MPI_Comm_create_group(MPI_COMM_WORLD, cppGroup, CPP_ONLY_COMM_CREATE_TAG, &newComm);
                std::cout << "[CPP_ONLY_COMM][CREATE_EXIT] rank=" << rank
                          << " commNull=" << (newComm == MPI_COMM_NULL ? "true" : "false")
                          << " tag=" << CPP_ONLY_COMM_CREATE_TAG << std::endl;
                MPI_Group_free(&cppGroup); MPI_Group_free(&worldGroup);
                commCppOnly = newComm;
                if (commCppOnly == MPI_COMM_NULL) {
                    std::cerr << "Failed to create commCppOnly on rank " << rank << std::endl;
                }
            }
        }

        if (!amCpp) {
            // This rank is reserved for learner (Python). It must not enter C++ collectives.
            MPI_Finalize();
            return 0;
        }
        if (commCppOnly == MPI_COMM_NULL) {
            std::cerr << "[FATAL] commCppOnly is NULL on rank " << rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 92);
        }

        MPI_Comm perKernelComm = MPI_COMM_NULL;
        {
            int color = (isKernel || isAgent) ? simRankGlobal : MPI_UNDEFINED;
            MPI_Comm newComm = MPI_COMM_NULL;
            MPI_Comm_split(commCppOnly, color, rank, &newComm);
            perKernelComm = newComm;
        }

        MPI_Comm epochComm = MPI_COMM_NULL;
        {
            int color = (isKernel || isAgent) ? 1 : MPI_UNDEFINED;
            MPI_Comm newComm = MPI_COMM_NULL;
            MPI_Comm_split(commCppOnly, color, rank, &newComm);
            epochComm = newComm;
        }
        if ((isKernel || isAgent) && epochComm == MPI_COMM_NULL) {
            std::cerr << "[FATAL] epochComm is NULL for a kernel/agent rank=" << rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 91);
        }

        std::vector<int> discoveredKernelRanks;
        {
            const char* envKernels = std::getenv("DESMAR_KERNEL_RANKS");
            if (envKernels && *envKernels) {
                std::stringstream ss(envKernels);
                std::string tok; while (std::getline(ss, tok, ',')) {
                    try { int v = std::stoi(tok); if (v >= 0) discoveredKernelRanks.push_back(v); } catch(...) {}
                }
                std::sort(discoveredKernelRanks.begin(), discoveredKernelRanks.end());
                discoveredKernelRanks.erase(std::unique(discoveredKernelRanks.begin(), discoveredKernelRanks.end()), discoveredKernelRanks.end());
            } else {
                if (!learnerRanksCfg.empty()) {
                    throw std::runtime_error("DESMAR_KERNEL_RANKS is required in MPMD (learner enabled) to avoid MPI_COMM_WORLD collectives");
                }
                MPICommunicationManager tmpMgr; tmpMgr.initialize(false);
                discoveredKernelRanks = tmpMgr.discoverKernelRanks(simRankGlobal);
            }
        }

        // Baseline mode (for experiments):
        //
        // IMPORTANT:
        // A correct "cut+balance baseline" must NOT change connectivity semantics.
        // Otherwise the speedup might come from removing full-mesh traffic rather than from partitioning/balancing.
        //
        // Therefore we split baseline behavior into independent switches:
        // - DESMAR_BASELINE=true:
        //     disable migration / graph optimization (Mt-KaHyPar + assignment update), but keep basket evaluation.
        // - DESMAR_BASELINE_FULL_MESH=true (optional legacy):
        //     force full-mesh connectivity (kernel<->kernel, kernel<->cross), ignoring sparse MultiKernel mapping.
        // - Baseline must still update the next-day *connectivity* topology from baskets
        //   (kernel_cross_<date>.json), but MUST NOT repartition agents / do load balancing.
        //   This guarantees that baseline isolates the effect of cut/balance only.
        const bool baseline = envBoolStrict("DESMAR_BASELINE", false);
        const bool baselineNoMigration = baseline;
        const bool baselineFullMesh = envBoolStrict("DESMAR_BASELINE_FULL_MESH", false);
        const bool baselineTopoFromBaskets = baseline;
        std::cout << "[RunConfig] rank=" << rank
                  << " DESMAR_EPOCHS=" << epochCount
                  << " DESMAR_MASTER_SEED=" << masterSeed
                  << " DESMAR_BASKET_CAP=" << basketCapN
                  << " DESMAR_ASSET_EVAL_N=" << assetEvalN
                  << " DESMAR_BASELINE=" << (baseline ? "true" : "false")
                  << " DESMAR_BASELINE_FULL_MESH=" << (baselineFullMesh ? "true" : "false")
                  << std::endl;
        static bool s_printBaselineOnce = false;
        if ((baseline || baselineFullMesh) && !s_printBaselineOnce) {
            s_printBaselineOnce = true;
            std::cout << "[BASELINE] enabled:"
                      << " DESMAR_BASELINE=" << (baseline?"1":"0")
                      << " DESMAR_BASELINE_FULL_MESH=" << (baselineFullMesh?"1":"0")
                      << std::endl;
        }

        // Barrier tracing (always on): helps locate which rank fails to reach a barrier.
        // This prints 2 lines per barrier per rank: [Barrier][Enter] and [Barrier][Exit].
        auto tracedBarrier = [&](MPI_Comm comm, const char* name, int epochIdx, const std::string& epochDate) {
            if (comm == MPI_COMM_NULL) {
                // Non-participating ranks (e.g., learner) do not enter epoch pipeline barriers.
                return;
            }
            MPI_Comm c = comm;
            int csize = -1;
            MPI_Comm_size(c, &csize);
            std::cout << "[Barrier][Enter] rank=" << rank
                      << " role=" << (isKernel ? "kernel" : (isAgent ? "agent" : (isLearner ? "learner" : "other")))
                      << " epoch=" << epochIdx
                      << " date=" << epochDate
                      << " name=" << (name ? name : "")
                      << " commSize=" << csize
                      << std::endl;
            MPI_Barrier(c);
            std::cout << "[Barrier][Exit] rank=" << rank
                      << " role=" << (isKernel ? "kernel" : (isAgent ? "agent" : (isLearner ? "learner" : "other")))
                      << " epoch=" << epochIdx
                      << " date=" << epochDate
                      << " name=" << (name ? name : "")
                      << " commSize=" << csize
                      << std::endl;
        };

        if (isKernel) {
            std::cout << "Starting distributed simulation core on rank " << rank << " (epochs=" << epochCount << ")" << std::endl;

            std::string curDate = baseDate;
            std::string prevEpochDate;

            // Ensure base-date assignment exists (used by cross-rank offline evaluation).
            if (rank == 0) {
                const auto p = assignmentFilePath(baseDate);
                // IMPORTANT:
                // In "one epoch per process" runs, the next epoch's assignment file may already exist
                // (written by the previous process). Do NOT overwrite it here.
                if (!std::filesystem::exists(p)) {
                    auto initAssign = buildInitialAssignmentFromConfig(rootNode, baseDate);
                    (void)writeJsonAtomic(p, initAssign);
                }
            }
            tracedBarrier(epochComm, "init_assignment_ready", /*epochIdx*/-1, baseDate);

            for (int epochLocal = 0; epochLocal < epochCount; ++epochLocal) {
                // Rank0 is the global epoch coordinator (as requested).
                char dateBuf[9] = {0};
                uint64_t daySeed = 0;
                if (rank == 0) {
                    if (epochLocal == 0) {
                        curDate = baseDate;
                    } else {
                        curDate = nextTradingDateSkipWeekend(curDate);
                    }
                    daySeed = deriveDaySeed(masterSeed, curDate);
                    std::snprintf(dateBuf, sizeof(dateBuf), "%s", curDate.c_str());
                }
                MPI_Bcast(dateBuf, 9, MPI_CHAR, 0, epochComm);
                MPI_Bcast(&daySeed, 1, MPI_UNSIGNED_LONG_LONG, 0, epochComm);
                std::string epochDate(dateBuf);
                if (epochDate.size() != 8) epochDate = baseDate;
                // IMPORTANT: global epochIdx is derived from (experimentBaseDate, epochDate),
                // so it remains correct in both in-process multi-epoch and process-level restart modes.
                const int epochIdx = tradingDayIndexFromStart(experimentBaseDate, epochDate);

                // Overwrite stdout/stderr per epoch to avoid unbounded log growth.
                // Kernel ranks are under <logRoot>/<asset>/<rank>/stdout|stderr.
                overwriteStdoutStderrPerEpoch(logRoot, assetNameForLogs, rank);

                // Apply in-memory overrides to XML (do NOT write back to disk).
                if (!rootNode.attribute("date").empty()) {
                    rootNode.attribute("date").set_value(epochDate.c_str());
                } else {
                    rootNode.append_attribute("date").set_value(epochDate.c_str());
                }
                // One-epoch-per-process support: if a kernel_cross_<date>.json exists (written by previous day),
                // apply it for THIS epoch so connectivity/placement stays consistent without in-process epoch chaining.
                if (!baselineFullMesh && !crossAgentRanksCfg.empty()) {
                    auto k2r_now = readKernelCrossMapForDate(epochDate);
                    if (epochDate != baseDate && k2r_now.empty()) {
                        throw std::runtime_error("[Topology][KernelCross][FATAL] missing kernel_cross map for date=" + epochDate);
                    }
                    if (!k2r_now.empty()) {
                        applyKernelCrossMapToMultiKernel(rootNode, k2r_now);
                    }
                }
                {
                    auto ga = rootNode.child("GlobalAgentConfig");
                    if (ga) {
                        // Do NOT override GlobalSeed per epoch. GlobalSeed is intended to be a single
                        // global seed shared across all ranks (including cross ranks) so that shared
                        // modules (e.g., FundamentalValueModel) remain consistent.
                        //
                        // But we DO derive per-epoch per-asset seed for agent instantiation randomness.
                        const uint64_t assetSeed = deriveAssetDaySeed(daySeed, assetNameForLogs);
                        pugi::xml_node as = ga.child("AssetSeed");
                        if (!as) as = ga.append_child("AssetSeed");
                        auto a = as.attribute("value");
                        if (a.empty()) a = as.append_attribute("value");
                        a.set_value(std::to_string(assetSeed).c_str());

                    }
                }

                std::cout << "[Epoch] rank=" << rank
                          << " epoch=" << epochIdx
                          << " date=" << epochDate
                          << " daySeed=" << daySeed
                          << std::endl;

                // Baseline: ensure per-day assignment_<date>.json exists and stays stable (no migration),
                // so basket evaluation can still find which SPT agents belong to which cross rank.
                if (baseline && rank == 0) {
                    const auto p = assignmentFilePath(epochDate);
                    // In "one epoch per process" runs, assignment_<date>.json might have been produced by the previous day
                    // (carry-forward). Do NOT overwrite it here.
                    if (!std::filesystem::exists(p)) {
                        auto assign = buildInitialAssignmentFromConfig(rootNode, epochDate);
                        (void)writeJsonAtomic(p, assign);
                    }
                }
                tracedBarrier(epochComm, "epoch_assignment_ready", epochIdx, epochDate);

                // ===== Per-asset FundamentalValueModel config sync (MPMD) =====
                // All ranks in epochComm must call this collective in the same order.
                // Kernel ranks contribute (assetName -> cfg) from their own XML; non-kernel ranks contribute empty.
                auto fundamentalCfgsByAsset =
                    allgatherFundamentalCfgsFromKernels(epochComm, /*isKernel=*/true, assetNameForLogs, rootNode);
                (void)fundamentalCfgsByAsset; // kernel uses its local asset cfg; map is for cross/agent side

                // Day-2+ opening price adjustment:
                // Use previous trading day's last TRADE price as mid, then +/- 2 ticks (default tick=0.01).
                // Only applied on kernel ranks because SetupAgent is on CoreRank.
                {
                    // No manual env var needed:
                    // - If we're in-process multi-epoch, prevEpochDate is set at end of previous epoch.
                    // - If we're doing process-level restarts (one day per process), infer previous trading day
                    //   from the current date (skip weekends).
                    const std::string prevDateForOpen = prevEpochDate.empty()
                                                        ? prevTradingDateSkipWeekend(epochDate)
                                                        : prevEpochDate;
                    bool ok = updateSetupAgentPricesFromPrevClose(rootNode, assetNameForLogs, prevDateForOpen, /*tickCents=*/1, /*spreadTicks=*/2);
                    if (ok) {
                        std::cout << "[SetupAgent][OpenFromPrevClose] asset=" << assetNameForLogs
                                  << " prevDate=" << prevDateForOpen
                                  << " -> bid/ask updated" << std::endl;
                    } else {
                        std::cout << "[SetupAgent][OpenFromPrevClose][Skip] asset=" << assetNameForLogs
                                  << " prevDate=" << prevDateForOpen
                                  << " (no TRADE found or log missing; keep XML bid/ask)" << std::endl;
                    }
                }

                auto mkTopo = buildMultiKernelCommunicationTopology(rootNode, discoveredKernelRanks);
                auto componentView = buildComponentCommunicationView(
                    mkTopo, rank, simRankGlobal, /*isKernel=*/true, /*isCross=*/false, baselineFullMesh);
                MPI_Comm componentComm = createGroupCommunicatorForMembers(componentView.members, rank);
                if (componentComm == MPI_COMM_NULL) {
                    std::cerr << "[FATAL][COMPONENT] rank=" << rank
                              << " role=kernel"
                              << " componentId=" << componentView.componentId
                              << " members=" << formatIntList(componentView.members)
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 93);
                }
                std::cout << "[TOPO][COMPONENT] rank=" << rank
                          << " role=kernel"
                          << " componentId=" << componentView.componentId
                          << " kernels=" << formatIntList(componentView.kernelRanks)
                          << " agents=" << formatIntList(componentView.agentRanks)
                          << " cross=" << formatIntList(componentView.crossRanks)
                          << " members=" << formatIntList(componentView.members)
                          << std::endl;

                auto __epoch_start = std::chrono::steady_clock::now();
                auto parameters = std::make_unique<ParameterStorage>();
                auto simulation = std::make_unique<DistributedSimulation>(parameters.get());
                simulation->configure(rootNode, "");
                simulation->getCommunication().setPerKernelCommunicator(perKernelComm);
                if (componentComm != MPI_COMM_NULL) simulation->getCommunication().setSimulationCommunicator(componentComm);
                if (!componentView.kernelRanks.empty()) {
                    simulation->getCommunication().setKernelTargetsList(componentView.kernelRanks);
                }
                std::vector<int> localKernelCrossRanks;
                if (auto itCross = componentView.crossRanksByKernel.find(simRankGlobal);
                    itCross != componentView.crossRanksByKernel.end()) {
                    localKernelCrossRanks = itCross->second;
                    std::sort(localKernelCrossRanks.begin(), localKernelCrossRanks.end());
                    localKernelCrossRanks.erase(std::unique(localKernelCrossRanks.begin(), localKernelCrossRanks.end()),
                                                localKernelCrossRanks.end());
                }
                if (!localKernelCrossRanks.empty()) {
                    simulation->getCommunication().setCrossAgentRanks(localKernelCrossRanks);
                }
                std::cout << "[TOPO][DIRECT_CROSS] rank=" << rank
                          << " role=kernel"
                          << " kernel=" << simRankGlobal
                          << " cross=" << formatIntList(localKernelCrossRanks)
                          << std::endl;

                // Configure + load true fundamental checkpoint for this epoch (kernel role).
                // IMPORTANT: do this AFTER simulation->configure(), because CppAgentBatch::configure()
                // configures the FundamentalValueModel and would otherwise wipe the loaded cache.
                //
                // Kernel only needs its own asset's cfg/state (its XML is correct for this asset).
                // Cross ranks will use fundamentalCfgsByAsset in the agent branch.
                {
                    auto cfg = fundamentalCfgFromRoot(rootNode);
                    FundamentalValueModel::instance().configure(cfg);
                    FundamentalValueModel::instance().configureForAsset(assetNameForLogs, cfg);
                    logFundamentalCfgOneLine("kernel_local_xml", rank, "kernel", assetNameForLogs, cfg);
                    if (cfg.enabled && cfg.checkpointEnabled) {
                        Timestamp startNs = computeEpochStartNsFromRoot(rootNode);
                        bool ok = FundamentalValueModel::instance().loadCheckpointForAsset(assetNameForLogs, startNs);
                        std::cout << "[FundamentalCkpt]"
                                  << " role=kernel"
                                  << " rank=" << rank
                                  << " asset=" << assetNameForLogs
                                  << " loadOk=" << (ok ? "true" : "false")
                                  << std::endl;
                    }
                }

                std::vector<int> neighborKernelRanks;
                {
                    neighborKernelRanks.clear();
                    if (baselineFullMesh) {
                        neighborKernelRanks = discoveredKernelRanks;
                        if (neighborKernelRanks.empty()) neighborKernelRanks.push_back(simRankGlobal);
                    } else {
                        neighborKernelRanks.push_back(simRankGlobal);
                    }
                    auto mk = rootNode.child("MultiKernel");
                    if (mk) {
                        std::vector<int> selfCross;
                        std::vector<std::pair<int, std::vector<int>>> others;
                        for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                            int kr = kn.attribute("rank").as_int(-1);
                            if (kr < 0) continue;
                            std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                            if (crossAgents.empty()) continue;
                            std::vector<int> lst;
                            std::stringstream ss(crossAgents); std::string it;
                            while (std::getline(ss, it, ',')) {
                                if (!it.empty()) {
                                    try { lst.push_back(std::stoi(it)); } catch(...) {}
                                }
                            }
                            if (lst.empty()) continue;
                            std::sort(lst.begin(), lst.end());
                            lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
                            if (kr == simRankGlobal) selfCross = lst; else others.emplace_back(kr, std::move(lst));
                        }
                        if (!selfCross.empty()) {
                            for (const auto& kv : others) {
                                int otherKr = kv.first;
                                const auto& otherCross = kv.second;
                                bool dependent = false;
                                for (int x : selfCross) {
                                    if (std::binary_search(otherCross.begin(), otherCross.end(), x)) { dependent = true; break; }
                                }
                                if (!baselineFullMesh && dependent) neighborKernelRanks.push_back(otherKr);
                            }
                            std::sort(neighborKernelRanks.begin(), neighborKernelRanks.end());
                            neighborKernelRanks.erase(std::unique(neighborKernelRanks.begin(), neighborKernelRanks.end()), neighborKernelRanks.end());
                        }
                    }
                }
                simulation->setKernelRanks(neighborKernelRanks);
                {
                    std::ostringstream oss;
                    oss << "[TOPO][KERNEL_NEIGHBORS] rank=" << simRankGlobal
                        << " neighbors={";
                    for (size_t i = 0; i < neighborKernelRanks.size(); ++i) {
                        oss << neighborKernelRanks[i];
                        if (i + 1 < neighborKernelRanks.size()) oss << ",";
                    }
                    oss << "}";
                    std::cout << oss.str() << std::endl;
                }

                bool enableInterKernelSync = false;
                if (componentView.kernelRanks.size() > 1) {
                    simulation->getCommunication().createKernelOnlyCommunicator(componentView.kernelRanks);
                    enableInterKernelSync = true;
                }
                simulation->setInterKernelSyncEnabled(enableInterKernelSync);

                std::string kernelLogDir = rankLogDirFor(rank);
                std::string commLogDirKernel = (fs::path(kernelLogDir) / "communication_logs").string();
                { std::error_code ec; std::filesystem::create_directories(commLogDirKernel, ec); }

                size_t rmaWindowMB = 0;
                auto commConfig2 = rootNode.child("CommunicationConfig");
                if (commConfig2) {
                    auto kernelSizeNode = commConfig2.child("KernelRMAWindowMB");
                    if (kernelSizeNode) {
                        auto v = kernelSizeNode.text().as_uint(0);
                        if (v > 0) rmaWindowMB = v;
                    }
                    unsigned int pollMicros = commConfig2.child("RMAPollIntervalMicros").text().as_uint(10);
                    simulation->setRMAPollIntervalMicros(pollMicros);
                    unsigned int putBackoffMax = commConfig2.child("RMAPutBackoffMicrosMax").text().as_uint(200);
                    simulation->getCommunication().setRMAPutBackoffMicrosMax(putBackoffMax);
                    bool enableRMAStats = commConfig2.child("EnableRMAStats").text().as_bool(false);
                    unsigned int rmaStatsFlushMs = commConfig2.child("RMAStatsFlushIntervalMs").text().as_uint(1000);
                    simulation->getCommunication().configureRMAStats(enableRMAStats, rmaStatsFlushMs, commLogDirKernel);
                    bool enableCommTimeStats = commConfig2.child("EnableCommunicationTimeStats").text().as_bool(false);
                    DesmarMpiApiProfiler::Configure(enableCommTimeStats, commLogDirKernel);
                    bool enableCpuStats = commConfig2.child("EnableCPULoadStats").text().as_bool(false);
                    bool enableMsgStats = commConfig2.child("EnableMessageStats").text().as_bool(false);
                    unsigned int rankStatsFlushMs = commConfig2.child("RankStatsFlushIntervalMs").text().as_uint(rmaStatsFlushMs);
                    simulation->configureRankStats(enableCpuStats, enableMsgStats, rankStatsFlushMs, commLogDirKernel);
                    simulation->setLogDir(commLogDirKernel);
                    unsigned int lbtsMicros = commConfig2.child("LBTSPollIntervalMicrosKernel").text().as_uint(100);
                    simulation->setLBTSPollIntervalMicros(lbtsMicros);
                }

                size_t bytes = static_cast<size_t>(rmaWindowMB) * 1024ull * 1024ull;
                auto crossRanksByKernel = componentView.crossRanksByKernel;

                // IMPORTANT (scalability + correctness):
                // For each kernel rank, the sender set MUST be exactly the per-kernel senders specified by MultiKernel:
                //   Kernel@agentRanks + Kernel@crossAgentRanks
                // No fallback to "all cross ranks" when crossAgentRanks is empty.
                std::vector<int> mergedAgents;
                if (auto itAgents = componentView.agentRanksByKernel.find(simRankGlobal);
                    itAgents != componentView.agentRanksByKernel.end()) {
                    mergedAgents.insert(mergedAgents.end(), itAgents->second.begin(), itAgents->second.end());
                }
                if (auto itCross = componentView.crossRanksByKernel.find(simRankGlobal);
                    itCross != componentView.crossRanksByKernel.end()) {
                    mergedAgents.insert(mergedAgents.end(), itCross->second.begin(), itCross->second.end());
                }
                if (mergedAgents.empty()) {
                    // Fallback to agentRanksGlobal only (no cross ranks), to avoid deadlock and keep topology sparse.
                    mergedAgents = agentRanksGlobal;
                }
                if (baselineFullMesh) {
                    // Optional legacy baseline: every kernel connects to all cross-agent ranks (full mesh).
                    for (int cr : componentView.crossRanks) mergedAgents.push_back(cr);
                }
                std::sort(mergedAgents.begin(), mergedAgents.end());
                mergedAgents.erase(std::unique(mergedAgents.begin(), mergedAgents.end()), mergedAgents.end());
                std::vector<int> broadcastTargets = mergedAgents;
                broadcastTargets.push_back(simRankGlobal);
                std::sort(broadcastTargets.begin(), broadcastTargets.end());
                broadcastTargets.erase(std::unique(broadcastTargets.begin(), broadcastTargets.end()), broadcastTargets.end());
                simulation->setBroadcastTargetRanks(broadcastTargets);
                simulation->enableRMAMode(bytes, simRankGlobal, mergedAgents);

                size_t agentWinMB = 0;
                if (auto comm = rootNode.child("CommunicationConfig")) {
                    if (auto agentSz = comm.child("AgentRMAWindowMB")) {
                        agentWinMB = agentSz.text().as_uint(0);
                    }
                    if (agentWinMB == 0) agentWinMB = 8;
                    if (comm.child("RMALockAll").text().as_bool(false)) {
                        simulation->getCommunication().enableRMALockAll();
                    }
                }
                simulation->setRemoteWindowLayout(0, agentWinMB * 1024ull * 1024ull);
                // In full-mesh mode, componentView.crossAgentSenders already expands each cross rank
                // to all kernels. Keep the topology explicit so kernel->cross RMA layout is valid.
                if (!componentView.crossAgentSenders.empty()) {
                    simulation->getCommunication().setCrossAgentWindowTopology(componentView.crossAgentSenders);
                }
                bool enableMainDoorbell = false;
                bool enableSyncDoorbell = true;
                unsigned int mainDoorbellSleepMicros = 1;
                unsigned int syncDoorbellSleepMicros = 1;
                if (auto comm = rootNode.child("CommunicationConfig")) {
                    const bool hasMainEnable = static_cast<bool>(comm.child("EnableMainMessageDoorbell"));
                    const bool hasMainSleep = static_cast<bool>(comm.child("MainMessageDoorbellShortSleepMicros"));
                    const bool hasSyncEnable = static_cast<bool>(comm.child("EnableSyncDoorbell"));
                    const bool hasSyncSleep = static_cast<bool>(comm.child("SyncDoorbellShortSleepMicros"));
                    const bool legacyEnableSyncDoorbell = comm.child("EnableAdaptiveLBTS").text().as_bool(true);
                    const unsigned int legacyDoorbellShortSleep = comm.child("DoorbellShortSleepMicrosKernel").text().as_uint(1);

                    enableMainDoorbell = hasMainEnable
                        ? comm.child("EnableMainMessageDoorbell").text().as_bool(false)
                        : false;
                    mainDoorbellSleepMicros = hasMainSleep
                        ? comm.child("MainMessageDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
                        : legacyDoorbellShortSleep;
                    enableSyncDoorbell = hasSyncEnable
                        ? comm.child("EnableSyncDoorbell").text().as_bool(legacyEnableSyncDoorbell)
                        : legacyEnableSyncDoorbell;
                    syncDoorbellSleepMicros = hasSyncSleep
                        ? comm.child("SyncDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
                        : legacyDoorbellShortSleep;
                    if (mainDoorbellSleepMicros == 0) mainDoorbellSleepMicros = 1;
                    if (syncDoorbellSleepMicros == 0) syncDoorbellSleepMicros = 1;
                }
                simulation->getCommunication().setMainDoorbellEnabled(enableMainDoorbell);
                simulation->getCommunication().setMainDoorbellShortSleepMicros(mainDoorbellSleepMicros);
                simulation->getCommunication().setSyncDoorbellEnabled(enableSyncDoorbell);
                std::cout << "[DOORBELL][MODE] rank=" << rank
                          << " mainEnabled=" << (enableMainDoorbell ? "true" : "false")
                          << " mainSleepMicros=" << mainDoorbellSleepMicros
                          << " syncEnabled=" << (enableSyncDoorbell ? "true" : "false")
                          << " syncSleepMicros=" << syncDoorbellSleepMicros
                          << " anyDoorbell=" << (simulation->getCommunication().isAnyDoorbellEnabled() ? "enabled" : "disabled")
                          << std::endl;
                simulation->startCommunicationWorkers();

                simulation->start();
                if (DesmarMpiApiProfiler::Enabled()) {
                    DesmarMpiApiProfiler::StartWindow("Unknown");
                }
                simulation->runToCompletion();
                if (DesmarMpiApiProfiler::Enabled()) {
                    DesmarMpiApiProfiler::StopAndDump();
                }

                simulation->stop();
                // Save per-asset fundamental checkpoint at end-of-day (kernel ranks only).
                // Use the kernel's final simulation time as close timestamp.
                saveFundamentalModelForEpoch(rootNode, assetNameForLogs, simulation->currentTimestamp());
                std::string alignedStatsFile = commLogDirKernel + "/Aligned_time_stats_rank" + std::to_string(rank) + ".txt";
                simulation->dumpTimeAlignmentStats(alignedStatsFile);

                simulation->shutdownThreads();

                auto __epoch_end = std::chrono::steady_clock::now();
                auto __epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__epoch_end - __epoch_start).count();
                std::cout << "[Epoch] kernel rank=" << rank
                          << " epoch=" << epochIdx
                          << " date=" << epochDate
                          << " wall_ms=" << __epoch_ms << std::endl;

                prevEpochDate = epochDate;

                // ===== Cross-epoch coordinator (fixed pipeline) =====
                // 1) barrier: ensure all ranks finished the day and flushed outputs
                // DESMAR_MPI_MODE=proxy: stop the single MPI progress thread before entering MPI_Barrier
                // on the main thread (keeps the "single MPI thread while running" contract).
                // quiesce() now also drains any in-flight Iallreduce session so the epoch-local
                // component communicator can be freed safely after object destruction.
                {
                    const char* v = std::getenv("DESMAR_MPI_MODE");
                    bool proxy = (v && (*v=='p' || *v=='P')); // "proxy" fast check
                    if (proxy) {
                        std::cout << "[DESMAR_MPI_PROXY][QUIESCE] rank=" << rank
                                  << " before epoch barrier: epoch_end_all_ranks_finished" << std::endl;
                        simulation->getCommunication().quiesce();
                    }
                }
                tracedBarrier(epochComm, "epoch_end_all_ranks_finished", epochIdx, epochDate);

                // IMPORTANT:
                // MPI windows are freed in MPICommunicationManager::~MPICommunicationManager() via shutdown().
                // Epoch-local Iallreduce requests are also drained there before componentComm is freed.
                // Win_free/unlock_all are collective; destroying per-epoch objects BEFORE the global barrier
                // can make fast ranks appear "hung" waiting for slow ranks (or deadlock if some ranks never reach cleanup).
                // Therefore, delay destruction until AFTER the barrier.
                simulation.reset();
                parameters.reset();
                if (componentComm != MPI_COMM_NULL) {
                    MPI_Comm_free(&componentComm);
                }

                // 2) rank0 merges DataFactory outputs to _merged (OHLCV+LOB) and deletes per-agent dirs
                if (rank == 0) {
                    mergeDedupDataFactoryToMergedDir(epochDate);
                    // Snapshot per-rank communication logs for this epoch (cpu/msg/rma/comm_time/LBTS, etc.).
                    // This avoids changing the existing log output paths while preserving per-epoch stats for experiments.
                    archiveCommunicationLogsForEpoch(logRoot, epochDate);
                }

                // 3) barrier: merged ready for all ranks
                tracedBarrier(epochComm, "merged_ready", epochIdx, epochDate);

                // 4) cross ranks evaluate baskets offline using merged day data + checkpoint, write BasketFinal/<date>/rank*.jsonl
                {
                    std::unordered_map<std::string,int> asset2kernel;
                    std::vector<std::string> allAssets;
                    (void)parseMultiKernelTargets(rootNode, asset2kernel, allAssets);
                    auto assign = readJsonFile(assignmentFilePath(epochDate));
                    if (std::find(crossAgentRanksCfg.begin(), crossAgentRanksCfg.end(), rank) != crossAgentRanksCfg.end()) {
                        evaluateBasketsForCrossRankAndWriteJsonl(
                            rank, epochDate, allAssets, assign,
                            basketCapN, assetEvalN,
                            daySeed, static_cast<uint64_t>(epochIdx)
                        );
                    }
                }

                // 5) barrier: baskets ready
                tracedBarrier(epochComm, "baskets_ready", epochIdx, epochDate);

                // 6) rank0 runs METIS on coarsened groups and writes assignment_<nextDate>.json
                if (rank == 0) {
                    const std::string nextDate = nextTradingDateSkipWeekend(epochDate);
                    // When no cross ranks are configured, skip ALL cross-epoch topology updates.
                    // This allows multi-epoch runs to proceed cleanly in "no-cross" experiments.
                    if (!crossAgentRanksCfg.empty()) {
                        if (!baselineNoMigration) {
                            buildAndWriteNextAssignmentViaMtKaHyPar(epochDate, nextDate, rootNode, crossAgentRanksCfg, worldRankToNodeId);
                        } else if (baselineTopoFromBaskets) {
                            buildAndWriteNextKernelCrossMapNoMigration(epochDate, nextDate, rootNode, crossAgentRanksCfg, worldRankToNodeId);
                        }
                    }
                }

                // 7) barrier: next-day assignment ready before next epoch starts
                tracedBarrier(epochComm, "next_assignment_ready", epochIdx, epochDate);

                // 8) apply next-day kernel->cross mapping into in-memory XML for the upcoming epoch (all ranks)
                if (!crossAgentRanksCfg.empty() &&
                    (((!baselineNoMigration) || (baselineNoMigration && baselineTopoFromBaskets)) && epochIdx + 1 < epochCount)) {
                    const std::string nextDate = nextTradingDateSkipWeekend(epochDate);
                    auto k2r = readKernelCrossMapForDate(nextDate);
                    applyKernelCrossMapToMultiKernel(rootNode, k2r);
                }
            }

            std::cout << "Kernel completed all epochs, finalizing..." << std::endl;
            tracedBarrier(epochComm, "kernel_finalizing", /*epochIdx*/epochCount-1, prevEpochDate.empty()? baseDate : prevEpochDate);
            {
                auto __proc_end = std::chrono::steady_clock::now();
                auto __proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__proc_end - __kernel_proc_start).count();
                std::cout << "[Perf] Process total wall time (kernel role): "
                          << __proc_ms << " ms (" << (__proc_ms / 1000.0) << " s)" << std::endl;
            }
            MPI_Finalize();
            return 0;
            
        } else if (isAgent) {
            std::cout << "Starting agent router on rank " << rank << " (epochs=" << epochCount << ")" << std::endl;
            const bool isCross = (std::find(crossAgentRanksCfg.begin(), crossAgentRanksCfg.end(), rank) != crossAgentRanksCfg.end());

            std::string curDate = baseDate;
            // Ensure base-date assignment exists (used by cross-rank offline evaluation).
            if (rank == 0) {
                const auto p = assignmentFilePath(baseDate);
                // IMPORTANT: do not overwrite an existing assignment file (supports one-epoch-per-process runs).
                if (!std::filesystem::exists(p)) {
                    auto initAssign = buildInitialAssignmentFromConfig(rootNode, baseDate);
                    (void)writeJsonAtomic(p, initAssign);
                }
            }
            tracedBarrier(epochComm, "agent_init_assignment_ready", /*epochIdx*/-1, baseDate);

            for (int epochLocal = 0; epochLocal < epochCount; ++epochLocal) {
                char dateBuf[9] = {0};
                uint64_t daySeed = 0;
                if (rank == 0) {
                    if (epochLocal == 0) curDate = baseDate;
                    else curDate = nextTradingDateSkipWeekend(curDate);
                    daySeed = deriveDaySeed(masterSeed, curDate);
                    std::snprintf(dateBuf, sizeof(dateBuf), "%s", curDate.c_str());
                }
                MPI_Bcast(dateBuf, 9, MPI_CHAR, 0, epochComm);
                MPI_Bcast(&daySeed, 1, MPI_UNSIGNED_LONG_LONG, 0, epochComm);
                std::string epochDate(dateBuf);
                if (epochDate.size() != 8) epochDate = baseDate;
                const int epochIdx = tradingDayIndexFromStart(experimentBaseDate, epochDate);

                // Overwrite stdout/stderr per epoch to avoid unbounded log growth.
                // Cross ranks are under <logRoot>/CROSS/<rank>/stdout|stderr; normal agents under <logRoot>/<asset>/<rank>/...
                overwriteStdoutStderrPerEpoch(logRoot, isCross ? std::string("CROSS") : assetNameForLogs, rank);

                if (!rootNode.attribute("date").empty()) rootNode.attribute("date").set_value(epochDate.c_str());
                else rootNode.append_attribute("date").set_value(epochDate.c_str());
                // One-epoch-per-process support: apply kernel_cross_<date>.json (if present) for THIS epoch.
                if (!baselineFullMesh && !crossAgentRanksCfg.empty()) {
                    auto k2r_now = readKernelCrossMapForDate(epochDate);
                    if (epochDate != baseDate && k2r_now.empty()) {
                        throw std::runtime_error("[Topology][KernelCross][FATAL] missing kernel_cross map for date=" + epochDate);
                    }
                    if (!k2r_now.empty()) {
                        applyKernelCrossMapToMultiKernel(rootNode, k2r_now);
                    }
                }
                {
                    auto ga = rootNode.child("GlobalAgentConfig");
                    if (ga) {
                        // Do NOT override GlobalSeed per epoch. GlobalSeed is intended to be a single
                        // global seed shared across all ranks (including cross ranks) so that shared
                        // modules (e.g., FundamentalValueModel) remain consistent.
                        const std::string tag = isCross ? std::string("CROSS") : assetNameForLogs;
                        const uint64_t assetSeed = deriveAssetDaySeed(daySeed, tag);
                        pugi::xml_node as = ga.child("AssetSeed");
                        if (!as) as = ga.append_child("AssetSeed");
                        auto a = as.attribute("value");
                        if (a.empty()) a = as.append_attribute("value");
                        a.set_value(std::to_string(assetSeed).c_str());

                    }
                }

                std::cout << "[Epoch] rank=" << rank
                          << " epoch=" << epochIdx
                          << " date=" << epochDate
                          << " daySeed=" << daySeed
                          << std::endl;

                // Baseline: ensure per-day assignment_<date>.json exists and stays stable (no migration),
                // so basket evaluation can still find which SPT agents belong to which cross rank.
                if (baseline && rank == 0) {
                    const auto p = assignmentFilePath(epochDate);
                    if (!std::filesystem::exists(p)) {
                        auto assign = buildInitialAssignmentFromConfig(rootNode, epochDate);
                        (void)writeJsonAtomic(p, assign);
                    }
                }
                tracedBarrier(epochComm, "agent_epoch_assignment_ready", epochIdx, epochDate);

                // ===== Per-asset FundamentalValueModel config sync (MPMD) =====
                // All ranks in epochComm must call this collective in the same order.
                // Kernel ranks contribute (assetName -> cfg) from their own XML; non-kernel ranks contribute empty.
                auto fundamentalCfgsByAsset =
                    allgatherFundamentalCfgsFromKernels(epochComm, /*isKernel=*/false, assetNameForLogs, rootNode);

                auto mkTopo = buildMultiKernelCommunicationTopology(rootNode, discoveredKernelRanks);
                auto componentView = buildComponentCommunicationView(
                    mkTopo, rank, simRankGlobal, /*isKernel=*/false, /*isCross=*/isCross, baselineFullMesh);
                MPI_Comm componentComm = createGroupCommunicatorForMembers(componentView.members, rank);
                if (componentComm == MPI_COMM_NULL) {
                    std::cerr << "[FATAL][COMPONENT] rank=" << rank
                              << " role=" << (isCross ? "cross" : "agent")
                              << " componentId=" << componentView.componentId
                              << " members=" << formatIntList(componentView.members)
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 94);
                }
                std::cout << "[TOPO][COMPONENT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent")
                          << " componentId=" << componentView.componentId
                          << " kernels=" << formatIntList(componentView.kernelRanks)
                          << " agents=" << formatIntList(componentView.agentRanks)
                          << " cross=" << formatIntList(componentView.crossRanks)
                          << " members=" << formatIntList(componentView.members)
                          << std::endl;

                auto parameters = std::make_unique<ParameterStorage>();
                std::string routerName = "ROUTER_RANK_" + std::to_string(rank);
                std::unique_ptr<AgentRankRouter> router;
                if (isCross) router = std::make_unique<CrossAgentRankRouter>(rank, routerName);
                else router = std::make_unique<AgentRankRouter>(rank, routerName);
                router->setSimulationRank(simRankGlobal);
                if (componentComm != MPI_COMM_NULL) router->getCommunication().setSimulationCommunicator(componentComm);
                std::cout << "[AGENT_BOOT][INIT_ENTER] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                if (!router->initialize()) {
                    std::cerr << "Failed to initialize router on rank " << rank << std::endl;
                    MPI_Finalize();
                    return 1;
                }
                std::cout << "[AGENT_BOOT][INIT_EXIT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;

                auto proxySimulation = std::make_unique<ProxySimulation>(parameters.get(), router.get());
                pugi::xml_node agentRankNode;
                for (auto node : rootNode.children("AgentRank")) {
                    if (node.attribute("rank").as_int() == rank) { agentRankNode = node; break; }
                }
                if (agentRankNode.empty()) {
                    for (auto node : rootNode.children("CrossAgentRank")) {
                        if (node.attribute("rank").as_int() == rank) { agentRankNode = node; break; }
                    }
                }
                if (!agentRankNode.empty()) {
                    auto batch = std::make_unique<CppAgentBatch>(proxySimulation.get());
                    batch->configure(agentRankNode, "");
                    router->configureDelayFromConfig(agentRankNode, rootNode);

                    // Configure + load true fundamental checkpoints for this epoch (all ranks).
                    // IMPORTANT: do this AFTER batch->configure(), because CppAgentBatch::configure()
                    // configures the FundamentalValueModel and would otherwise wipe the loaded cache.
                    //
                    // MPMD multi-asset:
                    // - Normal agent ranks: use their own XML (asset-specific) for this asset.
                    // - Cross ranks: started with first_cfg, so MUST collect per-asset cfgs from all kernels.
                    Timestamp startNs = computeEpochStartNsFromRoot(rootNode);
                    if (isCross) {
                        // Always configure per-asset cfgs for cross (even if checkpointing is off).
                        for (const auto& kv : fundamentalCfgsByAsset) {
                            FundamentalValueModel::instance().configureForAsset(kv.first, kv.second);
                        }

                        // Print once per epoch (only the smallest cross rank) for quick verification.
                        int minCross = rank;
                        if (!crossAgentRanksCfg.empty()) {
                            minCross = *std::min_element(crossAgentRanksCfg.begin(), crossAgentRanksCfg.end());
                        }
                        if (rank == minCross) {
                            std::cout << "[FundamentalSync]"
                                      << " role=cross"
                                      << " rank=" << rank
                                      << " localXmlAssetNameForLogs=" << assetNameForLogs
                                      << " gatheredAssets=" << fundamentalCfgsByAsset.size()
                                      << std::endl;
                            for (const auto& kv : fundamentalCfgsByAsset) {
                                logFundamentalCfgOneLine("gathered_from_kernels", rank, "cross", kv.first, kv.second);
                            }
                        }

                        // Load checkpoints for ALL assets (so r*(t) continues across days).
                        // If cfgMap is empty, fall back to previous behavior (best effort).
                        if (!fundamentalCfgsByAsset.empty()) {
                            for (const auto& kv : fundamentalCfgsByAsset) {
                                bool ok = FundamentalValueModel::instance().loadCheckpointForAsset(kv.first, startNs);
                                if (rank == minCross) {
                                    std::cout << "[FundamentalCkpt]"
                                              << " role=cross"
                                              << " rank=" << rank
                                              << " asset=" << kv.first
                                              << " loadOk=" << (ok ? "true" : "false")
                                              << std::endl;
                                }
                            }
                        } else {
                            std::unordered_map<std::string,int> asset2kernel;
                            std::vector<std::string> allAssets;
                            (void)parseMultiKernelTargets(rootNode, asset2kernel, allAssets);
                            configureAndLoadFundamentalModelForEpoch(rootNode, allAssets);
                        }
                    } else {
                        // Normal agent: configure per-asset cfg from its own XML, then load its asset checkpoint.
                        auto cfg = fundamentalCfgFromRoot(rootNode);
                        FundamentalValueModel::instance().configure(cfg);
                        FundamentalValueModel::instance().configureForAsset(assetNameForLogs, cfg);
                        logFundamentalCfgOneLine("agent_local_xml", rank, "agent", assetNameForLogs, cfg);
                        bool ok = FundamentalValueModel::instance().loadCheckpointForAsset(assetNameForLogs, startNs);
                        std::cout << "[FundamentalCkpt]"
                                  << " role=agent"
                                  << " rank=" << rank
                                  << " asset=" << assetNameForLogs
                                  << " loadOk=" << (ok ? "true" : "false")
                                  << std::endl;
                    }

                    if (isCross) {
                        size_t expMin = agentRankNode.attribute("expBatchMin").as_uint(8);
                        unsigned int expTimeoutMs = agentRankNode.attribute("expBatchTimeoutMs").as_uint(10);
                        if (auto expNode = agentRankNode.child("ExperienceBatchConfig")) {
                            auto a = expNode.attribute("expBatchMin"); if (!a.empty()) expMin = a.as_uint(expMin);
                            auto b = expNode.attribute("expBatchTimeoutMs"); if (!b.empty()) expTimeoutMs = b.as_uint(expTimeoutMs);
                        }
                        auto* cr = static_cast<CrossAgentRankRouter*>(router.get());
                        cr->setExpBatchMin(expMin);
                        cr->setExpBatchTimeoutMs(expTimeoutMs);
                    }
                }

                std::string agentLogDir;
                if (isCross) {
                    fs::path p = logRoot;
                    p /= "CROSS";
                    p /= std::to_string(rank);
                    std::error_code ec;
                    fs::create_directories(p, ec);
                    agentLogDir = p.string();
                } else {
                    agentLogDir = rankLogDirFor(rank);
                }
                std::string commLogDirAgent = (fs::path(agentLogDir) / "communication_logs").string();
                { std::error_code ec; std::filesystem::create_directories(commLogDirAgent, ec); }

                size_t rmaWindowMB = 0;
                auto commConfig2 = rootNode.child("CommunicationConfig");
                if (commConfig2) {
                    auto agentSizeNode = commConfig2.child("AgentRMAWindowMB");
                    if (agentSizeNode) {
                        auto v = agentSizeNode.text().as_uint(0);
                        if (v > 0) rmaWindowMB = v;
                    }
                    unsigned int pollMicros = commConfig2.child("RMAPollIntervalMicros").text().as_uint(10);
                    router->setRMAPollIntervalMicros(pollMicros);
                    unsigned int putBackoffMax = commConfig2.child("RMAPutBackoffMicrosMax").text().as_uint(200);
                    router->getCommunication().setRMAPutBackoffMicrosMax(putBackoffMax);
                    bool enableRMAStats = commConfig2.child("EnableRMAStats").text().as_bool(false);
                    unsigned int rmaStatsFlushMs = commConfig2.child("RMAStatsFlushIntervalMs").text().as_uint(1000);
                    router->getCommunication().configureRMAStats(enableRMAStats, rmaStatsFlushMs, commLogDirAgent);
                    bool enableCommTimeStats = commConfig2.child("EnableCommunicationTimeStats").text().as_bool(false);
                    DesmarMpiApiProfiler::Configure(enableCommTimeStats, commLogDirAgent);
                    router->setLogDir(commLogDirAgent);
                    bool enableCpuStats = commConfig2.child("EnableCPULoadStats").text().as_bool(false);
                    bool enableMsgStats = commConfig2.child("EnableMessageStats").text().as_bool(false);
                    unsigned int rankStatsFlushMs = commConfig2.child("RankStatsFlushIntervalMs").text().as_uint(rmaStatsFlushMs);
                    router->configureRankStats(enableCpuStats, enableMsgStats, rankStatsFlushMs, commLogDirAgent);
                    unsigned int lbtsMicrosAgent = commConfig2.child("LBTSPollIntervalMicrosAgent").text().as_uint(100);
                    router->setLBTSPollIntervalMicros(lbtsMicrosAgent);
                    bool enableAgentTime = commConfig2.child("EnableLBTSLogAgent").text().as_bool(false);
                    unsigned int everyItersAgent = commConfig2.child("LBTSLogEveryItersAgent").text().as_uint(1000);
                    router->configureLBTSLogging(enableAgentTime, everyItersAgent);
                    bool enableAlignedStatsAgent = commConfig2.child("EnableAlignedMessageStatsAgent").text().as_bool(false);
                    router->setAlignedStatsEnabled(enableAlignedStatsAgent);
                }

                size_t bytes = static_cast<size_t>(rmaWindowMB) * 1024ull * 1024ull;
                if (isCross) {
                    std::unordered_map<std::string,int> asset2kernel;
                    std::vector<int> kernelRanksMulti;
                    std::unordered_map<int, std::vector<int>> ranksByKernel;
                    std::unordered_map<int, size_t> remoteWinByKernel;
                    std::unordered_map<int, std::vector<int>> crossRanksByKernel;
                    if (auto mk = rootNode.child("MultiKernel")) {
                        if (auto attr = mk.attribute("targets"); attr && *attr.value()) {
                            std::string s = attr.as_string(); size_t start = 0;
                            while (start < s.size()) {
                                size_t sep = s.find(';', start);
                                std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
                                size_t colon = item.find(':');
                                if (colon != std::string::npos) {
                                    int kr = std::stoi(item.substr(0, colon));
                                    std::string asset = item.substr(colon+1);
                                    if (!asset.empty()) { asset2kernel[asset] = kr; }
                                }
                                if (sep==std::string::npos) break;
                                start = sep + 1;
                            }
                        }
                        std::set<int> myKernels;
                        for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                            int kr = kn.attribute("rank").as_int(-1);
                            std::string agents = kn.attribute("agentRanks").as_string();
                            if (kr>=0 && !agents.empty()) {
                                std::vector<int> tmp; std::stringstream ss(agents); std::string it;
                                while (std::getline(ss, it, ',')) { if (!it.empty()) tmp.push_back(std::stoi(it)); }
                                if (!tmp.empty()) ranksByKernel[kr] = tmp;
                            }
                            std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                            if (kr>=0 && !crossAgents.empty()) {
                                std::vector<int> tmpCross; std::stringstream ssCross(crossAgents); std::string itCross;
                                while (std::getline(ssCross, itCross, ',')) {
                                    if (!itCross.empty()) {
                                        int crossRank = std::stoi(itCross);
                                        tmpCross.push_back(crossRank);
                                        if (crossRank == rank) myKernels.insert(kr);
                                    }
                                }
                                if (!tmpCross.empty()) crossRanksByKernel[kr] = tmpCross;
                            }
                            size_t rmb = kn.attribute("kernelWindowMB").as_uint(0);
                            if (kr>=0 && rmb>0) remoteWinByKernel[kr] = rmb * 1024ull * 1024ull;
                        }
                        kernelRanksMulti.assign(myKernels.begin(), myKernels.end());
                    }
                    if (baselineFullMesh) {
                        // Optional legacy baseline: cross ranks connect to ALL kernels and every kernel accepts ALL cross ranks.
                        kernelRanksMulti = componentView.kernelRanks;
                        std::sort(kernelRanksMulti.begin(), kernelRanksMulti.end());
                        kernelRanksMulti.erase(std::unique(kernelRanksMulti.begin(), kernelRanksMulti.end()), kernelRanksMulti.end());
                        crossRanksByKernel.clear();
                        for (int kr : kernelRanksMulti) {
                            crossRanksByKernel[kr] = componentView.crossRanks;
                        }
                    }
                    for (auto it = crossRanksByKernel.begin(); it != crossRanksByKernel.end(); ) {
                        if (std::find(componentView.kernelRanks.begin(), componentView.kernelRanks.end(), it->first) == componentView.kernelRanks.end()) {
                            it = crossRanksByKernel.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    for (auto it = remoteWinByKernel.begin(); it != remoteWinByKernel.end(); ) {
                        if (std::find(componentView.kernelRanks.begin(), componentView.kernelRanks.end(), it->first) == componentView.kernelRanks.end()) {
                            it = remoteWinByKernel.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    std::vector<int> directCrossRanks;
                    for (int kr : kernelRanksMulti) {
                        auto it = crossRanksByKernel.find(kr);
                        if (it == crossRanksByKernel.end()) continue;
                        directCrossRanks.insert(directCrossRanks.end(), it->second.begin(), it->second.end());
                    }
                    std::sort(directCrossRanks.begin(), directCrossRanks.end());
                    directCrossRanks.erase(std::unique(directCrossRanks.begin(), directCrossRanks.end()), directCrossRanks.end());

                    if (!kernelRanksMulti.empty()) router->getCommunication().setKernelTargetsList(kernelRanksMulti);
                    if (!directCrossRanks.empty()) router->getCommunication().setCrossAgentRanks(directCrossRanks);
                    std::cout << "[TOPO][DIRECT_CROSS] rank=" << rank
                              << " role=" << (isCross ? "cross" : "agent")
                              << " kernels=" << formatIntList(kernelRanksMulti)
                              << " cross=" << formatIntList(directCrossRanks)
                              << std::endl;
                    if (!componentView.crossAgentSenders.empty()) {
                        router->getCommunication().setCrossAgentWindowTopology(componentView.crossAgentSenders);
                    }
                    std::vector<int> mergedAgentsB;
                    for (int kr : kernelRanksMulti) {
                        auto it = ranksByKernel.find(kr);
                        if (it == ranksByKernel.end()) continue;
                        mergedAgentsB.insert(mergedAgentsB.end(), it->second.begin(), it->second.end());
                    }
                    for (int cr : directCrossRanks) mergedAgentsB.push_back(cr);
                    std::sort(mergedAgentsB.begin(), mergedAgentsB.end());
                    mergedAgentsB.erase(std::unique(mergedAgentsB.begin(), mergedAgentsB.end()), mergedAgentsB.end());
                    router->enableRMAMode(bytes, simRankGlobal, mergedAgentsB);
                    if (!asset2kernel.empty()) {
                        auto* cr = static_cast<CrossAgentRankRouter*>(router.get());
                        std::sort(kernelRanksMulti.begin(), kernelRanksMulti.end());
                        kernelRanksMulti.erase(std::unique(kernelRanksMulti.begin(), kernelRanksMulti.end()), kernelRanksMulti.end());
                        cr->setAssetKernelMapping(asset2kernel);
                        cr->setKernelRanks(kernelRanksMulti);
                        std::unordered_map<int, std::vector<int>> mergedRanksByKernel = ranksByKernel;
                        for (auto it = mergedRanksByKernel.begin(); it != mergedRanksByKernel.end(); ) {
                            if (std::find(componentView.kernelRanks.begin(), componentView.kernelRanks.end(), it->first) == componentView.kernelRanks.end()) {
                                it = mergedRanksByKernel.erase(it);
                            } else {
                                ++it;
                            }
                        }
                        for (const auto& kv : crossRanksByKernel) {
                            int kr = kv.first;
                            const auto& crossRanks = kv.second;
                            auto& agentList = mergedRanksByKernel[kr];
                            for (int cr2 : crossRanks) agentList.push_back(cr2);
                            std::sort(agentList.begin(), agentList.end());
                            agentList.erase(std::unique(agentList.begin(), agentList.end()), agentList.end());
                        }
                        if (!mergedRanksByKernel.empty()) router->getCommunication().enableRMAModeMultiTopologies(kernelRanksMulti, mergedRanksByKernel, bytes);
                        if (!remoteWinByKernel.empty()) router->getCommunication().setRemoteWindowLayoutForKernels(remoteWinByKernel);
                        if (!crossRanksByKernel.empty()) router->getCommunication().setCrossAgentRanksByKernel(crossRanksByKernel);
                    }
                } else {
                    std::vector<int> myKernelCrossRanks;
                    if (auto mk = rootNode.child("MultiKernel")) {
                        for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                            int kr = kn.attribute("rank").as_int(-1);
                            if (kr == simRankGlobal) {
                                std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                                if (!crossAgents.empty()) {
                                    std::stringstream ssCross(crossAgents); std::string itCross;
                                    while (std::getline(ssCross, itCross, ',')) { if (!itCross.empty()) myKernelCrossRanks.push_back(std::stoi(itCross)); }
                                }
                                break;
                            }
                        }
                    }
                    if (baselineFullMesh) {
                        // Optional legacy baseline: all kernels talk to all cross ranks (must match kernel-side RMA slice layout).
                        myKernelCrossRanks = componentView.crossRanks;
                    }
                    // IMPORTANT: do not fallback to "all cross ranks" when the kernel's crossAgentRanks is empty.
                    std::vector<int> mergedAgentsA = agentRanksGlobal;
                    for (int cr : myKernelCrossRanks) mergedAgentsA.push_back(cr);
                    std::sort(mergedAgentsA.begin(), mergedAgentsA.end());
                    mergedAgentsA.erase(std::unique(mergedAgentsA.begin(), mergedAgentsA.end()), mergedAgentsA.end());
                    router->enableRMAMode(bytes, simRankGlobal, mergedAgentsA);
                }
                size_t kernelWinMB = 0;
                if (auto comm = rootNode.child("CommunicationConfig")) {
                    if (auto kerSz = comm.child("KernelRMAWindowMB")) {
                        kernelWinMB = kerSz.text().as_uint(0);
                    }
                    if (kernelWinMB == 0) kernelWinMB = 64;
                    if (comm.child("RMALockAll").text().as_bool(false)) {
                        router->getCommunication().enableRMALockAll();
                    }
                }
                router->setRemoteWindowLayout(kernelWinMB * 1024ull * 1024ull, 0);
                bool enableMainDoorbell = false;
                bool enableSyncDoorbell = true;
                unsigned int mainDoorbellSleepMicros = 1;
                unsigned int syncDoorbellSleepMicros = 1;
                if (auto comm = rootNode.child("CommunicationConfig")) {
                    const bool hasMainEnable = static_cast<bool>(comm.child("EnableMainMessageDoorbell"));
                    const bool hasMainSleep = static_cast<bool>(comm.child("MainMessageDoorbellShortSleepMicros"));
                    const bool hasSyncEnable = static_cast<bool>(comm.child("EnableSyncDoorbell"));
                    const bool hasSyncSleep = static_cast<bool>(comm.child("SyncDoorbellShortSleepMicros"));
                    const bool legacyEnableSyncDoorbell = comm.child("EnableAdaptiveLBTS").text().as_bool(true);
                    const unsigned int legacyDoorbellShortSleep = comm.child("DoorbellShortSleepMicrosKernel").text().as_uint(1);

                    enableMainDoorbell = hasMainEnable
                        ? comm.child("EnableMainMessageDoorbell").text().as_bool(false)
                        : false;
                    mainDoorbellSleepMicros = hasMainSleep
                        ? comm.child("MainMessageDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
                        : legacyDoorbellShortSleep;
                    enableSyncDoorbell = hasSyncEnable
                        ? comm.child("EnableSyncDoorbell").text().as_bool(legacyEnableSyncDoorbell)
                        : legacyEnableSyncDoorbell;
                    syncDoorbellSleepMicros = hasSyncSleep
                        ? comm.child("SyncDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
                        : legacyDoorbellShortSleep;
                    if (mainDoorbellSleepMicros == 0) mainDoorbellSleepMicros = 1;
                    if (syncDoorbellSleepMicros == 0) syncDoorbellSleepMicros = 1;
                }
                router->getCommunication().setMainDoorbellEnabled(enableMainDoorbell);
                router->getCommunication().setMainDoorbellShortSleepMicros(mainDoorbellSleepMicros);
                router->getCommunication().setSyncDoorbellEnabled(enableSyncDoorbell);
                std::cout << "[DOORBELL][MODE] rank=" << rank
                          << " mainEnabled=" << (enableMainDoorbell ? "true" : "false")
                          << " mainSleepMicros=" << mainDoorbellSleepMicros
                          << " syncEnabled=" << (enableSyncDoorbell ? "true" : "false")
                          << " syncSleepMicros=" << syncDoorbellSleepMicros
                          << " anyDoorbell=" << (router->getCommunication().isAnyDoorbellEnabled() ? "enabled" : "disabled")
                          << std::endl;
                // IMPORTANT:
                // - In DESMAR_MPI_MODE=proxy, communicator setup MUST happen before starting MPI worker threads,
                //   otherwise a non-MPI thread would call MPI_Comm_group/translate while the proxy MPI thread is running.
                // - In DESMAR_MPI_MODE=multiple, keep legacy order to avoid behavior differences.
                {
                    const char* v = std::getenv("DESMAR_MPI_MODE");
                    bool proxy = (v && (*v=='p' || *v=='P')); // "proxy" fast check
                    if (proxy) {
                        router->getCommunication().setPerKernelCommunicator(perKernelComm);
                        router->getCommunication().startWorkers();
                    } else {
                router->getCommunication().startWorkers();
                router->getCommunication().setPerKernelCommunicator(perKernelComm);
                    }
                }

                Timestamp startNs = 0;
                {
                    pugi::xml_attribute att;
                    std::string date = rootNode.attribute("date").as_string();
                    if (!(att = rootNode.attribute("start")).empty()) {
                        std::string startStr = att.as_string();
                        if (!date.empty()) startNs = DateTimeConverter::dateTimeToNs(date, startStr);
                        else if (std::all_of(startStr.begin(), startStr.end(), ::isdigit)) startNs = DateTimeConverter::marketTimeToNs(std::stoull(startStr));
                        else if (startStr.find(':') != std::string::npos) startNs = DateTimeConverter::timeStringToNs(startStr);
                        else startNs = static_cast<Timestamp>(att.as_ullong());
                    }
                }
                if (startNs > 0) router->setInitialLVT(startNs);

                // Inject epoch date to agents BEFORE preload, so agents can reliably load per-day history/checkpoints.
                router->setEpochDateForAgents(epochDate);

                // Preload phase (BEFORE READY): allow agents to do heavy I/O/parse outside START critical path.
                // This is especially important for cross ranks with many local agents (e.g., SPT checkpoints).
                router->preloadLocalAgents();

                std::cout << "[AGENT_BOOT][START_ENTER] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                router->start();
                std::cout << "[AGENT_BOOT][START_EXIT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                std::cout << "[AGENT_BOOT][READY_ENTER] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                router->sendReadySignalToKernel();
                std::cout << "[AGENT_BOOT][READY_EXIT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                std::cout << "[AGENT_BOOT][BARRIER_ENTER] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                router->getCommunication().barrierPerKernel();
                std::cout << "[AGENT_BOOT][BARRIER_EXIT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                // Run inbound message processing on the process main thread (no idle while+sleep loop).
                std::cout << "[AGENT_BOOT][INCOMING_LOOP_ENTER] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                router->runIncomingLoopOnThisThread();
                std::cout << "[AGENT_BOOT][INCOMING_LOOP_EXIT] rank=" << rank
                          << " role=" << (isCross ? "cross" : "agent") << std::endl;
                router->stop();

                std::string alignedStatsFileAgent = commLogDirAgent + "/Aligned_time_stats_rank" + std::to_string(rank) + ".txt";
                router->dumpTimeAlignmentStats(alignedStatsFileAgent);

                // ===== Cross-epoch coordinator (fixed pipeline) =====
                tracedBarrier(epochComm, "agent_epoch_end_all_ranks_finished", epochIdx, epochDate);

                // IMPORTANT:
                // Router destruction triggers MPICommunicationManager::~MPICommunicationManager() -> shutdown() -> freeWindows(),
                // and shutdown() now drains any in-flight epoch-local Iallreduce before componentComm is freed.
                // which calls MPI_Win_free / (optional) MPI_Win_unlock_all. These are collective operations on the window communicator.
                // Destroying routers before the barrier can make some ranks block here while others are still simulating.
                // Delay destruction until after the barrier.
                router.reset();
                parameters.reset();
                if (componentComm != MPI_COMM_NULL) {
                    MPI_Comm_free(&componentComm);
                }

                if (rank == 0) {
                    mergeDedupDataFactoryToMergedDir(epochDate);
                }

                tracedBarrier(epochComm, "agent_merged_ready", epochIdx, epochDate);

                {
                    std::unordered_map<std::string,int> asset2kernel;
                    std::vector<std::string> allAssets;
                    (void)parseMultiKernelTargets(rootNode, asset2kernel, allAssets);
                    auto assign = readJsonFile(assignmentFilePath(epochDate));
                    if (isCross) {
                        evaluateBasketsForCrossRankAndWriteJsonl(
                            rank, epochDate, allAssets, assign,
                            basketCapN, assetEvalN,
                            daySeed, static_cast<uint64_t>(epochIdx)
                        );
                    }
                }

                tracedBarrier(epochComm, "agent_baskets_ready", epochIdx, epochDate);

                if (rank == 0) {
                    const std::string nextDate = nextTradingDateSkipWeekend(epochDate);
                    // When no cross ranks are configured, skip ALL cross-epoch topology updates.
                    // This allows multi-epoch runs to proceed cleanly in "no-cross" experiments.
                    if (!crossAgentRanksCfg.empty()) {
                        if (!baselineNoMigration) {
                            buildAndWriteNextAssignmentViaMtKaHyPar(epochDate, nextDate, rootNode, crossAgentRanksCfg, worldRankToNodeId);
                        } else if (baselineTopoFromBaskets) {
                            buildAndWriteNextKernelCrossMapNoMigration(epochDate, nextDate, rootNode, crossAgentRanksCfg, worldRankToNodeId);
                        }
                    }
                }

                tracedBarrier(epochComm, "agent_next_assignment_ready", epochIdx, epochDate);

                if (!crossAgentRanksCfg.empty() &&
                    (((!baselineNoMigration) || (baselineNoMigration && baselineTopoFromBaskets)) && epochIdx + 1 < epochCount)) {
                    const std::string nextDate = nextTradingDateSkipWeekend(epochDate);
                    auto k2r = readKernelCrossMapForDate(nextDate);
                    applyKernelCrossMapToMultiKernel(rootNode, k2r);
                }
            }

            std::cout << "Agent rank " << rank << " completed all epochs, finalizing..." << std::endl;
            tracedBarrier(epochComm, "agent_finalizing", /*epochIdx*/epochCount-1, curDate);
            {
                auto __proc_end = std::chrono::steady_clock::now();
                auto __proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__proc_end - __kernel_proc_start).count();
                std::cout << "[Perf] Process total wall time (agent role): "
                          << __proc_ms << " ms (" << (__proc_ms / 1000.0) << " s)" << std::endl;
            }
            MPI_Finalize();
            return 0;
        } else if (isLearner) {
            std::cout << "Learner rank " << rank << ": handled by Python (learner_mpi.py). Exiting C++ process." << std::endl;
            MPI_Finalize();
            return 0;
        } else {
            const char* allowIdle = std::getenv("DESMAR_ALLOW_UNKNOWN_RANKS");
            bool idle = (allowIdle && (*allowIdle=='1' || *allowIdle=='y' || *allowIdle=='Y' || *allowIdle=='t' || *allowIdle=='T'));
            if (idle) {
                std::cout << "Rank " << rank << " is not in this config's roles; idling (DESMAR_ALLOW_UNKNOWN_RANKS=1)" << std::endl;
                while (true) { std::this_thread::sleep_for(std::chrono::seconds(60)); }
            } else {
                std::cout << "Rank " << rank << " not in SimulationRank or AgentRanks for this config. Exiting." << std::endl;
                {
                    auto __proc_end = std::chrono::steady_clock::now();
                    auto __proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(__proc_end - __kernel_proc_start).count();
                    std::cout << "[Perf] Process total wall time (unused role): "
                              << __proc_ms << " ms (" << (__proc_ms / 1000.0) << " s)" << std::endl;
                }
                MPI_Finalize();
                return 0;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
    }
    
    return 0;
}