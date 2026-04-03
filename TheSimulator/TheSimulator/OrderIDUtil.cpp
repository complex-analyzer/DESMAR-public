#include "OrderIDUtil.h"
#include <algorithm>
#include <cctype>

std::string extractSourceAgentFromOrderID(const std::string& orderId) {
    size_t lastUnderscorePos = orderId.rfind('_');
    if (lastUnderscorePos == std::string::npos) {
        return orderId;
    }
    
    std::string lastPart = orderId.substr(lastUnderscorePos + 1);
    bool isLastPartNumeric = !lastPart.empty() && 
                            std::all_of(lastPart.begin(), lastPart.end(), 
                                      [](char c) { return std::isdigit(c); });
    
    if (isLastPartNumeric) {
        return orderId.substr(0, lastUnderscorePos);
    } else {
        return orderId;
    }
} 