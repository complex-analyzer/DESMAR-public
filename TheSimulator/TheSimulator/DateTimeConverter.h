#pragma once

#include "Timestamp.h"
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>


class DateTimeConverter {
public:
    /**
     * Convert a date string and a time string into a nanosecond timestamp.
     * @param dateStr Date string in format "YYYYMMDD"
     * @param timeStr Time string in format "HH:MM:SS", "HH:MM:SS.mmm",
     *               or market time format (e.g. 93000000)
     * @return Nanosecond timestamp
     */
    static Timestamp dateTimeToNs(const std::string& dateStr, const std::string& timeStr) {
        // Initialize date structure and parse date string
        std::tm tm = {};
        std::istringstream dateSs(dateStr);
        dateSs >> std::get_time(&tm, "%Y%m%d");
        
        // Set time to 00:00:00 for the given day
        tm.tm_hour = 0;
        tm.tm_min = 0;
        tm.tm_sec = 0;
        
        // Convert to time point
        std::time_t dateTime = std::mktime(&tm);
        Timestamp dateNs = static_cast<Timestamp>(dateTime) * 1000000000LL;
        
        // Parse time part
        Timestamp timeNs = 0;
        if (timeStr.find(':') != std::string::npos) {
            // If contains ':', treat as time string format (e.g. "09:30:00")
            timeNs = timeStringToNs(timeStr);
        } else if (std::all_of(timeStr.begin(), timeStr.end(), ::isdigit)) {
            // Check whether it is market time format (e.g. 93000000)
            Timestamp marketTime = std::stoull(timeStr);
            if (marketTime >= 90000000 && marketTime <= 160000000) {
                // Convert market time format to nanosecond timestamp
                timeNs = marketTimeToNs(marketTime);
            } else {
                // Assume it is already a nanosecond timestamp
                timeNs = marketTime;
            }
        } else {
            // Default: treat as raw numeric value
            timeNs = std::stoull(timeStr);
        }
        
        return dateNs + timeNs;
    }
    
    /**
     * Convert a nanosecond timestamp to a date-time string.
     * @param nsTimestamp Nanosecond timestamp
     * @return Date-time string in format "YYYY-MM-DD HH:MM:SS.mmm"
     */
    static std::string nsToDateTimeString(Timestamp nsTimestamp) {
        // Extract date part (seconds-level timestamp)
        std::time_t seconds = nsTimestamp / 1000000000LL;
        std::tm* tm = std::localtime(&seconds);
        
        // Extract millisecond part
        Timestamp milliseconds = (nsTimestamp % 1000000000LL) / 1000000LL;
        
        // Format as string
        std::ostringstream oss;
        oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "." 
            << std::setfill('0') << std::setw(3) << milliseconds;
        
        return oss.str();
    }
    
    /**
     * Convert a time string to a nanosecond timestamp (within the same day).
     * @param timeStr Time string in format "HH:MM:SS" or "HH:MM:SS.mmm"
     * @return Nanosecond timestamp
     */
    static Timestamp timeStringToNs(const std::string& timeStr) {
        std::tm tm = {};
        std::istringstream ss(timeStr);
        
        // Try to parse time format
        if (timeStr.find('.') != std::string::npos) {
            // With milliseconds, format "HH:MM:SS.mmm"
            ss >> std::get_time(&tm, "%H:%M:%S");
            
            // Extract millisecond part
            std::string msStr = timeStr.substr(timeStr.find('.') + 1);
            int ms = std::stoi(msStr);
            
            // Convert to nanoseconds
            return (tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec) * 1000000000LL + ms * 1000000LL;
        } else {
            // Without milliseconds, format "HH:MM:SS"
            ss >> std::get_time(&tm, "%H:%M:%S");
            
            // Convert to nanoseconds
            return (tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec) * 1000000000LL;
        }
    }
    
    /**
     * Convert market time format (e.g. 93000000 means 9:30:00.000) to nanoseconds.
     * @param marketTime Market time format
     * @return Nanosecond timestamp
     */
    static Timestamp marketTimeToNs(Timestamp marketTime) {
        // Extract hour, minute, second and millisecond
        Timestamp hour = marketTime / 10000000;
        Timestamp minute = (marketTime % 10000000) / 100000;
        Timestamp second = (marketTime % 100000) / 1000;
        Timestamp millisecond = marketTime % 1000;
        
        // Convert to nanoseconds
        return (hour * 3600 + minute * 60 + second) * 1000000000LL + millisecond * 1000000LL;
    }
    
    /**
     * Convert a nanosecond timestamp to market time format.
     * @param nsTime Nanosecond timestamp
     * @return Market time format
     */
    static Timestamp nsToMarketTime(Timestamp nsTime) {
        // Extract hour, minute, second and millisecond
        Timestamp totalSeconds = nsTime / 1000000000LL;
        Timestamp hour = totalSeconds / 3600;
        Timestamp minute = (totalSeconds % 3600) / 60;
        Timestamp second = totalSeconds % 60;
        Timestamp millisecond = (nsTime % 1000000000LL) / 1000000LL;
        
        // Convert to market time format
        return hour * 10000000 + minute * 100000 + second * 1000 + millisecond;
    }
    
    /**
     * Convert a nanosecond timestamp to a human-readable time string.
     * @param nsTime Nanosecond timestamp
     * @return Time string in format "HH:MM:SS.mmm"
     */
    static std::string nsToTimeString(Timestamp nsTime) {
        // Extract hour, minute, second and millisecond
        Timestamp totalSeconds = nsTime / 1000000000LL;
        Timestamp hour = totalSeconds / 3600;
        Timestamp minute = (totalSeconds % 3600) / 60;
        Timestamp second = totalSeconds % 60;
        Timestamp millisecond = (nsTime % 1000000000LL) / 1000000LL;
        
        // Format as string
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << hour << ":"
            << std::setfill('0') << std::setw(2) << minute << ":"
            << std::setfill('0') << std::setw(2) << second << "."
            << std::setfill('0') << std::setw(3) << millisecond;
        
        return oss.str();
    }
};
