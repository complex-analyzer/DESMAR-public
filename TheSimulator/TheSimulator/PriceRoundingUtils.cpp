#include "PriceRoundingUtils.h"

namespace PriceRounding {

int roundHalfEvenToInt(double x) {
    if (!std::isfinite(x)) return 0;
    double ip = 0.0;
    double fp = std::modf(x, &ip);
    if (fp > 0.5) return static_cast<int>(ip) + 1;
    if (fp < -0.5) return static_cast<int>(ip) - 1;
    if (fp > -0.5 && fp < 0.5) return static_cast<int>(ip);
    long long base = static_cast<long long>(ip);
    bool even = (base % 2LL) == 0LL;
    if (even) return static_cast<int>(base);
    return fp > 0 ? static_cast<int>(base + 1LL) : static_cast<int>(base - 1LL);
}

long long roundHalfEvenToLL(double x) {
    if (!std::isfinite(x)) return 0LL;
    double ip = 0.0;
    double fp = std::modf(x, &ip);
    if (fp > 0.5) return static_cast<long long>(ip) + 1LL;
    if (fp < -0.5) return static_cast<long long>(ip) - 1LL;
    if (fp > -0.5 && fp < 0.5) return static_cast<long long>(ip);
    long long base = static_cast<long long>(ip);
    bool even = (base % 2LL) == 0LL;
    if (even) return base;
    return fp > 0 ? base + 1LL : base - 1LL;
}

int moneyToCentsHalfEven(const Money& price) {
    long long internal = price.internalValueRaw();
    long long sign = (internal < 0) ? -1LL : 1LL;
    long long abs_internal = internal < 0 ? -internal : internal;
    const long long CENT = Money::CENT_OFFSET; // 1 cent in internal units
    long long base_cents = abs_internal / CENT;           // truncate toward zero
    long long rem = abs_internal % CENT;                  // sub-cent remainder

    if (rem > CENT / 2) {
        base_cents += 1;
    } else if (rem == CENT / 2) {
        if ((base_cents % 2LL) != 0LL) {
            base_cents += 1;
        }
    }
    long long cents_signed = sign * base_cents;
    return static_cast<int>(cents_signed);
}

int averageCentsHalfEven(int a, int b) {
    long long sum = static_cast<long long>(a) + static_cast<long long>(b);
    long long base = sum / 2; // floor toward zero, fine for non-negative sums here
    if ((sum & 1LL) == 0LL) {
        return static_cast<int>(base);
    }
    if ((base % 2LL) != 0LL) {
        base += 1LL;
    }
    return static_cast<int>(base);
}

} // namespace PriceRounding


