#pragma once

#include <cmath>
#include "Money.h"

namespace PriceRounding {

// Banker's rounding (round half to even) to nearest int
int roundHalfEvenToInt(double x);

// Banker's rounding to nearest long long
long long roundHalfEvenToLL(double x);

// Convert Money to integer cents with ties-to-even using only internal representation
int moneyToCentsHalfEven(const Money& price);

// Average two integer-cent prices with ties-to-even
int averageCentsHalfEven(int a, int b);

// Build Money from integer cents (supports negative)
inline Money centsToMoney(int cents) {
    long long wholes = cents / 100;
    unsigned int cents_abs = static_cast<unsigned int>(std::abs(cents % 100));
    return Money(static_cast<signed long long int>(wholes), cents_abs);
}

} // namespace PriceRounding


