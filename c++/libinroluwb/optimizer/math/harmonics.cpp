#include "libinroluwb/optimizer/math/harmonics.hpp"

namespace inrol {
  int index2degree(int k) {
    int l = 0;
    while (true) {
      if (k < (l + 1) * (l + 1)) {
        break;
      }
      l++;
    }
    return l;
  }

  int index2order(int k, int l) {
    return (k - l * l) - l;
  }
}  // namespace inrol
