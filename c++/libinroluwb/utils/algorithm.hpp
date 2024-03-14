#pragma once

#include <vector>
#include <optional>

namespace inrol {
  using std::optional;
  using std::vector;

  template <typename T>
  optional<size_t> find_index_in_vector(vector<T> vector, T value) {
    optional<size_t> result = std::nullopt;
    size_t i = 0;
    for (auto const entry : vector) {
      if (entry == value) {
        result = i;
      }
      i++;
    }
    return result;
  }
}  // namespace inrol
