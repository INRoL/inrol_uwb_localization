#pragma once

#include <glog/logging.h>
#include <ceres/ceres.h>

#include <cmath>

namespace inrol {
  int index2degree(int k);
  int index2order(int k, int l);

  template <typename T>
  T legendre(int l, int m, T x) {
    CHECK(m >= 0) << "`m` of associated legendre should not be less than 0";
    CHECK(m <= l) << "`m` should not be greater than `l`";

    T Pmm = T(1.0);
    if (m > 0) {
      T somx2 = ceres::sqrt((T(1.0) - x) * (T(1.0) + x));
      for (int i = 1; i <= m; i++) {
        Pmm *= -T(2 * i - 1) * somx2;
      }
    }
    if (l == m) {
      return Pmm;
    }
    T Pmmp1 = x * T(2.0 * m + 1.0) * Pmm;
    if (l == m + 1) {
      return Pmmp1;
    }

    T Plm = T(0.0);
    for (int ll = m + 2; ll <= l; ll++) {
      Plm = (T(2.0 * ll - 1.0) * x * Pmmp1 - T(ll + m - 1.0) * Pmm) / T(ll - m);
      Pmm = Pmmp1;
      Pmmp1 = Plm;
    }
    return Plm;
  }

  template <typename T>
  T dlegendre(int l, int m, T x) {
    CHECK(m >= 0) << "`m` of associated legendre should not be less than 0";
    CHECK(m <= l) << "`m` should not be greater than `l`";
    T Plm = legendre(l, m, x);
    T Pl1m = legendre(l + 1, m, x);
    return (-(l + 1) * x * Plm + (l - m + 1) * Pl1m) / (x * x - 1);
  }

  template <typename T>
  T normal_term(int l, int m) {
    double factor = (2.0 * l + 1.0) * std::tgamma(l - m + 1) /
      (4.0 * M_PI * std::tgamma(l + m + 1));
    return ceres::sqrt(T(factor));
  }

  template <typename T>
  T real_spherical_harmonics(int l, int m, T theta, T phi) {
    T x = ceres::cos(theta);
    T phase = T(m) * phi;
    T sqrt2 = T(std::sqrt(2.0));

    T sign = T(1.0);
    if (m % 2 != 0)
      sign = T(-1.0);

    if (m == 0) {
      T Plm = legendre(l, m, x);
      T N = normal_term<T>(l, m);
      return N * Plm;
    } else if (m > 0) {
      T Plm = legendre(l, m, x);
      T N = normal_term<T>(l, m);
      return sign * sqrt2 * N * Plm * ceres::cos(phase);
    } else {
      T Plm = legendre(l, -m, x);
      T N = normal_term<T>(l, -m);
      return sign * sqrt2 * N * Plm * ceres::sin(-phase);
    }
  }

  template <typename T>
  T dYdphi(int l, int m, T theta, T phi) {
    T x = ceres::cos(theta);
    T phase = T(m) * phi;
    T sqrt2 = T(std::sqrt(2.0));

    T sign = T(1.0);
    if (m % 2 == 0)
      sign = T(-1.0);

    if (m == 0) {
      return T(0.0);
    } else if (m > 0) {
      T Plm = legendre(l, m, x);
      T N = normal_term<T>(l, m);
      return sign * sqrt2 * m * N * std::sin(phase) * Plm;
    } else {
      T Plm = legendre(l, -m, x);
      T N = normal_term<T>(l, -m);
      return sign * sqrt2 * m * N * std::cos(-phase) * Plm;
    }
  }

  template <typename T>
  T dYdtheta(int l, int m, T theta, T phi) {
    T x = ceres::cos(theta);
    T phase = T(m) * phi;
    T sqrt2 = T(std::sqrt(2.0));

    T sign = T(1.0);
    if (m % 2 == 0)
      sign = T(-1.0);

    if (m == 0) {
      T dPlm = dlegendre(l, m, x);
      T N = normal_term<T>(l, m);
      return -N * std::sin(theta) * dPlm;
    } else if (m > 0) {
      T dPlm = dlegendre(l, m, x);
      T N = normal_term<T>(l, m);
      return sign * sqrt2 * N * std::cos(phase) * std::sin(theta) * dPlm;
    } else {
      T dPlm = dlegendre(l, -m, x);
      T N = normal_term<T>(l, -m);
      return sign * sqrt2 * N * std::sin(-phase) * std::sin(theta) * dPlm;
    }
  }
}  // namespace inrol
