// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#ifndef MDFT_WIGNER_HPP
#define MDFT_WIGNER_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include "MDFT_Asserts.hpp"
#include "MDFT_Concepts.hpp"

namespace MDFT {
namespace Impl {

// Copy pasted from module_mathematica.f90
// last few digits are different from those given by AI
// need to check
static constexpr double factorial[35] = {1.0,
                                         1.0,
                                         2.0,
                                         6.0,
                                         24.0,
                                         120.0,
                                         720.0,
                                         5040.0,
                                         40320.0,
                                         362880.0,
                                         3628800.0,
                                         39916800.0,
                                         479001600.0,
                                         6227020800.0,
                                         87178291200.0,
                                         1307674368000.0,
                                         20922789888000.0,
                                         355687428096000.0,
                                         6402373705728000.0,
                                         1.2164510040883200e17,
                                         2.4329020081766400e18,
                                         5.1090942171709440e19,
                                         1.1240007277776077e21,
                                         2.5852016738884978e22,
                                         6.2044840173323941e23,
                                         1.5511210043330986e25,
                                         4.0329146112660565e26,
                                         1.0888869450418352e28,
                                         3.0488834461171384e29,
                                         8.8417619937397008e30,
                                         2.6525285981219103e32,
                                         8.2228386541779224e33,
                                         2.6313083693369352e35,
                                         8.6833176188118859e36,
                                         2.9523279903960412e38};

/// \brief r^m_{mu,mup}(\theta)
///        theta is the angle in radian.
///        see
///        http://pdg.lbl.gov/2015/reviews/rpp2014-rev-clebsch-gordan-coefs.pdf
///        <= we have the same normalization etc for d^m_{mu,mup}(theta)
///
template <typename ScalarType>
ScalarType wigner_small_d(int m, int mu, int mup, ScalarType theta) {
  // Quick return if possible
  if (Kokkos::abs(mu) > m || Kokkos::abs(mup) > m) {
    return 0.0;
  }

  if (m == 0) {
    return 1.0;
  }

  if (mu == 0 || mup == 0) {
    auto mu0    = mu;
    auto theta0 = theta;  // je mets le 0 en second
    if (mu == 0) {
      mu0    = mup;
      theta0 = -theta;
    }
    ScalarType x = 1.0;  // si mu negatif, ca vaut (-1)**mu * valeur pour -mu
    if (mu0 < 0) {
      x   = std::pow(-1.0, mu0);
      mu0 = -mu0;
    }

    ScalarType cc  = std::cos(theta0);
    ScalarType pm1 = 0.0;
    ScalarType pm  = 1.0;

    for (int l = mu0 + 1; l <= m; ++l) {
      ScalarType pm2 = pm1;
      pm1            = pm;
      pm             = (cc * static_cast<ScalarType>(2 * l - 1) * pm1 -
            static_cast<ScalarType>(l + mu0 - 1) * pm2) /
           static_cast<ScalarType>(l - mu0);
    }

    MDFT::Impl::Throw_If((m - mu0 < 0 || m + mu0 < 0 || 2 * mu0 < 0 || mu0 < 0),
                         "Factorial index out of range");

    return x * std::pow(-1.0, mu0) *
           std::sqrt(factorial[m - mu0] / factorial[m + mu0]) *
           factorial[2 * mu0] / (std::pow(2.0, mu0) * factorial[mu0]) *
           std::pow(std::sin(theta0), mu0) * pm;
  } else {
    ScalarType tmp_wigner_small_d = 0.0;
    ScalarType cc                 = std::cos(0.5 * theta);
    ScalarType ss                 = std::sin(0.5 * theta);

    for (int it = std::max(0, mu - mup); it <= std::min(m + mu, m - mup);
         ++it) {
      MDFT::Impl::Throw_If(
          (m + mu - it < 0 || m - mup - it < 0 || it - mu + mup < 0),
          "Factorial index out of range");
      tmp_wigner_small_d += std::pow(-1.0, it) /
                            (factorial[m + mu - it] * factorial[m - mup - it] *
                             factorial[it] * factorial[it - mu + mup]) *
                            std::pow(cc, 2 * m + mu - mup - 2 * it) *
                            std::pow(ss, 2 * it - mu + mup);
    }

    MDFT::Impl::Throw_If(
        (m + mu < 0 || m - mu < 0 || m + mup < 0 || m - mup < 0),
        "Factorial index out of range");

    return std::sqrt(factorial[m + mu] * factorial[m - mu] *
                     factorial[m + mup] * factorial[m - mup]) *
           tmp_wigner_small_d;
  }
}

}  // namespace Impl
}  // namespace MDFT

#endif
