// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <cmath>
#include <random>
#include "MDFT_Wigner.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestWigner : public ::testing::Test {
  using float_type             = T;
  std::vector<int> m_all_sizes = {0, 1, 2, 5, 10};
};

TYPED_TEST_SUITE(TestWigner, float_types);

template <typename T>
void test_wigner_small_d_quick_return(int m, int mu, int mup) {
  ASSERT_NO_THROW(({
    [[maybe_unused]] auto d = MDFT::Impl::wigner_small_d(m, mu, mup, 0.0);
  }));
  auto d = MDFT::Impl::wigner_small_d(m, mu, mup, 0.0);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  if (Kokkos::abs(mu) > m || Kokkos::abs(mup) > m) {
    EXPECT_LT(Kokkos::abs(d - 0.0), epsilon);
  } else {
    if (m == 0) {
      EXPECT_LT(Kokkos::abs(d - 1.0), epsilon);
    }
  }
  EXPECT_TRUE(d >= 0.0);
}

template <typename T>
void test_wigner_small_d_mu0_or_mup0(int m, int mu, int mup) {
  // Exclude already tested cases
  if (Kokkos::abs(mu) > m || Kokkos::abs(mup) > m) return;
  if (m == 0) return;
  if (mu == 0 || mup == 0) {
    // Give theta by a random number [0, 1]
    std::default_random_engine generator;
    std::uniform_real_distribution<T> unif(0, 1);
    T theta = unif(generator);

    // Make a reference for (mu == 0 || mup == 0) case
    auto mu0    = mu;
    auto theta0 = theta;  // je mets le 0 en second
    if (mu == 0) {
      mu0    = mup;
      theta0 = -theta;
    }
    T x = 1.0;  // si mu negatif, ca vaut (-1)**mu * valeur pour -mu
    if (mu0 < 0) {
      x   = std::pow(-1.0, mu0);
      mu0 = -mu0;
    }

    T cc  = std::cos(theta0);
    T pm1 = 0.0;
    T pm  = 1.0;

    for (int l = mu0 + 1; l <= m; ++l) {
      T pm2 = pm1;
      pm1   = pm;
      pm    = (cc * static_cast<T>(2 * l - 1) * pm1 -
            static_cast<T>(l + mu0 - 1) * pm2) /
           static_cast<T>(l - mu0);
    }

    auto ref_wigner_small_d =
        x * std::pow(-1.0, mu0) *
        std::sqrt(MDFT::Impl::factorial[m - mu0] /
                  MDFT::Impl::factorial[m + mu0]) *
        MDFT::Impl::factorial[2 * mu0] /
        (std::pow(2.0, mu0) * MDFT::Impl::factorial[mu0]) *
        std::pow(std::sin(theta0), mu0) * pm;

    auto d    = MDFT::Impl::wigner_small_d(m, mu, mup, theta);
    T epsilon = std::numeric_limits<T>::epsilon() * 100;
    EXPECT_LT(Kokkos::abs(d - ref_wigner_small_d), epsilon);
  } else {
    return;
  }
}

template <typename T>
void test_wigner_small_d_non_zero(int m, int mu, int mup) {
  // Exclude already tested cases
  if (Kokkos::abs(mu) > m || Kokkos::abs(mup) > m) return;
  if (m == 0) return;
  if (mu == 0 || mup == 0) return;

  // Give theta by a random number [0, 1]
  std::default_random_engine generator;
  std::uniform_real_distribution<T> unif(0, 1);
  T theta = unif(generator);

  // Make a reference for the else case
  T cc = std::cos(0.5 * theta);
  T ss = std::sin(0.5 * theta);

  T tmp_wigner_small_d = 0;
  for (int it = std::max(0, mu - mup); it <= std::min(m + mu, m - mup); ++it) {
    tmp_wigner_small_d +=
        std::pow(-1.0, it) /
        (MDFT::Impl::factorial[m + mu - it] *
         MDFT::Impl::factorial[m - mup - it] * MDFT::Impl::factorial[it] *
         MDFT::Impl::factorial[it - mu + mup]) *
        std::pow(cc, 2 * m + mu - mup - 2 * it) *
        std::pow(ss, 2 * it - mu + mup);
  }

  auto ref_wigner_small_d =
      std::sqrt(MDFT::Impl::factorial[m + mu] * MDFT::Impl::factorial[m - mu] *
                MDFT::Impl::factorial[m + mup] *
                MDFT::Impl::factorial[m - mup]) *
      tmp_wigner_small_d;

  auto d    = MDFT::Impl::wigner_small_d(m, mu, mup, theta);
  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_LT(Kokkos::abs(d - ref_wigner_small_d), epsilon);
}

TYPED_TEST(TestWigner, QuickReturn) {
  using float_type = typename TestFixture::float_type;

  for (auto m : this->m_all_sizes) {
    for (int mu = -m; mu <= m; ++mu) {
      for (int mup = -m; mup <= m; ++mup) {
        test_wigner_small_d_quick_return<float_type>(m, mu, mup);
      }
    }
  }
}

TYPED_TEST(TestWigner, MuIs0) {
  using float_type = typename TestFixture::float_type;

  for (auto m : this->m_all_sizes) {
    for (int mu = -m; mu <= m; ++mu) {
      for (int mup = -m; mup <= m; ++mup) {
        test_wigner_small_d_mu0_or_mup0<float_type>(m, mu, mup);
      }
    }
  }
}

TYPED_TEST(TestWigner, NonZero) {
  using float_type = typename TestFixture::float_type;

  for (auto m : this->m_all_sizes) {
    for (int mu = -m; mu <= m; ++mu) {
      for (int mup = -m; mup <= m; ++mup) {
        test_wigner_small_d_non_zero<float_type>(m, mu, mup);
      }
    }
  }
}

}  // namespace
