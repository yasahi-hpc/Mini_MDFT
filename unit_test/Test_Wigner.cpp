#include <gtest/gtest.h>
#include <tuple>
#include <vector>
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

}  // namespace
