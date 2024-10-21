#include <gtest/gtest.h>
#include "MDFT_Math_Utils.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestMathUtils : public ::testing::Test {
  using float_type                     = T;
  std::vector<std::size_t> m_all_sizes = {1, 2, 5, 10};
  float_type m_delta                   = 0.1;
};

TYPED_TEST_SUITE(TestMathUtils, float_types);

template <typename T>
void test_gauss_legendre(int n) {
  using ViewType = Kokkos::View<T*, execution_space>;
  ViewType x("x", n);
  ViewType w("w", n);
  MDFT::Impl::gauss_legendre(x, w);
  T sum = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, n},
      KOKKOS_LAMBDA(int i, T& lsum) { lsum += w(i); }, sum);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_LT(Kokkos::abs(sum - 1.0), epsilon);
}

template <typename T>
void test_uniform_mesh(int n, T dx) {
  using ViewType = Kokkos::View<T*, execution_space>;
  ViewType x("x", n), x_ref("x_ref", n);
  MDFT::Impl::uniform_mesh(execution_space(), x, dx);

  // Reference
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  for (int i = 0; i < n; ++i) {
    h_x_ref(i) = i * dx;
  }
  Kokkos::deep_copy(x_ref, h_x_ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(execution_space(), x, x_ref, epsilon));
}

TYPED_TEST(TestMathUtils, GaussLegendre) {
  using float_type = typename TestFixture::float_type;

  for (auto n : this->m_all_sizes) {
    test_gauss_legendre<float_type>(n);
  }
}

TYPED_TEST(TestMathUtils, UniformMesh) {
  using float_type = typename TestFixture::float_type;

  for (auto n : this->m_all_sizes) {
    test_uniform_mesh<float_type>(n, this->m_delta);
  }
}

}  // namespace
