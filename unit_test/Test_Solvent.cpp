#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <Kokkos_Random.hpp>
#include "MDFT_Grid.hpp"
#include "MDFT_Solvent.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestSolvent : public ::testing::Test {
  using float_type = T;
};

TYPED_TEST_SUITE(TestSolvent, float_types);

template <typename T>
void test_get_delta_rho() {
  using ViewType = Kokkos::View<Kokkos::complex<T>******, execution_space>;
  const int n0 = 2, n1 = 2, n2 = 2, n3 = 2, n4 = 2, n5 = 2;
  ViewType xi("xi", n0, n1, n2, n3, n4, n5),
      delta_rho("delta_rho", n0, n1, n2, n3, n4, n5),
      delta_rho_ref("delta_rho_ref", n0, n1, n2, n3, n4, n5);

  T rho0 = 0.3;

  // Initialize xi with random values
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(xi, random_pool, z);

  // Reference
  auto h_xi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
  auto h_delta_rho_ref = Kokkos::create_mirror_view(delta_rho_ref);
  for (int i0 = 0; i0 < n0; i0++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i3 = 0; i3 < n3; i3++) {
          for (int i4 = 0; i4 < n4; i4++) {
            for (int i5 = 0; i5 < n5; i5++) {
              // delta_rho = rho0 * (xi^2 - 1)
              h_delta_rho_ref(i0, i1, i2, i3, i4, i5) =
                  rho0 *
                  (h_xi(i0, i1, i2, i3, i4, i5) * h_xi(i0, i1, i2, i3, i4, i5) -
                   1.0);
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(delta_rho_ref, h_delta_rho_ref);

  // Compute delta_rho
  execution_space exec_space;
  MDFT::get_delta_rho(exec_space, xi, delta_rho, rho0);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(exec_space, delta_rho, delta_rho_ref, epsilon));
}

TYPED_TEST(TestSolvent, delta_rho) {
  using float_type = typename TestFixture::float_type;
  test_get_delta_rho<float_type>();
}

}  // namespace
