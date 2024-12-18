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
  using ViewType = Kokkos::View<T******, execution_space>;
  const int n0 = 2, n1 = 2, n2 = 2, n3 = 2, n4 = 2, n5 = 2;
  ViewType xi("xi", n0, n1, n2, n3, n4, n5),
      delta_rho("delta_rho", n0, n1, n2, n3, n4, n5),
      delta_rho_ref("delta_rho_ref", n0, n1, n2, n3, n4, n5);

  T rho0 = 0.3;

  // Initialize xi with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(xi, random_pool, 1.0);

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

template <typename T>
void test_get_delta_f() {
  using View3DType = Kokkos::View<T***, execution_space>;
  using View6DType = Kokkos::View<T******, execution_space>;
  const int n0 = 2, n1 = 2, n2 = 2, n3 = 2, n4 = 2, n5 = 2;
  View3DType vexc("vexc", n3, n4, n5), w("w", n3, n4, n5);
  View6DType xi("xi", n0, n1, n2, n3, n4, n5),
      delta_f("delta_f", n0, n1, n2, n3, n4, n5),
      delta_f_ref("delta_f_ref", n0, n1, n2, n3, n4, n5);
  T ff_ref = 0;
  T rho0 = 0.3, prefactor = 0.334;

  // Initialize xi with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(xi, random_pool, 1.0);
  Kokkos::fill_random(vexc, random_pool, 1.0);
  Kokkos::fill_random(w, random_pool, 1.0);

  // Reference
  auto h_xi   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
  auto h_vexc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vexc);
  auto h_w    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
  auto h_delta_f_ref = Kokkos::create_mirror_view(delta_f_ref);
  for (int i0 = 0; i0 < n0; i0++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i3 = 0; i3 < n3; i3++) {
          for (int i4 = 0; i4 < n4; i4++) {
            for (int i5 = 0; i5 < n5; i5++) {
              h_delta_f_ref(i0, i1, i2, i3, i4, i5) =
                  2.0 * rho0 * h_xi(i0, i1, i2, i3, i4, i5) *
                  h_vexc(i3, i4, i5);

              ff_ref +=
                  rho0 *
                  (h_xi(i0, i1, i2, i3, i4, i5) * h_xi(i0, i1, i2, i3, i4, i5) -
                   1.0) *
                  prefactor * h_w(i3, i4, i5) * h_vexc(i3, i4, i5);
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(delta_f_ref, h_delta_f_ref);

  // Compute delta_f and ff
  execution_space exec_space;
  T ff = 0;
  MDFT::get_delta_f(exec_space, xi, vexc, w, delta_f, ff, rho0, prefactor);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(exec_space, delta_f, delta_f_ref, epsilon));
  EXPECT_LE(Kokkos::abs(ff - ff_ref), epsilon * Kokkos::abs(ff_ref));
}

TYPED_TEST(TestSolvent, delta_rho) {
  using float_type = typename TestFixture::float_type;
  test_get_delta_rho<float_type>();
}

TYPED_TEST(TestSolvent, delta_f) {
  using float_type = typename TestFixture::float_type;
  test_get_delta_f<float_type>();
}

}  // namespace
