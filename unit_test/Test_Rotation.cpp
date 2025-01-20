#include <gtest/gtest.h>
#include "MDFT_Rotation.hpp"
#include "MDFT_Math_Utils.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestRotation : public ::testing::Test {
  using float_type   = T;
  using RotArrayType = Kokkos::Array<T, 3>;

  const float_type kx0                 = 0.209439510239320;
  std::vector<std::size_t> m_all_sizes = {0, 1, 2, 3, 4, 5};
  std::vector<RotArrayType> m_all_q    = {
      {0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}, {kx0, kx0, kx0}};
};

TYPED_TEST_SUITE(TestRotation, float_types);

template <typename T>
void test_coeff_initialization() {
  ASSERT_NO_THROW(({ MDFT::RotationCoeffs<execution_space, T> coeff; }));
}

template <typename T>
void test_spherical_harmonics_lu(int mmax, Kokkos::Array<T, 3> q) {
  using complex_type = Kokkos::complex<T>;
  using ViewType     = Kokkos::View<complex_type***, execution_space>;
  using RotArrayType = Kokkos::Array<T, 3>;
  // ArrayType q{1.0, 2.0, 3.0};
  ViewType R("R", mmax + 1, 2 * mmax + 1, 2 * mmax + 1),
      R_ref("R_ref", mmax + 1, 2 * mmax + 1, 2 * mmax + 1);
  MDFT::RotationCoeffs<execution_space, T> coeff;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        MDFT::Impl::rotation_matrix_between_complex_spherical_harmonics_lu(
            q, coeff.m_c, coeff.m_d, coeff.m_a, coeff.m_b, R);
      });

  // Make a reference at host
  int mmax_max = coeff.m_a.extent(0) + 1;
  T epsilon    = std::numeric_limits<T>::epsilon() * 1e3;
  auto h_a =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeff.m_a);
  auto h_b =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeff.m_b);
  auto h_c =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeff.m_c);
  auto h_d =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coeff.m_d);
  auto h_R_ref = Kokkos::create_mirror_view(R_ref);

  if (mmax == 0) {
    // R = (0, 0) in this case
    Kokkos::deep_copy(R_ref, Kokkos::complex<T>(1.0, 0.0));
    EXPECT_TRUE(allclose(execution_space(), R, R_ref, epsilon));
    return;
  }

  // m == 0
  h_R_ref(0, mmax, mmax) = Kokkos::complex<T>(1.0, 0.0);

  RotArrayType rmat0, rmat1, rmat2;
  if (q[0] == 0.0 && q[1] == 0.0 && q[2] == 0.0) {
    // theta definied as zero.
    rmat2 = {0.0, 0.0, 1.0};
  } else {
    rmat2 = q;
  }

  // Then deal with X and Y
  if (rmat2[0] != 0.0 || rmat2[1] != 0.0) {
    // if rmat2 is along with axis z, the GSH is null, and we don't carre about
    // phi. in the MDFT definition of Omega, the rotation axes are z-y-z.
    rmat1 = MDFT::Impl::cross_product(RotArrayType{0, 0, 1}, rmat2);
  } else {
    rmat1 = MDFT::Impl::cross_product(rmat2, RotArrayType{1, 0, 0});
  }

  // Normalize
  // to avoid round up error if rmat2 is so closed to z.
  rmat2 = MDFT::Impl::L2normalize(rmat2);
  rmat1 = MDFT::Impl::L2normalize(rmat1);
  rmat0 = MDFT::Impl::cross_product(rmat1, rmat2);

  // m == 1
  const T inv_sqrt2 = 1.0 / Kokkos::sqrt(2);
  h_R_ref(1, mmax - 1, mmax - 1) =
      complex_type((rmat1[1] + rmat0[0]) * 0.5, (rmat0[1] - rmat1[0]) * 0.5);
  h_R_ref(1, mmax - 1, mmax) =
      complex_type(rmat2[0] * inv_sqrt2, rmat2[1] * inv_sqrt2);
  h_R_ref(1, mmax - 1, mmax + 1) =
      complex_type((rmat1[1] - rmat0[0]) * 0.5, (-rmat0[1] - rmat1[0]) * 0.5);

  h_R_ref(1, mmax, mmax - 1) =
      complex_type(rmat0[2] * inv_sqrt2, -rmat1[2] * inv_sqrt2);
  h_R_ref(1, mmax, mmax) = complex_type(rmat2[2], 0.0);
  h_R_ref(1, mmax, mmax + 1) =
      complex_type(-rmat0[2] * inv_sqrt2, -rmat1[2] * inv_sqrt2);

  h_R_ref(1, mmax + 1, mmax - 1) =
      complex_type((rmat1[1] - rmat0[0]) * 0.5, (rmat0[1] + rmat1[0]) * 0.5);
  h_R_ref(1, mmax + 1, mmax) =
      complex_type(-rmat2[0] * inv_sqrt2, rmat2[1] * inv_sqrt2);
  h_R_ref(1, mmax + 1, mmax + 1) =
      complex_type((rmat1[1] + rmat0[0]) * 0.5, (rmat1[0] - rmat0[1]) * 0.5);

  if (mmax == 1) {
    Kokkos::deep_copy(R_ref, h_R_ref);
    EXPECT_TRUE(allclose(execution_space(), R, R_ref, epsilon));
    return;
  }

  // mmax > 1
  for (int l = 2; l <= mmax; ++l) {
    auto l1 = l - 1;
    for (int m = -l; m <= l; ++m) {
      int m1min = m > 0 ? 1 : 0;
      for (int m1 = m1min; m1 <= l - 1; ++m1) {
        if (m == -l) {
          auto b_tmp  = h_b(l - 2, mmax_max - m, mmax_max + m1);
          auto R0_tmp = h_R_ref(1, mmax - 1, mmax);
          auto R1_tmp = h_R_ref(l1, mmax + m + 1, mmax + m1);

          auto f = b_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g = b_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else if (m == l) {
          auto b_tmp  = h_b(l - 2, mmax_max + m, mmax_max + m1);
          auto R0_tmp = h_R_ref(1, mmax + 1, mmax);
          auto R1_tmp = h_R_ref(l1, mmax + m - 1, mmax + m1);

          auto f = b_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g = b_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else {
          auto a_tmp     = h_a(l - 2, mmax_max + m, mmax_max + m1);
          auto b_pos_tmp = h_b(l - 2, mmax_max + m, mmax_max + m1);
          auto b_neg_tmp = h_b(l - 2, mmax_max - m, mmax_max + m1);
          auto R0_tmp    = h_R_ref(1, mmax, mmax);
          auto R1_tmp    = h_R_ref(1, mmax + 1, mmax);
          auto R2_tmp    = h_R_ref(1, mmax - 1, mmax);
          auto Rm0_tmp   = h_R_ref(l1, mmax + m, mmax + m1);
          auto Rm1_tmp   = h_R_ref(l1, mmax + m + 1, mmax + m1);
          auto Rm2_tmp   = h_R_ref(l1, mmax + m - 1, mmax + m1);

          auto f = a_tmp * (R0_tmp.real() * Rm0_tmp.real()) +
                   b_pos_tmp * (R1_tmp.real() * Rm2_tmp.real() -
                                R1_tmp.imag() * Rm2_tmp.imag()) +
                   b_neg_tmp * (R2_tmp.real() * Rm1_tmp.real() -
                                R2_tmp.imag() * Rm1_tmp.imag());
          auto g = a_tmp * (R0_tmp.real() * Rm0_tmp.imag()) +
                   b_pos_tmp * (R1_tmp.real() * Rm2_tmp.imag() +
                                R1_tmp.imag() * Rm2_tmp.real()) +
                   b_neg_tmp * (R2_tmp.real() * Rm1_tmp.imag() +
                                R2_tmp.imag() * Rm1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        }
        auto pow_tmp = Kokkos::pow(-1.0, m + m1);
        auto f_conj  = pow_tmp * h_R_ref(l, mmax + m, mmax + m1).real();
        auto g_conj  = -pow_tmp * h_R_ref(l, mmax + m, mmax + m1).imag();
        h_R_ref(l, mmax - m, mmax - m1) = complex_type(f_conj, g_conj);
      }

      int m1 = l;
      {
        if (m == -l) {
          auto d_tmp  = h_d(l - 2, mmax_max - m);
          auto R0_tmp = h_R_ref(1, mmax - 1, mmax + 1);
          auto R1_tmp = h_R_ref(l1, mmax + m + 1, mmax + m1 - 1);
          auto f      = d_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g      = d_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else if (m == l) {
          auto d_tmp  = h_d(l - 2, mmax_max + m);
          auto R0_tmp = h_R_ref(1, mmax + 1, mmax + 1);
          auto R1_tmp = h_R_ref(l1, mmax + m - 1, mmax + m1 - 1);
          auto f      = d_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g      = d_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else {
          auto c_tmp     = h_c(l - 2, mmax_max + m);
          auto d_pos_tmp = h_d(l - 2, mmax_max + m);
          auto d_neg_tmp = h_d(l - 2, mmax_max - m);
          auto R0_tmp    = h_R_ref(1, mmax, mmax + 1);
          auto R1_tmp    = h_R_ref(1, mmax + 1, mmax + 1);
          auto R2_tmp    = h_R_ref(1, mmax - 1, mmax + 1);
          auto Rm0_tmp   = h_R_ref(l1, mmax + m, mmax + m1 - 1);
          auto Rm1_tmp   = h_R_ref(l1, mmax + m + 1, mmax + m1 - 1);
          auto Rm2_tmp   = h_R_ref(l1, mmax + m - 1, mmax + m1 - 1);

          auto f = c_tmp * (R0_tmp.real() * Rm0_tmp.real() -
                            R0_tmp.imag() * Rm0_tmp.imag()) +
                   d_pos_tmp * (R1_tmp.real() * Rm2_tmp.real() -
                                R1_tmp.imag() * Rm2_tmp.imag()) +
                   d_neg_tmp * (R2_tmp.real() * Rm1_tmp.real() -
                                R2_tmp.imag() * Rm1_tmp.imag());
          auto g = c_tmp * (R0_tmp.real() * Rm0_tmp.imag() +
                            R0_tmp.imag() * Rm0_tmp.real()) +
                   d_pos_tmp * (R1_tmp.real() * Rm2_tmp.imag() +
                                R1_tmp.imag() * Rm2_tmp.real()) +
                   d_neg_tmp * (R2_tmp.real() * Rm1_tmp.imag() +
                                R2_tmp.imag() * Rm1_tmp.real());
          h_R_ref(l, mmax + m, mmax + m1) = complex_type(f, g);
        }
        auto pow_tmp = Kokkos::pow(-1.0, m + m1);
        auto f_conj  = pow_tmp * h_R_ref(l, mmax + m, mmax + m1).real();
        auto g_conj  = -pow_tmp * h_R_ref(l, mmax + m, mmax + m1).imag();
        h_R_ref(l, mmax - m, mmax - m1) = complex_type(f_conj, g_conj);
      }
    }
  }

  Kokkos::deep_copy(R_ref, h_R_ref);
  EXPECT_TRUE(allclose(execution_space(), R, R_ref, epsilon, epsilon));
}

TYPED_TEST(TestRotation, CoeffInitialization) {
  using float_type = typename TestFixture::float_type;

  test_coeff_initialization<float_type>();
}

TYPED_TEST(TestRotation, SphericalHarmonicsLU) {
  using float_type = typename TestFixture::float_type;
  for (auto m : this->m_all_sizes) {
    for (auto q : this->m_all_q) {
      test_spherical_harmonics_lu<float_type>(m, q);
    }
  }
}

}  // namespace
