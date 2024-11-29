#include <gtest/gtest.h>
#include "MDFT_Rotation.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestRotation : public ::testing::Test {
  using float_type                     = T;
  std::vector<std::size_t> m_all_sizes = {1, 2, 5, 10};
  float_type m_delta                   = 0.1;
};

TYPED_TEST_SUITE(TestRotation, float_types);

template <typename T>
void test_coeff_initialization() {
  ASSERT_NO_THROW(({ MDFT::RotationCoeffs<execution_space, T> coeff; }));
}

template <typename T>
void test_spherical_harmonics_lu() {
  using ViewType  = Kokkos::View<Kokkos::complex<T>***, execution_space>;
  using ArrayType = Kokkos::Array<T, 3>;
  ArrayType q{1.0, 2.0, 3.0};
  int mmax = 6;
  ViewType R("R", mmax + 1, 2 * mmax + 1, 2 * mmax + 1);
  MDFT::RotationCoeffs<execution_space, T> coeff;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        MDFT::Impl::rotation_matrix_between_complex_spherical_harmonics_lu(
            q, coeff.m_c, coeff.m_d, coeff.m_a, coeff.m_b, R);
      });

  // Need to add checks. Do not know how to do it yet
}

TYPED_TEST(TestRotation, CoeffInitialization) {
  using float_type = typename TestFixture::float_type;

  test_coeff_initialization<float_type>();
}

TYPED_TEST(TestRotation, SphericalHarmonicsLU) {
  using float_type = typename TestFixture::float_type;

  for (auto n : this->m_all_sizes) {
    test_spherical_harmonics_lu<float_type>();
  }
}

}  // namespace
