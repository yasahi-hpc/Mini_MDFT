#include <gtest/gtest.h>
#include "MDFT_Math_Utils.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;
using int_types       = ::testing::Types<int, long>;

template <typename T>
struct TestMathUtils : public ::testing::Test {
  using float_type                     = T;
  std::vector<std::size_t> m_all_sizes = {1, 2, 5, 10};
  float_type m_delta                   = 0.1;
};

template <typename T>
struct TestMathUtilsIntOp : public ::testing::Test {
  using int_type                       = T;
  std::vector<std::size_t> m_all_sizes = {1, 2, 5, 10};
};

TYPED_TEST_SUITE(TestMathUtils, float_types);
TYPED_TEST_SUITE(TestMathUtilsIntOp, int_types);

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
  MDFT::Impl::uniform_mesh(x, dx);

  // Reference
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  for (int i = 0; i < n; ++i) {
    h_x_ref(i) = i * dx;
  }
  Kokkos::deep_copy(x_ref, h_x_ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(execution_space(), x, x_ref, epsilon));
}

template <typename T>
void test_cross_product() {
  using ViewType  = Kokkos::View<T*, execution_space>;
  using ArrayType = Kokkos::Array<T, 3>;
  ArrayType A{1.0, 2.0, 3.0}, B{4.0, 5.0, 6.0};
  ArrayType Ex{1.0, 0.0, 0.0}, Ey{0.0, 1.0, 0.0}, Ez{0.0, 0.0, 1.0};

  ViewType AxB("AxB", 3), AxEx("AxEx", 3), AxEy("AxEy", 3), AxEz("AxEz", 3);
  ViewType AxB_ref("AxB_ref", 3), AxEx_ref("AxEx_ref", 3),
      AxEy_ref("AxEy_ref", 3), AxEz_ref("AxEz_ref", 3);

  // Set reference
  auto h_AxB_ref  = Kokkos::create_mirror_view(AxB_ref);
  auto h_AxEx_ref = Kokkos::create_mirror_view(AxEx_ref);
  auto h_AxEy_ref = Kokkos::create_mirror_view(AxEy_ref);
  auto h_AxEz_ref = Kokkos::create_mirror_view(AxEz_ref);

  {
    h_AxEx_ref(0) = 0.0;
    h_AxEx_ref(1) = 3.0;
    h_AxEx_ref(2) = -2.0;
    h_AxEy_ref(0) = -3.0;
    h_AxEy_ref(1) = 0.0;
    h_AxEy_ref(2) = 1.0;
    h_AxEz_ref(0) = 2.0;
    h_AxEz_ref(1) = -1.0;
    h_AxEz_ref(2) = 0.0;
    h_AxB_ref(0)  = A[1] * B[2] - A[2] * B[1];
    h_AxB_ref(1)  = A[2] * B[0] - A[0] * B[2];
    h_AxB_ref(2)  = A[0] * B[1] - A[1] * B[0];
  }

  Kokkos::deep_copy(AxB_ref, h_AxB_ref);
  Kokkos::deep_copy(AxEx_ref, h_AxEx_ref);
  Kokkos::deep_copy(AxEy_ref, h_AxEy_ref);
  Kokkos::deep_copy(AxEz_ref, h_AxEz_ref);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        auto tmp_AxEx = MDFT::Impl::cross_product(A, Ex);
        auto tmp_AxEy = MDFT::Impl::cross_product(A, Ey);
        auto tmp_AxEz = MDFT::Impl::cross_product(A, Ez);
        auto tmp_AxB  = MDFT::Impl::cross_product(A, B);

        for (int i = 0; i < 3; ++i) {
          AxEx(i) = tmp_AxEx[i];
          AxEy(i) = tmp_AxEy[i];
          AxEz(i) = tmp_AxEz[i];
          AxB(i)  = tmp_AxB[i];
        }
      });

  Kokkos::fence();

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(execution_space(), AxEx, AxEx_ref, epsilon));
  EXPECT_TRUE(allclose(execution_space(), AxEy, AxEy_ref, epsilon));
  EXPECT_TRUE(allclose(execution_space(), AxEz, AxEz_ref, epsilon));
  EXPECT_TRUE(allclose(execution_space(), AxB, AxB_ref, epsilon));
}

template <typename T>
void test_norm2() {
  using ViewType    = Kokkos::View<int, execution_space>;
  using Array1DType = Kokkos::Array<T, 1>;
  using Array2DType = Kokkos::Array<T, 2>;
  using Array3DType = Kokkos::Array<T, 3>;
  Array1DType A1{1.5};
  Array2DType A2{1.2, 2.5};
  Array3DType A3{0.7, 2.8, 0.3};

  ViewType error1("error1"), error2("error2"), error3("error3");

  // Set reference
  T norm1_ref = std::sqrt(A1[0] * A1[0]);
  T norm2_ref = std::sqrt(A2[0] * A2[0] + A2[1] * A2[1]);
  T norm3_ref = std::sqrt(A3[0] * A3[0] + A3[1] * A3[1] + A3[2] * A3[2]);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        if (Kokkos::abs(MDFT::Impl::norm2(A1) - norm1_ref) > epsilon)
          Kokkos::atomic_store(error1.data(), 1);
        if (Kokkos::abs(MDFT::Impl::norm2(A2) - norm2_ref) > epsilon)
          Kokkos::atomic_store(error2.data(), 1);
        if (Kokkos::abs(MDFT::Impl::norm2(A3) - norm3_ref) > epsilon)
          Kokkos::atomic_store(error3.data(), 1);
      });

  auto h_error1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error1);
  auto h_error2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error2);
  auto h_error3 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, error3);

  ASSERT_EQ(h_error1(), 0);
  ASSERT_EQ(h_error2(), 0);
  ASSERT_EQ(h_error3(), 0);
}

template <typename T>
void test_L2normalize() {
  using ViewType  = Kokkos::View<T*, execution_space>;
  using ArrayType = Kokkos::Array<T, 3>;
  ArrayType A{1.0, 2.0, 3.0}, B{4.0, 5.0, 6.0};
  ArrayType Ex{1.0, 0.0, 0.0}, Ey{0.0, 1.0, 0.0}, Ez{0.0, 0.0, 1.0};

  ViewType A_norm("A_norm", 3), B_norm("B_norm", 3), Ex_norm("Ex_norm", 3),
      Ey_norm("Ey_norm", 3), Ez_norm("Ez_norm", 3);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        auto tmp_A_norm  = MDFT::Impl::L2normalize(A);
        auto tmp_B_norm  = MDFT::Impl::L2normalize(B);
        auto tmp_Ex_norm = MDFT::Impl::L2normalize(Ex);
        auto tmp_Ey_norm = MDFT::Impl::L2normalize(Ey);
        auto tmp_Ez_norm = MDFT::Impl::L2normalize(Ez);

        for (int i = 0; i < 3; ++i) {
          A_norm(i)  = tmp_A_norm[i];
          B_norm(i)  = tmp_B_norm[i];
          Ex_norm(i) = tmp_Ex_norm[i];
          Ey_norm(i) = tmp_Ey_norm[i];
          Ez_norm(i) = tmp_Ez_norm[i];
        }
      });

  Kokkos::fence();

  auto h_A_norm =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A_norm);
  auto h_B_norm =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, B_norm);
  auto h_Ex_norm =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Ex_norm);
  auto h_Ey_norm =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Ey_norm);
  auto h_Ez_norm =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Ez_norm);

  T sum_A = 0.0, sum_B = 0.0, sum_Ex = 0.0, sum_Ey = 0.0, sum_Ez = 0.0;
  for (int i = 0; i < 3; ++i) {
    sum_A += h_A_norm(i) * h_A_norm(i);
    sum_B += h_B_norm(i) * h_B_norm(i);
    sum_Ex += h_Ex_norm(i) * h_Ex_norm(i);
    sum_Ey += h_Ey_norm(i) * h_Ey_norm(i);
    sum_Ez += h_Ez_norm(i) * h_Ez_norm(i);
  }

  // Norm should be 1
  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_LT(Kokkos::abs(Kokkos::sqrt(sum_A) - 1.0), epsilon);
  EXPECT_LT(Kokkos::abs(Kokkos::sqrt(sum_B) - 1.0), epsilon);
  EXPECT_LT(Kokkos::abs(Kokkos::sqrt(sum_Ex) - 1.0), epsilon);
  EXPECT_LT(Kokkos::abs(Kokkos::sqrt(sum_Ey) - 1.0), epsilon);
  EXPECT_LT(Kokkos::abs(Kokkos::sqrt(sum_Ez) - 1.0), epsilon);
}

template <typename T>
void test_prevent_underflow() {
  using ViewType        = Kokkos::View<T, execution_space>;
  using ComplexViewType = Kokkos::View<Kokkos::complex<T>, execution_space>;
  ViewType x("x"), x2("x2"), x_ref("x_ref"), x2_ref("x2_ref");
  ComplexViewType z("z"), z2("z2"), z_ref("z_ref"), z2_ref("z2_ref");

  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  auto h_x2    = Kokkos::create_mirror_view(x2);

  auto h_z_ref = Kokkos::create_mirror_view(z_ref);
  auto h_z2    = Kokkos::create_mirror_view(z2);

  h_x_ref() = 1.0;       // Big value
  h_x2()    = 1.0e-100;  // Small value

  h_z_ref() = Kokkos::complex<T>(2.0, 3.0);           // Big value
  h_z2()    = Kokkos::complex<T>(1.0e-100, 1.0e-50);  // Small value

  Kokkos::deep_copy(x, h_x_ref);
  Kokkos::deep_copy(x2, h_x2);

  Kokkos::deep_copy(z, h_z_ref);
  Kokkos::deep_copy(z2, h_z2);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<execution_space, Kokkos::IndexType<int>>{0, 1},
      KOKKOS_LAMBDA(int) {
        // Real number
        x()  = MDFT::Impl::prevent_underflow(x(), epsilon);
        x2() = MDFT::Impl::prevent_underflow(x2(), epsilon);

        // Complex number
        z()  = MDFT::Impl::prevent_underflow(z(), epsilon);
        z2() = MDFT::Impl::prevent_underflow(z2(), epsilon);
      });

  Kokkos::fence();

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, z);

  // Norm should be 1
  EXPECT_LT(Kokkos::abs(h_x() - h_x_ref()), epsilon);
  EXPECT_LT(Kokkos::abs(h_x2() - 0.0), epsilon);

  EXPECT_LT(Kokkos::abs(h_z().real() - h_z_ref().real()), epsilon);
  EXPECT_LT(Kokkos::abs(h_z().imag() - h_z_ref().imag()), epsilon);
  EXPECT_LT(Kokkos::abs(h_z2().real() - 0.0), epsilon);
  EXPECT_LT(Kokkos::abs(h_z2().imag() - 0.0), epsilon);
}

template <typename T>
void test_chop() {
  T x0 = 1.0, x1 = 1.0e-9, x2 = 1.0e-12;

  // With default delta
  T x0_ref = 1.0, x1_ref = 1.0e-9, x2_ref = 0.0;

  // With custom delta of 1.0e-8
  T x0_ref2 = 1.0, x1_ref2 = 0.0, x2_ref2 = 0.0;

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x0) - x0_ref), epsilon);
  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x1) - x1_ref), epsilon);
  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x2) - x2_ref), epsilon);

  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x0, 1.0e-8) - x0_ref2), epsilon);
  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x1, 1.0e-8) - x1_ref2), epsilon);
  EXPECT_LT(Kokkos::abs(MDFT::Impl::chop(x2, 1.0e-8) - x2_ref2), epsilon);
}

template <typename T>
void test_inv_index(T n) {
  // Check if do not crash
  for (T i = 0; i < n; ++i) {
    int result = MDFT::Impl::inv_index(i, n);
    EXPECT_TRUE(result >= 0 && result < n);
  }

  // if n is even (used in MDFT)
  for (T i = 0; i < n; ++i) {
    int result     = MDFT::Impl::inv_index(i, n);
    int result_ref = i == 0 ? 0 : n - i;
    EXPECT_EQ(result, result_ref);
  }
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

TYPED_TEST(TestMathUtils, CrossProduct) {
  using float_type = typename TestFixture::float_type;
  test_cross_product<float_type>();
}

TYPED_TEST(TestMathUtils, Norm2) {
  using float_type = typename TestFixture::float_type;
  test_norm2<float_type>();
}

TYPED_TEST(TestMathUtils, L2normalize) {
  using float_type = typename TestFixture::float_type;
  test_L2normalize<float_type>();
}

TYPED_TEST(TestMathUtils, PreventUnderflow) {
  using float_type = typename TestFixture::float_type;
  test_prevent_underflow<float_type>();
}

TYPED_TEST(TestMathUtils, Chop) {
  using float_type = typename TestFixture::float_type;
  if constexpr (std::is_same_v<float_type, double>) {
    test_chop<float_type>();
  }
}

TYPED_TEST(TestMathUtilsIntOp, InvIndex) {
  using int_type = typename TestFixture::int_type;
  for (auto n : this->m_all_sizes) {
    test_inv_index<int_type>(static_cast<int_type>(n));
  }
}

}  // namespace
