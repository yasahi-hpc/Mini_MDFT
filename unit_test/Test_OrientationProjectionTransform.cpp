#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <Kokkos_Random.hpp>
#include "MDFT_Grid.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestOrientationProjectionMap : public ::testing::Test {
  using float_type             = T;
  using scalar_array_type      = Kokkos::Array<T, 3>;
  using int_array_type         = Kokkos::Array<int, 3>;
  std::vector<int> m_all_sizes = {5, 10};
};

template <typename T>
struct TestOrientationProjection : public ::testing::Test {
  using float_type             = T;
  using scalar_array_type      = Kokkos::Array<T, 3>;
  using int_array_type         = Kokkos::Array<int, 3>;
  std::vector<int> m_all_sizes = {5, 10};
};

TYPED_TEST_SUITE(TestOrientationProjectionMap, float_types);
TYPED_TEST_SUITE(TestOrientationProjection, float_types);

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_orientation_projection_map_init(int n) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);

  ASSERT_NO_THROW(({
    MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);
  }));

  // Add checks for the indices order
  // mapping is different from fortran version
}

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_orientation_projection_init(int n) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);

  ASSERT_NO_THROW(({
    MDFT::OrientationProjectionTransform<execution_space, T> opt(
        grid, angular_grid, map);
  }));
}

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_orientation_projection_identity(int n) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);
  MDFT::OrientationProjectionTransform<execution_space, T> opt(
      grid, angular_grid, map);

  using OViewType = Kokkos::View<T****, execution_space>;
  using PViewType = Kokkos::View<Kokkos::complex<T>****, execution_space>;

  auto [nx, ny, nz] = grid.m_n_nodes;
  auto ntheta = angular_grid.m_ntheta, nphi = angular_grid.m_nphi,
       npsi = angular_grid.m_npsi;
  auto np   = angular_grid.m_np;

  OViewType o("o", nx * ny * nz, ntheta, nphi, npsi),
      o_inv("o_inv", nx * ny * nz, ntheta, nphi, npsi),
      o_ref("o_ref", nx * ny * nz, ntheta, nphi, npsi);
  PViewType p("p", np, nx, ny, nz);

  // Initialize o with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(o, random_pool, 1.0);

  Kokkos::deep_copy(o_ref, o);

  opt.angl2proj(o, p);
  opt.proj2angl(p, o_inv);

  // FIXME Does not work for now, probably something is wrong
  // T epsilon = std::numeric_limits<T>::epsilon() * 100;
  // EXPECT_TRUE(allclose(execution_space(), o_inv, o_ref, epsilon));
}

TYPED_TEST(TestOrientationProjectionMap, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  for (auto m : this->m_all_sizes) {
    test_orientation_projection_map_init<float_type, int_array_type,
                                         scalar_array_type>(m);
  }
}

TYPED_TEST(TestOrientationProjection, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  for (auto m : this->m_all_sizes) {
    test_orientation_projection_init<float_type, int_array_type,
                                     scalar_array_type>(m);
  }
}

TYPED_TEST(TestOrientationProjection, Identity) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  for (auto m : this->m_all_sizes) {
    test_orientation_projection_identity<float_type, int_array_type,
                                         scalar_array_type>(m);
  }
}

}  // namespace
