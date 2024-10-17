#include <gtest/gtest.h>
#include "MDFT_Grid.hpp"

namespace {

using float_types = ::testing::Types<float, double>;

template <typename T>
struct TestSpatialGrid : public ::testing::Test {
  using float_type        = T;
  using scalar_array_type = Kokkos::Array<T, 3>;
  using int_array_type    = Kokkos::Array<int, 3>;
};

template <typename T>
struct TestAngularGrid : public ::testing::Test {
  using float_type        = T;
  using scalar_array_type = Kokkos::Array<T, 3>;
  using int_array_type    = Kokkos::Array<int, 3>;
};

TYPED_TEST_SUITE(TestSpatialGrid, float_types);
TYPED_TEST_SUITE(TestAngularGrid, float_types);

TYPED_TEST(TestSpatialGrid, NegativeMeshSize) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;

  ASSERT_THROW(
      {
        MDFT::SpatialGrid grid(int_array_type{10, 10, -10},
                               scalar_array_type{1, 1, 1});
      },
      std::runtime_error);
}

TYPED_TEST(TestSpatialGrid, NegativeMeshLength) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;
  ASSERT_THROW(
      {
        MDFT::SpatialGrid grid(int_array_type{10, 10, 10},
                               scalar_array_type{1, 1, -1});
      },
      std::runtime_error);
}

TYPED_TEST(TestSpatialGrid, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;
  ASSERT_NO_THROW({
    MDFT::SpatialGrid grid(int_array_type{10, 10, 10},
                           scalar_array_type{1, 1, 1});
  });

  int nx        = 10;
  float_type lx = 1.5;
  MDFT::SpatialGrid grid(int_array_type{nx, nx, nx},
                         scalar_array_type{lx, lx, lx});
  float_type ref_v  = lx * lx * lx;
  float_type dx     = lx / static_cast<float_type>(nx);
  float_type ref_dv = dx * dx * dx;

  EXPECT_EQ(grid.m_v, ref_v);
  EXPECT_EQ(grid.m_dv, ref_dv);
}

TYPED_TEST(TestAngularGrid, NegativeMeshSize) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;

  ASSERT_THROW(({
                 int mmax = -1, molrotsymorder = 10;
                 MDFT::AngularGrid<float_type> angular_grid(mmax,
                                                            molrotsymorder);
               }),
               std::runtime_error);

  ASSERT_THROW(({
                 int mmax = 1, molrotsymorder = -10;
                 MDFT::AngularGrid<float_type> angular_grid(mmax,
                                                            molrotsymorder);
               }),
               std::runtime_error);
}

}  // namespace
