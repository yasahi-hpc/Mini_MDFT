#include <gtest/gtest.h>
#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestGrid : public ::testing::Test {
  using float_type           = T;
  using scalar_array_type    = Kokkos::Array<T, 3>;
  using int_array_type       = Kokkos::Array<int, 3>;
  using SpatialGridType      = MDFT::SpatialGrid<execution_space, float_type>;
  using AngularGridType      = MDFT::AngularGrid<execution_space, float_type>;
  std::string m_setting_file = "dft.json";
  // Executed from build/unit_test
  std::string m_file_path = "../../input";

  int_array_type m_n_nodes_ref   = {64, 64, 64};
  scalar_array_type m_length_ref = {30, 30, 30};
  int m_mmax_ref                 = 5;
  int m_molrotsymorder_ref       = 2;
};

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

TYPED_TEST_SUITE(TestGrid, float_types);
TYPED_TEST_SUITE(TestSpatialGrid, float_types);
TYPED_TEST_SUITE(TestAngularGrid, float_types);

TYPED_TEST(TestGrid, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using spatial_grid_type = typename TestFixture::SpatialGridType;
  using angular_grid_type = typename TestFixture::AngularGridType;

  std::unique_ptr<spatial_grid_type> spatial_grid;
  std::unique_ptr<angular_grid_type> angular_grid;
  MDFT::Settings<float_type> settings(this->m_file_path + "/" +
                                      this->m_setting_file);

  ASSERT_NO_THROW(({ init_grid(settings, spatial_grid, angular_grid); }));

  // Check values are correctly set
  float_type epsilon = std::numeric_limits<float_type>::epsilon() * 100;
  for (std::size_t i = 0; i < 3; ++i) {
    ASSERT_EQ(spatial_grid->m_n_nodes[i], this->m_n_nodes_ref[i]);
    ASSERT_TRUE(Kokkos::abs(spatial_grid->m_length[i] - this->m_length_ref[i]) <
                epsilon);
  }
  ASSERT_EQ(angular_grid->m_mmax, this->m_mmax_ref);
  ASSERT_EQ(angular_grid->m_molrotsymorder, this->m_molrotsymorder_ref);
}

TYPED_TEST(TestSpatialGrid, NegativeMeshSize) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;

  ASSERT_THROW(({
                 MDFT::SpatialGrid<execution_space, float_type> grid(
                     int_array_type{10, 10, -10}, scalar_array_type{1, 1, 1});
               }),
               std::runtime_error);
}

TYPED_TEST(TestSpatialGrid, NegativeMeshLength) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;
  ASSERT_THROW(({
                 MDFT::SpatialGrid<execution_space, float_type> grid(
                     int_array_type{10, 10, 10}, scalar_array_type{1, 1, -1});
               }),
               std::runtime_error);
}

TYPED_TEST(TestSpatialGrid, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;
  using int_array_type    = typename TestFixture::int_array_type;
  ASSERT_NO_THROW(({
    MDFT::SpatialGrid<execution_space, float_type> grid(
        int_array_type{10, 10, 10}, scalar_array_type{1, 1, 1});
  }));

  int nx        = 10;
  float_type lx = 1.5;
  MDFT::SpatialGrid<execution_space, float_type> grid(
      int_array_type{nx, nx, nx}, scalar_array_type{lx, lx, lx});
  float_type ref_v  = lx * lx * lx;
  float_type dx     = lx / static_cast<float_type>(nx);
  float_type ref_dv = dx * dx * dx;

  EXPECT_EQ(grid.m_v, ref_v);
  EXPECT_EQ(grid.m_dv, ref_dv);
}

TYPED_TEST(TestAngularGrid, NegativeMeshSize) {
  using float_type = typename TestFixture::float_type;

  ASSERT_THROW(({
                 int mmax = -1, molrotsymorder = 10;
                 MDFT::AngularGrid<execution_space, float_type> angular_grid(
                     mmax, molrotsymorder);
               }),
               std::runtime_error);

  ASSERT_THROW(({
                 int mmax = 1, molrotsymorder = -10;
                 MDFT::AngularGrid<execution_space, float_type> angular_grid(
                     mmax, molrotsymorder);
               }),
               std::runtime_error);
}

TYPED_TEST(TestAngularGrid, Initialization) {
  using float_type = typename TestFixture::float_type;

  ASSERT_NO_THROW(({
    int mmax = 10, molrotsymorder = 10;
    MDFT::AngularGrid<execution_space, float_type> angular_grid(mmax,
                                                                molrotsymorder);
  }));
}

}  // namespace
