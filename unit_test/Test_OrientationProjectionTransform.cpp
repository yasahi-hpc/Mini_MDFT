#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include "MDFT_Grid.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestOrientationProjection : public ::testing::Test {
  using float_type             = T;
  using scalar_array_type      = Kokkos::Array<T, 3>;
  using int_array_type         = Kokkos::Array<int, 3>;
  std::vector<int> m_all_sizes = {5, 10};
};

TYPED_TEST_SUITE(TestOrientationProjection, float_types);

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_orientation_projection_init(int n) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);

  ASSERT_NO_THROW(({
    MDFT::OrientationProjectionTransform<execution_space, T> opt(grid,
                                                                 angular_grid);
  }));
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

}  // namespace
