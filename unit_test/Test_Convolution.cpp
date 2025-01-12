#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include "MDFT_Grid.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "MDFT_Convolution.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestConvolution : public ::testing::Test {
  using float_type             = T;
  using scalar_array_type      = Kokkos::Array<T, 3>;
  using int_array_type         = Kokkos::Array<int, 3>;
  std::vector<int> m_all_sizes = {5, 10};

  // Related to Luc data
  // Executed from build/unit_test
  std::string m_file_path = "../../data/dcf/tip3p";
  std::string m_filename  = "tip3p-ck_nonzero_nmax3_ml";

  const int m_np_luc = 252;
};

TYPED_TEST_SUITE(TestConvolution, float_types);

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_convolution_init(int n, std::string file_path, int np_luc) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);

  ASSERT_NO_THROW(({
    MDFT::Convolution<execution_space, T> conv(file_path, grid, angular_grid,
                                               map, np_luc);
  }));

  // Check the conv is initialized correctly
  MDFT::Convolution<execution_space, T> conv(file_path, grid, angular_grid, map,
                                             np_luc);

  // Check if gamma_p_map is correct
  // These gamma_p must be accessed only once, and thus
  // gamma_p_isok must be 0 before set as 1
  int nx = n, ny = n, nz = n;
  using HostIntView3DType = Kokkos::View<int***, Kokkos::HostSpace>;
  HostIntView3DType h_gamma_p_isok("gamma_p_isok", nx, ny, nz);
  auto h_gamma_p_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           conv.gamma_p_map());
  int N              = h_gamma_p_map.extent(0);

  for (int idx = 0; idx < N; ++idx) {
    const int ix = h_gamma_p_map(idx, 0), iy = h_gamma_p_map(idx, 1),
              iz    = h_gamma_p_map(idx, 2);
    const int ix_mq = MDFT::Impl::inv_index(ix, nx);
    const int iy_mq = MDFT::Impl::inv_index(iy, ny);
    const int iz_mq = MDFT::Impl::inv_index(iz, nz);

    EXPECT_FALSE(h_gamma_p_isok(ix, iy, iz));
    EXPECT_FALSE(h_gamma_p_isok(ix_mq, iy_mq, iz_mq));

    // Set the value to 1
    h_gamma_p_isok(ix, iy, iz)          = 1;
    h_gamma_p_isok(ix_mq, iy_mq, iz_mq) = 1;
  }
}

TYPED_TEST(TestConvolution, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  std::string file_path = this->m_file_path + "/" + this->m_filename;
  for (auto m : this->m_all_sizes) {
    test_convolution_init<float_type, int_array_type, scalar_array_type>(
        m, file_path, this->m_np_luc);
  }
}

}  // namespace
