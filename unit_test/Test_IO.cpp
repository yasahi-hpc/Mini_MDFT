#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <filesystem>
#include "IO/MDFT_IO_Utils.hpp"
#include "IO/MDFT_ReadCLuc.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

struct TestIOUtils : public ::testing::Test {
  std::vector<std::string> m_all_files = {
      "ck_tip3p012-013_khi_max5", "tip3p-ck_nonzero_nmax1_ml",
      "tip3p-ck_nonzero_nmax5_ml", "ck_tip3p02-023_khi_max5",
      "tip3p-ck_nonzero_nmax3_ml"};

  // Executed from build/unit_test
  std::string m_file_path = "../../data/dcf/tip3p";
};

template <typename T>
struct TestLucData : public ::testing::Test {
  using float_type                     = T;
  std::vector<std::string> m_all_files = {
      "ck_tip3p012-013_khi_max5", "tip3p-ck_nonzero_nmax5_ml",
      "ck_tip3p02-023_khi_max5", "tip3p-ck_nonzero_nmax3_ml"};

  // Executed from build/unit_test
  std::string m_file_path = "../../data/dcf/tip3p";
};

TYPED_TEST_SUITE(TestLucData, float_types);

TEST_F(TestIOUtils, IsFileExists) {
  for (auto file : this->m_all_files) {
    std::string file_path = this->m_file_path + "/" + file;
    EXPECT_TRUE(MDFT::IO::Impl::is_file_exists(file_path));

    std::string wrong_file_path = this->m_file_path + "/" + file + "wrong";
    EXPECT_FALSE(MDFT::IO::Impl::is_file_exists(wrong_file_path));
  }
}

TEST_F(TestIOUtils, LineCount) {
  for (auto file : this->m_all_files) {
    std::string file_path = this->m_file_path + "/" + file;
    if (file == "tip3p-ck_nonzero_nmax1_ml") {
      // I do not know why this file is empty
      EXPECT_EQ(MDFT::IO::Impl::count_lines_in_file(file_path), 0);
    } else {
      EXPECT_GT(MDFT::IO::Impl::count_lines_in_file(file_path), 0);
    }

    std::string wrong_file_path = this->m_file_path + "/" + file + "wrong";
    EXPECT_EQ(MDFT::IO::Impl::count_lines_in_file(wrong_file_path), -1);
  }
}

// Testing ReadCLuc related functions
TYPED_TEST(TestLucData, LucDataInit) {
  using float_type = typename TestFixture::float_type;
  int mmax = 5, molrotsymorder = 2, np = 252;
  float_type qmaxwanted = 11.6083159310990;
  MDFT::AngularGrid<execution_space, float_type> angular_grid(mmax,
                                                              molrotsymorder);
  for (auto file : this->m_all_files) {
    std::string file_path = this->m_file_path + "/" + file;
    ASSERT_NO_THROW(({
      MDFT::IO::LucData<execution_space, float_type> luc_data(
          file_path, angular_grid, np, qmaxwanted);
    }));

    std::string wrong_file_path = this->m_file_path + "/" + file + "wrong";
    ASSERT_THROW(({
                   MDFT::IO::LucData<execution_space, float_type> luc_data(
                       wrong_file_path, angular_grid, np, qmaxwanted);
                 }),
                 std::runtime_error);
  }
}

TYPED_TEST(TestLucData, deltaAbscissa) {
  using float_type = typename TestFixture::float_type;
  int mmax = 5, molrotsymorder = 2, np = 252;
  float_type qmaxwanted = 11.6083159310990;
  MDFT::AngularGrid<execution_space, float_type> angular_grid(mmax,
                                                              molrotsymorder);
  std::string file_path = this->m_file_path + "/" + "tip3p-ck_nonzero_nmax3_ml";

  MDFT::IO::LucData<execution_space, float_type> luc_data(
      file_path, angular_grid, np, qmaxwanted);

  // Check results
  float_type epsilon = std::numeric_limits<float_type>::epsilon() * 100;

  int ref_nq               = 191;
  float_type ref_dAbscissa = 6.135923151542601e-2;
  ASSERT_EQ(luc_data.m_nq, ref_nq);
  EXPECT_LT(Kokkos::abs(luc_data.m_dq - ref_dAbscissa), epsilon);
}

TYPED_TEST(TestLucData, readLuc) {
  using float_type = typename TestFixture::float_type;
  int mmax = 5, molrotsymorder = 2, np = 252, nq = 191;
  double qmaxwanted         = 11.6083159310990;
  float_type tmp_qmaxwanted = static_cast<float_type>(qmaxwanted);
  MDFT::AngularGrid<execution_space, float_type> angular_grid(mmax,
                                                              molrotsymorder);
  std::string file_path = this->m_file_path + "/" + "tip3p-ck_nonzero_nmax3_ml";

  MDFT::IO::LucData<execution_space, float_type> luc_data(
      file_path, angular_grid, np, tmp_qmaxwanted);

  // Only checks first and last elements
  auto h_m =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luc_data.m_m);
  auto h_n =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luc_data.m_n);
  auto h_mu =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luc_data.m_mu);
  auto h_nu =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luc_data.m_nu);
  auto h_khi =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), luc_data.m_khi);

  ASSERT_EQ(h_m(0), 3);
  ASSERT_EQ(h_m(h_m.size() - 1), 0);
  ASSERT_EQ(h_n(0), 3);
  ASSERT_EQ(h_n(h_n.size() - 1), 0);
  ASSERT_EQ(h_mu(0), 2);
  ASSERT_EQ(h_mu(h_mu.size() - 1), 0);
  ASSERT_EQ(h_nu(0), 2);
  ASSERT_EQ(h_nu(h_nu.size() - 1), 0);
  ASSERT_EQ(h_khi(0), 3);
  ASSERT_EQ(h_khi(h_khi.size() - 1), 0);

  // Check cmnmunukhi
  auto epsilon      = std::numeric_limits<float_type>::epsilon() * 100;
  auto h_cmnmunukhi = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), luc_data.m_cmnmunukhi);

  // (ip, iq) = (0, 0)
  int ip = 0, iq = 0;
  Kokkos::complex<float_type> ref_cmnmunukhi_0_0(0.158540502400000, 0.0);
  EXPECT_LT(Kokkos::abs(h_cmnmunukhi(ip, iq) - ref_cmnmunukhi_0_0), epsilon);

  // (ip, iq) = (np-1, 0)
  ip = np - 1;
  Kokkos::complex<float_type> ref_cmnmunukhi_np_0(-11.9370828317000, 0.0);
  EXPECT_LT(Kokkos::abs(h_cmnmunukhi(ip, iq) - ref_cmnmunukhi_np_0), epsilon);

  // (ip, iq) = (0, nq-1)
  ip = 0;
  iq = nq - 1;
  Kokkos::complex<float_type> ref_cmnmunukhi_0_nq(9.241650000000000e-5,
                                                  -8.986700000000000e-6);
  EXPECT_LT(Kokkos::abs(h_cmnmunukhi(ip, iq) - ref_cmnmunukhi_0_nq), epsilon);

  // (ip, iq) = (np-1, nq-1)
  ip = np - 1;
  Kokkos::complex<float_type> ref_cmnmunukhi_np_nq(1.76792295e-2, 0.0);
  EXPECT_LT(Kokkos::abs(h_cmnmunukhi(ip, iq) - ref_cmnmunukhi_np_nq), epsilon);
}

}  // namespace
