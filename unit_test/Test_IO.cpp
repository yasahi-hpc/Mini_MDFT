#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <filesystem>
#include "IO/MDFT_IO_Utils.hpp"
#include "IO/MDFT_ReadCLuc.hpp"
#include "Test_Utils.hpp"

namespace {

struct TestIOUtils : public ::testing::Test {
  std::vector<std::string> m_all_files = {
      "ck_tip3p012-013_khi_max5", "tip3p-ck_nonzero_nmax1_ml",
      "tip3p-ck_nonzero_nmax5_ml", "ck_tip3p02-023_khi_max5",
      "tip3p-ck_nonzero_nmax3_ml"};

  // Executed from build/unit_test
  std::string m_file_path = "../../data/dcf/tip3p";
};

TEST_F(TestIOUtils, IsFileExists) {
  for (auto file : this->m_all_files) {
    std::string file_path = m_file_path + "/" + file;
    EXPECT_TRUE(MDFT::IO::Impl::is_file_exists(file_path));

    std::string wrong_file_path = m_file_path + "/" + file + "wrong";
    EXPECT_FALSE(MDFT::IO::Impl::is_file_exists(wrong_file_path));
  }
}

TEST_F(TestIOUtils, LineCount) {
  for (auto file : this->m_all_files) {
    std::string file_path = m_file_path + "/" + file;
    if (file == "tip3p-ck_nonzero_nmax1_ml") {
      // I do not know why this file is empty
      EXPECT_EQ(MDFT::IO::Impl::count_lines_in_file(file_path), 0);
    } else {
      EXPECT_GT(MDFT::IO::Impl::count_lines_in_file(file_path), 0);
    }

    std::string wrong_file_path = m_file_path + "/" + file + "wrong";
    EXPECT_EQ(MDFT::IO::Impl::count_lines_in_file(wrong_file_path), -1);
  }
}

TEST_F(TestIOUtils, LucDataInit) {
  for (auto file : this->m_all_files) {
    std::string file_path = m_file_path + "/" + file;
    ASSERT_NO_THROW(({ MDFT::IO::LucData<double> luc_data(file_path); }));

    std::string wrong_file_path = m_file_path + "/" + file + "wrong";
    ASSERT_THROW(({ MDFT::IO::LucData<double> luc_data(wrong_file_path); }),
                 std::runtime_error);
  }
}

}  // namespace
