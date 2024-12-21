#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>
#include "MDFT_System.hpp"
#include "MDFT_Solute.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestSolute : public ::testing::Test {
  using float_type                     = T;
  using SiteType                       = MDFT::Site<float_type>;
  std::vector<std::string> m_all_files = {"solute.json"};
  std::vector<SiteType> m_sites;

  // Executed from build/unit_test
  std::string m_file_path = "../../input";

  virtual void SetUp() {
    m_sites.push_back(SiteType("I", 1.0, 2.93, 0.759,
                               Kokkos::Array<T, 3>({0.0, 0.0, 0.0}), 17));
  }
};

TYPED_TEST_SUITE(TestSolute, float_types);

template <typename T>
void test_solute_init(std::string filename, MDFT::Site<T> ref_site) {
  ASSERT_NO_THROW(({ MDFT::Solute<execution_space, T> solute(filename); }));

  // Check values are correctly set
  MDFT::Solute<execution_space, T> solute(filename);
  ASSERT_EQ(solute.m_nsite, 1);
  auto site = solute.m_site.at(0);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  ASSERT_EQ(site.m_name, ref_site.m_name);
  ASSERT_TRUE(Kokkos::abs(site.m_q - ref_site.m_q) < epsilon);
  ASSERT_TRUE(Kokkos::abs(site.m_sig - ref_site.m_sig) < epsilon);
  ASSERT_TRUE(Kokkos::abs(site.m_eps - ref_site.m_eps) < epsilon);
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(Kokkos::abs(site.m_r[i] - ref_site.m_r[i]) < epsilon);
  }
  ASSERT_EQ(site.m_z, ref_site.m_z);
}

TYPED_TEST(TestSolute, Initialization) {
  using float_type = typename TestFixture::float_type;
  for (int i = 0; i < this->m_all_files.size(); i++) {
    std::string file            = this->m_all_files[i];
    MDFT::Site<float_type> site = this->m_sites[i];
    test_solute_init<float_type>(this->m_file_path + "/" + file, site);
  }
}

}  // namespace
