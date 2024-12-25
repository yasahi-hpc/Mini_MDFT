#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>
#include <memory>
#include "MDFT_System.hpp"
#include "MDFT_Thermo.hpp"
#include "MDFT_Constants.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestThermo : public ::testing::Test {
  using float_type           = T;
  using ThermoType           = MDFT::Thermo<float_type>;
  std::string m_setting_file = "dft.json";

  ThermoType m_thermo;

  // Executed from build/unit_test
  std::string m_file_path = "../../input";

  virtual void SetUp() {
    float_type t = 298, p = 1;
    m_thermo.m_t    = t;
    m_thermo.m_p    = p;
    m_thermo.m_kbt  = MDFT::Constants::Boltz * MDFT::Constants::Navo * t * 1e-3;
    m_thermo.m_beta = 1.0 / m_thermo.m_kbt;
  }
};

TYPED_TEST_SUITE(TestThermo, float_types);

template <typename T>
void test_thermo_init(std::string setting_filename,
                      MDFT::Thermo<T> &ref_thermo) {
  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check values are correctly set
  MDFT::Settings<T> settings(setting_filename);
  MDFT::Thermo<T> thermo(settings);

  // Check values
  ASSERT_TRUE(Kokkos::abs(thermo.m_t - ref_thermo.m_t) < epsilon);
  ASSERT_TRUE(Kokkos::abs(thermo.m_p - ref_thermo.m_p) < epsilon);
  ASSERT_TRUE(Kokkos::abs(thermo.m_kbt - ref_thermo.m_kbt) < epsilon);
  ASSERT_TRUE(Kokkos::abs(thermo.m_beta - ref_thermo.m_beta) < epsilon);
}

TYPED_TEST(TestThermo, Initialization) {
  using float_type = typename TestFixture::float_type;

  test_thermo_init<float_type>(this->m_file_path + "/" + this->m_setting_file,
                               this->m_thermo);
}

}  // namespace
