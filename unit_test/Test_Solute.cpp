#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>
#include <memory>
#include "MDFT_System.hpp"
#include "MDFT_Solute.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

template <typename T>
struct TestSolute : public ::testing::Test {
  using float_type           = T;
  using scalar_array_type    = Kokkos::Array<T, 3>;
  using int_array_type       = Kokkos::Array<int, 3>;
  using SettingsType         = MDFT::Settings<float_type>;
  using SiteType             = MDFT::Site<float_type>;
  std::string m_setting_file = "dft.json";
  std::string m_solute_file  = "solute.json";

  std::unique_ptr<SettingsType> m_settings;
  std::unique_ptr<SiteType> m_site;

  // Executed from build/unit_test
  std::string m_file_path = "../../input";

  virtual void SetUp() {
    m_settings = std::make_unique<SettingsType>(
        "spce", 1, Kokkos::Array<int, 3>({64, 64, 64}),
        Kokkos::Array<T, 3>({30, 30, 30}), 5, 35, 1.0, 1.0, true, false, 15,
        298.0, true);

    m_site = std::make_unique<SiteType>(
        "I", 1.0 * 1.0, 2.93, 0.759, Kokkos::Array<T, 3>({0.0, 0.0, 0.0}), 17);
  }
};

TYPED_TEST_SUITE(TestSolute, float_types);

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_solute_init(int n, std::string setting_filename,
                      std::string solute_filename,
                      MDFT::Settings<T> &ref_setting, MDFT::Site<T> &ref_site) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  ASSERT_NO_THROW(({
    MDFT::Solute<execution_space, T> solute(grid, setting_filename,
                                            solute_filename);
  }));

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check values are correctly set
  MDFT::Solute<execution_space, T> solute(grid, setting_filename,
                                          solute_filename);

  // Check settings
  auto setting = *solute.m_settings;
  ASSERT_EQ(setting.m_solvent, ref_setting.m_solvent);
  ASSERT_EQ(setting.m_nb_solvent, ref_setting.m_nb_solvent);
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(setting.m_boxnod[i], ref_setting.m_boxnod[i]);
    ASSERT_TRUE(Kokkos::abs(setting.m_boxlen[i] - ref_setting.m_boxlen[i]) <
                epsilon);
  }

  ASSERT_EQ(setting.m_mmax, ref_setting.m_mmax);
  ASSERT_EQ(setting.m_maximum_iteration_nbr,
            ref_setting.m_maximum_iteration_nbr);
  ASSERT_TRUE(Kokkos::abs(setting.m_precision_factor -
                          ref_setting.m_precision_factor) < epsilon);
  ASSERT_TRUE(Kokkos::abs(setting.m_solute_charges_scale_factor -
                          ref_setting.m_solute_charges_scale_factor) < epsilon);
  ASSERT_TRUE(Kokkos::abs(setting.m_hard_sphere_solute -
                          ref_setting.m_hard_sphere_solute) < epsilon);
  ASSERT_TRUE(Kokkos::abs(setting.m_hard_sphere_solute_radius -
                          ref_setting.m_hard_sphere_solute_radius) < epsilon);
  ASSERT_TRUE(Kokkos::abs(setting.m_temperature - ref_setting.m_temperature) <
              epsilon);
  ASSERT_EQ(setting.m_restart, ref_setting.m_restart);

  // Check solute
  ASSERT_EQ(solute.m_nsite, 1);
  auto site = solute.m_site.at(0);

  ASSERT_EQ(site.m_name, ref_site.m_name);
  ASSERT_TRUE(Kokkos::abs(site.m_q - ref_site.m_q) < epsilon);
  ASSERT_TRUE(Kokkos::abs(site.m_sig - ref_site.m_sig) < epsilon);
  ASSERT_TRUE(Kokkos::abs(site.m_eps - ref_site.m_eps) < epsilon);
  for (int i = 0; i < 3; i++) {
    auto m_r = setting.m_translate_solute_to_center
                   ? ref_site.m_r[i] + grid.m_length[i] / 2.0
                   : ref_site.m_r[i];
    ASSERT_TRUE(Kokkos::abs(site.m_r[i] - m_r) < epsilon);
  }
  ASSERT_EQ(site.m_z, ref_site.m_z);
}

TYPED_TEST(TestSolute, Initialization) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  test_solute_init<float_type, int_array_type, scalar_array_type>(
      10, this->m_file_path + "/" + this->m_setting_file,
      this->m_file_path + "/" + this->m_solute_file, *(this->m_settings),
      *(this->m_site));
}

}  // namespace
