#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <Kokkos_Random.hpp>
#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Solvent.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;
using float_types     = ::testing::Types<float, double>;

struct TestSolventConstants : public ::testing::Test {
  // Solvent names
  std::vector<std::string> m_allowed_solvent_names = {
      "spce", "spce-h", "spce-m", "tip3p", "tip3p-m"};
  std::vector<std::string> m_solvent_names = {"spce",  "spce-h",  "spce-m",
                                              "tip3p", "tip3p-m", "unknown"};
};

template <typename T>
struct TestSolvent : public ::testing::Test {
  using float_type           = T;
  using scalar_array_type    = Kokkos::Array<T, 3>;
  using int_array_type       = Kokkos::Array<int, 6>;
  using SettingsType         = MDFT::Settings<float_type>;
  using SiteType             = MDFT::Site<float_type>;
  using SolventType          = MDFT::Solvent<execution_space, float_type>;
  std::string m_setting_file = "dft.json";

  SolventType m_solvent_ref;
  std::vector<int> m_all_sizes = {5, 10};

  // Executed from build/unit_test
  std::string m_file_path = "../../input";

  virtual void SetUp() {
    std::string name     = "spce";
    float_type hs_radius = 0;
    int nsite            = 3;
    int molrotsymorder   = 2;
    std::vector<float_type> q({-0.8476, 0.4238, 0.4238});
    std::vector<float_type> sig({3.166, 0., 0.});
    std::vector<float_type> eps({0.65, 0., 0.});
    std::vector<float_type> r0({0.0, 0.0, 0.0});
    std::vector<float_type> r1({0.816495, 0.0, 0.5773525});
    std::vector<float_type> r2({-0.816495, 0.0, 0.5773525});
    std::vector<int> Z({8, 1, 1});
    float_type n0 = 0.0332891;
    float_type rho0 =
        n0 / (8 * MDFT::Constants::pi * MDFT::Constants::pi / molrotsymorder);
    float_type relativePermittivity = 71;
    int_array_type npluc({1, 6, 75, 252, 877, 2002});
    int n_line_cfile = 1024;

    m_solvent_ref.init(name, hs_radius, nsite, molrotsymorder, q, sig, eps, r0,
                       r1, r2, Z, n0, rho0, relativePermittivity, npluc,
                       n_line_cfile);
  }
};

TYPED_TEST_SUITE(TestSolvent, float_types);

template <typename T, typename SolventType>
void test_solvent_init(int n, std::string setting_filename,
                       SolventType& ref_solvent) {
  MDFT::Settings<T> settings(setting_filename);
  MDFT::SpatialGrid<execution_space, T> grid(Kokkos::Array<int, 3>{n, n, n},
                                             Kokkos::Array<T, 3>{1, 1, 1});
  int mmax = settings.m_mmax, molrotsymorder = 2;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::Thermo<T> thermo(settings);

  ASSERT_NO_THROW(({
    MDFT::Solvents<execution_space, T> solvents(grid, angular_grid, settings,
                                                thermo);
  }));

  // Check values are correctly set
  MDFT::Solvents<execution_space, T> solvents(grid, angular_grid, settings,
                                              thermo);
  ASSERT_EQ(solvents.m_solvents.at(0).m_name, ref_solvent.m_name);
  ASSERT_EQ(solvents.m_solvents.at(0).m_molrotsymorder,
            ref_solvent.m_molrotsymorder);
  ASSERT_EQ(solvents.m_solvents.at(0).m_nsite, ref_solvent.m_nsite);
  ASSERT_EQ(solvents.m_solvents.at(0).m_n0, ref_solvent.m_n0);
  ASSERT_EQ(solvents.m_solvents.at(0).m_rho0, ref_solvent.m_rho0);
  ASSERT_EQ(solvents.m_solvents.at(0).m_relativePermittivity,
            ref_solvent.m_relativePermittivity);
  for (int i = 0; i < 6; i++) {
    ASSERT_EQ(solvents.m_solvents.at(0).m_npluc[i], ref_solvent.m_npluc[i]);
  }
  ASSERT_EQ(solvents.m_solvents.at(0).m_n_line_cfile,
            ref_solvent.m_n_line_cfile);

  // Check monopole, dipole, quadrupole, octupole and hexadecapole
  T epsilon    = std::numeric_limits<T>::epsilon() * 100;
  auto solvent = solvents.m_solvents.at(0);
  EXPECT_EQ(solvent.m_monopole, ref_solvent.m_monopole);
  EXPECT_TRUE(allclose(execution_space(), solvent.m_dipole,
                       ref_solvent.m_dipole, epsilon));
  EXPECT_TRUE(allclose(execution_space(), solvent.m_quadrupole,
                       ref_solvent.m_quadrupole, epsilon));
  EXPECT_TRUE(allclose(execution_space(), solvent.m_octupole,
                       ref_solvent.m_octupole, epsilon));
  EXPECT_TRUE(allclose(execution_space(), solvent.m_hexadecupole,
                       ref_solvent.m_hexadecupole, epsilon));

  // Check xi
  using ViewType  = Kokkos::View<T****, execution_space>;
  ViewType xi_ref = ViewType("xi_ref", n, n, n, angular_grid.m_no);
  Kokkos::deep_copy(xi_ref, 1.0);  // Initialize xi_ref with 1.0

  auto xi = solvents.m_solvents.at(0).m_xi;
  for (std::size_t i = 0; i < xi.rank(); i++) {
    EXPECT_EQ(xi.extent(i), xi_ref.extent(i));
  }

  EXPECT_TRUE(allclose(execution_space(), xi, xi_ref, epsilon));
}

template <typename T>
void test_get_delta_rho() {
  using ViewType = Kokkos::View<T****, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  ViewType xi("xi", n0, n1, n2, n3), delta_rho("delta_rho", n0, n1, n2, n3),
      delta_rho_ref("delta_rho_ref", n0, n1, n2, n3);

  T rho0 = 0.3;

  // Initialize xi with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(xi, random_pool, 1.0);

  // Reference
  auto h_xi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
  auto h_delta_rho_ref = Kokkos::create_mirror_view(delta_rho_ref);
  for (int i0 = 0; i0 < n0; i0++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i3 = 0; i3 < n3; i3++) {
          // delta_rho = rho0 * (xi^2 - 1)
          h_delta_rho_ref(i0, i1, i2, i3) =
              rho0 * (h_xi(i0, i1, i2, i3) * h_xi(i0, i1, i2, i3) - 1.0);
        }
      }
    }
  }
  Kokkos::deep_copy(delta_rho_ref, h_delta_rho_ref);

  // Compute delta_rho
  execution_space exec_space;
  MDFT::get_delta_rho(exec_space, xi, delta_rho, rho0);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(exec_space, delta_rho, delta_rho_ref, epsilon));
}

template <typename T>
void test_get_delta_f() {
  using View1DType = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
  using View4DType = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5;
  View1DType w("w", n3);
  View4DType xi("xi", n0, n1, n2, n3), vexc("vexc", n0, n1, n2, n3),
      delta_f("delta_f", n0, n1, n2, n3),
      delta_f_ref("delta_f_ref", n0, n1, n2, n3);
  T ff_ref = 0;
  T rho0 = 0.3, prefactor = 0.334;

  // Initialize xi with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(xi, random_pool, 1.0);
  Kokkos::fill_random(vexc, random_pool, 1.0);
  Kokkos::fill_random(w, random_pool, 1.0);

  // Reference
  auto h_xi   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
  auto h_vexc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vexc);
  auto h_w    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
  auto h_delta_f_ref = Kokkos::create_mirror_view(delta_f_ref);
  for (int i0 = 0; i0 < n0; i0++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i3 = 0; i3 < n3; i3++) {
          h_delta_f_ref(i0, i1, i2, i3) =
              2.0 * rho0 * h_xi(i0, i1, i2, i3) * h_vexc(i0, i1, i2, i3);

          ff_ref += rho0 * (h_xi(i0, i1, i2, i3) * h_xi(i0, i1, i2, i3) - 1.0) *
                    prefactor * h_w(i3) * h_vexc(i0, i1, i2, i3);
        }
      }
    }
  }
  Kokkos::deep_copy(delta_f_ref, h_delta_f_ref);

  // Compute delta_f and ff
  execution_space exec_space;
  T ff = 0;
  MDFT::get_delta_f(exec_space, xi, vexc, w, delta_f, ff, rho0, prefactor);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  EXPECT_TRUE(allclose(exec_space, delta_f, delta_f_ref, epsilon));
  EXPECT_LE(Kokkos::abs(ff - ff_ref), epsilon * Kokkos::abs(ff_ref));
}

TEST_F(TestSolventConstants, SwitchByString) {
  for (const auto& solvent_name : this->m_solvent_names) {
    switch (MDFT::Impl::solvent_index(solvent_name)) {
      case MDFT::Impl::const_solvent_index("spce"):
        ASSERT_EQ(solvent_name, "spce");
        break;

      case MDFT::Impl::const_solvent_index("spce-h"):
        ASSERT_EQ(solvent_name, "spce-h");
        break;

      case MDFT::Impl::const_solvent_index("spce-m"):
        ASSERT_EQ(solvent_name, "spce-m");
        break;

      case MDFT::Impl::const_solvent_index("tip3p"):
        ASSERT_EQ(solvent_name, "tip3p");
        break;

      case MDFT::Impl::const_solvent_index("tip3p-m"):
        ASSERT_EQ(solvent_name, "tip3p-m");
        break;

      default:
        ASSERT_FALSE(is_included(solvent_name, this->m_allowed_solvent_names));
        break;
    }
  }
}

TYPED_TEST(TestSolvent, Initialization) {
  using float_type  = typename TestFixture::float_type;
  using SolventType = typename TestFixture::SolventType;
  for (auto m : this->m_all_sizes) {
    test_solvent_init<float_type, SolventType>(
        m, this->m_file_path + "/" + this->m_setting_file, this->m_solvent_ref);
  }
}

TYPED_TEST(TestSolvent, GetDeltaRho) {
  using float_type = typename TestFixture::float_type;
  test_get_delta_rho<float_type>();
}

TYPED_TEST(TestSolvent, GetDeltaF) {
  using float_type = typename TestFixture::float_type;
  test_get_delta_f<float_type>();
}

}  // namespace
