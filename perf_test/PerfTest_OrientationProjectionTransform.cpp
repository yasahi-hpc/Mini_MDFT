// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include <benchmark/benchmark.h>

#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "MDFT_Grid.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "unit_test/Test_Utils.hpp"

using execution_space      = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

template <typename T>
struct TestOrientationProjection : public ::testing::Test {
  using float_type             = T;
  using scalar_array_type      = Kokkos::Array<T, 3>;
  using int_array_type         = Kokkos::Array<int, 3>;
  std::vector<int> m_all_sizes = {5, 10};
};

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_orientation_projection_forward(int n) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  int mmax = 10, molrotsymorder = 10;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);
  MDFT::OrientationProjectionTransform<execution_space, T> opt(
      grid, angular_grid, map);

  using OViewType = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;
  using PViewType = Kokkos::View<Kokkos::complex<T>****, Kokkos::LayoutRight,
                                 execution_space>;

  auto [nx, ny, nz] = grid.m_n_nodes;
  auto ntheta = angular_grid.m_ntheta, nphi = angular_grid.m_nphi,
       npsi = angular_grid.m_npsi, mrso = angular_grid.m_molrotsymorder;
  auto np = angular_grid.m_np;

  OViewType o("o", nx * ny * nz, ntheta, nphi, npsi),
      o_inv("o_inv", nx * ny * nz, ntheta, nphi, npsi),
      o_ref("o_ref", nx * ny * nz, ntheta, nphi, npsi);
  PViewType p("p", np, nx, ny, nz), p_ref("p_ref", np, nx, ny, nz);

  // Initialize o with random values
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(o, random_pool, 1.0);

  opt.angl2proj(o, p);

  // Make a reference at host
  using HostView1DType =
      Kokkos::View<T*, Kokkos::LayoutRight, host_execution_space>;
  using HostView2DType =
      Kokkos::View<T**, Kokkos::LayoutRight, host_execution_space>;
  using HostComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, Kokkos::LayoutRight,
                   host_execution_space>;
  using HostComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, Kokkos::LayoutRight,
                   host_execution_space>;

  auto p_map    = map.p();
  int mmax_p1   = p_map.extent(0);
  int mmax2_p1  = p_map.extent(1);
  int mmax_mrso = p_map.extent(2);

  auto wtheta         = angular_grid.m_wthetaofntheta;
  auto wigner_small_d = map.wigner_small_d();

  HostView1DType fm("fm", mmax_p1);

  for (int im = 0; im <= mmax; im++) {
    fm(im) = std::sqrt(2.0 * static_cast<T>(im) + 1.0);
  }

  HostView2DType my_r2d("my_r2d", nphi, npsi);
  HostComplexView2DType my_c2d("my_c2d", nphi, npsi / 2 + 1);
  HostComplexView3DType my_f_theta_mu2_mup("my_f_theta_mu2_mup", ntheta,
                                           mmax_mrso, mmax2_p1);

  auto h_wtheta =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wtheta);
  auto h_wigner_small_d =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wigner_small_d);

  auto h_p_ref = Kokkos::create_mirror_view(p_ref);
  auto h_o     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), o);
  int nxyz     = nx * ny * nz;

  using value_type = typename PViewType::non_const_value_type;
  using PView2DType =
      Kokkos::View<value_type**, Kokkos::LayoutRight, host_execution_space>;
  PView2DType p2d(h_p_ref.data(), p_ref.extent(0), nxyz);

  for (int ixyz = 0; ixyz < nxyz; ixyz++) {
    auto sub_o =
        Kokkos::subview(h_o, ixyz, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto sub_p = Kokkos::subview(p2d, Kokkos::ALL, ixyz);
    for (int itheta = 0; itheta < ntheta; itheta++) {
      for (int iphi = 0; iphi < nphi; iphi++) {
        for (int ipsi = 0; ipsi < npsi; ipsi++) {
          my_r2d(iphi, ipsi) = sub_o(itheta, iphi, ipsi);
        }
      }
      KokkosFFT::rfft2(host_execution_space(), my_r2d, my_c2d,
                       KokkosFFT::Normalization::forward);
      Kokkos::fence();
      for (int imu2 = 0; imu2 < mmax_mrso; imu2++) {
        for (int imup = 0; imup <= mmax; imup++) {
          my_f_theta_mu2_mup(itheta, imu2, imup + mmax) =
              Kokkos::conj(my_c2d(imup, imu2));
        }
        for (int imup = 0; imup < mmax; imup++) {
          my_f_theta_mu2_mup(itheta, imu2, imup) =
              Kokkos::conj(my_c2d(imup + mmax + 1, imu2));
        }
      }
    }

    int ip = 0;
    for (int im = 0; im <= mmax; im++) {
      for (int imup = -im; imup <= im; imup++) {
        for (int imu2 = 0; imu2 <= im / mrso; imu2++) {
          value_type sum = 0;
          for (int itheta = 0; itheta < ntheta; itheta++) {
            sum += my_f_theta_mu2_mup(itheta, imu2, imup + mmax) *
                   h_wigner_small_d(itheta, ip) * h_wtheta(itheta) * fm(im);
          }
          sub_p(ip) = sum;
          ip++;
        }
      }
    }
  }
  Kokkos::deep_copy(p_ref, h_p_ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 1e3;
  bool passed = allclose(execution_space(), p, p_ref, epsilon);
}

BENCHMARK_TEMPLATE_DEFINE_F(TestOrientationProjection, Angl2ProjFloat, float) {
  int n = state.range(0);
  test_orientation_projection_forward<float_type, int_array_type,
                                      scalar_array_type>(n);
}

BENCHMARK_TEMPLATE_DEFINE_F(TestOrientationProjection, Angl2ProjDouble, double) {
  int n = state.range(0);
  test_orientation_projection_forward<float_type, int_array_type,
                                      scalar_array_type>(n);
}

static constexpr double WARMUP_TIME = 0.1;

BENCHMARK_REGISTER_F(TestOrientationProjection, Angl2ProjFloat)
    ->Arg(2)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();
BENCHMARK_REGISTER_F(TestOrientationProjection, Angl2ProjDouble)
    ->Arg(2)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();

