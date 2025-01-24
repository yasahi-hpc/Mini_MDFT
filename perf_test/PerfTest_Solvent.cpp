// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include <benchmark/benchmark.h>

#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Solvent.hpp"
#include "unit_test/Test_Utils.hpp"

using execution_space = Kokkos::DefaultExecutionSpace;

template <typename T>
using View4DType = Kokkos::View<T****, execution_space>;

template <typename T>
using View1DLayRType = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;

template <typename T>
using View4DLayRType =
    Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

template <typename T>
struct TestSolvent : public ::benchmark::Fixture {
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

  void SetUp() {
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

template <typename T>
auto prep_delta_rho(const int n0 = 2, const int n1 = 3, const int n2 = 4,
                    const int n3 = 5) {
  View4DType<T> xi("xi", n0, n1, n2, n3),
      delta_rho("delta_rho", n0, n1, n2, n3),
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

  return std::make_tuple(xi, delta_rho, delta_rho_ref, rho0);
}

template <typename T>
auto test_get_delta_rho(const View4DType<T>& xi, const View4DType<T>& delta_rho,
                        const View4DType<T>& delta_rho_ref, const T rho0) {
  // Compute delta_rho
  execution_space exec_space;
  exec_space.fence();
  Kokkos::Timer timer;
  MDFT::get_delta_rho(exec_space, xi, delta_rho, rho0);
  exec_space.fence();
  double time_delta_rho = timer.seconds();
  T epsilon             = std::numeric_limits<T>::epsilon() * 100;
  bool passed = allclose(exec_space, delta_rho, delta_rho_ref, epsilon);

  return std::make_tuple(time_delta_rho, passed);
}

BENCHMARK_TEMPLATE_DEFINE_F(TestSolvent, GetDeltaRhoFloat, float)
(benchmark::State& state) {
  int n0 = state.range(0), n1 = state.range(0), n2 = state.range(0),
      n3 = state.range(1);

  auto [xi, delta_rho, delta_rho_ref, rho0] =
      prep_delta_rho<float_type>(n0, n1, n2, n3);
  for (auto _ : state) {
    const auto [time_delta_rho, passed] =
        test_get_delta_rho<float_type>(xi, delta_rho, delta_rho_ref, rho0);

    state.SetIterationTime(time_delta_rho);
    state.counters["Passed"] = passed;
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TestSolvent, GetDeltaRhoDouble, double)
(benchmark::State& state) {
  int n0 = state.range(0), n1 = state.range(0), n2 = state.range(0),
      n3 = state.range(1);

  auto [xi, delta_rho, delta_rho_ref, rho0] =
      prep_delta_rho<float_type>(n0, n1, n2, n3);
  for (auto _ : state) {
    const auto [time_delta_rho, passed] =
        test_get_delta_rho<float_type>(xi, delta_rho, delta_rho_ref, rho0);

    state.SetIterationTime(time_delta_rho);
    state.counters["Passed"] = passed;
  }
}

template <typename T>
auto prep_delta_f(const int n0 = 2, const int n1 = 3, const int n2 = 4,
                  const int n3 = 5) {
  View1DLayRType<T> w("w", n3);
  View4DLayRType<T> xi("xi", n0, n1, n2, n3), vexc("vexc", n0, n1, n2, n3),
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

  return std::make_tuple(xi, vexc, w, delta_f, delta_f_ref, ff_ref, rho0,
                         prefactor);
}

template <typename T>
auto test_get_delta_f(const View4DLayRType<T>& xi,
                      const View4DLayRType<T>& vexc, const View1DLayRType<T>& w,
                      const View4DLayRType<T>& delta_f,
                      const View4DLayRType<T>& delta_f_ref, const T ff_ref,
                      const T rho0, const T prefactor) {
  execution_space exec_space;
  T ff = 0;
  exec_space.fence();
  Kokkos::Timer timer;
  MDFT::get_delta_f(exec_space, xi, vexc, w, delta_f, ff, rho0, prefactor);
  exec_space.fence();
  double time_delta_f = timer.seconds();

  T epsilon   = std::numeric_limits<T>::epsilon() * 100;
  bool passed = allclose(exec_space, delta_f, delta_f_ref, epsilon) &&
                (Kokkos::abs(ff - ff_ref) < epsilon * Kokkos::abs(ff_ref));

  return std::make_tuple(time_delta_f, passed);
}

BENCHMARK_TEMPLATE_DEFINE_F(TestSolvent, GetDeltaFFloat, float)
(benchmark::State& state) {
  int n0 = state.range(0), n1 = state.range(0), n2 = state.range(0),
      n3 = state.range(1);

  auto [xi, vexc, w, delta_f, delta_f_ref, ff_ref, rho0, prefactor] =
      prep_delta_f<float_type>(n0, n1, n2, n3);
  for (auto _ : state) {
    const auto [time_delta_f, passed] = test_get_delta_f<float_type>(
        xi, vexc, w, delta_f, delta_f_ref, ff_ref, rho0, prefactor);

    state.SetIterationTime(time_delta_f);
    state.counters["Passed"] = passed;
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(TestSolvent, GetDeltaFDouble, double)
(benchmark::State& state) {
  int n0 = state.range(0), n1 = state.range(0), n2 = state.range(0),
      n3 = state.range(1);

  auto [xi, vexc, w, delta_f, delta_f_ref, ff_ref, rho0, prefactor] =
      prep_delta_f<float_type>(n0, n1, n2, n3);
  for (auto _ : state) {
    const auto [time_delta_f, passed] = test_get_delta_f<float_type>(
        xi, vexc, w, delta_f, delta_f_ref, ff_ref, rho0, prefactor);

    state.SetIterationTime(time_delta_f);
    state.counters["Passed"] = passed;
  }
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  for (int i = 30; i <= 100; i += 20)
    for (int j = 20; j <= 200; j += 20)
      b->Args({i, j});
}

static constexpr double WARMUP_TIME = 0.1;

BENCHMARK_REGISTER_F(TestSolvent, GetDeltaRhoFloat)
    ->Apply(CustomArgs)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();
BENCHMARK_REGISTER_F(TestSolvent, GetDeltaRhoDouble)
    ->Apply(CustomArgs)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();
BENCHMARK_REGISTER_F(TestSolvent, GetDeltaFFloat)
    ->Apply(CustomArgs)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();
BENCHMARK_REGISTER_F(TestSolvent, GetDeltaFDouble)
    ->Apply(CustomArgs)
    ->MinWarmUpTime(WARMUP_TIME)
    ->UseManualTime();

