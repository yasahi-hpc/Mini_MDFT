// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <Kokkos_Random.hpp>
#include "MDFT_Grid.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "MDFT_Convolution.hpp"
#include "Test_Utils.hpp"

namespace {

using execution_space      = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;
using float_types          = ::testing::Types<float, double>;

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

template <typename T, typename IntArrayType, typename ScalarArrayType>
void test_convolution_execute(int n, std::string file_path, int np_luc) {
  MDFT::SpatialGrid<execution_space, T> grid(IntArrayType{n, n, n},
                                             ScalarArrayType{1, 1, 1});
  using complex_type = Kokkos::complex<T>;
  using ViewType =
      Kokkos::View<complex_type****, Kokkos::LayoutRight, execution_space>;

  int mmax = 3, molrotsymorder = 2;
  MDFT::AngularGrid<execution_space, T> angular_grid(mmax, molrotsymorder);
  MDFT::OrientationProjectionMap<execution_space, T> map(angular_grid);
  MDFT::Convolution<execution_space, T> conv(file_path, grid, angular_grid, map,
                                             np_luc);

  auto [nx, ny, nz] = grid.m_n_nodes;
  auto np           = angular_grid.m_np;
  auto mrso         = angular_grid.m_molrotsymorder;

  ViewType deltarho_p("deltarho_p", np, nx, ny, nz),
      deltarho_p_ref("deltarho_p_ref", np, nx, ny, nz);

  // Initialize p with random values
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(deltarho_p, random_pool, z);
  Kokkos::deep_copy(deltarho_p_ref, deltarho_p);

  conv.execute(deltarho_p);

  // Make a reference on host
  auto h_kx =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_kx);
  auto h_ky =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_ky);
  auto h_kz =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_kz);
  auto p_map =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), map.p());
  auto p_to_m =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), map.m());
  auto p_to_mup =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), map.mup());
  auto p_to_mu2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), map.mu2());
  auto h_mnmunukhi_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           conv.mnmunukhi_q());
  int np_new         = h_mnmunukhi_q.extent(0);
  using HostIntView3DType =
      Kokkos::View<int***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using HostComplexView1DType =
      Kokkos::View<complex_type*, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using HostComplexView3DType =
      Kokkos::View<complex_type***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  HostIntView3DType h_gamma_p_isok("gamma_p_isok", nx, ny, nz);
  HostComplexView1DType h_deltarho_p_q("deltarho_p_q", np),
      h_deltarho_p_mq("deltarho_p_mq", np), h_gamma_p_q("gamma_p_q", np),
      h_gamma_p_mq("gamma_p_mq", np), h_ceff("ceff", np_new);
  HostComplexView3DType h_R("R", mmax + 1, 2 * mmax + 1, 2 * mmax + 1);

  // Copy coefficients to the host
  auto dq                  = conv.dq();
  using RotationCoeffsType = RotationCoeffs<host_execution_space, T>;
  RotationCoeffsType coeffs;
  auto h_a = coeffs.m_a;
  auto h_b = coeffs.m_b;
  auto h_c = coeffs.m_c;
  auto h_d = coeffs.m_d;
  auto h_deltarho_p =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), deltarho_p_ref);

  T epsilon = std::numeric_limits<T>::epsilon();
  // Start the kernel
  using axis_type = KokkosFFT::axis_type<3>;
  KokkosFFT::fftn(host_execution_space(), h_deltarho_p, h_deltarho_p,
                  axis_type{-3, -2, -1}, KokkosFFT::Normalization::none);

  for (int iz_q = 0; iz_q < nz / 2 + 1; iz_q++) {
    for (int iy_q = 0; iy_q < ny; iy_q++) {
      for (int ix_q = 0; ix_q < nx; ix_q++) {
        const int ix_mq   = MDFT::Impl::inv_index(ix_q, nx);
        const int iy_mq   = MDFT::Impl::inv_index(iy_q, ny);
        const int iz_mq   = MDFT::Impl::inv_index(iz_q, nz);
        ScalarArrayType q = {h_kx(ix_q), h_ky(iy_q), h_kz(iz_q)};

        // gamma_p_isok is a logical array. If gamma(ix_q,iy_q,iz_q) has already
        // been calculated, it is .true.
        if (h_gamma_p_isok(ix_q, iy_q, iz_q) &&
            h_gamma_p_isok(ix_mq, iy_mq, iz_mq))
          continue;

        // pay attention to the special case(s) where q=-q
        // this should only happen for ix=0 and ix=nx/2
        bool q_eq_mq = (ix_mq == ix_q && iy_mq == iy_q && iz_mq == iz_q);

        // Prepare R^m_mup_khi(q)
        MDFT::Impl::rotation_matrix_between_complex_spherical_harmonics_lu(
            q, h_c, h_d, h_a, h_b, h_R);

        // Prevent underflow for all the elements
        auto* R_data = h_R.data();
        for (int idx = 0; idx < h_R.size(); idx++) {
          R_data[idx] = MDFT::Impl::prevent_underflow(R_data[idx], epsilon);
        }

        // Eq. 1.23 We don't need to compute gshrot for -q since there are
        // symetries between R(q) and R(-q). Thus, we do q and -q at the same
        // time. That's the most important point in doing all q but half of
        // mu. Lu decided to do all mu but half of q in her code

        //  Rotation to molecular (q) frame
        //  on  a       deltarho_p_q(m,khi,mu2) =  sum/mup  @
        //  gamma_p_q(m,mup,mu2) * R(m,mup,khi)
        //  =>              gamma_p_q(mup,m,mu2) * R(mup,m,khi)
        int ip_tmp = 0;
        auto sub_deltarho_p_q =
            Kokkos::subview(h_deltarho_p, Kokkos::ALL, ix_q, iy_q, iz_q);
        auto sub_deltarho_p_mq =
            Kokkos::subview(h_deltarho_p, Kokkos::ALL, ix_mq, iy_mq, iz_mq);

        Kokkos::deep_copy(h_gamma_p_q, sub_deltarho_p_q);
        Kokkos::deep_copy(h_gamma_p_mq, sub_deltarho_p_mq);
        for (int im = 0; im <= mmax; im++) {
          for (int ikhi = -im; ikhi <= im; ikhi++) {
            for (int imu2 = 0; imu2 <= im / mrso; imu2++) {
              complex_type deltarho_p_q_loc(0.0), deltarho_p_mq_loc(0.0);
              for (int imup = -im; imup <= im; imup++) {
                auto ip_mapped = p_map(im, imup + mmax, imu2);
                deltarho_p_q_loc +=
                    h_gamma_p_q(ip_mapped) * h_R(im, imup + mmax, ikhi + mmax);
                deltarho_p_mq_loc += h_gamma_p_mq(ip_mapped) *
                                     h_R(im, imup + mmax, -ikhi + mmax);
              }

              h_deltarho_p_q(ip_tmp) = deltarho_p_q_loc;
              h_deltarho_p_mq(ip_tmp) =
                  deltarho_p_mq_loc * Kokkos::pow(-1.0, im);
              ip_tmp++;
            }
          }
        }

        // c^{m,n}_{mu,nu,chi}(|q|) is tabulated for c%nq values of |q|.
        // Find the tabulated value that is closest to |q|. Its index is iq.
        // Note |q| = |-q| so iq is the same for both vectors.

        // iq = int( norm2(q) /c%dq +0.5) +1
        // ceff(:) = c%mnmunukhi_q(:,iq)

        auto effectiveiq = MDFT::Impl::norm2(q) / dq;

        // the lower bound. The upper bound is iq+1
        int iq = static_cast<int>(effectiveiq);

        // linear interpolation    y=alpha*upperbound + (1-alpha)*lowerbound
        T alpha = effectiveiq - static_cast<T>(iq);

        for (int ip = 0; ip < np_new; ip++) {
          // Linear interpolation
          h_ceff(ip) = alpha * h_mnmunukhi_q(ip, iq + 1) +
                       (1.0 - alpha) * h_mnmunukhi_q(ip, iq);
        }

        // Ornstein-Zernike in the molecular frame
        // We do OZ for q and -q at the same time

        Kokkos::deep_copy(h_gamma_p_q, 0.0);
        Kokkos::deep_copy(h_gamma_p_mq, 0.0);
        int ia_tmp = 0;
        for (int ip = 0; ip < np; ip++) {
          auto im   = p_to_m(ip);
          auto ikhi = p_to_mup(ip);
          auto imu2 = p_to_mu2(ip);

          for (int in = Kokkos::abs(ikhi); in <= mmax; in++) {
            for (int inu2 = -in / mrso; inu2 <= in / mrso; inu2++) {
              auto ceff = h_ceff(ia_tmp);
              if (inu2 < 0) {
                auto ip_mapped = p_map(in, ikhi + mmax, Kokkos::abs(inu2));
                h_gamma_p_q(ip) += ceff * h_deltarho_p_q(ip_mapped);
                h_gamma_p_mq(ip) += ceff * h_deltarho_p_mq(ip_mapped);
              } else {
                auto ip_mapped = p_map(in, ikhi + mmax, inu2);
                h_gamma_p_q(ip) +=
                    ceff * Kokkos::conj(h_deltarho_p_mq(ip_mapped));
                h_gamma_p_mq(ip) +=
                    ceff * Kokkos::conj(h_deltarho_p_q(ip_mapped));
              }
              ia_tmp++;
            }
          }
        }

        // Rotation from molecular frame to fix frame
        // le passage retour au repaire fixe se fait avec simplement le conjugue
        // complexe de l'harm sph generalisee we use deltarho_p_q and
        // deltarho_p_mq as temp arrays since they're not used after MOZ
        for (int idx = 0; idx < h_R.size(); idx++) {
          R_data[idx] = Kokkos::conj(R_data[idx]);
        }

        // prevent underflow in gamma_p_q/mq * R if gamma_p is very low
        for (int ip = 0; ip < np; ip++) {
          h_gamma_p_q(ip) =
              MDFT::Impl::prevent_underflow(h_gamma_p_q(ip), epsilon);
          h_gamma_p_mq(ip) =
              MDFT::Impl::prevent_underflow(h_gamma_p_mq(ip), epsilon);
        }

        ip_tmp = 0;
        Kokkos::deep_copy(h_deltarho_p_q, 0.0);
        Kokkos::deep_copy(h_deltarho_p_mq, 0.0);
        for (int im = 0; im <= mmax; im++) {
          for (int imup = -im; imup <= im; imup++) {
            for (int imu2 = 0; imu2 <= im / mrso; imu2++) {
              for (int ikhi = -im; ikhi <= im; ikhi++) {
                h_deltarho_p_q(ip_tmp) +=
                    h_gamma_p_q(p_map(im, ikhi + mmax, imu2)) *
                    h_R(im, imup + mmax, ikhi + mmax);
                h_deltarho_p_mq(ip_tmp) +=
                    h_gamma_p_mq(p_map(im, ikhi + mmax, imu2)) *
                    h_R(im, imup + mmax, -ikhi + mmax);
              }
              h_deltarho_p_mq(ip_tmp) =
                  h_deltarho_p_mq(ip_tmp) * Kokkos::pow(-1.0, im);
              ip_tmp++;
            }
          }
        }

        // Move the result for this given vector q to the big array containing
        // all results. First, for q,
        for (int ip = 0; ip < np; ip++) {
          h_deltarho_p(ip, ix_q, iy_q, iz_q) = h_deltarho_p_q(ip);
          // Then, for -q. Again, pay attention to the singular mid-k point
          if (q_eq_mq &&
              (ix_mq == nx / 2 || iy_mq == ny / 2 || iz_mq == nz / 2)) {
            h_deltarho_p(ip, ix_mq, iy_mq, iz_mq) =
                Kokkos::conj(h_deltarho_p_mq(ip));
          } else {
            h_deltarho_p(ip, ix_mq, iy_mq, iz_mq) = h_deltarho_p_mq(ip);
          }
        }

        // And store you have already done the job
        h_gamma_p_isok(ix_q, iy_q, iz_q)    = 1;
        h_gamma_p_isok(ix_mq, iy_mq, iz_mq) = 1;
      }  // for (int ix_q = 0; ix_q < nx; ix_q++)
    }    // for (int iy_q = 0; iy_q < ny; iy_q++)
  }      // for (int iz_q = 0; iz_q < nz/2+1; iz_q++)

  KokkosFFT::ifftn(host_execution_space(), h_deltarho_p, h_deltarho_p,
                   axis_type{-3, -2, -1}, KokkosFFT::Normalization::none);

  Kokkos::deep_copy(deltarho_p_ref, h_deltarho_p);

  EXPECT_TRUE(
      allclose(execution_space(), deltarho_p, deltarho_p_ref, epsilon * 1e3));
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

TYPED_TEST(TestConvolution, Execute) {
  using float_type        = typename TestFixture::float_type;
  using int_array_type    = typename TestFixture::int_array_type;
  using scalar_array_type = typename TestFixture::scalar_array_type;

  std::string file_path = this->m_file_path + "/" + this->m_filename;
  for (auto m : this->m_all_sizes) {
    test_convolution_execute<float_type, int_array_type, scalar_array_type>(
        m, file_path, this->m_np_luc);
  }
}

}  // namespace
