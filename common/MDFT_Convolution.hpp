#ifndef MDFT_CONVOLUTION_HPP
#define MDFT_CONVOLUTION_HPP

#include <memory>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Wigner.hpp"
#include "MDFT_Rotation.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"

namespace MDFT {
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
class Convolution {
 private:
  constexpr static std::size_t FFT_DIM = 3;
  using IntType                        = int;
  using ComplexType                    = Kokkos::complex<ScalarType>;
  using ArrayType                      = Kokkos::Array<ScalarType, 3>;
  using IntView1DType                  = Kokkos::View<IntType*, ExecutionSpace>;
  using IntView2DType = Kokkos::View<IntType**, ExecutionSpace>;
  using IntView3DType = Kokkos::View<IntType***, ExecutionSpace>;
  using View1DType    = Kokkos::View<ScalarType*, ExecutionSpace>;
  using View2DType    = Kokkos::View<ScalarType**, ExecutionSpace>;
  using View3DType    = Kokkos::View<ScalarType***, ExecutionSpace>;
  using View4DType    = Kokkos::View<ScalarType****, ExecutionSpace>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<ScalarType>*, ExecutionSpace>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<ScalarType>**, ExecutionSpace>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<ScalarType>***, ExecutionSpace>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<ScalarType>****, ExecutionSpace>;
  using SpatialGridType    = SpatialGrid<ExecutionSpace, ScalarType>;
  using AngularGridType    = AngularGrid<ExecutionSpace, ScalarType>;
  using RotationCoeffsType = RotationCoeffs<ExecutionSpace, ScalarType>;
  using OrientationProjectionMapType =
      OrientationProjectionMap<ExecutionSpace, ScalarType>;

  // Internal Scratch View Type
  using ScratchView1DType =
      Kokkos::View<Kokkos::complex<ScalarType>*,
                   typename ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using ScratchView3DType =
      Kokkos::View<Kokkos::complex<ScalarType>***,
                   typename ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  // FFT plan for 3D batched FFT (inplace)
  using C2CPlanType = KokkosFFT::Plan<ExecutionSpace, ComplexView4DType,
                                      ComplexView4DType, FFT_DIM>;

  // MDRange policy used internally
  using MDRangeType = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right> >;
  using TileType  = typename MDRangeType::tile_type;
  using PointType = typename MDRangeType::point_type;

  //! Dynamically allocatable inplace fft plan.
  std::unique_ptr<C2CPlanType> m_forward_plan, m_backward_plan;

  RotationCoeffsType m_coeffs;

  // tabulation des harmoniques sphériques r(m,mup,mu,theta) en un tableau
  OrientationProjectionMapType m_map;

  // Angular grid
  int m_mmax, m_np, m_molrotsymorder;

  // Spatial grid
  int m_nx, m_ny, m_nz;
  View1DType m_kx, m_ky, m_kz;

  // Instead of gamma_p_isok, we use a map array without overlapping (length, 3)
  IntView2DType m_gamma_p_map;

  IntView3DType m_ia_map;

  //
  ComplexView1DType m_ceff;

  ComplexView2DType m_mnmunukhi_q;

 public:
  Convolution()  = delete;
  ~Convolution() = default;

  // [TO DO] We need OrientationProjectionMap for this class
  Convolution(const SpatialGridType& spatial_grid,
              const AngularGridType& angular_grid,
              const OrientationProjectionMapType& map)
      : m_map(map),
        m_mmax(angular_grid.m_mmax),
        m_np(angular_grid.m_np),
        m_molrotsymorder(angular_grid.m_molrotsymorder),
        m_nx(spatial_grid.m_nx),
        m_ny(spatial_grid.m_ny),
        m_nz(spatial_grid.m_nz),
        m_kx(spatial_grid.m_kx),
        m_ky(spatial_grid.m_ky),
        m_kz(spatial_grid.m_kz) {
    // Prepare FFT plans
    // 3D batched plan (nbatch, nz, ny, nx)
    ExecutionSpace exec;
    auto nx = spatial_grid.m_nx;
    auto ny = spatial_grid.m_ny;
    auto nz = spatial_grid.m_nz;
    auto np = angular_grid.m_np;

    // Terrible implementation, needed to improve
    int np_new = 0;
    for (int m = 0; m <= m_mmax; ++m) {
      for (int khi = -m; khi <= m; ++khi) {
        for (int mu2 = 0; mu2 <= m / m_molrotsymorder; ++mu2) {
          for (int n = Kokkos::abs(khi); n <= m_mmax; ++n) {
            for (int nu2 = -n / m_molrotsymorder; nu2 <= n / m_molrotsymorder;
                 ++nu2) {
              np_new++;
            }
          }
        }
      }
    }

    using HostIntView3DType = Kokkos::View<int***, Kokkos::HostSpace>;
    HostIntView3DType h_gamma_p_isok("gamma_p_isok", nx, ny, nz);

    // Instead of gamma_p_isok, we use a map array without overlapping
    int length = 0;
    for (int iz = 0; iz < nz / 2 + 1; ++iz) {
      int iz_mq = MDFT::Impl::inv_index(iz, nz);
      for (int iy = 0; iy < ny; ++iy) {
        int iy_mq = MDFT::Impl::inv_index(iy, ny);
        for (int ix = 0; ix < nx; ++ix) {
          int ix_mq = MDFT::Impl::inv_index(ix, nx);
          if (h_gamma_p_isok(ix, iy, iz) == 1 &&
              h_gamma_p_isok(ix_mq, iy_mq, iz_mq) == 1)
            continue;

          // Store you have already done the job
          h_gamma_p_isok(ix, iy, iz)          = 1;
          h_gamma_p_isok(ix_mq, iy_mq, iz_mq) = 1;
          length++;
        }
      }
    }

    // Reset h_gamma_p_isok
    Kokkos::deep_copy(h_gamma_p_isok, 0);

    m_gamma_p_map      = IntView2DType("gamma_p_map", length, 3);
    auto h_gamma_p_map = Kokkos::create_mirror_view(m_gamma_p_map);
    length             = 0;
    for (int iz = 0; iz < nz / 2 + 1; ++iz) {
      int iz_mq = MDFT::Impl::inv_index(iz, nz);
      for (int iy = 0; iy < ny; ++iy) {
        int iy_mq = MDFT::Impl::inv_index(iy, ny);
        for (int ix = 0; ix < nx; ++ix) {
          int ix_mq = MDFT::Impl::inv_index(ix, nx);
          if (h_gamma_p_isok(ix, iy, iz) == 1 &&
              h_gamma_p_isok(ix_mq, iy_mq, iz_mq) == 1)
            continue;

          // Store you have already done the job
          h_gamma_p_map(length, 0) = ix;
          h_gamma_p_map(length, 1) = iy;
          h_gamma_p_map(length, 2) = iz;

          h_gamma_p_isok(ix, iy, iz)          = 1;
          h_gamma_p_isok(ix_mq, iy_mq, iz_mq) = 1;
          length++;
        }
      }
    }
    Kokkos::deep_copy(m_gamma_p_map, h_gamma_p_map);

    // Allocate views
    int nq = 10;  // Is there a way to get this value without reading file?
    m_ceff = ComplexView1DType("ceff", np_new);
    m_mnmunukhi_q = ComplexView2DType("mnmunukhi_q", np_new, nq);

    // Can we make empty views just with shapes?
    ComplexView4DType delta_rho("delta_rho", np, nz, ny, nx);

    // We need an inplace transform
    using axes_type = KokkosFFT::axis_type<3>;
    axes_type axes  = {-3, -2, -1};
    m_forward_plan  = std::make_unique<C2CPlanType>(
        exec, delta_rho, delta_rho, KokkosFFT::Direction::forward, axes);
    m_backward_plan = std::make_unique<C2CPlanType>(
        exec, delta_rho, delta_rho, KokkosFFT::Direction::backward, axes);
  }

 public:
  // \brief
  // \tparam View Orientation view, needs to be a Complex View
  //
  // \param deltarho_p [in/out] Orientation (nm, nmup, nmu, nz, ny, nx)
  template <KokkosView View>
    requires KokkosViewAccesible<ExecutionSpace, View>
  void execute(const View& deltarho_p) {
    // delta_rho_hat^m_{\mu', \mu}(q) = FFT [delta_rho^m_{\mu', \mu}(r)]
    m_forward_plan->execute(deltarho_p, deltarho_p,
                            KokkosFFT::Normalization::none);

    // For all vectors q and -q handled simultaneously.
    // ix_q,iy_q,iz_q are the coordinates of vector q, while ix_mq,iy_mq_iz_mq
    // are those of vector -q

    using member_type =
        typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
    // std::size_t N = m_nx * m_ny * (m_nz/2 + 1);
    std::size_t N             = m_gamma_p_map.extent(0);
    std::size_t mmax_p1       = static_cast<std::size_t>(m_mmax + 1);
    std::size_t mmax2_p1      = static_cast<std::size_t>(2 * m_mmax + 1);
    std::size_t request_size  = mmax_p1 * mmax2_p1 * mmax2_p1;
    std::size_t request_size2 = m_np;
    int scratch_size          = ScratchView3DType::shmem_size(request_size) +
                       ScratchView1DType::shmem_size(request_size2 * 4);
    int level = 1;  // using global memory
    auto team_policy =
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(scratch_size));
    auto a = m_coeffs.m_a;
    auto b = m_coeffs.m_b;
    auto c = m_coeffs.m_c;
    auto d = m_coeffs.m_d;

    auto p_map    = m_map.p();
    auto p_to_m   = m_map.m();
    auto p_to_mup = m_map.mup();
    auto p_to_mu  = m_map.mu();

    auto gamma_p_map = m_gamma_p_map;
    auto ia_map      = m_ia_map;
    auto ceff        = m_ceff;
    auto mnmunukhi_q = m_mnmunukhi_q;

    auto kx = m_kx, ky = m_ky, kz = m_kz;
    int nx = m_nx, ny = m_ny, nz = m_nz;
    int np   = m_np;
    int mrso = m_molrotsymorder;
    int mmax = m_mmax;
    ScalarType dq =
        1.0;  // Coming from
              // read_c_luc(1,c%mnmunukhi_q,mmax,mrso,qmaxnecessary,c%np,c%nq,c%dq,c%m,c%n,c%mu,c%nu,c%khi,c%ip)
    ScalarType epsilon = std::numeric_limits<ScalarType>::epsilon() * 10;
    Kokkos::parallel_for(
        "convolution", team_policy,
        KOKKOS_LAMBDA(const member_type& team_member) {
          const auto idx = team_member.league_rank();

          /*
          const int ix = idx / (ny * nz);
          const int iyz = idx % (ny * nz);
          const int iy = iyz / nz;
          const int iz = iyz % nz;
          */
          const int ix = gamma_p_map(idx, 0), iy = gamma_p_map(idx, 1),
                    iz = gamma_p_map(idx, 2);
          ArrayType q  = {kx(ix), ky(iy), kz(iz)};

          const int ix_mq = MDFT::Impl::inv_index(ix, nx);
          const int iy_mq = MDFT::Impl::inv_index(iy, ny);
          const int iz_mq = MDFT::Impl::inv_index(iz, nz);

          // pay attention to the special case(s) where q=-q
          // this should only happen for ix=1 and ix=nx/2
          bool q_eq_mq = (ix_mq == ix && iy_mq == iy && iz_mq == iz);

          // Prepare R^m_mup_khi(q)
          ScratchView3DType s_R(team_member.team_scratch(level), mmax_p1,
                                mmax2_p1, mmax2_p1);
          MDFT::Impl::rotation_matrix_between_complex_spherical_harmonics_lu(
              q, c, d, a, b, s_R);

          Kokkos::parallel_for(
              Kokkos::ThreadVectorMDRange<Kokkos::Rank<3>, member_type>(
                  team_member, mmax_p1, mmax2_p1, mmax2_p1),
              [&](const int m, const int mup, const int mu2) {
                s_R(m, mup, mu2) =
                    MDFT::Impl::prevent_underflow(s_R(m, mup, mu2), epsilon);
              });

          // Eq. 1.23 We don't need to compute gshrot for -q since there are
          // symetries between R(q) and R(-q). Thus, we do q and -q at the same
          // time. That's the most important point in doing all q but half of
          // mu. Lu decided to do all mu but half of q in her code

          // Rotation to molecular (q) frame
          // on  a       deltarho_p_q(m,khi,mu2) =  sum/mup  @
          // gamma_p_q(m,mup,mu2) * R(m,mup,khi)
          // => gamma_p_q(mup,m,mu2) * R(mup,m,khi)

          ScratchView1DType s_gamma_p_q(team_member.team_scratch(level), np),
              s_gamma_p_mq(team_member.team_scratch(level), np),
              s_deltarho_p_q(team_member.team_scratch(level), np),
              s_deltarho_p_mq(team_member.team_scratch(level), np);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                s_gamma_p_q(ip)  = deltarho_p(ip, iz, iy, ix);
                s_gamma_p_mq(ip) = deltarho_p(ip, iz_mq, iy_mq, ix_mq);
                auto im          = p_to_m(ip);
                auto ikphi       = p_to_mup(ip);
                auto imu2        = p_to_mu(ip);
                Kokkos::complex<ScalarType> deltarho_p_q_loc(0.0),
                    deltarho_p_mq_loc(0.0);
                for (int imup = 0; imup < im * 2 + 1; imup++) {
                  deltarho_p_q_loc += s_gamma_p_q(ip) * s_R(im, imup, imu2);
                  deltarho_p_mq_loc += s_gamma_p_mq(ip) * s_R(im, imup, imu2);
                }
                s_deltarho_p_q(ip)  = deltarho_p_q_loc;
                s_deltarho_p_mq(ip) = deltarho_p_mq_loc * Kokkos::pow(-1.0, im);
              });
          team_member.team_barrier();

          // c^{m,n}_{mu,nu,chi}(|q|) is tabulated for c%nq values of |q|.
          // Find the tabulated value that is closest to |q|. Its index is iq.
          // Note |q| = |-q| so iq is the same for both vectors.

          // norm(q)/dq is in [0,n] while our iq should be in [1,n+1]. Thus, add
          // +1.
          auto effectiveiq = MDFT::Impl::norm2(q) / dq + 1.0;

          // the lower bound. The upper bound is iq+1
          int iq = static_cast<int>(effectiveiq);

          // linear interpolation    y=alpha*upperbound + (1-alpha)*lowerbound
          ScalarType alpha = effectiveiq - static_cast<ScalarType>(iq);

          // This loop is incorrect
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, np),
                               [&](const int ip) {
                                 ceff(ip) = alpha * mnmunukhi_q(ip, iq + 1) +
                                            (1.0 - alpha) * mnmunukhi_q(ip, iq);
                               });

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                s_gamma_p_q(ip)  = Kokkos::complex<ScalarType>(0.0, 0.0);
                s_gamma_p_mq(ip) = Kokkos::complex<ScalarType>(0.0, 0.0);

                auto im   = p_to_m(ip);
                auto ikhi = p_to_mup(ip);
                auto imu2 = p_to_mu(ip);

                // int ia = ia_map(ip, n, nu2);
                for (int n = Kokkos::abs(ikhi); n < mmax_p1; n++) {
                  for (int nu2 = -n / mrso; nu2 < n / mrso; nu2++) {
                    auto ia = ia_map(ip, n, nu2);
                    if (nu2 < 0) {
                      s_gamma_p_q(ip) =
                          s_gamma_p_q(ip) +
                          ceff(ia) *
                              s_deltarho_p_q(p_map(n, ikhi, Kokkos::abs(nu2)));
                      s_gamma_p_mq(ip) =
                          s_gamma_p_mq(ip) +
                          ceff(ia) *
                              s_deltarho_p_mq(p_map(n, ikhi, Kokkos::abs(nu2)));
                    } else {
                      s_gamma_p_q(ip) = s_gamma_p_q(ip) +
                                        ceff(ia) * Kokkos::conj(s_deltarho_p_mq(
                                                       p_map(n, ikhi, nu2)));
                      s_gamma_p_mq(ip) =
                          s_gamma_p_mq(ip) +
                          ceff(ia) *
                              Kokkos::conj(s_deltarho_p_q(p_map(n, ikhi, nu2)));
                    }
                  }
                }
              });

          // Rotation from molecular frame to fix frame
          // R = conjg(R) Do this isinde parallel region
          // le passage retour au repaire fixe se fait avec simplement le
          // conjugue complexe de l'harm sph generalisee we use deltarho_p_q and
          // deltarho_p_mq as temp arrays since they're not used after MOZ

          // prevent underflow in gamma_p_q/mq * R if gamma_p is very low
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team_member, np), [&](const int ip) {
                s_gamma_p_q(ip) =
                    MDFT::Impl::prevent_underflow(s_gamma_p_q(ip), epsilon);
                s_gamma_p_mq(ip) =
                    MDFT::Impl::prevent_underflow(s_gamma_p_mq(ip), epsilon);
              });

          bool is_singular_mid_k =
              q_eq_mq &&
              (ix == nx / 2 + 1 && iy == ny / 2 + 1 && iz == nz / 2 + 1);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                auto im   = p_to_m(ip);
                auto imup = p_to_mup(ip);
                auto imu2 = p_to_mu(ip);

                ComplexType sum_deltarho_p_q = 0, sum_deltarho_p_mq = 0;
                for (int ikhi = -im; ikhi <= im; ikhi++) {
                  sum_deltarho_p_q +=
                      s_gamma_p_q(p_map(im, ikhi, imu2)) *
                      Kokkos::conj(s_R(im, imup + mmax, ikhi + mmax));
                  sum_deltarho_p_mq +=
                      s_gamma_p_mq(p_map(im, ikhi, imu2)) *
                      Kokkos::conj(s_R(im, imup + mmax, -ikhi + mmax));
                }

                // what is the order of s_deltarho_p_q and deltarho_p_q?
                s_deltarho_p_q(ip)  = sum_deltarho_p_q;
                s_deltarho_p_mq(ip) = sum_deltarho_p_mq * Kokkos::pow(-1.0, im);

                deltarho_p(ip, iz, iy, ix) = s_deltarho_p_q(ip);
                deltarho_p(ip, iz_mq, iy_mq, ix_mq) =
                    is_singular_mid_k ? Kokkos::conj(s_deltarho_p_mq(ip))
                                      : s_deltarho_p_mq(ip);
              });
        });

    // gamma^m_{\mu', \mu}(r) = IFFT [\hat{gamma}^m_{\mu', \mu}(q)]
    m_backward_plan->execute(deltarho_p, deltarho_p);
  }
};
};  // namespace MDFT

#endif
