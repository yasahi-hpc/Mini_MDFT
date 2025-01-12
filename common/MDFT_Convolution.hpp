#ifndef MDFT_CONVOLUTION_HPP
#define MDFT_CONVOLUTION_HPP

#include <memory>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Wigner.hpp"
#include "MDFT_Rotation.hpp"
#include "MDFT_OrientationProjectionTransform.hpp"
#include "IO/MDFT_ReadCLuc.hpp"

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
  using LucDataType = MDFT::IO::LucData<ExecutionSpace, ScalarType>;

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

  // Helper to read luc data
  std::unique_ptr<LucDataType> m_luc_data;

  // Angular grid
  int m_mmax, m_np, m_molrotsymorder;

  // Spatial grid
  int m_nx, m_ny, m_nz;
  View1DType m_kx, m_ky, m_kz;

  ScalarType m_dq;

  // Instead of gamma_p_isok, we use a map array without overlapping (length, 3)
  IntView2DType m_gamma_p_map;
  IntView2DType m_ia_map;

  ComplexView2DType m_mnmunukhi_q;

 public:
  Convolution()  = delete;
  ~Convolution() = default;

  Convolution(const std::string filename, const SpatialGridType& spatial_grid,
              const AngularGridType& angular_grid,
              const OrientationProjectionMapType& map, const int np_luc)
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
    auto nx = spatial_grid.m_nx;
    auto ny = spatial_grid.m_ny;
    auto nz = spatial_grid.m_nz;
    auto np = angular_grid.m_np;

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

    // Initialize m_ia_map
    auto h_p_to_mup =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_map.mup());

    int na = 0;
    for (int ip = 0; ip < np; ip++) {
      auto ikhi = h_p_to_mup(ip);
      for (int in = Kokkos::abs(ikhi); in <= m_mmax; in++) {
        for (int inu2 = -in / m_molrotsymorder; inu2 < in / m_molrotsymorder;
             inu2++) {
          na++;
        }
      }
    }

    m_ia_map      = IntView2DType("ia_map", na, 3);
    auto h_ia_map = Kokkos::create_mirror_view(m_ia_map);

    int ia = 0;
    for (int ip = 0; ip < np; ip++) {
      auto ikhi = h_p_to_mup(ip);
      for (int in = Kokkos::abs(ikhi); in <= m_mmax; in++) {
        for (int inu2 = -in / m_molrotsymorder; inu2 < in / m_molrotsymorder;
             inu2++) {
          h_ia_map(ia, 0) = ip;
          h_ia_map(ia, 1) = in;
          h_ia_map(ia, 2) = inu2;
          ia++;
        }
      }
    }
    Kokkos::deep_copy(m_ia_map, h_ia_map);

    // We need an inplace transform
    // Can we make empty views just with shapes?
    ComplexView4DType delta_rho("delta_rho", np, nx, ny, nz);

    ExecutionSpace exec;
    using axes_type = KokkosFFT::axis_type<3>;
    axes_type axes  = {-3, -2, -1};
    m_forward_plan  = std::make_unique<C2CPlanType>(
        exec, delta_rho, delta_rho, KokkosFFT::Direction::forward, axes);
    m_backward_plan = std::make_unique<C2CPlanType>(
        exec, delta_rho, delta_rho, KokkosFFT::Direction::backward, axes);

    // Read Luc's direct correlation function c^{m,n}_{mu,nu_,chi}(|q|)
    // projected on generalized spherical harmonics
    // in the intermolecular frame
    // normq is norm of q, |q|, that correspond to the index iq in ck(ia,iq)
    auto h_kx   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      spatial_grid.m_kx);
    auto h_ky   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      spatial_grid.m_ky);
    auto h_kz   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      spatial_grid.m_kz);
    auto sub_kz = Kokkos::subview(h_kz, Kokkos::pair<int, int>(0, nz / 2 + 1));

    using host_space = Kokkos::DefaultHostExecutionSpace;
    auto kx_max =
        Kokkos::Experimental::max_element("kx_max", host_space(), h_kx);
    auto ky_max =
        Kokkos::Experimental::max_element("ky_max", host_space(), h_ky);
    auto kz_max =
        Kokkos::Experimental::max_element("kz_max", host_space(), sub_kz);

    ArrayType k_max    = {*kx_max, *ky_max, *kz_max};
    auto qmaxnecessary = MDFT::Impl::norm2(k_max);
    m_luc_data = std::make_unique<LucDataType>(filename, angular_grid, np_luc,
                                               qmaxnecessary);

    // Copy dq value into the member of this class
    m_dq = m_luc_data->m_dq;

    // Terrible implementation, needed to improve
    int np_new = 0;
    for (int im = 0; im <= m_mmax; ++im) {
      for (int ikhi = -im; ikhi <= im; ++ikhi) {
        for (int imu2 = 0; imu2 <= im / m_molrotsymorder; ++imu2) {
          for (int in = Kokkos::abs(ikhi); in <= m_mmax; ++in) {
            for (int inu2 = -in / m_molrotsymorder;
                 inu2 <= in / m_molrotsymorder; ++inu2) {
              np_new++;
            }
          }
        }
      }
    }

    // Allocate Views
    auto nq = m_luc_data->m_nq;

    // Initialize h_n_new, h_nu_new, h_khi_new, and h_cnu2nmu2khim_q_new
    using HostIntView1D = Kokkos::View<IntType*, host_space>;
    HostIntView1D h_n_new("n_new", np_new);
    HostIntView1D h_nu_new("nu_new", np_new);
    HostIntView1D h_khi_new("khi_new", np_new);

    // Allocate a member variable
    m_mnmunukhi_q      = ComplexView2DType("mnmunukhi_q", np_new, nq);
    auto h_mnmunukhi_q = Kokkos::create_mirror_view(m_mnmunukhi_q);

    auto h_mnmunukhi_q_fromfile = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), m_luc_data->m_cmnmunukhi);
    auto h_ip = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    m_luc_data->m_p);

    int mmax_mrso = m_mmax / m_molrotsymorder;
    int i         = 0;
    for (int im = 0; im <= m_mmax; ++im) {
      for (int ikhi = -im; ikhi <= im; ++ikhi) {
        for (int imu2 = 0; imu2 <= im / m_molrotsymorder; ++imu2) {
          for (int in = Kokkos::abs(ikhi); in <= m_mmax; ++in) {
            for (int inu2 = -in / m_molrotsymorder;
                 inu2 <= in / m_molrotsymorder; ++inu2) {
              h_n_new(i)   = in;
              h_nu_new(i)  = inu2 * m_molrotsymorder;
              h_khi_new(i) = ikhi;
              for (int iq = 0; iq < nq; iq++) {
                int ip = h_ip(im, in, imu2 + mmax_mrso, inu2 + mmax_mrso,
                              ikhi + m_mmax);
                h_mnmunukhi_q(i, iq) = h_mnmunukhi_q_fromfile(ip, iq);
              }
              i++;
            }
          }
        }
      }
    }

    // Move prefactors of MOZ inside the direct correlation function so that one
    // does not need to compute, for instance,
    // (-1)**(khi+nu) inside the inner loop of MOZ
    for (int ip = 0; ip < np_new; ip++) {
      auto inu = h_nu_new(ip);
      if (inu < 0) {
        auto ikhi = h_khi_new(ip);
        for (int iq = 0; iq < nq; iq++) {
          h_mnmunukhi_q(ip, iq) =
              std::pow(-1.0, (ikhi + inu)) * h_mnmunukhi_q(ip, iq);
        }
      } else {
        auto in = h_n_new(ip);
        for (int iq = 0; iq < nq; iq++) {
          h_mnmunukhi_q(ip, iq) = std::pow(-1.0, in) * h_mnmunukhi_q(ip, iq);
        }
      }
    }
    Kokkos::deep_copy(m_mnmunukhi_q, h_mnmunukhi_q);
  }

 public:
  auto gamma_p_map() const { return m_gamma_p_map; }

  // \brief
  // \tparam View Orientation view, needs to be a Complex View
  //
  // \param deltarho_p [in/out] Orientation (nm * nmup * nmu, nx, ny, nz)
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
    std::size_t na            = m_ia_map.extent(0);
    std::size_t mmax_p1       = static_cast<std::size_t>(m_mmax + 1);
    std::size_t mmax2_p1      = static_cast<std::size_t>(2 * m_mmax + 1);
    std::size_t request_size  = mmax_p1 * mmax2_p1 * mmax2_p1;
    std::size_t request_size2 = m_np;
    int scratch_size          = ScratchView3DType::shmem_size(request_size) +
                       ScratchView1DType::shmem_size(request_size2 * 5);
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
    auto mnmunukhi_q = m_mnmunukhi_q;

    auto kx = m_kx, ky = m_ky, kz = m_kz;
    int nx = m_nx, ny = m_ny, nz = m_nz;
    int np             = m_np;
    int mrso           = m_molrotsymorder;
    int mmax           = m_mmax;
    ScalarType dq      = m_dq;
    ScalarType epsilon = std::numeric_limits<ScalarType>::epsilon() * 10;
    // Loop over nx * ny * (nz/2+1) without overlapping
    Kokkos::parallel_for(
        "convolution", team_policy,
        KOKKOS_LAMBDA(const member_type& team_member) {
          const auto idx = team_member.league_rank();
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
              s_deltarho_p_mq(team_member.team_scratch(level), np),
              s_ceff(team_member.team_scratch(level), np);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                s_gamma_p_q(ip)  = deltarho_p(ip, iz, iy, ix);
                s_gamma_p_mq(ip) = deltarho_p(ip, iz_mq, iy_mq, ix_mq);
                auto im          = p_to_m(ip);
                auto ikphi       = p_to_mup(ip);
                auto imu2        = p_to_mu(ip);
                Kokkos::complex<ScalarType> deltarho_p_q_loc(0.0),
                    deltarho_p_mq_loc(0.0);
                for (int imup = -im; imup <= im; imup++) {
                  auto ip_mapped = p_map(im, imup + mmax, imu2);
                  deltarho_p_q_loc +=
                      s_gamma_p_q(ip) * s_R(im, imup + mmax, ikphi + mmax);
                  deltarho_p_mq_loc +=
                      s_gamma_p_mq(ip) * s_R(im, imup + mmax, -ikphi + mmax);
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

          // We should parallelize over ia
          // Then map it to ip, n, and nu2
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, na), [&](const int ia) {
                const int ip = ia_map(ia, 0), in = ia_map(ia, 1),
                          inu2 = ia_map(ia, 2);

                s_gamma_p_q(ip)  = Kokkos::complex<ScalarType>(0.0, 0.0);
                s_gamma_p_mq(ip) = Kokkos::complex<ScalarType>(0.0, 0.0);

                // Linear interpolation
                s_ceff(ip) = alpha * mnmunukhi_q(ip, iq + 1) +
                             (1.0 - alpha) * mnmunukhi_q(ip, iq);

                auto im   = p_to_m(ip);
                auto ikhi = p_to_mup(ip);

                auto ceff = s_ceff(ia);
                if (inu2 < 0) {
                  auto ip_mapped = p_map(in, ikhi, Kokkos::abs(inu2));
                  s_gamma_p_q(ip) += ceff * s_deltarho_p_q(ip_mapped);
                  s_gamma_p_mq(ip) += ceff * s_deltarho_p_mq(ip_mapped);
                } else {
                  auto ip_mapped = p_map(in, ikhi, inu2);
                  s_gamma_p_q(ip) +=
                      ceff * Kokkos::conj(s_deltarho_p_mq(ip_mapped));
                  s_gamma_p_mq(ip) +=
                      ceff * Kokkos::conj(s_deltarho_p_q(ip_mapped));
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
              q_eq_mq && (ix == nx / 2 && iy == ny / 2 && iz == nz / 2);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                auto im   = p_to_m(ip);
                auto imup = p_to_mup(ip);
                auto imu2 = p_to_mu(ip);

                ComplexType sum_deltarho_p_q(0.0, 0.0),
                    sum_deltarho_p_mq(0.0, 0.0);
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
