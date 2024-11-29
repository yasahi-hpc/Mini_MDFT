#ifndef MDFT_ORIENTATIONPROJECTIONTRANSFORM_HPP
#define MDFT_ORIENTATIONPROJECTIONTRANSFORM_HPP

#include <memory>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Wigner.hpp"

namespace MDFT {

// \brief Storing the mapping between the orientation and the projection
// \tparam ExecutionSpace Execution space
// \tparam ScalarType Scalar type
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct OrientationProjectionMap {
 private:
  using IntType         = int;
  using IntView1DType   = Kokkos::View<IntType*, ExecutionSpace>;
  using IntView3DType   = Kokkos::View<IntType***, ExecutionSpace>;
  using View2DType      = Kokkos::View<ScalarType**, ExecutionSpace>;
  using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;

  //! index of the projection corresponding to m, mup, mu
  IntView3DType m_p;

  //! m for projection 1 to np
  IntView1DType m_m;

  //! mup for projection 1 to np. mup corresponds to phi
  IntView1DType m_mup;

  //! mu for projection 1 to np. mu corresponds to psi
  IntView1DType m_mu;

  //! tabulation des harmoniques sphériques r(m,mup,mu,theta) en un tableau
  //! r(itheta,p)
  View2DType m_wigner_small_d;

 public:
  OrientationProjectionMap()  = delete;
  ~OrientationProjectionMap() = default;

  OrientationProjectionMap(const AngularGridType& angular_grid) {
    auto ntheta = angular_grid.m_ntheta;
    auto nphi   = angular_grid.m_nphi;
    auto npsi   = angular_grid.m_npsi;
    auto np     = angular_grid.m_np;
    auto mmax   = angular_grid.m_mmax;
    auto mrso   = angular_grid.m_molrotsymorder;

    // Allocate Views
    m_p   = IntView3DType("p", mmax + 1, 2 * mmax + 1, mmax / mrso + 1);
    m_m   = IntView1DType("m", np);
    m_mup = IntView1DType("mup", np);
    m_mu  = IntView1DType("mu", np);

    auto h_p = Kokkos::create_mirror_view(m_p);
    auto h_m = Kokkos::create_mirror_view(m_m);

    // h_mup     [0, 1, ..., m-1, m, m+1, ..., 2m-1, 2m]
    // h_mup_neg [-m, -m+1, ... -1, 0, 1, ..., m-1, m]
    auto h_mup     = Kokkos::create_mirror_view(m_mup);
    auto h_mup_neg = Kokkos::create_mirror_view(m_mup);
    auto h_mu      = Kokkos::create_mirror_view(m_mu);

    int ip = 0;

    // [TO DO] Should this be negative index?
    for (int m = 0; m <= mmax; ++m) {
      for (int mup = -m; mup <= m; ++mup) {
        for (int mu = 0; mu <= m / mrso; ++mu) {
          MDFT::Impl::Throw_If(ip >= np, "ip must be smaller than np");
          h_p(m, mup + m, mu) = ip;
          h_m(ip)             = m;
          h_mup(ip)           = mup + m;
          h_mup_neg(ip)       = mu;
          h_mu(ip)            = mu;
          ip++;
        }
      }
    }
    MDFT::Impl::Throw_If(ip != np, "ip after the loops must be equal to np");

    Kokkos::deep_copy(m_p, h_p);
    Kokkos::deep_copy(m_m, h_m);
    Kokkos::deep_copy(m_mup, h_mup);
    Kokkos::deep_copy(m_mu, h_mu);

    // Initialization made on host
    using host_space = Kokkos::DefaultHostExecutionSpace;

    int amax_m = 0, amax_mup = 0, amax_mu = 0;
    Kokkos::parallel_reduce(
        "maximum", Kokkos::RangePolicy<host_space>(0, np),
        KOKKOS_LAMBDA(const int ip, int& lmax_m, int& lmax_mup, int& lmax_mu) {
          lmax_m   = Kokkos::max(lmax_m, Kokkos::abs(h_m(ip)));
          lmax_mup = Kokkos::max(lmax_mup, Kokkos::abs(h_mup_neg(ip)));
          lmax_mu  = Kokkos::max(lmax_mu, Kokkos::abs(h_mu(ip)));
        },
        amax_m, amax_mup, amax_mu);

    MDFT::Impl::Throw_If(amax_m > mmax || amax_mup > mmax || amax_mu > mmax,
                         "Invalid m, mup, mu values");

    // L'initialisation des racines et des poids de la quadrature angulaire a
    // été faite beaucoup plus tot dans le module_grid. On fait attention qu'on
    // a deux tableaux des racines et des poids:
    // - le premier tableau, grid%theteaofntheta, est un tableau de taille  le
    // nombre de theta (ntheta)
    // - le deuxième, grid%theta, est de taille le nombre total d'angle. Il
    // retourne le theta qui correspond à l'indice de chaque angle. Si on veut
    // la liste de tous les theta, on utilisera donc le tableau
    // grid%thetaofntheta.

    // Allocate views
    // (ntheta, mmax + 1, mmax/mrso, mmax*2 + 1)
    m_wigner_small_d = View2DType("wigner_small_d", ntheta, np);

    auto h_wigner_small_d = Kokkos::create_mirror_view(m_wigner_small_d);
    auto h_thetaofntheta  = Kokkos::create_mirror_view_and_copy(
         Kokkos::HostSpace(), angular_grid.m_thetaofntheta);
    // Tabulate generalized spherical harmonics in array
    // p3%wigner_small_d(theta,proj) where theta can be any of the GaussLegendre
    // integration roots for theta where proj is an index related to a tuple
    // {m,mup,mu} Blum's notation : m is related to theta mup is related to phi
    // mu is related to psi
    // TODO: a remplacer par la routine de luc, et utiliser la notation alpha
    // plutot que m,mup,mu a ce moment
    for (int p = 0; p < np; p++) {
      auto m   = h_m(p);
      auto mup = h_mup_neg(p);
      auto mu  = h_mu(p);
      for (int itheta = 0; itheta < ntheta; itheta++) {
        // Pour chaque theta, calcule la fonction de Wigner-d correspondant à
        // toutes les projections avec la méthode de Wigner.
        auto theta = h_thetaofntheta(itheta);
        h_wigner_small_d(itheta, p) =
            MDFT::Impl::wigner_small_d(m, mup, mu, theta);
      }
    }
    Kokkos::deep_copy(m_wigner_small_d, h_wigner_small_d);
  }

  View2DType wigner_small_d() const { return m_wigner_small_d; }
  IntView3DType p() const { return m_p; }
  IntView1DType m() const { return m_m; }
  IntView1DType mup() const { return m_mup; }
  IntView1DType mu() const { return m_mu; }
};

// \brief Class to handle the orientation projection transform
// \tparam ExecutionSpace Execution space
// \tparam ScalarType Scalar type
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
class OrientationProjectionTransform {
 private:
  constexpr static std::size_t FFT_DIM = 2;
  using IntType                        = int;
  using IntView1DType                  = Kokkos::View<IntType*, ExecutionSpace>;
  using IntView3DType = Kokkos::View<IntType***, ExecutionSpace>;
  using View1DType    = Kokkos::View<ScalarType*, ExecutionSpace>;
  using View2DType    = Kokkos::View<ScalarType**, ExecutionSpace>;
  using View3DType    = Kokkos::View<ScalarType***, ExecutionSpace>;
  using View4DType    = Kokkos::View<ScalarType****, ExecutionSpace>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<ScalarType>***, ExecutionSpace>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<ScalarType>****, ExecutionSpace>;
  using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
  using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;
  using OrientationProjectionMapType =
      OrientationProjectionMap<ExecutionSpace, ScalarType>;
  using ForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, View4DType, ComplexView4DType, FFT_DIM>;
  using BackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, ComplexView4DType, View4DType, FFT_DIM>;
  // Internal Scratch View Type
  using ScratchViewType =
      Kokkos::View<Kokkos::complex<ScalarType>***,
                   typename ExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<ForwardPlanType> m_rfft2_plan;
  std::unique_ptr<BackwardPlanType> m_irfft2_plan;

  View1DType m_wtheta;

  //! sqrt(2m+1), shape (mmax+1)
  View1DType m_fm;

  // tabulation des harmoniques sphériques r(m,mup,mu,theta) en un tableau
  OrientationProjectionMapType m_map;

  //! Complex buffer for the rfft2 and irfft2 (nx * ny * nz, ntheta, nphi, npsi)
  ComplexView4DType m_o_hat;

 public:
  OrientationProjectionTransform()  = delete;
  ~OrientationProjectionTransform() = default;

  OrientationProjectionTransform(const SpatialGridType& spatial_grid,
                                 const AngularGridType& angular_grid,
                                 const OrientationProjectionMapType& map)
      : m_map(map) {
    auto ntheta = angular_grid.m_ntheta;
    auto nphi   = angular_grid.m_nphi;
    auto npsi   = angular_grid.m_npsi;
    auto m_max  = angular_grid.m_mmax;

    // Allocate Views
    m_wtheta = View1DType("wtheta", ntheta);
    m_fm     = View1DType("fm", m_max + 1);

    // Initialization
    Kokkos::deep_copy(m_wtheta, angular_grid.m_thetaofntheta);
    auto h_fm = Kokkos::create_mirror_view(m_fm);
    for (int m = 0; m <= m_max; ++m) {
      h_fm(m) = std::sqrt(2.0 * static_cast<ScalarType>(m) + 1.0);
    }
    Kokkos::deep_copy(m_fm, h_fm);

    // Prepare FFT plans
    // 2D batched plan (nbatch, nphi, npsi)
    ExecutionSpace exec;
    auto nx            = spatial_grid.m_nx;
    auto ny            = spatial_grid.m_ny;
    auto nz            = spatial_grid.m_nz;
    std::size_t nbatch = nz * ny * nx;

    // Can we make empty views just with shapes?
    View4DType o("o", nbatch, ntheta, nphi, npsi);
    m_o_hat = ComplexView4DType("o_hat", nbatch, ntheta, nphi, npsi / 2 + 1);

    using axes_type = KokkosFFT::axis_type<2>;
    axes_type axes  = {-2, -1};
    m_rfft2_plan    = std::make_unique<ForwardPlanType>(
        exec, o, m_o_hat, KokkosFFT::Direction::forward, axes);
    m_irfft2_plan = std::make_unique<BackwardPlanType>(
        exec, m_o_hat, o, KokkosFFT::Direction::backward, axes);
  }

 public:
  // \brief
  // \tparam OView Orientation view, needs to be a rank 4 Real View
  // \tparam PView Projection view, needs to be a rank 2 Complex View
  //
  // \param o [in] Orientation (nx * ny * nz, theta, phi, psi)
  // \param p [out] Projection (np, nx * ny * nz)
  // [R.K] mu2 -> psi, mup -> phi
  template <KokkosView OView, KokkosView PView>
  requires KokkosViewAccesible<ExecutionSpace, OView> &&
      KokkosViewAccesible<ExecutionSpace, PView>
  void angl2proj(const OView& o, const PView& p) {
    int N = o.extent(0), ntheta = o.extent(1), nphi = o.extent(2),
        npsi = o.extent(3);

    // We may use an inplace transform here
    // Need to represent o and o_hat in another way using Unmanaged views
    // delta_rho_hat (r, ktheta, kphi, kpsi) = FFT [delta_rho (r, theta, phi,
    // psi)]
    m_rfft2_plan->execute(
        o, m_o_hat,
        KokkosFFT::Normalization::none);  // Plan should have extents method?

    auto p_map    = m_map.p();
    int mmax_p1   = p_map.extent(0);
    int mmax2_p1  = p_map.extent(1);
    int mmax_mrso = p_map.extent(2);
    int mmax      = mmax_p1 - 1;
    using member_type =
        typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

    // std::size_t request_size = static_cast<std::size_t>(ntheta) * mmax2_p1 *
    // mmax_mrso;
    int scratch_size = ScratchViewType::shmem_size(ntheta, mmax2_p1, mmax_mrso);
    int level        = 1;  // using global memory
    auto team_policy =
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(scratch_size));

    // Need to represent o_hat
    // delta_rho_hat^m_{\mu', \mu} (r) = fm \int \int \int delta_rho_hat (r,
    // ktheta, kphi, kpsi) R(m, mup, mu, ktheta, kphi, kpsi) d\Omega
    auto o_hat_tmp = m_o_hat;
    auto fm        = m_fm;
    auto p_to_m    = m_map.m();
    auto p_to_mup  = m_map.mup();
    auto p_to_mu   = m_map.mu();

    auto wtheta         = m_wtheta;
    auto wigner_small_d = m_map.wigner_small_d();
    int np              = p_to_m.extent(0);

    using value_type = typename PView::non_const_value_type;
    Kokkos::parallel_for(
        "to_projection", team_policy,
        KOKKOS_LAMBDA(const member_type& team_member) {
          const auto idx = team_member.league_rank();

          auto sub_o_hat = Kokkos::subview(o_hat_tmp, idx, Kokkos::ALL,
                                           Kokkos::ALL, Kokkos::ALL);
          auto sub_p     = Kokkos::subview(p, Kokkos::ALL, idx);

          // s_f, Shape (ntheta, mmax*2 + 1, mmax/mrso)
          ScratchViewType s_f(team_member.team_scratch(level), ntheta, mmax2_p1,
                              mmax_mrso);

          // Loop over theta, 0:mmax, mmax/mrso
          Kokkos::parallel_for(
              Kokkos::ThreadVectorMDRange<Kokkos::Rank<3>, member_type>(
                  team_member, ntheta, mmax_p1, mmax_mrso),
              [&](const int itheta, const int iphi, const int ipsi) {
                int iphi_shift = iphi + mmax;
                int iphi_neg   = iphi - 1;

                // Used for the update of negative part
                // src: mmax -> dst: mmax * 2, which is also updated by the
                // positive part
                if (iphi == 0) iphi_neg = mmax * 2;

                // [0, 1, 2, ...] -> [mmax, mmax+1, mmax+2, ...]
                s_f(itheta, iphi_shift, ipsi) =
                    Kokkos::conj(sub_o_hat(itheta, iphi, ipsi)) /
                    static_cast<value_type>(nphi * npsi);

                // [mmax, mmax+1, ...]-> [-mmax, -mmax+1, -mmax+2, ...]
                s_f(itheta, iphi_neg, ipsi) =
                    Kokkos::conj(sub_o_hat(itheta, iphi_shift, ipsi)) /
                    static_cast<value_type>(nphi * npsi);
              });

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, np), [&](const int ip) {
                auto m_idx   = p_to_m(ip);
                auto mup_idx = p_to_mup(ip);
                auto mu_idx  = p_to_mu(ip);
                auto sub_f = Kokkos::subview(s_f, Kokkos::ALL, mup_idx, mu_idx);
                auto sub_wigner_small_d =
                    Kokkos::subview(wigner_small_d, Kokkos::ALL, ip);
                auto tmp_fm    = fm(m_idx);
                value_type sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, ntheta),
                    [&](const int itheta, value_type& lsum) {
                      lsum += sub_f(itheta) * sub_wigner_small_d(itheta) *
                              wtheta(itheta) * tmp_fm;
                    },
                    sum);
                sub_p(ip) = sum;
              });
        });
  }

  // \brief
  // \tparam PView Projection view, needs to be a rank 2 Complex View
  // \tparam OView Orientation view, needs to be a rank 4 Real View
  //
  // \param p [in] Projection (np, nx * ny * nz)
  // \param o [out] Orientation (nx * ny * nz, theta, phi, psi)
  template <KokkosView PView, KokkosView OView>
  requires KokkosViewAccesible<ExecutionSpace, PView> &&
      KokkosViewAccesible<ExecutionSpace, OView>
  void proj2angl(const PView& p, const OView& o) {
    int N = o.extent(0), ntheta = o.extent(1), nphi = o.extent(2),
        npsi = o.extent(3);

    // Need to represent p
    auto fm             = m_fm;
    auto p_to_m         = m_map.m();
    auto p_map          = m_map.p();
    auto wtheta         = m_wtheta;
    auto wigner_small_d = m_map.wigner_small_d();
    int np              = p_to_m.extent(0);

    auto o_hat_tmp = m_o_hat;

    int mrso      = 0;
    int mmax_p1   = p_map.extent(0);
    int mmax2_p1  = p_map.extent(1);
    int mmax_mrso = p_map.extent(2);
    int mmax      = mmax_p1 - 1;
    using member_type =
        typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

    // std::size_t request_size = static_cast<std::size_t>(ntheta) * mmax2_p1 *
    // mmax_mrso;
    int scratch_size = ScratchViewType::shmem_size(ntheta, mmax2_p1, mmax_mrso);
    int level        = 1;  // using global memory
    auto team_policy =
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(scratch_size));

    using value_type = typename PView::non_const_value_type;
    Kokkos::parallel_for(
        "to_angle", team_policy, KOKKOS_LAMBDA(const member_type& team_member) {
          const auto idx = team_member.league_rank();

          auto sub_p     = Kokkos::subview(p, Kokkos::ALL, idx);
          auto sub_o_hat = Kokkos::subview(o_hat_tmp, idx, Kokkos::ALL,
                                           Kokkos::ALL, Kokkos::ALL);

          // s_f, Shape (ntheta, mmax*2 + 1, mmax/mrso)
          ScratchViewType s_f(team_member.team_scratch(level), ntheta, mmax2_p1,
                              mmax_mrso);

          // Loop over theta, mmax*2 + 1, mmax/mrso
          Kokkos::parallel_for(
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, member_type>(
                  team_member, ntheta, mmax2_p1, mmax_mrso),
              [&](const int itheta, const int imup, const int imu2) {
                value_type sum = 0;
                int m_init =
                    Kokkos::max(Kokkos::abs(imup), mrso * Kokkos::abs(imu2));
                for (int im = m_init; im < mmax_p1; ++im) {
                  auto ip = p_map(im, imup, imu2);
                  sum += sub_p(ip) * wigner_small_d(itheta, ip) * fm(im);
                }

                s_f(itheta, imup, imu2) = Kokkos::conj(sum);
              });

          team_member.team_barrier();

          // Loop over theta, 0:mmax, mmax/mrso
          Kokkos::parallel_for(
              Kokkos::ThreadVectorMDRange<Kokkos::Rank<3>, member_type>(
                  team_member, ntheta, mmax_p1, mmax_mrso),
              [&](const int itheta, const int iphi, const int ipsi) {
                int iphi_shift = iphi + mmax;
                int iphi_neg   = iphi - 1;

                // Used for the update of negative part
                // src: mmax -> dst: mmax * 2, which is also updated by the
                // positive part
                if (iphi == 0) iphi_neg = mmax * 2;

                // [mmax, mmax+1, mmax+2, ...] -> [0, 1, 2, ...]
                sub_o_hat(itheta, iphi, ipsi) = s_f(itheta, iphi_shift, ipsi);

                // [-mmax, -mmax+1, -mmax+2, ...] -> [mmax, mmax+1, mmax+2, ...]
                sub_o_hat(itheta, iphi_shift, ipsi) =
                    s_f(itheta, iphi_neg, ipsi);
              });
        });

    // We may use an inplace transform here
    // gamma (r, thata, phi, psi) = IFFT [gamma (r, ktheta, kphi, kpsi)]
    m_irfft2_plan->execute(m_o_hat, o, KokkosFFT::Normalization::none);
  }
};
};  // namespace MDFT

#endif
