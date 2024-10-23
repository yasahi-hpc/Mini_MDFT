#ifndef MDFT_ORIENTATIONPROJECTIONTRANSFORM_HPP
#define MDFT_ORIENTATIONPROJECTIONTRANSFORM_HPP

#include <memory>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Wigner.hpp"

namespace MDFT {
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
class OrientationProjectionTransform {
 private:
  constexpr static std::size_t FFT_DIM = 2;
  using IntType                        = int;
  using IntView1DType                  = Kokkos::View<IntType*, ExecutionSpace>;
  using IntView3DType = Kokkos::View<IntType***, ExecutionSpace>;
  using View1DType    = Kokkos::View<ScalarType*, ExecutionSpace>;
  using View2DType    = Kokkos::View<ScalarType**, ExecutionSpace>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<ScalarType>**, ExecutionSpace>;
  using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
  using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;
  using ForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, View2DType, ComplexView2DType, FFT_DIM>;
  using BackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, ComplexView2DType, View2DType, FFT_DIM>;
  //! Dynamically allocatable fft plan.
  std::unique_ptr<ForwardPlanType> m_forward_plan;
  std::unique_ptr<BackwardPlanType> m_backward_plan;

  View1DType m_wtheta;

  //! sqrt(2m+1)
  View1DType m_fm;

  //! tabulation des harmoniques sphériques r(m,mup,mu,theta) en un tableau
  //! r(itheta,p)
  View2DType m_wigner_small_d;

  //! index of the projection corresponding to m, mup, mu
  IntView3DType m_p;

  //! m for projection 1 to np
  IntView1DType m_m;

  //! mup for projection 1 to np. mup corresponds to phi
  IntView1DType m_mup;

  //! mu for projection 1 to np. mu corresponds to psi
  IntView1DType m_mu;

 public:
  OrientationProjectionTransform()  = delete;
  ~OrientationProjectionTransform() = default;

  OrientationProjectionTransform(SpatialGridType& spatial_grid,
                                 AngularGridType& angular_grid) {
    auto ntheta = angular_grid.m_ntheta;
    auto nphi   = angular_grid.m_nphi;
    auto npsi   = angular_grid.m_npsi;
    auto np     = angular_grid.m_np;
    auto mmax   = angular_grid.m_mmax;
    auto mrso   = angular_grid.m_molrotsymorder;

    // Allocate Views
    m_wtheta = View1DType("wtheta", ntheta);
    m_p      = IntView3DType("p", mmax + 1, 2 * mmax + 1, mmax / mrso + 1);
    m_m      = IntView1DType("m", np);
    m_mup    = IntView1DType("mup", np);
    m_mu     = IntView1DType("mu", np);

    auto h_p   = Kokkos::create_mirror_view(m_p);
    auto h_m   = Kokkos::create_mirror_view(m_m);
    auto h_mup = Kokkos::create_mirror_view(m_mup);
    auto h_mu  = Kokkos::create_mirror_view(m_mu);

    int ip = 0;

    // [TO DO] Should this be negative index?
    for (int m = 0; m <= mmax; ++m) {
      for (int mup = -m; mup <= m; ++mup) {
        for (int mu = 0; mu <= m / mrso; ++mu) {
          MDFT::Impl::Throw_If(ip >= np, "ip must be smaller than np");
          h_p(m, mup + m, mu) = ip;
          h_m(ip)             = m;
          h_mup(ip)           = mup;
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
          lmax_mup = Kokkos::max(lmax_mup, Kokkos::abs(h_mup(ip)));
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
    Kokkos::deep_copy(m_wtheta, angular_grid.m_wthetaofntheta);

    // Allocate views
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
      auto mup = h_mup(p);
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
};
};  // namespace MDFT

#endif