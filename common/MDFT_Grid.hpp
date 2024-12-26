#ifndef MDFT_GRID_HPP
#define MDFT_GRID_HPP

#include <numeric>
#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosFFT.hpp>
#include "MDFT_Asserts.hpp"
#include "MDFT_Concepts.hpp"
#include "MDFT_System.hpp"
#include "MDFT_Math_Utils.hpp"

namespace MDFT {
/// \brief A class that manages Spatial Grid
///
/// \tparam ScalarType The type of the scalar values.
///
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct SpatialGrid {
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 3>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;
  using View1DType      = typename Kokkos::View<ScalarType*, ExecutionSpace>;

  // Spatial grid
  // number of grid nodes in direction x, y and z
  IntArrayType m_n_nodes, m_n;
  IntType m_nx, m_ny, m_nz;

  // total length in direction x, y and z
  ScalarArrayType m_length, m_l;
  ScalarType m_lx, m_ly, m_lz;

  // elemental distance between two nodes in direction x, y and z
  ScalarArrayType m_dl;
  ScalarType m_dx, m_dy, m_dz;
  ScalarType m_dv;  // elemental volume
  ScalarType m_v;
  ScalarType m_buffer_length;  // length of free space between the extremam of
                               // the solute.

  View1DType m_kx, m_ky, m_kz;

  SpatialGrid(IntArrayType n_nodes, ScalarArrayType length)
      : m_n_nodes(n_nodes), m_length(length) {
    MDFT::Impl::Throw_If(
        std::any_of(Kokkos::begin(m_length), Kokkos::end(m_length),
                    [](ScalarType value) { return value <= 0; }),
        "The supercell cannot have negative length.");
    MDFT::Impl::Throw_If(
        std::any_of(Kokkos::begin(m_n_nodes), Kokkos::end(m_n_nodes),
                    [](ScalarType value) { return value <= 0; }),
        "The space is divided into grid nodes.");
    m_nx = m_n_nodes[0], m_ny = m_n_nodes[1], m_nz = m_n_nodes[2];
    m_lx = m_length[0], m_ly = m_length[1], m_lz = m_length[2];
    // std::tie(m_nx, m_ny, m_nz) = m_n_nodes;
    // std::tie(m_lx, m_ly, m_lz) = m_length;

    std::transform(
        Kokkos::begin(m_length), Kokkos::end(m_length),
        Kokkos::begin(m_n_nodes), Kokkos::begin(m_dl),
        [](ScalarType l, IntType n) { return l / static_cast<ScalarType>(n); });

    m_v  = std::accumulate(Kokkos::begin(m_length), Kokkos::end(m_length), 1.0,
                           std::multiplies<ScalarType>());
    m_dv = std::accumulate(Kokkos::begin(m_dl), Kokkos::end(m_dl), 1.0,
                           std::multiplies<ScalarType>());

    m_kx = KokkosFFT::fftfreq(ExecutionSpace(), m_nx, m_dl[0]);
    m_ky = KokkosFFT::fftfreq(ExecutionSpace(), m_ny, m_dl[1]);
    m_kz = KokkosFFT::fftfreq(ExecutionSpace(), m_nz, m_dl[2]);
  }
};

template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct AngularGrid {
  using execSpace       = ExecutionSpace;
  using IntType         = int;
  using IntArrayType    = std::array<IntType, 3>;
  using ScalarArrayType = std::array<ScalarType, 3>;
  using View1DType      = typename Kokkos::View<ScalarType*, ExecutionSpace>;
  using View3DType      = typename Kokkos::View<ScalarType***, ExecutionSpace>;

  // Angular grid .. angular quadrature
  IntType m_mmax, m_molrotsymorder, m_ntheta, m_nphi, m_npsi, m_no, m_np;
  ScalarType m_dphi, m_dpsi;
  ScalarType m_quadrature_norm = 8.0 * M_PI * M_PI;

  View1DType m_theta, m_phi, m_psi, m_wtheta, m_wphi, m_wpsi, m_w;
  View1DType m_thetaofntheta, m_wthetaofntheta;
  View1DType m_phiofnphi, m_psiofnpsi, m_wphiofnphi, m_wpsiofnpsi;
  View3DType m_indo;  // table of index of orientations
  View3DType m_io;
  View1DType m_rotxx, m_rotxy, m_rotxz, m_rotyx, m_rotyy, m_rotyz, m_rotzx,
      m_rotzy, m_rotzz;
  View1DType m_OMx, m_OMy, m_OMz;

  AngularGrid(int mmax, int molrotsymorder)
      : m_mmax(mmax), m_molrotsymorder(molrotsymorder) {
    MDFT::Impl::Throw_If(mmax <= 0, "The grid mmax must be greater than zero.");
    MDFT::Impl::Throw_If(molrotsymorder <= 0,
                         "The grid molrotsymorder must be greater than zero.");

    m_ntheta = m_mmax + 1;
    m_nphi   = 2 * m_mmax + 1;
    m_npsi   = 2 * (m_mmax / m_molrotsymorder) + 1;

    // Number of orientations in the Euler representation
    m_no   = m_ntheta * m_nphi * m_npsi;
    m_dphi = M_PI / static_cast<ScalarType>(m_nphi);
    m_dpsi = M_PI / static_cast<ScalarType>(m_molrotsymorder) /
             static_cast<ScalarType>(m_npsi);
    m_np = 0;

    for (int m = 0; m <= m_mmax; ++m) {
      for (int mup = -m; mup <= m; ++mup) {
        for (int mu = 0; mu <= m / m_molrotsymorder; ++mu) {
          m_np++;
        }
      }
    }

    // Allocate views
    m_theta          = View1DType("theta", m_no);
    m_phi            = View1DType("phi", m_no);
    m_psi            = View1DType("psi", m_no);
    m_wtheta         = View1DType("wtheta", m_no);
    m_wphi           = View1DType("wphi", m_no);
    m_wpsi           = View1DType("wpsi", m_no);
    m_wthetaofntheta = View1DType("wthetaofntheta", m_ntheta);
    m_w              = View1DType("w", m_no);
    m_thetaofntheta  = View1DType("thetaofntheta", m_ntheta);
    m_phiofnphi      = View1DType("phiofnphi", m_nphi);
    m_psiofnpsi      = View1DType("psiofnpsi", m_npsi);
    m_wphiofnphi     = View1DType("wphiofnphi", m_nphi);
    m_wpsiofnpsi     = View1DType("wpsiofnpsi", m_npsi);

    // Initialization made on host
    using host_space  = Kokkos::DefaultHostExecutionSpace;
    auto h_wphiofnphi = Kokkos::create_mirror_view(m_wphiofnphi);
    auto h_wpsiofnpsi = Kokkos::create_mirror_view(m_wpsiofnpsi);
    Kokkos::Experimental::fill(host_space(), h_wphiofnphi,
                               1.0 / static_cast<ScalarType>(m_nphi));
    Kokkos::Experimental::fill(
        host_space(), h_wpsiofnpsi,
        1.0 / static_cast<ScalarType>(m_npsi * m_molrotsymorder));
    Kokkos::deep_copy(m_wphiofnphi, h_wphiofnphi);
    Kokkos::deep_copy(m_wpsiofnpsi, h_wpsiofnpsi);

    auto h_psiofnpsi = Kokkos::create_mirror_view(m_psiofnpsi);
    auto h_phiofnphi = Kokkos::create_mirror_view(m_phiofnphi);
    MDFT::Impl::uniform_mesh(host_space(), h_psiofnpsi, m_dpsi);
    MDFT::Impl::uniform_mesh(host_space(), h_phiofnphi, m_dphi);
    Kokkos::deep_copy(m_psiofnpsi, h_psiofnpsi);
    Kokkos::deep_copy(m_phiofnphi, h_phiofnphi);

    // This function works on device views
    MDFT::Impl::gauss_legendre(m_thetaofntheta, m_wthetaofntheta);

    auto h_thetaofntheta = Kokkos::create_mirror_view(m_thetaofntheta);
    Kokkos::Experimental::for_each(
        "acos", host_space(), h_thetaofntheta,
        KOKKOS_LAMBDA(ScalarType & theta) { theta = Kokkos::acos(theta); });
    Kokkos::deep_copy(m_thetaofntheta, h_thetaofntheta);

    m_indo = View3DType("indo", m_ntheta, m_nphi, m_npsi);
    m_io   = View3DType("io", m_ntheta, m_nphi, m_npsi);

    using range_type = Kokkos::MDRangePolicy<
        host_space,
        Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
    using tile_type  = typename range_type::tile_type;
    using point_type = typename range_type::point_type;

    range_type range(point_type{0, 0, 0}, point_type{m_npsi, m_nphi, m_ntheta},
                     tile_type{4, 4, 4});

    auto h_theta          = Kokkos::create_mirror_view(m_theta);
    auto h_phi            = Kokkos::create_mirror_view(m_phi);
    auto h_psi            = Kokkos::create_mirror_view(m_psi);
    auto h_io             = Kokkos::create_mirror_view(m_io);
    auto h_indo           = Kokkos::create_mirror_view(m_indo);
    auto h_wtheta         = Kokkos::create_mirror_view(m_wtheta);
    auto h_wphi           = Kokkos::create_mirror_view(m_wphi);
    auto h_wpsi           = Kokkos::create_mirror_view(m_wpsi);
    auto h_w              = Kokkos::create_mirror_view(m_w);
    auto h_wthetaofntheta = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), m_wthetaofntheta);

    auto quadrature_norm = m_quadrature_norm;
    auto npsi            = m_npsi;
    auto nphi            = m_nphi;
    Kokkos::parallel_for(
        "init_angles", range, KOKKOS_LAMBDA(int ipsi, int iphi, int itheta) {
          int io                   = ipsi + iphi * npsi + itheta * npsi * nphi;
          h_theta(io)              = h_thetaofntheta(itheta);
          h_phi(io)                = h_phiofnphi(iphi);
          h_psi(io)                = h_psiofnpsi(ipsi);
          h_io(itheta, iphi, ipsi) = io;
          h_indo(itheta, iphi, ipsi) = io;
          h_wtheta(io)               = h_wthetaofntheta(itheta);
          h_wphi(io)                 = h_wphiofnphi(iphi);
          h_wpsi(io)                 = h_wpsiofnpsi(ipsi);
          h_w(io) = h_wtheta(io) * h_wphi(io) * h_wpsi(io) * quadrature_norm;
        });
    Kokkos::deep_copy(m_theta, h_theta);
    Kokkos::deep_copy(m_phi, h_phi);
    Kokkos::deep_copy(m_psi, h_psi);
    Kokkos::deep_copy(m_io, h_io);
    Kokkos::deep_copy(m_indo, h_indo);
    Kokkos::deep_copy(m_wtheta, h_wtheta);
    Kokkos::deep_copy(m_wphi, h_wphi);
    Kokkos::deep_copy(m_wpsi, h_wpsi);
    Kokkos::deep_copy(m_w, h_w);

    // check if sum(m_w) == quadrature_norm/molrotsymorder
    ScalarType sum_w = 0;
    for (int i = 0; i < m_no; ++i) {
      sum_w += h_w(i);
    }
    auto eps       = std::numeric_limits<ScalarType>::epsilon() * 100;
    bool not_close = Kokkos::abs(sum_w - quadrature_norm / m_molrotsymorder) >
                     (eps * Kokkos::abs(sum_w));

    MDFT::Impl::Throw_If(not_close,
                         "sum(m_w) != quadrature_norm/molrotsymorder");

    // Allocate views
    m_OMx   = View1DType("OMx", m_no);
    m_OMy   = View1DType("OMy", m_no);
    m_OMz   = View1DType("OMz", m_no);
    m_rotxx = View1DType("rotxx", m_no);
    m_rotxy = View1DType("rotxy", m_no);
    m_rotxz = View1DType("rotxz", m_no);
    m_rotyx = View1DType("rotyx", m_no);
    m_rotyy = View1DType("rotyy", m_no);
    m_rotyz = View1DType("rotyz", m_no);
    m_rotzx = View1DType("rotzx", m_no);
    m_rotzy = View1DType("rotzy", m_no);
    m_rotzz = View1DType("rotzz", m_no);

    auto h_OMx   = Kokkos::create_mirror_view(m_OMx);
    auto h_OMy   = Kokkos::create_mirror_view(m_OMy);
    auto h_OMz   = Kokkos::create_mirror_view(m_OMz);
    auto h_rotxx = Kokkos::create_mirror_view(m_rotxx);
    auto h_rotxy = Kokkos::create_mirror_view(m_rotxy);
    auto h_rotxz = Kokkos::create_mirror_view(m_rotxz);
    auto h_rotyx = Kokkos::create_mirror_view(m_rotyx);
    auto h_rotyy = Kokkos::create_mirror_view(m_rotyy);
    auto h_rotyz = Kokkos::create_mirror_view(m_rotyz);
    auto h_rotzx = Kokkos::create_mirror_view(m_rotzx);
    auto h_rotzy = Kokkos::create_mirror_view(m_rotzy);
    auto h_rotzz = Kokkos::create_mirror_view(m_rotzz);

    Kokkos::parallel_for(
        "init_rotations", range, KOKKOS_LAMBDA(int ipsi, int iphi, int itheta) {
          int io         = ipsi + iphi * npsi + itheta * npsi * nphi;
          auto cos_theta = Kokkos::cos(h_theta(io));
          auto sin_theta = Kokkos::sin(h_theta(io));
          auto cos_phi   = Kokkos::cos(h_phi(io));
          auto sin_phi   = Kokkos::sin(h_phi(io));
          auto cos_psi   = Kokkos::cos(h_psi(io));
          auto sin_psi   = Kokkos::sin(h_psi(io));
          h_OMx(io)      = sin_theta * cos_phi;
          h_OMy(io)      = sin_theta * sin_phi;
          h_OMz(io)      = cos_theta;

          h_rotxx(io) = cos_theta * cos_phi * cos_psi - sin_phi * sin_psi;
          h_rotxy(io) = -cos_theta * cos_phi * sin_psi - sin_phi * cos_psi;
          h_rotxz(io) = sin_theta * cos_phi;

          h_rotyx(io) = cos_theta * sin_phi * cos_psi + cos_phi * sin_psi;
          h_rotyy(io) = -cos_theta * sin_phi * sin_psi + cos_phi * cos_psi;
          h_rotyz(io) = sin_theta * sin_phi;

          h_rotzx(io) = -sin_theta * cos_psi;
          h_rotzy(io) = sin_theta * sin_psi;
          h_rotzz(io) = cos_theta;
        });
    Kokkos::deep_copy(m_OMx, h_OMx);
    Kokkos::deep_copy(m_OMy, h_OMy);
    Kokkos::deep_copy(m_OMz, h_OMz);
    Kokkos::deep_copy(m_rotxx, h_rotxx);
    Kokkos::deep_copy(m_rotxy, h_rotxy);
    Kokkos::deep_copy(m_rotxz, h_rotxz);
    Kokkos::deep_copy(m_rotyx, h_rotyx);
    Kokkos::deep_copy(m_rotyy, h_rotyy);
    Kokkos::deep_copy(m_rotyz, h_rotyz);
    Kokkos::deep_copy(m_rotzx, h_rotzx);
    Kokkos::deep_copy(m_rotzy, h_rotzy);
    Kokkos::deep_copy(m_rotzz, h_rotzz);
  }
};

template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
void init_grid(
    const Settings<ScalarType>& settings,
    std::unique_ptr<SpatialGrid<ExecutionSpace, ScalarType>>& grid,
    std::unique_ptr<AngularGrid<ExecutionSpace, ScalarType>>& angular_grid) {
  grid = std::make_unique<SpatialGrid<ExecutionSpace, ScalarType>>(
      settings.m_boxnod, settings.m_boxlen);
  angular_grid = std::make_unique<AngularGrid<ExecutionSpace, ScalarType>>(
      settings.m_mmax, settings.m_molrotsymorder);
}
}  // namespace MDFT

#endif  // MDFT_GRID_HPP
