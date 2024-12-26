#ifndef MDFT_SOLVENT_HPP
#define MDFT_SOLVENT_HPP

#include <array>
#include <memory>
#include <string_view>
#include <nlohmann/json.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Constants.hpp"
#include "MDFT_Thermo.hpp"
#include "IO/MDFT_IO_Utils.hpp"

namespace MDFT {

namespace Impl {
static constexpr auto allowed_solvents = std::array<std::string_view, 5>{
    "spce", "spce-h", "spce-m", "tip3p", "tip3p-m"};

// use consteval to eliminate runtime conversions, zero runtime overhead!
consteval int const_solvent_index(std::string_view s) {
  for (std::size_t i = 0; i < allowed_solvents.size(); ++i) {
    if (std::string_view{s} == allowed_solvents[i]) return int(i);
  }
  return -1;
}

int solvent_index(std::string_view s) {
  for (std::size_t i = 0; i < allowed_solvents.size(); ++i) {
    if (std::string_view{s} == allowed_solvents[i]) return int(i);
  }
  return -1;
}

}  // namespace Impl

template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct Correlationfunction {
  using View1DType = typename Kokkos::View<ScalarType*, ExecutionSpace>;

  std::string m_filename;

  View1DType m_x;
  View1DType m_y;
};

/**
 * @brief A structure representing a solvent in the MDFT framework.
 *
 * @tparam ExecutionSpace The Kokkos execution space.
 * @tparam ScalarType The type of scalar values used.
 */
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct Solvent {
  using IntType          = int;
  using IntArrayType     = Kokkos::Array<IntType, 6>;
  using StaticView1DType = typename Kokkos::View<ScalarType[3], ExecutionSpace>;
  using StaticView2DType =
      typename Kokkos::View<ScalarType[3][3], ExecutionSpace>;
  using StaticView3DType =
      typename Kokkos::View<ScalarType[3][3][3], ExecutionSpace>;
  using StaticView4DType =
      typename Kokkos::View<ScalarType[3][3][3][3], ExecutionSpace>;
  using View4DType = typename Kokkos::View<ScalarType****, ExecutionSpace>;
  using ComplexView4DType =
      typename Kokkos::View<Kokkos::complex<ScalarType>****, ExecutionSpace>;
  using ComplexView5DType =
      typename Kokkos::View<Kokkos::complex<ScalarType>*****, ExecutionSpace>;
  using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
  using SiteType        = Site<ScalarType>;
  using SettingsType    = Settings<ScalarType>;
  using CorrelationfunctionType =
      Correlationfunction<ExecutionSpace, ScalarType>;

  /**
   * @brief Name of the solvent.
   */
  std::string m_name;

  /**
   * @brief Molecular rotational symmetry order.
   */
  int m_molrotsymorder;

  /**
   * @brief Number of sites of the solvent molecule.
   */
  int m_nsite;

  /**
   * @brief Number of solvent species.
   */
  // int m_nspec;

  /**
   * @brief Monopole moment of the solute.
   */
  ScalarType m_monopole;

  /**
   * @brief Dipole moment of the solute.
   */
  StaticView1DType m_dipole;

  /**
   * @brief Quadrupole moment of the solute.
   */
  StaticView2DType m_quadrupole;

  /**
   * @brief Octupole moment of the solute.
   */
  StaticView3DType m_octupole;

  /**
   * @brief Hexadecupole moment of the solute.
   */
  StaticView4DType m_hexadecupole;

  ScalarType m_hs_radius;

  /**
   * @brief Hard sphere diameter of the solute.
   */
  ScalarType m_diameter;

  /**
   * @brief xi**2=rho/rho0 (io, ix, iy, iz)
   */
  View4DType m_xi;

  /**
   * @brief Vector of sites in the solute.
   */
  std::vector<SiteType> m_site;

  // number density of the homogeneous reference fluid in molecules per
  // Angstrom^3, e.g., 0.033291 molecule.A**-3 for water
  ScalarType m_n0;

  // number density per orientation of the homogeneous reference fluid in
  // molecules per Angstrom^3 per orient
  ScalarType m_rho0;

  // charge factor
  ComplexView4DType m_sigma_k;

  // molecule polarization factor
  ComplexView5DType m_molec_polar_k;

  View4DType m_vext;

  View4DType m_vextq;

  // 36.something is the maximum value of v so that exp(-beta.v) does not return
  // under
  ScalarType m_vext_threeshold = Constants::qfact / 2.0;

  ScalarType m_mole_fraction = 1.0;

  CorrelationfunctionType m_cs;
  CorrelationfunctionType m_cdelta;
  CorrelationfunctionType m_cd;

  // relative permittivity == static dielectric constant = dielectric constant =
  // coonstante diélectrique
  ScalarType m_relativePermittivity;

  IntArrayType m_npluc;

  IntType m_n_line_cfile;

 public:
  void init(std::string name, ScalarType hs_radius, int nsite,
            int molrotsymorder, std::vector<ScalarType> q,
            std::vector<ScalarType> sig, std::vector<ScalarType> eps,
            std::vector<ScalarType> r0, std::vector<ScalarType> r1,
            std::vector<ScalarType> r2, std::vector<IntType> Z, ScalarType n0,
            ScalarType rho0, ScalarType relativePermittivity,
            IntArrayType npluc, IntType n_line_cfile) {
    m_name           = name;
    m_hs_radius      = hs_radius;
    m_nsite          = nsite;
    m_molrotsymorder = molrotsymorder;

    for (int i = 0; i < m_nsite; i++) {
      m_site.push_back(
          SiteType("", q.at(i), sig.at(i), eps.at(i),
                   Kokkos::Array<ScalarType, 3>({r0.at(i), r1.at(i), r2.at(i)}),
                   Z.at(i)));
    }

    m_n0                   = n0;
    m_rho0                 = rho0;
    m_relativePermittivity = relativePermittivity;
    m_npluc                = npluc;
    m_n_line_cfile         = n_line_cfile;

    // compute monopole, dipole, quadrupole, octupole and hexadecapole of each
    // solvent species

    // View allocation
    m_dipole       = StaticView1DType("dipole");
    m_quadrupole   = StaticView2DType("quadrupole");
    m_octupole     = StaticView3DType("octupole");
    m_hexadecupole = StaticView4DType("hexadecupole");

    auto h_dipole       = Kokkos::create_mirror_view(m_dipole);
    auto h_quadrupole   = Kokkos::create_mirror_view(m_quadrupole);
    auto h_octupole     = Kokkos::create_mirror_view(m_octupole);
    auto h_hexadecupole = Kokkos::create_mirror_view(m_hexadecupole);

    // monopole = net charge
    ScalarType monopole = 0.0;
    for (int n = 0; n < m_nsite; n++) {
      monopole += m_site.at(n).m_q;
    }
    m_monopole = MDFT::Impl::chop(monopole);

    // dipole
    for (int i = 0; i < 3; i++) {
      ScalarType dipole = 0.0;
      for (int n = 0; n < m_nsite; n++) {
        auto site = m_site.at(n);
        dipole += site.m_q * site.m_r[i];
      }
      h_dipole(i) = MDFT::Impl::chop(dipole);
    }

    // quadrupole
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        ScalarType quadrupole = 0.0;
        for (int n = 0; n < m_nsite; n++) {
          auto site = m_site.at(n);
          quadrupole += site.m_q * site.m_r[i] * site.m_r[j];
        }
        h_quadrupole(i, j) = MDFT::Impl::chop(quadrupole);
      }
    }

    // octupole
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          ScalarType octupole = 0.0;
          for (int n = 0; n < m_nsite; n++) {
            auto site = m_site.at(n);
            octupole += site.m_q * site.m_r[i] * site.m_r[j] * site.m_r[k];
          }
          h_octupole(i, j, k) = MDFT::Impl::chop(octupole);
        }
      }
    }

    // hexadecapole
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            ScalarType hexadecapole = 0.0;
            for (int n = 0; n < m_nsite; n++) {
              auto site = m_site.at(n);
              hexadecapole += site.m_q * site.m_r[i] * site.m_r[j] *
                              site.m_r[k] * site.m_r[l];
            }
            h_hexadecupole(i, j, k, l) = MDFT::Impl::chop(hexadecapole);
          }
        }
      }
    }

    Kokkos::deep_copy(m_dipole, h_dipole);
    Kokkos::deep_copy(m_quadrupole, h_quadrupole);
    Kokkos::deep_copy(m_octupole, h_octupole);
    Kokkos::deep_copy(m_hexadecupole, h_hexadecupole);
  }
  void init_density(const int nx, const int ny, const int nz, const int no) {
    // guess density (from module_density.f90)
    m_xi = View4DType("xi", nx, ny, nz, no);

    // Initialization made on device on a flatten view
    // Note vext is not allocated so, m_xi is initialized with 1.0
    using exec_space = ExecutionSpace;
    using View1DType = typename Kokkos::View<ScalarType*, ExecutionSpace>;
    View1DType xi(m_xi.data(), m_xi.size());
    Kokkos::Experimental::fill(exec_space(), xi, 1.0);
  }

 private:
};

/**
 * @brief A structure representing solvents in the MDFT framework.
 *
 * @tparam ExecutionSpace The Kokkos execution space.
 * @tparam ScalarType The type of scalar values used.
 */
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct Solvents {
  using json            = nlohmann::json;
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 6>;
  using LucArrayType    = Kokkos::Array<IntType, 6>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;
  using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
  using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;
  using SolventType     = Solvent<ExecutionSpace, ScalarType>;
  using SettingsType    = Settings<ScalarType>;

  /**
   * @brief Settings of the MDFT simulation.
   */
  SettingsType m_settings;

  std::vector<SolventType> m_solvents;

  SpatialGridType m_spatial_grid;
  AngularGridType m_angular_grid;

  Solvents()  = delete;
  ~Solvents() = default;

  /**
   * @brief Constructs a Solvents object and reads its properties from a file.
   *
   * @param filename The path to the file containing solvent properties.
   */
  Solvents(const SpatialGridType& spatial_grid,
           const AngularGridType& angular_grid, const SettingsType& settings)
      : m_spatial_grid(spatial_grid),
        m_angular_grid(angular_grid),
        m_settings(settings) {
    read_solvent();
  }

 private:
  // Read solvent atomic positions, charge, and lennard jones values in
  // solvent.in charge in electron units, sigma in Angstroms, epsilon in KJ/mol.
  void read_solvent() {
    for (int i = 0; i < m_settings.m_nb_solvent; i++) {
      m_solvents.push_back(SolventType());
    }

    read_mole_fractions();
    set_solvent();
    init_density();
  }

  // This SUBROUTINE open the array input_line which contains every line of
  // input/dft.in It then reads every line of input_line and looks for the tag
  // mole_fractions Then, it reads, one line after the other, the mole fractions
  // of every constituant.
  void read_mole_fractions() {
    switch (m_settings.m_nb_solvent) {
      case 1: m_solvents.at(0).m_mole_fraction = 1.0; break;
      default:
        for (int i = 0; i < m_settings.m_nb_solvent; i++) {
          m_solvents.at(i).m_mole_fraction =
              1.0 / static_cast<ScalarType>(m_settings.m_nb_solvent);
        }
        break;
    }

    ScalarType sum = 0;
    for (int i = 0; i < m_settings.m_nb_solvent; i++) {
      sum += m_solvents.at(i).m_mole_fraction;
    }

    ScalarType epsilon = std::numeric_limits<ScalarType>::epsilon() * 100;
    MDFT::Impl::Throw_If(
        std::abs(sum - 1.0) > epsilon,
        "Critial error. Sum of all mole fraction should be equal to one.");

    std::vector<ScalarType> mole_fractions;
    for (std::size_t i = 0; i < m_settings.m_nb_solvent; i++) {
      mole_fractions.push_back(m_solvents.at(i).m_mole_fraction);
    }

    MDFT::Impl::Throw_If(
        std::any_of(mole_fractions.begin(), mole_fractions.end(),
                    [](ScalarType value) { return value < 0 || value > 1; }),
        "Critical errror. Mole fractions should be between 0 and 1");
  }

  void set_solvent() {
    // Get the information about the solvent
    for (std::size_t i = 0; i < m_settings.m_nb_solvent; i++) {
      std::string solvent_name = m_settings.m_solvent;
      auto mole_fraction       = m_solvents.at(i).m_mole_fraction;
      std::cout << "Solvent number " << i << " is " << solvent_name
                << " with a molecular fraction of " << mole_fraction
                << std::endl;

      switch (Impl::solvent_index(solvent_name)) {
        case Impl::const_solvent_index("spce"):
        case Impl::const_solvent_index("spce-h"): {
          ScalarType hs_radius = 0;
          int nsite            = 3;
          int molrotsymorder   = 2;
          std::vector<ScalarType> q({-0.8476, 0.4238, 0.4238});
          std::vector<ScalarType> sig({3.166, 0., 0.});
          std::vector<ScalarType> eps({0.65, 0., 0.});
          std::vector<ScalarType> r0({0.0, 0.0, 0.0});
          std::vector<ScalarType> r1({0.816495, 0.0, 0.5773525});
          std::vector<ScalarType> r2({-0.816495, 0.0, 0.5773525});
          std::vector<IntType> Z({8, 1, 1});
          ScalarType n0 = 0.0332891 * mole_fraction;
          ScalarType rho0 =
              n0 / (8 * Constants::pi * Constants::pi / molrotsymorder);
          ScalarType relativePermittivity = 71;
          IntArrayType npluc({1, 6, 75, 252, 877, 2002});
          IntType n_line_cfile = 1024;

          m_solvents.at(i).init("spce", hs_radius, nsite, molrotsymorder, q,
                                sig, eps, r0, r1, r2, Z, n0, rho0,
                                relativePermittivity, npluc, n_line_cfile);
          MDFT::Impl::Throw_If(
              m_angular_grid.m_mmax > 5 || m_angular_grid.m_mmax < 0,
              "solvent spce only avail with mmax between 0 and 5");
        } break;
        case Impl::const_solvent_index("spce-m"): {
          std::string tmp_name = "spce";
          ScalarType hs_radius = 0;
          int nsite            = 3;
          int molrotsymorder   = 2;
          std::vector<ScalarType> q({-1.0, 0.5, 0.5});
          std::vector<ScalarType> sig({3.166, 0., 0.});
          std::vector<ScalarType> eps({0.65, 0., 0.});
          std::vector<ScalarType> r0({0.0, 0.0, 0.0});
          std::vector<ScalarType> r1({0.816495, 0.0, 0.5773525});
          std::vector<ScalarType> r2({-0.816495, 0.0, 0.5773525});
          std::vector<IntType> Z({8, 1, 1});
          ScalarType n0 = 0.0332891 * mole_fraction;
          ScalarType rho0 =
              n0 / (8 * Constants::pi * Constants::pi / molrotsymorder);
          ScalarType relativePermittivity = 71;
          IntArrayType npluc({1, 6, 75, 252, 877, 2002});
          IntType n_line_cfile = 1024;

          m_solvents.at(i).init(tmp_name, hs_radius, nsite, molrotsymorder, q,
                                sig, eps, r0, r1, r2, Z, n0, rho0,
                                relativePermittivity, npluc, n_line_cfile);
          MDFT::Impl::Throw_If(
              m_angular_grid.m_mmax > 5 || m_angular_grid.m_mmax < 0,
              "solvent spce only avail with mmax between 0 and 5");

          break;
        }
        case Impl::const_solvent_index("tip3p"): {
          std::cout << "case tip3p" << std::endl;
          ScalarType hs_radius = 0;
          int nsite            = 3;
          int molrotsymorder   = 2;
          std::vector<ScalarType> q({-0.834, 0.417, 0.417});
          std::vector<ScalarType> sig({3.15061, 0., 0.});
          std::vector<ScalarType> eps({0.636386, 0., 0.});
          std::vector<ScalarType> r0({0.0, 0.0, 0.0});
          std::vector<ScalarType> r1({0.756950, 0.0, 0.585882});
          std::vector<ScalarType> r2({-0.756950, 0.0, 0.585882});
          std::vector<IntType> Z({8, 1, 1});
          ScalarType n0 = 0.03349459 * mole_fraction;
          ScalarType rho0 =
              n0 / (8 * Constants::pi * Constants::pi / molrotsymorder);
          ScalarType relativePermittivity =
              91;  // cf mail de Luc du 16/12/2016 :
          IntArrayType npluc({1, 6, 75, 252, 877, 2002});
          IntType n_line_cfile = 1024;

          m_solvents.at(i).init("tip3p", hs_radius, nsite, molrotsymorder, q,
                                sig, eps, r0, r1, r2, Z, n0, rho0,
                                relativePermittivity, npluc, n_line_cfile);
          MDFT::Impl::Throw_If(
              m_angular_grid.m_mmax > 5 || m_angular_grid.m_mmax < 0,
              "solvent tip3p only avail with mmax between 0 and 5");

          break;
        }
        case Impl::const_solvent_index("tip3p-m"): {
          std::string tmp_name = "tip3p";
          ScalarType hs_radius = 0;
          int nsite            = 3;
          int molrotsymorder   = 2;
          std::vector<ScalarType> q({-0.95, 0.475, 0.475});
          std::vector<ScalarType> sig({3.15061, 0., 0.});
          std::vector<ScalarType> eps({0.636386, 0., 0.});
          std::vector<ScalarType> r0({0.0, 0.0, 0.0});
          std::vector<ScalarType> r1({0.756950, 0.0, 0.585882});
          std::vector<ScalarType> r2({-0.756950, 0.0, 0.585882});
          std::vector<IntType> Z({8, 1, 1});
          ScalarType n0 = 0.03349459 * mole_fraction;
          ScalarType rho0 =
              n0 / (8 * Constants::pi * Constants::pi / molrotsymorder);
          ScalarType relativePermittivity =
              91;  // cf mail de Luc du 16/12/2016 :
          IntArrayType npluc({1, 6, 75, 252, 877, 2002});
          IntType n_line_cfile = 1024;

          m_solvents.at(i).init(tmp_name, hs_radius, nsite, molrotsymorder, q,
                                sig, eps, r0, r1, r2, Z, n0, rho0,
                                relativePermittivity, npluc, n_line_cfile);
          MDFT::Impl::Throw_If(
              m_angular_grid.m_mmax > 5 || m_angular_grid.m_mmax < 0,
              "solvent tip3p only avail with mmax between 0 and 5");

          // Je connais ce site. C'est bizarre, la ref.3 pour epsilon(tip3p) n'a
          // pas fait tip3p! Il y aussi J.Chem.Phys.108, 10220 (1998) qui donne
          // 82, 94, 86 suivant N et paramètres de réaction field. Ma simulation
          // rapide N=100 donne 100, et MC/HNC résultant donne 91. Luc
          break;
        }
        default:
          MDFT::Impl::Throw_If(true, "Critical error. Solvent: " +
                                         solvent_name + " not recognized.");
          break;
      }
    }

    std::vector<ScalarType> all_molrotsymorder;
    for (std::size_t i = 0; i < m_settings.m_nb_solvent; i++) {
      all_molrotsymorder.push_back(m_solvents.at(i).m_molrotsymorder);
    }

    auto molrotsymorder = m_angular_grid.m_molrotsymorder;
    MDFT::Impl::Throw_If(
        std::any_of(all_molrotsymorder.begin(), all_molrotsymorder.end(),
                    [molrotsymorder](ScalarType value) {
                      return value != molrotsymorder;
                    }),
        "at least one solvent molrotsymorder is different "
        "from the grid molrotsymorder");
  }

  void init_density() {
    for (auto& solvent : m_solvents) {
      solvent.init_density(m_spatial_grid.m_nx, m_spatial_grid.m_ny,
                           m_spatial_grid.m_nz, m_angular_grid.m_no);
    }
  }
};

// \brief Compute delta_rho(r, Omega) = rho0 * (xi(r, Omega) ^2 - 1)
// These arrays are stored in the same order, so we just recast this into 1D
// View and perform the operation with 1D parallel for loop
// \tparam ExecutionSpace Execution space
// \tparam ViewType View type
// \tparam ScalarType Scalar type
//
// \param exec_space [in] Execution space instance
// \param xi [in] 6D View of xi, shape(nx, ny, nz, ntheta, nphi, npsi)
// \param delta_rho [out] 6D View of delta_rho, shape(nx, ny, nz, ntheta, nphi,
// npsi)
// \param rho0 [in] Reference density
template <KokkosExecutionSpace ExecutionSpace, KokkosView ViewType,
          typename ScalarType>
  requires KokkosViewAccesible<ExecutionSpace, ViewType>
void get_delta_rho(const ExecutionSpace& exec_space, const ViewType& xi,
                   const ViewType& delta_rho, const ScalarType rho0) {
  const std::size_t n = xi.size();
  MDFT::Impl::Throw_If(delta_rho.size() != n,
                       "size of delta_rho must be the same as the size of xi");

  // Flatten Views for simplicity
  using ValueType  = typename ViewType::non_const_value_type;
  using View1DType = Kokkos::View<ValueType*, ExecutionSpace>;
  View1DType xi_1d(xi.data(), n), delta_rho_1d(delta_rho.data(), n);

  Kokkos::parallel_for(
      "delta_rho",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, n),
      KOKKOS_LAMBDA(std::size_t i) {
        delta_rho_1d(i) = rho0 * (xi_1d(i) * xi_1d(i) - 1.0);
      });
}

// \brief Gather projections into gamma
// \tparam ExecutionSpace Execution space
// \tparam ViewType View type
// \tparam ScalarType Scalar type
//
// \param exec_space [in] Execution space instance
// \param xi [in] 6D View of xi, shape(nx, ny, nz, ntheta, nphi, npsi)
// \param vexc [in] 3D View of vexc, shape(ntheta, nphi, npsi)
// \param w [in] 3D View of w, shape(ntheta, nphi, npsi) (volume element in
// angular space?)
// \param delta_f [out] 6D View of delta_f, shape(nx, ny, nz,
// ntheta, nphi, npsi)
// \param ff [out] density fluctuation
// \param rho0 [in] Reference density
// \param prefactor [in] Coefficient prefactor
template <KokkosExecutionSpace ExecutionSpace, KokkosView View6DType,
          KokkosView View3DType, typename ScalarType>
  requires KokkosViewAccesible<ExecutionSpace, View3DType> &&
           KokkosViewAccesible<ExecutionSpace, View6DType>
void get_delta_f(const ExecutionSpace& exec_space, const View6DType& xi,
                 const View3DType& vexc, const View3DType& w,
                 const View6DType& delta_f, ScalarType& ff,
                 const ScalarType rho0, const ScalarType prefactor) {
  const std::size_t nx = xi.extent(0), ny = xi.extent(1), nz = xi.extent(2),
                    ntheta = xi.extent(3), nphi = xi.extent(4),
                    npsi = xi.extent(5);

  for (int i = 0; i < 3; i++) {
    MDFT::Impl::Throw_If(xi.extent(i + 3) != vexc.extent(i),
                         "angular grid size of xi and vexc must be the same");
  }

  // Flatten Views for simplicity
  const std::size_t nxyz = nx * ny * nz, nangle = ntheta * nphi * npsi;
  using ValueType  = typename View6DType::non_const_value_type;
  using View1DType = Kokkos::View<ValueType*, ExecutionSpace>;
  using View2DType = Kokkos::View<ValueType**, ExecutionSpace>;
  View1DType vexc_1d(vexc.data(), nangle), w_1d(w.data(), nangle);
  View2DType delta_f_2d(delta_f.data(), nxyz, nangle),
      xi_2d(xi.data(), nxyz, nangle);

  ff                = 0;
  using member_type = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  auto team_policy =
      Kokkos::TeamPolicy<ExecutionSpace>(exec_space, nxyz, Kokkos::AUTO);
  Kokkos::parallel_reduce(
      "delta_f", team_policy,
      KOKKOS_LAMBDA(const member_type& team_member, ValueType& l_ff) {
        const auto ixyz = team_member.league_rank();
        ValueType sum   = 0;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(team_member, nangle),
            [&](const int ip, ValueType& lsum) {
              lsum += vexc_1d(ip) * w_1d(ip) *
                      (xi_2d(ixyz, ip) * xi_2d(ixyz, ip) - 1.0);
              delta_f_2d(ixyz, ip) = 2.0 * rho0 * xi_2d(ixyz, ip) * vexc_1d(ip);
            },
            sum);
        l_ff += rho0 * sum * prefactor;
      },
      ff);
}

};  // namespace MDFT

#endif
