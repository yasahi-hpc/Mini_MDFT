#ifndef MDFT_SOLUTE_HPP
#define MDFT_SOLUTE_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <Kokkos_Core.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_System.hpp"
#include "IO/MDFT_IO_Utils.hpp"

namespace MDFT {

/**
 * @brief A structure representing a solute in the MDFT framework.
 *
 * @tparam ExecutionSpace The Kokkos execution space.
 * @tparam ScalarType The type of scalar values used.
 */
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct Solute {
  using json             = nlohmann::json;
  using StaticView1DType = typename Kokkos::View<ScalarType[3], ExecutionSpace>;
  using StaticView2DType =
      typename Kokkos::View<ScalarType[3][3], ExecutionSpace>;
  using StaticView3DType =
      typename Kokkos::View<ScalarType[3][3][3], ExecutionSpace>;
  using StaticView4DType =
      typename Kokkos::View<ScalarType[3][3][3][3], ExecutionSpace>;
  using ComplexView3DType =
      typename Kokkos::View<Kokkos::complex<ScalarType>***, ExecutionSpace>;
  using SiteType = Site<ScalarType>;

  /**
   * @brief Name of the solute.
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
  int m_nspec;

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

  /**
   * @brief Hard sphere diameter of the solute.
   */
  ScalarType m_diameter;

  /**
   * @brief Charge factor of the solute.
   */
  ComplexView3DType m_sigma_k;

  /**
   * @brief Vector of sites in the solute.
   */
  std::vector<SiteType> m_site;

  /**
   * @brief Constructs a Solute object and reads its properties from a file.
   *
   * @param filename The path to the file containing solute properties.
   */
  Solute(std::string filename) { read_solute(filename); }

 private:
  /**
   * @brief Reads solute properties from a file.
   *
   * @param filename The path to the file containing solute properties.
   */
  void read_solute(std::string filename) {
    MDFT::Impl::Throw_If(!IO::Impl::is_file_exists(filename),
                         "File: " + filename + "does not exist.");
    std::ifstream f(filename);
    json json_data = json::parse(f);

    // Read nsite
    m_nsite                            = json_data["nsite"].get<int>();
    std::vector<ScalarType> charge     = json_data["charge"];
    std::vector<ScalarType> sigma      = json_data["sigma"];
    std::vector<ScalarType> epsilon    = json_data["epsilon"];
    std::vector<ScalarType> x          = json_data["x"];
    std::vector<ScalarType> y          = json_data["y"];
    std::vector<ScalarType> z          = json_data["z"];
    std::vector<int> Zatomic           = json_data["Zatomic"];
    std::vector<std::string> Atom_Name = json_data["Atom Name"];
    std::vector<std::string> Surname   = json_data["Surname"];

    for (int i = 0; i < m_nsite; i++) {
      m_site.push_back(SiteType(
          Atom_Name[i], charge[i], sigma[i], epsilon[i],
          Kokkos::Array<ScalarType, 3>({x[i], y[i], z[i]}), Zatomic[i]));
    }
  }
};

}  // namespace MDFT

#endif
