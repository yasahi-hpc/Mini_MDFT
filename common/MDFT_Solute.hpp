// SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#ifndef MDFT_SOLUTE_HPP
#define MDFT_SOLUTE_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <Kokkos_Core.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_System.hpp"
#include "MDFT_Grid.hpp"
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
  using SpatialGridType = SpatialGrid<ExecutionSpace, ScalarType>;
  using SiteType        = Site<ScalarType>;
  using SettingsType    = Settings<ScalarType>;

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

  SpatialGridType m_spatial_grid;

  Solute()  = delete;
  ~Solute() = default;
  /**
   * @brief Constructs a Solute object and reads its properties from a file.
   *
   * @param filename The path to the file containing solute properties.
   */
  Solute(const SpatialGridType& spatial_grid, const SettingsType& settings,
         std::string solute_filename)
      : m_spatial_grid(spatial_grid) {
    read_solute(solute_filename, settings.m_solute_charges_scale_factor);
    if (settings.m_translate_solute_to_center) {
      mv_solute_to_center();
    }
  }

 private:
  /**
   * @brief Reads solute properties from a file.
   *
   * @param filename The path to the file containing solute properties.
   */
  void read_solute(std::string filename,
                   ScalarType solute_charges_scale_factor) {
    MDFT::Impl::Throw_If(!IO::Impl::is_file_exists(filename),
                         "File: " + filename + " does not exist.");
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
      auto q = charge[i] * solute_charges_scale_factor;
      m_site.push_back(SiteType(
          Atom_Name[i], q, sigma[i], epsilon[i],
          Kokkos::Array<ScalarType, 3>({x[i], y[i], z[i]}), Zatomic[i]));
    }
  }

  void mv_solute_to_center() {
    // what are the coordinates of the middle of the simulation box ?
    ScalarType coo_midbox_x = m_spatial_grid.m_lx / 2.0;
    ScalarType coo_midbox_y = m_spatial_grid.m_ly / 2.0;
    ScalarType coo_midbox_z = m_spatial_grid.m_lz / 2.0;

    // what are the coordinates of the center of mass of the solute?
    // we don't now the mass of the sites as of mdft-dev 2016-07-20.
    // we'll thus say all sites have the same mass.
    // Thus, the coordinates of the center of mass is the mean coordinate
    ScalarType solute_mean_x = 0.0, solute_mean_y = 0.0, solute_mean_z = 0.0;
    for (int i = 0; i < m_nsite; i++) {
      solute_mean_x += m_site.at(i).m_r[0] / static_cast<ScalarType>(m_nsite);
      solute_mean_y += m_site.at(i).m_r[1] / static_cast<ScalarType>(m_nsite);
      solute_mean_z += m_site.at(i).m_r[2] / static_cast<ScalarType>(m_nsite);
    }

    // Now, we translate this center of mass to the center of the box
    // by shifting all coordinates (and thus the center of mass).
    // Removing solute_mean_x, y and z translates the center of mass to
    // coordinate 0,0,0 Then add coo_midbox_x, y and z to translate the center
    // of mass to center of the box.
    for (int i = 0; i < m_nsite; i++) {
      m_site.at(i).m_r[0] += coo_midbox_x - solute_mean_x;
      m_site.at(i).m_r[1] += coo_midbox_y - solute_mean_y;
      m_site.at(i).m_r[2] += coo_midbox_z - solute_mean_z;
    }

    // check if some positions are out of the supercell
    // j is a test tag. We loop over this test until every atom is in the box.
    // This allows for instance, if a site is two boxes too far to still be
    // ok.
    for (int i = 0; i < m_nsite; i++) {
      for (int d = 0; d < 3; d++) {
        m_site.at(i).m_r[d] =
            std::fmod(m_site.at(i).m_r[d], m_spatial_grid.m_length[d]);
      }
    }
  }
};

}  // namespace MDFT

#endif
