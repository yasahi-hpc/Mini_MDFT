#ifndef MDFT_SYSTEM_HPP
#define MDFT_SYSTEM_HPP

#include <string>
#include <nlohmann/json.hpp>
#include <Kokkos_Core.hpp>
#include "MDFT_Asserts.hpp"
#include "IO/MDFT_IO_Utils.hpp"

namespace MDFT {

/**
 * @brief Represents a site in the MDFT system.
 *
 * @tparam ScalarType The scalar type used for the site properties.
 */
template <typename ScalarType>
struct Site {
  using IntType         = int;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;

  /**
   * @brief The name of the site.
   */
  std::string m_name;

  /**
   * @brief The charge of the site.
   */
  ScalarType m_q;

  /**
   * @brief The sigma parameter of the site.
   */
  ScalarType m_sig;

  /**
   * @brief The epsilon parameter of the site.
   */
  ScalarType m_eps;

  /**
   * @brief The position of the site as an array of scalars.
   */
  ScalarArrayType m_r;

  /**
   * @brief The first lambda parameter of the site.
   */
  ScalarType m_lambda1;

  /**
   * @brief The second lambda parameter of the site.
   */
  ScalarType m_lambda2;

  /**
   * @brief The atomic number of the site.
   */
  int m_z;

  Site()  = delete;
  ~Site() = default;

  Site(std::string name, ScalarType q, ScalarType sig, ScalarType eps,
       ScalarArrayType r, IntType Z)
      : m_name(name), m_q(q), m_sig(sig), m_eps(eps), m_r(r), m_z(Z) {}
};

/**
 * @brief Configuration of the MDFT simulation
 *
 * @tparam ScalarType The scalar type used for the site properties.
 */
template <typename ScalarType>
struct Settings {
  using json            = nlohmann::json;
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 3>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;

  /**
   * @brief The name of the solvent.
   * This may be a list of strings
   */
  std::string m_solvent;

  /**
   * @brief The number of the solvent.
   */
  IntType m_nb_solvent = 1;

  /**
   * @brief Grid sizes of the simulation.
   */
  IntArrayType m_boxnod;

  /**
   * @brief Grid lengths of the simulation.
   */
  ScalarArrayType m_boxlen;

  /**
   * @brief molrotsymorder
   */
  IntType m_molrotsymorder = 2;

  /**
   * @brief mu_max
   */
  IntType m_mmax;

  /**
   * @brief Maximum number of iterations.
   */
  IntType m_maximum_iteration_nbr;

  /**
   * @brief The precision factor.
   */
  ScalarType m_precision_factor;

  /**
   * @brief solute_charges_scale_factor
   */
  ScalarType m_solute_charges_scale_factor;

  /**
   * @brief add Lx/2, Ly/2, Lz/2 to all solute coord
   */
  bool m_translate_solute_to_center = true;

  /**
   * @brief hard sphere solute
   */
  bool m_hard_sphere_solute;

  /**
   * @brief radius of hard sphere solute
   */
  ScalarType m_hard_sphere_solute_radius;

  /**
   * @brief Temperature of the simulation.
   */
  ScalarType m_temperature = 300;

  /**
   * @brief Pressure of the simulation.
   */
  ScalarType m_pressure = 1;

  /**
   * @brief Whether to restart the simulation.
   */
  bool m_restart;

  Settings()  = default;
  ~Settings() = default;

  Settings(std::string filename) {
    MDFT::Impl::Throw_If(!IO::Impl::is_file_exists(filename),
                         "File: " + filename + " does not exist.");
    std::ifstream f(filename);
    json json_data = json::parse(f);

    // Read settings
    std::vector<int> boxnod        = json_data["boxnod"];
    std::vector<ScalarType> boxlen = json_data["boxlen"];

    m_boxnod  = Kokkos::Array<int, 3>({boxnod[0], boxnod[1], boxnod[2]});
    m_boxlen  = Kokkos::Array<ScalarType, 3>({boxlen[0], boxlen[1], boxlen[2]});
    m_solvent = json_data["solvent"].get<std::string>();

    if (json_data.contains("nb_solvent")) {
      m_nb_solvent = json_data["nb_solvent"].get<int>();
    }

    m_mmax = json_data["mmax"].get<int>();
    if (json_data.contains("molrotsymorder")) {
      m_molrotsymorder = json_data["molrotsymorder"].get<int>();
    }

    m_maximum_iteration_nbr = json_data["maximum_iteration_nbr"].get<int>();
    m_precision_factor      = json_data["precision_factor"].get<ScalarType>();
    m_solute_charges_scale_factor =
        json_data["solute_charges_scale_factor"].get<ScalarType>();
    if (json_data.contains("translate_solute_to_center")) {
      m_translate_solute_to_center =
          json_data["translate_solute_to_center"].get<bool>();
    }
    m_hard_sphere_solute = json_data["hard_sphere_solute"].get<bool>();
    m_hard_sphere_solute_radius =
        json_data["hard_sphere_solute_radius"].get<ScalarType>();
    if (json_data.contains("temperature")) {
      m_temperature = json_data["temperature"].get<ScalarType>();
    }
    m_restart = json_data["restart"].get<bool>();
  }
};

}  // namespace MDFT

#endif
