#ifndef MDFT_THERMO_HPP
#define MDFT_THERMO_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <Kokkos_Core.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_System.hpp"
#include "MDFT_Constants.hpp"

namespace MDFT {

/**
 * @brief A structure representing a thermodynamic conditions in the MDFT
 * framework.
 *
 * @tparam ScalarType The type of scalar values used.
 */
template <typename ScalarType>
struct Thermo {
  using SettingsType = Settings<ScalarType>;

  /**
   * @brief Temperature of the system.
   */
  ScalarType m_t;

  /**
   * @brief Pressure of the system.
   */
  ScalarType m_p;

  /**
   * @brief temperature energy unit
   */
  ScalarType m_kbt;

  /**
   * @brief 1/kbt
   */
  ScalarType m_beta;

  Thermo()  = default;
  ~Thermo() = default;

  Thermo(const SettingsType& settings) { read_settings(settings); }

 private:
  void read_settings(const SettingsType& settings) {
    // Read settings
    m_t = settings.m_temperature;
    m_p = settings.m_pressure;

    m_kbt  = Constants::Boltz * Constants::Navo * m_t * 1e-3;
    m_beta = 1.0 / m_kbt;

    std::cout << "Temperature = " << m_t << " (K)" << std::endl;
    std::cout << "kT = " << m_kbt << " (kJ/mol)" << std::endl;
    std::cout << "P = " << m_p << " (bar)" << std::endl;
  }
};

}  // namespace MDFT

#endif
