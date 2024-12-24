#ifndef MDFT_SYSTEM_HPP
#define MDFT_SYSTEM_HPP

#include <string>
#include <Kokkos_Core.hpp>

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
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 3>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;

  /**
   * @brief The name of the solvent.
   */
  std::string m_solvent;

  /**
   * @brief The number of the solvent.
   */
  IntType m_nb_solvent;

  /**
   * @brief Grid sizes of the simulation.
   */
  IntArrayType m_boxnod;

  /**
   * @brief Grid lengths of the simulation.
   */
  ScalarArrayType m_boxlen;

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
  bool m_translate_solute_to_center;

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
  ScalarType m_temperature;

  /**
   * @brief Whether to restart the simulation.
   */
  bool m_restart;

  Settings(std::string name, IntType nb_solvent, IntArrayType boxnod,
           ScalarArrayType boxlen, IntType mmax, IntType maximum_iteration_nbr,
           ScalarType precision_factor, ScalarType solute_charges_scale_factor,
           bool translate_solute_to_center, bool hard_sphere_solute,
           ScalarType hard_sphere_solute_radius, ScalarType temperature,
           bool restart)
      : m_solvent(name),
        m_nb_solvent(nb_solvent),
        m_boxnod(boxnod),
        m_boxlen(boxlen),
        m_mmax(mmax),
        m_maximum_iteration_nbr(maximum_iteration_nbr),
        m_precision_factor(precision_factor),
        m_solute_charges_scale_factor(solute_charges_scale_factor),
        m_translate_solute_to_center(translate_solute_to_center),
        m_hard_sphere_solute(hard_sphere_solute),
        m_hard_sphere_solute_radius(hard_sphere_solute_radius),
        m_temperature(temperature),
        m_restart(restart) {}
};

}  // namespace MDFT

#endif
