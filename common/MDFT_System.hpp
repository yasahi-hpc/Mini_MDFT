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
   * @brief The position of the site as an array of scalars.
   */
  ScalarArrayType m_r;

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

}  // namespace MDFT

#endif
