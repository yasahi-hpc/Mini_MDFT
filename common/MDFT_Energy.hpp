#ifndef MDFT_ENERGY_HPP
#define MDFT_ENERGY_HPP

#include <string>

namespace MDFT {

/**
 * @brief A structure representing the total energy in the MDFT
 * framework.
 *
 * @tparam ScalarType The type of scalar values used.
 */
template <typename ScalarType>
struct Energy {
  using IntType                                           = int;
  ScalarType m_id                                         = 0;
  ScalarType m_ext                                        = 0;
  ScalarType m_exc_cs                                     = 0;
  ScalarType m_exc_cdeltacd                               = 0;
  ScalarType m_exc_cproj                                  = 0;
  ScalarType m_exc_ck_angular                             = 0;
  ScalarType m_exc_fmt                                    = 0;
  ScalarType m_exc_wca                                    = 0;
  ScalarType m_exc_3b                                     = 0;
  ScalarType m_exc_b                                      = 0;
  ScalarType m_exc_dipolar                                = 0;
  ScalarType m_exc_multipolar_without_coupling_to_density = 0;
  ScalarType m_exc_multipolar_with_coupling_to_density    = 0;
  ScalarType m_exc_hydro                                  = 0;
  ScalarType m_exc_nn_cs_plus_nbar                        = 0;
  ScalarType m_tot                                        = 0;
  ScalarType m_pscheme_correction                         = -999;
  ScalarType m_pbc_correction                             = -999;
  IntType m_ieval                                         = 0;
};

}  // namespace MDFT

#endif
