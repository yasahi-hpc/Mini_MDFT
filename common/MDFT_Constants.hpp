#ifndef MDFT_CONSTANTS_HPP
#define MDFT_CONSTANTS_HPP

#include <string>
#include <Kokkos_Core.hpp>

namespace MDFT {

/** \brief A structure representing a solute in the MDFT framework.
 * This module defines the fundamental constants
 * It has to be the most complete possible and very well documented in order to
 * be easily trackable and modified and shareable The best website to my
 * knowledge to get this kind of information is the National Institute of
 * Standards and Technology (NIST) http://physics.nist.gov/cuu/Constants/ In
 * order to be easily sharable and trackable, please add date and short
 * description of your modification below. 20110922 16h08 Maximilien Levesque
 * creation from a basis by Daniel Borgis 20111120 20h00 Maximilien Levesque
 * addition sqpi ! square root of pi 20111216 08h36 Maximilien Levesque addition
 * of infty = huge(1.0_dp)
 */
struct Constants {
  using ScalarType                      = double;
  static constexpr ScalarType ln2       = 0.69314718055994530941723212145818;
  static constexpr ScalarType pi        = Kokkos::numbers::pi_v<double>;
  static constexpr ScalarType twopi     = 2.0 * pi;
  static constexpr ScalarType fourpi    = 4.0 * pi;
  static constexpr ScalarType eightpi   = 8.0 * pi;
  static constexpr ScalarType eightpiSQ = 8.0 * pi * pi;
  //static constexpr ScalarType sqpi      = sqrt(pi);

  // Electric constant ie Vacuum permittivity in Farad per meter == Coulomb^2
  // per (Joule x meter)
  static constexpr ScalarType eps0 = 8.854187817e-12;

  // Elementary charge in Coulomb ; [C]
  static constexpr ScalarType qunit = 1.602176565e-19;

  // Avogadro constant in mol^(-1)
  static constexpr ScalarType Navo = 6.02214129e23;

  // electrostatic potential unit so that QFACT*q*q/r is kJ/mol
  static constexpr ScalarType qfact =
      qunit * qunit * 1.0e-3 * Navo / (fourpi * eps0 * 1.0e-10);

  // Boltzmann constant in Joule per Kelvin, [J].[K]^{-1}
  static constexpr ScalarType Boltz = 1.38064852e-23;

  static constexpr ScalarType infty = std::numeric_limits<ScalarType>::max();

  // what is numerically considered as zero
  static constexpr ScalarType epsN = std::numeric_limits<ScalarType>::epsilon();
};

}  // namespace MDFT

#endif
