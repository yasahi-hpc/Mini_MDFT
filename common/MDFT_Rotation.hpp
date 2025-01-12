#ifndef MDFT_ROTATION_HPP
#define MDFT_ROTATION_HPP

#include <memory>
#include <KokkosFFT.hpp>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "MDFT_Wigner.hpp"
#include "MDFT_Math_Utils.hpp"

namespace MDFT {

// \brief Storing the rotation coefficients
// \tparam ExecutionSpace Execution space
// \tparam ScalarType Scalar type
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct RotationCoeffs {
  using IntType         = int;
  using IntArrayType    = Kokkos::Array<IntType, 3>;
  using ScalarArrayType = Kokkos::Array<ScalarType, 3>;
  using View2DType      = typename Kokkos::View<ScalarType **, ExecutionSpace>;
  using View3DType      = typename Kokkos::View<ScalarType ***, ExecutionSpace>;

  const IntType m_mmax_max = 6;
  View3DType m_a, m_b;
  View2DType m_c, m_d;

  RotationCoeffs() {
    // View allocation
    m_a =
        View3DType("a", m_mmax_max - 1, 2 * m_mmax_max + 1, 2 * m_mmax_max + 1);
    m_b =
        View3DType("b", m_mmax_max - 1, 2 * m_mmax_max + 1, 2 * m_mmax_max + 1);
    m_c = View2DType("c", m_mmax_max - 1, 2 * m_mmax_max + 1);
    m_d = View2DType("d", m_mmax_max - 1, 2 * m_mmax_max + 1);

    // Initialization at host
    auto h_a = Kokkos::create_mirror_view(m_a);
    auto h_b = Kokkos::create_mirror_view(m_b);
    auto h_c = Kokkos::create_mirror_view(m_c);
    auto h_d = Kokkos::create_mirror_view(m_d);

    // In Fortran
    // sqrtof(-1:2*mmax_max +1) = [0._dp, 0._dp, 1._dp, sqrt(2._dp),
    // sqrt(3._dp), sqrt(4._dp), sqrt(5._dp),& sqrt(6._dp), sqrt(7._dp),
    // sqrt(8._dp), sqrt(9._dp), sqrt(10._dp),
    // sqrt(11._dp),sqrt(12._dp),sqrt(13._dp) ]
    std::vector<ScalarType> sqrtof(2 * m_mmax_max + 2, 0.0);
    for (int i = 2; i < 14; ++i) {
      sqrtof.at(i) = std::sqrt(static_cast<ScalarType>(i - 1));
    }
    ScalarType sqrt2 = std::sqrt(2.0);

    // Note that the l index starts from 0 in C++ but starts from 2 in Fortran
    for (int l = 2; l <= m_mmax_max; ++l) {
      for (int m = -l; m <= l; ++m) {
        for (int m1 = -l + 1; m1 < l; ++m1) {
          h_a(l - 2, m + m_mmax_max, m1 + m_mmax_max) =
              sqrtof.at(l + m + 1) * sqrtof.at(l - m + 1) /
              (sqrtof.at(l + m1 + 1) * sqrtof.at(l - m1 + 1));
          h_b(l - 2, m + m_mmax_max, m1 + m_mmax_max) =
              sqrtof.at(l + m + 1) * sqrtof.at(l + m) /
              (sqrt2 * sqrtof.at(l + m1 + 1) * sqrtof.at(l - m1 + 1));
        }
        int m1 = l;
        h_c(l - 2, m + m_mmax_max) =
            sqrt2 * sqrtof.at(l + m + 1) * sqrtof.at(l - m + 1) /
            (sqrtof.at(l + m1 + 1) * sqrtof.at(l + m1));
        h_d(l - 2, m + m_mmax_max) =
            sqrtof.at(l + m + 1) * sqrtof.at(l + m) /
            (sqrtof.at(l + m1 + 1) * sqrtof.at(l + m1));
      }
    }

    Kokkos::deep_copy(m_a, h_a);
    Kokkos::deep_copy(m_b, h_b);
    Kokkos::deep_copy(m_c, h_c);
    Kokkos::deep_copy(m_d, h_d);
  }
};

namespace Impl {

/// \brief Compute Rotation matrix between complex spherical harmonics
///
/// \tparam ArrayType: Array type, needs to be a 1D Array
/// \tparam RealView2DType: Real 2D View type
/// \tparam RealView3DType: Real 3D View type
/// \tparam ComplexView3DType: Complex 3D View type
///
/// \param q: Array of 3 elements
/// \param c: Real 2D View
/// \param d: Real 2D View
/// \param a: Real 3D View
/// \param b: Real 3D View
/// \param R: Complex 3D View (0:mmax,-mmax:mmax,-mmax:mmax)
///
template <KokkosArray ArrayType, KokkosView RealView2DType,
          KokkosView RealView3DType, KokkosView ComplexView3DType>
KOKKOS_INLINE_FUNCTION void
rotation_matrix_between_complex_spherical_harmonics_lu(
    const ArrayType &q, const RealView2DType &c, const RealView2DType &d,
    const RealView3DType &a, const RealView3DType &b,
    const ComplexView3DType &R) {
  using complex_type = typename ComplexView3DType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<complex_type>;
  using RotArrayType = Kokkos::Array<float_type, 3>;
  const int mmax     = R.extent(0) - 1;

  if (mmax == 0) {
    for (std::size_t i = 0; i < R.size(); ++i) {
      auto R_data = R.data();
      R_data[i]   = complex_type(1.0, 0.0);
    }
    return;
  }

  // m == 0
  R(0, mmax, mmax) = complex_type(1.0, 0.0);

  // Build q-frame XYZ
  // Start by aligining the new Z along q
  RotArrayType rmat0, rmat1, rmat2;
  if (q[0] == 0.0 && q[1] == 0.0 && q[2] == 0.0) {
    // theta definied as zero.
    rmat2 = {0.0, 0.0, 1.0};
  } else {
    rmat2 = {q[0], q[1], q[2]};
  }

  // Then deal with X and Y
  if (rmat2[0] != 0.0 || rmat2[1] != 0.0) {
    // if rmat2 is along with axis z, the GSH is null, and we don't carre about
    // phi. in the MDFT definition of Omega, the rotation axes are z-y-z.
    rmat1 = cross_product(RotArrayType{0, 0, 1}, rmat2);
  } else {
    rmat1 = cross_product(rmat2, RotArrayType{1, 0, 0});
  }

  // Normalize
  rmat2 = L2normalize(rmat2);

  // to avoid round up error if rmat2 is so close to z
  rmat1 = L2normalize(rmat1);
  rmat0 = cross_product(rmat1, rmat2);

  // m == 1
  const float_type inv_sqrt2 = 1.0 / Kokkos::sqrt(2);
  R(1, mmax - 1, mmax - 1) =
      complex_type((rmat1[1] + rmat0[0]) * 0.5, (rmat0[1] - rmat1[0]) * 0.5);
  R(1, mmax - 1, mmax) =
      complex_type(rmat2[0] * inv_sqrt2, rmat2[1] * inv_sqrt2);
  R(1, mmax - 1, mmax + 1) =
      complex_type((rmat1[1] - rmat0[0]) * 0.5, (-rmat0[1] - rmat1[0]) * 0.5);

  R(1, mmax, mmax - 1) =
      complex_type(rmat0[2] * inv_sqrt2, -rmat1[2] * inv_sqrt2);
  R(1, mmax, mmax) = complex_type(rmat2[2], 0.0);
  R(1, mmax, mmax + 1) =
      complex_type(-rmat0[2] * inv_sqrt2, -rmat1[2] * inv_sqrt2);

  R(1, mmax + 1, mmax - 1) =
      complex_type((rmat1[1] - rmat0[0]) * 0.5, (rmat0[1] + rmat1[0]) * 0.5);
  R(1, mmax + 1, mmax) =
      complex_type(-rmat2[0] * inv_sqrt2, rmat2[1] * inv_sqrt2);
  R(1, mmax + 1, mmax + 1) =
      complex_type((rmat1[1] + rmat0[0]) * 0.5, (rmat1[0] - rmat0[1]) * 0.5);

  if (mmax == 1) return;

  // mmax > 1
  for (int l = 2; l <= mmax; ++l) {
    auto l1 = l - 1;
    for (int m = -l; m <= l; ++m) {
      int m1min = m > 0 ? 1 : 0;
      for (int m1 = m1min; m1 <= l - 1; ++m1) {
        if (m == -l) {
          auto b_tmp  = b(l - 2, mmax - m, mmax + m1);
          auto R0_tmp = R(1, mmax - 1, mmax);
          auto R1_tmp = R(l1, mmax + m + 1, mmax + m1);

          auto f                    = b_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g                    = b_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else if (m == l) {
          auto b_tmp  = b(l - 2, mmax + m, mmax + m1);
          auto R0_tmp = R(1, mmax + 1, mmax);
          auto R1_tmp = R(l1, mmax + m - 1, mmax + m1);

          auto f                    = b_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g                    = b_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else {
          auto a_tmp     = a(l - 2, mmax + m, mmax + m1);
          auto b_pos_tmp = b(l - 2, mmax + m, mmax + m1);
          auto b_neg_tmp = b(l - 2, mmax - m, mmax + m1);
          auto R0_tmp    = R(1, mmax, mmax);
          auto R1_tmp    = R(1, mmax + 1, mmax);
          auto R2_tmp    = R(1, mmax - 1, mmax);
          auto Rm0_tmp   = R(1, mmax + m, mmax + m1);
          auto Rm1_tmp   = R(1, mmax + m + 1, mmax + m1);
          auto Rm2_tmp   = R(1, mmax + m - 1, mmax + m1);

          auto f = a_tmp * (R0_tmp.real() * Rm0_tmp.real()) +
                   b_pos_tmp * (R1_tmp.real() * Rm2_tmp.real() -
                                R1_tmp.imag() * Rm2_tmp.imag()) +
                   b_neg_tmp * (R2_tmp.real() * Rm1_tmp.real() -
                                R2_tmp.imag() * Rm1_tmp.imag());
          auto g = a_tmp * (R0_tmp.real() * Rm0_tmp.imag()) +
                   b_pos_tmp * (R1_tmp.real() * Rm2_tmp.imag() +
                                R1_tmp.imag() * Rm2_tmp.real()) +
                   b_neg_tmp * (R2_tmp.real() * Rm1_tmp.imag() +
                                R2_tmp.imag() * Rm1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        }
        auto pow_tmp              = Kokkos::pow(-1.0, m + m1);
        auto f_conj               = pow_tmp * R(l, mmax + m, mmax + m1).real();
        auto g_conj               = -pow_tmp * R(l, mmax + m, mmax + m1).imag();
        R(l, mmax - m, mmax - m1) = complex_type(f_conj, g_conj);
      }

      int m1 = l;
      {
        if (m == -l) {
          auto d_tmp                = d(l - 2, mmax - m);
          auto R0_tmp               = R(1, mmax - 1, mmax + 1);
          auto R1_tmp               = R(l1, mmax + m + 1, mmax + m1 - 1);
          auto f                    = d_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g                    = d_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else if (m == l) {
          auto d_tmp                = d(l - 2, mmax + m);
          auto R0_tmp               = R(1, mmax + 1, mmax + 1);
          auto R1_tmp               = R(l1, mmax + m - 1, mmax + m1 - 1);
          auto f                    = d_tmp * (R0_tmp.real() * R1_tmp.real() -
                            R0_tmp.imag() * R1_tmp.imag());
          auto g                    = d_tmp * (R0_tmp.real() * R1_tmp.imag() +
                            R0_tmp.imag() * R1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        } else {
          auto c_tmp     = c(l - 2, mmax + m);
          auto d_pos_tmp = d(l - 2, mmax + m);
          auto d_neg_tmp = d(l - 2, mmax - m);
          auto R0_tmp    = R(1, mmax, mmax + 1);
          auto R1_tmp    = R(1, mmax + 1, mmax + 1);
          auto R2_tmp    = R(1, mmax - 1, mmax + 1);
          auto Rm0_tmp   = R(1, mmax + m, mmax + m1 - 1);
          auto Rm1_tmp   = R(1, mmax + m + 1, mmax + m1 - 1);
          auto Rm2_tmp   = R(1, mmax + m - 1, mmax + m1 - 1);

          auto f = c_tmp * (R0_tmp.real() * Rm0_tmp.real() -
                            R0_tmp.imag() * Rm0_tmp.imag()) +
                   d_pos_tmp * (R1_tmp.real() * Rm2_tmp.real() -
                                R1_tmp.imag() * Rm2_tmp.imag()) +
                   d_neg_tmp * (R2_tmp.real() * Rm1_tmp.real() -
                                R2_tmp.imag() * Rm1_tmp.imag());
          auto g = c_tmp * (R0_tmp.real() * Rm0_tmp.imag() +
                            R0_tmp.imag() * Rm0_tmp.real()) +
                   d_pos_tmp * (R1_tmp.real() * Rm2_tmp.imag() +
                                R1_tmp.imag() * Rm2_tmp.real()) +
                   d_neg_tmp * (R2_tmp.real() * Rm1_tmp.imag() +
                                R2_tmp.imag() * Rm1_tmp.real());
          R(l, mmax + m, mmax + m1) = complex_type(f, g);
        }
        auto pow_tmp              = Kokkos::pow(-1.0, m + m1);
        auto f_conj               = pow_tmp * R(l, mmax + m, mmax + m1).real();
        auto g_conj               = -pow_tmp * R(l, mmax + m, mmax + m1).imag();
        R(l, mmax - m, mmax - m1) = complex_type(f_conj, g_conj);
      }
    }
  }
}

}  // namespace Impl
}  // namespace MDFT

#endif
