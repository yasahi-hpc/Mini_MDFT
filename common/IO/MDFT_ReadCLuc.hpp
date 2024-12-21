#ifndef MDFT_READCLUC_HPP
#define MDFT_READCLUC_HPP

#include <string>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "IO/MDFT_IO_Utils.hpp"

namespace MDFT {
namespace IO {

// \brief Storing the data read from luc files
// \tparam ScalarType Scalar type
template <typename ScalarType>
struct LucData {
  using IntType = int;

  IntType m_np, m_nq;
  ScalarType m_dq;

  LucData(std::string filename) {
    MDFT::Impl::Throw_If(!Impl::is_file_exists(filename),
                         "File: " + filename + "does not exist.");
  }
};

}  // namespace IO
}  // namespace MDFT

#endif
