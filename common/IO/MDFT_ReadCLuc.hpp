#ifndef MDFT_READCLUC_HPP
#define MDFT_READCLUC_HPP

#include <string>
#include "MDFT_Concepts.hpp"
#include "MDFT_Asserts.hpp"
#include "MDFT_Grid.hpp"
#include "IO/MDFT_IO_Utils.hpp"

namespace MDFT {
namespace IO {

// \brief Storing the data read from luc files
// \tparam ScalarType Scalar type
template <KokkosExecutionSpace ExecutionSpace, typename ScalarType>
struct LucData {
  using IntType       = int;
  using IntView1DType = Kokkos::View<IntType*, ExecutionSpace>;
  using IntView5DType = Kokkos::View<IntType*****, ExecutionSpace>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<ScalarType>**, ExecutionSpace>;
  using AngularGridType = AngularGrid<ExecutionSpace, ScalarType>;

  IntType m_np, m_nq;
  ScalarType m_dq;

  IntView1DType m_m;
  IntView1DType m_n;
  IntView1DType m_mu;
  IntView1DType m_nu;
  IntView1DType m_khi;

  IntView5DType m_p;
  ComplexView2DType m_cmnmunukhi;

  LucData(std::string filename, const AngularGridType& angular_grid, int np,
          ScalarType qmaxwanted)
      : m_np(np) {
    MDFT::Impl::Throw_If(!Impl::is_file_exists(filename),
                         "File: " + filename + "does not exist.");

    // Allocate views
    auto mmax            = angular_grid.m_mmax;
    auto mrso            = angular_grid.m_molrotsymorder;
    std::size_t mmax_p1  = static_cast<std::size_t>(mmax + 1);
    std::size_t mmax2_p1 = static_cast<std::size_t>(2 * mmax + 1);
    std::size_t mmax_pm  = static_cast<std::size_t>(2 * mmax / mrso + 1);

    m_m   = IntView1DType("m", np);
    m_n   = IntView1DType("n", np);
    m_mu  = IntView1DType("mu", np);
    m_nu  = IntView1DType("nu", np);
    m_khi = IntView1DType("khi", np);
    m_p   = IntView5DType("p", mmax_p1, mmax_p1, mmax_pm, mmax_pm, mmax2_p1);

    // Reading luc files
    double tmp_qmaxwanted = static_cast<double>(qmaxwanted);
    int linesToSkip       = 17;
    double dq             = deltaAbscissa(filename, linesToSkip);
    int nq                = static_cast<int>(tmp_qmaxwanted / dq + 0.5) + 2;

    m_cmnmunukhi = ComplexView2DType("cmnmunukhi", np, nq);

    // Copy to member variables
    m_dq = static_cast<ScalarType>(dq);
    m_nq = nq;

    std::vector<int> m_vec, n_vec, mu_vec, nu_vec, khi_vec;
    std::vector<std::complex<double>> cmnmunukhi_vec;

    read_luc(filename, tmp_qmaxwanted, np, nq, dq, m_vec, n_vec, mu_vec, nu_vec,
             khi_vec, cmnmunukhi_vec);

    // Fill the array that hash map from a projection index to m, n, mu, nu, khi
    auto h_m         = Kokkos::create_mirror_view(m_m);
    auto h_n         = Kokkos::create_mirror_view(m_n);
    auto h_mu        = Kokkos::create_mirror_view(m_mu);
    auto h_nu        = Kokkos::create_mirror_view(m_nu);
    auto h_khi       = Kokkos::create_mirror_view(m_khi);
    auto h_p         = Kokkos::create_mirror_view(m_p);
    auto h_mnmunukhi = Kokkos::create_mirror_view(m_cmnmunukhi);

    for (int ip = 0; ip < np; ip++) {
      int im    = m_vec.at(ip);
      int in    = n_vec.at(ip);
      int imu   = m_vec.at(ip);
      int inu   = n_vec.at(ip);
      int ikhi  = khi_vec.at(ip);
      h_m(ip)   = im;
      h_n(ip)   = in;
      h_mu(ip)  = imu;  // mu(:) and nu(:) contains the read value of mu and nu
      h_nu(ip)  = inu;
      h_khi(ip) = ikhi;
      // but p(:,:,:,:,:) uses mu/mrso and nu/mrso, we often call them mu2 and
      // nu2
      h_p(im, in, imu / mrso, inu / mrso, ikhi) = ip;

      for (int iq = 0; iq < nq; iq++) {
        using complex_value_type = ComplexView2DType::non_const_value_type;
        auto cmnmunukhi          = cmnmunukhi_vec.at(ip + np * iq);
        h_mnmunukhi(ip, iq) =
            complex_value_type(cmnmunukhi.real(), cmnmunukhi.imag());
      }
    }

    Kokkos::deep_copy(m_m, h_m);
    Kokkos::deep_copy(m_n, h_n);
    Kokkos::deep_copy(m_mu, h_mu);
    Kokkos::deep_copy(m_nu, h_nu);
    Kokkos::deep_copy(m_khi, h_khi);
    Kokkos::deep_copy(m_p, h_p);
    Kokkos::deep_copy(m_cmnmunukhi, h_mnmunukhi);
  }

 private:
  void read_luc(std::string filename, double qmaxwanted, int np, int nq,
                double dq, std::vector<int>& m_vec, std::vector<int>& n_vec,
                std::vector<int>& mu_vec, std::vector<int>& nu_vec,
                std::vector<int>& khi_vec,
                std::vector<std::complex<double>>& cmnmunukhi_vec) {
    std::ifstream file(filename);

    // Allocate vectors
    m_vec.resize(np);
    n_vec.resize(np);
    mu_vec.resize(np);
    nu_vec.resize(np);
    khi_vec.resize(np);
    cmnmunukhi_vec.resize(np * nq);

    // Skip the first 10 lines
    std::string dummy;
    for (int i = 0; i < 10; ++i) std::getline(file, dummy);

    // Read arrays m, n, mu, nu, khi
    std::string label;
    std::getline(file, label);

    file >> label;
    for (int i = 0; i < np; i++) file >> m_vec[i];

    file >> label;
    for (int i = 0; i < np; i++) file >> n_vec[i];

    file >> label;
    for (int i = 0; i < np; i++) file >> mu_vec[i];

    file >> label;
    for (int i = 0; i < np; i++) file >> nu_vec[i];

    file >> label;
    for (int i = 0; i < np; i++) file >> khi_vec[i];

    std::getline(file, label);
    std::getline(file, label);

    // Read q, cmnmunukhi(q)
    double q, actualdq;

    // if you want more than available, use all that is available. If you need
    // less, use less.
    for (int iq = 0; iq < nq; ++iq) {
      std::string line;
      std::getline(file, line);
      std::istringstream iss(line);
      std::string token;

      iss >> q;

      for (int ip = 0; ip < np; ++ip) {
        std::complex<double> z;
        iss >> z;
        cmnmunukhi_vec[ip + np * iq] = z;
      }
      if (iq == 0) actualdq = q;
      if (iq == 1) actualdq = q - actualdq;
      MDFT::Impl::Throw_If(q > qmaxwanted + 2.0 * dq,
                           "q > qmaxwanted in file.");
    }
  }

  double deltaAbscissa(const std::string& filename, int linesToSkip) {
    std::ifstream file(filename);
    // Skip the specified number of lines
    std::string line;
    for (int i = 0; i < linesToSkip; ++i) {
      if (!std::getline(file, line)) {
        throw std::runtime_error(
            "Unexpected EOF while skipping lines in file: " + filename);
      }
    }

    // Read the first two abscissa values to compute dAbscissa
    double previousAbscissa, abscissa;
    for (int i = 0; i < 2; ++i) {
      if (!(file >> previousAbscissa)) {
        throw std::runtime_error("Error reading first abscissa in file: " +
                                 filename);
      }
      file.ignore(std::numeric_limits<std::streamsize>::max(),
                  '\n');  // Ignore the rest of the line
      if (!(file >> abscissa)) {
        throw std::runtime_error("Error reading second abscissa in file: " +
                                 filename);
      }
      file.ignore(std::numeric_limits<std::streamsize>::max(),
                  '\n');  // Ignore the rest of the line
    }

    double dAbscissa = abscissa - previousAbscissa;
    int n_lines      = MDFT::IO::Impl::count_lines_in_file(filename);

    // Reset and re-read the file to check uniformity
    file.clear();
    file.seekg(0, std::ios::beg);
    for (int i = 0; i < linesToSkip; ++i) {
      if (!std::getline(file, line)) {
        throw std::runtime_error(
            "Unexpected EOF while skipping lines in file: " + filename);
      }
    }

    if (!(file >> abscissa)) {
      throw std::runtime_error("Error reading second abscissa in file: " +
                               filename);
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(),
                '\n');  // Ignore the rest of the line

    // Validate uniformity of abscissa values
    for (int i = 0; i < n_lines - 1; ++i) {
      previousAbscissa = abscissa;
      try {
        file >> abscissa;
      } catch (const std::runtime_error& e) {
        return dAbscissa;
        // throw std::runtime_error("Error reading second abscissa in file: " +
        // filename);
      }

      file.ignore(std::numeric_limits<std::streamsize>::max(),
                  '\n');  // Ignore the rest of the line
      double difference = abscissa - previousAbscissa;

      // This check looks useless
      // if (std::abs(difference) / dAbscissa > 1e-5) {
      MDFT::Impl::Throw_If(
          (difference - dAbscissa) / dAbscissa > 1e-5,
          "Non-uniform abscissa detected in file: " + filename);
    }
    return dAbscissa;
  }
};

}  // namespace IO
}  // namespace MDFT

#endif
