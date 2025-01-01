#include "mini-mdft.hpp"
#include <stdexcept>
#include <iostream>

using execution_space = Kokkos::DefaultExecutionSpace;
using float_type      = double;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    execution_space exec;
    MDFT::Solver<execution_space, float_type> solver(exec);

    try {
      solver.initialize(&argc, &argv);
      solver.run();
      solver.finalize();

      return 0;
    } catch (std::runtime_error e) {
      std::cerr << e.what() << std::endl;
      solver.finalize();
    }
  }
  Kokkos::finalize();
  return 0;
}
