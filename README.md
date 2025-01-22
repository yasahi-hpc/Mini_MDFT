<!--
SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

# Mini-MDFT

[![CI](https://github.com/yasahi-hpc/Mini_MDFT/actions/workflows/build_test.yml/badge.svg)](https://github.com/yasahi-hpc/Mini_MDFT/actions/workflows/build_test.yml)
[![Nightly](https://github.com/yasahi-hpc/Mini_MDFT/actions/workflows/nightly.yml/badge.svg)](https://github.com/yasahi-hpc/Mini_MDFT/actions/workflows/nightly.yml)

The Molecular Density Functional Theory is a collaborative work led by Daniel Borgis and Maximilien Levesque:

## Using Mini-MDFT
First of all, you need to clone this repo.
```bash
git clone --recursive https://github.com/yasahi-hpc/Mini_MDFT.git
```

### Prerequisites
Mini-MDFT is developed with C++20, thus you need newer compilers.
To use Mini-MDFT, we need the followings:
* `CMake 3.22+`
* `Kokkos 4.4+`
* `gcc 11.0.0+` (CPUs, c++20 support is necessary)
* `IntelLLVM 2025.0.0+` (CPUs, Intel GPUs)
* `nvcc 11.0.0+` (NVIDIA GPUs)
* `rocm 5.6.0+` (AMD GPUs)

### Compile and run

For compilation, we basically rely on the CMake options for Kokkos. For example, the compile options for A100 GPU is as follows.
```bash
cmake -B build \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DMini_MDFT_ENABLE_TESTS=ON
cmake --build build -j 8
```

To run the tests, please run the follwoing command.
```bash
cd build
ctest --output-on-failure
```

## Development strategy

1. Port [Mini-MDFT](https://github.com/LuJeMa/Mini_MDFT) function by function with unit-testing
2. Add docstrings for each function
3. Reconstruct I/O with `json/yaml` and `hdf5/netcdf`
4. Create a minimal benchmark case for integrated testing and compare the result with Fortran
5. Making a PR for each modification and ask for a review if you wish
6. Formatting your files by `clang-format -i <file_you_have_modified>`
7. Feel free to use AI tools to convert Fortran code into C++, adding unit-test and docstrings, etc

## LICENSE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mini-MDFT is distributed under the MIT license.
