# SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(
    Mini_MDFT
    VERSION 0.1.0
    LANGUAGES CXX
)

# Options
option(Mini_MDFT_ENABLE_TESTS "Build Mini-MDFT tests" OFF)
option(Mini_MDFT_ENABLE_INTERNAL_KOKKOS "Build internal Kokkos instead of relying on external one" OFF)
option(Mini_MDFT_ENABLE_INTERNAL_KOKKOSFFT "Build internal Kokkos-fft instead of relying on external one" OFF)

# Version information
set(MINI_MDFT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(MINI_MDFT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(MINI_MDFT_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(KOKKOS_REQUIRED_VERSION 4.4.0)
set(KOKKOSFFT_REQUIRED_VERSION 0.2.1)

if (NOT Mini_MDFT_ENABLE_INTERNAL_KOKKOS)
    # First check, Kokkos is added as subdirectory or not
    if(NOT TARGET Kokkos::kokkos)
        find_package(Kokkos ${KOKKOS_REQUIRED_VERSION} REQUIRED)
    endif()
else ()
    add_subdirectory(tpls/kokkos)
endif ()

if (NOT Mini_MDFT_ENABLE_INTERNAL_KOKKOSFFT)
    # First check, Kokkos is added as subdirectory or not
    if(NOT TARGET KokkosFFT::fft)
        find_package(KokkosFFT ${KOKKOSFFT_REQUIRED_VERSION} REQUIRED)
    endif()
else ()
    add_subdirectory(tpls/kokkos-fft)
endif ()

# Not decided how to deal with json yet
add_subdirectory(tpls/json)

# ==================================================================
# CMake Summary
# ==================================================================

message("")
message(STATUS "Mini-MDFT version: ${MINI_MDFT_VERSION_MAJOR}.${MINI_MDFT_VERSION_MINOR}.${MINI_MDFT_VERSION_PATCH}")

# GoogleTest
if(Mini_MDFT_ENABLE_TESTS)
    include(CTest)
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        add_subdirectory(tpls/googletest)
    endif()
    add_subdirectory(unit_test)
endif()

if(Mini_MDFT_ENABLE_BENCHMARKS)
  add_subdirectory(perf_test)
endif()

add_subdirectory(common)
add_subdirectory(src)
