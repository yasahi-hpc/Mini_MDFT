cmake_minimum_required(VERSION 3.22)
project(
    Mini_MDFT
    VERSION 0.0.0
    LANGUAGES CXX
)

# Options
option(Mini_MDFT_ENABLE_TESTS "Build Mini-MDFT tests" OFF)

add_subdirectory(tpls/json)
add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)

# GoogleTest
if(Mini_MDFT_ENABLE_TESTS)
    include(CTest)
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        add_subdirectory(tpls/googletest)
    endif()
    add_subdirectory(unit_test)
endif()

add_subdirectory(common)
