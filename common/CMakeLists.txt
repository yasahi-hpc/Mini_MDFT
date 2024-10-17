add_library(Mini-MDFT-Common INTERFACE)

target_link_libraries(Mini-MDFT-Common
    INTERFACE
    Kokkos::kokkos
    KokkosFFT::fft
)

target_include_directories(Mini-MDFT-Common INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)