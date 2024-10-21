#ifndef MDFT_CONCEPTS_HPP
#define MDFT_CONCEPTS_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

namespace MDFT {

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename ExecutionSpace, typename ViewType>
concept KokkosViewAccesible = requires(ExecutionSpace e, ViewType v) {
  [] {
    static_assert(Kokkos::SpaceAccessibility<
                  ExecutionSpace, typename ViewType::memory_space>::accessible);
  }();
};

}  // namespace MDFT

#endif
