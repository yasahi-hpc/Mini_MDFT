#ifndef MDFT_CONCEPTS_HPP
#define MDFT_CONCEPTS_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

namespace MDFT {

template <class>
struct is_kokkos_array : public std::false_type {};

template <class T, std::size_t N>
struct is_kokkos_array<Kokkos::Array<T, N>> : public std::true_type {};

template <class T, std::size_t N>
struct is_kokkos_array<const Kokkos::Array<T, N>> : public std::true_type {};

template <typename T>
concept KokkosArray = is_kokkos_array<T>::value;

template <typename T>
concept KokkosView = Kokkos::is_view_v<T>;

template <typename T>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<T>;

template <typename ExecutionSpace, typename ViewType>
concept KokkosViewAccesible = (bool)Kokkos::SpaceAccessibility<
    ExecutionSpace, typename ViewType::memory_space>::accessible;

}  // namespace MDFT

#endif
