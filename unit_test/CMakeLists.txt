add_executable(unit-tests-mini-mdft
    Test_Main.cpp
    Test_Grid.cpp
    Test_Wigner.cpp
    Test_Math_Utils.cpp
)

target_compile_features(unit-tests-mini-mdft PUBLIC cxx_std_20)
target_link_libraries(unit-tests-mini-mdft PUBLIC Mini-MDFT-Common GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-mini-mdft PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
