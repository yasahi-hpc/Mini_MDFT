#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <filesystem>
#include "IO/MDFT_String_Utils.hpp"
#include "Test_Utils.hpp"

namespace {

struct TestStringUtils : public ::testing::Test {};

TEST_F(TestStringUtils, zfill) {
  ASSERT_EQ(MDFT::IO::Impl::zfill(1), "0001");
  ASSERT_EQ(MDFT::IO::Impl::zfill(123), "0123");
  ASSERT_EQ(MDFT::IO::Impl::zfill(123, 5), "00123");
}

TEST_F(TestStringUtils, trimRight) {
  ASSERT_EQ(MDFT::IO::Impl::trimRight("abc", "a"), "abc");
  ASSERT_EQ(MDFT::IO::Impl::trimRight("abc", "b"), "abc");
  ASSERT_EQ(MDFT::IO::Impl::trimRight("abc", "c"), "ab");
}

TEST_F(TestStringUtils, trimLeft) {
  ASSERT_EQ(MDFT::IO::Impl::trimLeft("abc", "a"), "bc");
  ASSERT_EQ(MDFT::IO::Impl::trimLeft("abc", "b"), "abc");
  ASSERT_EQ(MDFT::IO::Impl::trimLeft("abc", "c"), "abc");
}

TEST_F(TestStringUtils, trim) {
  ASSERT_EQ(MDFT::IO::Impl::trim("abc", "a"), "bc");
  ASSERT_EQ(MDFT::IO::Impl::trim("abc", "b"), "abc");
  ASSERT_EQ(MDFT::IO::Impl::trim("abc", "c"), "ab");
}

}  // namespace
