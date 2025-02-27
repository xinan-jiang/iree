// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thread>

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {

class SemaphoreTest : public CtsTestBase {};

// Tests that a semaphore that is unused properly cleans itself up.
TEST_P(SemaphoreTest, NoOp) {
  iree_hal_semaphore_t* semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 123ull, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(123ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Tests that a semaphore will accept new values as it is signaled.
TEST_P(SemaphoreTest, NormalSignaling) {
  iree_hal_semaphore_t* semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(2ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 40ull));
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(40ull, value);

  iree_hal_semaphore_release(semaphore);
}

// Note: Behavior is undefined when signaling with decreasing values, so we
// can't reliably test it across backends. Some backends may return errors,
// while others may accept the new, decreasing, values.

// Tests semaphore failure handling.
TEST_P(SemaphoreTest, Failure) {
  iree_hal_semaphore_t* semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3ull));
  uint64_t value;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(3ull, value);

  iree_hal_semaphore_fail(semaphore,
                          iree_status_from_code(IREE_STATUS_UNKNOWN));
  EXPECT_TRUE(
      iree_status_is_unknown(iree_hal_semaphore_query(semaphore, &value)));

  // Signaling again is undefined behavior. Some backends may return a sticky
  // failure status while others may silently process new signal values.

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on no semaphores.
TEST_P(SemaphoreTest, EmptyWait) {
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ANY, NULL, IREE_TIME_INFINITE_FUTURE));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ALL, NULL, IREE_TIME_INFINITE_FUTURE));

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_timeout(
      device_, IREE_HAL_WAIT_MODE_ANY, NULL, IREE_DURATION_INFINITE));
  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_timeout(
      device_, IREE_HAL_WAIT_MODE_ALL, NULL, IREE_DURATION_INFINITE));
}

// Tests waiting on a semaphore that has already been signaled.
TEST_P(SemaphoreTest, WaitAlreadySignaled) {
  iree_hal_semaphore_t* semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  // Test both previous and current values.
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      semaphore, 1ull, IREE_TIME_INFINITE_FUTURE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      semaphore, 2ull, IREE_TIME_INFINITE_FUTURE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_timeout(semaphore, 1ull,
                                                      IREE_DURATION_INFINITE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_timeout(semaphore, 2ull,
                                                      IREE_DURATION_INFINITE));

  iree_hal_semaphore_release(semaphore);
}

// Tests waiting on a semaphore that has not been signaled.
TEST_P(SemaphoreTest, WaitUnsignaled) {
  iree_hal_semaphore_t* semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 2ull, &semaphore));

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_semaphore_wait_with_deadline(
      semaphore, 3ull, IREE_TIME_INFINITE_PAST));

  iree_hal_semaphore_release(semaphore);
}

// Waiting on a failed semaphore is undefined behavior. Some backends may
// return UnknownError while others may succeed.

// Tests IREE_HAL_WAIT_MODE_ALL when not all are signaled.
TEST_P(SemaphoreTest, WaitAllButNotAllSignaled) {
  iree_hal_semaphore_t* semaphore_a;
  iree_hal_semaphore_t* semaphore_b;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ALL, &semaphore_list,
      IREE_TIME_INFINITE_PAST));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ALL when all are signaled.
TEST_P(SemaphoreTest, WaitAllAndAllSignaled) {
  iree_hal_semaphore_t* semaphore_a;
  iree_hal_semaphore_t* semaphore_b;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  // NOTE: we don't actually block here because otherwise we'd lock up.
  // Result status is undefined - some backends may return DeadlineExceededError
  // while others may return success.
  IREE_IGNORE_ERROR(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ALL, &semaphore_list,
      IREE_TIME_INFINITE_FUTURE));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests IREE_HAL_WAIT_MODE_ANY.
// **Fails using timeline semaphore emulation**
TEST_P(SemaphoreTest, DISABLED_WaitAny) {
  iree_hal_semaphore_t* semaphore_a;
  iree_hal_semaphore_t* semaphore_b;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore_a));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 1ull, &semaphore_b));

  iree_hal_semaphore_list_t semaphore_list;
  iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore_a, semaphore_b};
  semaphore_list.count = IREE_ARRAYSIZE(semaphore_ptrs);
  semaphore_list.semaphores = semaphore_ptrs;
  uint64_t payload_values[] = {1ull, 1ull};
  semaphore_list.payload_values = payload_values;

  IREE_ASSERT_OK(iree_hal_device_wait_semaphores_with_deadline(
      device_, IREE_HAL_WAIT_MODE_ANY, &semaphore_list,
      IREE_TIME_INFINITE_FUTURE));

  iree_hal_semaphore_release(semaphore_a);
  iree_hal_semaphore_release(semaphore_b);
}

// Tests threading behavior by ping-ponging between the test main thread and
// a little thread.
TEST_P(SemaphoreTest, PingPong) {
  iree_hal_semaphore_t* a2b;
  iree_hal_semaphore_t* b2a;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &a2b));
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &b2a));
  std::thread thread([&]() {
    // Should advance right past this because the value is already set.
    IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
        a2b, 0ull, IREE_TIME_INFINITE_FUTURE));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(b2a, 1ull));
    // Jump ahead (blocking at first).
    IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
        a2b, 4ull, IREE_TIME_INFINITE_FUTURE));
  });
  // Block until thread signals.
  IREE_ASSERT_OK(iree_hal_semaphore_wait_with_deadline(
      b2a, 1ull, IREE_TIME_INFINITE_FUTURE));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(a2b, 4ull));
  thread.join();

  iree_hal_semaphore_release(a2b);
  iree_hal_semaphore_release(b2a);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, SemaphoreTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
