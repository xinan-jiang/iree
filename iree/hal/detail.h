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

#ifndef IREE_HAL_DETAIL_H_
#define IREE_HAL_DETAIL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dispatches a method on a HAL object vtable.
//
// In the future we can use this to compile in a mode where all indirect
// dispatches are replaced by direct calls to static methods. For example,
// by changing the macro to resolve to `iree_hal_[resource]_[method_name]` we
// can rely on LTO to perform cross-compilation unit inlining/strip unused HAL
// calls/etc. This will be particularly useful for super tiny builds
// (web/embedded) where there's only ever one usable backend and debugging
// features like command buffer validation aren't required.
//
// Some changes (mostly whackamole) are still required to fully support this and
// it's critical there's a CI building with the setting as it's not hard to keep
// working but very easy to accidentally break (by not routing through this
// interface, using the vtable for object instance comparison, etc).
#define IREE_HAL_VTABLE_DISPATCH(resource, type_prefix, method_name)       \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource)) \
       ->vtable)                                                           \
      ->method_name

// Defines the iree_hal_<type_name>_retain/_release methods.
#define IREE_HAL_API_RETAIN_RELEASE(type_name)                           \
  IREE_API_EXPORT void IREE_API_CALL iree_hal_##type_name##_destroy(     \
      iree_hal_##type_name##_t* type_name) {                             \
    if (IREE_LIKELY(type_name)) {                                        \
      IREE_HAL_VTABLE_DISPATCH(type_name, iree_hal_##type_name, destroy) \
      (type_name);                                                       \
    }                                                                    \
  }                                                                      \
  IREE_API_EXPORT void IREE_API_CALL iree_hal_##type_name##_retain(      \
      iree_hal_##type_name##_t* type_name) {                             \
    if (IREE_LIKELY(type_name)) {                                        \
      iree_atomic_ref_count_inc(                                         \
          &((iree_hal_resource_t*)(type_name))->ref_count);              \
    }                                                                    \
  }                                                                      \
  IREE_API_EXPORT void IREE_API_CALL iree_hal_##type_name##_release(     \
      iree_hal_##type_name##_t* type_name) {                             \
    if (IREE_LIKELY(type_name) &&                                        \
        iree_atomic_ref_count_dec(                                       \
            &((iree_hal_resource_t*)(type_name))->ref_count) == 1) {     \
      iree_hal_##type_name##_destroy(type_name);                         \
    }                                                                    \
  }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DETAIL_H_
