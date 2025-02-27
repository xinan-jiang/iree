// Copyright 2019 Google LLC
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

#include "iree/hal/vulkan/debug_reporter.h"

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

struct iree_hal_vulkan_debug_reporter_s {
  iree_allocator_t host_allocator;
  VkInstance instance;
  iree::hal::vulkan::DynamicSymbols* syms;
  const VkAllocationCallbacks* allocation_callbacks;
  VkDebugUtilsMessengerEXT messenger;
};

// NOTE: |user_data| may be nullptr if we are being called during instance
// creation. Otherwise it is a pointer to the DebugReporter instance.
//
// NOTE: this callback must be thread safe and must be careful not to reach too
// far outside of the call - it is called in-context from arbitrary threads with
// some amount of Vulkan state on the stack. Assume that creating or deleting
// Vulkan objects, issuing most Vulkan commands, etc are off-limits.
static VKAPI_ATTR VkBool32 VKAPI_CALL
iree_hal_vulkan_debug_utils_message_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    IREE_LOG(ERROR) << callback_data->pMessage;
  } else {
    IREE_VLOG(1) << callback_data->pMessage;
  }
  return VK_FALSE;  // VK_TRUE is reserved for future use.
}

// Populates |create_info| with an instance-agnostic callback.
// This can be used during instance creation by chaining the |create_info| to
// VkInstanceCreateInfo::pNext.
//
// Only use if VK_EXT_debug_utils is present.
static void iree_hal_vulkan_debug_reporter_populate_create_info(
    VkDebugUtilsMessengerCreateInfoEXT* out_create_info) {
  out_create_info->sType =
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  out_create_info->pNext = nullptr;
  out_create_info->flags = 0;

  // TODO(benvanik): only enable the severities that logging has enabled.
  out_create_info->messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

  // TODO(benvanik): allow filtering by category as a flag.
  out_create_info->messageType =
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

  out_create_info->pfnUserCallback =
      iree_hal_vulkan_debug_utils_message_callback;
  out_create_info->pUserData = nullptr;
}

iree_status_t iree_hal_vulkan_debug_reporter_allocate(
    VkInstance instance, iree::hal::vulkan::DynamicSymbols* syms,
    const VkAllocationCallbacks* allocation_callbacks,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_debug_reporter_t** out_reporter) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(out_reporter);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate our struct first as we need to pass the pointer to the userdata
  // of the messager instance when we create it.
  iree_hal_vulkan_debug_reporter_t* reporter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*reporter),
                                (void**)&reporter));
  reporter->host_allocator = host_allocator;
  reporter->instance = instance;
  reporter->syms = syms;
  reporter->allocation_callbacks = allocation_callbacks;

  VkDebugUtilsMessengerCreateInfoEXT create_info;
  iree_hal_vulkan_debug_reporter_populate_create_info(&create_info);
  create_info.pUserData = reporter;
  iree_status_t status = VK_RESULT_TO_STATUS(
      syms->vkCreateDebugUtilsMessengerEXT(
          instance, &create_info, allocation_callbacks, &reporter->messenger),
      "vkCreateDebugUtilsMessengerEXT");

  if (iree_status_is_ok(status)) {
    *out_reporter = reporter;
  } else {
    iree_hal_vulkan_debug_reporter_free(reporter);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_vulkan_debug_reporter_free(
    iree_hal_vulkan_debug_reporter_t* reporter) {
  if (!reporter) return;
  iree_allocator_t host_allocator = reporter->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (reporter->messenger != VK_NULL_HANDLE) {
    reporter->syms->vkDestroyDebugUtilsMessengerEXT(
        reporter->instance, reporter->messenger,
        reporter->allocation_callbacks);
  }
  iree_allocator_free(host_allocator, reporter);

  IREE_TRACE_ZONE_END(z0);
}
