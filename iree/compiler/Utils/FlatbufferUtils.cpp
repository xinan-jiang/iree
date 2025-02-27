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

#include "iree/compiler/Utils/FlatbufferUtils.h"

#include <vector>

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

// Combines all pages of the flatbuffer builder into a single contiguous byte
// buffer and returns the result.
//
// NOTE: this is a alloc/copy. We need to have a single contiguous buffer to
// pass into the elements factory function and the data we have in the
// builder is paged. If we end up with a custom attribute type for this that
// does not support storage uniquing then we can directly allocate and copy
// the pages into the buffer without the extra copy.
static SmallVector<uint8_t, 32> cloneBufferIntoContiguousBytes(
    FlatbufferBuilder &fbb) {
  size_t packedSize = flatcc_builder_get_buffer_size(fbb);
  SmallVector<uint8_t, 32> packedData(packedSize);
  void *result =
      flatcc_builder_copy_buffer(fbb, packedData.data(), packedData.size());
  assert(result && "flatcc_emitter_t impl failed (non-default?)");
  return packedData;
}

FlatbufferBuilder::FlatbufferBuilder() { flatcc_builder_init(&builder); }

FlatbufferBuilder::~FlatbufferBuilder() { flatcc_builder_clear(&builder); }

flatbuffers_uint8_vec_ref_t FlatbufferBuilder::streamUint8Vec(
    std::function<bool(raw_ostream &stream)> fn) {
  flatbuffers_uint8_vec_start(*this);
  raw_flatbuffer_uint8_vec_ostream stream(*this);
  if (!fn(stream)) {
    return 0;
  }
  stream.flush();
  return flatbuffers_uint8_vec_end(*this);
}

DenseIntElementsAttr FlatbufferBuilder::getBufferAttr(MLIRContext *context) {
  // We require direct access to the flatbuffer bytes so we can pass them to
  // the attribute constructor (which needs to inspect them all for uniquing).
  auto bufferData = cloneBufferIntoContiguousBytes(*this);

  // NOTE: ew. OpaqueAttr may be better? It does equality checks but won't try
  // to unique and would let us get a mutable buffer out.
  return DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(bufferData.size())},
                      IntegerType::get(context, 8)),
      std::move(bufferData));
}

LogicalResult FlatbufferBuilder::copyToStream(llvm::raw_ostream &output) {
  // NOTE: expected to be the default emitter.
  auto *E = reinterpret_cast<flatcc_emitter_t *>(
      flatcc_builder_get_emit_context(*this));

  if (!E->front) {
    return failure();
  }
  if (E->front == E->back) {
    output.write(reinterpret_cast<char *>(E->front_cursor), E->used);
    return success();
  }
  size_t len = FLATCC_EMITTER_PAGE_SIZE - E->front_left;
  output.write(reinterpret_cast<char *>(E->front_cursor), len);
  flatcc_emitter_page_t *p = E->front->next;
  while (p != E->back) {
    output.write(reinterpret_cast<char *>(p->page), FLATCC_EMITTER_PAGE_SIZE);
    p = p->next;
  }
  output.write(reinterpret_cast<char *>(p->page),
               FLATCC_EMITTER_PAGE_SIZE - E->back_left);
  return success();
}

LogicalResult FlatbufferBuilder::printJsonToStream(
    bool pretty, bool includeDefaults, print_json_fn_t print_json_fn,
    llvm::raw_ostream &output) {
  // The printer requires direct access to the flatbuffer bytes so clone here.
  auto bufferData = cloneBufferIntoContiguousBytes(*this);

  flatcc_json_printer_t printer;
  flatcc_json_printer_init_dynamic_buffer(&printer, /*buffer_size=*/0);
  flatcc_json_printer_set_indent(&printer, pretty ? 2 : 0);
  flatcc_json_printer_set_skip_default(&printer, !includeDefaults);
  flatcc_json_printer_set_force_default(&printer, includeDefaults);

  // Print into the dynamically-resizing buffer. May fail if OOM.
  int rv =
      print_json_fn(&printer, reinterpret_cast<const char *>(bufferData.data()),
                    bufferData.size());
  if (rv == -1) {
    flatcc_json_printer_clear(&printer);
    return failure();
  }

  // Take the buffer from the printer; note that it is 0 terminated and can be
  // used directly as a cstr if needed.
  size_t outputSize = 0;
  char *outputBytes = reinterpret_cast<char *>(
      flatcc_json_printer_finalize_dynamic_buffer(&printer, &outputSize));
  output.write(outputBytes, outputSize);
  free(outputBytes);

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
