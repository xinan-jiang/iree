#!/bin/bash

set -x

export IREE_LLVMAOT_LINKER_PATH="/usr/bin/ld"

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
BACKEND=${1}
ARGS=${@:2}
for MLIR_FILE_PATH in $SCRIPTPATH/*.mlir; do
    MLIR_FILE_NAME=$(basename $MLIR_FILE_PATH)
    TARGET_VM_FILE=/tmp/$MLIR_FILE_NAME.fbvm
    ${IREE_RELEASE_DIR}/iree/tools/iree-translate --iree-hal-target-backends=${BACKEND} --iree-mlir-to-vm-bytecode-module ${ARGS} ${MLIR_FILE_PATH} -o ${TARGET_VM_FILE}
    ${IREE_RELEASE_DIR}/iree/tools/iree-benchmark-module --driver=dylib --module_file=${TARGET_VM_FILE}
done
