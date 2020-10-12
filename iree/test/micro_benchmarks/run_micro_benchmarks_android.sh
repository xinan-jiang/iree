#!/bin/bash

set -x

export IREE_LLVMAOT_LINKER_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++  -static-libstdc++ -ffp-contract=fast "

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
BACKEND=${1}
ARGS=${@:2}

adb shell 'mkdir /data/local/tmp'

for MLIR_FILE_PATH in $SCRIPTPATH/*.mlir; do
    MLIR_FILE_NAME=$(basename $MLIR_FILE_PATH)
    TARGET_VM_FILE=/tmp/$MLIR_FILE_NAME.fbvm
    ${IREE_RELEASE_ANDROID_DIR}/host/bin/iree-translate --iree-hal-target-backends=${BACKEND} --iree-mlir-to-vm-bytecode-module --iree-llvm-target-triple=aarch64-linux-android ${ARGS} ${MLIR_FILE_PATH} -o ${TARGET_VM_FILE}
    adb push ${TARGET_VM_FILE} '/data/local/tmp'
    adb shell </dev/null taskset 80 "data/local/tmp/iree-benchmark-module --driver=dylib  --module_file=/data/local/tmp/${MLIR_FILE_NAME}.fbvm";
done
