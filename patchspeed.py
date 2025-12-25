import os
os.environ['DS_BUILD_OPS'] = '0'
os.environ['DS_SKIP_CUDA_CHECK'] = '1'

# Monkey patch the CUDA version check
import deepspeed.ops.op_builder.builder as builder

original_installed_cuda_version = builder.installed_cuda_version

def patched_installed_cuda_version():
    return (12, 8)

builder.installed_cuda_version = patched_installed_cuda_version

# Now import deepspeed normally
import deepspeed.launcher.runner as ds_runner
import sys

if __name__ == "__main__":
    ds_runner.main()
