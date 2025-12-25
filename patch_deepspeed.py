import os
os.environ['DS_BUILD_OPS'] = '0'
os.environ['DS_SKIP_CUDA_CHECK'] = '1'

# Monkey patch before DeepSpeed imports
import sys
import importlib.util

# Patch the builder module before it gets imported
spec = importlib.util.find_spec("deepspeed.ops.op_builder.builder")
if spec:
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    # Override the installed_cuda_version function
    def fake_cuda_version():
        return (12, 8)

    builder.installed_cuda_version = fake_cuda_version
    sys.modules['deepspeed.ops.op_builder.builder'] = builder

# Import DeepSpeed *after* the patch is applied
import deepspeed.launcher.runner as ds_runner

if __name__ == "__main__":
    # Note: If this script is intended to *launch* a DeepSpeed job,
    # the command line arguments for the actual training script (e.g., train.py)
    # need to be passed to ds_runner.main().
    # ds_runner.main()
    pass # Use this script as a pre-patch runner, and then execute the actual training
