#!/bin/bash
echo "Setting up local editable environment..."
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
pip install https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.4/ml_switcheroo_ir-0.0.4-py3-none-any.whl || pip install -e ../ml-switcheroo-ir
pip install https://github.com/SamuelMarks/ml-switcheroo-compiler/releases/download/v0.0.1/ml_switcheroo_compiler-0.0.1-py3-none-any.whl || pip install --no-deps -e ../ml-switcheroo-compiler
if [ -d "../griffe/packages/griffelib" ]; then
  pip install --no-deps -e ../griffe/packages/griffelib -e ../griffe/packages/griffecli -e ../griffe
fi
pip install --no-deps https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl || pip install --no-deps -e ../ml-ecosystem-snapshots
pip install --no-deps -e ../zero-jax
pip install --no-deps -e ../zero-chex
pip install --no-deps -e ../zero-flax
pip install --no-deps -e ../zero-grain
pip install --no-deps -e ../zero-optax
pip install --no-deps -e .
echo "Done!"
