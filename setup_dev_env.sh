#!/bin/bash
echo "Setting up local editable environment..."
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-switcheroo-compiler
if [ -d "../griffe/packages/griffelib" ]; then
  pip install -e ../griffe/packages/griffelib -e ../griffe/packages/griffecli -e ../griffe
fi
pip install https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl || pip install -e ../ml-ecosystem-snapshots
pip install -e ../zero-jax
pip install -e .
echo "Done!"
