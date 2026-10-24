#!/bin/bash
echo "Setting up local editable environment..."
pip install -e ../ml-switcheroo-ir
pip install -e ../ml-switcheroo-compiler
pip install https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.4/ml_ecosystem_snapshots-0.0.4-py3-none-any.whl || pip install -e ../ml-ecosystem-snapshots
pip install -e ../zero-jax
pip install -e .
echo "Done!"
