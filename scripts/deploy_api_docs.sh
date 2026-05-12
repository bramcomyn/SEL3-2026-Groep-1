#!/bin/bash

# Build API docs
cd docs/api
rm -rf build source/api
uv run sphinx-apidoc -o source/api ../../src/brittle_star_locomotion
uv run make html

# Copy to api-docs
cd -
mkdir -p api-docs
cp -r docs/api/build/html/* api-docs/

# Deploy
ghp-import -n -p -f api-docs/
