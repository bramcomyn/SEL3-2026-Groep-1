Command to generate new docs when something changes in the codebase:
```bash
cd docs/api
rm -rf build source/api
uv run sphinx-apidoc -o source/api ../../src/brittle_star_locomotion
uv run make html
```
