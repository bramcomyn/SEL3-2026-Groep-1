Command to generate new docs when something changes in the codebase:
```bash
rm -rf docs/build docs/source/api
uv run sphinx-apidoc -o docs/source/api src/brittle_star_locomotion
uv run make -C docs html
```
