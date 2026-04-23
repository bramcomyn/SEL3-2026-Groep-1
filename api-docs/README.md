Command to generate new docs when something changes in the codebase:
```bash
rm -rf api-docs/build api-docs/source/api
uv run sphinx-apidoc -o api-docs/source/api src/brittle_star_locomotion
uv run make -C api-docs html
```
