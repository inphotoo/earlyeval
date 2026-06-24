# core

`core/` contains shared contracts, path handling, table IO, prefix-builder
bridges, and split validation helpers. It should stay lightweight and should
not start training jobs at import time.

Important files:

- `contracts.py`: dataclasses for prefix rows, prediction rows, and policies.
- `io.py`: lightweight table and JSON IO helpers.
- `paths.py`: project path resolution.
- `legacy_paths.py`: compatibility path helpers. Active runtime code resolves
  to the vendored earlyeval source tree.
- `prefix.py`: bridge functions for the vendored prefix/step builders.
- `splits.py`: split validation helpers.

