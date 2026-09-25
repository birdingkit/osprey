# osprey

Photo burst grouper CLI: `src/osprey/cli.py` — find photos + same-name sidecars, split into bursts by EXIF capture time (gap < 1 s), rename in place `NNNN_<orig>`.

## Build & Test

| Gate | Command |
|---|---|
| Lint | `uv run ruff check src tests && uv run ruff format --check src tests` |
| Unit tests | `uv run pytest -q` |
| End-to-end | `rm -rf /tmp/osprey-e2e && cp -R examples /tmp/osprey-e2e && uv run osprey /tmp/osprey-e2e` → 3 photos in 3 bursts, `0001_DSC06266`, `0002_DSC07300`, `0003_DSC07714` (copy first: osprey renames files) |

`examples/` photos are local only (gitignored, 56 MB).

## Public repo guard

Repo is public. `.githooks/pre-commit` blocks photos/raw/video/CSV/XMP, files > 1 MB, absolute home paths and gitleaks hits. Enable per clone: `git config core.hooksPath .githooks` (needs `brew install gitleaks`). Never bypass with `--no-verify`.
