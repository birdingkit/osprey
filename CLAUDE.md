# osprey

Bird-photo sorter CLI: `src/osprey/` — `cli.py` (find photos + same-name sidecars, split into bursts by EXIF time, rename in place `NNNN-RR_` sharpest first), `detect.py` (OWL-ViT bird/dolphin/whale box + SAM 2 mask), `quality.py` (masked re-blur sharpness).

## Build & Test

| Gate | Command |
|---|---|
| Lint | `uv run ruff check src tests && uv run ruff format --check src tests` |
| Unit tests | `uv run pytest -q` |
| End-to-end | `rm -rf /tmp/osprey-e2e && cp -R examples /tmp/osprey-e2e && uv run osprey /tmp/osprey-e2e` → scores DSC06266 Brown Booby 60.3, DSC07300 Bridled Tern 70.2, DSC07714 Bulwer's Petrel 64.1; each its own burst `000N-01_` (copy first: osprey renames files) |

`examples/` photos are local only (gitignored, 56 MB).

## Public repo guard

Repo is public. `.githooks/pre-commit` blocks photos/raw/video/CSV/XMP, files > 1 MB, absolute home paths and gitleaks hits. Enable per clone: `git config core.hooksPath .githooks` (needs `brew install gitleaks`). Never bypass with `--no-verify`.
