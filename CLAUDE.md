# osprey

Bird-photo sorter CLI: `src/osprey/` — `cli.py` (pipeline, burst ranking, CSV), `detect.py` (Mask R-CNN bird box/mask), `quality.py` (masked re-blur sharpness, clipped-pixel exposure), `taxa.py` (iNaturalist local species, cached), `species.py` (BioCLIP 2.5 zero-shot over local species).

## Build & Test

| Gate | Command |
|---|---|
| Lint | `uv run ruff check src tests && uv run ruff format --check src tests` |
| Unit tests | `uv run pytest -q` |
| End-to-end | `uv run osprey examples --place Taiwan -o /tmp/osprey.csv` → DSC06266 Brown Booby soft, DSC07300 Bridled Tern sharp, DSC07714 Bulwer's Petrel sharp; all exposure ok |

`examples/` photos are local only (gitignored, 56 MB).

## Public repo guard

Repo is public. `.githooks/pre-commit` blocks photos/raw/video/CSV/XMP, files > 1 MB, absolute home paths and gitleaks hits. Enable per clone: `git config core.hooksPath .githooks` (needs `brew install gitleaks`). Never bypass with `--no-verify`.
