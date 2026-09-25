# osprey

Bird-photo sorter CLI: `src/osprey/` — `cli.py` (find photos + same-name sidecars, move into A_sharp/B_soft/C_blurry/D_no_bird), `detect.py` (Mask R-CNN bird box/mask), `quality.py` (masked re-blur sharpness).

## Build & Test

| Gate | Command |
|---|---|
| Lint | `uv run ruff check src tests && uv run ruff format --check src tests` |
| Unit tests | `uv run pytest -q` |
| End-to-end | `rm -rf /tmp/osprey-e2e && cp -R examples /tmp/osprey-e2e && uv run osprey /tmp/osprey-e2e` → DSC06266 Brown Booby `B_soft/`, DSC07300 Bridled Tern `A_sharp/`, DSC07714 Bulwer's Petrel `A_sharp/` (copy first: osprey moves files) |

`examples/` photos are local only (gitignored, 56 MB).

## Public repo guard

Repo is public. `.githooks/pre-commit` blocks photos/raw/video/CSV/XMP, files > 1 MB, absolute home paths and gitleaks hits. Enable per clone: `git config core.hooksPath .githooks` (needs `brew install gitleaks`). Never bypass with `--no-verify`.
