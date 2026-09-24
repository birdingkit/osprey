# osprey

Bird-photo sorter CLI: `src/osprey/` — `cli.py` (pipeline + CSV), `detect.py` (Mask R-CNN bird box/mask), `quality.py` (masked re-blur sharpness), `taxa.py` (iNaturalist local species, cached), `species.py` (BioCLIP 2.5 zero-shot over local species).

## Build & Test

| Gate | Command |
|---|---|
| Lint | `uv run ruff check src tests && uv run ruff format --check src tests` |
| Unit tests | `uv run pytest -q` |
| End-to-end | `uv run osprey examples --place Taiwan -o /tmp/osprey.csv` → DSC06266 白腹鰹鳥 soft, DSC07300 白眉燕鷗 sharp, DSC07714 穴鳥 sharp |

`examples/` photos are local only (gitignored, 56 MB).
