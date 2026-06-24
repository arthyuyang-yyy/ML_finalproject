# experiments/

Independent experiment layer for the meeting-memory project. **Does not modify** any file under `src/`, `scripts/`, or `data/`. The whole layer imports the project's existing functions and calls `main.py` for pipeline runs.

## Layout

```
experiments/
├── README.md              ← you are here
├── EVAL_PLAN.md           ← 4D matrix + scoring logic + reuse map
├── matrix/                ← 24 cells × 8 meetings grid definition (yaml)
├── runs/                  ← per (cell × meeting) run outputs
│   └── <cell_id>/<meeting_id>/
│       ├── outputs/                   ← what main.py writes (verbatim)
│       ├── run_meta.json              ← config + wall_time + detector counts
│       └── evaluation.json            ← all scores (this layer's output)
├── results/               ← aggregates (per_cell / per_meeting / summary)
├── clips/                 ← optional 5-min pre-cut wavs (smoke tests)
└── scripts/
    ├── build_clips.py     ← cut 5-min wavs from Eval_Ali_far
    ├── build_manifest.py  ← rewrite macOS paths → local
    ├── build_matrix.py    ← emit matrix/*.yaml
    ├── run_matrix.py      ← batch main.py invocations (with resume)
    ├── evaluate_runs.py   ← per-run scorer (this file is the one to read first)
    └── aggregate.py       ← per-cell / per-axis slice → summary.csv / summary.md
```

## 4D experiment matrix (EVAL_PLAN.md has the detail)

| Axis | Levels |
|---|---|
| A. ASR backend | `faster-whisper` · `funasr` · `whisperx` |
| B. OSD | `pyannote` · `energy_fallback` |
| C. LLM resolver | `none` · `deepseek` |
| D. Speech separation | `none` · `sepformer` |

= **3 × 2 × 2 × 2 = 24 cells × 8 meetings = 192 runs** at full scale.

## Run

```bash
# 1. (optional) cut 5-min clips for fast smoke tests
uv run python experiments/scripts/build_clips.py

# 2. (optional) rewrite paths and emit matrix yaml
uv run python experiments/scripts/build_manifest.py
uv run python experiments/scripts/build_matrix.py

# 3. batch pipeline
uv run python experiments/scripts/run_matrix.py

# 4. score every run
uv run python experiments/scripts/evaluate_runs.py

# 5. aggregate
uv run python experiments/scripts/aggregate.py
```

## Status

- `experiments/runs/asr=funasr_osd=pyannote_resolver=deepseek_sep=none/R8001_M8004_MS801/` — full 26-min audio single-cell run; smoke test for structure + scoring
- Aggregator and matrix scaffolding pending
