# Evaluation Data

This folder holds the human-annotated meeting corpora used for **Phase 2 / Step 7b
— overlap-threshold calibration** and later evaluation. The audio and derived
annotations are **never committed to Git** (see `.gitignore`); only this README,
the preparation scripts, and final metric results are versioned.

## Why we need it

Overlap-threshold calibration needs ground-truth speaker timing. When two or more
speakers are active at the same time, that span is a true overlap region. Comparing
the detector's `overlap_score` against these true regions lets us pick the routing
threshold and report accuracy / precision / recall / F1.

## Primary dataset: AliMeeting Eval

- Real Chinese meeting recordings, 2–4 speakers, varied speech overlap.
- Far-field (microphone array) and near-field (headset) audio.
- Eval split is ~3.42 GB — the right size for a course project.
- License: **CC BY-SA 4.0**.

Download (use the **Eval** split only; do **not** download the ~96 GB training set):

- OpenSLR page: https://www.openslr.org/119/
- Direct: https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz

Citation: F. Yu et al., "M2MeT: The ICASSP 2022 Multi-Channel Multi-Party Meeting
Transcription Challenge," ICASSP 2022.

### Fallback / alternative corpora

- **AISHELL-4** (Chinese, 4–8 speakers, harder): Test ~5.2 GB — https://www.openslr.org/111/
- **AMI** (English, tiny annotation download ~22 MB, good for testing the parser
  first): https://groups.inf.ed.ac.uk/ami/corpus/ — License CC BY 4.0.

The final report should include Chinese data; AMI is only for validating the
annotation-parsing code quickly.

## Local layout (git-ignored)

```text
data/
├── README.md                 # this file (committed)
└── alimeeting/               # git-ignored — real data lives here
    ├── audio/                # *.wav from Eval_Ali
    └── annotations/          # *.TextGrid from Eval_Ali, and prepared *.json
```

## One shared copy, accessed by env var

Do not have every member re-download 3.42 GB. Put one copy on a shared
server / network drive, and point every machine at it:

```bash
export ALIMEETING_ROOT=/path/to/Eval_Ali       # Windows PowerShell: $env:ALIMEETING_ROOT="..."
```

Scripts read the root from this variable.

## Who needs to download

- **Needs it:** members working on data prep, overlap detection, and the
  threshold experiment.
- **Does not need it:** members on UI, LLM, Memory, or docs.
- **CI must not** download or depend on this corpus. The unit tests use small
  synthetic fixtures only.

## Workflow

```bash
# 1. Download Eval_Ali.tar.gz, extract under data/alimeeting/ (or your shared root).

# 2. Convert AliMeeting TextGrid annotations into the project format:
#    {meeting_id, speaker, start_time, end_time} turns + ground-truth overlap regions.
python scripts/prepare_alimeeting.py        # reads ALIMEETING_ROOT

# 3. (next step) Run overlap-threshold calibration and report acc / P / R / F1.
#    experiments/overlap_threshold/ — to be added.
```

### Detector under calibration: pyannote OSD

Step 7b calibrates the routing threshold for the **pyannote overlapped-speech
detector**: pyannote OSD produces the predicted `overlap_score` from the audio,
and the ground-truth overlap regions generated here (from AliMeeting speaker
turns) are the reference labels. This is a real calibration of a real detector,
not an oracle.

Enabling pyannote requires `pip install pyannote.audio` and an `HF_TOKEN` — see
the **Enable pyannote** steps in the main README. For an honest baseline, also run
the default **energy** detector (capped at 0.39): it is expected to route nothing
to the high-overlap path, which quantifies the gap between the tokenless fallback
and the real detector.
