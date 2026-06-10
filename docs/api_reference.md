# API Reference

Quick reference of all public functions, classes, and configuration objects. Full docstrings are in the source code.

## Pipeline

### `src/pipeline/run_pipeline.py`

```python
def run_meeting_pipeline(
    input_audio_path: str,
    meeting_id: str,
    config: PipelineConfig | None = None,
) -> dict[str, Any]
```
End-to-end orchestration from audio file to per-meeting artifacts. See [pipeline_walkthrough.md](pipeline_walkthrough.md).

### `src/pipeline/config.py`

```python
@dataclass(frozen=True)
class PipelineConfig:
    outputs_root: Path = Path("outputs")
    target_sample_rate: int = 16000
    overlap_threshold: float = 0.4       # from DEFAULT_OVERLAP_THRESHOLD
    language: str = "und"

    def meeting_dir(self, meeting_id: str) -> Path
```

### `src/pipeline/io.py`

```python
def ensure_meeting_dirs(meeting_dir: str | Path) -> dict[str, Path]
def write_json(path: str | Path, payload: Any) -> None
def read_json(path: str | Path) -> Any
```

---

## Audio

### `src/audio/preprocess.py`

```python
TARGET_SAMPLE_RATE = 16000

def load_audio(audio_path: str, target_sample_rate: int = 16000,
               normalize: bool = True, target_peak: float = 0.97) -> tuple[np.ndarray, int]

def to_mono(samples: np.ndarray) -> np.ndarray

def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray

def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray

def peak_normalize(samples: np.ndarray, target_peak: float = 0.97) -> np.ndarray

def frame_rms(samples: np.ndarray, frame_length: int, hop_length: int
              ) -> tuple[np.ndarray, np.ndarray]

def energy_vad(samples: np.ndarray, sample_rate: int = 16000,
               frame_ms: float = 25.0, hop_ms: float = 10.0,
               threshold_ratio: float = 0.3, min_speech_ms: float = 1000.0,
               min_silence_ms: float = 300.0, speech_pad_ms: float = 50.0,
               merge_short_gap_s: float = 1.0,
               max_segment_s: float = 20.0, target_segment_s: float = 12.0
               ) -> list[tuple[float, float]]

def segment_waveform(samples: np.ndarray, sample_rate: int = 16000,
                     meeting_id: str = "meeting", **vad_kwargs) -> list[dict[str, Any]]

def segment_audio(audio_path: str, meeting_id: str = "meeting", **vad_kwargs) -> list[dict[str, Any]]

def preprocess_audio(input_path: str, output_path: str,
                     target_sample_rate: int = 16000, target_peak: float = 0.97,
                     target_sr: int | None = None) -> tuple[np.ndarray, int]
```

### `src/audio/clipper.py`

```python
def write_segment_clips(samples: np.ndarray, sample_rate: int,
                        segments: list[dict[str, Any]], clips_dir: str | Path
                        ) -> list[dict[str, Any]]
```

---

## Overlap Detection

### `src/overlap_detector.py`

```python
DEFAULT_OVERLAP_THRESHOLD = 0.4

def estimate_segment_overlap_scores(
    samples: np.ndarray, segments: list[dict[str, Any]],
    sample_rate: int = 16000, audio_path: str | None = None,
    overlap_regions: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]

def detect_pyannote_overlap_regions(
    audio_path: str, model_name: str = "pyannote/overlapped-speech-detection",
    auth_token: str | None = None,
) -> list[tuple[float, float]] | None

def estimate_overlap_score(audio_path: str) -> float

def detect_overlap_segments(audio_path: str, threshold: float = 0.4) -> list[dict]
```

---

## Routing

### `src/dual_path_router.py`

```python
def route_segment(overlap_score: float, threshold: float = 0.4) -> str
# Returns "low_overlap_cluster" or "high_overlap_candidate"
```

---

## ASR

### `src/asr.py`

```python
def logprob_to_confidence(avg_logprob: float, no_speech_prob: float = 0.0) -> float

class ASRAdapter:
    name = "base"
    def transcribe_array(self, samples: np.ndarray, sample_rate: int = 16000) -> dict
    def transcribe_file(self, audio_path: str) -> dict

class MockASRAdapter(ASRAdapter):    name = "mock"
class WhisperXAdapter(ASRAdapter):   name = "whisperx"
class WhisperAdapter(ASRAdapter):    name = "whisper"
class FunASRAdapter(ASRAdapter):     name = "funasr"

def get_adapter(name: str = "mock", **kwargs) -> ASRAdapter
def transcribe_audio(audio_path: str, adapter=None, model="mock") -> dict
def transcribe_segments(samples, segments, adapter, sample_rate=16000) -> list[dict]
```

**Transcript shape:** `{"text", "language", "model", "asr_confidence", "segments": [{"start_time", "end_time", "text", "asr_confidence"}]}`

---

## Diarization

### `src/diarization.py`

```python
def diarize_audio(audio_path: str) -> list[dict]
def cluster_speakers(segments: list[dict]) -> list[dict]
def assign_speakers_to_segments(segments: list[dict], diarization_turns=None) -> list[dict]
def diarize_with_pyannote(audio_path: str, model_name="pyannote/speaker-diarization-3.1", auth_token=None) -> list[dict] | None
```

---

## Low-Overlap Path

### `src/low_overlap.py`

```python
LOW_OVERLAP_PATH = "low_overlap_cluster"

def process_low_overlap_segments(
    samples,
    segments,
    asr_adapter,
    sample_rate=16000,
    diarization_turns=None,
) -> list[dict]
```

Returns low-overlap segment records with stable `text`, `speaker`, `start_time`, `end_time`, `asr_confidence`, `speaker_confidence`, empty `candidates`, and empty `uncertainty_note`.

---

## Candidate Generation

### `src/high_overlap.py`

```python
HIGH_OVERLAP_PATH = "high_overlap_candidate"
HIGH_OVERLAP_SPEAKER = "MIXED"

def process_high_overlap_segments(
    samples,
    segments,
    sample_rate=16000,
    language=None,
) -> list[dict]
```

Returns high-overlap segment records with `speaker="MIXED"`, empty main `text`, low `speaker_confidence`, and populated `candidates`.

### `src/candidate_generator.py`

```python
def generate_high_overlap_candidates(
    segment: dict,
    samples=None,
    sample_rate=16000,
    language=None,
    decode_configs=None,
    max_candidates=4,
) -> list[dict]
# Returns list of {"candidate_id", "speaker", "text", "confidence", "uncertainty_note"}
```

When waveform samples are provided and faster-whisper is installed, candidates are generated with multiple beam/temperature/language decode settings. Otherwise explicit fallback candidates preserve the uncertainty contract.

---

## Metadata & Validation

### `src/metadata_builder.py`

```python
def build_metadata_segment(
    meeting_id, segment_id, speaker, start_time, end_time, text,
    processing_path, overlap_score, asr_confidence, speaker_confidence,
    candidates=None, uncertainty_note="",
    evidence_id=None, audio_clip_path="", source_audio_path="",
    language="und", route_reason="",
) -> dict[str, Any]
```

### `src/schema_validation.py`

```python
VALID_PROCESSING_PATHS = {"low_overlap_cluster", "high_overlap_candidate"}

def validate_candidate(candidate: Any, index: int = 0) -> dict[str, Any]
def validate_metadata_segment(record: Any) -> dict[str, Any]
def validate_meeting(segments: Any) -> list[dict[str, Any]]
```

---

## LLM

### `src/llm/event_extractor.py`

```python
def extract_meeting_events(
    evidence_segments: list[dict[str, Any]],
    client: GemmaClient | None = None,
) -> list[dict[str, Any]]
```

### `src/llm/event_validator.py`

```python
def validate_meeting_event(event: Any, known_evidence_ids: set[str] | None = None) -> dict[str, Any]
```

**Event shape:** `{"meeting_id", "event_id", "summary", "evidence_ids", "confidence", "uncertainty_note"}`

### `src/llm/gemma_client.py`

```python
class GemmaClient:
    def generate_json(self, prompt: str) -> dict[str, Any]
```

### `src/llm/prompts.py`

```python
def build_event_extraction_prompt(evidence_segments: list[dict[str, Any]]) -> str
```

### `src/llm_postprocess.py`

```python
SYSTEM_INSTRUCTIONS: str

def build_llm_prompt_with_metadata(segments: list[dict], memory_context=None) -> str
def uncertainty_aware_correction(segments: list[dict]) -> list[dict]  # stub
def generate_evidence_based_summary(segments: list[dict]) -> dict      # stub
```

---

## Episodic Memory

### `src/episodic_memory.py`

```python
def create_episode_from_segments(segments: list[dict]) -> dict[str, Any]
def store_episode(episode: dict, path: str | Path = "outputs/episodic_memory.jsonl") -> None
def search_episodes(query: str, top_k: int = 5,
                    path: str | Path = "outputs/episodic_memory.jsonl") -> list[dict]
```

**Episode shape:** `{"meeting_id", "episode_id", "start_time", "end_time", "speakers", "topic", "summary", "evidence_ids", "evidence", "confidence", "uncertainty_note"}`

---

## QA

### `src/rag_qa.py`

```python
def retrieve_relevant_memory(query: str, top_k: int = 5) -> list[dict]
def answer_question_with_evidence(query: str, retrieved_episodes: list[dict]) -> dict
```

**QA answer shape:** `{"answer", "evidence", "speaker", "timestamp", "confidence", "uncertainty_note", "query"}`

---

## Evaluation

### `src/evaluation.py`

```python
def edit_distance(reference: list, hypothesis: list) -> dict[str, int]
def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]
def character_error_rate(reference: str, hypothesis: str) -> dict[str, Any]
def evaluate_overlap_routing(predictions: list[str], references: list[str]) -> dict[str, Any]
def speaker_attribution_accuracy(reference: list[str], hypothesis: list[str]) -> dict[str, Any]
def evaluate_evidence_support(predictions: list[dict], references: list[dict]) -> dict  # stub
```

---

## Data Synthesis

### `src/data_synthesis.py`

```python
def signal_power(samples: np.ndarray) -> float
def scale_to_snr(reference: np.ndarray, signal: np.ndarray, snr_db: float) -> np.ndarray
def mix_two_speakers(speaker_a, speaker_b, overlap_s, sample_rate=16000,
                     snr_db=0.0, speaker_a_label="SPEAKER_00",
                     speaker_b_label="SPEAKER_01", meeting_id="synth"
                     ) -> tuple[np.ndarray, dict]
def overlap_intervals(segments: list[dict]) -> list[tuple[float, float]]
def total_overlap_duration(segments: list[dict]) -> float
def overlap_ratio(segments: list[dict], duration_s: float) -> float
def build_annotation(segments, sample_rate, total_samples, meeting_id) -> dict
def to_annotation_rows(annotation: dict) -> list[dict]
```

---

## UI

### `src/ui/gradio_app.py`

```python
def build_app()       # returns gradio.Blocks
def launch() -> None  # starts the Gradio server
```

---

## Utilities

### `src/utils.py`

```python
def validate_score(score: float, name: str = "score") -> float  # clamps to [0, 1]
```
