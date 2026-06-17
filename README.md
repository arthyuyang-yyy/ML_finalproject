# Overlap-aware Dual-path ASR with Episodic Memory

中文名称：面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

本项目面向多人会议音频，目标是生成可追溯的结构化会议记录和 Episodic Memory。系统会识别说话人、估计语音重叠程度、对低重叠和高重叠片段走不同处理路径，并把后续问答绑定到 evidence segment、时间戳和原始音频片段。

当前工程目标是保留一条清晰、可运行、可演示的主流程，避免过度抽象和过度文档化。

## 项目目标

系统输入一段会议音频，输出：

- 带时间戳的会议片段；
- 说话人标签；
- 低/高重叠路由结果；
- ASR 文本和置信度；
- 高重叠片段的候选转写、最终解析文本和决策原因；
- 统一的 `evidence_segments.json`；
- 结构化会议事件；
- 可检索的 `episodic_memory.json`；
- 带 evidence 引用的问答结果。

核心思路不是把所有音频交给一个端到端模型，而是先根据重叠程度分流：

- 低重叠片段：走稳定 ASR 和说话人归属路径；
- 高重叠片段：保留多个候选转写，再由 LLM resolver 或 fallback resolver 选择最终结果。

## Pipeline

```text
input audio
-> preprocess audio
-> VAD segmentation
-> diarization
-> overlap scoring
-> route by overlap_score
   -> low_overlap_cluster: ASR + speaker attribution
   -> high_overlap_candidate: candidate generation + resolver
-> evidence_segments.json
-> meeting_events.json
-> episodic_memory.json
-> retrieval QA
```

## 当前项目结构

```text
.
├── app.py                         # Gradio demo entry point
├── main.py                        # CLI pipeline entry point
├── requirements.txt               # Unified dependency file
├── README.md
├── Project_task.md                 # Current task scope and acceptance notes
├── TODO.md                         # Current follow-up tracker
├── data/
│   ├── raw_audio/                 # Raw meeting audio, ignored except placeholders
│   ├── processed_audio/           # Processed audio, ignored except placeholders
│   ├── fixtures/                  # Lightweight test/demo fixtures
│   ├── annotations/               # Annotation templates
│   └── manifests/                 # Dataset manifests, if used
├── memory/
│   └── episodic_memory.json       # Long-term memory store
├── docs/                          # Compact reference notes for optional areas
├── outputs/                       # Generated pipeline artifacts
├── scripts/                       # Optional dataset/benchmark helpers
├── src/
│   ├── audio/                     # Audio loading, preprocessing, VAD, clipping
│   ├── asr/                       # ASR adapters and transcription helpers
│   ├── candidates/                # High-overlap candidate generation
│   ├── datasets/                  # Optional dataset manifest helpers
│   ├── diarization/               # Speaker diarization adapters
│   ├── evaluation/                # Lightweight objective metrics
│   ├── evidence/                  # Evidence segment schema, builder, validator
│   ├── fallbacks/                 # Deterministic fallback backends
│   ├── llm/                       # Gemma client, event extraction, high-overlap resolver
│   ├── memory/                    # Episodic memory store and retriever
│   ├── overlap/                   # Overlap detection and routing
│   ├── pipeline/                  # End-to-end orchestration
│   ├── qa/                        # Evidence-backed QA
│   └── ui/                        # Gradio app
└── tests/                         # Unit and integration tests
```

## 关键模块

### Audio

`src/audio/preprocess.py` 负责：

- 读取音频；
- 转单声道；
- 重采样到 16 kHz；
- 峰值归一化；
- 基于能量的 VAD 分段。

`src/audio/clipper.py` 会根据 evidence segment 的时间戳导出每段音频 clip。

### Diarization and Overlap

`src/diarization/` 负责说话人分离。优先使用 pyannote；没有模型或 token 时，代码保持可导入、可测试。

`src/overlap/` 负责估计每个片段的 `overlap_score` 并做路由：

```python
if overlap_score < 0.4:
    processing_path = "low_overlap_cluster"
else:
    processing_path = "high_overlap_candidate"
```

### Low-overlap Path

低重叠片段进入 `src/low_overlap.py`：

- 根据 diarization 结果分配 speaker；
- 调用 ASR adapter 转写；
- 输出文本、ASR 置信度、说话人置信度。

### High-overlap Path

高重叠片段进入 `src/high_overlap.py` 和 `src/candidates/generator.py`：

- 生成多个候选转写；
- 可选调用 speech separation adapter 生成 separated-source 候选；
- 保留候选 speaker、文本、置信度和 decode 配置；
- 不直接丢弃不确定性。

随后进入 `src/llm/resolver.py`：

- 如果配置了 Gemma/Ollama，则让 LLM 基于候选结果输出最终文本；
- 如果没有 LLM，则选择最高置信候选作为 `fallback_resolved`；
- 最终 evidence segment 会保留：
  - `text`
  - `speaker`
  - `candidates`
  - `source`
  - `decision_reason`
  - `uncertainty_note`

Speech separation 当前是可选增强，不是默认主路径。`src/speech_separation.py` 保留 `none`、`mock`、`nmf` 和 `sepformer` adapter；pipeline 默认 `speech_separation_backend="none"`，需要时才为高重叠片段补充 separated-source candidates。resolver 不替代 speech separation，而是在候选生成之后负责最终选择或合并。

### Evidence

`src/evidence/` 统一低重叠和高重叠输出，生成 `evidence_segments.json`。

核心字段包括：

| Field | Meaning |
| --- | --- |
| `meeting_id` | Meeting identifier |
| `segment_id` | Segment identifier |
| `evidence_id` | Evidence identifier |
| `speaker` | Speaker label or resolved speaker |
| `start_time`, `end_time` | Segment timestamps |
| `text` | Final transcript text |
| `processing_path` | `low_overlap_cluster` or `high_overlap_candidate` |
| `route_reason` | Routing explanation |
| `overlap_score` | Overlap score in `[0, 1]` |
| `asr_confidence` | ASR / resolved confidence |
| `speaker_confidence` | Speaker confidence |
| `audio_clip_path` | Exported clip path |
| `source_audio_path` | Original audio path |
| `language` | Language code |
| `candidates` | High-overlap candidates |
| `uncertainty_note` | Uncertainty explanation |
| `source` | Optional resolver source |
| `decision_reason` | Optional resolver reason |

### Episodic Memory and Retrieval

`src/memory/episodic_store.py` 把会议事件转成 episode，并写入：

```text
outputs/{meeting_id}/episodic_memory.json
memory/episodic_memory.json
```

`src/memory/retriever.py` 使用轻量检索：

- 自定义 BLAKE2 character n-gram hash embedding；
- 简单 keyword score；
- 默认分数：`0.70 * embedding_similarity + 0.30 * keyword_score`；
- 支持 meeting、speaker、time range 过滤。

自定义 embedding 实现在：

```text
src/fallbacks/embeddings.py
```

这是当前项目保留的轻量特色能力，不依赖 `sentence-transformers`。

当前检索是 MVP 权衡：优先保证 deterministic、低依赖和可测试，不引入 transformer embedding、recency decay、importance prior 或跨会议个性化排序。QA 返回的 evidence IDs、timestamps 和 uncertainty 信息用于降低误答风险；后续如果有人工评估集，再重新加入更复杂的排序信号。

## Architecture Notes

精简架构说明见 `docs/system_architecture.md`。该文档只保留当前主流程、关键契约和可选后端边界，避免恢复过多历史文档。

### QA

`src/qa/answerer.py` 只基于检索到的 episodes 回答问题，并返回 evidence IDs、timestamps、speakers 和 uncertainty 信息。

## 输出目录

运行一次 pipeline 后会生成：

```text
outputs/{meeting_id}/
├── preprocessed.wav
├── vad_segments.json
├── diarization.json
├── overlap.json
├── low_overlap_segments.json
├── high_overlap_candidates.json
├── evidence_segments.json
├── meeting_events.json
├── episodic_memory.json
└── clips/
```

## 安装

所有依赖统一在一个文件：

```bash
python -m pip install -r requirements.txt
```

说明：

- 只跑测试时，mock ASR 和 fallback backend 可以避免下载大型模型；
- Whisper、WhisperX、pyannote、FunASR 等生产后端会在实际使用时按需加载；
- 模型权重不要提交到仓库。

## 运行测试

```bash
python -m pytest -q
python -m ruff check src tests main.py app.py
```

当前验证状态：

```text
488 passed, 6 skipped, 2 warnings, 7 subtests passed
ruff: All checks passed
```

## 运行 CLI Pipeline

使用 mock ASR：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 --asr mock
```

使用 faster-whisper 和本地 Ollama Gemma：

```bash
python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --asr faster-whisper \
  --gemma-backend ollama \
  --gemma-model gemma3:4b
```

## 启动 Gradio Demo

```bash
python app.py
```

Demo 包含：

- 音频上传；
- pipeline 执行；
- 时间线展示；
- 高重叠候选展示；
- Episodic Memory 展示；
- 证据问答。

## 当前状态

已完成：

- 端到端 pipeline；
- 音频预处理和 VAD；
- diarization / overlap adapter；
- 低重叠 ASR 路径；
- 高重叠候选生成；
- 高重叠 resolver；
- evidence schema 和 validator；
- Episodic Memory；
- 自定义 BLAKE2 hash embedding 检索；
- evidence-backed QA；
- Gradio demo；
- 自动化测试。

仍建议后续优先补：

- 真实会议音频样例；
- 人工标注的 overlap / speaker / transcript 评估集；
- 高重叠 resolver 的质量评估；
- WhisperX / faster-whisper / pyannote 在真实音频上的运行记录。

## Git 注意事项

不要提交：

- 大型音频文件；
- 模型权重；
- `outputs/` 下生成结果；
- 本地环境文件。

仓库应优先保持主流程清楚、能运行、能演示。
