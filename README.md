📢 【重要】机器学习大作业 GitHub 协作规范与工作要求
各位组员，为了保证我们项目（ML_finalproject）的代码质量，防止覆盖彼此的代码，也为了让大家更好地分工，我们已经开通了 GitHub 的自动化分支保护机制与代码审查流程。

以下是后续开发中所有人必须严格遵守的工作要求：

1. 🛑 绝对禁止直接 Push 到 main 分支
现在 main 分支已经完全锁死。任何人都无法（也禁止尝试）直接把代码推送到 main 分支。所有的日常开发和新功能编写，都必须在自己的本地分支上进行。

2. 💻 标准开发流程（每次写代码请遵循这 4 步）：
第一步：同步主分支
在本地切换到 main 分支并拉取最新代码，确保你的基础代码是最新的：
git checkout main ➡️ git pull origin main

第二步：切出自己的功能分支
根据你负责的模块，切出一个独立的特性分支（分支名用小写，可以用下划线或连字符，如 dev_data_preprocessing 或 feature_model_training）：
git checkout -b 你的分支名

第三步：在自己的分支上开发并提交
在这个分支上写代码，完成后提交并推送到 GitHub：
git add . ➡️ git commit -m "增加了XX模块/修复了XX问题" ➡️ git push origin 你的分支名

第四步：在 GitHub 网页端发起 Pull Request (PR)
登录 GitHub 网页，点击 Compare & pull request，请求将你的分支合并到 main 分支。

3. 🛡️ 代码终审权（PR 审批规则）
为了严格把关代码质量，系统已经配置了 Code Owners（代码所有者） 自动化机制：

任何组员提交 PR 之后，系统会自动指定 @arthuyuyang-yyy 以及另外两位负责人作为核心评审人（Reviewers）。

核心硬性规则：一个 PR 必须在 3 位负责人中拿到至少 2 个 Approve（通过），绿色的合并按钮才会解锁，代码才能合入 main。

修改意见处理：负责人在 Code Review（代码审查）时如果对某行代码提出了修改意见，会在 PR 里留下一个讨论（Conversation）。在作者修改完代码并点击 “Resolve conversation” 之前，该 PR 将被系统强制锁死，无法合并。

4. 📝 良好的 Commit 习惯
请不要在 commit 信息里写 “111”、“update” 这种模糊的字眼。请用简短的一句话说清楚你这次改了什么，方便大家以后回滚代码和写大作业的最终报告。

请大家务必按照这个规范来提交代码，第一次走流程如果遇到 git 报错或者网络问题，随时在群里呼叫负责人协助，大家加油！🚀
# Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding

[English](#project-title) | [中文完整翻译](#中文完整翻译) | [独立中文版](README.zh-CN.md)

## Project Title

**Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding**

Chinese title: 面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

## Motivation

Meeting assistants often compress imperfect transcripts into fluent summaries. This can hide speaker-attribution errors, overlapping speech, and unsupported conclusions. This project is not only a meeting summarization system. It is a **verifiable meeting memory system** that retains uncertainty and links later answers, decisions, and action items back to timestamped evidence.

## Difference from the Reference Thesis

The reference thesis already combines ASR, speaker diarization, low-overlap clustering, high-overlap speech separation, LLM correction, and structured meeting summaries. This project extends that foundation instead of reproducing it.

**Reference system**

`ASR -> speaker diarization -> LLM correction -> structured summary`

**Our system**

`ASR + overlap-aware routing -> uncertainty-aware candidate generation -> metadata-aware LLM post-processing -> Episodic Memory -> traceable QA and meeting recall`

The key change is that high-overlap speech is not forced into one confident transcript. Candidate interpretations and confidence metadata remain available to downstream reasoning and retrieval.

## Core Innovations

1. **Overlap-aware routing:** route low-overlap audio to lightweight speaker clustering and high-overlap audio to separation or candidate generation.
2. **Uncertainty-aware candidate generation:** preserve multiple plausible transcripts and speaker hypotheses for ambiguous regions.
3. **Metadata-aware LLM post-processing:** reason over timestamps, confidence, overlap, candidates, and prior memory rather than plain text alone.
4. **Episodic Memory:** store meaningful meeting events with evidence for traceable QA, action-item retrieval, and cross-meeting recall.
5. **Evaluation beyond WER and DER:** measure routing, candidate usefulness, uncertainty preservation, evidence quality, and hallucination.

## System Pipeline

1. Preprocess audio and create timestamped segments.
2. Estimate overlap scores.
3. Route each segment:
   - Low overlap: VAD, speaker embedding, clustering, and ASR.
   - High overlap: speech separation or multiple candidate interpretations.
4. Build a common metadata record for every segment.
5. Use an LLM to correct text, preserve uncertainty, and extract evidence-backed meeting events.
6. Convert related segments into Episodic Memory records.
7. Retrieve episodes to answer questions with speakers, timestamps, confidence, and uncertainty notes.

See [docs/system_architecture.md](docs/system_architecture.md) for the module-level design.

## Repository Structure

```text
.
├── docs/                  # Bilingual research design and experiment plans
├── data/                  # Raw/processed audio and annotation templates
├── outputs/               # Generated artifacts, ignored except placeholders
├── src/                   # Modular pipeline implementation
│   ├── audio/             # Audio preprocessing, normalization, export, and clipping
│   ├── pipeline/          # End-to-end orchestration, configuration, and I/O helpers
│   ├── overlap/           # Overlap detection facade
│   ├── evidence/          # Metadata builder and validator facade
│   ├── llm/               # LLM event extraction, validation, and prompt construction
│   ├── memory/            # Episodic memory facade
│   ├── qa/                # QA facade
│   ├── candidates/        # Candidate generation facade
│   └── ui/                # Gradio interactive demo
├── tests/                 # Unit tests (70 cases)
├── app.py                 # Gradio interactive demo entry point
├── main.py                # CLI pipeline entry point
├── README.md
└── README.zh-CN.md
```

## Metadata Schema

Each processed segment uses a shared schema:

| Field | Meaning |
| --- | --- |
| `meeting_id` | Stable meeting identifier |
| `segment_id` | Stable segment identifier |
| `evidence_id` | Unique evidence record ID (usually mirrors segment_id) |
| `speaker` | Speaker label or uncertain speaker hypothesis |
| `start_time` | Evidence start time in seconds |
| `end_time` | Evidence end time in seconds |
| `text` | Current transcript |
| `processing_path` | `low_overlap_cluster` or `high_overlap_candidate` |
| `route_reason` | Human-readable routing decision explanation |
| `overlap_score` | Estimated overlap likelihood [0, 1] |
| `asr_confidence` | ASR confidence estimate [0, 1] |
| `speaker_confidence` | Speaker-attribution confidence [0, 1] |
| `audio_clip_path` | Path to exported audio clip file |
| `source_audio_path` | Original input audio path |
| `language` | Language code (default `"und"`) |
| `candidates` | Alternative transcript/speaker interpretations |
| `uncertainty_note` | Human-readable reason for uncertainty |

## Episodic Memory Design

An episode represents a meaningful meeting event or coherent segment group. It stores meeting and episode IDs, timestamp range, speakers, topic, original and corrected transcripts, overlap and confidence information, candidates, decisions, action items, evidence text, and a future embedding vector.

Episodes support:

- evidence-backed meeting QA;
- historical and cross-meeting recall;
- action-item and decision retrieval;
- speaker-specific search;
- traceability from an answer to exact timestamps.

## Planned Experiments

| Experiment | Goal | Status |
| --- | --- | --- |
| 1. Overlap routing | Compare predicted overlap routes with manual labels | Infrastructure ready; pyannote adapter and energy fallback implemented; annotation set pending |
| 2. High-overlap candidates | Compare candidate generation with forced single-output transcription | Candidate interface implemented; formal experiment pending |
| 3. Metadata-aware LLM | Compare plain-text, speaker-aware, and full-metadata LLM post-processing | LLM event extraction implemented; metadata-input ablation pending |
| 4. Episodic Memory QA | Compare summary QA, transcript RAG, and speaker-aware memory QA | Storage, retrieval, and baseline QA implemented; formal experiment pending |
| 5. Hallucination and evidence | Measure hallucination rate and timestamped evidence hit rate | Metric interfaces defined (stub); formal experiment pending |

Full details are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Current Status

The project is currently in the **baseline infrastructure stage**. Formal experiment results have not been produced yet.

Completed:

- bilingual research design, architecture, innovation points, and experiment plan;
- shared evidence-packet metadata schema (17 fields), validation rules, and sample meeting fixture;
- audio loading, mono conversion, polyphase resampling, peak normalization, and energy-based VAD segmentation (with merging and splitting);
- audio clip export per evidence segment (`src/audio/clipper.py`);
- controlled two-speaker overlap synthesis with SNR control and ground-truth overlap annotations;
- objective WER, CER, overlap-routing classification, and best-mapping speaker-attribution metrics;
- pluggable ASR adapters (Mock/WhisperX/Whisper/Paraformer) with calibrated confidence;
- overlap detection with pyannote OSD adapter (priority) and conservative energy fallback;
- dual-path routing (threshold 0.4), low-overlap ASR + speaker-attribution path, and high-overlap candidate generation without forcing one transcript;
- metadata construction, schema validation, and LLM event extraction;
- event-level episodic memory creation, atomic JSON upsert by meeting, and BM25 + embedding hybrid retrieval;
- top-k-only Gemma QA with validated evidence/timestamp citations and deterministic fallback;
- Gradio interactive UI demo;
- end-to-end pipeline orchestration (`src/pipeline/run_pipeline.py`);
- 75 unit tests covering the implemented baseline infrastructure.

Pending before formal experiments:

- manually annotated evaluation split;
- pyannote model download and calibration experiments;
- heavy-model integrations (Whisper, FunASR) with accuracy comparisons;
- metadata-input ablation experiments and evidence-quality evaluation.

Current verification note: pass the full 75-test suite with `python -m unittest discover -s tests -v`. Heavy models such as faster-whisper, WhisperX, Whisper, pyannote, and speech separation models remain intentionally unloaded without their respective backends.

## How to Run

Install the lightweight baseline dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the end-to-end pipeline:

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
```

Launch the Gradio interactive demo:

```bash
python -m pip install -r requirements-demo.txt
python app.py
```

The demo includes audio upload, an overlap-aware timeline, high-overlap candidate inspection, structured meeting memory, and evidence-cited QA over the current meeting.

Keep large audio files, model weights, and generated outputs outside Git.

---

# 中文完整翻译

## 项目名称

**面向多人会议理解的重叠感知双路径语音处理与情景记忆系统**

## 项目背景

常见的会议助手会将存在错误的转写压缩成流畅的摘要，这可能掩盖说话人归属错误、重叠语音和缺乏证据支持的结论。本项目不仅是一个会议摘要系统，更是一个**可验证的会议记忆系统**。系统会保留不确定性，并将后续回答、决策和行动项关联到带时间戳的原始证据。

## 与参考论文的区别

参考论文已经结合了 ASR、说话人日志、低重叠说话人聚类、高重叠语音分离、LLM 纠错和结构化会议摘要。本项目将在这一基础上进行扩展，而不是简单复现参考系统。

## 系统流程

1. 预处理音频并创建带时间戳的片段。
2. 估计每个片段的重叠分数（优先 pyannote OSD，不可用时使用能量 fallback）。
3. 对每个片段进行路由（阈值 0.4）：低重叠路径或高重叠候选路径。
4. 为每个片段构建统一的元信息记录（17 字段）。
5. 导出每段音频 clip，schema 验证，LLM 事件提取。
6. 相关片段转换为 Episodic Memory 记录并持久化。
7. 检索 episode 回答问题。

## 元信息 Schema（17 字段）

| 字段 | 含义 |
| --- | --- |
| `meeting_id` | 稳定会议标识 |
| `segment_id` | 稳定片段标识 |
| `evidence_id` | 证据记录唯一 ID |
| `speaker` | 说话人标签或假设 |
| `start_time`, `end_time` | 以秒为单位的证据时间范围 |
| `text` | 当前转写文本 |
| `processing_path` | `low_overlap_cluster` 或 `high_overlap_candidate` |
| `route_reason` | 路由决策说明 |
| `overlap_score` | 估计的重叠概率 [0, 1] |
| `asr_confidence` | ASR 置信度 [0, 1] |
| `speaker_confidence` | 说话人归属置信度 [0, 1] |
| `audio_clip_path` | 导出音频 clip 路径 |
| `source_audio_path` | 原始输入音频路径 |
| `language` | 语言代码 |
| `candidates` | 备选转写和说话人解释 |
| `uncertainty_note` | 对不确定原因的可读说明 |

## 当前进度

项目处于**基础设施与基线准备阶段**，已有可运行的端到端 pipeline（`src/pipeline/run_pipeline.py`）。已完成 70 项单元测试。

已完成：
- 完整的 pipeline 编排、音频预处理、VAD、重叠检测（pyannote + 能量 fallback）、双路径路由、ASR 适配器、候选生成、元数据构建、schema 验证、LLM 事件提取、情景记忆存储与检索、Gradio 交互演示。

正式实验前仍需：人工标注评估集构建、pyannote 模型校准、重模型（Whisper/FunASR）集成、元信息消融实验。

## 运行方式

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
python app.py
```

独立中文版详见 [README.zh-CN.md](README.zh-CN.md)。大型音频文件、模型权重和生成结果不应提交到 Git。
