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

`audio preprocessing -> overlap-aware dual paths -> Evidence -> structured events -> Episodic Memory -> traceable retrieval and QA`

The key change is not merely adding an LLM. High-overlap speech is not forced into one confident transcript: candidates, confidence, and uncertainty remain available to structured events, memory, and final answers. LLMs are optional, replaceable components for event extraction and answer wording; the core pipeline remains runnable and testable without them.

## Core Innovations

1. **Uncertainty propagation for overlapped speech:** low-overlap segments produce stable transcripts, while high-overlap segments retain multiple transcript and speaker candidates with confidence. Later stages must not silently turn uncertain candidates into confirmed facts.
2. **Traceable event-level meeting memory and RAG:** episodes bind meeting events to speakers, timestamps, confidence, evidence IDs, and playable audio clips. Retrieval and QA can trace claims back to exact evidence.
3. **Trust-oriented evaluation:** evaluate routing, candidate usefulness, uncertainty preservation, evidence hits, content support, and unsupported claims in addition to WER and DER.

LLM integration is not itself an innovation. It is an optional experimental variable for comparing deterministic rules, plain LLM extraction, and evidence-constrained LLM extraction.

## System Pipeline

1. Preprocess audio and create timestamped VAD segments.
2. Estimate overlap scores and route segments to low- or high-overlap processing.
3. Produce one stable ASR result for low-overlap segments and multiple confidence-bearing candidates for high-overlap segments.
4. Build and validate the shared Evidence representation (17 required + 1 optional field).
5. Extract structured events with deterministic rules or an optional evidence-constrained LLM.
6. Convert events into persistent, traceable Episodic Memory.
7. Retrieve episodes with BM25 + embeddings, then answer with templates or an optional LLM. Every supported answer must cite real evidence IDs and timestamps.

See [docs/system_architecture.md](docs/system_architecture.md) for the module-level design.

## Repository Structure

```text
.
├── docs/                  # Bilingual research design and experiment plans
├── data/
│   ├── raw_audio/         # Raw meeting audio files
│   └── processed_audio/   # Processed/derived audio
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
│   ├── fallbacks/         # Deterministic lightweight fallback backends
│   └── ui/                # Gradio interactive demo
├── tests/                 # Unit and integration tests
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
| `cluster_similarity_distribution` | *Optional.* Relative, uncalibrated `{speaker: similarity}` distribution from the embedding-clustering fallback (defaults to `{}`) |

## Episodic Memory Design

An episode represents a meaningful meeting event or coherent segment group. Episodes inherit event IDs and topics from extracted meeting events, while uncovered segments fall back to time-gap grouping. Each episode preserves supporting evidence IDs, evidence segments, timestamps, speakers, confidence, overlap score, importance, and uncertainty.

Episodes support:

- evidence-backed meeting QA;
- historical and cross-meeting recall;
- action-item and decision retrieval;
- semantic retrieval when `sentence-transformers` is available, with CJK-aware lexical fallback;
- meeting-, speaker-, and time-filtered search;
- relevance-gated ranking with importance, recency, and overlap-aware adjustments;
- traceability from an answer to exact timestamps.

## Planned Experiments

| Experiment | Goal | Status |
| --- | --- | --- |
| 1. Overlap routing | Compare predicted overlap routes with manual labels | Infrastructure ready; pyannote adapter and energy fallback implemented; annotation set pending |
| 2. High-overlap candidates | Compare candidate generation with forced single-output transcription | Candidate interface and metrics implemented; real high-overlap runs and separation pending |
| 3. Structured event extraction | Compare rules, plain LLMs, and Evidence-constrained LLMs | Rule fallback, LLM interface, and validation implemented; real-model ablation pending |
| 4. Episodic Memory QA | Compare summary QA, transcript RAG, and speaker-aware memory QA | Event-grouped storage, hybrid retrieval, filters, and baseline evidence-backed QA implemented; formal experiment pending |
| 5. Evidence and uncertainty | Measure evidence hits, content support, unsupported claims, and uncertainty preservation | Core metrics and a seed experiment exist; annotated real pipeline outputs are pending |

Full details are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Current Status

The project is currently in the **runnable infrastructure, pending real high-overlap processing and formal experiments** stage. The lightweight pipeline validates software integration, but its Mock ASR and deterministic fallbacks do not demonstrate real meeting quality.

Implemented and runnable:

- bilingual research design, architecture, innovation points, and experiment plan;
- shared evidence-packet metadata schema (17 required + 1 optional field), validation rules, and sample meeting fixture;
- audio loading, mono conversion, polyphase resampling, peak normalization, and energy-based VAD segmentation (with merging and splitting);
- audio clip export per evidence segment (`src/audio/clipper.py`);
- controlled two-speaker overlap synthesis with SNR control and ground-truth overlap annotations;
- objective WER, CER, overlap-routing classification, best-mapping speaker-attribution, citation-rate, and timestamp-citation-rate metrics;
- configurable ASR adapters (auto/WhisperX/faster-whisper/Whisper/FunASR/mock) with calibrated confidence;
- overlap scoring that fuses pyannote OSD, diarization overlap, speaker changes, optional ASR instability, and a conservative energy fallback;
- dual-path routing (threshold 0.4), low-overlap ASR + speaker-attribution path, and high-overlap candidate generation without forcing one transcript;
- optional high-overlap speech separation via replaceable adapters — a dependency-free from-scratch NMF baseline and an optional SpeechBrain SepFormer baseline — with per-source ASR candidates and the existing multi-decode fallback;
- metadata construction, schema validation, evidence-only JSON prompts, LLM output repair/validation, and deterministic evidence-linked event fallback;
- event-grouped episodic memory creation, atomic JSON persistence, semantic/lexical retrieval, relevance gating, and meeting/speaker/time filters;
- baseline evidence-backed QA with evidence IDs, timestamps, confidence, uncertainty, and retrieval metadata;
- a five-area Gradio workflow with selectable ASR/Gemma backends, timeline, candidates, long-term memory, and QA;
- end-to-end pipeline orchestration (`src/pipeline/run_pipeline.py`);
- automated tests covering the pipeline, runtime backends, retrieval filters, memory, QA, and evaluation.

Current run capability:

- the lightweight dependency set can run the complete CLI pipeline from WAV input to per-meeting artifacts, episodic memory, and evidence-backed fallback QA;
- without optional heavy models, `auto` ASR uses Mock ASR, diarization/overlap/event extraction/QA use deterministic fallbacks, and output is suitable for integration validation rather than recognition-quality claims;
- real WhisperX/faster-whisper/Whisper/FunASR, pyannote, sentence-transformers, Ollama Gemma, and Gradio require their optional dependencies, model access, tokens, or services.

Pending before the project can claim full experimental completion:

- manually annotated evaluation split;
- overlap-threshold calibration and routing experiments against human labels;
- real high-overlap processing and an optional speech-separation baseline;
- real heavy-model runs and accuracy comparisons for WhisperX, faster-whisper, Whisper, FunASR, pyannote, and Ollama Gemma;
- formal validation of decision/action-item/deadline extraction;
- an annotated set built from real pipeline outputs for content support, uncertainty preservation, and candidate usefulness;
- rules/plain-LLM/Evidence-constrained-LLM ablations and Summary QA/Transcript RAG/Episodic Memory QA comparisons.

Verified on June 13, 2026 with the lightweight virtual environment: `280` unit/integration tests passed and `1` optional Gradio component test was skipped because Gradio was not installed. The seed evidence-evaluation experiment and a lightweight end-to-end smoke run completed successfully. Heavy models remain optional and are loaded lazily.

## How to Run

Install the lightweight baseline dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the end-to-end pipeline:

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 --asr auto

# Local Gemma through Ollama
python main.py data/raw_audio/meeting_001.wav --gemma-backend ollama --gemma-model gemma3:4b
```

Launch the Gradio interactive demo:

```bash
python app.py
```

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
4. 为每个片段构建统一的元信息记录（17 必填字段 + 1 可选）。
5. 导出每段音频 clip 并完成 schema 验证。
6. 使用规则或可选的证据约束 LLM 提取结构化事件。
7. 将事件转换为 Episodic Memory，并通过模板或可选 LLM 基于检索证据回答问题。

## 元信息 Schema（17 必填字段 + 1 可选）

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
| `cluster_similarity_distribution` | *可选。* 聚类 fallback 的相对相似度分布（未校准信号，默认 `{}`） |

## 当前进度

项目处于**可运行基线与正式实验准备阶段**。轻量依赖环境已经能够从 WAV 输入完整运行到证据片段、会议事件、Episodic Memory 和证据问答，但尚未产生真实重模型与人工标注数据上的正式实验结果。

已完成：
- pipeline 编排、音频预处理、VAD、音频切片、重叠检测（pyannote OSD + 能量 fallback）、双路径路由、ASR 适配器、高重叠候选生成、元数据构建与 schema 验证；
- evidence-only JSON Prompt、LLM 输出修复与校验、确定性事件提取 fallback；
- 按事件分组的 Episodic Memory、原子 JSON 持久化、混合检索与会议/说话人/时间过滤；
- 引用 evidence ID 与时间戳、保留不确定性并拒绝无证据回答的 QA；
- 五区 Gradio 工作流和端到端 CLI Pipeline；
- WER、CER、重叠路由、说话人归属、引用率和时间戳引用率等基础指标。

当前可运行程度：
- 仅安装轻量依赖时可以完整运行 Pipeline，但会使用 Mock ASR 与确定性 fallback，适合验证系统流程，不代表真实识别效果；
- 配置可选依赖、模型、Hugging Face token 和 Ollama 服务后，可切换真实 ASR、pyannote、Gemma 与 Gradio。

正式实验前仍需：人工标注评估集、重叠阈值校准、真实高重叠处理与语音分离、真实重模型对比、正式事件抽取验证，以及计划中的消融和对比实验。

2026 年 6 月 13 日验证：轻量环境下 `280` 个测试通过，`1` 个可选 Gradio 组件测试因未安装 Gradio 跳过；证据评估种子实验和轻量端到端 smoke run 成功。

## 运行方式

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
python app.py
```

独立中文版详见 [README.zh-CN.md](README.zh-CN.md)。大型音频文件、模型权重和生成结果不应提交到 Git。
