# TODO / 项目任务进度

This file tracks implementation progress. The authoritative requirements,
acceptance criteria, schemas, and final deliverables are defined in
[`Project_task.md`](Project_task.md).

本文档同时提供中文说明。状态以源码、测试和 `Project_task.md` 的验收目标为准：

- `[x]` 已实现并有基础测试 / implemented with baseline tests
- `[ ]` 尚未完成或仍需正式实验验证 / pending or awaiting formal validation

## Confirmed Research Route / 已确认研究路线

The core project must remain runnable without an LLM:

`preprocessing -> overlap detection and routing -> low/high-overlap processing -> Evidence -> structured events -> Episodic Memory -> retrieval and QA`

项目主线必须在不接入 LLM 时也能够运行：

`预处理 -> 重叠检测与分流 -> 低/高重叠处理 -> Evidence -> 结构化事件 -> Episodic Memory -> 检索与问答`

Core contributions / 核心贡献：

- propagate uncertainty from high-overlap candidates into events, memory, and answers;
- store traceable event-level memory linked to evidence IDs, timestamps, and audio clips;
- evaluate evidence support, unsupported claims, and uncertainty preservation beyond WER/DER.

LLMs are optional experimental components for structured event extraction and
answer wording. RAG retrieval, evidence validation, memory storage, and the
uncertainty rules must not depend on an external LLM.

LLM 是结构化事件抽取和答案组织的可选实验组件。RAG 检索、证据校验、记忆存储和不确定性
规则不能依赖外部 LLM。

## Phase 1: Interfaces and Data / 阶段一：接口与数据

- [x] Define bilingual research positioning and system architecture. / 完成双语研究定位和系统架构。
- [x] Define segment metadata and annotation schemas. / 完成片段元数据和标注 Schema。
- [x] Create module interfaces without loading heavy models. / 完成不依赖重模型加载的模块接口。
- [x] Add schema validation and small fixture datasets. / 完成 Schema 校验和小型 fixture 数据。

## Phase 2: Audio and Dual-path Baselines / 阶段二：音频与双路径基线

- [x] Step 1 - Audio preprocessing: soundfile/PyAV demux and decode for common
  meeting formats, optional denoising, mono conversion, one-time 16 kHz
  resampling, normalization, and standard WAV export.
  / 常见会议音频格式解封装与解码、可选降噪、单声道转换、单次 16 kHz 重采样、归一化和 WAV 导出。
- [x] Step 2 - Energy-based VAD with timestamped segments. / 基于能量的 VAD 与时间戳切段。
- [x] Step 3 - Per-segment audio clip export with `audio_clip_path`. / 按片段导出音频 clip。
- [x] Step 4 - Pluggable ASR adapters and confidence normalization:
  Mock, WhisperX, faster-whisper, Whisper, and FunASR. / 可插拔 ASR 与置信度规范化。
  `faster-whisper small` is the selected first real baseline and has passed a
  real-audio smoke run; other adapters remain available for later comparisons.
  / 第一版真实 ASR 已选择 `faster-whisper small` 并通过真实音频 smoke run，其他适配器保留用于后续对比。
- [x] Step 5 - Optional pyannote diarization adapter. / 可选 pyannote 说话人日志接口。
- [x] Step 6 - Integrate diarization and speaker assignment into the pipeline,
  including dominant speaker, `MIXED`, and `UNKNOWN` rules. / 集成说话人归属规则。
- [x] Step 7a - Baseline overlap detection with pyannote OSD, fused signals,
  and a conservative energy fallback. / 完成基础重叠检测和保守 fallback。
- [ ] Step 7b - Calibrate the overlap threshold against human labels and report
  routing metrics and cost/quality trade-offs. / 使用人工标注校准阈值并报告实验结果。
- [x] Step 8 - Configurable dual-path router with default threshold `0.4`. / 可配置双路径路由。
- [x] Step 9 - Low-overlap path producing speaker, transcript, timestamps, and
  confidence values. / 完成低重叠稳定转写路径。
- [x] Step 10 - High-overlap path preserving multiple candidates and
  uncertainty instead of forcing one transcript. / 完成高重叠多候选路径。
- [x] Step 11 - Optional speech separation baseline for high-overlap segments:
  dependency-free single-channel NMF separator (numpy) with a soft Wiener mask,
  an optional learned Conv-TasNet backend, and an opt-in
  `process_high_overlap_segments(..., separate=True)` hook that generates
  candidates per separated stream. / 完成可选语音分离基线(numpy NMF + 软掩码,
  可选 Conv-TasNet 后端),并在高重叠路径提供 `separate=True` 开关按分离流生成候选。
- [x] Step 12 - Evidence-segment schema builder and validator. / 完成统一 Evidence Schema 构建与校验。
- [x] Validate that every emitted `audio_clip_path` exists on disk. / 在校验器中检查每个音频路径真实存在。

## Phase 3: Evidence, Memory, and QA / 阶段三：证据、记忆与问答

- [x] Step 13 - Implement deterministic structured-event fallback so the core
  pipeline does not require an LLM. / 完成确定性结构化事件 fallback，确保主线不依赖 LLM。
- [x] Step 14 - Implement optional evidence-only, JSON-only LLM prompts, evidence citations,
  owner uncertainty, small-talk exclusion, and high-overlap confidence rules. / 完成证据约束 Prompt 和不确定性规则。
- [x] Step 15 - Add LLM JSON parse, repair, regeneration, and evidence-ID
  validation. / 完成 LLM JSON 修复、重试和证据 ID 校验。
- [ ] Step 16a - Define and evaluate a deterministic/rule-based structured-event
  baseline for decisions, action items, deadlines, open questions, and uncertainty.
  / 定义并评估规则式结构化事件抽取基线。
- [ ] Step 16b - Compare rule extraction, plain LLM extraction, and full-Evidence
  constrained LLM extraction on the same annotated meetings.
  / 在同一标注集上比较规则、普通 LLM 和完整 Evidence 约束 LLM。
- [x] Step 17 - Create event-grouped Episodic Memory and atomically persist it
  as JSON. / 完成事件级情景记忆与原子 JSON 持久化。
- [x] Step 18 - Add keyword and semantic/hybrid retrieval with relevance gating
  and meeting, speaker, and time filters. / 完成关键词与混合检索。
- [x] Step 19 - Implement evidence-backed QA that cites evidence IDs and
  timestamps, refuses unsupported answers, and surfaces uncertainty. / 完成可追溯证据问答。
- [ ] Step 20 - Compare template QA, transcript RAG, and Episodic Memory RAG;
  use an LLM only as an optional answer-writing condition.
  / 比较模板问答、Transcript RAG 和 Episodic Memory RAG；LLM 仅作为可选答案生成条件。
- [x] Emit unified per-meeting pipeline artifacts under `outputs/<meeting_id>/`.
  / 按会议输出统一 Pipeline 产物。

## Gradio Demo / Gradio 演示

- [x] Page 1 - Audio upload and Run Pipeline workflow. / 音频上传与运行 Pipeline。
- [x] Page 2 - Timeline with speaker, route, overlap score, transcript, and uncertainty. / 证据时间线。
- [x] Page 3 - High-overlap candidate drill-down with audio playback. / 高重叠候选与音频播放。
- [x] Page 4 - Structured meeting-memory view. / 结构化会议记忆。
- [x] Page 5 - Evidence-cited QA with timestamp traceability. / 带证据和时间戳的问答。
- [ ] Run and verify the full UI with the optional Gradio dependency installed.
  / 安装可选 Gradio 依赖后运行并验证完整 UI。

## Phase 4: Evaluation / 阶段四：实验评估

- [ ] Build the manually annotated evaluation split. / 构建人工标注评估集。
- [ ] Experiment 1 - Sweep overlap-routing thresholds and report accuracy,
  precision, recall, F1, and cost/quality trade-offs. / 完成重叠路由阈值实验。
- [ ] Experiment 2 - Compare multi-candidate high-overlap processing with a
  forced single transcript. / 完成高重叠多候选对比实验。
- [ ] Experiment 3 - Compare deterministic rules, plain LLM extraction, and
  full-Evidence constrained LLM extraction. / 比较规则、普通 LLM 和完整 Evidence 约束 LLM。
- [ ] Experiment 4 - Compare Episodic Memory QA with summary QA and transcript
  RAG. / 完成 Episodic Memory QA 对比实验。
- [x] Implement evidence-ID support, uncertainty-preservation, candidate-usefulness,
  and seed experiment infrastructure. / 完成证据 ID、不确定性保留、候选有效性和种子实验基础设施。
- [ ] Build annotations from real pipeline outputs and run content-support,
  unsupported-claim, uncertainty-preservation, and candidate-usefulness evaluation.
  / 基于真实 Pipeline 输出构建标注并完成内容支持、无支持声明、不确定性和候选有效性实验。

## Shared Infrastructure / 共享基础设施

- [x] Add controlled two-speaker overlap synthesis with SNR control and
  ground-truth labels. / 完成双说话人可控重叠合成。
- [x] Implement WER, CER, overlap-routing, speaker-attribution, citation-rate,
  and timestamp-citation-rate metrics. / 完成基础客观指标。
- [x] Add pipeline orchestration, configuration, I/O helpers, and package
  facades. / 完成 Pipeline 编排、配置和 I/O。
- [x] Add deterministic event-extraction and QA fallbacks. / 完成确定性事件抽取与问答 fallback。
- [x] Validate emitted audio clip paths and add seed evidence-evaluation experiments.
  / 完成音频片段路径校验与种子证据评估实验。
- [x] Verify the environment on June 14, 2026: 282 tests passed,
  1 optional Gradio test skipped, seed evidence evaluation passed, and the
  lightweight pipeline and real faster-whisper ASR smoke runs passed.
  / 环境验证完成：282 个测试通过，1 个可选 Gradio 测试跳过，证据种子实验、
  轻量 Pipeline 与真实 faster-whisper ASR smoke run 均通过。

## Project Goal / 项目最终要实现什么

Build a verifiable multi-speaker meeting-understanding system that detects
overlapped speech, preserves uncertainty, extracts structured meeting events,
stores Episodic Memory, and answers questions with evidence IDs, timestamps,
and traceable audio clips.

构建一个可验证的多人会议理解系统：识别重叠语音、保留不确定性、提取结构化会议事件、形成
Episodic Memory，并使用 evidence ID、时间戳和原始音频片段进行可追溯问答。

## Current Capability / 当前已经实现什么

The repository now contains a complete lightweight end-to-end pipeline:
audio preprocessing -> VAD -> overlap scoring -> dual-path processing ->
evidence schema -> event extraction/fallback -> Episodic Memory -> retrieval ->
evidence-backed QA, plus a five-area Gradio workflow.

仓库已经具备完整轻量端到端流程：音频预处理、VAD、重叠评分、双路径处理、统一证据、
事件抽取/fallback、Episodic Memory、混合检索、证据问答，以及五区 Gradio 工作流。

## Can It Run Completely? / 能否完整运行

Yes, the lightweight baseline can run completely with `requirements.txt`.
Without optional heavy models it uses Mock ASR and deterministic fallbacks, so
it validates the full software workflow but does not demonstrate production
recognition quality. Full experimental completion still requires real models,
manual annotations, speech separation, metric completion, and all planned
experiments.

可以。安装 `requirements.txt` 后，轻量基线能够完整运行；未安装重模型时会使用 Mock ASR
和确定性 fallback，因此可以验证完整软件流程，但不能代表真实识别效果。项目要达到正式完成，
仍需真实模型、人工标注、语音分离、完整评估指标和全部计划实验。
