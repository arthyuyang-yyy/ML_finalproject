# 系统架构

## 设计目标

系统架构将音频处理、不确定性表示、LLM 推理和记忆检索分离。所有后续结论都应能够追溯到原始音频片段。

## 数据流程（实际实现）

```text
音频
  -> preprocess_audio（归一化为 16kHz 单声道 float32 WAV）
  -> load_audio
  -> segment_waveform（基于能量的 VAD，含段落合并与分割）
  -> estimate_segment_overlap_scores
       -> pyannote OSD（如有 HF_TOKEN）
       -> 显式重叠区域
       -> 能量 fallback（保守，上限 0.39）
  -> route_segment（阈值 0.4）
       -> low_overlap_cluster
          -> process_low_overlap_segments
             -> WhisperX/Whisper/Paraformer/Mock ASR
             -> pyannote/WhisperX 说话人日志或确定性 fallback
       -> high_overlap_candidate
          -> process_high_overlap_segments
             -> faster-whisper 多参数解码候选或显式 fallback 候选
  -> build_metadata_segment（17 字段证据记录）
  -> write_segment_clips（导出每段 WAV）
  -> validate_metadata_segment
  -> extract_meeting_events（LLM 或 fallback）
  -> create_episode_from_segments
  -> store_episode（JSONL）
  -> write_json（按会议写出多份 JSON artifact）
```

完整 14 步调用链见 [pipeline_walkthrough.md](pipeline_walkthrough.md)。

## 模块职责

### 核心 Pipeline

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 预处理 | `src/audio/preprocess.py` | 加载、单声道转换、polyphase 重采样、峰值归一化、VAD 分段、导出 float32 WAV |
| Clip 导出 | `src/audio/clipper.py` | 将每个证据片段导出为 WAV clip |
| 重叠检测 | `src/overlap_detector.py` | 评分重叠：pyannote OSD 适配器（优先）、显式区域覆盖或能量 fallback（上限 0.39） |
| 双路径路由 | `src/dual_path_router.py` | 按重叠阈值（默认 0.4）路由片段 |
| 低重叠路径 | `src/low_overlap.py` | 为低重叠片段产出稳定文本、speaker、时间戳、ASR 置信度和说话人置信度 |
| ASR | `src/asr.py` | 可插拔适配器（Mock/WhisperX/Whisper/Paraformer），带校准置信度；低重叠重模型优先推荐 WhisperX |
| 说话人日志 | `src/diarization.py` | 配置后使用 pyannote 说话人 turns；否则使用确定性 fallback |
| 语音分离 | `src/speech_separation.py` | 分离接口（stub — 待模型集成） |
| 高重叠路径 | `src/high_overlap.py` | 保留 mixed-speaker 记录，主转写为空，并保存多个候选 |
| 候选生成 | `src/candidate_generator.py` | 使用 faster-whisper beam/temperature/language 变化生成多个转写/说话人假设；轻量运行时使用 fallback 候选 |
| 元数据构建 | `src/metadata_builder.py` | 将输出统一为共享的 17 字段证据 schema |
| Schema 验证 | `src/schema_validation.py` | 验证证据记录、候选结构和 meeting 列表 |
| LLM 后处理 | `src/llm_postprocess.py` | 构建元信息感知约束 prompt；不确定性纠错接口（stub） |

### Pipeline 编排

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 运行 pipeline | `src/pipeline/run_pipeline.py` | `run_meeting_pipeline()` — 端到端编排 |
| 配置 | `src/pipeline/config.py` | `PipelineConfig` 不可变 dataclass |
| I/O | `src/pipeline/io.py` | 目录创建、JSON 读写 |

### LLM 子系统

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 事件提取 | `src/llm/event_extractor.py` | 从证据片段提取会议事件（LLM 或确定性 fallback） |
| 事件验证 | `src/llm/event_validator.py` | 验证 LLM 提取事件，强制 evidence_id 引用 |
| Gemma 客户端 | `src/llm/gemma_client.py` | 可插拔 Gemma JSON 生成接口 |
| Prompt 构建 | `src/llm/prompts.py` | 构建证据感知的事件提取 prompt |

### 记忆与问答

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 情景记忆 | `src/episodic_memory.py` | 从片段创建 episode、JSONL 持久化、关键词检索 |
| RAG 问答 | `src/rag_qa.py` | 检索相关 episode 并使用证据回答 |

### 评估与数据

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 评估 | `src/evaluation.py` | WER、CER、重叠路由指标、说话人归属准确率、证据质量（stub） |
| 数据合成 | `src/data_synthesis.py` | 可控双人重叠语音合成，含 SNR 和真值标注 |
| 工具 | `src/utils.py` | `validate_score()` |

### UI

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| Gradio 应用 | `src/ui/gradio_app.py` | 通过 Gradio 运行交互式 pipeline 演示 |

### 包 Facade

| 包 | 重新导出 |
| --- | --- |
| `src/overlap/` | `detect_overlap_segments`, `estimate_segment_overlap_scores`, `detect_pyannote_overlap_regions`, `DEFAULT_OVERLAP_THRESHOLD` |
| `src/evidence/` | `build_metadata_segment`, `validate_metadata_segment`, `validate_meeting`, `validate_candidate` |
| `src/llm/` | `extract_meeting_events`, `validate_meeting_event` |
| `src/memory/` | `create_episode_from_segments`, `store_episode`, `search_episodes` |
| `src/qa/` | `answer_question_with_evidence`, `retrieve_relevant_memory` |
| `src/candidates/` | `generate_high_overlap_candidates` |

## 关键约定

- 分数范围统一为 `[0.0, 1.0]`。
- 时间为从会议音频开始计算的秒数。
- 高重叠记录必须保留候选列表和不确定性说明。
- 高重叠主记录使用 `speaker="MIXED"` 且 `text=""`；转写内容保存在 `candidates` 中，而不是强制确定为一个答案。
- 能量 fallback 重叠分数上限为 0.39（低于默认路由阈值 0.4，不会误触高重叠路由）。
- 低重叠记录是单一假设证据记录：稳定 `text`、`speaker`、时间戳、`asr_confidence`、`speaker_confidence`，`candidates` 为空，`uncertainty_note` 为空。
- 决策与行动项必须携带带时间戳的证据。
- 早期实验中的存储与检索后端应可替换。

## IO Artifact 路径

每场会议的产物写入 `outputs/{meeting_id}/`：

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 预处理音频 | `preprocessed.wav` | 16kHz 单声道 float32 WAV |
| VAD 片段 | `vad_segments.json` | 带时间戳的语音区域 |
| 重叠分数 | `overlap.json` | 附加重叠分数的 VAD 片段 |
| 低重叠片段 | `low_overlap_segments.json` | 路由到低重叠路径的证据记录 |
| 高重叠候选 | `high_overlap_candidates.json` | 路由到高重叠路径的证据记录 |
| 证据片段 | `evidence_segments.json` | 所有验证通过的证据记录 |
| 会议事件 | `meeting_events.json` | LLM 提取的会议事件 |
| 情景记忆 | `episodic_memory.json` | Episode 记录 |
| 音频 clip | `clips/{evidence_id}.wav` | 每段 WAV 导出 |

## 实施状态

| 阶段 | 状态 |
| --- | --- |
| 1. 验证元数据与标注约定 | 已完成 |
| 2. 基础重叠检测、ASR 和说话人日志 | 已完成（pyannote 适配器 + 保守 fallback、低重叠 ASR/speaker 路径、测试默认 mock） |
| 3. 候选生成与不确定性感知 prompt | 已完成（多参数候选接口、fallback 候选、LLM 事件提取） |
| 4. 本地 episode 存储与检索 | 已完成（JSONL 持久化、关键词检索） |
| 5. 消融实验与证据质量评估 | 待完成（需标注评估集、重模型集成） |
