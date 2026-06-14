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
       -> pyannote OSD（如有 HF_TOKEN；配置后失败将明确报错）
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
  -> build_episodes（事件级 Episode，高重叠强制 uncertainty）
  -> upsert_episodes（按 meeting ID 原子更新 JSON）
  -> write_json（按会议写出多份 JSON artifact）
```

完整 14 步调用链见 [pipeline_walkthrough.md](pipeline_walkthrough.md)。

## 模块职责

### 核心 Pipeline

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 预处理 | `src/audio/preprocess.py` | 使用 soundfile/PyAV 解封装与解码常见格式、可选降噪、单声道转换、单次 polyphase 重采样、峰值归一化、VAD 分段、导出 float32 WAV |
| Clip 导出 | `src/audio/clipper.py` | 将每个证据片段导出为 WAV clip |
| 重叠检测 | `src/overlap/detector.py` | 评分重叠：pyannote OSD 适配器（优先）、显式区域覆盖或能量 fallback（上限 0.39） |
| 双路径路由 | `src/overlap/router.py` | 按重叠阈值（默认 0.4）路由片段 |
| 低重叠路径 | `src/low_overlap.py` | 为低重叠片段产出稳定文本、speaker、时间戳、ASR 置信度和说话人置信度 |
| ASR | `src/asr/core.py` | 可插拔适配器（Mock/WhisperX/Whisper/Paraformer），带校准置信度；低重叠重模型优先推荐 WhisperX |
| 说话人日志 | `src/diarization/core.py` | 配置后使用 pyannote 说话人 turns；否则使用确定性 fallback |
| 语音分离 | `src/speech_separation.py` | 兼容接口与占位实现，等待模型集成 |
| 高重叠路径 | `src/high_overlap.py` | 保留 mixed-speaker 记录，主转写为空，并保存多个候选 |
| 候选生成 | `src/candidates/generator.py` | 使用 faster-whisper beam/temperature/language 变化生成多个转写/说话人假设；轻量运行时使用 fallback 候选 |
| Evidence 构建 | `src/evidence/builder.py` | 合并低/高重叠结果，规范化候选，按时间排序并输出共享的 17 字段证据 schema |
| Schema 验证 | `src/evidence/validator.py` | 验证证据记录、候选结构和 meeting 列表 |
### 回退后端（确定性轻量级后端）

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| ASR 回退 | `src/fallbacks/asr.py` | ASR 后端自动选择 |
| 候选回退 | `src/fallbacks/candidates.py` | 保留不确定性的候选生成 |
| 说话人日志回退 | `src/fallbacks/diarization.py` | 确定性说话人聚类 |
| 重叠回退 | `src/fallbacks/overlap.py` | 基于能量的重叠估计 |
| 事件回退 | `src/fallbacks/events.py` | 确定性会议事件提取 |
| QA 回退 | `src/fallbacks/qa.py` | 基于证据引用的确定性 QA |
| 嵌入回退 | `src/fallbacks/embeddings.py` | 基于哈希的字符 n-gram 嵌入 |

### Pipeline 编排

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 运行 pipeline | `src/pipeline/run_pipeline.py` | `run_meeting_pipeline()` — 端到端编排 |
| 配置 | `src/pipeline/config.py` | `PipelineConfig` 不可变 dataclass |
| I/O | `src/pipeline/io.py` | 目录创建、JSON 读写 |

### LLM 子系统

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 事件提取 | `src/llm/event_extractor.py` | 从证据片段生成结构化会议文档；支持 JSON 修复、失败重试、无效事件删除和确定性 fallback |
| 事件验证 | `src/llm/event_validator.py` | 验证 event 类型、真实 evidence_id、speaker/owner 来源、action item 字段与高重叠置信度约束 |
| Gemma 客户端 | `src/llm/gemma_client.py` | 可注入本地或远程 Gemma JSON 生成函数 |
| Prompt 构建 | `src/llm/prompts.py` | 强制 JSON Schema、证据引用和禁止编造的事件抽取/修复 prompt |

### 记忆与问答

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 情景记忆 | `src/memory/episodic_store.py` | 将结构化事件转为可追溯 Episode，强制高重叠不确定性，并按会议原子更新长期 JSON Memory |
| 混合检索 | `src/memory/retriever.py` | 使用 BM25、embedding、重要度、时效性和重叠惩罚排序 Episode |
| RAG 问答 | `src/qa/answerer.py` | Gemma 只读取 top-k episode；校验引用，输出无效时安全回退 |

### 评估与数据

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 评估 | `src/evaluation/core.py` | WER、CER、重叠路由指标、说话人归属准确率、证据质量（stub） |
| 数据合成 | `src/data_synthesis.py` | 可控双人重叠语音合成，含 SNR 和真值标注 |
| 工具 | `src/utils.py` | `validate_score()` |

### UI

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| Gradio 应用 | `src/ui/gradio_app.py` | 五区交互演示：上传、证据时间线、高重叠候选、会议记忆和当前会议 QA |

### 包 Facade

| 包 | 重新导出 |
| --- | --- |
| `src/overlap/` | `detect_overlap_segments`, `estimate_segment_overlap_scores`, `detect_pyannote_overlap_regions`, `DEFAULT_OVERLAP_THRESHOLD` |
| `src/evidence/` | `build_evidence_segments`, `build_evidence_file`, `build_metadata_segment`, `validate_metadata_segment`, `validate_meeting`, `validate_candidate` |
| `src/llm/` | `extract_meeting_events`, `validate_meeting_event` |
| `src/memory/` | `build_episodes`, `build_episodes_file`, `upsert_episodes`, `read_episodes`, `retrieve_episodes` |
| `src/qa/` | `answer_question`、`validate_qa_answer`、Prompt 构建器和兼容入口 |
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
| 长期情景记忆 | `memory/episodic_memory.json` | 跨会议持久化 Episode；重复处理同一会议时替换旧记录 |
| 音频 clip | `clips/{evidence_id}.wav` | 每段 WAV 导出 |

## 实施状态

| 阶段 | 状态 |
| --- | --- |
| 1. 验证元数据与标注约定 | 已完成 |
| 2. 基础重叠检测、ASR 和说话人日志 | 已完成（pyannote 适配器 + 保守 fallback、低重叠 ASR/speaker 路径、测试默认 mock） |
| 3. 候选生成与不确定性感知 prompt | 已完成（多参数候选接口、fallback 候选、LLM 事件提取） |
| 4. 本地 episode 存储与检索 | 已完成（长期 JSON 原子更新、BM25 + embedding 混合检索） |
| 5. 消融实验与证据质量评估 | 待完成（需标注评估集、重模型集成） |
