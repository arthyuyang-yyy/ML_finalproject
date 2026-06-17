# Project Task

项目名称：Overlap-aware Dual-path ASR with Episodic Memory

中文名称：面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

## 1. 当前项目目标

本项目实现一个面向多人会议音频的最小可运行系统：

1. 输入会议音频。
2. 预处理并切分语音片段。
3. 进行说话人分离和重叠语音检测。
4. 根据 `overlap_score` 分为低重叠和高重叠两条路径。
5. 低重叠片段直接进行 ASR 和说话人归属。
6. 高重叠片段先生成候选转写，再通过 resolver 得到最终文本。
7. 生成统一的 `evidence_segments.json`。
8. 抽取结构化会议事件。
9. 构建 Episodic Memory。
10. 支持基于 evidence 的问答。

当前重点是主流程清楚、可运行、可调试、可演示，不追求复杂插件系统、复杂配置层或过度文档化。

## 2. Pipeline

```text
input audio
-> preprocess_audio
-> segment_waveform
-> diarize_with_pyannote
-> estimate_segment_overlap_scores
-> route_segment
   -> low_overlap_cluster
      -> process_low_overlap_segments
   -> high_overlap_candidate
      -> process_high_overlap_segments
      -> resolve_high_overlap_segments
-> build_evidence_segments
-> extract_meeting_events
-> build_episodes
-> retrieve_episodes
-> answer_question
```

## 3. 当前目录结构

```text
.
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── Project_task.md
├── TODO.md
├── docs/
│   ├── asr_baseline.zh-CN.md
│   ├── future_work_guide.zh-CN.md
│   ├── speech_separation.zh-CN.md
│   └── system_architecture.md
├── data/
│   ├── raw_audio/
│   ├── processed_audio/
│   ├── fixtures/
│   ├── annotations/
│   └── manifests/
├── memory/
│   └── episodic_memory.json
├── outputs/
├── scripts/
├── src/
│   ├── audio/
│   ├── asr/
│   ├── candidates/
│   ├── datasets/
│   ├── diarization/
│   ├── evaluation/
│   ├── evidence/
│   ├── fallbacks/
│   ├── llm/
│   ├── memory/
│   ├── overlap/
│   ├── pipeline/
│   ├── qa/
│   └── ui/
└── tests/
```

多个 requirements 文件已合并为 `requirements.txt`。项目主说明统一维护在 `README.md`、`Project_task.md` 和 `TODO.md`；`docs/` 只保留精简参考说明和可选后端/future work 记录。

## 4. 关键实现约定

### 4.1 音频预处理

文件：

```text
src/audio/preprocess.py
src/audio/clipper.py
```

职责：

- 读取音频；
- 转单声道；
- 重采样到 16 kHz；
- 峰值归一化；
- energy-based VAD；
- 导出每个 evidence segment 的音频 clip。

### 4.2 说话人分离与重叠检测

文件：

```text
src/diarization/
src/overlap/
src/speech_separation.py
```

约定：

```python
if overlap_score < 0.4:
    processing_path = "low_overlap_cluster"
else:
    processing_path = "high_overlap_candidate"
```

pyannote 可用时优先使用 pyannote；不可用时保留轻量 fallback，保证测试和演示流程不会因为重模型缺失而崩溃。Speech separation 当前是高重叠路径的可选增强，默认 `none`，需要时可启用 `mock`、`nmf` 或 `sepformer` adapter 来补充 separated-source candidates。

### 4.3 低重叠路径

文件：

```text
src/low_overlap.py
src/asr/
```

输出必须包含：

- `speaker`
- `text`
- `start_time`
- `end_time`
- `asr_confidence`
- `speaker_confidence`
- `candidates: []`
- `uncertainty_note: ""`

### 4.4 高重叠路径

文件：

```text
src/high_overlap.py
src/candidates/generator.py
src/llm/resolver.py
```

当前规则：

1. `process_high_overlap_segments` 负责生成候选，不负责最终确认文本。
2. `resolve_high_overlap_segments` 负责从候选中解析最终文本。
3. 有 Gemma/Ollama client 时，resolver 调用 LLM。
4. 没有 LLM 时，resolver 选择最高置信候选作为 `fallback_resolved`。
5. LLM 输出无效或置信度越界时，resolver 回退到最高置信候选。
6. 无候选时，resolver 标记为 `unresolved`，不伪造文本。
7. 最终高重叠 evidence 必须继续保留候选列表和不确定性说明。

高重叠 evidence 可包含：

```json
{
  "processing_path": "high_overlap_candidate",
  "text": "resolved transcript",
  "speaker": "SPEAKER_01",
  "candidates": [],
  "source": "llm_resolved",
  "decision_reason": "selected candidate 1 based on local context",
  "uncertainty_note": "High-overlap segment; speaker attribution is uncertain."
}
```

实际输出中 `candidates` 不应为空；上例省略候选内容仅为了展示字段。

### 4.5 Evidence Segment

文件：

```text
src/evidence/
```

核心字段：

| Field | Required | Meaning |
| --- | --- | --- |
| `meeting_id` | yes | Meeting ID |
| `segment_id` | yes | Segment ID |
| `evidence_id` | yes | Evidence ID |
| `speaker` | yes | Speaker label |
| `start_time` | yes | Start timestamp |
| `end_time` | yes | End timestamp |
| `text` | yes | Final transcript text |
| `processing_path` | yes | Low/high path |
| `route_reason` | yes | Routing explanation |
| `overlap_score` | yes | Overlap score |
| `asr_confidence` | yes | ASR/resolver confidence |
| `speaker_confidence` | yes | Speaker confidence |
| `audio_clip_path` | yes | Exported clip path |
| `source_audio_path` | yes | Original audio path |
| `language` | yes | Language code |
| `candidates` | yes | Candidate list |
| `uncertainty_note` | yes | Uncertainty note |
| `source` | optional | `llm_resolved`, `fallback_resolved`, etc. |
| `decision_reason` | optional | Resolver decision explanation |

### 4.6 Episodic Memory

文件：

```text
src/memory/episodic_store.py
src/memory/retriever.py
src/fallbacks/embeddings.py
```

系统生成 per-meeting memory：

```text
outputs/{meeting_id}/episodic_memory.json
```

同时维护长期记忆：

```text
memory/episodic_memory.json
```

检索使用自定义 BLAKE2 字符 n-gram hash embedding，默认分数为：

```text
0.70 * embedding_similarity + 0.30 * keyword_score
```

该 embedding 是当前项目保留的轻量特色能力，不依赖外部 embedding 模型。

检索当前刻意保持为 MVP 版本：不使用 transformer embedding，不加入 recency decay、importance prior 或复杂 reranker。这样可以保证结果稳定、依赖轻、测试可重复；代价是排序不会捕捉深层语义相似度、会议时序偏好或事件重要性偏好。后续只有在有标注评估集后再扩展这些信号。

### 4.7 QA

文件：

```text
src/qa/answerer.py
```

QA 只能基于检索到的 episodes 回答，不应直接让 LLM 自由阅读全部 memory。回答应带有：

- evidence IDs；
- timestamps；
- speakers；
- confidence；
- uncertainty note。

## 5. 输出文件

一次 pipeline 运行生成：

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

## 6. 运行方式

安装：

```bash
python -m pip install -r requirements.txt
```

测试：

```bash
python -m pytest -q
python -m ruff check src tests main.py app.py
```

CLI：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 --asr mock
```

Gradio：

```bash
python app.py
```

## 7. 当前验收状态

当前已通过：

```text
488 passed, 6 skipped, 2 warnings, 7 subtests passed
ruff: All checks passed
```

## 8. 后续优先级

优先继续做：

1. 准备真实会议音频样例。
2. 建立小规模人工标注集。
3. 验证 pyannote、faster-whisper、WhisperX 在真实音频上的效果。
4. 评估高重叠 resolver 是否优于直接选择单一 ASR 输出。
5. 保持 README、Project_task、TODO 和精简 docs 与代码同步。

暂不优先做：

- 新增大量重复文档；
- 新增复杂配置系统；
- 新增未使用的插件式抽象；
- 在没有评估数据前继续扩展复杂指标。
