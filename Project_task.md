
# 一、项目最终目标

项目名称可以定为：

Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding

中文：

面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

最终要做的不是普通会议总结器，而是一个：

可识别重叠语音  
可保留不确定性  
可生成结构化会议事件  
可形成 Episodic Memory  
可基于证据回答问题  
可追溯到时间戳和原始音频片段的会议理解系统

---

# 二、核心开发目标

## 目标 1：输入一段中英文混合会议音频

系统支持用户上传一段会议音频，例如：

```text
data/raw/meeting_001.wav
```

音频可能包含：

多人说话  
中英文混合  
打断  
抢话  
高重叠语音  
普通低重叠语音

---

## 目标 2：将音频切成带时间戳的片段

输出：

```json
[
  {
    "segment_id": "m1_seg_001",
    "start_time": 0.0,
    "end_time": 6.4
  },
  {
    "segment_id": "m1_seg_002",
    "start_time": 6.4,
    "end_time": 12.8
  }
]
```

这个阶段只负责切段，不负责理解内容。

---

## 目标 3：给每个片段估计 overlap_score

每个片段都要得到一个重叠分数：

```json
{
  "segment_id": "m1_seg_003",
  "overlap_score": 0.78
}
```

含义：

```text
0.0 = 几乎没有重叠
1.0 = 严重多人同时说话
```

然后根据阈值路由：

```python
if overlap_score < 0.4:
    processing_path = "low_overlap_cluster"
else:
    processing_path = "high_overlap_candidate"
```

---

## 目标 4：低重叠片段走稳定 ASR + 说话人路径

低重叠片段要输出：

说话人  
文本  
时间戳  
ASR 置信度  
说话人置信度

示例：

```json
{
  "meeting_id": "meeting_001",
  "segment_id": "m1_seg_004",
  "speaker": "SPEAKER_00",
  "start_time": 12.4,
  "end_time": 18.2,
  "text": "I think we should first build the WhisperX baseline.",
  "processing_path": "low_overlap_cluster",
  "overlap_score": 0.12,
  "asr_confidence": 0.91,
  "speaker_confidence": 0.88,
  "candidates": [],
  "uncertainty_note": ""
}
```

---

## 目标 5：高重叠片段不强行确定，而是生成候选

高重叠片段不能强行输出一个确定结果。

应该输出多个候选：

```json
{
  "meeting_id": "meeting_001",
  "segment_id": "m1_seg_009",
  "speaker": "MIXED",
  "start_time": 42.3,
  "end_time": 48.6,
  "text": "",
  "processing_path": "high_overlap_candidate",
  "overlap_score": 0.78,
  "asr_confidence": 0.54,
  "speaker_confidence": 0.35,
  "candidates": [
    {
      "candidate_id": "m1_seg_009_c1",
      "speaker": "SPEAKER_00",
      "text": "We should test WhisperX first.",
      "confidence": 0.62
    },
    {
      "candidate_id": "m1_seg_009_c2",
      "speaker": "SPEAKER_01",
      "text": "但是重点是 overlap detection.",
      "confidence": 0.58
    }
  ],
  "uncertainty_note": "High-overlap segment; speaker attribution is uncertain."
}
```

这一点是项目的核心创新之一：

高重叠语音不被伪装成确定 transcript，而是以 uncertainty-aware candidates 的形式保留下来。

---

## 目标 6：构建统一 evidence_segments.json

无论低重叠还是高重叠，最终都必须统一成一个中间格式：

```text
outputs/meeting_001/evidence_segments.json
```

统一格式是后续 LLM、Memory、QA 的唯一输入。

---

## 目标 7：Gemma 只做证据融合和结构化抽取

Gemma 不应该被设计成直接替代所有 ASR。

Gemma 的主要职责是：

修正文本  
保留不确定性  
提取会议事件  
生成 Episodic Memory  
回答问题  
基于 evidence_id 做追溯

它不能自由总结，必须按 JSON Schema 输出。

---

## 目标 8：生成 Episodic Memory

Episodic Memory 是本项目的核心后端记忆层。

它保存的不是普通 summary，而是：

某次会议  
某个时间段  
谁说了什么  
发生了什么事件  
是否高重叠  
置信度多少  
证据 ID 是什么  
音频片段在哪里

示例：

```json
{
  "episode_id": "m1_ep_001",
  "meeting_id": "meeting_001",
  "event_type": "action_item",
  "topic": "ASR baseline",
  "content": "SPEAKER_01 will test WhisperX and pyannote alignment.",
  "speakers": ["SPEAKER_01"],
  "start_time": 60.2,
  "end_time": 68.4,
  "evidence_ids": ["m1_seg_012"],
  "evidence_text": "我来测试 WhisperX 和 pyannote 的对齐。",
  "overlap_score": 0.08,
  "confidence": "high",
  "importance": 0.90,
  "audio_clip_paths": [
    "outputs/meeting_001/clips/m1_seg_012.wav"
  ]
}
```

---

## 目标 9：用户问答时，先检索，再让 Gemma 回答

不能让 Gemma 自己看全部 memory。

正确流程是：

```text
用户问题
↓
外部检索器检索 episodic_memory.json
↓
取 top-k 相关 episodes
↓
把这些 evidence 给 Gemma
↓
Gemma 只基于这些 evidence 回答
↓
回答必须引用 evidence_id 和时间戳
```

---

## 目标 10：Gradio Web Demo

最终要有一个 Gradio 页面，至少包含：

音频上传  
运行 pipeline  
时间线展示  
高重叠片段展示  
结构化会议记忆展示  
会议问答窗口  
证据追溯显示

---

# 三、推荐代码目录结构

建议目录如下：

```text
.
├── app.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
├── outputs/
│   └── meeting_001/
│       ├── preprocessed.wav
│       ├── vad_segments.json
│       ├── low_overlap_segments.json
│       ├── high_overlap_candidates.json
│       ├── evidence_segments.json
│       ├── meeting_events.json
│       ├── episodic_memory.json
│       └── clips/
├── memory/
│   ├── episodic_memory.json
│   ├── semantic_memory.json
│   ├── task_memory.json
│   ├── reflection_memory.json
│   └── vector_index/
├── src/
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── vad.py
│   │   └── clipper.py
│   ├── overlap/
│   │   ├── detector.py
│   │   └── router.py
│   ├── asr/
│   │   ├── whisper_backend.py
│   │   └── sensevoice_backend.py
│   ├── diarization/
│   │   ├── pyannote_backend.py
│   │   └── speaker_assign.py
│   ├── candidates/
│   │   ├── generator.py
│   │   └── separation_optional.py
│   ├── evidence/
│   │   ├── schema.py
│   │   ├── builder.py
│   │   └── validator.py
│   ├── llm/
│   │   ├── gemma_client.py
│   │   ├── prompts.py
│   │   └── json_repair.py
│   ├── memory/
│   │   ├── episodic_store.py
│   │   ├── retriever.py
│   │   └── memory_schema.py
│   ├── qa/
│   │   └── answerer.py
│   └── evaluation/
│       ├── routing_metrics.py
│       ├── evidence_metrics.py
│       └── qa_metrics.py
└── tests/
```

---

# 四、具体开发步骤

## Step 1：音频预处理模块

文件：

```text
src/audio/preprocess.py
```

目标：

读取音频  
转 mono  
重采样到 16kHz  
归一化  
保存标准 wav

函数：

```python
def preprocess_audio(input_path: str, output_path: str, target_sr: int = 16000) -> str:
    """
    Load audio, convert to mono, resample to 16kHz,
    normalize amplitude, and save to output_path.
    """
```

输入：

```text
data/raw/meeting_001.wav
```

输出：

```text
outputs/meeting_001/preprocessed.wav
```

验收标准：

能成功生成 16kHz mono wav  
音频长度不变  
音频无明显爆音  
后续模块可以读取

---

## Step 2：VAD 分段模块

文件：

```text
src/audio/vad.py
```

目标：

将音频切成有语音的片段。

可选实现：

energy-based VAD  
Silero VAD

推荐先做 energy-based，保证无重依赖可跑；之后再加 Silero。

函数：

```python
def segment_audio(audio_path: str, meeting_id: str) -> list[dict]:
    """
    Return timestamped speech segments.
    """
```

输出：

```json
[
  {
    "meeting_id": "meeting_001",
    "segment_id": "m1_seg_001",
    "start_time": 0.0,
    "end_time": 5.6
  }
]
```

验收标准：

每个 segment 有 start_time / end_time  
没有明显空白段  
长静音被切开  
片段长度大致在 1–30 秒之间

---

## Step 3：音频片段裁剪模块

文件：

```text
src/audio/clipper.py
```

目标：

根据 VAD 时间戳，把原音频切成 clip。

函数：

```python
def export_clips(audio_path: str, segments: list[dict], output_dir: str) -> list[dict]:
    """
    Save each segment as an audio clip and attach audio_clip_path.
    """
```

输出字段增加：

```json
{
  "segment_id": "m1_seg_001",
  "audio_clip_path": "outputs/meeting_001/clips/m1_seg_001.wav"
}
```

验收标准：

每个 segment 都有对应 wav clip  
clip 时长和 start/end 对齐  
Gradio 后面可以播放

---

## Step 4：ASR baseline 模块

文件：

```text
src/asr/whisper_backend.py
```

目标：

对每个 clip 生成 transcript。

推荐：

faster-whisper  
WhisperX

函数：

```python
def transcribe_clip(audio_clip_path: str, language: str | None = None) -> dict:
    """
    Return text and ASR confidence for an audio clip.
    """
```

输出：

```json
{
  "text": "I think we should first build the baseline.",
  "asr_confidence": 0.91
}
```

中英文混合处理：

第一版可以 language=None 自动识别  
如果效果差，高重叠候选中增加 zh/en 两种 decode

验收标准：

低重叠片段能得到可读 transcript  
中英文混合不崩溃  
每段都有 text 字段

---

## Step 5：说话人 diarization 模块

文件：

```text
src/diarization/pyannote_backend.py
```

目标：

得到 speaker 时间段。

函数：

```python
def diarize_audio(audio_path: str) -> list[dict]:
    """
    Return speaker-labeled time regions.
    """
```

输出：

```json
[
  {
    "speaker": "SPEAKER_00",
    "start_time": 0.0,
    "end_time": 4.2
  },
  {
    "speaker": "SPEAKER_01",
    "start_time": 4.4,
    "end_time": 9.1
  }
]
```

验收标准：

至少能区分 2 个 speaker  
能和 VAD segment 对齐  
输出 speaker label

---

## Step 6：speaker assignment 模块

文件：

```text
src/diarization/speaker_assign.py
```

目标：

把 diarization 的 speaker label 分配到每个 VAD segment。

函数：

```python
def assign_speaker_to_segments(segments: list[dict], diarization: list[dict]) -> list[dict]:
    """
    Assign speaker label and speaker_confidence based on time overlap.
    """
```

规则：

如果一个 speaker 覆盖该 segment 超过 70%，则赋给该 speaker  
如果多个 speaker 时间重叠明显，则 speaker = "MIXED"  
如果没有明显匹配，则 speaker = "UNKNOWN"

示例：

```json
{
  "segment_id": "m1_seg_004",
  "speaker": "SPEAKER_00",
  "speaker_confidence": 0.86
}
```

验收标准：

每个 segment 都有 speaker  
speaker_confidence 在 0–1 之间  
高混乱段不会被强行归给单人

---

## Step 7：overlap detector 模块

文件：

```text
src/overlap/detector.py
```

目标：

为每个 segment 生成 overlap_score。

第一版可以规则实现：

```python
overlap_score = (
    0.4 * diarization_overlap_score
    + 0.3 * asr_instability_score
    + 0.2 * speaker_change_score
    + 0.1 * energy_complexity_score
)
```

函数：

```python
def estimate_overlap_score(segment: dict, diarization: list[dict], asr_outputs: list[dict]) -> float:
    """
    Estimate overlap likelihood between 0 and 1.
    """
```

各部分含义：

diarization_overlap_score：同一时间段是否出现多个 speaker  
asr_instability_score：多次 ASR decode 是否差异大  
speaker_change_score：短时间内 speaker 是否频繁变化  
energy_complexity_score：能量曲线是否异常复杂

验收标准：

明显单人段 overlap_score 低  
明显抢话段 overlap_score 高  
所有分数范围在 0–1

---

## Step 8：router 模块

文件：

```text
src/overlap/router.py
```

目标：

根据 overlap_score 把片段分成低重叠 / 高重叠。

函数：

```python
def route_segment(segment: dict, threshold: float = 0.4) -> str:
    """
    Return low_overlap_cluster or high_overlap_candidate.
    """
```

输出字段：

```json
{
  "processing_path": "low_overlap_cluster"
}
```

或：

```json
{
  "processing_path": "high_overlap_candidate"
}
```

验收标准：

每个 segment 都有 processing_path  
低重叠和高重叠分开输出  
threshold 可调

---

## Step 9：低重叠处理模块

文件：

```text
src/evidence/builder.py
```

目标：

对 low_overlap_cluster 片段构建标准 evidence segment。

输入：

ASR text  
speaker  
time  
confidence  
overlap_score

输出：

```json
{
  "meeting_id": "meeting_001",
  "segment_id": "m1_seg_004",
  "speaker": "SPEAKER_00",
  "start_time": 12.4,
  "end_time": 18.2,
  "text": "I think we should first build the WhisperX baseline.",
  "processing_path": "low_overlap_cluster",
  "overlap_score": 0.12,
  "asr_confidence": 0.91,
  "speaker_confidence": 0.88,
  "candidates": [],
  "uncertainty_note": "",
  "audio_clip_path": "outputs/meeting_001/clips/m1_seg_004.wav"
}
```

验收标准：

字段完整  
低重叠片段 text 不为空  
candidates 为空数组  
uncertainty_note 可以为空

---

## Step 10：高重叠候选生成模块

文件：

```text
src/candidates/generator.py
```

目标：

对 high_overlap_candidate 片段生成多个候选解释。

第一版不做语音分离，只做多次 ASR decode。

候选生成方式：

beam_size 不同  
temperature 不同  
language 设置不同  
可选：不同 ASR 模型

函数：

```python
def generate_candidates(audio_clip_path: str, segment: dict) -> list[dict]:
    """
    Generate multiple transcript/speaker candidates for high-overlap segment.
    """
```

输出：

```json
[
  {
    "candidate_id": "m1_seg_009_c1",
    "speaker": "SPEAKER_00",
    "text": "We should test WhisperX first.",
    "confidence": 0.62
  },
  {
    "candidate_id": "m1_seg_009_c2",
    "speaker": "SPEAKER_01",
    "text": "但是重点是 overlap detection.",
    "confidence": 0.58
  }
]
```

验收标准：

每个高重叠片段至少有 1–3 个候选  
候选保留 confidence  
不能强行合并成唯一 transcript  
uncertainty_note 必须存在

---

## Step 11：可选语音分离模块

文件：

```text
src/candidates/separation_optional.py
```

目标：

如果时间允许，对高重叠片段做 speech separation。

可选模型：

SpeechBrain SepFormer  
Demucs  
Asteroid

输出：

```text
outputs/meeting_001/separated/m1_seg_009_spk1.wav
outputs/meeting_001/separated/m1_seg_009_spk2.wav
```

然后分别 ASR，加入 candidates。

注意：

这个是 bonus，不作为主线验收。

---

## Step 12：Evidence Segment Validator

文件：

```text
src/evidence/validator.py
```

目标：

确保 evidence_segments.json 可用、可信、格式统一。

函数：

```python
def validate_evidence_segments(segments: list[dict]) -> list[str]:
    """
    Return validation errors.
    """
```

检查规则：

每个 segment 必须有 meeting_id  
每个 segment 必须有 segment_id  
每个 segment 必须有 start_time / end_time  
processing_path 只能是 low_overlap_cluster 或 high_overlap_candidate  
overlap_score 必须在 0–1  
high_overlap_candidate 必须有 uncertainty_note  
low_overlap_cluster 应该有 text  
high_overlap_candidate 应该有 candidates  
speaker 不能为空  
audio_clip_path 应该存在

验收标准：

非法 JSON 能报错  
缺字段能报错  
高重叠没有 uncertainty_note 能报错  
低重叠没有 text 能报错

---

## Step 13：Gemma 调用模块

文件：

```text
src/llm/gemma_client.py
```

目标：

调用 Gemma 3 / Gemma 3n / Gemma 量化小模型。

实际选择：

如果本地 3090 跑得动，使用量化版本  
如果本地环境复杂，可以先封装接口，允许替换为其他 LLM

函数：

```python
def run_gemma(prompt: str) -> str:
    """
    Run Gemma and return raw text output.
    """
```

推荐模型策略：

主线使用量化小参数模型推理  
不做端到端音频微调  
如果要微调，只做文本侧 LoRA，让模型更稳定输出 JSON

验收标准：

能输入 prompt  
能返回文本  
能在 GPU 或 CPU fallback 下运行  
接口和具体模型解耦

---

## Step 14：Prompt 模板模块

文件：

```text
src/llm/prompts.py
```

目标：

写稳定的结构化抽取 prompt。

Prompt 核心规则：

只能基于 evidence_segments  
必须输出 JSON  
每个事件必须引用 evidence_ids  
不能编造  
高重叠片段不能作为高置信度事实  
如果 owner 不确定，写 uncertain  
不保存寒暄

核心模板：

```text
You are an episodic memory extraction module for a meeting understanding system.

Input:
A list of evidence segments. Each segment contains:
segment_id, start_time, end_time, speaker, text, processing_path,
overlap_score, asr_confidence, speaker_confidence, candidates, uncertainty_note.

Task:
Extract meaningful meeting events.

A memory should be created only if the segment contains:
1. a decision
2. an action item
3. a deadline
4. an open question
5. a disagreement
6. an uncertainty caused by overlapped speech

Output valid JSON only.

Rules:
- Every event must cite existing evidence_ids.
- Do not invent information.
- If overlap_score > 0.6, mark confidence as low or medium unless confirmed by low-overlap evidence.
- If the owner of an action item is unclear, set owner to "uncertain".
- Do not include small talk.
```

---

## Step 15：LLM JSON 修复和校验模块

文件：

```text
src/llm/json_repair.py
```

目标：

LLM 可能输出非法 JSON，需要修复或重试。

函数：

```python
def parse_or_repair_json(raw_output: str) -> dict:
    """
    Parse LLM output into JSON. If invalid, attempt repair or request regeneration.
    """
```

检查：

是否是合法 JSON  
是否有 meeting_id  
是否有 events  
每个 event 是否有 event_id  
每个 event 是否有 evidence_ids  
evidence_ids 是否真实存在

验收标准：

普通格式错误可以修复  
严重缺字段可以报错  
不会把无证据内容写入 memory

---

## Step 16：会议事件抽取模块

文件：

```text
src/llm/event_extractor.py
```

目标：

从 evidence_segments 中抽取 meeting_events。

输出：

```json
{
  "meeting_id": "meeting_001",
  "meeting_summary": "The team discussed an overlap-aware ASR pipeline.",
  "events": [
    {
      "event_id": "ev_001",
      "event_type": "decision",
      "content": "Use WhisperX and pyannote as the front-end baseline.",
      "speakers": ["SPEAKER_00"],
      "evidence_ids": ["m1_seg_004", "m1_seg_005"],
      "confidence": "high"
    },
    {
      "event_id": "ev_002",
      "event_type": "action_item",
      "task": "Test WhisperX and pyannote alignment.",
      "owner": "SPEAKER_01",
      "deadline": "this Friday",
      "evidence_ids": ["m1_seg_012"],
      "confidence": "high"
    }
  ]
}
```

验收标准：

能生成 decision  
能生成 action_item  
能生成 uncertainty  
每个事件都有 evidence_ids  
不会把高重叠片段强行标 high confidence

---

## Step 17：Episodic Memory Store

文件：

```text
src/memory/episodic_store.py
```

目标：

把 meeting_events 转换成 episodic_memory。

函数：

```python
def build_episodes(meeting_events: dict, evidence_segments: list[dict]) -> list[dict]:
    """
    Convert LLM-extracted meeting events into episodic memory records.
    """
```

输出：

```json
{
  "episode_id": "m1_ep_001",
  "meeting_id": "meeting_001",
  "event_type": "action_item",
  "topic": "ASR baseline",
  "content": "SPEAKER_01 will test WhisperX and pyannote alignment.",
  "speakers": ["SPEAKER_01"],
  "start_time": 60.2,
  "end_time": 68.4,
  "evidence_ids": ["m1_seg_012"],
  "evidence_text": "我来测试 WhisperX 和 pyannote 的对齐。",
  "overlap_score": 0.08,
  "confidence": "high",
  "importance": 0.90,
  "audio_clip_paths": [
    "outputs/meeting_001/clips/m1_seg_012.wav"
  ]
}
```

验收标准：

episode 能追溯到 evidence segment  
时间范围来自 evidence  
speaker 来自 evidence  
高重叠 episode 标为 uncertainty 或 low confidence

---

## Step 18：Memory Retriever

文件：

```text
src/memory/retriever.py
```

目标：

用户提问时，从 episodic_memory.json 中检索相关 episode。

第一版：

关键词检索

第二版：

embedding 检索

第三版：

BM25 + embedding 混合检索

函数：

```python
def retrieve_episodes(question: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve relevant episodic memories for a user question.
    """
```

推荐打分：

```text
final_score =
0.45 * embedding_similarity
+ 0.25 * keyword_score
+ 0.15 * importance
+ 0.10 * recency
- 0.20 * overlap_penalty
```

验收标准：

问 “谁负责 WhisperX？” 能找到对应 action item  
问 “有哪些不确定片段？” 能找到 uncertainty episodes  
问 “上次决定了什么？” 能找到 decision episodes

---

## Step 19：QA Answerer

文件：

```text
src/qa/answerer.py
```

目标：

让 Gemma 只基于检索出来的 episodes 回答。

函数：

```python
def answer_question(question: str, retrieved_episodes: list[dict]) -> str:
    """
    Generate evidence-backed answer using retrieved episodic memories.
    """
```

Prompt 规则：

```text
Use only the retrieved episodes.
Every factual claim must cite evidence_id and timestamp.
If evidence is insufficient, say you cannot determine.
If evidence is high-overlap or low-confidence, explicitly mention uncertainty.
Do not invent speaker names, tasks, or deadlines.
```

输出示例：

```text
SPEAKER_01 负责测试 WhisperX 和 pyannote 的对齐。证据来自 m1_seg_012，时间范围是 60.2–68.4 秒，置信度为 high。
```

验收标准：

回答必须带 evidence_id  
回答必须带时间范围  
证据不足时必须说无法确定  
不能凭空编造 owner / deadline / decision

---

# 五、Gradio Demo 开发目标

文件：

```text
app.py
```

## 页面 1：上传音频

组件：

```text
Audio upload
Run Pipeline button
```

用户上传：

```text
meeting_001.wav
```

点击运行后生成：

```text
evidence_segments.json
episodic_memory.json
```

---

## 页面 2：Timeline 展示

表格字段：

```text
time_range
speaker
processing_path
overlap_score
text
uncertainty_note
```

示例：

|Time|Speaker|Path|Overlap|Text|
|---|---|---|---|---|
|00:00–00:06|SPEAKER_00|low_overlap|0.08|Today we discuss...|
|00:06–00:12|SPEAKER_01|low_overlap|0.12|我觉得先用 WhisperX|
|00:12–00:17|MIXED|high_overlap|0.78|candidates available|

---

## 页面 3：High-overlap Candidates

点击一个 high-overlap segment，显示：

```json
{
  "segment_id": "m1_seg_013",
  "overlap_score": 0.82,
  "candidates": [
    "SPEAKER_00: We can use Gemma for post-processing.",
    "SPEAKER_01: But not directly for full ASR."
  ],
  "uncertainty_note": "Speaker attribution is uncertain."
}
```

---

## 页面 4：Meeting Memory

展示：

decision  
action item  
open question  
uncertainty

表格：

|Type|Content|Evidence|Confidence|
|---|---|---|---|
|decision|Use WhisperX + pyannote baseline|m1_seg_004|high|
|action_item|Test alignment|m1_seg_012|high|
|uncertainty|Gemma usage discussion unclear|m1_seg_013|low|

---

## 页面 5：QA

输入：

```text
谁负责测试 WhisperX？
```

输出：

```text
SPEAKER_01 负责测试 WhisperX 和 pyannote 的对齐。证据来自 m1_seg_012，时间范围 60.2–68.4 秒。
```

---

# 六、实验目标与具体做法

## 实验 1：Overlap Routing 实验

目标：

测试系统能不能正确区分低重叠和高重叠片段。

数据：

人工标注 20–50 个片段即可。

标签：

```text
low_overlap
high_overlap
```

指标：

accuracy  
precision  
recall  
F1

输出表：

|Threshold|Accuracy|Precision|Recall|F1|
|---|--:|--:|--:|--:|
|0.3|||||
|0.4|||||
|0.5|||||

---

## 实验 2：高重叠候选 vs 强制单输出

目标：

证明高重叠片段保留候选，比强行输出唯一 transcript 更可靠。

对比方法：

普通 ASR 单输出  
多候选 high-overlap 输出

人工评价：

candidate_usefulness：1–5  
uncertainty_correctness：1–5  
speaker_safety：1–5

输出表：

|Method|Candidate usefulness|Uncertainty correctness|Speaker safety|
|---|--:|--:|--:|
|Forced single ASR||||
|Multi-candidate ASR||||

---

## 实验 3：Metadata-aware LLM 消融

目标：

证明给 LLM metadata 比只给纯文本更好。

对比三种输入：

1. Plain transcript
    
2. Transcript + speaker
    
3. Full metadata：speaker + timestamp + overlap + confidence + candidates
    

评价指标：

action item accuracy  
decision accuracy  
evidence citation rate  
uncertainty preservation rate  
hallucination rate

输出表：

|Input Type|Action Item Acc|Evidence Rate|Uncertainty Rate|Hallucination Rate|
|---|--:|--:|--:|--:|
|Plain transcript|||||
|+ Speaker|||||
|+ Full metadata|||||

---

## 实验 4：Episodic Memory QA

目标：

证明 memory QA 比普通 summary QA 更可追溯。

对比：

Summary-only QA  
Transcript RAG QA  
Episodic Memory QA

问题类型：

谁负责某任务  
为什么做某决定  
有哪些未解决问题  
哪些片段不确定  
Gemma 的角色是什么  
WhisperX 为什么作为 baseline

指标：

answer correctness  
evidence hit rate  
timestamp citation rate  
hallucination rate

输出表：

|Method|Correctness|Evidence Hit|Timestamp Citation|Hallucination|
|---|--:|--:|--:|--:|
|Summary QA|||||
|Transcript RAG QA|||||
|Episodic Memory QA|||||
