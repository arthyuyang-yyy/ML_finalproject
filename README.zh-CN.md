# 面向多人会议理解的重叠感知双路径语音处理与情景记忆系统

## 项目名称

**面向多人会议理解的重叠感知双路径语音处理与情景记忆系统**

英文名称：Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding

## 项目背景

常见会议助手会把存在错误的转写直接压缩成流畅摘要，从而掩盖说话人归属错误、重叠语音和缺乏证据的结论。本项目不仅是会议摘要系统，更是一个**可验证的会议记忆系统**：它保留不确定性，并将问答、决策和行动项追溯到带时间戳的证据。

## 与参考论文的区别

参考论文已经包含 ASR、说话人日志、低重叠聚类、高重叠语音分离、LLM 纠错和结构化会议摘要。本项目在此基础上扩展，而不是简单复现。

**参考系统**

`ASR -> 说话人日志 -> LLM 纠错 -> 结构化摘要`

**本项目系统**

`音频预处理 -> 重叠检测与双路径处理 -> Evidence -> 结构化事件 -> Episodic Memory -> 可追溯检索与问答`

核心区别不是简单增加一个 LLM，而是：高重叠语音不会被强制写成唯一且“确定”的转写；候选、置信度和不确定性会继续传递到结构化事件、会议记忆和最终答案。LLM 只作为可替换的事件抽取与答案组织组件，系统在不接入 LLM 时也应能够运行和评估。

## 核心创新点

1. **重叠语音的不确定性传播：**低重叠片段生成稳定转写；高重叠片段保留多个带置信度的转写与说话人候选。后续事件、记忆和答案不能悄悄把不确定候选压缩为确定事实。
2. **可追溯的事件级会议记忆与 RAG：**会议内容保存为绑定说话人、时间戳、置信度、Evidence ID 和原始音频片段的 Episode。检索和问答可以追溯到具体证据，并支持行动项、决策、不确定内容和跨会议记录查询。
3. **面向可信会议理解的评估：**除 WER 与 DER 外，评估路由准确率、候选有效性、不确定性保留、证据命中、内容支持和无支持声明。

LLM 接入本身不是创新点。项目会把它作为可选实验变量，比较确定性规则、普通 LLM 和读取完整 Evidence 元信息的受约束 LLM。

## 系统流程

1. **预处理：**统一采样率和声道，使用 VAD 生成带时间戳的语音片段。
2. **重叠检测与分流：**估计 `overlap_score`，将片段路由到低重叠或高重叠路径。
3. **双路径处理：**低重叠片段执行 ASR 和说话人归属；高重叠片段生成多个带置信度的候选，真实语音分离仍是待完成任务。
4. **Evidence 证据层：**将文本、候选、说话人、时间戳、置信度、路由原因和原始音频路径统一为 17 字段记录（外加 1 个可选的聚类相似度分布字段）并校验。
5. **结构化事件：**从 Evidence 中提取 decision、action item、open question 和 uncertainty。可以使用规则，也可以使用受证据约束的 LLM。
6. **Episodic Memory：**将事件转换为可持久化、可检索、可回溯原始证据的 Episode。
7. **检索与问答：**先通过 BM25 + embedding 检索 Episode，再使用模板或可选 LLM 组织回答；回答必须引用真实 Evidence ID 和时间戳。

模块设计见 [docs/system_architecture.zh-CN.md](docs/system_architecture.zh-CN.md)，目录职责与扩展约定见 [docs/project_structure.zh-CN.md](docs/project_structure.zh-CN.md)，完整 pipeline 调用链见 [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md)。

## 仓库结构

```text
.
├── docs/                  # 双语研究设计、系统架构与实验计划
├── data/
│   ├── raw_audio/         # 原始会议音频
│   └── processed_audio/   # 预处理/派生音频
├── outputs/               # 生成结果，除占位文件外不纳入 Git
├── src/                   # 模块化 pipeline 实现
│   ├── audio/             # 音频预处理、归一化、导出与 clip 输出
│   ├── pipeline/          # 端到端编排、配置与 I/O 工具
│   ├── overlap/           # 重叠检测 facade
│   ├── evidence/          # 元数据构建与验证 facade
│   ├── llm/               # LLM 事件提取、验证与 prompt 构建
│   ├── memory/            # 情景记忆 facade
│   ├── qa/                # 问答 facade
│   ├── candidates/        # 候选生成 facade
│   ├── fallbacks/         # 确定性轻量回退后端
│   └── ui/                # Gradio 交互演示
├── tests/                 # 单元测试与集成测试
├── app.py                 # Gradio 交互演示入口
├── main.py                # 命令行 pipeline 入口
├── README.md
└── README.zh-CN.md
```

## 元信息 Schema

每个处理后的片段采用统一 Schema（17 个必填字段 + 1 个可选字段）：

| 字段 | 含义 |
| --- | --- |
| `meeting_id` | 稳定的会议标识 |
| `segment_id` | 稳定的片段标识 |
| `evidence_id` | 证据记录唯一 ID（通常与 segment_id 相同） |
| `speaker` | 说话人标签或不确定的说话人假设 |
| `start_time` | 以秒计的证据起始时间 |
| `end_time` | 以秒计的证据结束时间 |
| `text` | 当前转写文本 |
| `processing_path` | `low_overlap_cluster` 或 `high_overlap_candidate` |
| `route_reason` | 路由决策的可读说明 |
| `overlap_score` | 估计的重叠概率 [0, 1] |
| `asr_confidence` | ASR 置信度 [0, 1] |
| `speaker_confidence` | 说话人归属置信度 [0, 1] |
| `audio_clip_path` | 导出音频 clip 的文件路径 |
| `source_audio_path` | 原始输入音频路径 |
| `language` | 语言代码（默认 "und"） |
| `candidates` | 备选转写与说话人解释列表 |
| `uncertainty_note` | 对不确定原因的可读说明 |
| `cluster_similarity_distribution` | *可选。* 聚类 fallback 输出的相对相似度分布 `{说话人: 相似度}`，是未校准的相对信号而非校准后验，默认 `{}` |

## Episodic Memory / 情景记忆设计

一个 episode 表示一个有意义且有证据支持的会议事件。它保存会议与事件 ID、时间范围、说话人、主题、内容、证据 ID、证据文本、重叠分数、置信度、重要度、写入时间和音频片段路径。长期记忆写入 `memory/episodic_memory.json`，按 meeting ID 原子更新，并通过 BM25 + embedding 混合检索。

情景记忆支持：

- 基于证据的会议问答；
- 历史会议与跨会议回忆；
- 行动项和决策检索；
- 指定说话人检索；
- 从回答追溯到精确时间戳。

## 实验计划

| 实验 | 目标 | 当前状态 |
| --- | --- | --- |
| 1. 重叠路由 | 将预测的重叠路由与人工标注比较 | 基础设施就绪；pyannote 适配器和能量 fallback 已实现；标注集待构建 |
| 2. 高重叠候选 | 比较候选生成与强制单一转写 | 候选接口和指标已实现；真实高重叠音频实验与语音分离待完成 |
| 3. 结构化事件抽取 | 比较规则、普通 LLM 和完整 Evidence 元信息约束 LLM | 规则 fallback、LLM 接口和校验已实现；真实模型消融待进行 |
| 4. Episodic Memory 问答 | 比较摘要问答、纯转写 RAG 和说话人感知记忆问答 | 存储、检索和基线 QA 已实现；正式实验待进行 |
| 5. 证据与不确定性 | 测量证据命中、内容支持、无支持声明和不确定性保留 | 基础指标与种子实验已实现；真实 Pipeline 输出标注集待构建 |

完整计划见 [docs/experiment_plan.zh-CN.md](docs/experiment_plan.zh-CN.md)。

## 当前进度

项目目前处于**主线基础设施可运行、真实高重叠处理与正式实验待完成**阶段。轻量环境可以完整验证软件流程，但当前输出大量依赖 Mock ASR 和确定性 fallback，不能代表真实会议处理效果。

已完成：

- 双语研究设计、系统架构、创新点阐述和实验计划；
- 统一的 evidence-packet 元信息 Schema（17 字段）、校验规则和示例会议 fixture；
- 音频加载：常规格式优先使用 soundfile，M4A/AAC/MP4/WMA 等容器回退到 PyAV 解封装与解码；随后执行可选降噪、单声道转换、单次 polyphase 重采样、峰值归一化和基于能量的 VAD 分段（含段落合并与分割）；
- 音频 clip 导出（`src/audio/clipper.py`）；
- 双说话人可控重叠语音合成，支持 SNR 控制和重叠真值标注；
- WER、CER、重叠路由分类、最优映射说话人归属、引用率和时间戳引用率等基础评估指标；
- 可插拔 ASR 适配器（Mock/WhisperX/faster-whisper/Whisper/FunASR）与置信度校准；
- 重叠检测：pyannote OSD 适配器（有 HF token 时）+ 保守能量 fallback（上限 0.39，不会误触发高重叠路由）；
- 双路径路由（阈值 0.4）、低重叠 ASR + 说话人归属路径、高重叠候选生成（不强行确定单一转写）；
- 可选高重叠语音分离（可替换适配器）：零依赖自写 NMF 基线 + 可选 SpeechBrain SepFormer；分离出的声源分别执行 ASR，未启用或失败时保留原有多参数候选回退；
- 元数据构建、schema 验证、evidence-only JSON Prompt、LLM 输出修复与校验，以及确定性事件提取 fallback；
- 事件级情景记忆创建、按会议原子更新的 JSON 持久化与 BM25 + embedding 混合检索；
- 模板 QA 和可选 Gemma QA 均只基于 top-k Episode，并校验证据 ID、时间戳、说话人和不确定性；
- 基于 Gradio 的交互式 UI 演示；
- 端到端 pipeline 编排（`src/pipeline/run_pipeline.py`）；
- 自动化测试覆盖已实现的基础设施、运行时后端、检索、记忆、QA 与评估。

当前可运行程度：

- 仅安装 `requirements.txt` 时，可以使用 Mock ASR 和确定性 fallback 完整运行 CLI Pipeline，生成每场会议的全部核心 artifact、Episodic Memory 和证据问答结果；
- 轻量模式适合验证端到端系统流程，不代表真实 ASR、说话人识别、重叠检测或事件抽取质量；
- 安装并配置可选依赖、模型、Hugging Face token 和 Ollama 服务后，可以切换真实 ASR、pyannote、Gemma 与 Gradio。

正式实验前仍需完成：

- 人工标注评估集构建；
- 重叠阈值校准和 pyannote 正式实验；
- 真实高重叠语音分离质量评估与模型对比；
- faster-whisper/WhisperX/Whisper/FunASR、pyannote 与 Ollama Gemma 的真实运行和精度对比；
- decision、action item、deadline 等真实会议事件抽取验证；
- 使用真实 Pipeline 输出构建内容支持、不确定性保留和候选有效性标注集；
- 规则/普通 LLM/Evidence 约束 LLM、Summary QA/Transcript RAG/Episodic Memory QA 等消融与对比实验。

2026 年 6 月 13 日验证结果：

- `280` 个单元与集成测试通过；
- `1` 个可选 Gradio 组件测试因未安装 Gradio 跳过；
- 证据评估种子实验可运行并生成结果表；
- 使用轻量依赖和 Mock ASR 的端到端 smoke run 成功。

## 运行方式

安装轻量级基线依赖并运行测试：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

安装 Step 4 选定的真实 ASR baseline：

```bash
python -m pip install -r requirements-asr.txt
```

Phase 2 Step 11 的零依赖 NMF 语音分离基线无需额外安装。可选的重模型 SepFormer baseline 需要：

```bash
python -m pip install -r requirements-separation.txt
```

运行端到端 pipeline。CLI 默认使用 `faster-whisper small`，首次运行会下载模型：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
```

启用高重叠语音分离（`nmf` 零依赖，`sepformer` 需重依赖）：

```bash
# 零依赖 NMF 基线
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 \
  --speech-separation nmf

# 可选重模型 SepFormer
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 \
  --speech-separation sepformer
```

两者默认关闭。SepFormer 首次启用时会下载 `speechbrain/sepformer-whamr16k`。分离模型不可用、
加载失败或未生成有效分轨候选时，Pipeline 会自动回到原有 faster-whisper 多参数候选路径。

输入可以是 WAV、FLAC、OGG、MP3、M4A、AAC、MP4 或 WMA。所有格式都会先统一为
16 kHz 单声道 float32 WAV，因此 Step 4 的 ASR 后端不需要针对每种文件格式分别适配。
可使用 `--denoise --denoise-strength 0.5` 开启可选静态噪声抑制；该功能默认关闭，
需要另外安装 `noisereduce`。

仅验证软件流程时可使用 `--asr mock`。详细配置见
[`docs/asr_baseline.zh-CN.md`](docs/asr_baseline.zh-CN.md)。

启动 Gradio 交互演示：

```bash
python -m pip install -r requirements-demo.txt
python app.py
```

Demo 包含音频上传、重叠感知时间线、高重叠候选查看、结构化会议记忆，以及仅针对当前会议的证据引用问答。

大型音频文件、模型权重和生成结果不应提交到 Git。
