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

`ASR + 重叠感知路由 -> 不确定性感知候选生成 -> 元信息感知 LLM 后处理 -> Episodic Memory -> 可追溯问答与会议回忆`

核心区别是：系统不会强制将高重叠语音变成唯一且“确定”的转写，而会将候选解释与置信度继续传递给后续推理和检索模块。

## 核心创新点

1. **重叠感知路由：**低重叠语音进入轻量说话人聚类路径，高重叠语音进入分离或候选生成路径。
2. **不确定性感知候选生成：**为模糊区域保留多个可能的转写和说话人假设。
3. **元信息感知 LLM 后处理：**LLM 同时使用时间戳、置信度、重叠程度、候选解释和历史记忆，而不是只读取纯文本。
4. **Episodic Memory / 情景记忆：**保存带证据的会议事件，支持可追溯问答、行动项检索和跨会议回忆。
5. **超越 WER 与 DER 的评估：**评估路由准确性、候选有效性、不确定性保留、证据质量和幻觉率。

## 系统流程

1. 预处理音频并生成带时间戳的片段。
2. 估计各片段的重叠分数（优先 pyannote OSD，不可用时使用能量 fallback）。
3. 根据重叠程度路由：
   - 低重叠（< 阈值）：VAD、说话人聚类和 ASR。
   - 高重叠（>= 阈值）：生成多个候选解释。
4. 为所有片段构建统一元信息记录（17 字段）。
5. 导出每段音频 clip，进行 schema 验证。
6. 使用 LLM 提取有证据支持的会议事件。
7. 将片段转换为情景记忆记录并持久化。
8. 检索情景记忆，以说话人、时间戳、置信度和不确定性说明回答问题。

模块设计见 [docs/system_architecture.zh-CN.md](docs/system_architecture.zh-CN.md)，目录职责与扩展约定见 [docs/project_structure.zh-CN.md](docs/project_structure.zh-CN.md)，完整 pipeline 调用链见 [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md)。

## 仓库结构

```text
.
├── docs/                  # 双语研究设计、系统架构与实验计划
├── data/                  # 原始/处理后音频、标注模板与测试 fixture
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
│   └── ui/                # Gradio 交互演示
├── tests/                 # 单元测试（75 项）
├── app.py                 # Gradio 交互演示入口
├── main.py                # 命令行 pipeline 入口
├── README.md
└── README.zh-CN.md
```

## 元信息 Schema

每个处理后的片段采用统一 17 字段 Schema：

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
| 2. 高重叠候选 | 比较候选生成与强制单一转写 | 候选接口已实现；正式实验待进行 |
| 3. 元信息感知 LLM | 比较纯文本、说话人感知和完整元信息 LLM 后处理 | LLM 事件提取已实现；元信息 prompt 消融实验待进行 |
| 4. Episodic Memory 问答 | 比较摘要问答、纯转写 RAG 和说话人感知记忆问答 | 存储、检索和基线 QA 已实现；正式实验待进行 |
| 5. 幻觉与证据 | 测量幻觉率和带时间戳的证据命中率 | 指标接口已定义（stub）；正式实验待进行 |

完整计划见 [docs/experiment_plan.zh-CN.md](docs/experiment_plan.zh-CN.md)。

## 当前进度

项目目前处于**基础设施与基线准备阶段**，已有可运行的端到端 pipeline，正式实验结果尚未产生。

已完成：

- 双语研究设计、系统架构、创新点阐述和实验计划；
- 统一的 evidence-packet 元信息 Schema（17 字段）、校验规则和示例会议 fixture；
- 音频加载、单声道转换、polyphase 重采样、峰值归一化和基于能量的 VAD 分段（含段落合并与分割）；
- 音频 clip 导出（`src/audio/clipper.py`）；
- 双说话人可控重叠语音合成，支持 SNR 控制和重叠真值标注；
- WER、CER、重叠路由分类和最优映射说话人归属等客观评估指标；
- 可插拔 ASR 适配器（Mock/WhisperX/Whisper/Paraformer）与置信度校准；
- 重叠检测：pyannote OSD 适配器（有 HF token 时）+ 保守能量 fallback（上限 0.39，不会误触发高重叠路由）；
- 双路径路由（阈值 0.4）、低重叠 ASR + 说话人归属路径、高重叠候选生成（不强行确定单一转写）；
- 元数据构建、schema 验证和 LLM 事件提取；
- 事件级情景记忆创建、按会议原子更新的 JSON 持久化与 BM25 + embedding 混合检索；
- Gemma 仅基于 top-k Episode 回答，并校验证据 ID、时间戳、说话人和不确定性；
- 基于 Gradio 的交互式 UI 演示；
- 端到端 pipeline 编排（`src/pipeline/run_pipeline.py`）；
- 75 项单元测试覆盖已实现基础设施。

正式实验前仍需完成：

- 人工标注评估集构建；
- pyannote 模型下载与校准实验；
- faster-whisper/WhisperX/Whisper/FunASR 重模型集成与精度对比；
- 元信息输入消融实验和证据质量评估。

## 运行方式

安装轻量级基线依赖并运行测试：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

运行端到端 pipeline：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
```

启动 Gradio 交互演示：

```bash
python -m pip install -r requirements-demo.txt
python app.py
```

Demo 包含音频上传、重叠感知时间线、高重叠候选查看、结构化会议记忆，以及仅针对当前会议的证据引用问答。

大型音频文件、模型权重和生成结果不应提交到 Git。
