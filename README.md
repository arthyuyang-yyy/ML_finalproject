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
├── src/                   # Modular pipeline interfaces
├── app.py                 # Future interactive application entry point
├── main.py                # Pipeline entry point
├── README.md
└── README.zh-CN.md
```

## Metadata Schema

Each processed segment uses a shared schema:

| Field | Meaning |
| --- | --- |
| `meeting_id`, `segment_id` | Stable meeting and segment identifiers |
| `speaker` | Speaker label or uncertain speaker hypothesis |
| `start_time`, `end_time` | Evidence timestamp range in seconds |
| `text` | Current transcript |
| `processing_path` | `low_overlap_cluster` or `high_overlap_candidate` |
| `overlap_score` | Estimated overlap likelihood |
| `asr_confidence` | ASR confidence estimate |
| `speaker_confidence` | Speaker-attribution confidence |
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
| 1. Overlap routing | Compare predicted overlap routes with manual labels | Infrastructure ready; baseline detector and experiment run pending |
| 2. High-overlap candidates | Compare candidate generation with forced single-output transcription | Candidate interface defined; implementation and experiment pending |
| 3. Metadata-aware LLM | Compare plain-text, speaker-aware, and full-metadata LLM post-processing | Prompt constraints defined; LLM integration and ablation pending |
| 4. Episodic Memory QA | Compare summary QA, transcript RAG, and speaker-aware memory QA | Interfaces defined; storage, retrieval, and experiment pending |
| 5. Hallucination and evidence | Measure hallucination rate and timestamped evidence hit rate | Metric contract and experiment pending |

Full details are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Current Status

The project is currently in the **baseline infrastructure stage**. Formal experiment results have not been produced yet.

Completed:

- bilingual research design, architecture, experiment plan, and module interfaces;
- shared evidence-packet metadata schema, validation rules, and sample meeting fixture;
- audio loading interface, mono conversion, linear resampling, peak normalization, and energy-based VAD segmentation;
- controlled two-speaker overlap synthesis with SNR control and ground-truth overlap annotations;
- objective WER, CER, overlap-routing, and best-mapping speaker-attribution metrics;
- 46 unit tests covering the implemented baseline infrastructure.

Pending before formal experiments:

- calibrated overlap detector and routing-threshold study;
- ASR, speaker diarization/clustering, and high-overlap candidate-generation baselines;
- uncertainty-aware LLM integration and metadata-input ablations;
- persistent Episodic Memory, retrieval, and evidence-backed QA;
- manually annotated evaluation split and finalized evidence/hallucination metrics.

Current verification note: 25 dependency-free tests pass in the present environment. The remaining 21 NumPy-based preprocessing and data-synthesis tests require installing the dependencies in `requirements.txt`. Heavy models such as Whisper, pyannote, and speech separation models remain intentionally unloaded.

## How to Run

Install the lightweight baseline dependencies and run the infrastructure tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`main.py` and `app.py` remain placeholders because the end-to-end ASR, memory, and QA pipeline is not implemented yet. Keep large audio files, model weights, and generated outputs outside Git.

---

# 中文完整翻译

## 项目名称

**面向多人会议理解的重叠感知双路径语音处理与情景记忆系统**

英文名称：Overlap-aware Dual-path ASR with Episodic Memory for Multi-speaker Meeting Understanding

## 项目背景

常见的会议助手会将存在错误的转写压缩成流畅的摘要，这可能掩盖说话人归属错误、重叠语音和缺乏证据支持的结论。本项目不仅是一个会议摘要系统，更是一个**可验证的会议记忆系统**。系统会保留不确定性，并将后续回答、决策和行动项关联到带时间戳的原始证据。

## 与参考论文的区别

参考论文已经结合了 ASR、说话人日志、低重叠说话人聚类、高重叠语音分离、LLM 纠错和结构化会议摘要。本项目将在这一基础上进行扩展，而不是简单复现参考系统。

**参考系统**

`ASR -> 说话人日志 -> LLM 纠错 -> 结构化摘要`

**本项目系统**

`ASR + 重叠感知路由 -> 不确定性感知候选生成 -> 元信息感知 LLM 后处理 -> Episodic Memory -> 可追溯问答与会议回忆`

核心变化是：系统不会强制将高重叠语音转换成一个看似确定的转写结果。候选解释和置信度元信息会被保留，并继续用于后续推理和检索。

## 核心创新点

1. **重叠感知路由：**将低重叠音频路由到轻量级说话人聚类路径，将高重叠音频路由到语音分离或候选生成路径。
2. **不确定性感知候选生成：**为模糊区域保留多个可能的转写和说话人假设。
3. **元信息感知 LLM 后处理：**LLM 基于时间戳、置信度、重叠程度、候选解释和历史记忆进行推理，而不是只处理纯文本。
4. **Episodic Memory / 情景记忆：**保存带证据的有意义会议事件，支持可追溯问答、行动项检索和跨会议回忆。
5. **超越 WER 与 DER 的评估：**评估路由准确性、候选有效性、不确定性保留、证据质量和幻觉率。

## 系统流程

1. 预处理音频并创建带时间戳的片段。
2. 估计每个片段的重叠分数。
3. 对每个片段进行路由：
   - 低重叠：VAD、说话人嵌入、聚类和 ASR。
   - 高重叠：语音分离或生成多个候选解释。
4. 为每个片段构建统一的元信息记录。
5. 使用 LLM 纠正文本、保留不确定性，并提取有证据支持的会议事件。
6. 将相关片段转换成 Episodic Memory 记录。
7. 检索 episode，并通过说话人、时间戳、置信度和不确定性说明回答问题。

模块级设计见 [docs/system_architecture.zh-CN.md](docs/system_architecture.zh-CN.md)。

## 仓库结构

```text
.
├── docs/                  # 双语研究设计与实验计划
├── data/                  # 原始/处理后音频与标注模板
├── outputs/               # 生成结果，除占位文件外不纳入 Git
├── src/                   # 模块化流程接口
├── app.py                 # 后续交互应用入口
├── main.py                # 流程入口
├── README.md
└── README.zh-CN.md
```

## 元信息 Schema

每个处理后的片段使用统一 Schema：

| 字段 | 含义 |
| --- | --- |
| `meeting_id`, `segment_id` | 稳定的会议与片段标识 |
| `speaker` | 说话人标签或不确定的说话人假设 |
| `start_time`, `end_time` | 以秒为单位的证据时间范围 |
| `text` | 当前转写文本 |
| `processing_path` | `low_overlap_cluster` 或 `high_overlap_candidate` |
| `overlap_score` | 估计的语音重叠概率 |
| `asr_confidence` | ASR 置信度估计 |
| `speaker_confidence` | 说话人归属置信度 |
| `candidates` | 备选转写和说话人解释 |
| `uncertainty_note` | 对不确定原因的可读说明 |

## Episodic Memory / 情景记忆设计

一个 episode 表示有意义的会议事件或一组连贯的会议片段。它保存会议 ID 和事件 ID、时间范围、说话人、主题、原始与纠正后的转写、重叠与置信度信息、候选解释、决策、行动项、证据文本，以及后续用于检索的嵌入向量。

Episode 支持：

- 基于证据的会议问答；
- 历史会议与跨会议回忆；
- 行动项和决策检索；
- 指定说话人检索；
- 从回答追溯到精确时间戳。

## 实验计划

| 实验 | 目标 | 当前状态 |
| --- | --- | --- |
| 1. 重叠路由 | 将预测的重叠路由与人工标注比较 | 实验基础设施已完成；基础检测器和正式实验待完成 |
| 2. 高重叠候选 | 比较候选生成与强制单一转写 | 候选接口已定义；实现和正式实验待完成 |
| 3. 元信息感知 LLM | 比较纯文本、说话人感知和完整元信息 LLM 后处理 | 提示词约束已定义；LLM 集成和消融实验待完成 |
| 4. Episodic Memory 问答 | 比较摘要问答、纯转写 RAG 和说话人感知记忆问答 | 接口已定义；存储、检索和正式实验待完成 |
| 5. 幻觉与证据 | 测量幻觉率和带时间戳的证据命中率 | 指标定义和正式实验待完成 |

完整实验计划见 [docs/experiment_plan.zh-CN.md](docs/experiment_plan.zh-CN.md)。

## 当前进度

项目目前处于**基础设施与基线准备阶段**，尚未产生正式实验结果。

已完成：

- 双语研究设计、系统架构、实验计划和模块接口；
- 统一的 evidence-packet 元信息 Schema、校验规则和示例会议 fixture；
- 音频加载接口、单声道转换、线性重采样、峰值归一化和基于能量的 VAD 分段；
- 支持 SNR 控制和重叠真值标注的双说话人可控重叠语音合成；
- WER、CER、重叠路由和最优映射说话人归属等客观评估指标；
- 覆盖已实现基础设施的 46 项单元测试。

正式实验前仍需完成：

- 校准后的重叠检测器和路由阈值实验；
- ASR、说话人日志/聚类和高重叠候选生成基线；
- 不确定性感知 LLM 集成和元信息输入消融实验；
- 持久化 Episodic Memory、检索和基于证据的问答；
- 人工标注评估集，以及最终的证据与幻觉指标定义。

当前验证说明：现有环境中不依赖 NumPy 的 25 项测试已通过；其余 21 项预处理和数据合成测试需要先安装 `requirements.txt` 中的依赖。Whisper、pyannote 和语音分离模型等重模型仍按计划保持未加载状态。

## 运行方式

安装轻量级基线依赖并运行基础设施测试：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`main.py` 和 `app.py` 仍为占位入口，因为端到端 ASR、记忆与问答流程尚未实现。大型音频文件、模型权重和生成结果不应提交到 Git。
