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

1. Compare predicted overlap routes with manual labels.
2. Compare high-overlap candidate generation with forced single-output transcription.
3. Compare plain-text, speaker-aware, and full-metadata LLM post-processing.
4. Compare summary QA, transcript RAG, and speaker-aware Episodic Memory QA.
5. Measure hallucination rate and timestamped evidence hit rate.

Full details are in [docs/experiment_plan.md](docs/experiment_plan.md).

## Current Status

The repository currently contains the first-stage research design, annotation schema, and clean module interfaces. Heavy models such as Whisper, pyannote, and speech separation models are intentionally not loaded yet.

## How to Run

The current code is an interface-only scaffold:

```bash
python main.py
python app.py
```

Implement the TODOs in `src/` before running real audio experiments. Keep large audio files, model weights, and generated outputs outside Git.

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

1. 将预测的重叠路由与人工标注进行比较。
2. 比较高重叠候选生成与强制单一转写。
3. 比较纯文本、带说话人信息和完整元信息三种 LLM 后处理方式。
4. 比较摘要问答、纯转写 RAG 和说话人感知的 Episodic Memory 问答。
5. 测量幻觉率和带时间戳的证据命中率。

完整实验计划见 [docs/experiment_plan.zh-CN.md](docs/experiment_plan.zh-CN.md)。

## 当前进度

仓库目前包含第一阶段研究设计、标注 Schema 和清晰的模块接口。Whisper、pyannote 和语音分离模型等重模型暂未加载。

## 运行方式

当前代码仅为接口骨架：

```bash
python main.py
python app.py
```

在进行真实音频实验之前，需要先实现 `src/` 中的 TODO。大型音频文件、模型权重和生成结果不应提交到 Git。
