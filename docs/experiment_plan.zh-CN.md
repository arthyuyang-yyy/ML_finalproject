# 实验计划

## 通用设置

建立人工标注的评估集，包含时间戳、说话人标签、重叠标签、转写文本、主题、决策和行动项。报告总体指标，并分别报告低重叠与高重叠区域结果。

**合成数据**可通过 `src/data_synthesis.py` 生成可控的双人重叠语音混合，包含已知真值（重叠时长、SNR、说话人片段），用于正式标注前的预评估校准。

## 实验 1：重叠路由

将预测的 `low_overlap_cluster` 和 `high_overlap_candidate` 路由与人工重叠标签比较。

**指标**（实现于 `src/evaluation.py` — `evaluate_overlap_routing()`）：

- **准确率 (Accuracy)**：`(TP + TN) / total`
- **精确率 (Precision)**：`TP / (TP + FP)`
- **召回率 (Recall)**：`TP / (TP + FN)`
- **F1**：`2 * precision * recall / (precision + recall)`

正类 = `high_overlap_candidate`。报告不同阈值下的下游成本与质量权衡。

**检测器选项**（实现于 `src/overlap_detector.py`）：
- pyannote OSD（需 `HF_TOKEN` 和 `pip install pyannote.audio`）
- 能量 fallback（始终可用，上限 0.39）

**状态：** 基础设施就绪。pyannote 适配器和能量 fallback 已实现。标注评估集待构建。

## 实验 2：高重叠候选生成

比较候选输出与强制单一转写。

**指标**（相关函数已实现于 `src/evaluation.py`）：

- Oracle 候选 WER：top-K 候选中最佳 WER 对比参考
- Top-1 WER：最高置信度候选的 WER
- 说话人假设覆盖率：至少一位候选命中正确说话人的片段比例
- 候选有效性：人工评定有用信息是否能在模糊重叠区域中保留

**候选生成基线**（实现于 `src/high_overlap.py` 与 `src/candidate_generator.py`）：高重叠主记录保持 mixed/空转写，候选由 faster-whisper 多参数解码生成（beam size、temperature、language）。如未安装 faster-whisper，则输出显式 fallback 候选以保留同一套不确定性 schema。

**状态：** 候选接口已实现。oracle/top-1 WER 和人工评定需标注数据。

## 实验 3：元信息感知 LLM 后处理

比较三种 LLM 输入配置：

1. 纯转写 + LLM
2. 转写 + 说话人标签 + LLM
3. 转写 + 说话人标签 + 重叠/置信度元信息 + LLM

**指标：**
- 纠错质量（原始转写与纠正后转写的 WER）
- 说话人归属准确率（`speaker_attribution_accuracy()` in `src/evaluation.py`）
- 不确定性保留质量（不确定片段是否仍被标记）
- 决策提取准确性
- 行动项提取准确性

**LLM 集成**（实现于 `src/llm/event_extractor.py`）：当前使用确定性 fallback；Gemma 客户端接口已就绪，待接入真实 LLM。

**状态：** LLM 事件提取基础设施已实现。元信息消融实验需真实 LLM 集成和人工评估。

## 实验 4：Episodic Memory 问答

比较三种问答方式：

1. 普通摘要问答
2. 基于纯转写的 RAG
3. 说话人感知的 Episodic Memory 问答

**指标：**
- 问答准确率（事实正确性）
- 说话人检索准确率
- 行动项检索精确率/召回率
- 跨会议回忆
- 证据命中率（回答是否引用正确片段）

**记忆基础设施**（实现于 `src/episodic_memory.py` 和 `src/rag_qa.py`）：
- 从证据片段创建 episode
- JSONL 持久化
- 关键词检索基线
- QA 响应格式（含 evidence、speaker、timestamp、confidence）

**状态：** 存储、检索和基线 QA 已实现。语义/向量检索和正式 QA 评估待进行。

## 实验 5：幻觉与证据评估

评估回答、决策和行动项是否得到带时间戳证据的支持。

**指标**（接口定义于 `src/evaluation.py` — `evaluate_evidence_support()`，当前为 stub）：

- 证据精确率：引用证据中正确的比例
- 证据召回率：正确声明中引用证据的比例
- 无支持声明率：无支持证据的声明比例
- 幻觉率：未在任何源片段中找到的声明比例
- 置信度校准：置信度分数与正确性的相关性

**状态：** 指标接口已定义（stub）。正式实验需人工标注证据支持关系。

## 评估函数参考

| 指标 | 函数 (`src/evaluation.py`) | 状态 |
|------|--------------------------|------|
| 编辑距离 | `edit_distance()` | 已实现 |
| 词错误率 (WER) | `word_error_rate()` | 已实现 |
| 字符错误率 (CER) | `character_error_rate()` | 已实现 |
| 重叠路由准确率 | `evaluate_overlap_routing()` | 已实现 |
| 说话人归属准确率 | `speaker_attribution_accuracy()` | 已实现 |
| 证据支持 | `evaluate_evidence_support()` | Stub |
