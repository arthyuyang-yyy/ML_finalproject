# 系统架构

## 设计目标

系统架构将音频处理、不确定性表示、LLM 推理和记忆检索分离。所有后续结论都应能够追溯到原始音频片段。

## 数据流程

```text
音频
  -> 预处理
  -> 重叠检测
  -> 双路径路由
       -> 低重叠：说话人日志/聚类 + ASR
       -> 高重叠：语音分离和/或候选生成
  -> 统一元信息片段
  -> 不确定性感知 LLM 后处理
  -> Episodic Memory
  -> 检索与基于证据的问答
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `audio/preprocess.py` | 验证、归一化、导出并分段输入音频 |
| `overlap_detector.py` | 估计重叠分数并返回重叠区域 |
| `dual_path_router.py` | 选择低重叠或高重叠路径 |
| `asr.py` | 可插拔 ASR 适配器（mock/Whisper/Paraformer），带校准置信度 |
| `diarization.py` | 提供说话人标注接口 |
| `speech_separation.py` | 提供高重叠语音分离接口 |
| `candidate_generator.py` | 表示多个高重叠候选解释 |
| `metadata_builder.py` | 将输出统一为共享 Schema |
| `llm_postprocess.py` | 构建约束提示词和基于证据的输出 |
| `episodic_memory.py` | 创建、存储和检索会议 episode |
| `rag_qa.py` | 检索 episode 并使用证据回答 |
| `evaluation.py` | 定义超越 WER 与 DER 的指标 |

## 关键约定

- 分数范围统一为 `[0.0, 1.0]`。
- 时间为从会议音频开始计算的秒数。
- 高重叠记录必须保留候选列表和不确定性说明。
- 决策与行动项必须携带带时间戳的证据。
- 早期实验中的存储与检索后端应可替换。

## 计划实施阶段

1. 使用少量手工示例验证元信息与标注约定。
2. 添加基础重叠检测、ASR 和说话人日志适配器。
3. 实现候选生成与不确定性感知提示词。
4. 添加本地 episode 存储与检索。
5. 运行消融实验与证据质量评估。
