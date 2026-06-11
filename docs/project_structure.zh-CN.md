# 项目结构与扩展约定

本仓库按“音频处理 -> 重叠路由 -> 证据构建 -> 事件抽取 -> 情景记忆 -> 检索问答”组织。重模型均通过后端适配器延迟加载，默认测试和基础流水线不下载模型。

## 目录职责

```text
src/
├── audio/          # 预处理、VAD、音频片段导出
├── overlap/        # 重叠检测与双路径路由
├── asr/            # WhisperX、Whisper、FunASR/SenseVoice 兼容后端
├── diarization/    # pyannote 后端与 speaker assignment
├── candidates/     # 高重叠多候选与可选语音分离
├── evidence/       # 统一 EvidenceSegment schema、构建与校验
├── llm/            # Gemma 接口、prompt、JSON 修复、事件抽取
├── memory/         # Episode schema、事件转记忆、存储与检索
├── qa/             # 仅基于检索结果的证据问答
├── evaluation/     # 路由、证据与 QA 指标
├── pipeline/       # 端到端编排、配置与 artifact I/O
└── ui/             # Gradio 演示
```

## 兼容约定

- 公共入口优先从包导入，例如 `from src.asr import get_adapter`。
- 后端实现放在对应子包，流水线不直接依赖具体模型库。
- `evidence_segments.json` 是 LLM、Memory 和 QA 的唯一事实输入。
- 高重叠片段保持 `speaker="MIXED"`、主文本为空，并保留候选和不确定性说明。
- 每个 Episode 必须保留 `evidence_ids`、时间范围和音频片段路径。

## 扩展顺序

1. 在 `src/asr/` 或 `src/diarization/` 增加新后端适配器。
2. 保持后端输出符合现有字典结构，不修改下游 Evidence Schema。
3. 在 `PipelineConfig` 中增加可选配置，并在流水线编排层注入适配器。
4. 为新增入口添加单元测试；重模型测试应标记为可选集成测试。
