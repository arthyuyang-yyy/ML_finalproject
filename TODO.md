# TODO

本文件只跟踪当前工程状态和下一步任务。完整项目说明见 `README.md`、`Project_task.md` 和 `docs/system_architecture.md`。

## 已完成

- [x] 单一依赖文件：`requirements.txt`
- [x] 精简说明文档：保留 `README.md`、`Project_task.md`、`TODO.md`
- [x] 音频预处理：单声道、16 kHz 重采样、归一化
- [x] Energy-based VAD 分段
- [x] 音频 clip 导出
- [x] pyannote diarization / overlap adapter
- [x] overlap score 估计与路由
- [x] 低重叠 ASR + speaker attribution
- [x] 高重叠候选生成
- [x] Speech separation adapter 保留为可选候选增强，默认关闭
- [x] 高重叠 resolver：LLM 可用时用 LLM，无 LLM 时 fallback 到最高置信候选
- [x] Evidence segment schema、builder、validator
- [x] Meeting event extraction 和 fallback
- [x] Episodic Memory 构建与长期存储
- [x] 自定义 BLAKE2 字符 n-gram hash embedding 检索
- [x] Evidence-backed QA
- [x] Gradio demo
- [x] 自动化测试和 ruff 检查

## 当前验证状态

```text
python3 -m pytest -q
491 passed, 6 skipped, 2 warnings, 7 subtests passed

python3 -m ruff check src tests main.py app.py
All checks passed
```

## 待做：高优先级

- [ ] 准备 1-3 段可公开演示的真实或合成会议音频。
- [ ] 为这些音频手工标注 speaker、overlap、transcript 和关键事件。
- [ ] 用真实音频跑通 `--asr faster-whisper` 或 `--asr whisperx`。
- [ ] 用本地 Ollama Gemma 跑通高重叠 resolver 和事件抽取。
- [ ] 对比高重叠 resolver 与“直接选择最高 ASR 候选”的效果。

## 待做：中优先级

- [ ] 校准默认 overlap threshold `0.4`。
- [ ] 补一份小型演示数据的运行结果截图或输出样例。
- [ ] 明确哪些 optional heavy backend 是最终展示必须安装的。
- [ ] 有人工评估集后，重新评估是否恢复 recency、importance、overlap penalty 等检索排序信号。
- [ ] 清理或归档暂时不用的数据集脚本，如果最终展示不需要它们。

## 暂不优先

- [ ] 基于真实音频评估 speech separation 可选后端的收益。
- [ ] 大规模数据集实验。
- [ ] 复杂 evaluation 指标体系。
- [ ] 新增更多文档文件。
- [ ] 新增插件系统或复杂配置系统。
