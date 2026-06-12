# 待实现任务清单

> 基于当前 `main`（含 PR #20）核对仓库实际代码后整理。系统框架与 Gradio
> 演示已基本完成，本清单聚焦**尚未实现 / 待补充**的部分，按对论文与答辩的
> 影响排优先级。每条都标注了对应的代码位置或 `TODO.md` 条目，便于认领。

## 完成度速览

| 创新点 | 状态 | 说明 |
| --- | --- | --- |
| 1. 重叠感知路由 | 🟡 框架完成 | detector + router 可运行，阈值未校准、未用人工标签验证 |
| 2. 不确定性感知高重叠处理 | 🟡 候选完成 | 多候选生成 done，可选语音分离未实现 |
| 3. 元信息感知 LLM 后处理 | 🔴 仅 fallback | 真实事件抽取与 evidence-only prompt 未实现 |
| 4. Episodic Memory | 🟢 基本完成 | 存储 + 混合检索 + 证据引用问答 |
| 5. 超越 WER/DER 的评估 | 🔴 部分实现 | 证据命中率 / 幻觉率 / 不确定性保留质量未实现 |

---

## P0 · 直接影响创新点与论文结论（必须补）

- [ ] **实现证据可追溯评估指标** —— `src/evaluation/core.py::evaluate_evidence_support`
      目前直接 `raise NotImplementedError`。需定义：证据命中率（evidence hit
      rate）、不被支持的主张检测、幻觉率、置信度校准。这是创新点 5 的核心，
      也是"可追溯"主张唯一的量化依据。
- [ ] **实现不确定性保留质量指标** —— 衡量高重叠候选是否被下游悄悄压成单一
      答案（对应创新点 2 的论点）。
- [ ] **真实 LLM 会议事件抽取** —— `TODO.md` Step 16。当前决策 / 行动项 /
      截止时间 / 开放问题 / 分歧 / 不确定性全部走确定性 fallback，未体现
      "LLM 推理"价值（创新点 3）。
- [ ] **完善 evidence-only / JSON-only prompt 约束** —— `TODO.md` Step 14，
      落实只基于证据作答、不确定性规则。
- [ ] **构建人工标注评估集** —— `TODO.md`。带时间戳、说话人标签、重叠标注；
      没有它，上面的指标和所有正式实验都无法跑。

## P1 · 实证收尾（拿到真实数字才有说服力）

- [ ] **重叠阈值校准** —— `TODO.md` Step 7b，对照人工标签扫描阈值并报告
      accuracy / precision / recall。
- [ ] **接入并运行真实重模型** —— WhisperX / Whisper / FunASR / pyannote /
      Gemma，给出与 mock 基线的真实效果对比。
- [ ] **Experiment 1 · 重叠路由阈值扫描** —— `TODO.md` Experiment 1。
- [ ] **Experiment 2 · 多候选 vs 强制单输出** —— `TODO.md` Experiment 2。
- [ ] **Experiment 3 · 元信息感知 LLM 消融** —— `TODO.md` Experiment 3。
- [ ] **Experiment 4 · Episodic Memory QA vs 摘要 / 全文 QA** —— `TODO.md`
      Experiment 4。

## P2 · 工程收尾（改动小但应完成）

- [ ] **校验音频片段路径存在性** —— `TODO.md` 第 39 行。确保每个输出的
      `audio_clip_path` 在磁盘上真实存在。
- [ ] **语音分离基线** —— `src/speech_separation.py::separate_speakers` 当前为
      `NotImplementedError`。要么接一个可替换的分离适配器（`TODO.md` Step 11），
      要么在论文中明确写为"已留接口、列为未来工作"。

---

## 备注

- 完成 P0 后即可支撑论文核心结论；P1 提供实验数字；P2 为工程收尾。
- 各条目状态以代码与 `TODO.md` 为准，完成后请同步勾选并更新 `TODO.md`。
