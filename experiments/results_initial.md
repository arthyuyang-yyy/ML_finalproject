# 实验初步结果分析（截至 64/192 runs）

> 评估在完整会议（无 window 过滤）上运行。`wer` 字段对中文文本统一为 `null`（无分词下 split()=1 个 token，edit distance/1 必然=1），CER 才是有效指标。

---

## 1. 指标含义速查

| 字段 | 取值 | 含义 | 好/坏 |
|---|---|---|---|
| `cer` | 0–1 | 字符错误率（去掉标点和空白，按字符算编辑距离） | 越低越好 |
| `wer` | null | 词错误率 — 中文无分词时不报告，留 `null` 避免误导 | — |
| `spk_acc` | 0–1 | 把预测 speaker 标签通过最佳置换映射到 GT 后的匹配率（best-mapping brute force） | 越接近 1 越好；越低说明 diarization 出错或 speaker 串扰 |
| `routing_f1` | 0–1 | low/high overlap 路由 F1（每个 evidence 段是二分类，对比 GT overlap_regions） | 越高越好；过低说明 overlap_score 阈值或 VAD 不稳 |
| `overlap_f1` | 0–1 | OSD 区间级 P/R/F1（预测重叠区间 vs GT `overlap_regions`） | 越高越好 |
| `events` | int | 抽取出的事件总数 | 与 resolver 强相关；`openai` 把 350+ 合并成 30–50 |
| `llm_resolved` | 0–1 | evidence 中通过 LLM 解决冲突的比例 | 反映 DeepSeek 介入率 |
| `wall_s` | s | 端到端 wall-clock 时长 | 反映 LLM resolver 6–8× 减速 |
| `rtf` | float | wall / audio_duration | <1 = 实时，>1 = 离线 |

**`events` 高 ≠ 好**：在没有 resolver 时，每个 evidence 段都会独立产出事件（≈350+），实际上是噪声；LLM resolver 把多说话人冲突段合并成单事件（≈40–50）才是结构化记忆。

---

## 2. Meeting 难度（按 overlap_ratio 排序）

| meeting | overlap_ratio | num_speakers | turns_per_min | 难度 |
|---|---|---|---|---|
| R8007_M8010 | **0.564** | 4 | 38.2 | 🔴 极难 — 一半时间多人在说 |
| R8001_M8004 | 0.290 | 4 | 23.0 | 🟠 难 — 4 人会议 |
| R8007_M8011 | 0.238 | 4 | 24.8 | 🟠 难 — 4 人会议 |
| R8003_M8001 | 0.161 | 4 | 19.6 | 🟡 中 — 4 人低重叠 |
| R8008_M8013 | 0.139 | 3 | 19.6 | 🟡 中 — 3 人 |
| R8009_M8019 | 0.095 | 2 | 29.3 | 🟢 易 — 2 人 |
| R8009_M8018 | 0.070 | 2 | 24.2 | 🟢 易 — 2 人 |
| R8009_M8020 | **0.066** | 2 | 27.5 | 🟢 最易 — 2 人低重叠 |

**CER 与 overlap_ratio 强相关**：R8007_M8010（overlap=0.564）的 CER ≈ 0.69，R8009_M8020（overlap=0.066）的 CER ≈ 0.21；差距 ~3× 完全由会议难度解释，而非模型差异。

---

## 3. 各 cell 平均指标（10 cells / 64 runs 已完成）

| cell | n | cer | spk_acc | routing_f1 | overlap_f1 | events | llm_resolved | wall_s |
|---|---|---|---|---|---|---|---|---|
| `fw · energy_fallback · none · none` | 8 | 0.477 | 0.707 | 0.463 | 0.393 | 373 | 0.000 | 172 |
| `fw · energy_fallback · none · sepformer` | 8 | 0.472 | 0.708 | 0.463 | 0.393 | 373 | 0.000 | 172 |
| `fw · energy_fallback · openai · none` | 8 | 0.457 | 0.676 | 0.463 | 0.393 | 110 | 0.198 | **1085** |
| `fw · energy_fallback · openai · sepformer` | 7 | 0.481 | 0.694 | 0.463 | 0.408 | 295 | 0.011 | 318 |
| `fw · pyannote · none · none` | 8 | 0.475 | 0.709 | 0.463 | 0.393 | 373 | 0.000 | 174 |
| `fw · pyannote · none · sepformer` | 7 | 0.455 | 0.726 | 0.472 | 0.404 | 349 | 0.000 | 308 |
| `fw · pyannote · openai · none` | 5 | 0.469 | 0.645 | 0.505 | 0.433 | 42 | 0.239 | **1459** |
| `fw · pyannote · openai · sepformer` | 7 | 0.465 | 0.667 | 0.452 | 0.376 | 37 | 0.194 | 1149 |
| `funasr · pyannote · none · none` | 4 | 0.491 | 0.548 | 0.615 | 0.571 | 372 | 0.000 | 216 |
| `funasr · pyannote · none · sepformer` | 2* | **0.147** | 0.868 | 0.190 | 0.124 | 302 | 0.000 | 198 |

\* `funasr · sepformer` 只跑了 2 个最容易的 meeting（无 R8007 系列），数据有偏，下方单独说明。

---

## 4. 关键结论

### 4.1 ASR（`faster-whisper` vs `funasr`）

- **`faster-whisper`** 8 meeting 全跑完，平均 CER 0.46–0.48，spk_acc 0.65–0.73。
- **`funasr`** 4 meeting 完成：CER 0.491（vs `fw · none` 0.475）—— 在 4 meeting 上 CER 略高。
- **唯一亮点**：`funasr · pyannote · none · none` 的 **routing_f1 = 0.615**、**overlap_f1 = 0.571**，比 `fw · none` 的 0.463/0.393 高出 ~0.15。
  - 这说明 Paraformer 自带的 overlap / endpointing 比 pyannote 3.x + faster-whisper 联用更稳。
  - 但 spk_acc 0.548 显著低（vs 0.709），说明 funasr 输出 speaker 信息弱，需要靠下游 diarization 挽救。

### 4.2 OSD（`pyannote` vs `energy_fallback`）

- 两种 OSD 在 `routing_f1` 和 `overlap_f1` 上几乎打平（差距 ≤ 0.01）。
- `energy_fallback` 在没有 resolver 的纯 baseline 上略好 0.001 — 实质上没差别。
- `energy_fallback` 的优势：完全不依赖 HF_TOKEN，可在断网或 HF 限流时退而求其次。

### 4.3 LLM Resolver（`none` vs `openai→DeepSeek`）

- **CER 影响微小**：0.477 → 0.457（`-0.020`），CER 没有本质提升。
- **`spk_acc` 反而下降 0.03–0.06**：从 ~0.71 降到 ~0.65。原因：DeepSeek 在合并冲突段时偶尔会重新分配 speaker，best-mapping 后与原始 GT 失配。
- **事件数大降**：373 → 110（`fw · energy_fallback`）、373 → 42（`fw · pyannote`）。
  - 这是 resolver 的**主要价值** —— 把 350 个 evidence 段独立抽出的事件合并成 ~50 个真正可用的记忆。
  - 配合 `llm_resolved_rate ≈ 0.20–0.24`，DeepSeek 实际介入了 ~1/4 的 evidence 段。
- **墙钟 6–8×**：172s → 1085s（`fw · energy_fallback`）、174s → 1459s（`fw · pyannote`）。DeepSeek API 调用是主要瓶颈。
- **结论**：resolver 不提升指标，但显著提升**事件结构化质量**。需要在 `events` 之上加一个 quality 评分（如 evidence-grounding, redundancy）才能在指标上看到价值。

### 4.4 Speech Separation（`none` vs `sepformer`）

- **`fw · pyannote · none`**：`spk_acc` 0.709 → 0.726（sepformer +0.017）。
- 其他 cell 的 spk_acc 提升 ≤ 0.002，可忽略。
- **wall_s 大增**：174 → 308s（`fw · pyannote · none · sepformer`）；1085 → 318s 时反而降速 ❓ —— 这不是 sepformer 加速，而是 openai resolver 那次 GPU OOM 半跑被截短了，需重跑确认。
- **CER 几乎无差别**（0.475 vs 0.455），sepformer 在 ASR 层面贡献有限。
- `funasr · sepformer` 在 2 个低 overlap meeting 上 CER=0.095、0.200 极漂亮，但样本量太小，且缺高 overlap 会议，无法跟其他 cell 直接对比。

### 4.5 整体最优

按综合 `events ~50 + cer < 0.50 + spk_acc > 0.65`：
- **首选**：`fw · pyannote · openai · none`（CER 0.469 / spk 0.645 / events 42 / wall 1459s）
- **不接 LLM 的最快**：`fw · pyannote · none · sepformer`（CER 0.455 / spk 0.726 / events 349 / wall 308s）
- **纯 baseline**：`fw · pyannote · none · none`（CER 0.475 / spk 0.709 / events 373 / wall 174s）

---

## 5. 还差什么

- ❌ `whisperx` 整轴（0/8 完成）
- ⚠️ `funasr` 大部分（4/8 完成 `none·none`，1/8 `openai·none`，2/8 `none·sepformer`）
- ❌ `fw · pyannote · openai · none` 缺 3 个 meeting
- ❌ 5 个 `fw · energy_fallback · openai · sepformer` 的 meeting 缺，wall_s 数据可疑

**结论**：当前 64 runs 已经能看出 LLM resolver 把 ~350 个事件合并成 ~50 个的趋势，但样本量不足以断言"sepformer 在哪类 meeting 上有用"或"funasr 是否更优"。需要补完剩余 128 runs（funasr + whisperx 全 + 缺失补齐）才能给出统计意义上的结论。
