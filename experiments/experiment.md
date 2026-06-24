# 实验层总览（experiments/）

> **目的**：在 AliMeeting Eval_Ali_far 8 个远场会议（8 通道 / 16kHz / ~26 min）上跑 4D 矩阵，对比 ASR / OSD / LLM resolver / 分离 4 个技术栈的所有组合，把每个 cell 在每个 meeting 上的指标落地成 CSV。

---

## 1. 4D 实验矩阵

| 轴 | 取值 | 维度 |
|---|---|---|
| **A. ASR** | `faster-whisper` · `funasr` · `whisperx` | 3 |
| **B. OSD**（overlap/speaker） | `pyannote` · `energy_fallback` | 2 |
| **C. LLM resolver** | `none` · `openai`（→ DeepSeek） | 2 |
| **D. Speech separation** | `none` · `sepformer` | 2 |

**3 × 2 × 2 × 2 = 24 cells × 8 meetings = 192 runs**（再加 1 个 mock sanity cell）

实际有效的 meeting（Eval_Ali_far 各 mic 不同）：

| meeting_id | mic |
|---|---|
| R8001_M8004_MS801 | 801 |
| R8003_M8001_MS801 | 801 |
| R8007_M8010_MS803 | 803 |
| R8007_M8011_MS806 | 806 |
| R8008_M8013_MS807 | 807 |
| R8009_M8018_MS809 | 809 |
| R8009_M8019_MS810 | 810 |
| R8009_M8020_MS810 | 810 |

---

## 2. Pipeline 12 阶段概览

以 `R8001_M8004_MS801` 为例（26min 8ch → mono 16kHz）：

| # | 阶段 | 干的事 | 实际耗时（funasr/pyannote cell） |
|---|---|---|---|
| 01 | preprocess | 8ch→mono + 16kHz 重采样 + 可选降噪 | 2.2 s |
| 02 | vad | silero VAD 切出语音段 | 6.4 s |
| 03 | diarize | pyannote/segmentation 给 speaker turns | 25.5 s |
| 04 | overlap_score | 给每段算 overlap 分数（pyannote/energy） | 1.7 s |
| 05 | route_split | 分流 low_overlap/high_overlap，重叠处再切 | < 0.1 s |
| 06 | low_overlap_asr | ASR 转写低重叠段 | 62.2 s（funasr GPU）/ ~40 s（faster-whisper） |
| 07 | high_overlap | 高重叠段（可选 sepformer 拆）再 ASR | 71.8 s |
| 08 | resolve_llm | DeepSeek 在多候选中二选一 | 16.0 s |
| 09 | build_evidence | low + resolved-high 合成 344 段 evidence | < 0.1 s |
| 10 | clips_and_validate | 每段写 wav clip + schema 校验 | 13.5 s |
| 11 | event_extraction | DeepSeek 从 text 抽 events | 0.16 s |
| 12 | persist | 写所有 JSON + upsert long-term memory | 0.1 s |
| **合计** | | | **≈ 200 s**（GPU 修好前 1422 s） |

stage 时长写 `outputs/<meeting>/stage_timings.json`（每个 stage 完都 flush，崩了也能续）。

---

## 3. 关键修复（不修就跑不起来）

| # | 文件 | Bug | 修复 |
|---|---|---|---|
| F1 | `src/pipeline/run_pipeline.py` `_adapter_kwargs` | funasr 不接 `--asr-device`，永远 CPU | `device` 透传给 funasr（不算 `compute_type`） |
| F2 | `experiments/scripts/run_one.py` | 启动前没 `load_dotenv(REPO/.env)`，`HF_TOKEN` 被 `setdefault` 成空串 → pyannote 无 turns | 启动前 load `.env` 再 `os.environ.get("HF_TOKEN")` |
| F3 | `src/llm/backends.py` `OpenAIBackend` | DeepSeek v4 默认开 reasoning；用 `gemma3:4b`（Ollama 别名）传过去直接 400；`max_tokens` 默认 4K 被 reasoning 吃光 → content 0 token | model 自动按 base_url 选 `deepseek-v4-flash`；`max_tokens=65536`；支持 `DEEPSEEK_DISABLE_THINKING=1` 关 thinking |
| F4 | `.env` | `OPENAI_MODEL=deepseek-chat`（v4 账户不认） | 改 `deepseek-v4-flash` |
| F5 | `experiments/scripts/run_one.py` | `--gemma-backend openai` 时不传 model | 自动加 `--gemma-model deepseek-v4-flash` |
| F6 | `experiments/scripts/run_matrix.py` `DEFAULT_MEETINGS` | 全部硬编 MS801，但实际各 meeting mic 不同 | 用实际 mic（MS801/803/806/807/809/810） |
| F7 | `run_matrix.py` `audio_path_for` | 找不到 wav 直接 `SystemExit`，整个 matrix 死 | 返回 `None`，main loop skip+continue |
| F8 | `run_matrix.py` `is_complete` | 只看 `meta.exit_code==0`，脆弱 | 改成看 `outputs/<meeting>/{evidence_segments,meeting_events,episodic_memory}.json` 都存在 |
| F9 | `experiments/scripts/evaluate_runs.py` `_strip_mic_suffix` | 只剥 MS801/802/803 | 用正则 `_MS\d{3}$` 剥任何 mic 号 |
| F10 | `main.py` | 没配 logging，stage 进度看不到 | `_configure_logging()` 走 stderr |

---

## 4. 全流程（从零跑）

### 4.1 环境

```bash
cd /data_8t_1/wcj_2/meeting-memory-deploy
pip install soundfile faster-whisper funasr pyannote.audio
# .env 必须含 OPENAI_API_KEY（DeepSeek）+ HF_TOKEN（pyannote gated）
```

### 4.2 构建矩阵

```bash
python experiments/scripts/build_matrix.py    # 写 experiments/matrix.json
```

### 4.3 启动全量（后台）

```bash
nohup python -u experiments/scripts/run_matrix.py --gpu 0 --deepseek \
    > experiments/runs/_matrix.log 2>&1 &
```

会顺序跑 24 cells × 8 meetings。GPU 锁在 0，单卡独占。

### 4.4 监控

```bash
# 总进度
tail -f experiments/runs/_matrix.log

# 当前 run 的 stage
tail -f experiments/runs/<cell>/<meeting>/run.log

# 已完成数
grep -c "done" experiments/runs/_matrix.log

# GPU
watch -n 5 nvidia-smi
```

### 4.5 优雅停止 / 恢复

```bash
kill -INT <MATRIX_PID>   # 跑完当前 cell 后退出
kill <MATRIX_PID>         # 强制

# 恢复（自动跳过已完成）
nohup python -u experiments/scripts/run_matrix.py --gpu 0 --deepseek \
    > experiments/runs/_matrix.log 2>&1 &
```

### 4.6 评分 + 聚合

```bash
# 给所有 run 算 evaluation.json
python experiments/scripts/evaluate_runs.py --runs-root experiments/runs

# 重新算（参数改了，比如窗口大小）
python experiments/scripts/evaluate_runs.py --runs-root experiments/runs --force

# 出 CSV + Markdown
python experiments/scripts/aggregate.py
# → experiments/results/summary_per_cell_meeting.csv
# → experiments/results/meeting_difficulty.csv
# → experiments/results/summary.md
```

---

## 5. 文件布局

```
experiments/
├── README.md                       # 老入口
├── EVAL_PLAN.md                    # 评测方案
├── matrix.json                     # 24 cells 配置（build_matrix.py 生成）
├── experiment.md                   # 本文件
├── scripts/
│   ├── run_one.py                  # 单 cell × 单 meeting
│   ├── evaluate_runs.py            # 单 run 评分（生成 evaluation.json）
│   ├── aggregate.py                # 聚合 → CSV + MD
│   ├── build_matrix.py             # 枚举 4D 矩阵
│   └── run_matrix.py               # 全量调度（nohup 友好、断点续跑）
├── runs/
│   ├── _matrix.log                 # matrix runner 总日志
│   ├── _matrix_status.json         # 每个 run 的 rc / wall
│   └── <cell_id>/<meeting_id>/
│       ├── run_meta.json           # cell config + exit_code + wall
│       ├── run.log                 # stage 日志
│       ├── evaluation.json         # 评分（evaluator 生成）
│       └── outputs/<meeting_id>/
│           ├── preprocessed.wav
│           ├── stage_timings.json  # 每 stage 的秒数
│           ├── vad_segments.json
│           ├── diarization.json
│           ├── overlap.json
│           ├── routed_segments.json
│           ├── low_overlap_segments.json
│           ├── high_overlap_candidates.json
│           ├── evidence_segments.json
│           ├── meeting_events.json
│           ├── episodic_memory.json
│           └── clips/*.wav
└── results/
    ├── summary_per_cell_meeting.csv
    ├── meeting_difficulty.csv
    └── summary.md
```

---

## 6. 指标速查（详见 `summary_per_cell_meeting.csv` 表头）

| 块 | 字段 | 含义 | 越好 |
|---|---|---|---|
| CER | `cer_concat / cer_low / cer_high` | 字符错误率（ref vs hyp） | ↓ |
| CER | `wer_concat` | 词错误率 | ↓ |
| CER | `subs / ins / del / ref_chars / hyp_chars` | 编辑距离拆解 | — |
| Speaker | `spk_best_mapping_acc` | 枚举所有 1-to-1 配对后最佳 IoU | ↑ |
| Speaker | `spk_known_coverage` | GT speaker 在 hyp 出现的占比 | ↑ |
| Speaker | `spk_unknown_coverage` | 假阳性 speaker 占比 | ↓ |
| Routing | `routing_accuracy / f1` | low/high_overlap 分类 | ↑ |
| Overlap | `overlap_recall / precision / f1` | 重叠检测（hyp vs GT） | ↑ |
| Events | `events_count` | 抽出的事件数 | 看 meeting |
| Events | `llm_resolved_rate` | 多候选时 LLM 选了非 fallback | ↑ |
| Events | `fallback_resolved_rate` | 无候选走 fallback 的比例 | 看场景 |
| Timing | `wall_time_s / rtf` | 单 run 耗时 / < 1.0 算比实况快 | ↓ |

---

## 7. 当前进度（截至最近一次聚合）

- 8 cells 跑完（全部 `asr=faster-whisper`），56 / 192 runs
- per-cell × 8-meeting 平均 CER 在 0.39-0.46，spk 0.64-0.77，routing_f1 0.41-0.52
- **明显观察**：DeepSeek resolver 把 events 从 ~373 砍到 ~40-110（事件合并 / 去重）；sepformer 在 pyannote+resolver=none 时 spk +3%
- 16 cells 还在跑（funasr / whisperx × 8），估计剩余 12-14h

---

## 8. 已知问题 / 待办

| 问题 | 影响 | 状态 |
|---|---|---|
| WER 全 1.000 | evaluate_runs 用 [0,300s] window，ref_words 太小 saturate | 待修：去掉 window 限制 |
| whisperx cell 可能缺依赖（ctranslate2 版本冲突） | 该轴跑不起来 | 待验证 |
| energy_fallback 不真生效（main.py 没读 `MM_OSD_MODE`） | 该轴实际走的还是 pyannote | 待修 |
| mock cell 不在主实验矩阵里 | 单元测试用 | 已隔离 |
