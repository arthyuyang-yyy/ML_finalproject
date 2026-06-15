# Step 11：高重叠语音分离 Baseline

## 可选后端

语音分离作为可替换适配器（`SpeechSeparationAdapter`）接入，提供两个真实后端：

| 后端 | 选项 | 依赖 | 定位 |
| --- | --- | --- | --- |
| **NMF**（自写 numpy 基线） | `--speech-separation nmf` | 无 | 零依赖、确定性、可解释的对比基线；CI 默认可端到端跑通 |
| **SepFormer**（SpeechBrain） | `--speech-separation sepformer` | `speechbrain` + PyTorch | 重模型升级，质量更高，需下载 `speechbrain/sepformer-whamr16k` |

NMF 在幅度谱上做非负矩阵分解 `V≈W·H`，把基向量聚成 N 路说话人，用软 Wiener 掩码 +
混音相位重建每路源（实现于 `src/nmf_separation.py`，与项目自写 AHC 聚类一脉相承）。它靠
声源的**时间激活差异**分离，对稳态/频谱完全相同的语音本就分不开——这是单通道方法的
固有边界，因此下游仍按「高重叠 = 不确定」处理。

Pipeline 的输入已经统一为 16 kHz，因此不需要在默认路径额外重采样。语音分离默认关闭。
启用后只处理被路由为 `high_overlap_candidate` 的片段：

```text
高重叠混合音频
  -> NMF / SepFormer 分离声源
  -> 每条声源分别执行 faster-whisper
  -> 保存为多个高重叠 candidates
```

分离声源的顺序不代表已确认的说话人身份，因此候选始终使用
`SEPARATED_SOURCE_01`、`SEPARATED_SOURCE_02` 等临时标签。只有后续增加可靠的
声纹匹配后，才能将这些声源绑定到 diarization speaker。

## 安装与运行

NMF 后端零依赖，无需额外安装：

```bash
python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --speech-separation nmf
```

SepFormer 后端需要先装重依赖：

```bash
python -m pip install -r requirements-separation.txt

python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --speech-separation sepformer
```

SepFormer 首次运行会下载模型到用户缓存目录。可以通过
`--sepformer-model` 和 `--speech-separation-device` 更换模型或设备。

## 回退行为

以下情况不会阻断整个会议 Pipeline：

- 未启用语音分离；
- SepFormer 的 SpeechBrain、PyTorch 或模型不可用；
- 模型推理失败（NMF 或 SepFormer）；
- 分轨成功但 ASR 没有生成有效文本。

发生这些情况时，高重叠路径继续使用已有的 faster-whisper 多参数解码候选；如果
faster-whisper 也不可用，则继续输出显式 fallback candidates。

## 后续实验

Phase 4 需要在真实人工标注会议上比较：

- NMF、SepFormer 分轨候选与直接 multi-decode 候选三路的 Oracle/Top-1 WER；
- 分轨对说话人归属和候选有效性的影响；
- 零依赖 NMF 与 SepFormer/MossFormer2 等重模型的质量与运行成本权衡。
