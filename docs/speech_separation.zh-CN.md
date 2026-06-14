# Step 11：高重叠语音分离 Baseline

## 当前选择

项目第一版语音分离 baseline 使用 SpeechBrain 的
`speechbrain/sepformer-whamr16k`。Pipeline 的输入已经统一为 16 kHz，因此不需要在
默认路径额外重采样。

语音分离默认关闭。启用后只处理被路由为 `high_overlap_candidate` 的片段：

```text
高重叠混合音频
  -> SepFormer 分离声源
  -> 每条声源分别执行 faster-whisper
  -> 保存为多个高重叠 candidates
```

分离声源的顺序不代表已确认的说话人身份，因此候选始终使用
`SEPARATED_SOURCE_01`、`SEPARATED_SOURCE_02` 等临时标签。只有后续增加可靠的
声纹匹配后，才能将这些声源绑定到 diarization speaker。

## 安装与运行

```bash
python -m pip install -r requirements-separation.txt

python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --speech-separation sepformer
```

首次运行会下载模型到用户缓存目录。可以通过
`--sepformer-model` 和 `--speech-separation-device` 更换模型或设备。

## 回退行为

以下情况不会阻断整个会议 Pipeline：

- 未启用语音分离；
- SpeechBrain、PyTorch 或模型不可用；
- 模型推理失败；
- 分轨成功但 ASR 没有生成有效文本。

发生这些情况时，高重叠路径继续使用已有的 faster-whisper 多参数解码候选；如果
faster-whisper 也不可用，则继续输出显式 fallback candidates。

## 后续实验

Phase 4 需要在真实人工标注会议上比较：

- SepFormer 分轨候选与直接 multi-decode 候选的 Oracle/Top-1 WER；
- 分轨对说话人归属和候选有效性的影响；
- SepFormer 与 MossFormer2 等其他模型的质量和运行成本。
