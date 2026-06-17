# Step 4：faster-whisper ASR Baseline

## 当前选择

项目当前选择 `faster-whisper` 作为第一版真实 ASR baseline。其他 ASR 后端继续通过
`ASRAdapter` 接口保留，用于后续对比实验，但当前不要求安装。

默认配置：

| 配置 | 默认值 |
| --- | --- |
| 模型 | `small` |
| 设备 | `cpu` |
| 计算类型 | `int8` |
| beam size | `5` |
| temperature | `0.0` |
| condition on previous text | `False` |

## 安装

```bash
python -m pip install -r requirements.txt
```

模型权重会在第一次真实运行时下载。模型权重不提交到 Git。

## 运行完整 Pipeline

CLI 和 Gradio 默认使用 `faster-whisper`：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001
```

资源不足时可以使用更小模型：

```bash
python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --faster-whisper-model tiny
```

GPU 示例：

```bash
python main.py data/raw_audio/meeting_001.wav \
  --meeting-id meeting_001 \
  --asr-device cuda \
  --asr-compute-type float16
```

仅验证软件流程、不运行真实 ASR 时：

```bash
python main.py data/raw_audio/meeting_001.wav --meeting-id meeting_001 --asr mock
```

## 单独转写一个音频片段

```python
from src.asr.whisper_backend import transcribe_clip

result = transcribe_clip("data/raw_audio/meeting_001.wav", language=None)
print(result["text"])
print(result["asr_confidence"])
```

## Step 4 验收

- 真实会议音频能够完成转写，输出不是 `[mock transcript ...]`。
- 低重叠片段均包含 `text` 和 `[0, 1]` 范围内的 `asr_confidence`。
- 中文、英文和中英文混合音频运行时不崩溃。
- 明确选择真实后端但依赖、模型加载或推理失败时，程序给出错误，不静默切换到 Mock。
