# 云服务器部署与启动文档

本文档用于把项目打包到云服务器做测试。默认以 Linux 服务器、Python 3.11、CPU 推理为基线；如果服务器有 NVIDIA GPU，可在安装 PyTorch / ASR 后端时改用 CUDA 版本。

## 1. 部署内容

需要上传到服务器的内容：

- `src/`：核心处理流程；
- `app.py`、`streamlit_app.py`、`main.py`：Web demo 和命令行入口；
- `requirements.txt`：Python 依赖；
- `.env.example`：环境变量模板；
- `docs/`、`data/README.md`、`tests/`：说明、数据布局和验证用例；
- `Dockerfile`、`.dockerignore`：可选容器部署文件。

不要随项目包上传的内容：

- `.env`：包含 token 和密钥，服务器上单独创建；
- `outputs/`：本地实验输出，体积可能很大；
- `data/raw_audio/`、`data/processed_audio/`、`data/external/`：原始音频和数据集；
- `models/`、`*.pt`、`*.bin`、`*.safetensors`、`*.pth`：模型权重；
- `.venv/`、`__pycache__/`、`.pytest_cache/` 等本地缓存。

项目已提供 `.gitignore` 和 `.dockerignore`，用于避免把这些大文件带进 Git 或 Docker 镜像。

## 2. 推荐服务器规格

最小测试：

- CPU：4 核及以上；
- 内存：8 GB 及以上；
- 磁盘：30 GB 及以上；
- 系统：Ubuntu 22.04 / 24.04；
- Python：3.11。

较完整测试：

- CPU：8 核及以上；
- 内存：16 GB 及以上；
- 磁盘：80 GB 及以上；
- GPU：可选。跑 Whisper large、WhisperX、pyannote、SepFormer 或本地 Transformers LLM 时建议使用 NVIDIA GPU。

## 3. 系统依赖

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip ffmpeg libsndfile1 git build-essential
```

如果系统没有 `python3.11` 包，可以使用系统默认 Python 3.10+，但建议保持服务器和本地开发版本一致。

## 4. 上传项目

方式一：用 Git 拉取。

```bash
cd /srv
git clone <your-repo-url> meeting-memory
cd /srv/meeting-memory
```

方式二：从本机打包上传。

```bash
cd /Users/lymn/Downloads/Project
tar \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='outputs/*' \
  --exclude='data/raw_audio/*' \
  --exclude='data/processed_audio/*' \
  --exclude='data/external' \
  --exclude='models' \
  --exclude='*.wav' \
  --exclude='*.mp3' \
  --exclude='*.flac' \
  --exclude='*.m4a' \
  --exclude='*.pt' \
  --exclude='*.bin' \
  --exclude='*.safetensors' \
  -czf meeting-memory-deploy.tar.gz .
```

上传后在服务器解压：

```bash
sudo mkdir -p /srv/meeting-memory
sudo tar -xzf meeting-memory-deploy.tar.gz -C /srv/meeting-memory
cd /srv/meeting-memory
```

## 5. 创建运行环境

```bash
cd /srv/meeting-memory
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果服务器在国内网络环境，建议提前配置代理或镜像源。Hugging Face、Whisper、pyannote、SpeechBrain 相关模型可能需要外网访问。

## 6. 配置环境变量

```bash
cd /srv/meeting-memory
cp .env.example .env
nano .env
```

常用配置：

```bash
MEETING_DATA_ROOT=/srv/meeting-memory-data
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
HF_TOKEN=hf_xxx
GEMMA_BACKEND=none
```

说明：

- `GRADIO_SERVER_NAME=0.0.0.0` 用于让云服务器外部访问 Gradio；
- `HF_TOKEN` 只在启用 pyannote、部分 Hugging Face 模型或受限模型时需要；
- `GEMMA_BACKEND=none` 表示先用确定性 fallback，不依赖 LLM；
- 使用 Ollama 时设置 `GEMMA_BACKEND=ollama`、`GEMMA_BASE_URL=http://127.0.0.1:11434`、`GEMMA_MODEL=gemma3:4b`；
- 使用 DeepSeek 时设置 `GEMMA_BACKEND=deepseek` 和 `DEEPSEEK_API_KEY`；
- 使用 OpenAI 兼容接口时设置 `GEMMA_BACKEND=openai`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。

## 7. 预下载模型

先做最小测试时可以跳过本节，用 `--asr mock` 验证流程。

需要真实 ASR / diarization / overlap 后端时运行：

```bash
source .venv/bin/activate
python scripts/setup_models.py
```

注意：

- pyannote 模型需要 Hugging Face token，并且账号需要在模型页面接受使用条款；
- FunASR 模型通常从 ModelScope 下载；
- faster-whisper 模型默认使用 `small`，缓存到用户目录；
- 模型权重不要放进 Git，也不要打进 Docker 镜像，建议放在服务器缓存或独立数据盘。

## 8. 启动方式

### 8.1 命令行跑单个音频

最小连通性测试：

```bash
source .venv/bin/activate
python main.py /path/to/meeting.wav --meeting-id deploy_smoke --asr mock --language zh
```

中文会议真实 ASR 测试：

```bash
python main.py /path/to/meeting.wav \
  --meeting-id deploy_funasr_test \
  --asr funasr \
  --language zh \
  --gemma-backend none
```

使用 faster-whisper：

```bash
python main.py /path/to/meeting.wav \
  --meeting-id deploy_fw_test \
  --asr faster-whisper \
  --faster-whisper-model small \
  --asr-device cpu \
  --asr-compute-type int8 \
  --language zh
```

输出在：

```text
outputs/{meeting_id}/
```

### 8.2 启动 Gradio 页面

```bash
source .venv/bin/activate
python app.py
```

默认读取 `.env`：

- 地址：`0.0.0.0`
- 端口：`7860`

浏览器访问：

```text
http://<服务器公网 IP>:7860
```

云厂商安全组和服务器防火墙需要放行 7860 端口。公开暴露测试页面有风险，推荐只对自己的 IP 放行，或使用 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

如果使用 SSH 隧道，可把 `.env` 里的 `GRADIO_SERVER_NAME` 改回 `127.0.0.1`。

### 8.3 启动 Streamlit 页面

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

浏览器访问：

```text
http://<服务器公网 IP>:8501
```

## 9. 后台运行

临时测试可用 `nohup`：

```bash
cd /srv/meeting-memory
source .venv/bin/activate
nohup python app.py > gradio.log 2>&1 &
```

查看日志：

```bash
tail -f gradio.log
```

长期测试建议用 systemd。创建 `/etc/systemd/system/meeting-memory-gradio.service`：

```ini
[Unit]
Description=Meeting Memory Gradio Demo
After=network.target

[Service]
Type=simple
WorkingDirectory=/srv/meeting-memory
EnvironmentFile=/srv/meeting-memory/.env
ExecStart=/srv/meeting-memory/.venv/bin/python /srv/meeting-memory/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now meeting-memory-gradio
sudo systemctl status meeting-memory-gradio
```

日志：

```bash
journalctl -u meeting-memory-gradio -f
```

## 10. Docker 部署

构建镜像：

```bash
cd /srv/meeting-memory
docker build -t meeting-memory:latest .
```

启动 Gradio：

```bash
docker run --rm -it \
  --env-file .env \
  -p 7860:7860 \
  -v /srv/meeting-memory-data:/srv/meeting-memory-data \
  -v /srv/meeting-memory/outputs:/app/outputs \
  meeting-memory:latest
```

启动 Streamlit：

```bash
docker run --rm -it \
  --env-file .env \
  -p 8501:8501 \
  -v /srv/meeting-memory-data:/srv/meeting-memory-data \
  -v /srv/meeting-memory/outputs:/app/outputs \
  meeting-memory:latest \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Docker 镜像不包含音频数据和模型权重。首次运行真实后端时，模型仍会下载到容器内；正式测试建议把模型缓存目录也挂载出来。

## 11. 验证清单

部署完成后按顺序检查：

```bash
source .venv/bin/activate
python -m pytest -q
python main.py /path/to/short.wav --meeting-id deploy_smoke --asr mock --language zh
python app.py
```

成功标志：

- 测试通过或只剩与本机缺少重模型相关的跳过项；
- `outputs/deploy_smoke/` 下生成 `evidence_segments.json`、`meeting_events.json`、`episodic_memory.json`；
- Gradio 页面可上传音频并返回时间线、候选片段、会议记忆和 QA 结果。

## 12. 常见问题

端口打不开：

- 检查 `.env` 是否设置 `GRADIO_SERVER_NAME=0.0.0.0`；
- 检查云厂商安全组和系统防火墙；
- 检查服务日志里实际监听端口。

模型下载失败：

- 检查代理、DNS 和服务器出口网络；
- Hugging Face 受限模型需要 `HF_TOKEN` 和网页上接受条款；
- 国内环境下 FunASR / ModelScope 通常比 Hugging Face 更稳。

内存不够：

- 先用 `--asr mock` 或 `--asr faster-whisper --faster-whisper-model small --asr-compute-type int8`；
- 关闭 `sepformer` 和本地 Transformers LLM；
- 缩短测试音频，先用 1 到 3 分钟片段验证流程。

ASR 输出为空或质量差：

- 确认音频可被 `ffmpeg` 读取；
- 中文会议优先试 `funasr`；
- 英文或多语言可试 `faster-whisper`；
- 高重叠严重时可尝试调低 `suspected_overlap_threshold` 或启用候选路径，但会增加耗时。

LLM 不可用：

- `GEMMA_BACKEND=none` 时系统会使用确定性 fallback，pipeline 仍可运行；
- 使用 Ollama 时先确认 `ollama serve` 正常，模型已 `ollama pull`；
- 使用 DeepSeek/OpenAI 时确认 API key、base URL 和服务器网络。
