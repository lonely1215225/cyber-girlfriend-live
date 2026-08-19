# Cyber Girlfriend

数字人语音对话（AVTR-1 HTTP-FLV）。新机器解压后一键安装模型并启动。

## 机器要求

- Ubuntu 22.04+，root
- NVIDIA GPU（建议 Ampere 及以上，如 4090）+ 驱动
- 磁盘约 80GB（权重 + TensorRT + TTS + LLM）
- 能访问 Hugging Face、Ollama 模型库

若 AVTR-1 / TTS 下载被拦，先准备 `HF_TOKEN`。驱动报的 CUDA 不是 12.8 时，安装脚本会设置 `CONDA_OVERRIDE_CUDA=12.8`，无需改驱动。

## 一键部署

```bash
tar -xzf cyber-girlfriend-deploy.tar.gz
cd cyber-girlfriend
export PUBLIC_IP=你的公网IP          # 可选，不填则自动探测
export HF_TOKEN=hf_xxx               # 若下载被拦截再填
./install.sh                         # 下模型、编 TensorRT，耗时较长
./scripts/start.sh
```

浏览器打开 `https://公网IP:19800/`，信任自签证书，允许麦克风。

停止：`./scripts/stop.sh`  
状态：`./scripts/status.sh`

## 默认形象

设置页 / 右下角可选：小雅、海边近景、海边、白背心。图片在 `assets/looks/`。

## 目录

| 路径 | 说明 |
|------|------|
| `install.sh` | 安装依赖并下载模型 |
| `scripts/start.sh` | 启动 |
| `config.env` | 安装后生成，可改端口 / 人设 |
| `s2s/hf-realtime-voice/` | 网页 |
| `proxy/` | 网关、nginx、TTS 分流 |
| `third_party/avtr-1/` | 数字人渲染（权重安装时下载） |
| `models/` | TTS 等大模型（安装时下载） |

`logs/`、`run/`、`.cache/`、虚拟环境、模型权重和 AVTR-1 构建产物都属于本机运行数据，
不应提交到源码仓库。当前服务只使用 AVTR-1，不依赖 LiveTalking、Wav2Lip 或 MuseTalk。
