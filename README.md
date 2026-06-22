# Wakord

> Create your own wake words.

Train personalized wake words locally with just a few recordings.

---

## 为什么 openWakeWord 能做中文？会不会掉精度？

openWakeWord 的流水线是：

```
原始音频 → melspectrogram 模型 → Google speech-embedding（96维特征） → 你训练的小分类器
```

前两步（特征提取器）是**语言无关**的通用语音表征模型，中文同样适用。所谓"只支持英文"指的是它**官方发布的成品模型**是英文词，而不是框架本身。

本项目走的是 openWakeWord **官方推荐的自定义训练路线**：

1. 用中文 TTS（`sherpa-onnx` 的 `vits-zh-aishell3`）合成上千条「小元」，并**随机说话人 + 随机语速**制造多样性；
2. 用 openWakeWord 的特征提取器把音频转成 embedding；
3. 在 embedding 上训练一个小 CNN 二分类器。

因此中文唤醒词与英文走的是**同一套机制**，精度取决于样本质量/数量与负样本覆盖，而非语言本身 —— 不存在"为支持中文而损失精度"的问题。

---

## 目录结构

```
WakeUp_Project/
├── configs/config.yaml         # 可覆盖的配置（只写想改的项）
├── models/                     # 部署用成品模型 xiaoyuan.onnx（gitignore，建议走 Release）
├── src/wakeup/
│   ├── config.py               # 集中配置 + 派生路径
│   ├── cli.py                  # 命令行入口（wakeup ...）
│   ├── data/                   # 数据准备
│   │   ├── tts_generator.py    #   中文 TTS 合成正样本
│   │   ├── negatives.py        #   下载负样本特征
│   │   └── features.py         #   正样本特征提取
│   ├── training/               # 训练与导出
│   │   ├── model.py            #   CNN 分类器
│   │   ├── dataset.py          #   特征对齐 / 切分
│   │   ├── trainer.py          #   训练循环
│   │   ├── export.py           #   ONNX / TensorFlow 导出
│   │   └── pipeline.py         #   端到端编排
│   └── service/                # 常驻监听服务
│       ├── audio.py            #   麦克风采集
│       ├── vad.py              #   人声门控（省电）
│       ├── detector.py         #   VAD + 唤醒推理 + 冷却去重
│       ├── server.py           #   可被控制的 TCP 服务
│       ├── client.py           #   控制客户端
│       └── protocol.py         #   控制协议
├── examples/control_service.py # 用别的程序控制服务的示例
└── tests/                      # 轻量单元测试
```

---

## 安装（Anaconda）

```powershell
# 1. 创建并激活环境
conda env create -f environment.yml
conda activate wakeup

# 2. 以可编辑方式安装本项目（让 `wakeup` 命令可用）
pip install -e .
```

> GPU 训练（可选）：`environment.yml` 默认装 CPU 版 PyTorch，训练这个小模型足够。
> 想用 GPU 请按 <https://pytorch.org> 单独安装对应 CUDA 版本的 torch。

---

## 用法

### 1）训练模型

```powershell
wakeup train                 # 完整流程：合成正样本 → 特征 → 训练 → 导出 ONNX
wakeup train --skip-tts      # 已有正样本时跳过 TTS
wakeup train --export-tf     # 额外导出 TensorFlow（需 pip install -r requirements-export.txt）
```

产物：`models/xiaoyuan.onnx`（服务默认从这里加载）。

### 2）现场调阈值（不起服务，前台直跑）

```powershell
wakeup listen --show-score   # 对麦克风说「小元」，观察分数，决定 config 里的 threshold
```

### 3）启动常驻服务

```powershell
wakeup serve                 # 启动后默认不监听，等待外部 start 指令
wakeup serve --listen        # 启动后立即开始监听
```

### 4）用命令行控制服务

```powershell
wakeup ctl start             # 开始监听
wakeup ctl stop              # 停止监听（释放麦克风、省电）
wakeup ctl status            # 查询状态
wakeup ctl shutdown          # 关闭服务
wakeup events                # 持续打印唤醒事件
```

### 5）用你自己的程序控制（核心场景）

见 [`examples/control_service.py`](examples/control_service.py)：

```python
from wakeup.service.client import ServiceClient

with ServiceClient(host="127.0.0.1", port=8765) as c:
    c.start()                       # 开始监听
    for msg in c.messages():        # 接收事件
        if msg["type"] == "wake":
            print("唤醒!", msg["score"])
```

---

## 控制协议（任何语言都能集成）

本机 TCP（默认 `127.0.0.1:8765`），**按行收发 JSON**：

| 方向 | 消息 | 说明 |
|------|------|------|
| 客户端→服务 | `{"cmd":"start"}` | 开始监听 |
| 客户端→服务 | `{"cmd":"stop"}` | 停止监听（释放麦克风）|
| 客户端→服务 | `{"cmd":"status"}` | 查询状态 |
| 客户端→服务 | `{"cmd":"shutdown"}` | 关闭服务 |
| 服务→客户端 | `{"type":"wake","model":"xiaoyuan","score":0.97,"ts":...}` | **唤醒事件（广播给所有连接）** |
| 服务→客户端 | `{"type":"status","listening":true,...}` | 状态 |
| 服务→客户端 | `{"type":"ack","cmd":"...","ok":true}` | 命令确认 |

例如在任意语言里：连上 TCP → 发 `{"cmd":"start"}\n` → 逐行读取，遇到 `"type":"wake"` 即表示听到了唤醒词。

---

## 省电是怎么做到的

1. **VAD 门控**：每帧先用 WebRTC VAD（极廉价）判断有没有人声；**静默时根本不调用唤醒词模型**，CPU 几乎空转。
2. **前导回灌**：人声一开始就把前 ~1.3s 的缓冲帧回灌给模型补足上下文，保证「小元」这种短词也能被完整识别。
3. **停止即释放**：`stop` 后后台线程关闭麦克风音频流并阻塞等待，不占设备、不耗电；`start` 时再开启。

相关参数都在 `configs/config.yaml` 的 `service` 段（`vad_backend` / `vad_aggressiveness` / `hangover_frames` / `preroll_frames` / `threshold` / `cooldown_seconds`）。

---

## 提交到 GitHub

```powershell
git init
git add .
git commit -m "init: 中文语音唤醒（小元）训练与监听服务"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`artifacts/`、下载的大文件与训练好的模型已在 `.gitignore` 中排除。成品模型（`xiaoyuan.onnx`）建议用 **GitHub Release** 或 **Git LFS** 分发。

---

## 开发

```powershell
pip install pytest
pytest                       # 跑轻量单元测试（不需要音频/模型）
```
