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
│   │   ├── tts_generator.py    #   中文 TTS 合成正样本（sherpa-onnx，174 说话人）
│   │   ├── tts_edge.py         #   Edge TTS 多音色合成（扩充正样本多样性，可选）
│   │   ├── recorder.py         #   麦克风录制真实样本（few-shot 个性化，可选）
│   │   ├── augment.py          #   音频层增强（噪声/混响/增益/麦克风频响，缩小域差距）
│   │   ├── oww_assets.py       #   openWakeWord 特征/VAD 模型下载与缓存
│   │   ├── negatives.py        #   下载负样本特征
│   │   └── features.py         #   正样本特征提取
│   ├── training/               # 训练与导出
│   │   ├── model.py            #   CNN 分类器
│   │   ├── dataset.py          #   特征对齐 / 切分
│   │   ├── trainer.py          #   训练循环
│   │   ├── export.py           #   ONNX / TensorFlow 导出
│   │   └── pipeline.py         #   两阶段编排：prepare（数据+特征缓存）/ fit（训练+导出）
│   └── service/                # 常驻监听服务
│       ├── audio.py            #   麦克风采集
│       ├── vad.py              #   人声检测（仅用于 listen --debug 诊断）
│       ├── detector.py         #   持续推理 + 阈值 + 冷却去重（每帧跑模型保持缓冲预热）
│       ├── server.py           #   可被控制的 WebSocket 服务
│       ├── ws_client.py        #   WebSocket 控制客户端
│       └── protocol.py         #   控制协议
├── examples/control_service.py # 用别的程序控制服务的示例
└── tests/                      # 轻量单元测试
```

---

## 安装（推荐 uv）

```powershell
# 1. 创建可复现虚拟环境并安装运行+训练+测试依赖
uv sync --extra all

# 2. 使用项目命令
uv run wakeup --help
uv run wakeup prepare
uv run wakeup fit
```

`uv.lock` 会锁住依赖解析结果；换机器时优先用 `uv sync --frozen --extra all`。成品模型
`models/xiaoyuan.onnx` 不入库，发布时放到 GitHub Release，下载后放回同名路径即可。

## 安装（Anaconda，可选）

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

### 1）训练模型（全程自动合成语音，**无需录音**）

训练分成**两个阶段**，避免每次调参都重跑耗时的数据准备：

```powershell
# 阶段一【慢】：合成样本 → 下载负样本 → 提取并缓存所有特征（样本变化时才需重跑）
wakeup prepare                 # 自动合成上千条「小元」并缓存特征
wakeup prepare --gen-voices    # 额外用 Edge TTS 多音色扩充（需联网）

# 阶段二【快】：从缓存特征训练 + 导出 ONNX（调参时反复跑这个，秒级起步）
wakeup fit                     # 用缓存特征训练并导出 models/xiaoyuan.onnx
wakeup fit --epochs 10         # 快速试跑（临时少跑几轮）
wakeup fit --export-tf         # 额外导出 TensorFlow（需 pip install -r requirements-export.txt）
```

一条龙（= prepare + fit，首次最省心）：

```powershell
wakeup train                 # 等价于 prepare + fit
wakeup train --gen-voices    # 一条龙 + Edge TTS 多音色扩充
```

产物：`models/xiaoyuan.onnx`（服务默认从这里加载）。

> **「小元」语音是自动合成的，你不需要自己录制。** `prepare` 会用中文 TTS
> （sherpa-onnx `vits-zh-aishell3`，174 个说话人 + 随机语速）自动合成上千条「小元」
> 正样本。加 `--gen-voices` 还会用 Edge TTS 的几十种神经音色（普通话/台普/方言、男女
> 老少）跨引擎扩充，进一步提升对真实嗓音的泛化。
>
> 录音（下文「可选」一节）只是**锦上添花的 few-shot 个性化**，用于把召回率进一步拉高到
> 你本人的嗓音/麦克风上 —— 完全可以不录，直接训练即可得到可用模型。

**特征缓存机制**：`prepare` 把正样本/Edge/录音特征都缓存成 `.npy`，`fit` 直接读缓存，
所以调超参（epochs / 阈值 / neg_pos_ratio 等）只需重跑 `fit`，**跳过全部合成与特征提取**。
新增了录音或音色后，用 `wakeup prepare --force-features` 重建特征缓存。

### 1.5）可选：扩充多样性 / 个性化

```powershell
wakeup gen-voices            # 单独跑 Edge TTS 多音色合成（等价于 prepare --gen-voices 的合成步）
wakeup gen-voices --count 60 # 只合成 60 条（弱网/想快时）
wakeup record --count 30     # 可选：用麦克风录 30 条真实「小元」，混入训练提升对你本人的召回
```

新增样本后，跑 `wakeup prepare --force-features` 重建特征缓存，再 `wakeup fit` 即可。

### 2）现场调阈值（不起服务，前台直跑）

```powershell
wakeup listen --show-score   # 对麦克风说「小元」，观察分数，决定 config 里的 threshold
```

> **说「小元」分数很低 / 不灵敏？** 这是纯 TTS 训练的**域差距**：模型只听过合成音，
> 对你的真嗓音/麦克风是"没见过的分布"。不是 bug，调低阈值也救不了。两步解决：
>
> 1. **音频增强**（默认已开）：给 TTS 正样本叠噪声/混响/麦克风频响逼近真实录音。
>    在 `configs/config.yaml` 的 `data.audio_augment_variants` 调大（如 3~4）后
>    `wakeup prepare --force-features && wakeup fit` 重训。
> 2. **录几十条你自己的「小元」**（最有效）：`wakeup record --count 40`，再
>    `wakeup prepare --force-features && wakeup fit`。几十条就能把对你本人的召回拉满。
>
> 注意：训练日志里的"验证 F1/推荐阈值"是在**合成构造样本**上算的，偏乐观；真实可触发
> 阈值请以 `listen --show-score` 实测为准（通常比推荐值低不少）。

### 3）启动常驻服务

```powershell
wakeup serve                 # 启动后默认不监听，等待外部 start 指令
wakeup serve --listen        # 启动后立即开始监听
```

后台守护进程：

```powershell
wakeup daemon start --listen # 后台启动服务，并立即监听
wakeup daemon status         # 查看 pid、日志路径、服务连通性
wakeup daemon stop           # 关闭后台服务
wakeup daemon install --listen   # 注册开机/登录自启动（Windows: schtasks；Linux: systemd user）
wakeup daemon uninstall
```

日志说明：

- `wakeup serve` 常驻服务写滚动文件日志 `artifacts/logs/wakeup.log`（10MB × 5 份，UTF-8），可用环境变量 `WAKEUP_LOG_FILE` 覆盖路径、设为空串禁用
- `wakeup daemon` 后台模式的输出重定向到 `artifacts/run/wakeup-service.log`
- 日志格式与 ChatCaht 全家统一：`2026-07-06 10:09:29,554 INFO wakeup.service.server: 消息`，可被 ChatCaht Dashboard 直接解析

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
import asyncio

from wakeup.service.ws_client import WsServiceClient, wake_ws_url


async def main():
    url = wake_ws_url("127.0.0.1", 8766, "/v1/wake/ws")
    async with WsServiceClient(url) as c:
        await c.start()             # 开始监听
        async for msg in c.messages():
            if msg["type"] == "wake":
                print("唤醒!", msg["score"])


asyncio.run(main())
```

---

## 真实音频验收

训练集指标只能说明模型在构造样本上表现如何；上线前建议单独录一组真实验收音频：

```powershell
wakeup eval-record positive --count 20 --seconds 3  # 录「小元」
wakeup eval-record negative --count 30 --seconds 3  # 录环境声、闲聊、相似词
wakeup eval --threshold 0.5
```

默认目录是 `artifacts/data/eval/positive` 和 `artifacts/data/eval/negative`，报告输出到
`artifacts/model_output/real_eval.json` 与 `real_eval.csv`。验收报告会给出 precision、recall、
F1、false positive rate 和逐条峰值分数，方便决定 `configs/config.yaml` 里的 `service.threshold`。

---

## 控制协议（任何语言都能集成）

本机 WebSocket（默认 `ws://127.0.0.1:8766/v1/wake/ws`），收发 JSON 对象：

| 方向 | 消息 | 说明 |
|------|------|------|
| 客户端→服务 | `{"type":"ping"}` | 连通性测试 |
| 客户端→服务 | `{"type":"start"}` | 开始监听 |
| 客户端→服务 | `{"type":"stop"}` | 停止监听（释放麦克风）|
| 客户端→服务 | `{"type":"status"}` | 查询状态 |
| 客户端→服务 | `{"type":"shutdown"}` | 关闭服务 |
| 服务→客户端 | `{"type":"wake","model":"xiaoyuan","score":0.97,"ts":...}` | **唤醒事件（广播给所有连接）** |
| 服务→客户端 | `{"type":"status","listening":true,...}` | 状态 |
| 服务→客户端 | `{"type":"ack","cmd":"...","ok":true}` | 命令确认 |

例如在任意语言里：连接 WebSocket → 发 `{"type":"start"}` → 持续读取消息，遇到 `"type":"wake"` 即表示听到了唤醒词。

---

## 省电与可靠性是怎么平衡的

1. **停止即释放（主要省电点）**：`stop` 后后台线程关闭麦克风音频流并阻塞等待，不占设备、不耗电；`start` 时再开启。
2. **监听时持续推理**：监听状态下唤醒模型**每帧都跑**，让 openWakeWord 的流式缓冲常驻预热。
   openWakeWord 需要约 2s 连续音频才能填满特征窗口，**跳帧/按需唤起会让短词来不及识别而漏检**
   （早期"静默不跑模型 + 人声起再回灌前导帧"的省电做法正因此严重漏检，已改为持续推理）。
   每帧推理仅 ~1-2ms，开销很小。
3. **触发只看模型分 + 阈值 + 冷却**：openWakeWord 的分数比说词晚约 1s 才到峰值（窗口要先填满），
   而 VAD 在说词时就触发、结束得早，用 VAD 门控触发会把峰值挡在门外、漏检 >90%，因此**不做 VAD 门控**。
   模型本身已在负样本上训练为判别器；误报靠调高 `threshold` 与 `cooldown_seconds` 控制。

相关参数都在 `configs/config.yaml` 的 `service` 段（`threshold` / `cooldown_seconds`）。
`vad_backend` / `vad_aggressiveness` 现仅影响 `--debug` 诊断显示。

> 调阈值/诊断：`wakeup listen --show-score` 看实时分；`wakeup listen --debug` 每帧打印
> VAD/原始分/麦克风电平，用于定位"是 VAD 没抓到"还是"模型不认"。

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

## 开发

```powershell
pip install pytest
pytest                       # 跑轻量单元测试（不需要音频/模型）
```

### 本地验证（Windows / Anaconda）

如果当前 Anaconda 版本不支持 `conda run`，可以先执行 `conda activate wakeup`，或直接调用环境里的 Python：

```powershell
D:\APP\Anaconda3\envs\wakeup\python.exe -m compileall -q src tests main.py
D:\APP\Anaconda3\envs\wakeup\python.exe -m pytest -q
D:\APP\Anaconda3\envs\wakeup\python.exe -m wakeup.cli --help
```

PowerShell 若把中文显示成乱码，通常是终端编码显示问题；项目文件本身按 UTF-8 保存。
