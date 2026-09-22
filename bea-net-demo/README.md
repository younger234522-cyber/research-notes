# BEA-Net 学习用推理与失败分析工具

基于 BEA-Net 作者公开实现的学习记录。该目录提供在 AI 辅助下整理的推理、指标计算和对照图脚本，**不包含原创网络结构，不代表从头训练或完整复现论文结果**。

对应阅读笔记位于本仓库根目录：`阅读笔记：BEA-Net文献精读与个人思考.md`。

## 来源与贡献边界

- 原作者实现：[hulinkuang/BEA-Net](https://github.com/hulinkuang/BEA-Net)。网络和预处理均从该仓库导入，不在此重复分发。
- 验证使用的原作者提交：`610823cc656c00661e5760031aa26ac13929c0d9`。
- 原仓库根目录提供 [Apache-2.0 LICENSE](https://github.com/hulinkuang/BEA-Net/blob/610823cc656c00661e5760031aa26ac13929c0d9/LICENSE)；原代码及第三方组件遵循各自许可。数据和权重的使用条件需单独核对。
- 本目录新增：命令行入口、受限权重加载、逐图 Dice/Precision/Recall、误分/漏分可视化、运行记录及指标测试。
- 这里没有复制论文 PDF、模型权重、示例图片、临床照片、虚拟环境或个人账号配置。

## 运行环境

CPU 即可完成四张公开示例推理，不要求 GPU。已在 Windows / Python 3.12.4 的现有环境中验证脚本；不是对所有平台的兼容性保证，也尚未验证全新环境的一键安装。

验证环境：PyTorch `2.12.0+cu126`、NumPy `2.1.2`、SciPy `1.17.1`、Pillow `11.0.0`、Matplotlib `3.9.2`、Kornia `0.6.4`。这里采用较新的推理环境，不是作者原始训练环境。请不要为运行本示例修改作者网络或使用不受限的 pickle 加载。

## Windows 快速开始

先下载本仓库，在其 `bea-net-demo` 目录打开 PowerShell。电脑需已有 Python 3.12 和 Git。

### 1. 创建独立环境并安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 固定了本机验证过的软件版本；PyTorch 的安装构建未固定 CUDA 后缀，默认运行 CPU。如果没有 `py` 启动器，可用自己的 Python 3.12 可执行文件代替。不要把整个 `.venv` 上传 GitHub。

### 2. 获取作者代码并固定版本

```powershell
git clone https://github.com/hulinkuang/BEA-Net.git vendor/BEA-Net
git -C vendor/BEA-Net checkout 610823cc656c00661e5760031aa26ac13929c0d9
```

如果已有作者代码，可以跳过下载，在运行时通过 `--repo` 指定其路径。仅运行本脚本不需要安装作者的完整训练框架。

### 3. 从作者提供的入口获取权重

访问 [作者 README 的预训练模型说明](https://github.com/hulinkuang/BEA-Net#14-testing-demo-with-pre-trained-model)，自行下载 `model_final_checkpoint.model`，建议放入本目录的 `checkpoints` 文件夹。此仓库不重新分发权重。

本次验证的权重 SHA-256：

```text
83e54f3556025a90f1e15349016ef174577d67e3569e145c401367415c5f10b5
```

只加载可信来源的权重；脚本使用 `weights_only=True`，仅为这份旧权重中的已知 NumPy 元数据配置有限允许列表。若遇到其他不兼容权重，请停止核查，不要直接改成 `weights_only=False`。

### 4. 运行测试和示例

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_demo.py --checkpoint checkpoints/model_final_checkpoint.model --device cpu
```

已有作者代码时：

```powershell
.\.venv\Scripts\python.exe run_demo.py --repo "你的作者代码目录" --checkpoint "你的权重文件路径" --device cpu
```

也可使用 PowerShell 包装脚本：

```powershell
.\run_demo.ps1 -Checkpoint '.\checkpoints\model_final_checkpoint.model'
```

输出默认保存在 `results/`。指定另一个 `--output` 目录可保留不同运行结果；使用相同目录会覆盖同名输出。

## 输出与解释

| 文件 | 含义 |
| --- | --- |
| `*_prediction.png` | 二值预测掩膜 |
| `*_comparison.png` | 原图、参考标注、预测和错误位置；红色为多分，蓝色为漏分，灰色为重合 |
| `metrics.csv` | 每张示例的 Dice、Precision、Recall、假阳性/假阴性像素数和推理耗时 |
| `run_report.json` | 模型与权重信息、实际代码版本、是否有代码改动及协议说明 |

运行脚本会严格检查 296 个权重条目是否匹配。验证权重记录的 epoch 为 20，模型参数量为 8,249,409；不能据此认为它就是论文最终训练设置。

指标的空集约定：预测和真值都为空时 Dice 为 1；没有预测正例时 Precision 为 `null`，没有真实正例时 Recall 为 `null`。CSV 中相应值为空，不擅自填 0。

脚本调用作者的 RGB 预处理：256×256 三次插值、逐通道标准化；掩码按作者示例先二值化、线性缩放，再转整数。**这是兼容作者 demo 的协议，不是建议今后所有标签都用线性插值。** 不采用测试时增强，不进行训练。推理耗时包含设备间传输，不是严格速度基准。

## 已验证的四张示例

| 作者仓库中的图像编号 | Dice |
| --- | ---: |
| ISIC_0014624 | 0.9625 |
| ISIC_0014625 | 0.9420 |
| ISIC_0014628 | 0.8542 |
| ISIC_0014632 | 0.6752 |

平均 Dice 约为 0.8585。这些是作者仓库的公开皮肤镜示例，训练重叠情况未知，**不是独立测试集成绩、不是论文完整复现结果，也不是鲜红斑痣实验结果**。不同环境可能有微小数值差异。新模型、数据或权重不能预期得到相同成绩。

后续可据误分/漏分图整理失败类型，但当前没有开展训练、泛化实验或医学用途验证。不要上传临床照片、标注、患者编号或私人权重；`.gitignore` 只是辅助防护，提交前仍需检查文件清单。

## 论文引用

Kuang H, Wang Y, Liang Y, Liu J, Wang J. BEA-Net: Body and Edge Aware Network With Multi-Scale Short-Term Concatenation for Medical Image Segmentation. IEEE Journal of Biomedical and Health Informatics. 2023;27(10):4828–4839. [DOI](https://doi.org/10.1109/JBHI.2023.3304662).

作者仓库同时包含早期 BEA-SegNet 的实现与引用说明；使用其代码开展研究时请同时查看上游致谢及引用要求。
