# Transporter Networks 教学复现教程

本教程面向有深度学习基础的研究生，目标是沿着“输入坐标 → 抓取分布 → 条件放置分布 → 仿真执行 → 受控诊断”走通一次可检查的决策过程。它继承 [Transporter Networks](https://transporternets.github.io/) 的方法和 [Google Research Ravens](https://github.com/google-research/ravens) 的实现，是非官方教学整理；网络结构、官方检查点和任务实现归原作者。本项目补充环境整理、限量下载、流程检查、数值算例与可视化，不提出新的操作算法。

教程对应[精修稿](具身操作泛化_精修稿_20261011.pdf)的第 9.2 节和第 9.6 节。公式、图和代码的逐项对应见[教学索引](teaching-map.zh-CN.md)，结果的实验单位及验证边界见[结果记录](results.md)。精修稿还讨论 VLA、ReconVLA 和世界模型；这些内容没有被本仓库实现或评估。

## 1. 先选复现深度

| 路线 | 需要什么 | 运行什么 | 可以回答什么 |
| --- | --- | --- | --- |
| 阅读与数值教学 | Python、NumPy；画图另需绘图库和字体 | 三个合成算例、三通道内积图 | 表示与损失为什么产生这些结果 |
| 官方预训练推理 | Linux CPU 环境、两份官方权重、3 条测试示范 | `eval_pretrained.sh` | 当前安装能否执行完整抓放闭环 |
| 模型机制诊断 | 同上 | `model_diagnostics.py` | 改抓点、平移输入时，固定模型如何响应 |
| 可选训练通路检查 | 再加 10 条训练示范 | `smoke_train.sh` | 读取、反传、更新、保存与重新加载是否通畅 |

通常执行前三条就够了。最后一条严格检查一次训练迭代，不用于获得一个训练充分的新模型。不要将“下载的官方 10,000 步检查点”理解为本项目重新训练了 10,000 步。

## 2. 克隆和建立隔离环境

运行基线为 **Ubuntu 22.04、Python 3.10、CPU、无界面 PyBullet**。需要 Bash、Git，以及带 `venv` 支持的 Python 3.10。Windows 用户可在 Ubuntu WSL2 中执行下列 Bash 命令；本仓库没有验证 Windows 原生或 GPU 路径。

```bash
git clone https://github.com/PeterYanZH/transporter-networks-tutorial.git
cd transporter-networks-tutorial

# 默认解释器就是 python3.10；也可用 PYTHON_BIN 指定其路径。
PYTHON_BIN=python3.10 bash scripts/setup.sh
bash scripts/fetch_source.sh
bash scripts/check_environment.sh
```

环境建在仓库内 `.venv/`，上游源码放入 `upstream/`。固定源码提交为：

```text
4752ffab487ebd387781ad7890aa540ae2780970
```

后续阅读源码和加载权重都以这一提交为准。不要自行把 `upstream/` 换成跟随 `master` 的最新版本后仍沿用本教程的验证结论。

如果目标机器无法通过正常 HTTPS 访问 GitHub，可在一台能够正常验证 HTTPS 的可信机器上，从官方仓库制作完整 Git bundle：

```bash
git clone https://github.com/google-research/ravens.git ravens-source
git -C ravens-source checkout --detach 4752ffab487ebd387781ad7890aa540ae2780970
git -C ravens-source bundle create ../ravens.bundle HEAD
```

这里必须使用包含完整历史的克隆，不能加入 `--depth`。如果复用的是浅克隆，先执行 `git -C ravens-source fetch --unshallow`；`git -C ravens-source rev-parse --is-shallow-repository` 应输出 `false`。不要把从浅克隆直接制作的 bundle 当作完整离线源码。

将 `ravens.bundle` 传到目标机器，放在教程仓库的同级目录，再从教程仓库根目录执行：

```bash
RAVENS_SOURCE_BUNDLE="$(cd .. && pwd)/ravens.bundle" bash scripts/fetch_source.sh
```

入口会在空暂存仓库中核验 bundle，取出固定提交并检查精确 HEAD 和干净工作区。bundle 仍须来自可信机器上的官方源码；这种备用流程没有关闭 TLS 验证，也不使用非官方源码镜像。

关键依赖固定为 TensorFlow CPU 2.15.1、Keras 2.15.0、TensorFlow Addons 0.23.0、NumPy 1.26.4、SciPy 1.11.4、PyBullet 3.2.7、Gym 0.23.1 和 OpenCV headless 4.8.1.78。完整版本以仓库的锁定文件为准。这套组合保留旧实现所需 API，不是把所有包升级到最新版。

如需要镜像，仅对本次安装指定索引即可，无需改动系统配置：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  PYTHON_BIN=python3.10 bash scripts/setup.sh
```

检查结果写入 `results/environment.json`。进入交互环境时，在仓库根目录执行：

```bash
source env.sh
python -c 'import ravens, tensorflow as tf; print(tf.__version__)'
```

`env.sh` 选择 CPU 和项目解释器，不修改全局 CUDA、驱动或代理。默认线程配置为计算线程 8、跨算子线程 2；CPU 核数较少时可在调用前设置 `TF_NUM_INTRAOP_THREADS`、`TF_NUM_INTEROP_THREADS` 和 `OMP_NUM_THREADS`。

## 3. 只提取一个小任务的官方资产

```bash
bash scripts/fetch_assets.sh
```

默认只提取 `block-insertion`：

- 10 条训练示范、3 条测试示范。每条保留 `action/color/depth/reward/info` 五个字段。
- `block-insertion-transporter-10-0` 的两份 10,000 步权重：`attention-ckpt-10000.h5` 和 `transport-ckpt-10000.h5`。
- 合计 67 个资产文件；历史清单中选中内容的压缩大小为 121,634,962 字节，解压后为 286,601,400 字节。

模型和数据仍来自作者发布的 [checkpoints.zip](https://storage.googleapis.com/ravens-assets/checkpoints.zip) 与 [block-insertion.zip](https://storage.googleapis.com/ravens-assets/block-insertion.zip)。下载器使用 HTTP Range 读取 ZIP 目录并提取指定成员，避免下载约 5.49 GB 的完整两包。上述 121.6 MB 是选中内容的压缩大小，不包括目录读取、重试等额外网络流量。磁盘还需容纳源码、Python 环境、可视化输出及可选单步权重。

下载过程保存在 `results/fetch-manifest-*.json`，最终资产状态和校验信息保存在 `results/assets.json`。只有校验通过才进入后续运行。若连接中断，重新运行相同命令；已有文件须通过大小与 CRC 检查才会复用。服务端不支持分段响应，或下载期间远端对象身份发生变化时，程序会停止并报告错误。

模型名称中的 `10` 表示作者提供的 10 示范模型。这里提取的 10 条训练示范用于通路检查，不保证恰好是作者训练这份权重时随机抽到的那 10 条。测试场景种子固定为 1、3、5。

## 4. 先评估预训练模型

```bash
bash scripts/eval_pretrained.sh
```

这个入口加载官方权重，执行 3 个测试场景：重置场景、获取 RGB-D、融合俯视图、预测抓放动作、驱动机器人并计算奖励。无须手动打开显示窗口，无须先训练。

检查 `results/eval_pretrained.json` 和 `logs/eval_pretrained.log`。重点是实际场景数、种子、权重步数、各回合奖励及是否正常结束。历史运行的种子 1、3、5 均得到奖励 1.0；这只是三个固定场景的通路证据，不能宣称“复现了论文的 100% 泛化成功率”。重新执行同样种子也不会增加独立测试样本数。

如果与历史值不同，保留新结果及日志，先核对源码提交、权重、依赖、种子和 CPU 设置。无需为了得到相同数字而训练、挑选场景或删掉失败结果。

## 5. 沿代码追踪一次抓放

固定版本源码位于 `upstream/`。建议按下面顺序阅读：

| 入口 | 要找的运算 | 应检查的形状或约定 |
| --- | --- | --- |
| `ravens/agents/transporter.py` 的 `get_image()` | 三视角 RGB-D 融合、RGB 与高度拼接 | `320 × 160 × 6`；高度重复三次，不是六种独立传感模态 |
| `ravens/utils/utils.py` | `get_fused_heightmap()`、`xyz_to_pix()`、`pix_to_xyz()` | `(row, column)` 对应世界 `(Y, X)`；像素间距 3.125 mm |
| `ravens/models/attention.py` | 抓取分数、展平后全局 Softmax | `320 × 160 × 1`；选出抓点 `(r, c)` |
| `ravens/models/transport.py` 的 `forward()` | 围绕抓点旋转输入、裁剪，再送入 Query 网络 | 36 个方向；裁剪边长 64；Query 与 Key 输出通道数为 3 |
| 同文件的 `correlate()` | Query 作为动态核，与 Key 做互相关 | 通道内积与空间求和；输出 `320 × 160 × 36` |
| 同文件的 `train()` | 全张量展平、独热标签、交叉熵 | 放置候选数 1,843,200；没有逐像素独立 Sigmoid |
| `ravens/agents/transporter.py` 的 `act()` | 最大响应解码、像素转物理坐标、动作封装 | 方向索引与执行偏航角的符号由该实现坐标约定确定 |

Query 与 Key 的最后一层分别输出特征。二者随后在 `tf.nn.convolution` 中相匹配；“最后一层通道内积”不要误读为两条网络在各自卷积层内部已经完成了相互比较。

## 6. 重做数值教学图

仓库带有已生成的图和原始数组，阅读它们不需要机器人资产。运行数值算例也不会训练 CNN。

```bash
source env.sh

# 在临时目录验算并比对随仓数据，不覆盖发布图或历史结果。
python teaching/scripts/check_numerics.py

# 只计算和记录数值；不要求 Times New Roman。
python teaching/scripts/toy_experiments.py --out teaching/toys --no-plots
python teaching/scripts/plot_sliding_correlation.py --root teaching --no-plots
```

主要输出为 `teaching/toys/toy_metrics.json`、三个 `*_raw.npz`，以及 `teaching/figures/sliding_correlation.json`。教学实验中的 5,000 步只更新自由坐标或自由 logits，与机器人模型训练不同。

独立的 `check_numerics.py` 只需要 NumPy，适合尚未安装机器人环境的读者；它会重算三个算例及九个三通道窗口，并与随仓数据比较。公开整理时保存的检查结果为 `teaching/results/numerical_checks.json`。上面的后两条命令用于主动重新生成工作结果，而非仅作一致性检查。

| 算例 | 对应稿件 | 应核对的结论 |
| --- | --- | --- |
| 四个合法接触岛 | 图 9.3、式 (9.2.14)–(9.2.15) | MSE 最优位置 `(20,20)` 位于非法区域；分类保留四峰。两种参数化不同，不能当作公平的网络基线消融 |
| 三通道 Query/Key | 图 9.4、式 (9.2.18)–(9.2.19) | 先做对应三维向量内积，再对窗口四个位置求和；首窗口 6，右移一格 17 |
| 五候选梯度 | 图 9.5、式 (9.2.20)–(9.2.25) | `dL/dlogit = p - y`；梯度和约为零；有限差分校验闭式梯度 |
| L 形模板平移 | 补充算例 | 人工身份特征下的相关峰随整数平移移动；不是学习网络的严格等变性证明 |

三通道示例的完整评分图应为：

```text
 6  17  13
17   9  11
11  10   7
```

这里存在两个值为 17 的格点，不能把它写成唯一最优解；这些值是未归一化评分，不是概率。

若系统已安装 Times New Roman，可重绘矢量 PDF 和 PNG：

```bash
python teaching/scripts/plot_toys.py --root teaching
python teaching/scripts/plot_sliding_correlation.py --root teaching
```

所有绘图入口默认使用 Times New Roman。仓库不分发字体文件；若你有授权的字体副本，可以把它们放到自己的 `fonts/` 目录并为每个绘图命令加 `--font-dir fonts`。若使用已安装的开源替代字体，可显式添加 `--font-family 'Liberation Serif'`。替换字体后版面可能改变，请检查文字、负号和箭头，不要把替代字体图宣称为 Times New Roman 排版。

## 7. 重做固定模型的教学诊断

先完成环境、源码和官方资产准备，再运行：

```bash
source env.sh
python teaching/scripts/model_diagnostics.py --project-root . --out teaching
python teaching/scripts/plot_diagnostics.py --root teaching
python teaching/scripts/plot_rollouts.py --root teaching
```

字体选项与上一节相同，例如在未安装 Times New Roman 但已安装 Liberation Serif 的环境中，每条命令都显式添加 `--font-family 'Liberation Serif'`。诊断过程不更新模型参数；它会重新执行种子 1、3、5 的闭环动作，随后对种子 1 的固定观测做条件变化、输入平移和局部梯度计算。

主要输出：

- `teaching/results/model_metrics.json`：回合奖励、动作、三个抓点条件、六个平移条件和梯度统计。
- `teaching/results/model_arrays.npz`：用于条件响应、方向切片、平移与梯度图的原始数组。
- `teaching/results/rollouts_arrays.npz`：初始观测、抓取/放置响应和动作后的图像。
- `teaching/figures/`：模型回放、三抓点、平移、梯度及全工作区参考图。

也可只运行后两条绘图命令，利用仓库带有的原始数组重绘，不重复模型推理。复跑会更新上述同名工作结果；不可变的历史对照保存在 `evidence/historical/`。

读图时分清三个实验问题：

1. **抓点条件与刚体补偿**：固定同一张图，只改变抓点，比较放置峰和由物体真值位姿构造的几何参照。历史三项平面误差约为 4.42、8.84、3.13 mm；只有第一个条件执行过机器人，另两项是前向预测。真值用于计算诊断参照，不作为策略输入。这些误差也不是实测末端定位精度。
2. **一像素和八像素平移**：在输入张量上平移整个图像，并同步移动抓点。历史一像素平移的抓取分布 TV 约为 0.0647、0.1044；八像素平移的 TV 小于 `1e-7`。这与步长 8 的采样相位相符，但并非所有场景、所有平移的普遍保证。TV 的计算域先裁去四边各 32 像素，再对两个内部区域分别归一化。
3. **Query/Key 局部输入梯度**：对选定特征平方范数求输入导数。裁剪外 Query 梯度为零来自“先裁剪再编码”的计算图。局部梯度不是因果干预，不等价于遮挡区域后任务成败的变化。

主要热图使用全图最大值归一化后取 `log10`，只裁剪显示范围，不在局部窗口内重新归一化。亮度不能跨图比较绝对置信度；放置图沿方向取最大值，也不是沿方向求和的概率边缘分布。解释图 9.9、9.10 时，应同时查看原始指标和图注。

## 8. 可选：严格一次训练迭代

```bash
bash scripts/smoke_train.sh
```

一次迭代对抓取分支和放置分支各执行一次优化器更新，读取示范、增广、前向、反向、保存，最后重新加载检查。输出是 `results/train_smoke.json`、`logs/train_smoke.log` 和 `smoke_run/checkpoints/`；官方权重仍位于单独的 `checkpoints/`。

检查有限损失、实际更新数和重新加载状态即可。历史 attention loss 为 10.9863，transport loss 为 269.8396；一次更新的损失值既不能证明收敛，也不要求你重复运行直到数值变小。

不要直接套用上游训练脚本的默认循环。上游 `n_steps` 与 `interval` 共同决定循环；只改 `n_steps=1`、保留默认 `interval=1000`，并不能保证只做一次更新。本仓库入口已经限定流程。原版 `validate()` 的 `Skipping validation.` 也不应记成完成了验证集评估。

## 9. 常见故障按层定位

| 现象 | 首先检查 | 处理方向 |
| --- | --- | --- |
| 创建环境失败或找不到 wheel | `python3.10 --version`，Python 的 `venv` 支持，CPU 架构 | 使用 Python 3.10；按系统方式补齐 `python3.10-venv`。不要混用系统 Python 和 `.venv` |
| `No module named ravens` | 是否完成 `fetch_source.sh`、是否在根目录加载 `env.sh` | 检查 `upstream/ravens` 和环境变量，不要临时安装另一个同名包 |
| Keras/TensorFlow Addons 导入或旧 API 失败 | `results/environment.json` 与锁定版本 | 重建项目环境，不要单独升级到 Keras 3 或 TensorFlow 最新版 |
| 下载提示 Range/ETag/CRC 错误 | `results/fetch-manifest-*.json`、网络是否支持 206 响应 | 保留失败记录，确认官方源可访问后重试；不要跳过校验或关闭 TLS 验证 |
| 权重缺失、形状不符 | 两个权重文件名、10,000 步、固定上游提交 | 重新核对资产清单，不要拿单步权重代替官方预训练权重 |
| 仿真找不到 URDF/资源 | `upstream/ravens/environments/assets` | 重新核对源码完整性；入口采用无界面运行，无须桌面显示服务 |
| 字体错误但数值实验可运行 | Times New Roman 是否可找到 | 提供有授权字体目录，或显式选择已安装的开源字体；数值检查可用 `--no-plots` |
| 图看起来相同但 TV 非零 | 是否在比较最大位置而非完整分布 | 峰位不变不意味着其余像素概率不变；检查内部域归一化及原始数组 |
| 命令过慢或内存吃紧 | 同时运行的任务和线程数 | 顺序运行；减少 CPU 线程；保留默认 3 个场景，不要无意扩大任务 |

最后，可运行不依赖机器人完整训练的仓库检查：

```bash
python3 -m unittest discover -s tests -v
```

单元检查验证下载、边界或数值约定等具体行为；它不能替代预训练闭环回放。公开报告新结果时，附上提交、环境、检查点、种子、实际更新次数与原始结果文件，并区分数学算例、固定观测诊断和物理闭环试验。
