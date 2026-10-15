# 来源、继承关系与许可范围

## 原方法与官方实现

本项目是 Transporter Networks 的非官方教学复现整理，不由 Google、Google Research 或原论文作者维护，也不表示获得其背书。

- 原论文：Andy Zeng、Pete Florence、Jonathan Tompson 等，*Transporter Networks: Rearranging the Visual World for Robotic Manipulation*，CoRL 2020，<https://arxiv.org/abs/2010.14406>。
- 官方实现：The Ravens Authors，<https://github.com/google-research/ravens>，Apache License 2.0。
- 固定源码提交：`4752ffab487ebd387781ad7890aa540ae2780970`。
- 原项目页与预训练资源说明：<https://transporternets.github.io/>，以及 Ravens README 的数据与预训练模型部分。

抓取/放置网络、相关算子策略、仿真环境、奖励和原始模型/示范来自上游。本项目新增或整理的是环境入口、下载范围控制、中文教学说明、合成算例、绘图和有限场景诊断。运行时取得的 `upstream/` 保留原许可证及文件版权头。

## 本仓库新增内容

Copyright 2026 PeterYanZH and contributors.

除下列例外，本仓库新增的代码、测试、教程文本、原创教学示意及本项目生成的小型数值结果采用仓库根目录的 Apache License 2.0。来自已有 Transporter 教学材料的脚本已为公共发布调整路径、入口和参数，不代表对上游学习算法的改进。

## 单独保留来源的材料

- `docs/具身操作泛化_精修稿_20261011.pdf` 是维护者提供并授权公开的整合精修稿，按原文件提供阅读；其正文及内含第三方材料不纳入本仓库 Apache-2.0 授权。转载或另行出版请核对相应权利人的许可。
- 精修稿中的原论文插图、表格摘录和引用仍归相应作者或权利人，参见稿内图注与参考文献。仓库的 `teaching/figures/` 不另行打包原论文 Figure 3 或 Figure 10 图面。
- 官方预训练权重、专家示范与运行时仿真资产通过上游发布位置获取，不随仓库重新分发。请遵守其上游说明；本项目的许可不扩大第三方授权。
- Times New Roman 字体文件不随项目分发。已生成图中含嵌入字形；重新绘制请使用自己有使用权的字体安装，或显式选择已安装的开源替代字体。

## 结果归属与解释

`evidence/historical/` 是本项目在 2026 年 9 月记录的有限范围验证；不是原论文作者报告的完整基准。本项目图注及教程区分人工算例、预训练模型诊断、历史通路检查与原论文结果。不要把三次回放、有限场景平移测量或自由 logits 优化称为全任务泛化复现。
