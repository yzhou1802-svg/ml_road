# ML Journey: 16-Week Roadmap

> 从零到 AI for Physics 的学习路径
> 时间窗口：现在 → 2026 年 9 月初
> 优先级：IPhO 准备 > ML project。ML 用业余时间推进，不挤占物理。

---

## 📋 总览

| 阶段 | 周次 | 主题 | 核心 Deliverable |
|------|------|------|------------------|
| **0** | Pre-Week | 环境搭建 | 工具链就绪 |
| **1** | W1–W2 | NumPy 神经网络 | MNIST 分类器（无框架） |
| **2** | W3–W6 | nanoGPT | 自己的 Transformer + 实验 |
| **3a** | W7–W10 | Neural ODE | Lorenz 系统拟合 |
| **3b** | W11–W14 | Physics-Informed NN | Burgers 方程求解器 |
| **写作期** | W15–W16 | 整合 + 博客 | 公开博客 ×1–2 篇 |

**每周投入预算**：8–12 小时（工作日晚上 + 周末，灵活分配）
**绝对优先级**：IPhO Hausaufgabenrunde 截止周自动暂停 ML，不必内疚

---

## Pre-Week: 环境搭建（半天）

### Tasks

- [ ] 安装 Python 3.11+、NumPy、Matplotlib
- [ ] 安装 PyTorch（pytorch.org 选对应 CUDA 版本，无 GPU 用 CPU 版）
- [ ] 配置 VS Code + Python 扩展 + Jupyter 扩展
- [ ] 创建 GitHub repo: `ml-journey`，结构如下：
  ```
  ml-journey/
  ├── README.md           ← 这份 roadmap
  ├── 01-numpy-nn/
  ├── 02-nanogpt/
  ├── 03a-neural-ode/
  ├── 03b-pinn/
  └── LOG.md              ← 日志
  ```
- [ ] 注册 wandb.ai 免费账户
- [ ] 备份方案：Google Colab 或 Kaggle（免费 GPU）

### Checkpoint
能在终端跑：
```python
import torch
print(torch.cuda.is_available())  # 或 mps for Mac
```
能 push 一个 commit 到 GitHub。

---

## 阶段 1：NumPy 神经网络（W1–W2）

**目标**：能在纸上推导反向传播。理解 `.backward()` 在做什么。

### Week 1 — micrograd

跟随 Karpathy: *"The spelled-out intro to neural networks and backpropagation: building micrograd"*

- [ ] **Day 1–2**：看视频 + 手敲代码（不复制粘贴）
- [ ] **Day 3–4**：合上视频，从零重写 `Value` 类和反向传播
- [ ] **Day 5–7**：用自己的 micrograd 训练 MLP 解 `make_moons` 二分类，画决策边界

**Deliverable**: `01-numpy-nn/micrograd.py` + `moons_decision_boundary.png`

**Checkpoint**: 能在白纸上画出 `L = (a*b + c)**2` 的计算图，标出每个节点的梯度。

---

### Week 2 — NumPy MLP on MNIST

- [ ] **Day 1–3**：手写支持任意层数的 MLP 类
  - 前向：矩阵乘法
  - 反向：自己推导梯度（不用 autograd）
  - 激活：ReLU + softmax
  - **关键**：用 He 初始化 `np.random.randn() * sqrt(2/n_in)`
- [ ] **Day 4–5**：MNIST 训练，目标测试集 ≥ 97%
- [ ] **Day 6–7**：写 `01-numpy-nn/README.md`
  - 架构说明
  - 训练曲线（loss / accuracy vs epoch）
  - 超参数选择理由
  - 踩过的坑

**Deliverable**: GitHub 上 `01-numpy-nn/` 完整 + README

**Checkpoint**: 能解释为什么 cross-entropy + softmax 在反向传播时梯度形式特别简洁（提示：链式法则化简后 `dL/dz = p - y`）。

---

## 阶段 2：nanoGPT（W3–W6）

**目标**：能从零写出 Transformer。理解 attention 每一行代码。

### Week 3 — PyTorch + makemore

Karpathy *makemore* 系列 Part 1 & 2

- [ ] 看 Part 1 (bigram) + Part 2 (MLP language model)
- [ ] 掌握 PyTorch 核心 API：tensor 操作、`nn.Module`、optimizer、loss
- [ ] **小练习**：用 MLP 训练中文人名生成模型（找一份中文姓名数据集）

**Deliverable**: `02-nanogpt/makemore_chinese_names.ipynb` + 20 个生成样例

**Checkpoint**: 不查文档能写出 PyTorch 训练循环骨架（forward → loss → backward → step → zero_grad）。

---

### Week 4 — Attention 机制

Karpathy *"Let's build GPT from scratch, in code, spelled out"*（约 2 小时，关键视频）

- [ ] **Day 1–3**：跟视频实现 Tiny Shakespeare GPT
- [ ] **Day 4–5**：默写 self-attention forward pass
  - 理解 Q/K/V 的几何含义
  - 理解为什么除以 `sqrt(d_k)`
  - 理解 causal mask 为什么是上三角
- [ ] **Day 6–7**：扫描实验——不同 context length / model size 下的 loss

**Deliverable**: `02-nanogpt/tiny_shakespeare.py` + loss 曲线对比图

**Checkpoint**: 一句话解释 attention 是 permutation-equivariant，以及 positional encoding 解决了什么问题。

---

### Week 5 — 你自己的 GPT

选**一个**数据集（推荐围棋棋谱——和你的兴趣最契合）：

- **选项 A**：围棋棋谱（SGF → token 序列，下一手预测）
- **选项 B**：克莱斯特/德语经典文本生成
- **选项 C**：物理公式 LaTeX 生成

- [ ] **Day 1–3**：数据预处理 + tokenizer
- [ ] **Day 4–7**：训练 + 评估 + 生成样例

**Mini scaling study**（如果时间允许）：
固定数据集，训练 3–4 个不同大小模型（1M / 5M / 25M params），画 loss vs 计算量曲线。

**Deliverable**: `02-nanogpt/my_gpt/` 完整目录 + 生成样例 + scaling plot

**Checkpoint**: 能讲清楚 model size 翻倍时 loss 降低的趋势是什么样的（参考 Chinchilla 论文的直觉）。

---

### Week 6 — 现代改进 + 写作

选**一个**现代 Transformer 改进点实现并对比：

- [ ] **选项**：RoPE / GQA / RMSNorm 三选一
- [ ] **Day 1–4**：实现 + 对照实验（同步数，新版 vs 原版 loss）
- [ ] **Day 5–7**：写第一篇博客 *"Building GPT from scratch: what I learned"*
  - 中文或英文都行
  - 讲清 attention、列出实验数字、说明改进的效果

**Deliverable**:
- 改进版代码 + 对照实验图
- 博客发到个人网站、Medium 或 GitHub Pages

**Checkpoint**: 博客让一个不懂 ML 的同学读，能理解大致在做什么。

---

## 阶段 3a：Neural ODE（W7–W10）

**目标**：理解连续深度模型。能用 NN 学习物理动力学。

### Week 7 — 论文 + 基础

读 **Chen et al. 2018, "Neural Ordinary Differential Equations"**（NeurIPS best paper）

- [ ] **Day 1–3**：读论文 + 看作者讲解视频
  - 重点理解 adjoint method
  - 理解为什么能节省内存
- [ ] **Day 4–7**：安装 `torchdiffeq`，跑通官方 spiral ODE 示例

**Deliverable**: `03a-neural-ode/paper_notes.md`（你自己的论文笔记，标出哪里没懂）

**Checkpoint**: 能解释 Neural ODE 和 ResNet 的关系（提示：ResNet 是 forward Euler discretization）。

---

### Week 8 — Spiral ODE 复现

- [ ] **Day 1–4**：完整复现论文 Fig. 3 的 spiral 实验
- [ ] **Day 5–7**：变体实验——改变 hidden dim、积分器（Euler / RK4 / Dopri5），对比效果

**Deliverable**: `03a-neural-ode/spiral.py` + 训练动画（每个 epoch 的拟合曲线）

**Checkpoint**: 理解 `odeint` 函数的输入输出 shape。

---

### Week 9 — Lorenz 系统（核心实验）

- [ ] **Day 1–2**：生成 Lorenz 数据
  - 用 `scipy.integrate.solve_ivp` 或你之前学的 RK4 生成 1000 步轨迹
  - 经典参数 σ=10, β=8/3, ρ=28
- [ ] **Day 3–6**：训练 Neural ODE 拟合动力学
- [ ] **Day 7**：评估
  - 短期：能否精确预测 50 步？
  - 长期：能否学到奇异吸引子的形状（即使具体轨迹偏离）？

**Deliverable**: `03a-neural-ode/lorenz.py` + 吸引子 3D 可视化

**Checkpoint**: 能解释为什么混沌系统不可能长期精确预测（Lyapunov exponent 直觉）。

---

### Week 10 — 对比 + 写作

- [ ] **Day 1–4**：与传统方法对比
  - vs RK4 / RKF45（你之前学过的）
  - 精度对比表
  - 计算时间对比
  - **诚实评估**：什么情况下 Neural ODE 输？
- [ ] **Day 5–7**：写 `03a-neural-ode/README.md`，准备阶段 3b

**Deliverable**: 完整 README + 对比数字

**Checkpoint**: 能回答"我为什么觉得 Neural ODE 有意思？"——这个问题如果讲不清，说明还没真正理解。

---

## 阶段 3b：Physics-Informed NN（W11–W14）

**目标**：用 NN 解 PDE。能诚实评估 PINN 的能力边界。

### Week 11 — 论文 + 热方程

读 **Raissi et al. 2019, "Physics-Informed Neural Networks"** (JCP)

- [ ] **Day 1–2**：读论文 + 笔记
- [ ] **Day 3–7**：1D 热方程 `u_t = α u_xx`
  - 给定初始条件和边界条件
  - Loss = PDE residual + IC loss + BC loss
  - 与解析解对比

**Deliverable**: `03b-pinn/heat_1d.py` + 时空热图

**Checkpoint**: 能讲清 PINN 的 loss 三个分量分别在干什么。

---

### Week 12 — Burgers 方程

PINN 标准 benchmark。有 shock，能展示能力和局限。

- [ ] **Day 1–4**：复现 Raissi 论文 Burgers 实验
- [ ] **Day 5–7**：分析失败模式
  - 在 shock 附近 PINN 表现如何？
  - 增加 collocation points 在 shock 区域是否有帮助？

**Deliverable**: `03b-pinn/burgers.py` + shock 区域误差分析

**Checkpoint**: 能解释为什么 PINN 在 stiff 问题上常失败（gradient pathology）。

---

### Week 13 — 与传统方法对比

这是真正体现研究品味的一周。

- [ ] **Day 1–3**：实现有限差分（FDM）解同一个 Burgers 方程
- [ ] **Day 4–6**：建立对比矩阵
  - 精度（vs 解析解或高分辨率 FDM）
  - 计算时间
  - 内存
  - 实现复杂度
- [ ] **Day 7**：诚实总结表格："PINN wins when…, FDM wins when…"

**Deliverable**: `03b-pinn/comparison.md` + 对比图表

**Checkpoint**: 能客观说出 PINN 的两个明显劣势。

---

### Week 14 — 与 IPhO 物理的联系

可选但强烈推荐的协同：

- [ ] 选 IPhO 复习里的一个 PDE 问题（电磁/热传导/波动）
- [ ] 用 PINN 求解
- [ ] 对比解析或数值解

**Deliverable**: `03b-pinn/ipho_application.py` + 简短说明

**Checkpoint**: PINN 的求解过程让你对该物理问题有什么新的几何直觉？

---

## 写作期：W15–W16

很多人忽视这一步——而它的价值不亚于代码本身。

### Week 15 — 第二篇博客

主题：*"Learning physics with neural networks: Neural ODE & PINN"*

- [ ] **Day 1–3**：大纲 + 草稿
- [ ] **Day 4–5**：插图（吸引子图、Burgers 时空图、对比表）
- [ ] **Day 6–7**：润色 + 发布

**Deliverable**: 公开博客文章

---

### Week 16 — 整合 + 反思

- [ ] 更新 `ml-journey` 主 README：每个子目录的简介 + 链接
- [ ] GitHub Pages 个人页面（可选但推荐）：把博客和 project 链接起来
- [ ] 写 `RETROSPECTIVE.md`：
  - 哪个 project 最有收获？
  - 哪里花的时间超出预期？
  - 接下来想深挖什么方向？
- [ ] 给自己放一个长周末

**Deliverable**: 干净的 GitHub portfolio

---

## 📐 通用工作方法

### LOG.md 模板

每次坐下来工作前写：
```markdown
## 2026-MM-DD
**目标**: 今天要完成什么
**实际**: 做到了什么
**卡点**: 哪里卡住了，怎么解决/绕过的
**明日**: 下次接着做什么
```

三个月后回头看，这些日志就是你写博客和申请材料的原始素材。

### "够好就好" 原则

每个阶段达到 checkpoint 就进下一阶段。不要花两周优化 NumPy NN。完美主义是这种长期 project 最大的杀手。

### IPhO 优先级规则

- IPhO 截止周自动暂停 ML
- 物理直觉感觉变钝时立即停下，回去做物理题
- ML 和 IPhO 冲突时，IPhO 永远赢

### 协同机会

主动让 ML project 加深物理直觉：
- PINN 热方程 ↔ IPhO 热学
- PINN Laplace 方程 ↔ IPhO 静电学
- Neural ODE Duffing 振子 ↔ IPhO 非线性振动

---

## ⚠️ 防坑清单

- [ ] 不要因为某周状态好就提前赶下周内容——会透支
- [ ] 不要看到新论文就开新坑——记下来，做完当前阶段再看
- [ ] 不要追求 SOTA——这是学习 project，不是发论文
- [ ] 不要不写 README——没文档的 repo 等于没做
- [ ] 不要害怕公开发布博客——粗糙的博客 > 完美的草稿

---

## 🎯 16 周后你将拥有

- 4 个有 README 的 GitHub project
- 2 篇技术博客
- 对反向传播、Transformer、Neural ODE、PINN 的真实理解
- 一个能讲出"我为什么觉得 AI for Physics 有意思"的故事
- 更敏锐的物理直觉（PINN 副作用）

这套作品集放在大学申请或本科 research 申请中，比"做了 10 个 Kaggle 比赛"更有辨识度。

---

*最后更新：2026-05-17*
*下次回顾：每个阶段结束时*
