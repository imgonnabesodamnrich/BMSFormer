# 【BMSFormer论文笔记】高效电池健康状态估算——局部-全局融合注意力驱动的轻量化深度学习模型

> **Efficient State-of-Health Estimation of Lithium-ion Batteries**
>
> **Paper Title:** *BMSFormer: An efficient deep learning model for online state-of-health estimation of lithium-ion batteries under high-frequency early SOC data with strong correlated single health indicator*
>
> **Journal:** *Energy (Elsevier), Vol. 313, 2024*
>
> **DOI:** [10.1016/j.energy.2024.134030](https://doi.org/10.1016/j.energy.2024.134030)

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.energy.2024.134030-blue)](https://doi.org/10.1016/j.energy.2024.134030)
[![Status](https://img.shields.io/badge/Status-Published-brightgreen)](https://doi.org/10.1016/j.energy.2024.134030)
[![Topic](https://img.shields.io/badge/Topic-Efficiency%20Estimation-orange)](https://doi.org/10.1016/j.energy.2024.134030)

## 目录

- [1. 研究背景与核心问题](#1-研究背景与核心问题)
- [2. 健康指标（HI）的提取逻辑](#2-健康指标hi的提取逻辑)
- [3. BMSFormer 模型架构设计](#3-bmsformer-模型架构设计)
    - [3.1 局部-全局融合注意力机制 (LGFA)](#31-局部-全局融合注意力机制-lgfa)
    - [3.2 深度可分离卷积模块 (DSConv)](#32-深度可分离卷积模块-dsconv)
- [4. 实验验证与性能评估](#4-实验验证与性能评估)
    - [4.1 准确性对比](#41-准确性对比)
    - [4.2 部署效率对比](#42-部署效率对比)
    - [4.3 稳定性分析](#43-稳定性分析)
- [5. 结论](#5-结论)
- [论文链接](#论文链接)

### 1. 研究背景与核心问题

锂离子电池的健康状态（State-of-Health, SOH）定义为当前容量与额定容量的比值。由于电池内部降解（如活性物质损失、固体电解质界面膜增厚等）属于复杂的电化学过程，SOH 无法通过物理传感器直接测量，通常需要通过外部可观测信号（如电压、电流、温度）进行回归估算。

目前，工业界对电池管理系统（BMS）的 SOH 估算提出了两个核心诉求：
*   **高精度**：能够准确捕捉电池容量衰减过程中的非线性波动。
*   **低功耗**：由于 BMS 硬件资源（内存与算力）有限，深度学习模型必须足够轻量，以支持在线实时运行。

现有的方法如 LSTM 或传统的 Transformer，要么难以处理长序列的全局依赖，要么计算开销（O(N^2) 复杂度）过大，难以在嵌入式芯片上部署。

### 2. 健康指标（HI）的提取逻辑

在训练深度学习模型之前，输入数据的质量决定了模型的上限。论文中采用了一种高相关性单一健康指标提取方法：

*   **物理依据**：在电池每天的充电过程中，特定电压段的充电时长与电池老化的相关性极高。
*   **提取过程**：通过对高频早期荷电状态（SOC）段（如充电阶段 3.8 V–4.2 V，放电阶段 3.8 V–3.4 V）进行逐步窗口搜索。
*   **相关性验证**：实验结果表明，通过该方法确定的恒流充电时间（CCCT）与电池 SOH 的皮尔逊相关系数（PCC）平均超过了 **0.99**。这意味着单一指标就能提供足够丰富的信息，在简化输入特征的同时降低了预处理成本。

### 3. BMSFormer 模型架构设计

<div align="center">
  <img src="https://github.com/user-attachments/assets/b3e181a2-ec04-41d4-a0c6-3010abd4a78b" width="80%" />
  <p><em>图1 BMSFormer 框架 (a) LGFA 模块 (b) BMSFormer 块 (c) DSConv-L 模块 (d) BMSFormer 核心结构</em></p>
</div>

图1为BMSFormer框架图，为了兼顾计算效率与特征捕捉能力，BMSFormer 模型引入了两个核心改进：

#### 3.1 局部-全局融合注意力机制 (LGFA)

<div align="center">
  <img src="https://github.com/user-attachments/assets/8f2d9497-3296-4f2f-bef9-331156b00912" width="80%" />
  <p><em>图2 (a) 传统 Softmax 注意力机制 (b) 传统线性注意力机制与 (c) 所提出的局部-全局融合注意力模块之间的区别</em></p>
</div>
传统的自注意力机制（Softmax-based Attention）计算量随输入序列长度 N 的增长呈平方级增加。BMSFormer 采用了一种局部-全局融合注意力模块：

*   **线性复杂度**：利用关联属性（Associative Property）改变计算顺序，将复杂度从 O(N^2) 降为 O(N)。

*   **特征捕捉**：通过 ReLU 激活函数增强特征的表达能力，使其既能关注长期的容量衰减趋势（全局），也能敏感捕捉到容量再生或突发波动（局部）。

#### 3.2 深度可分离卷积模块 (DSConv)

<div align="center">
  <img src="https://github.com/user-attachments/assets/042411a8-0acf-42fe-a076-85704a234674" width="80%" />
  <p><em>图3 (a) 标准卷积 (b) 标准深度可分离卷积 (c) DSConv-S 模块 (d) DSConv-L 模块的基本结构</em></p>
</div>
模型中嵌入了两种不同倍数的深度可分离卷积：DSConv-S（小核）和 DSConv-L（大核）。

*   **原理**：将标准卷积分解为深度卷积（Depthwise）和逐点卷积（Pointwise）。

*   **作用**：在不增加过多参数的前提下，DSConv 能够融合多尺度和多通道特征，增加特征的多样性。这对于区分电池在不同老化阶段的细微特征起到了关键作用。

### 4. 实验验证与性能评估

论文使用了三类主流公开数据集（Oxford、NASA、CALCE）对模型进行验证。这些数据集涵盖了不同的正极材料（LCO、NCA、NCO-LCO）和不同的运行工况。

#### 4.1 准确性对比
在与 LSTM、Transformer、CNN-LSTM、CNN-Transformer 等模型的对比实验中，BMSFormer 在所有测试电池上均表现出更低的 MAE（平均绝对误差）和 RMSE（均方根误差）。即便在电池 SOH 出现剧烈波动的情况下，该预测曲线仍能紧密贴合真实值。

#### 4.2 部署效率对比
实验重点分析了模型的硬件成本。在同等精度水平下：

*   **参数量与存储**：BMSFormer 的存储大小约为 **36.37 KB**。

*   **计算速度**：由于采用了线性注意力机制，其训练时间显著缩短，且 FLOPs（浮点运算数）维持在较低水平。

#### 4.3 稳定性分析
通过 384 组不同的超参数组合（如层数、学习率、维度等）测试发现，BMSFormer 在不同参数设置下均能保持稳定的估算性能。这种鲁棒性对于实际工程应用非常重要，意味着模型不需要针对每种电池进行繁琐的参数微调。

### 5. 结论

BMSFormer 提供了一种平衡精度与效率的电池健康度评估方案。通过高效的注意力机制设计与轻量化卷积模块，可以在资源受限的硬件上实现较为可靠的 SOH 在线监测。这为下一代智能电池管理系统的开发提供了技术支撑。

---

### 论文链接

**Title:** BMSFormer: An efficient deep learning model for online state-of-health estimation of lithium-ion batteries under high-frequency early SOC data with strong correlated single health indicator

**Journal:** *Energy*, 2024

**DOI:** [10.1016/j.energy.2024.134030](https://doi.org/10.1016/j.energy.2024.134030)

```bibtex
@article{li2024bmsformer,
  title={BMSFormer: An efficient deep learning model for online state-of-health estimation of lithium-ion batteries under high-frequency early SOC data with strong correlated single health indicator},
  author={Li, Xiaopeng and Zhao, Minghang and Zhong, Shisheng and Li, Junfu and Fu, Song and Yan, Zhiqi},
  journal={Energy},
  volume={313},
  pages={134030},
  year={2024},
  publisher={Elsevier}
}
