## PTQ4DiT（DiT 后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PTQ4DiT（Wu et al., NeurIPS 2024, arXiv:2405.16005）是首个面向 Diffusion Transformer 的有效后训练量化方法，解决 DiT 量化两大难题：显著通道（salient channel）的激活/权重量化误差大（极值通道在均匀量化下误差显著，需截断）、显著激活跨去噪时间步剧烈变化（静态量化参数失效）。方法：Channel-wise Salience Balancing（CSB，用激活/权重分布统计出的 salience balancing matrix 做通道级变换，利用权重与激活显著通道的互补性——二者不会同时取极值）；Spearman ρ-guided Salience Calibration（SSC，沿时间维扩展通道显著性，加权到 CSB 收益最大的时间步）；重参数化把平衡矩阵离线吸收进相邻层（推理零额外开销）。效果：W8A8 接近全精度、W4A8 保持高质量生成。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文中的角色：所有硬件基线（A40 GPU、EXION、Ditto）统一按 PTQ4DiT 量化到 INT8 精度，其中激活 tensor-wise、权重 channel-wise（论文 A-Evaluation-Methodology 原文表述）。量化流程：校准集上统计激活/权重分布 → 求 salience balancing matrix → 变换激活与权重使显著通道误差减小 → 离线吸收进相邻层参数 → 推理时纯 INT8 GEMM。归一化基线：所有加速器归一化到 A40 峰值 37.4 TFLOPS，保证架构对比公平（排除硬件规模差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源实现见论文官方仓库（论文 Web 证据：arXiv:2405.16005，https://ar5iv.labs.arxiv.org/html/2405.16005 ）。使用：DiT 图像生成与动作规划的 INT8 部署基线；对动作规划 DiT 而言，INT8 量化与三层冗余消除正交——DiTPA 在 PTQ4DiT 之上进一步利用动作/去噪/多模态冗余获得 386.93×/13.22×/9.54× 加速。注意：论文只报告了采用 PTQ4DiT 的 INT8 基线设置，未给出量化感知的消融数据。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
