## Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

- baseline方法是什么？
  - **单模态方法**：仅使用某一 modality 做分类/诊断，如 3D CNN [17] 处理 MRI、VGG19 [43] + transfer learning 处理 2D MRI slices、ResNet-18 [45] 处理 fMRI、DLG (ResNet-34) [36] 处理 Genetic SNP、deep learning-assisted spectroscopy [29] 处理 Biospecimen。这些方法完全忽略多模态互补信息。
  - **多模态方法（仅处理全模态交集）**：如 Tensor Fusion Network (TF) [74] 使用 tensor fusion layer 融合多模态 embedding；MulT [57] 使用 cross-modal attention 捕获跨模态交互；MAG [52] 将多模态特征映射为 adaptation vector；LIMoE [44] 通过 contrastive learning+entropy regularization 处理多模态 MoE；FuseMoE [19] 直接通过 MoE 融合多模态数据。这些方法假设所有 modality 都可用（只取交集训练），对 missing modality 场景缺乏设计——FuseMoE 在少模态组合下甚至比全模态更低（FuseMoE 3-modality ACC=59.52 反而高于 full-modality 但依然不如 Flex-MoE）。
  - **多模态 Missing Modality 方法**：ShaSpec [60] 使用 spectral attention 增强跨模态特征；mmFormer [76] 使用 transformer-based attention fusion。这些方法虽声称处理 missing modalities，但未考虑 observed modality combination 与 missing modality 之间的关系。
  - **全栈执行例子（以 ADNI full modality "IGCB" 预测为例的 baseline FuseMoE）**：
    - **模型推理算法层**：FuseMoE 接收 4 个 modality 完整样本 → modality-specific encoders 编码 → MoE layer 做 sparse routing（所有 expert 对任意输入 token 均可被激活，无 modality combination-specific specialization）→ MLP head 输出 AD 分类。对缺失模态采用 zero-padding → encoder 训练被合成零值干扰，批次内不同 modality combination 导致 encoder 接收低质量输入。无 missing modality bank，无课程学习排序，无 expert 的 generalization→specialization 两步训练。
    - **系统框架层**：PyTorch 原生实现，batch-wise training，无特殊 serving/scheduling 优化。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel，MoE gate routing + expert FFN 为标准 GEMM。
    - **硬件架构层**：NVIDIA A100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Flex-MoE 方法**：核心是让 SMoE 架构显式建模 modality combination，分三阶段解决 baseline 缺陷：
    1. **Missing Modality Bank (解决 zero-padding/imputation 缺陷)**：构建 learnable embedding bank `B ∈ R^(2^|M|-1 × |M| × d)`，每个条目对应一种观测模态组合下的缺失模态 embedding。编码器仅用 observed 样本训练（不含合成数据），缺失部分按当前样本的观测模态组合索引查找 bank → encoder 训练质量不受 zero-padding 影响，bank 自动学习"缺少某模态时应该补充什么信息"。实验中 embedding bank 从 cosine similarity 验证了"共享更多观测模态的组合有更相似缺失 embedding"（full "ICBG" 与 "ICB" 相似度 0.56，与 "IC" 仅 0.46）。
    2. **Expert Generalization → Specialization (解决无 modality combination awareness 缺陷)**：训练分两阶段——(a) Warm-up 阶段：样本按可用模态数降序排列（课程学习），先用全模态样本 + G-Router + load/importance balancing loss 训练所有 expert 的通用知识；(b) Specialization 阶段：S-Router 通过 cross-entropy loss `L_ce = -Σ MC(x_j) log(max(S_Router(x_j)))` 将 top-1 gate 强制绑定到当前样本的 modality combination expert index，其余 top-(k-1) expert 做 load/importance balancing → 每个 expert 既保有全模态的通用知识，又获得自身模态组合的专有知识。实验验证：expert BCG 激活最多的两个输入 tokens 是 BCGI (通用) 和 BCG (专有)，expert BCI 激活最多的是 BCGI 和 BCI。
    3. **课程学习排序 (解决训练不稳定性)**：按 modality 数量降序排列样本先易后难 → 先 generalize 后 specialize，优于随机排列和升序排列（消融实验验证：降序 ACC=66.11, 随机=62.65, 升序=63.87）。

  - **全栈执行例子（Flex-MoE 在 ADNI 的 "IGC" (3-modality, missing B) 预测）**：
    - **模型推理算法层**：样本 i 有 {I, G, C} 三个模态输入 → (1) 各 encoder 仅用对应 modality 的 observed 数据训练，I 经 3D-CNN 得 e_i^I，G 经 ResNet-34 得 e_i^G，C 经 MLP 得 e_i^C；(2) 缺失 B 从 missing modality bank B[mc_idx=1 "IGC"][B] 取 embedding；(3) concat 得 h_i ∈ R^(4×128) → Transformer + SMoE layer。S-Router 计算 gate_logits，top-1 通过 L_ce 绑定到 expert_1 (MC index "IGC")，剩余 top-3 按 load balancing 选择 → y_i = gate_1·f_1(h_i) + gate_a·f_a(h_i) + gate_b·f_b(h_i) + gate_c·f_c(h_i)；(4) MLP head → 输出 Dementia/CN/MCI 分类概率。
    - **系统框架层**：PyTorch 实现，使用 batch_size=8 训练，50 epochs (5 warm-up + 45 specialization)，ADNI 16 experts/4 attention heads/128 hidden dim/MIMIC-IV 32 experts/3 attention heads/128 hidden dim。与 baseline 关键差异：encoder 训练与 modality completion 解耦（不需要 imputation），SMoE layer 的 routing 受 modality combination 监督信号约束。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel。与 baseline 的关键差异：GFLOPs 约 59.07（vs FuseMoE 59.76-59.74），因 SMoE 稀疏激活仅计算 top-k expert 的前向路径，参数量 36.9M（vs FuseMoE 340.9M）约减少 89%。
    - **硬件架构层**：NVIDIA A100 GPU。

  - **关键性能对比**（ADNI full modality "IGCB"）：
    - ACC: Flex-MoE 66.11 ±1.14 vs best baseline MAG 61.44 ±1.16 (+7.6%), vs FuseMoE 59.52 ±1.00 (+11.1%)
    - Macro-F1: Flex-MoE 64.73 ±2.01 vs best baseline MAG 61.38 ±1.32
    - AUC: Flex-MoE 81.67 ±0.54 vs best baseline mmFormer 73.93 ±5.97
    - Mean time/iter (IGCB): Flex-MoE 16.00s vs FuseMoE 20.71s (−22.7%)
    - # Params (IGCB): Flex-MoE 36.9M vs FuseMoE 340.9M (−89.2%)
    - 消融实验 (ACC): Flex-MoE 66.11, w/o Expert Specialization 62.75, w/o ES+EG 62.49, w/o embedding bank 63.87, w/o sorting (random) 62.65, w/o sorting (ascending) 63.87
