## Linearizing_Large_Language_Models

- baseline方法是什么？
  Baseline是标准softmax Transformer（Llama2-7B、Mistral-7B），使用因果多头自注意力（MHA with softmax(QK^T/√d)）作为token mixing机制。这些模型在高质量、大规模预训练数据上训练了数万亿tokens（Mistral约8T，Llama2约2T），在下游NLU benchmark上表现最强。同时对比各种从零预训练的循环模型（RWKV-5、Mamba、RetNet），这些模型使用线性注意力或状态空间模型实现O(1)推理，但训练成本高且性能落后于同参数量的Transformer。

  Baseline全栈执行例子（Llama2-7B/Mistral-7B推理时生成一个token）：
  - 算法pipeline：token → embedding lookup (1×4096) → L层Transformer block（每层: RMSNorm → MHA: W_Q/W_K/W_V投影 → RoPE应用到Q/K → causal softmax(QK^T/√d) → weighted sum V → W_O output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。每生成一个token需要对所有历史token计算QK内积（O(N) attention计算），且每层维护K/V cache ∈ R^{N×d_head}，总KV-Cache = 2·n_layers·N·D（32层7B模型：每token约512KB cache，10K context需约5GB）。
  - 系统框架：PyTorch + HuggingFace Transformers或vLLM。推理时使用FlashAttention-2优化attention计算，PagedAttention管理KV-Cache内存。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用FlashAttention等标准GPU kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **推理成本随序列长度线性增长**：softmax attention每个生成步骤需要访问完整KV-Cache并计算QK内积，FLOPs和内存访问均为O(N)，长序列推理延迟高。而RNN可用固定大小hidden state实现O(1) per-token推理。
  2. **从零训练循环模型成本极高**：Mamba/RWKV需要从头预训练（1-6T tokens），且受限于可用数据和计算资源，难以匹配强Transformer（如Mistral在8T高质量数据上训练的）的性能。
  3. **T2R等现有转换方法不稳定且无法扩展**：Kasai et al. (2021)的T2R方法通过MLP近似attention，但大规模uptraining时出现数值不稳定性（分母归一化、梯度问题），仅在小模型（~100M）上验证，且需约20%预训练tokens。
  4. **线性注意力本身的归一化问题**：传统线性注意力的分母归一化（除以Σsim(q,k)）在长序列中可能发散或数值不稳定，如TransNormer (Qin et al., 2022a)所指出的。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SUPRA（Scalable UPtraining for Recurrent Attention），将强预训练Transformer通过有限的继续训练（uptraining，仅需~5%预训练tokens）转换为RNN，而非从头训练。核心设计：(a) 用可学习MLP kernel（φ(x)=ReLU(Wx+b)，Q和K共享权重）替换softmax；(b) 用GroupNorm替换传统线性注意力的分母归一化（借鉴RetNet），解决训练稳定性；(c) 引入RoPE相对位置编码增强位置建模能力；(d) 使用固定衰减向量γ（借鉴RetNet）给更近的token更高权重。

  论文方法全栈执行例子（SUPRA推理时生成一个token，7B参数）：
  - 算法pipeline：token → embedding lookup → L层SUPRA block（每层: RMSNorm → Q/K/V投影 W_Q/W_K/W_V → MLP kernel: φ_q=ReLU(RoPE(qW+b)), φ_k=ReLU(RoPE(kW+b)) → 循环状态更新: s_i=diag(γ)·s_{i-1}+φ_k_i·v_i^T [O(1) per token，s∈R^{d_h×d_h}，固定大小] → GroupNorm(φ_q_i^T·s_i) → W_O output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。每生成一个token仅需O(1)计算，内存固定（s矩阵），无KV-Cache增长。
  - 系统框架：PyTorch + OpenLM fork（https://github.com/TRI-ML/linear_open_lm），集成Lightning Attention 2的Triton kernel。训练用FSDP分布式策略在H100集群上运行。
  - 编译框架：论文未明确说明。
  - kernel调度：使用Lightning Attention 2的Triton kernel实现高效线性注意力计算（训练时沿序列维度并行）。推理时切换为循环模式，O(1) per-token。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(N)推理成本）→ 线性注意力的循环形式将KV-Cache替换为固定大小的matrix state s∈R^{d_h×d_h}，每次更新为s_i=diag(γ)·s_{i-1}+φ(k_i)·v_i^T，读取为φ(q_i)^T·s_i，均为O(d_h²)常量操作。推理时无随序列长度增长的内存或计算开销。这使得SUPRA在理论上具备RNN的无限长度推理能力。
  - 缺陷2（从零训练成本高）→ SUPRA不需要从头训练。它从强预训练Transformer（Llama2/Mistral）初始化，仅用20B-100B tokens uptraining（约占5%预训练成本），即可达到与从零训练1.2T tokens的Mamba-7B竞争的性能（Mistral-SUPRA +100B avg 64.0 vs Mamba-7B avg 64.7）。这大大降低了研究线性模型的实验成本。
  - 缺陷3（T2R不稳定）→ 三项关键改进：(a) **归一化替换**：T2R用分母Σsim(q,k)做归一化，该分母在训练中可能发散/变为零导致梯度不稳定。SUPRA用GroupNorm（per-head, h个group）替代，每个head的输出独立做减均值除标准差，数值范围稳定可预测。Table 3消融直接证明了归一化策略的关键性——T2R uptraining 1B模型性能崩溃（HellaSwag 40.6），而SUPRA保持57.0（接近原始模型的62.1）；(b) **位置编码**：引入RoPE作为相对位置编码（φ(k)和φ(q)在MLP kernel后进行旋转），而T2R缺乏显式位置建模；(c) **decay因子**：γ^{i-j}衰减给近端token更高权重，模拟softmax中的位置偏置，同时短上下文性能更好。
  - 缺陷4（线性注意力归一化发散）→ GroupNorm方案有两个优势：一是数值稳定（每个head独立归一化，无累积操作），二是无需像RetNet那样维护额外的归一化状态（z_i向量）。消融实验（Table 3）证明：与T2R的基于除法的归一化相比，GroupNorm使uptraining从性能崩溃中恢复。

  训练时的并行-循环对偶性：
  - 训练（并行模式）：利用线性注意力的可并行性，使用Lightning Attention 2的Triton kernel做沿序列维度的并行计算（类似标准Transformer训练），避免BPTT。这使训练效率与标准Transformer可比。
  - 推理（循环模式）：纯RNN形式，s_i = diag(γ)·s_{i-1} + φ(k_i)·v_i^T，GroupNorm(φ(q_i)^T·s_i)，O(1) per token。
  - 两种模式在数学上等价，切换无需额外微调。

  不足之处（论文明确记录）：
  - **MMLU/In-context learning退化**：Mistral-SUPRA +100B的MMLU 5-shot仅34.2（vs Mistral原模型62.4），这是线性模型的已知弱点（Akyurek et al., 2024），表明线性化后失去了in-context learning能力。
  - **长上下文性能不达理论预期**：Table 2显示虽然SUPRA模型在超出训练长度后性能不崩溃（vs Transformer），但绝对性能仍显著低于经过位置编码扩展的Transformer（Llama2 + YaRN），且"decay因子限制了有效上下文窗口"。

  从SUPRA的训练流程看设计理念：
  ```
  Step 0: 选择最强的可用预训练Transformer（Mistral-7B）
  Step 1: 在每层attention中添加MLP kernel参数 (W, b)
  Step 2: 将attention计算替换为: GroupNorm(Σγ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)
  Step 3: 5%预训练tokens的uptraining（RefinedWeb, 100B tokens）
          Adam optimizer, cosine LR 3e-5→1e-5, 1000步warmup, FSDP
  Step 4: 推理时自动支持循环模式（数学等价，无需额外处理）
  Step 5: 获得约64% avg的7B RNN，训练成本仅为从零训练的1/20
  ```
  核心洞察：**不需要去近似softmax attention，直接替换为线性attention同时通过uptraining让模型适应新的计算范式**。附录A的热力图分析证明SUPRA的线性attention矩阵与原始softmax矩阵差异很大——模型学到的是不同的计算策略而非softmax的近似。
