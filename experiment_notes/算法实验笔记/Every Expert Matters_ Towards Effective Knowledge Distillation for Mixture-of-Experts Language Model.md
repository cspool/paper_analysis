## Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MoE 教师模型的知识蒸馏方法，包含两个核心创新：(1) **Knowledge Augmentation (KA)**：对每个输入进行 M 次前向传播，每次以概率 λ 随机采样 N-1 个 expert、以概率 1-λ 取 Top N-1 个 expert，从而增广来自不同 expert 组合的知识。使用 student 生成的 pseudo-target 和 reverse KL divergence 进行蒸馏。(2) **Student-Aware Router (SAR)**：先以 student 反馈（reverse KL divergence + auxiliary load balancing loss）训练 MoE 教师的路由器，再使用更新后的路由器聚合所有 expert 输出进行蒸馏。路由器训练中所有 expert 均被激活，输出通过加权求和聚合。

  实验比较：(a) KD baseline (forward KL, Sanh 2019) vs 论文方法；(b) GKD baseline (reverse KL + on-policy) vs 论文方法；(c) ALL（直接激活所有 expert 不训练 router）作为 SAR 的消融；(d) KA 中 M（增广样本数）和 λ（采样概率）的消融实验；(e) 评估 Sheared-Llama 2.7B 密集教师 vs Llama-MoE 教师的效果对比。

- 硬件平台是什么，配置是什么。
  4 张 Intel Gaudi v2 加速器，使用 SynapseAI 1.18.0 框架。

- 模型是什么。数据集和bench分别是什么。
  教师模型：Llama-MoE-3.5B (4/16, 2/8 variants) 和 Llama-MoE-3.0B (2/16)；密集教师：Sheared-Llama-2.7B。学生模型：Sheared-Llama-1.3B（密集模型）。数据集：Dolly (databricks-dolly-15k, 14k train / 500 val / 500 test)、SelfInst (252条)、Vicuna (80条)、S-NI (SUPER-NATURALINSTRUCTIONS test set 9k条)、UnNI (UNNATURALINSTRUCTIONS core set 10k条)。评估指标：ROUGE-L（5个随机种子取平均）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 KAIST，发表于 arXiv 2502.12947。**论文未提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。算法核心流程如下：

  **=== Knowledge Augmentation (KA) ===**
  输入: student qθ, 数据分布 p_x, 教师前向次数 M, 训练步数 K, 学习率 η
  输出: 训练后的 student θ

  MoE 教师前向传播使用 Noise Top-k Gating (Shazeer et al., 2017):
  ```
  H(x)_i = (x·W_g)_i + StandardNormal() · Softplus((x·W_noise)_i)   // gate logits
  G(x) = Softmax(KeepTopK(H(x), k))                                     // gate probs
  y = Σ_i G(x)_i · E_i(x)                                              // expert output aggregation
  ```

  KA 将 expert 选择扩展为 N-1 个:
  ```
  E = { Sampled N-1 experts (按 gate prob 采样)  w.p. λ,
        Top N-1 experts                        w.p. 1-λ }
  KA(v, E)_i = { v_i  if expert i ∈ E,
                 -∞   otherwise }
  G^KA(x) = Softmax(KA(H(x), E))
  ```

  KA 训练流程（Algorithm 1）:
  ```
  For each step k = 1..K:
      Sample request x from p_x
      Sample response y from qθ(·|x)          // student 生成 pseudo-target
      For m = 1..M:                            // 增广 M 次
          y_teacher = MoE_forward_with_KA(x)   // 用 KA 策略选 N-1 experts
          θ ← θ - η·∇ D_KL(qθ(·|x) || p_KA(·|x))  // reverse KL + student on-policy
  ```

  **=== Student-Aware Router (SAR) ===**
  SAR 在每次迭代包含两个阶段：路由器更新 + 知识蒸馏。

  SAR 路由器训练（Algorithm 2）:
  ```
  For each step k = 1..K:
      Sample request x from p_x
      Sample response y from qθ(·|x)          // student 生成 pseudo-target
      // 阶段1: 路由器更新（仅更新 W_g 和 W_noise）
      W_g ← W_g - η·∇ L_SAR
      W_noise ← W_noise - η·∇ L_SAR
      // L_SAR = D_KL(qθ(·|x) || p_all(·|x)) + β·L_b
      // 其中 p_all 使用所有 experts（全激活）
      // L_b = CV(m)² + CV(P)²  是 load balancing loss
      // 阶段2: 知识蒸馏
      y_teacher = MoE_with_updated_router(x)  // 用更新后 router，激活所有 experts
      θ ← θ - η·∇ D_KL(qθ(·|x) || p_SAR(·|x))
  ```

  超参数：λ=0.05, M=2, β=0.01, batch size=16, LR=1e-5 (student & router), epochs=10, max_seq_len=512, AdamW optimizer。temperature=1.0, top-k=0, top-p=1.0 for generation。

  算法pipeline分为两部分：QESC（量化）和 PESF（剪枝）。

  **=== QESC: Quantization with Expert-Selection Calibration ===**
  输入: MoE模型M（FP16），校准数据集C（WikiText2 128条序列），目标位宽B（expert 2/2.5/3-bit）
  输出: 量化后的MoE模型M_q
  ```
  For each layer l in [0..L-1]:
      1. 量化 MHSA 模块到 4-bit（使用 group-wise GPTQ, group size=128）
      2. 获取该层输入 x_l（从校准集前向传播）
      3. 路由器校准:
         对 MoE layer l 的每个 router:
           保存原始 router 输出标签: y_full = W_r * x_l  （W_r是全精度router权重）
           获取量化后输入: x_hat_l（经过量化MHSA和已量化expert的激活）
           计算 TopK-MSE Loss:
             L = (1/K) * Σ_{i∈top-K(W_r*x_l)} ((W_r*x_l)_i - (W_r*x_hat_l)_i)²
           优化 router 权重 W_r 以最小化 L（仅微调router，补偿量化干扰）
      4. 量化该层所有 experts 到 B-bit（GPTQ, group-wise asymmetric, group size=128）
      5. 保持 router 为原始精度（FP16）
  ```

  **=== PESF: Pruning based on Expert-Selection Frequency ===**
  输入: MoE模型M（已量化或全精度），输入序列seq（长度l），阈值α
  超参数: 每层N个expert，每token选K个expert
  ```
  For each MoE layer with N experts:
      计算阈值: T = (l * K / N) * α
      统计序列中该层每个expert被选中的次数 c_i
      For each expert i:
          if c_i < T:
              剪枝 expert i（不计算其输出）
          else:
              正常计算 expert i 输出
      对未剪枝的expert输出加权求和得到最终输出
  ```

  **=== 组合使用 EAC-MoE = QESC + PESF ===**
  ```
  1. 离线阶段: 使用QESC对模型进行3.03-bit量化（MHSA 4-bit, experts 3-bit, router FP16）
  2. 在线推理（prefill阶段）:
     对每个输入序列:
       For each MoE layer:
           使用量化expert权重 + BitBLAS混合精度计算
           执行PESF动态剪枝（α=0.3）
           仅计算被保留的expert
  ```

  **关键数学公式：**
  
  量化重建问题: argmin_{W_q} ||WX - W_qX||_2²  (GPTQ使用Hessian近似 H=2XX^T)
  
  专家选择概率（token x）: r = {r_0,...,r_{N-1}}, s = Softmax(r), 选top-K
  
  MoE输出: z = Σ_{j=0}^{K-1} (s_{e_j} / Σ_{i=0}^{K-1} s_{e_i}) · E_{e_j}(x)
  
  TopK-MSE Loss: L = (1/K) Σ_{i∈top-K(W_r*x)} ((W_r*x)_i - (W_r*x̂)_i)²
  
  剪枝条件: c_i < (l*K/N) * α  → 剪枝expert i

  **核心数据流**：
  全精度输入 x → 量化MHSA(4-bit) → Router(top-K选择+动态剪枝决策) → 量化Expert(B-bit, GPTQ group-wise) → 加权求和输出
  校准过程使用BitBLAS处理量化权重的混合精度BLAS操作。K值设置：Phi3.5-moe=8, Deepseek-moe-16b-base=20, Qwen1.5-MoE-A2.7B=20（通过MMLU上的网格搜索确定最优K值）。
