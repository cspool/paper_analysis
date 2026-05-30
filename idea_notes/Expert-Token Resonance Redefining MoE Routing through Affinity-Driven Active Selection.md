## Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

- baseline方法是什么？
  Baseline 是标准 MoE 架构下的 **Top-1 Token-Choice Routing (TCR)**，使用 capacity factor 1.1 约束 expert 容量（防止 token drop），搭配 auxiliary loss 做负载均衡。以 Mixtral 8×7B 为 backbone。Baseline 存在三个缺陷：(1) **路由效率瓶颈**：仅由 token 单向选择 expert（TCR），无法保证 expert 获得最适合处理的 token；训练后期 class-irrelevant token 呈各向同性分布时，TCR 成功率受限于 C/s（C 为容量，s 为 token 数）；(2) **Expert 同质化**：TCR 的 top-k softmax routing 无法主动维持 expert 间的专业化分工，导致多 expert 学到相似表征，冗余计算；(3) **All-to-All 通信气泡**：固定容量策略导致部分 expert 过载而其他空闲，padding 浪费计算和通信资源。

  **Baseline 全栈执行例子（以 Mixtral 8×7B + 32 Ascend NPU 下单个 MoE block 的一次训练前向为例）**：
  - **算法层**：输入 batch 包含 s 个 token（seq_len=32768），传统 MLP Router 计算 gate_logits = Softmax(W_g · x_t + ε)，W_g ∈ R^{d×n}（全参数矩阵，O(d²) 复杂度）。Top-1 选择最高 logit 的 expert 分配 token，capacity_factor=1.1 意味着每个 expert 容量上限 C = 1.1 × s/n。超出容量的 token 被 drop（残差连接直通）。auxiliary loss L_aux 惩罚 expert 使用不均。训练过程经历 Phase 1 (Router training，expert 学习接收对应类别 token) 和 Phase 2 (Expert training，expert 学习解决问题)，但两个阶段用同一 TCR 策略。
  - **系统框架层**：基于华为 MindSpeed-LLM（Megatron-LM 风格）训练框架，在 Ascend NPU 上运行。32N 配置：TP=4（张量切分 attention/FFN 权重）、PP=4（流水线切分 32 层 MoE block）、DP=2（数据并行）、EP=2（expert 并行，每张 NPU 持有 4 个 expert）。每步训练中的 All-to-All 通信完成 token 在 EP 维度上的重新分发和聚合——token dispatch（send）将 token 发送至持有目标 expert 的 NPU，token combine（recv）将 expert 输出聚合回原 NPU。
  - **编译框架/Kernel 层**：论文未详细说明编译器栈。CoC（Communication Over Computation）优化前，矩阵乘法和集合通信操作串行执行——即先完成 FFN 的 MatMul 计算，再发起 All-to-All 通信。Ascend 的 AI CORE 负责矩阵乘法/卷积，AI VECTOR CORE 负责向量并行计算，AI CPU 负责专用指令。论文未明确说明 kernel 编译细节。
  - **硬件层**：Ascend 910B3 NPU，单颗 20 AI Cores @ 1.8GHz（fp16 313T 算力），HBM 64GB @ 1.6GHz（1.6T 带宽）。8 颗 NPU 组成 Atlas 800T A2 服务器全 mesh 互联。服务器间通过 NPU 直连网络互联。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **ETR（Expert-Token Resonance）**，通过三方面设计解决 baseline 缺陷：

  **(1) GrAP 路由层替代 MLP Router → 解决计算复杂度和 expert 同质化**
  - 将传统全连接 Router W_g ∈ R^{d×n} 替换为 Grouped Average Pooling 生成的**对角稀疏亲和力矩阵** W_aff ∈ R^{d×n}（非零元仅在 d/n 分组内）
  - 参数量降至 1/n，计算 O(d²/n) vs 传统 O(d²)
  - 正交性天然确保各 expert 对应的 w_i 向量互相正交，防止多个 expert 学到相似路由模式
  - 使用**余弦相似度**（非 Softmax logit）作为亲和力分数 δ_{t,i} = cos(x_t, w_i)

  **(2) TCR+ECR 双向选择 → 解决路由质量和 expert 专业化**
  - Step 1 (TCR)：每个 token 按 δ 选 top-ℓ experts
  - Step 2 (ECR)：每个 expert 从已分配 token 中按 δ 选 top-C tokens
  - Expert 主动选择最相关 token，"共振效应"加速 expert 专业化
  - 理论证明（Theorem 5）：早期训练 TCR 更优（成功率 Θ(C·∑p_i/s)），后期 ECR 更优（成功率 → 1 当 C ≥ 2C*）；双向动态协调最大化全训练过程的成功率

  **(3) 自适应容量 + Locality Loss → 解决通信气泡和负载不均**
  - 容量 C 动态计算：C_min = (1/n)·exp(d·δ_max²/(2−δ_max²))，随训练进度自适应降低（最大降 40%）
  - Locality Loss L_loc = μ·KL(D_c||D_l)：鼓励 token 路由至本地 NPU 的 expert，减少跨节点 All-to-All 通信

  **ETR 全栈执行例子（以 Mixtral 8×7B + 32 Ascend NPU 下单个 MoE block 的一次训练前向为例）**：
  - **算法层**：输入 batch s 个 token，GrAP 生成 W_aff（对角稀疏，仅 d 个参数项，循环移位分组平均池化）。对每个 token t 和 expert i 计算 cos(x_t, w_i) 得亲和力矩阵 δ ∈ R^{s×n}。TCR 阶段：每个 token 取 top-ℓ experts（论文中 ℓ 由亲和力分数和阈值确定）。ECR 阶段：每个 expert i 从分配给它的 token 中 Bottom-C（保留最高亲和力的 C 个），动态 C = max(C_min, adaptive_by_progress)。仅 C 个最高亲和力 token 进入 expert FFN 计算。总 loss = task_loss + α·L_aux + β·L_loc（L_loc = μ·KL(current_dist || local_dist)，local_dist 偏向同节点 expert）。
  - **系统框架层**：基于 MindSpeed-LLM，增加双向路由模块（token dispatch 后并行执行 expert 侧 token 过滤）。Locality Loss 通过感知 EP 拓扑（哪些 expert 在同节点）计算 KL 散度。CoC 优化将 MatMul 和 All-to-All 通信融合为统一 kernel，通过 Ascend MTE（Memory Transfer Engine）的远程内存访问实现流水线并行——计算当前 batch 时预取下一 batch 的通信数据。Token rearrangement 引入的 TopK/IndexPutV2 操作有少量开销，但 FFN MatMul 实测达 17× 加速（相比 baseline）、2.6× 相比 LocMoE。
  - **编译框架/Kernel 层**：CoC 优化在 Ascend CANN（Compute Architecture for Neural Networks）编译框架层面将 MatMul + All-to-All 融合——论文未详细说明 CANN 版本或融合策略细节。Ascend AI CORE 执行 FFN MatMul（Cube 计算单元），AI VECTOR CORE 执行亲和力分数的 cosine 计算和 TopK/BottomC，AI CPU 执行 token rearrangement 的 IndexPutV2 等控制流操作。论文未明确说明 kernel 调度策略细节。
  - **硬件层**：同 baseline 的 Ascend 910B3 NPU 集群。但 locality loss 减少跨服务器通信量（token 优先本地 expert），通信 idle 时间占比下降（见图 5 的 3D 柱状图对比），显存峰值降低 4.57%-16.27%（因容量自适应减少 padding）。

- baseline方法是什么？
  Baseline 是 MoE-LLaVA [25] 提出的 **MoE-tuning** 三阶段方法：Pretraining（MLP Projector 对齐视觉和语言模态）→ Stage II MoE-tuning（复制 FFN 参数初始化多个 expert + 训练线性 router 做 top-k 选择）→ Stage III Instruction Tuning。Baseline 存在两个核心缺陷：(1) **Expert Uniformity**：通过复制（replication）初始化 MoE expert，导致 expert 趋同，失去 MoE 架构的多样化优势——实验证明 shuffle router 几乎不影响性能，说明 expert 之间没有实质差异；(2) **Router Rigidity**：使用静态线性 router 对所有 token 做统一路由，无法区分视觉 token 和文本 token 的差异（KDE 密度图显示两种模态的 logit 分布高度重叠），导致 router 输出对输入模态不敏感。

  **Baseline 全栈执行例子（以 MoE-LLaVA + Qwen-1.8B 单个 token 的前向推理为例）**：
  - **算法层**：视觉 token 经 CLIP-L 编码后通过 MLP Projector 映射到 LLM hidden space。每个 LLM decoder layer 的 FFN 被替换为 4 个 MoE expert（由原始 FFN 复制初始化），线性 router R 对每个 token 计算 logits = W_r · h，Softmax 后选 top-2 experts 做加权求和输出。训练时加载均衡辅助损失 L_aux 惩罚 expert 间 token 分配不均。
  - **系统框架层**：基于 LLaVA 1.5 代码框架实现（PyTorch + HuggingFace Transformers），使用 DeepSpeed ZeRO-2 做分布式训练。FFN 复制 + MoE 层替换在模型初始化阶段完成，不涉及运行时框架修改。
  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution，无自定义编译器 pass）。
  - **Kernel 调度层**：MoE FFN 的 expert 间计算为标准的串行循环或 batch-gemm，单个 token 的 top-2 expert 结果由加权 sum 合并。无自定义 CUDA kernel。通信层面 expert parallelism 使用标准 all-to-all 或 all-gather（论文未详述）。
  - **硬件架构层**：8x NVIDIA A100-80G GPU，无特定硬件加速。Expert 和 router 均在 GPU 通用计算单元上执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  EvoMoE 在 MoE-tuning 的 Stage II/III 分别引入两项算法创新对应解决两个缺陷：

  **(1) Expert Evolution → 解决 Expert Uniformity**

  将"复制初始化"改为"进化初始化"：仅训练一个 FFN（Expert 1），其他 expert 通过 EMA 形式从 Expert 1 的参数和梯度演化而来：
  ```
  θ_n ← β_n · θ_1 + (1 - β_n) · ∇θ_1   (n = 2, 3, 4)
  ```
  β_n 随机采样自不同范围（[0.9,0.99]、[0.8,0.89]、[0.7,0.79]），每个 expert 以不同速率"吸收"梯度信息，从而自然产生功能分化。对比：复制初始化的所有 expert 有相同的起点和几乎相同的梯度轨迹 → 趋同；进化初始化的 expert 因 β 不同而拥有不同的参数方向 → 多样。

  效果验证：独立评估每个演化后的 expert（不用 router）发现 Expert 2/3/4 在多个 benchmark 上一致优于 Expert 1（原 FFN），甚至 β=0.9（仅保留 10% 梯度更新）也能优于 Expert 1，证明演化产生的多样性是有效而非随机的。

  **(2) Dynamic Token-aware Router (DTR) → 解决 Router Rigidity**

  将"静态线性 router"替换为"hypernetwork 驱动的动态 router"：
  ```
  z' = MSA(LN(z_prev)) + z_prev
  Θ_up^τ, Θ_down^τ = H^τ(z')            # 两个 hypernetwork H_V / H_T
  E^τ = Θ_up^τ · SwiGLU(Θ_down^τ · z')   # 动态生成的 up/down 投影
  ρ^τ = φ(E^τ)                            # 最终 linear 层预测 expert 分布
  ```
  关键设计：视觉 token 和文本 token 分别通过不同的 hypernetwork 生成专属投影参数，确保：(a) 模态感知——视觉 token 经 H_V 路由，文本 token 经 H_T 路由；(b) token 级自适应——每个 token 拥有独立的路由计算权重，而非 shared linear layer。可视化表明 DTR 的 expert 分配在不同模态间有明显差异（MoE-tuning baseline 则几乎均匀），实现了"让 visual expert 处理图像、text expert 处理文本"的功能分化。

  **EvoMoE 全栈执行例子（与 baseline 同一 token 的前向推理对比）**：
  - **算法层**：与 baseline 相同流经 CLIP-L + MLP Projector → LLM decoder layers，但在每个 MoE layer：(a) expert 参数由 Expert Evolution 生成（4 个多样化 FFN 而非 4 个近似相同的复制 FFN）；(b) router 由 DTR 替代线性 router：H_V 或 H_T 动态生成投影矩阵 → SwiGLU → φ 输出 expert 概率 → 仅 top-1 expert 被激活（比 baseline 的 top-2 少一半激活参数）。Stage III 仅训练 H_V / H_T / φ（共约额外 34760 参数），experts 冻结。
  - **系统框架层**：同 baseline（LLaVA 1.5 + DeepSpeed ZeRO-2），但在模型定义中将 FFN 层替换为 EvoMoE 层（含 4 个演化 expert + DTR）。Stage II 仅 Expert 1 需要 optimizer state 和梯度，Stage III 仅 DTR 参数需要 optimizer state——训练参数量比 baseline 更少。
  - **编译框架层**：论文未明确说明（与 baseline 相同的 PyTorch eager execution）。
  - **Kernel 调度层**：论文未明确说明（与 baseline 相同的标准 expert computation，无自定义 kernel）。
  - **硬件架构层**：同 baseline（8x A100-80G）。论文未提出硬件层面的修改。
