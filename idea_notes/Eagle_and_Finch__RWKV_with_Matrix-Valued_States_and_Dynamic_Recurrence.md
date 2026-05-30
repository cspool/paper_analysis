## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- baseline方法是什么？
  Baseline是RWKV-4（Receptance Weighted Key Value），一种基于线性注意力的RNN架构，具有O(1) per-token推理和O(N)可并行训练的特性。RWKV-4核心机制：(1) Token Shift（静态learned lerp: lerp_□(a,b) = a + (b-a)⊙μ_□），使模型按channel分配新旧信息比例；(2) WKV attention with channel-wise additive decay: wkv_t = (Σ_{i=1}^{t-1} exp(-(t-1-i)w+k_i)⊙v_i + exp(u+k_t)⊙v_t) / (Σ exp(·))，每channel有独立learned decay rate w；(3) Sigmoid receptance作为归一化门控；(4) vector-valued state s∈R^D (head size=1, 相当于per-channel scalar state)。

  Baseline全栈执行例子（RWKV-4推理时生成一个token）：
  - 算法pipeline：输入token x_t → embedding lookup → L层RWKV block（每层: Pre-LayerNorm → Time Mixing [Token Shift(lerp): r_t,k_t,v_t = lerp_□(x_t,x_{t-1})W_□ → WKV: scalar decay, vector state s∈R^D的分母归一化 → σ(r_t)⊙wkv_t] → residual → Pre-LayerNorm → Channel Mixing [Token Shift → key/value projection → ReLU²(v_t) → σ(r'_t)⊙v'_t] → residual）→ LM head → logits → next token。每token计算O(1)，state size=5DL。
  - 系统框架：PyTorch + HuggingFace Transformers。训练支持time-parallel（沿序列维度并行，因RWKV-4的WKV可写成前缀和形式）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（RWKV-4使用标准PyTorch实现或基础CUDA kernel，无SRAM-resident优化）。
  - 硬件架构：NVIDIA GPU集群，论文未涉及RTL/模拟器层面。

  Baseline (RWKV-4) 缺陷：
  1. **状态表达力受限**：RWKV-4使用vector-valued state s∈R^D（head size=1），每个channel是标量state。这限制了模型记住和区分不同类型信息的能力，因为所有特征维度共享同一标量状态空间
  2. **分母归一化不稳定**：RWKV-4的WKV使用分母归一化（类似attention中的softmax分母），数值上可能在长序列中不稳定，且分母的除法操作增加计算开销
  3. **静态decay缺乏上下文感知**：decay rate w是learned但static的vector，对所有输入token使用相同decay行为，无法根据token内容动态调整信息保留/遗忘策略
  4. **Token Shift是静态的**：RWKV-4的Token Shift使用learned但data-independent的μ_□向量，新旧信息分配比例与输入内容无关
  5. **Sigmoid receptance限制梯度流**：Sigmoid激活在饱和区梯度接近零，可能限制深层网络的训练效率
  6. **MQAR（多查询联想记忆）能力不足**：Arora et al. (2023)实验表明RWKV-4在MQAR任务上存在性能差距，模型维度与序列长度之间存在相关性限制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两步渐进式改进：Eagle (RWKV-5) 和 Finch (RWKV-6)。

  **Eagle (RWKV-5) 的创新**：
  (a) Matrix-valued states: head size从1扩展至64（h=D/64），state变为s∈R^{(D/h)×(D/h)}的矩阵，每个head独立维护K^TV矩阵作为记忆库。这使state大小从5DL暴增至66DL（~13倍），提供更丰富的记忆存储空间
  (b) LayerNorm替代分母归一化：用per-head LayerNorm（等价GroupNorm with h groups）替代attention分母，消除除法操作，数值更稳定
  (c) SiLU gating + 移除Sigmoid receptance: receptance直接作为线性注意力中的query（无激活函数），添加独立的SiLU gate控制输出
  (d) 改进的参数初始化：针对不同参数类型使用差异化初始化策略（如time_decay初始化为-6+5·(i/(D-1))^{0.7+1.3r₀}），确保训练初期良好的数值分布

  **Finch (RWKV-6) 的创新**：
  (e) Data-dependent Token Shift (ddlerp): 将静态lerp替换为ddlerp_□(a,b) = a + (b-a)⊙lora_□(a+(b-a)⊙μ_x)，其中lora(x) = λ + tanh(xA)B。A∈R^{D×32}, B∈R^{32×D}是低秩矩阵，使token shift量成为输入内容的数据依赖函数
  (f) Time-varying decay w_t: decay从静态w=exp(-exp(ω))变为w_t=exp(-exp(d_t))，其中d_t = lora_d(ddlerp_d(x_t, x_{t-1}))，每个channel的decay rate在每个时间步根据当前和前一token的内容动态变化

  论文方法全栈执行例子（Finch推理时生成一个token）：
  - 算法pipeline：输入token x_t → embedding lookup → L层Finch block（每层: Pre-LayerNorm → Time Mixing [ddlerp Token Shift: r_t,k_t,v_t,g_t = ddlerp_□(x_t,x_{t-1})W_□, LoRA A∈R^{D×32}/B∈R^{32×D} → 计算d_t → w_t = exp(-exp(d_t)) → WKV: 矩阵state s∈R^{(D/h)×(D/h)}, s_t = diag(w_t)·s_{t-1} + k_t^T·v_t, wkv_cur = diag(u)·k_t^T·v_t → LayerNorm(r_t·(wkv_cur+s_{t-1})) → SiLU(g_t)⊙output → concat所有head → W_o output projection] → residual → Pre-LayerNorm → Channel Mixing [ddlerp → LoRA-augmented key/value → ReLU² → sigmoid gate] → residual）→ LM head → next token。每token O(1)计算+O(D²/h) state memory。Decay w_t和Token Shift现在都是data-dependent的。
  - 系统框架：PyTorch + HuggingFace Transformers。训练时有custom CUDA kernel将state操作保持在SRAM中，沿非时间维度并行。也有纯PyTorch time-parallel实现（基于GLA的associative scan方法）。
  - 编译框架：论文未明确说明。
  - kernel调度：Custom CUDA kernel for WKV computation：沿非时间维度并行+SRAM-resident state管理，避免反复HBM↔SRAM传输。Finch kernel在16k序列时比Flash Attention v2快4.2×，比Mamba省17%内存（A100 80GB）。
  - 硬件架构：NVIDIA A100/H800 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（状态表达力受限）→ Eagle引入matrix-valued states（每head的K^TV∈R^{64×64}），这等价于为每个head提供64个独立通道的记忆存储，每个通道可独立编码不同类型的信息模式。直觉上，K作为行选择器（input gate），V作为行值，矩阵的每个元素存储特定(输入通道, 值类型)pair的记忆。内部state从5DL膨胀至66DL（~13×），模型记忆容量大幅提升。Table 18消融（170M模型在Pile上训练330B tokens）证实RWKV6-Pile（avg 50.7%）超越RWKV4-Pile（47.7%）和Pythia（47.9%），接近Mamba（50.1%），证明了矩阵state的收益。
  - 缺陷2（分母归一化不稳定）→ Eagle用per-head LayerNorm替代分母除法。LayerNorm对每个head的WKV输出做减均值除标准差的归一化，数值范围稳定可预测，消除长序列中分母可能发散的风险。同时LayerNorm等效于GroupNorm on h groups，不引入跨head依赖。
  - 缺陷3（静态decay缺乏上下文感知）→ Finch引入data-dependent time-varying decay w_t = exp(-exp(d_t))。d_t由LoRA（低秩矩阵A∈R^{D×64}, B∈R^{64×D}）基于ddlerp后的输入生成，使decay rate在每个时间步、每个channel上根据输入内容动态调整。直觉上，重要token可以"标记"自己为需要更长保留时间（减小decay），不重要token可以加速遗忘（增大decay）。这使模型具备选择性记忆能力：在需要精确回忆的历史tokens上保持低decay，在无关tokens上加快遗忘。MQAR实验（Figure 4）显示Finch在MQAR任务上显著超越所有已知的非Transformer架构。
  - 缺陷4（Token Shift是静态的）→ Finch引入ddlerp：首先Eagle token shift（a+(b-a)⊙μ_x）对输入进行静态预调制，然后lora(a+(b-a)⊙μ_x) = λ + tanh((a+(b-a)⊙μ_x)A)B产生数据依赖的调制偏移。A,B是低秩矩阵（rank=32），参数开销小（~2×32×D per token-shift）。这允许模型根据输入内容决定从历史和当前token各吸收多少信息。Table 19 DDLerp消融证实完整DDLerp（loss=2.91）优于仅decay上DDLerp（2.923）和完全无DDLerp（2.926）。
  - 缺陷5（Sigmoid receptance限制梯度流）→ Eagle移除receptance的Sigmoid激活，使其直接作为线性注意力中的query项（类似标准attention的Q），梯度可通畅传播。同时引入独立的SiLU gate g_t来控制输出幅度，SiLU具有非饱和梯度的优势（x>0区域梯度线性）。这改善了训练效率和模型表达能力。
  - 缺陷6（MQAR能力不足）→ Finch在MQAR上的高准确率来自两个机制协同：(1) matrix-valued state提供更丰富的记忆存储；(2) data-dependent decay允许模型对关键token做选择性记忆增强。在MQAR任务中，模型可通过ddlerp识别key token并保留其信息（降低对应channel的w_t），通过matrix state的对应行存储value信息，最终通过receptance作为query精确检索。

  渐进式改进的效果证据：
  - RWKV-4→Eagle: Table 3多语言avg从51.8→54.3 (1.5B) / 53.9→56.5 (3B) / 56.4→58.2 (7B); Table 4英语avg从59.2→62.4 (1.5B) / 64.1→66.0 (3B) / 67.3→71.5 (7B)。Figure 5长上下文loss显著下降。
  - Eagle→Finch: Table 3多语言avg从54.3→55.0 (1.5/1.6B) / 56.5→57.1 (3B); Table 4英语avg从62.4→62.9 (1.5/1.6B) / 66.0→67.5 (3B)。MQAR上Finch达到极高高精度。
