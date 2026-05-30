## QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

- baseline方法是什么？
  - Baseline 方法：PTQ (Post-Training Quantization) 方法——以 PTQ4DM、Q-Diffusion、PTQ-D 为代表的典型扩散模型后训练量化方法。这些方法通过小规模校准集计算量化参数（scaling factor s、zero-point Z），尝试平衡裁剪误差（clipping error）和舍入误差（rounding error），实现 W8A8 高位宽下的有效量化。PTQ 在低比特下失效的核心原因：(1) **激活分布不均衡**：扩散模型激活值大多数集中在零附近，但存在稀疏的大值（例如范围 [-10, 34] 但多数值在 [-0.6, 1.7]），这些大值对生成质量很重要（消融实验表明破坏最大值 token 会导致图像严重退化）；(2) **低比特下舍入误差主导**：在 4-bit 下，舍入误差远大于裁剪误差，导致 PTQ 方法优化过程中过度裁剪（over-clipped），产生损坏图像；(3) **理论失效**：Proposition 3.1 指出基于重建的 PTQ 方法在低比特下失去理论保证——激活扰动 Δ 太大导致 Taylor 展开不准确（ḡ ≠ 0，不能简化为二次型）；(4) **无法按时间步动态调整**：部分方法虽支持分时间步量化参数，但需要存储多组参数抵消效率收益。
  - 全栈执行例子（Baseline: Q-Diffusion PTQ on LDM-4 LSUN-Bedrooms W4A4）：
    - **算法pipeline**：全精度 LDM-4 模型加载 → 构建校准集（真实图像经 encoder 得 latent，5120 样本） → 校准各层量化参数 s_w, s_a：W̃_ij = clamp(round(W_ij/s_w)+Z_w; 0, 15), x̃_k = clamp(round(x_k/s_a)+Z_a; 0, 15) → AdaRound 优化舍入策略 → 分块/分层重建：min_s Σ MSE(FP_block_output, Q_block_output) → W4A4 量化模型 → DDIM 200 步生成 → FID 评估。核心缺陷：由于 4-bit 仅 16 个量化等级，大值被过裁剪或舍入，W4A4 下 Q-Diffusion 生成失败（表 3、4 中 FID=N/A）。
    - **系统框架**：论文未明确说明特定 Serving 框架，PTQ 为离线量化流程，量化后的模型可用 PyTorch 直接推理。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT4 × INT4 矩阵乘法 kernel（cuBLAS 或类似），量化-反量化在计算前后插入。
    - **硬件架构**：NVIDIA A6000 GPU，标准的 Tensor Cores/INT4 推理路径。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuEST = 选择性渐进权重微调 + 数据无关训练 + TLA + CMA + 全局损失。对应解决 Baseline 缺陷：(1) **解决激活分布不均衡**：通过微调权重间接调整激活分布（图 2 显示范围从 [-10,34] 缩小到 [-4,14]），消除稀疏大值，使分布更紧凑——既保护了重要的大值不被过度裁剪，又使小值获得更细的量化粒度；(2) **解决低比特下舍入误差/裁剪误差难以平衡**：不直接优化 s 的 trade-off，而是修改模型本身使其在量化约束下仍然能保持性能——Theorem 3.2 将大扰动 Δ 分解为 K 小扰动 ε，证明微调权重 w_n 可使模型对量化扰动鲁棒；(3) **解决 PTQ 理论失效**：微调使模型"学习"量化后的输入分布，相当于在量化扰动样本上重新收敛到局部最小值，恢复 Taylor 展开的准确性；(4) **解决时间嵌入量化退化**：Property ❶——量化时间嵌入导致 FID 上升 15%（W4A8 下从 6.77→7.58），通过 TLA (L_TLA = Σ_{l∈C_TE} E_t[||O(t;w_l) - Õ(t;w_l,s_l)||²]) 微调时间嵌入层权值和量化参数，甚至超越全精度 baseline（FID 5.61 vs FP 6.77）；(5) **解决敏感层退化**：Property ❷——FeedForward 层在 6-bit 即失败（而其他线性层在 4-bit 才失败），通过 CMA (L_CMA = Σ_{l∈C_A} E_t[||O(z;w_l) - Õ(z̃;w_l,ŝ)||²]) 专门微调注意力相关层；(6) **全局损失补充局部对齐**：仅 TLA+CMA 只能对齐局部信息，L_G = E_t[||O(x_t;w) - Õ(x_t;w,s)||²] 提供网络级全局监督，且仅含 L_G 时性能反而退化（FID 退化 7.13），说明局部+全局结合的必要性；(7) **数据无关 + 高效**：校准集完全由高斯噪声通过全精度模型采样构造，无需真实数据；仅微调 <7% 参数（约 1% 时间嵌入 + ~5% 注意力层），时间和显存优于 EfficientDM 和 Full-finetune（0.45h vs 2.60h vs 0.85h）。
  - 全栈执行例子（QuEST on LDM-4 LSUN-Bedrooms W4A4, A6000）：
    - **算法pipeline**：(1) 全精度 LDM-4 加载 → 输入随机高斯噪声 x_T∼N(0,I)，在不同时间步 t 采样获得 256 样本/时间步的校准中间激活；(2) 量化初始化：W̃ = clamp(round(W/s_w^init)+Z_w; 0,15), s_a 初始化为 MinMax → 权重量化参数冻结；(3) 阶段一 TLA：仅微调时间嵌入层 w_TE + 对应 s_a_TE，优化 min ΣE_t[||FP_TE(t;w_TE) - Q_TE(t;w̃_TE,s_TE)||²]，约 0.5% 参数，Adam lr=1e-5；(4) 阶段二 CMA：冻结 w_TE/s_TE，微调注意力相关层 w_A + 所有剩余 s_a，优化 min ΣE_t[||FP_attn(z;w_A) - Q_attn(z̃;w̃_A,ŝ)||²]，约 5% 参数；(5) 阶段三：叠加 L_G = E_t[||FP_final(x_t;w) - Q_final(x_t;w̃,s)||²]，联合优化 min(L_TLA + L_CMA + 2L_G)，全流程 <7% 参数更新；(6) 推理：DDIM 200 步，每步 t → TimeEmbed(t) 经微调层，UNet 经量化层前向，端到端 FID 5.64 (W4A4)。
    - **系统框架**：论文未明确说明特定 Serving 框架，量化模型使用 PyTorch 直接推理。校准和微调流程为离线优化，可独立复现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT4 量化 kernel，矩阵乘法使用量化整数运算，量化/反量化和 clamp 操作在层间插入。无定制 kernel。
    - **硬件架构**：NVIDIA A6000 48GB GPU，Tensor Cores 支持 INT8/FP16 推理路径。
