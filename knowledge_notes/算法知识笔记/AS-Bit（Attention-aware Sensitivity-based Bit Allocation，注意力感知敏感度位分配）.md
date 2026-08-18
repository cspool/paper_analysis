## AS-Bit（Attention-aware Sensitivity-based Bit Allocation，注意力感知敏感度位分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AS-Bit 是 Omni-LUT（ISCA 2026）提出的 Key cache 自适应位分配算法。动机：BC-UQ/BCQ 对所有通道用固定位宽，但并非所有 Key 通道对 attention score 计算同等重要——每通道分布不同、对最终 QK^T 点积贡献不同，固定低位宽预算均匀惩罚所有通道，尤其伤害少数高敏感关键通道（小模型上更严重）。AS-Bit 的核心思想：给对量化误差更敏感的通道分配更高位宽，其余通道用低位宽。敏感度定义：attention A=QK^T 中 Key 的量化误差 ΔK 被对应 Query 项幅值放大，故有效敏感度 = 通道固有量化误差 × 对应 Query 通道能量。量化指标：(1) per-channel Query 能量 E[Q²]_d=(1/T_cal)Σ_{t=1}^{T_cal}(Q_{t,d})²（T_cal 校准 token 数）；(2) Key 在位宽 b 下的 per-channel 量化误差 MSE_b[d]=(1/T_cal)Σ_t(K_{t,d}−K_{q,b}[t,d])²；(3) 边际增益 ΔJ_d=E[Q²]_d·(MSE_{bℓ}[d]−MSE_{bh}[d])——从低位宽 bℓ 升到高位宽 bh 的 Key 误差减少量，用 Query 能量加权。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 离线校准流程：用校准数据对 Key 做双路径量化（dual-path quantization）分别算 MSE_{bℓ}[d] 与 MSE_{bh}[d]，并统计 Query 能量 E[Q²]_d → 对每通道算 ΔJ_d → 取 ΔJ_d 最大的 top k%（论文用 25%）通道分配高位宽 b_h，其余用低位宽 b_ℓ → 生成 per-channel 位分配表（如 b_h=4、b_ℓ=3，则 Key 有效位宽 = 0.25×4+0.75×3 = 3.25 bit；论文在 KV4 配置下给出 25% 高位 → 有效 4.25 bit）。在线：BEA 按位分配表逐通道用对应位宽编码 Key。效果（Fig.4）：仅 10-30% 通道用高位宽即可接近全高位精度；比只考虑 Key MSE 的分配收敛更快；跨模型（OPT/LLaMA2 等）趋势一致。直觉：ΔJ_d 中 E[Q²]_d 捕获"该通道误差被 Query 放大多少"、MSE 差捕获"加位宽能减多少误差"，两者乘积是加位宽的真实收益。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：PyTorch 离线校准（一次每模型），硬件侧由 BQU 的 Key Path 消费 per-channel 位分配结果（每个 Key 通道按分配位宽用 BEA 编码）；配合 LUT-based GEMM 加速器可处理可变位宽的灵活性。用途：任何 per-channel KV cache 量化 + 可变位宽 LUT 加速器的组合；相比 KIVI/KVQuant/Oaken 依赖 sparsity-based outlier 保留（有效 KV 位宽 4.8-5.0 bit），AS-Bit 不加任何额外位（Value 不加位）就达到更高有效位宽效率。论文未明确说明 bℓ/bh 的具体取值表与 top-k 超参搜索过程（仅给 25% 与 b=3/4 配置）。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
