## Gating Function / Token Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gating Function（门控函数/路由函数）是 MoE 层中决定每个 token 被分配到哪个（些）expert 的核心组件。输入为 token hidden state x ∈ R^M，输出为路由决策——包括每个 expert 的选中概率和 top-k 索引。不同路由函数在 MoE 训练稳定性和模型质量上有显著差异。FSMoE 预实现了 4 种主流路由函数：

1. **GShard Routing** (Lepikhin et al., 2020): 使用带噪声的 Top-k Gate——g(x) = softmax(KeepTopK(x·W_g + N(0,1)·Softplus(x·W_noise), k))，噪声帮助训练初期探索不同的 expert 分配。
2. **Sigmoid Routing** (Lewis et al., 2021 / BASE): 使用 sigmoid 替代 softmax，expert 输出按 σ(x·W_g)_i 缩放——若输出有益于训练目标则 gate 值增大，形成正反馈。
3. **X-MoE Routing** (Chi et al., 2022): 对 hidden state 做低秩投影 W_proj 后与 expert embedding W_g 做余弦相似度 s_i = cos(W_proj·x, W_g)，缓解表示坍缩问题。
4. **Expert Choice (EC) Routing** (Zhou et al., 2022): 从 expert 视角独立选择 top-k token，即 g(x) = softmax(KeepTopK((x·W_g)^T, k))，与 token-choice 路由对称。
5. **Top-P Routing** (Huang et al., 2024 / HMoE): 动态激活不同数量的 expert per token，而非固定 k。将 router 输出概率 P 从高到低排序，若最高概率 $P_{\max} \ge p$（threshold, e.g. 0.6），仅激活 1 个 expert；否则逐步累加直到累积概率 $\ge p$ 为止：$t = \operatorname{argmin}_k \sum_{j \le k} \tilde{P}_j \ge p$。核心优势：简单 token 可能仅需 1 个 expert（省计算），复杂 token 可激活更多 expert（保证质量），与 HMoE 的异构设计天然协同——二者都旨在按 token 复杂度差异化分配计算资源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 Gate 子模块的抽象接口和执行流程：

```
# Gate 子模块处理流程（以 GShard routing 为例）
输入: hidden_states = [B, L, M]  # batch × seqlen × d_model
Gate 参数: W_g [M, E], W_noise [M, E]

# Step 1: 计算干净 logits
logits = hidden_states @ W_g              # [B, L, E]

# Step 2: 添加可学习噪声 (GShard 特有)
noise = randn(B, L, E) * softplus(hidden_states @ W_noise)
logits_noisy = logits + noise

# Step 3: KeepTopK + Softmax
logits_topk = KeepTopK(logits_noisy, k)   # 非 top-k 位置置 -inf
gate_probs = softmax(logits_topk)         # [B, L, E]

# Step 4: 输出路由索引和概率
topk_idx = argtopk(gate_probs, k)         # [B, L, k]
topk_prob = gather(gate_probs, topk_idx)  # [B, L, k]
```

FSMoE 的 Gate 模块支持即插即用切换——调用 `LinearGate(gate_type="gshard")` 即可选择路由函数，无需修改下游的 Order/Dispatch/Expert 模块。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Gating Function 在主流 MoE 系统中通常作为 MoE block 的一部分实现。HuggingFace Transformers 的 `MixtralSparseMoeBlock` 使用标准 softmax top-k gate。FSMoE 通过 `GateBase` 抽象基类统一各种路由实现，用户可通过继承 `GateBase` 并实现 `forward()` 方法添加新路由函数，调度器通过在线 profiler 自动适配。FSMoE 在 Testbed-B（32×RTX2080Ti）上验证 4 种路由的端到端训练时间：相比 DeepSpeed-MoE，FSMoE 在 GShard/X-MoE/Sigmoid/EC 四种路由上分别获得 1.37×/1.42×/1.37×/1.33× 加速。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Layerwise Recurrent Router for Mixture-of-Experts
- LocMoE: A Low-overhead MoE for Large Language Model Training
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

**RMoE 的 GRU-based Cross-Layer Routing**：
RMoE 引入一种全新的路由范式——跨层循环路由。与标准 router 每层独立计算 gating scores 不同，RMoE 在每层 router 前插入跨层共享的 GRU 单元（state dim p=128），将路由决策从独立逐层计算改为跨层循环依赖。核心流程：$x_i' = \mathrm{Proj}_i(x_i)$（逐层独立投影降维），$h_i = \mathrm{GRU}(x_i', h_{i-1})$（共享 GRU 结合历史路由状态），$\mathrm{score}_i = \mathrm{softmax}(h_i \cdot G_i)$（基于 GRU 输出计算 gating）。关键设计：(1) 逐层独立 Proj_i（因为不同层 hidden state 分布差异大）；(2) GRU 跨层共享以引入跨层信息；(3) GRU 额外提供 Recurrent Gradient 路径优化 router 训练；(4) 该设计正交于现有路由方法，可与 XMoE/DeepSeekMoE 无缝组合。RMoE 仅引入额外 ~3.5M 参数（相对于 0.91B 模型），训练速度降低 <1%。代码开源：https://github.com/qiuzh20/RMoE。

**LocMoE 的 GrAP (Grouped Average Pooling) Routing**：

GrAP 是一种固定正交权重的门控计算方式，替代传统可学习 Dense 层。核心思想是将 token hidden state x_m ∈ R^d 均分为 n 组（n = expert 数量），每组取均值作为对应 expert 的门控值：

$$\text{gate\_logits}_i = \text{mean}(x_m[i \cdot d/n : (i+1) \cdot d/n]), \quad i \in [0, n-1]$$

等价于固定正交权重矩阵 ω 与 x_m 的内积，其中 ω_{i,j} = 1{i·d/n ≤ j < (i+1)·d/n} else 0。

GrAP 的关键特性：
- **正交性**：不同 expert 的 gating weight 相互正交（〈ω_i, ω_j〉= 0 for i ≠ j），使不相关 token 更可能被路由到不同 expert，增强语义区分能力。正交性也是 LocMoE 理论推导 expert capacity 下界的必要前提（满足 Lemma 2：各 expert 等概率被选）。
- **计算效率**：GrAP 仅需 O(d) 的均值计算，而 Dense 层门控需要 O(d·n) 的矩阵乘法，无需可学习参数（ω 固定为 0/1 矩阵）。
- **Top-1 路由**：i* = argmax_i(softmax(gate_logits))，仅激活概率最大的 expert。

GrAP 本质是将 Dense 门控简化为空间池化，在 PanGu-Σ 的 1.085T 参数 MoE 模型上验证了与 SwitchMoE (Dense gate) 和 HashMoE (无学习参数 hash) 的对比。

**Hunyuan-Large 的 Recycle Routing**：Hunyuan-Large 使用 mixed routing strategy——1 个 shared expert（所有 token 消费）+ 16 个 specialized experts（top-1 激活）。为解决 top-1 路由中 token dropping 问题，提出 Recycle Routing：对因 expert capacity overflow 被丢弃的 token，随机重新分配到未满 capacity 的其他 specialized experts，从而保留关键信息、提升训练稳定性。相比直接丢弃，recycle routing 确保每个 token 都参与梯度更新。

Flex-MoE 提出了两种独特的 Router 设计：**G-Router (Generalized Router)** 和 **S-Router (Specialized Router)**。G-Router 在 warm-up 阶段使用全模态样本训练，遵循标准 top-k gating + load/importance balancing loss；S-Router 在 specialization 阶段通过 cross-entropy loss $L_{ce} = -\sum_j MC(x_j) \log(\max(S\text{-Router}(x_j)))$ 将 top-1 强制绑定到目标 modality combination expert index，其余 top-(k-1) expert 继续使用 load/importance balancing。这种设计使每个 expert 同时具备通用知识（来自全模态样本）和专有知识（来自特定 modality combination 样本）。
