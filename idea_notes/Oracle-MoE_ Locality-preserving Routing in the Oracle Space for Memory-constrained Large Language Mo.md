## Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

- baseline方法是什么？
  - Baseline：Switch Transformer（token-level MoE routing）。每个 token 的嵌入 t_t ∈ R^d 通过线性 gate W_g ∈ R^{N×d} 独立投影到 N 个专家分数空间：g(t_t) = softmax(W_g · t_t)，然后选择 top-k 个专家。由于 token embedding 被 token-identity 特征主导（如图 2 左：不同 token ID 形成不同 cluster），相邻 token 即使语义相似也会被路由到不同专家，造成极高的 inter-token expert activation variation（CSD_token 大），在内存受限的边缘设备上导致频繁的 expert swapping（50%-85% 总延迟来自 I/O）。
  - Baseline 全栈执行例子（Switch Transformer, 729M, 8 MoE 层 × 16 专家, Jetson Xavier NX 8GiB GPU, 单用户请求, batch=1）：
    - **算法层**：每个生成的 token 独立通过 W_g 矩阵乘法做 top-1 routing → 相邻 token 的 W_g · t_t 投影在高维空间中方差极大 → 几乎每 2 个连续 token 就切换一次专家。
    - **系统框架层**：load-on-demand 策略（FIFO/LRU/SwapMoE）管理有限 GPU 内存中的专家 → 因为 token-level routing 变化频繁，几乎所有专家都需要在内存中轮转 → SwapMoE 离线统计专家频率但仍无法应对稀疏、不一致的激活模式。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：标准 Transformer FFN kernel（矩阵乘法 + SwiGLU），专家计算延迟 L_compute 固定，但 I/O 延迟 L_swap 主导总延迟（占 50%-85% 甚至 >99%）。
    - **硬件架构层**：Jetson Xavier NX 384 核 Volta GPU，8GiB 显存。完整 729M 模型需全部显存；50% 内存预算时仅能驻留约一半专家。Switch Transformer + FIFO/LRU/SwapMoE 在 50% 内存下的延迟是 full-model 的 15-30 倍。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Oracle-MoE 用基于 Oracle Space 的路由替代 token-level routing。核心洞察：连续 token 具有语义局部性（semantic locality），但 token embedding 被 token-identity 特征主导，掩盖了高层语义的局部性。Oracle-MoE 通过以下步骤提取高层语义并保持路由一致性：
    1. **语义组划分**：利用注意力分数矩阵（Q·K^T 内积）发现高层语义相关性——a_ij > ε 的 token 归入同一语义组。
    2. **Oracle Space 构建**：语义组嵌入 z_S = mean(token embeddings in S)，保留高层语义、压制 token-identity 噪声（理论证明 Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)）。
    3. **Oracle Space 路由**：在 Oracle Space 上做 K-means 聚类（k = 专家数），每个聚类中心对应一个专家。连续 token 的高层语义平滑变化（图 4），因此路由到相同专家的概率大幅提高。Theorem 1 证明 CSD_oracle < CSD_token 以高概率成立。
    4. **Expert Prediction 优化**：利用第一层 embedding 预测深层专家激活（准确率 85%-95%），预加载专家，进一步减少 10%-15% latency。
  - 对比 baseline 全栈执行例子（Oracle-MoE, 729M, 8 MoE 层 × 16 专家, Jetson Xavier NX 8GiB GPU, 单用户请求）：
    - **算法层**：不再用 token embedding 做 gate，而是：注意力分数矩阵 → 贪心语义组划分 → 组嵌入平均 → SVD 降维 → K-means 聚类中心最近邻 → 同一语义组内所有 token 路由到同一专家。因为一个序列（1024 token）通常只有不到 5 个语义组且同一序列的语义组往往属于同一聚类，连续数百个 token 可不切换专家（图 6）。
    - **系统框架层**：load-on-demand + 低 expert swapping 需求 → 只需在语义组切换时才加载新专家。因为 Oracle-MoE 的 CSD_oracle 极低，不同 swapping 策略（FIFO/LRU/SwapMoE）对延迟影响很小，论文以三种策略的平均值报告结果。
    - **编译框架层**：论文未明确说明。
    - **Kernel层**：路由计算从 W_g · t_t（token 级矩阵乘法）替换为 z_reduced = W_svd · z_S（组嵌入降维）+ ||z_reduced - c_k|| 欧氏距离计算（聚类中心最近邻）。降维后的低维空间使计算开销可忽略（2.5e-4s vs token-level 的 1e-4s，相比单次 forward-backward pass 的 3.5s 可忽略）。
    - **硬件架构层**：Jetson Xavier NX 384 核 Volta GPU，8GiB 显存。Oracle-MoE 在 25% 内存预算下（仅驻留 1/4 专家）仅比 full-model inference 多 3s 延迟；Switch Transformer 延迟增加高达 2000%。50% 内存预算时 Oracle-MoE 几乎无额外延迟。First token latency：Oracle-MoE 4.910s vs Switch+FIFO 22.395s / Switch+LRU 23.428s / Switch+SwapMoE 12.767s。下游任务性能：Oracle-MoE 平均持平甚至略优于 Switch Transformer（如 729M: Ours 36.35 vs Switch 35.86 avg score）。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（token-level routing 受 token-identity 主导 → 高 CSD → 频繁 expert swapping）→ Oracle Space Routing：用语义组嵌入（压制 token-identity 噪声，保留高层语义）替代 token embedding 做路由，CSD_oracle 远小于 CSD_token（理论保证 Theorem 1 + 实验验证激活不一致性从 53-82 降至 4-6 per 100 tokens）。
    - 缺陷2（load-on-demand 策略无法应对不可预测的激活模式）→ Semantic Locality Preservation：连续 token 的高层语义在 Oracle Space 中平滑缓慢变化（图 4），同一语义组内所有 token 路由到同一专家。即使话题切换的跨数据集 scenario，Oracle-MoE 仍仅每 100 token 换 12.2 次 vs Switch 的 90.54 次。
    - 缺陷3（Prefill 阶段也需频繁 swapping → 高 first token latency）→ Oracle-MoE 在 prefill 阶段对一个输入仅激活 1-2 个专家，仅需一次加载。First token latency 降至 4.910s（Switch+FIFO 22.395s）。
    - 缺陷4（无法预测深层专家激活 → 无法预加载）→ Expert Prediction：用第一层 embedding 预测深层专家激活准确率 85%-95%，进一步减少 10%-15% latency。此可预测性源自 Oracle-MoE 路由与高层语义的强关联。
