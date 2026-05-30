## Differential Expert Caching（差分专家缓存）

术语是什么？
Differential Expert Caching 是 Diff-MoE 提出的面向 batched MoE 推理的三级 GPU 缓存层级架构。其核心思想是：观察到 MoE 推理中 expert 激活具有显著的全局局部性（global locality：少数 expert 在整个推理过程中频繁激活）和时间局部性（temporal locality：某些 expert 在短解码窗口内反复激活），因此不应平等对待所有 expert——而是将 GPU 显存组织为三个不同优先级的缓存层，每层采用不同的管理策略以最大化缓存命中率和减少 host-GPU 数据传输。

三级缓存结构：
1. **High-Priority Cache (HPCi)**：每个 MoE layer i 独立持有，容量默认 2 experts（α=5% 缓存比下）。永久存储 globally hot experts（离线微调阶段按激活频率识别的 top-N experts），推理全程不驱逐。
2. **Medium-Priority Cache (MPCi)**：每个 MoE layer i 独立持有，容量为 HPCi 的 2 倍（默认 4 experts）。动态管理 locally hot experts（在线推理阶段因近期激活而优先级超过阈值的 experts），采用 Locality-Preserving Cache Replacement 策略驱逐降级 experts。
3. **Low-Priority Cache (LPC)**：所有 MoE layers 共享的临时缓冲区。分为两部分——当前层激活 experts 的加载缓冲区和预测下一层 experts 的预取缓冲区。当前层计算完成后，激活缓冲区中的 locally hot experts 可能晋升到 MPCi，其余 experts 立即驱逐以释放空间。

从系统架构角度拆解术语：
在 Diff-MoE 的在线推理循环（见 Algorithm 1）中，差分缓存的运作流程为：

1. **初始化（Lines 1-9）**：离线阶段统计各 MoE layer 的 expert 激活频次，每个 layer 取 top-N（默认 N=2）存入 HPCi 并设为 max priority（MaxP=2）。所有 non-global experts 优先级初始化为 0，参数留在 host DRAM。
2. **Gating 与 Expert 查找（Lines 10-16）**：Gating network 输出 activated experts 集合 A。对每个 E ∈ A，三级顺序查找：HPCi → MPCi → LPC。只有三级全部 miss 才触发 host→GPU 传输（cudaMemcpy 到 LPC）。
3. **Token Dispatch 与计算（Lines 17-22）**：Loaded experts 执行 SwiGLU FFN 计算（已缓存在 HPC/MPC 的命中 experts 无需等待加载，实现零延迟命中）。
4. **缓存更新（Lines 23-24）**：Locality-Preserving Replacement 将 LPC 中满足条件的 locally hot experts 晋升到 MPCi。LPC 清空（仅保留预取缓冲区的下一层 experts）。
5. **下一层预取（Lines 25-28）**：GRU predictor 预测下一层 likely experts，top-1~2 未缓存 experts 异步预取到 LPC。

关键参数：缓存比 α=5%（HPC 占 2/128, MPC 占 4/128 per layer），MaxP=2, threshold_hot=1, Δ_inc=1, Δ_dec_in=0.4, Δ_dec_out=0.2。

与 global shared cache（MoE-Infinity/LRU）的关键区别：Diff-MoE 使用 per-layer 独立 HPC/MPC 而非跨层共享，避免了一层的大量激活 experts 挤占其他层的缓存空间，在 batched 场景下维持稳定的 per-layer 命中率（batch=128 时仍保持 28.5% hit rate，而全局缓存崩溃至 <0.1%）。

术语一般如何实现？如何使用？
Diff-MoE 在 NVIDIA FasterTransformer v5.2 上实现三级缓存。具体实现方式：
- **参数粒度拆分**：将 HuggingFace 格式的整个模型 .bin 文件拆分为每个 expert 一个独立文件，支持 per-expert 粒度的 host↔GPU 传输。
- **缓存容量管理**：HPCi 和 MPCi 使用固定大小的 GPU memory pool，LPC 总量动态分配（当前层激活 experts 数变化）。总 GPU 内存占用约 16% of No-Offload（α=5% 时）。
- **替换策略实现**：维护 per-layer 的 priority list，每次 MPI update 时在 Python/PyTorch 层执行 priority score 更新和替换逻辑，替换仅在各 activated experts 在 MPCi 中全部 miss 时触发批处理。
- 适用场景：(1) 资源受限的单 GPU MoE 推理（GPU 显存不足以容纳全模型）；(2) 大 batch serving（batch_size ≥ 2），通信瓶颈严重时效果最佳；(3) 需高吞吐的 MoE 推理服务。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
