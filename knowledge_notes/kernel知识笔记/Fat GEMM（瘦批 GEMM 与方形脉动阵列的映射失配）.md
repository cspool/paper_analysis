## Fat GEMM（瘦批 GEMM 与方形脉动阵列的映射失配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMM 记为 (M,K)×(K,N)。长上下文 agentic LLM 推理中，KV cache 容量墙限制 batch，使 M（batch 相关维）远小于 K（隐藏维：LLaMA-3-8B 为 4096、LLaMA-3-70B 为 8192），形成"fat GEMM"（宽而扁：K 长、M 短）。方形 systolic array / Tensor Core 的 tile 假设 M≈N≈K，M 小时只用到阵列窄条，乘法器利用率骤降（图 2：同乘法器数下 8×512 扁平阵列 vs 64×64 方阵可达 FLOPs 差距显著）。FlashAttention 的 per-head GEMM（head_dim 小，如 LLaMA-3-70B 的 128；GQA 一个 K 头对多个 Q 头）是第二种 fat GEMM——计算维度小导致大阵列低利用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FFN fat GEMM：batch M 小、隐藏维 K 长（weight-activation GEMM 沿 K 归约）
Y[M, N] = X[M, K] @ W[N, K]^T      # M=4（容量墙压小的 batch）、K=8192
# 方形阵列 128x128：M 方向只占 4/128 → 利用率约 3%
# PLENA 扁平阵列 (BLEN, MLEN) = (32, 2048) 输出驻留：
#   BLEN=32 对齐 M，K 沿 MLEN 流式推进，PE 部分和驻留，全流水无气泡
#   M_SUM 加法树在 K 归约完成后做一次跨 sub-arr 部分和求和
# FlashAttention per-head fat GEMM（头级分解）：
for head in range(MLEN // HLEN):                  # 多个 Q 头并行
    S[BLEN, BLEN] += Q_head[BLEN, HLEN] @ K_head^T[HLEN, BLEN]
```
- 硬件对齐要素：FFN 阵列在 BLEN 对齐 batch 时利用率最优（Figure 12）；FlashAttention 计算模式与 batch 无关（每 head 固定 (BLEN,HLEN)），头级分解后利用率与有效 batch 解耦——decode 长上下文（有效 batch 小）仍满利用。预填充阶段 FFN 与 FA 都接近满利用，故扁平化收益集中在 decode。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PLENA flattened systolic array + output-stationary 数据流，把长 K 作为流水维、短 M 映射为 BLEN；跨 sub-arr 部分和用结果加法树一次归约（专用 M_SUM 指令，避免逐 tile 气泡）。Scale-Sim 支持矩形/扁平阵列仿真（阵列纵横比影响利用率的非线性结论），SARA 探索可重构阵列形状，但均未针对 autoregressive Transformer 的 fat GEMM + per-head GEMM 组合设计。使用：为 LLM FFN/attention GEMM 选择阵列形状与数据流——长上下文下方形阵列利用率受 M 限制，扁平阵列是 workload 驱动的替代；评估时按"同乘法器数、同 HBM 配置"比较可达 FLOPs 与利用率。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
