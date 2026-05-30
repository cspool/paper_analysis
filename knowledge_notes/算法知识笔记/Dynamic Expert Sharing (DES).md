## Dynamic Expert Sharing (DES)

术语解释
Dynamic Expert Sharing (DES) 是一种将 MoE 优化从 token-centric pruning 转变为 sequence-level coreset selection 的技术。通过识别紧凑、高效用的 expert 子集（coreset）服务整个并行解码 block，最大化 expert 复用，减少 HBM→SRAM weight fetching cost。

术语是什么？
DES 的核心公式：定义 Coreset Selection Function Φ: I → C，将运行时信息 I（如 router logits 或 hidden states）映射到共享 expert coreset C ⊂ {E_1, ..., E_M}。优化目标为：Φ* = argmin |Φ(I)|，满足 A(Φ(I)) ≥ A_base - ε。

DES 算法（Algorithm 1）：
**Stage 1: Sequence-level Consensus** — 通过 Φ 识别 compact high-utility expert set C。
**Stage 2: Constrained Local Routing** — 每 token 仅在 C 内进行 Top-K selection，重新归一化 gate weights。

延迟模型简化为：L_MoE(Φ) ≤ b·|Φ(I)| + a·(N·K)，unique expert weight cost 从 |∪S_n|（随 N 增长）降低为 |C|（与 N 解耦的可控变量）。

两种具体策略：DES-Seq 和 DES-Vote。

从算法pipeline角度拆解术语：
```
# DES Algorithm (Algorithm 1 from paper)
Input: I (sequence info: router logits N×M), Φ (coreset function), σ (activation), K
Output: Y (layer output N×d)

# Stage 1: Sequence-level Consensus
C = Φ(I)  # C ⊂ {1..M}, |C| << M, e.g. |C| ≈ β×M

# Stage 2: Constrained Local Routing
for each token n = 1..N:
    # Route within coreset only
    S_n = TopK(I_n[i] for i in C, K)     # top-K experts from C
    g_n = σ(I_n[i] for i in S_n)          # re-normalize gate weights
    y_n = Σ_{i∈S_n} g_{n,i} · E_i(x_n)   # weighted expert sum

return Y = [y_1, ..., y_N]
```

关键结果（LLaDA2.0-Mini 16B, N=32）：
- DES-Vote (β=0.15): unique experts 84→38 (-55%), MoE layer latency -38.0%, relative accuracy 99.5%
- DES-Vote (β=0.10): unique experts 84→25 (-70%), MoE layer latency 进一步降低, relative accuracy 96.4%

术语一般如何实现？如何使用？
- 无训练（training-free）：直接修改 inference 时的 routing 逻辑，无需重新训练模型
- 参数化：DES-Vote 用 budget factor β 控制 coreset size M_core = β×M；DES-Seq 用 local selection count k < K
- 超参数：β 越小→coreset 越小→memory 节省越大→accuracy 可能降低。β 调节灵活（连续值）
- 系统集成：在 dInfer 等 dLLM 框架中的每 MoE 层插入 coreset selection step
- 自定义 fused kernel 可消除算子碎片化开销
- 发现"re-activation"效应：从 coreset 中重新激活 expert（即使非原始 Top-K）几乎无 marginal cost，可恢复 accuracy

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
