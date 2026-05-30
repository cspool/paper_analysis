## GroupedGEMM in MoE Expert Computation（MoE 专家计算中的分组通用矩阵乘法）

术语是什么？
GroupedGEMM（Grouped General Matrix Multiplication）是 MoE 模型训练和推理中用于加速多个 expert FFN 层并行计算的核心 kernel。在 MoE 架构中，每个 token 被路由到 top-k 个 expert，不同 expert 接收不同数量的 token，因此需要对多个不同形状的小矩阵乘法进行批量计算。GroupedGEMM 将多个 expert 的矩阵乘法（每个 expert 执行 input_tokens × expert_weight）合并到一个 CUDA kernel 中执行，通过 cuFuncSetAttribute 精细控制每个 expert 的资源使用（shared memory、L1 cache、线程数），避免逐个 expert 串行调用的 kernel launch 开销。在 MegaScale-MoE 中，GroupedGEMM 是 SwiGLU FFN 的三次矩阵乘法（fc1、fc3 gate、fc2）的核心计算原语。

从 kernel 调度角度拆解术语：
MegaScale-MoE 中 GroupedGEMM 的完整计算流程（以 SwiGLU FFN 为例）：

```
输入: ffn_in [b*s*k/n, h]  // n 个 GPU 上的 token hidden states
      expert_weights = {fc1_weight_i, fc3_weight_i, fc2_weight_i for i in 1..E}

// Step 1: Token 路由信息预处理
token_to_expert = router(ffn_in)  // 每个 token 的 top-k expert index
expert_token_counts = count_tokens_per_expert(token_to_expert)

// Step 2: Scatter - 按 expert 分组 token
for expert_i in 1..E:
    expert_input_i = gather_tokens_for_expert(ffn_in, token_to_expert, expert_i)
    // expert_input_i shape: [num_tokens_i, h]

// Step 3: SwiGLU 三次 GroupedGEMM
// FC1: input → gate hidden
fc1_outputs = GroupedGEMM({
    expert_1: (expert_input_1, fc1_weight_1),  // [n1, h] × [h, fh] → [n1, fh]
    expert_2: (expert_input_2, fc1_weight_2),  // [n2, h] × [h, fh] → [n2, fh]
    ...
    expert_E: (expert_input_E, fc1_weight_E),  // [nE, h] × [h, fh] → [nE, fh]
})

// FC3 (gate): input → gate values
fc3_outputs = GroupedGEMM({
    expert_i: (expert_input_i, fc3_weight_i) for i in 1..E
})  // 每个 expert: [ni, h] × [h, fh] → [ni, fh]

// SwiGLU activation
fc2_inputs = {SiLU(fc1_outputs[i]) * fc3_outputs[i] for i in 1..E}

// FC2 (down projection): gate output → hidden
fc2_outputs = GroupedGEMM({
    expert_i: (fc2_inputs_i, fc2_weight_i) for i in 1..E
})  // 每个 expert: [ni, fh] × [fh, h] → [ni, h]
```

**GroupedGEMM 的 GPU 执行模型**：
- 单个 CUDA kernel 内通过动态形状处理多个不同 [m_i, k] × [k, n] 的矩阵乘法
- 使用 cuFuncSetAttribute 精细控制每个 expert 的 shared memory、L1 cache 和线程配置
- 输入/输出为动态形状 tensor（因 token 路由不均衡），频繁的动态内存分配可能引起 GPU 内存碎片化

术语一般如何实现？
- PyTorch 生态中的实现方式：
  - Megatron-LM 使用 Python for-loop 逐个 expert 调用 GEMM（简单但 kernel launch 开销大）
  - MegaScale-MoE 使用自定义 CUDA GroupedGEMM kernel，一次 kernel launch 处理所有 experts
  - PyTorch 官方 Triton Persistent Grouped GEMM（2025.08）：persistent kernel + TMA + grouped launch ordering，2.62x vs naive for-loop
  - SonicMoE（2025.12, Dao-AI Lab）：CuTe-DSL 实现，IO-computation overlap + token rounding，1.86x vs ScatterMoE
  - grouped_gemm 库（fanshiqing/grouped_gemm）：支持任意 expert 数量的批量 GEMM
- MegaScale-MoE 的观察：GroupedGEMM 中 expert intermediate dimension 远小于 dense FFN，导致 GPU 利用率低于 dense GEMM；GroupedGEMM 的细粒度资源控制可能引入同步延迟（straggler 来源之一）
- 从 MegaScale-MoE 的计算-通信比公式（R ≈ 3/2 × h_ffn × bandwidth/peak），expert intermediate dimension h_ffn 是决定 training efficiency 的关键参数——h_ffn 越大，GroupedGEMM 的 compute time 相对于 EP communication time 越充裕，越容易实现通信完全隐藏

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
