## Expert Pipeline Scheduler (EPS, 专家流水线调度器)

术语解释
Expert Pipeline Scheduler (EPS) 是一种针对 MoE 推理的细粒度流水线调度策略。通过在 kernel 级别将 MoE FFN（Gate/Up/Down GEMM）的自适应 GEMM 计算与 all-to-all 通信进行流水线重叠，并引入 SM 控制、GEMM 类型动态切换和输入水平切分，显著提升 MoE 模型 prefill 吞吐量。

术语是什么？
EPS-MoE 由三部分组成：
1. **并行策略**：对 Attention 使用 TP 或 DP（取决于注意力机制类型），对 MoE 块使用 EP。MLA Attention 使用 DP+EP（避免 TP 引入额外通信），MHA/GQA/MQA Attention 使用 TP+EP。
2. **Expert Pipeline Scheduler**：水平切分输入（按行）+ 权重按专家切分，将专家分 N 组顺序提交计算。
3. **计算-通信重叠**：通过 SM 控制实现 GEMM 与 all-to-all 在 kernel 级别的 pipeline 并行。

从系统架构角度拆解术语：
EPS-MoE 在一个 MoE 层中的完整调度流程：

```
=== EPS-MoE 调度流程 (MoE layer, DP+EP, 2 GPUs, 6 experts, PN=2) ===

初始化:
  GPU 0: Expert 0,1,2 (EP) + Attention params (DP复制)
  GPU 1: Expert 3,4,5 (EP) + Attention params (DP复制)

Step 1 - Local Attention (DP, 各GPU并行):
  GPU 0: Attention(seq_A_tokens) → x_A
  GPU 1: Attention(seq_B_tokens) → x_B

Step 2 - Gate Routing:
  GPU 0: Router(x_A) → token→expert映射
  GPU 1: Router(x_B) → token→expert映射

Step 3 - Dispatch: ReduceScatter + all2all (分设备发token)

Step 4 - Expert Pipeline (PN=2):
  # GPU 0上的 Expert Pipeline
  Stage A: [all2all E0-tokens] ══16 SM══╗
           [GEMM for E0]        ══116 SM╝  ← 并行
  Stage B: [all2all E1,E2-tokens] ══16 SM══╗
           [GEMM for E1,E2]      ══116 SM╝  ← 并行
  # GPU 1对称执行

Step 5 - Combine: all2all + AllGather 聚合

Step 6 → 下一层
```

并行策略选择：
```
if attention_type == "MLA":        # DeepSeekV2
    strategy = DP + EP             # 避免MLA额外TP通信
elif attention_type in ["MHA","GQA","MQA"]:  # Mixtral/DBRX
    strategy = TP + EP             # TP减少显存、提高计算效率
```

术语一般如何实现？如何使用？
- 集成到 vLLM 框架，替换 MoE FFN 的执行和通信调度路径
- PN 自动选择：argmax_N [min(T_comm/N, T_comp/N) × (N-1) - (kN+b)]，k为线性overhead系数
- 适用场景：prefill（m>1700）效果显著，decode 阶段（m小）效果有限
- 硬件：8xH800-80GB SXM NVLink，最佳PN通过profiling确定
- 与FP8通信协同使用效果更佳
- 局限性：token数过小（m<1700）时pipeline overhead超过收益

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
