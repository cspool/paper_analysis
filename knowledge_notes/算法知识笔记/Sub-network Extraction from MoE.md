## Sub-network Extraction from MoE

术语解释
Sub-network Extraction 是从大型稀疏 MoE 模型中按任务或条件提取仅包含部分 experts 的子网络直接用于推理部署的方法，与知识蒸馏（将 MoE 压缩为稠密模型）形成对比。

术语是什么？
与传统 MoE 推理（需加载全部 E 个 experts）不同，sub-network extraction 利用路由策略（如 task-level routing）使特定任务仅需要少量 experts，从而提取 sub-network 独立部署。核心公式：推理时 decoder 参数从 ΣE（全部 experts）降至 K（每 task 激活的 experts）。

Kudugunta et al. (2021) 的关键发现：蒸馏 token-MoE→dense 仅保留 32% BLEU 增益，而 task-MoE sub-network extraction 保留 **100%** BLEU 增益（且 decoder 参数量更小：25M vs 142M distilled dense model）。

从算法pipeline角度拆解术语。
```
# Token-MoE Inference (baseline): 需全部 E experts
for each decoding step:
    y_s = sum(TopK(Softmax(GATE(x_s)), k=2)[e] * FFN_e(x_s))
    # 不同 token → 不同 experts → 需加载全部 E experts + all-to-all 通信

# Task-MoE Sub-network Extraction (proposed):
# Step 1: 根据 task_id 确定 sub-network experts
task_experts = TopK(Softmax(GATE(task_emb[task_id])), k=2)  # e.g., {5, 17}

# Step 2: 仅加载 sub-network experts 到加速器
load_experts({FFN_5, FFN_17})  # K=2 vs E=32/128

# Step 3: 解码，所有 token 使用相同 experts
for each decoding step:
    y_s = G[5] * FFN_5(x_s) + G[17] * FFN_17(x_s)
    # 无 all-to-all，无跨设备通信
```

术语一般如何实现？如何使用？
- 适用于 task boundary 明确的 multi-task 场景
- 实现需修改 MoE router 接受 task_id 作为输入（而非 token embedding）
- 多 task 并行：不同 task 的 sub-networks 分配到不同设备，独立解码
- 与蒸馏的关系：sub-network extraction 是蒸馏的上位替代（当 task boundary 已知时），保留 100% MoE 增益，而蒸馏仅保留 ~32%

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---
