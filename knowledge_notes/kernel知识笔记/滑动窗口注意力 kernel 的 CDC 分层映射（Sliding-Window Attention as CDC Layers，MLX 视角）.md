## 滑动窗口注意力 kernel 的 CDC 分层映射（Sliding-Window Attention as CDC Layers，MLX 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sliding-Window Attention（SWA）是注意力矩阵只在局部窗口 W 内非零（token 只 attend 前 W 个 token）的线性复杂度注意力变体，用于长序列/流式推理降低 O(N²) 复杂度。MLX 论文把它作为"蝴蝶 kernel 之外的第二类结构化 workload"证明 MLX 不限于 FFT/蝴蝶类同构 kernel：SWA 的 tile 计算虽混合不同原语（矩阵累加、归约、指数、归一化），其数据流仍可表达为少量 CDC 层、每层阶段严格对齐、依赖链相邻——直接映射到 MLX 在同一 2D 阵列上的折叠执行（Fig.12）。
- 本地知识库旁证：已有 Sliding-Window Attention 条目（算法知识笔记）与 Hybrid Windows Attention（多方向滑动窗口注意力，EasyAnimate 语境）覆盖算法/模型视角；本条目补充 MLX 特有的 kernel/硬件映射视角——SWA 作为 CDC 分层的 kernel 调度对象。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SWA tile 的 CDC 分层 kernel 调度（MLX 折叠执行，窗口 W、块 Q）：
```
# CDC 层序列（相邻依赖链，每层只消费前层 CDC 边界输出 + tile 本地状态）：
# 层0: 窗口化 score 累加（QK^T，FMA 主导）
for i in 0..N-1:
    for j in max(0,i-W)..i:
        S[i,j] = sum_k Q[i,k] * K[j,k]        # FMA-dominant
# 层1: 行向 max 归约
m[i] = max_j S[i,j]                           # FMAX
# 层2: 指数与归一化统计（FEXP + sum/broadcast）
P[i,j] = exp(S[i,j] - m[i]) ;  l[i] = sum_j P[i,j]
# 层3: 加权累加与归一化（SV，FDIV/FMA）
O[i,:] = (sum_j P[i,j] * V[j,:]) / l[i]
```
调度要点：不同层压不同 FU 原语（FMA/FMAX/FEXP/FDIV）→ tagged-block 执行利用异构性；折叠使 CDC batch 部分 in-flight，层间通信只经显式 CDC-boundary xfer 操作（可检查、有界）；层粒度延迟窗口覆盖下达到稳态重叠，并发活跃层数有界（避免全矩阵中间驻留）。实验：SWA 上 MLX 平均归一化加速 3.6×/2.3×（vs AGX Orin/RTX-3090，batch 32，W/Q 两参数扫），FMA 利用率 43%-75%（vs GPU 10.8%-31%/8.9%-28%）；剩余缺口主要来自窗口 KV 流量的带宽损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) GPU 通用实现——SWA 通常作为融合 attention kernel（FlashAttention 变体支持窗口 mask，或流式 chunk 组织），O(N·W·D) 计算；(2) MLX 实现——把 SWA tile 编译为 4 个 CDC 层（上述伪代码）的 tagged blocks，折叠到 4×4 网格，FMA/FMAX/FEXP 异构单元在活跃窗口内重叠，全部层间通信经 CDC-boundary xfer；(3) 使用场景——长序列/流式 transformer 的注意力加速，验证 MLX 从"蝴蝶同构 kernel"扩展到"混合原语结构化 kernel"的通用性。局限：窗口 KV 流量仍占带宽（利用率天花板），更细动态模式需 predicated transfer（mask/segment 编码）与额外控制状态（论文 E 讨论的灵活性-效率权衡）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
