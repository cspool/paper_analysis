## ADAPT (Adaptive Dynamic Allocation for Parallel Tree / 自适应动态并行树分配)

术语是什么？通过联网搜索让回答具体和精准。
ADAPT是DFVG提出的budget-constrained integer programming方法，用于在speculative decoding中动态构建token tree。与传统静态预定义branch配置（如SpecInfer每层固定k分支）不同，ADAPT根据draft model在每个位置的confidence（概率分布）和FPGA硬件并行度约束（总branch预算B、每层最大branch数k_max、最小深度D_min=⌈T_verify/T_draft⌉）动态决定tree结构。目标函数max Σ p_{i,j,l}·x_{i,j,l}最大化expected accepted tokens。由于整数规划NP-hard，ADAPT使用Temperature-Controlled Probabilistic Gumbel Sampling作为贪心近似，时间复杂度O(D·k_max·log k_max)满足实时推理要求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ADAPT算法（BuildTreeADAPT，每iteration执行）：
```
Input: prefix S, draft model M_d, budget B, k_max, D_min, temperature T, threshold τ
Output: token tree T

// 1. Draft model forward → vocab-level probabilities
probs[1..V] = M_d(S)

// 2. 逐层构建tree
T = {root}
for depth d = 1 to D_max:
    N_d = {}  // candidate extension set
    for each node j in layer d-1:
        for token l where p_{d,j,l} > τ_d:
            // Path Cumulative Probability
            P_cum(d,j,l) = p_{d,j,l} · Π_{(k,a_k)∈path(d,j)} p_{k,par(a_k),a_k}
            N_d.add((d, j, l, P_cum))
    
    // Temperature-controlled Gumbel sampling
    for each candidate in N_d:
        P̃_cum = exp(P_cum / T) / Σ exp(P_cum / T)  // softmax
        G = -log(-log(U)) + log(P̃_cum)  // Gumbel, U~Uniform(0,1)
    
    k_i = min(k_max, |N_d|)
    selected = argmax_k(G, k_i)
    
    for each s in selected:
        T.add_node(child_of(s.parent), s.token)
    
    if total_nodes(T) >= B and d >= D_min: break

return T
```
关键：k_max设为FPGA PE array支持的并行数（如8/16/32）；D_min=⌈T_verify/T_draft⌉确保draft-verify可pipeline overlap；T控制exploration（T→0→deterministic top-k，T大→随机探索）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在DFVG中ADAPT实现在FPGA的Branch Management和Token Management硬件模块中。draft model forward后→Token Management计算confidence→Branch Management执行ADAPT→PE array并行生成多分支tokens。配置通过yaml文件（B, k_max, D_min, T, τ）。Qwen3-0.6B/8B pair上维持75%-85% acceptance rate，draft length呈long-tail分布自动适配不同token难度。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

