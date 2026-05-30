## Sliced Shapley Value (SSV / 切片 Shapley 值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sliced Shapley Value (SSV) 是 CoKV 提出的一种高效近似 Shapley Value 的方法，用于在 LLM 中评估每个 attention head 的合作贡献。传统 Shapley Value 需要枚举所有 coalition size j ∈ {1,...,n} 的期望 marginal contribution，计算复杂度为 O(n·T/ε²)（n 为 head 总数，T 为单次验证集推理时间）。SSV 的洞察是：在 LLM 中，不同 coalition size j 下 head 的 expected complementary contribution 分布高度相关（附录 B.3 实验验证），因此只需计算少数代表 coalition size H ⊆ {1,...,n} 的 complementary contribution 即可准确捕获 head 的相对重要性。

公式定义：$\mathcal{SSV}_i^{\mathcal{H}} = \frac{1}{|\mathcal{H}|} \sum_{j \in \mathcal{H}} \mathcal{SV}_{i,j}$，其中 $\mathcal{SV}_{i,j}$ 是 (i,j)-coalitions 的 expected complementary contribution。CoKV 使用 H={32,64,96,128}（n=256），计算复杂度降为 O(|H|·T/ε²)。

SSV 的关键优势：(1) 利用 complementary contribution U(S)-U(N\S) 可同时更新 coalition S 中所有 head 的估计值，一次采样助攻多个 head；(2) 只需 |H|≪n 个 coalition size 即可稳定估计，理论保证为 (ε,δ)-approximation；(3) 分布对称性（coalition size s 和 n-s 的分布几乎相同）使只需计算 s<n/2 的 coalition size。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SSV 计算的伪代码（Algorithm 1）**：

```
Input: Heads N = {h_1,...,h_n}, 采样次数 M, coalition sizes H={j_1,...,j_k}
Output: SSV_i^H for each head h_i

// 初始化累计矩阵
SV_{i,j} = 0, m_{i,j} = 0  for i,j in 1..n

for k = 1 to M:
    π^k = random_permutation(1..n)        // 随机排列 heads
    j = random_choice(H)                   // 随机选 coalition size
    S = {π^k(1), ..., π^k(j)}             // 前 j 个 heads 构成 coalition
    
    // 计算 complementary contribution
    U_S = model_accuracy(mask N\S)         // mask S 以外的 heads
    U_NS = model_accuracy(mask S)          // mask S 中的 heads
    u = U_S - U_NS
    
    // 更新 coalition S 中所有 head 的估计
    for t = 1 to j:
        SV_{π^k(t), j} += u
        m_{π^k(t), j} += 1

// 平均得到 SSV
for i = 1 to n:
    SSV_i^H = (1/|H|) * sum_{j in H} (SV_{i,j} / m_{i,j})

return {SSV_1^H, ..., SSV_n^H}
```

**SSV 在推理中的使用（Budget Allocation）**：
```
// Hyperparameters
α = #heads with zero extra budget (hyperparam, {1,5,10,15,20,30,40})
B = shared budget, s = local window size

// Step 1: Min-max normalize SSV
min_α = α-th smallest SSV
max_SSV = max(SSV)
NSV_i = 0  for α smallest SSV heads
NSV_i = (SSV_i - min_α) / (max_SSV - min_α)  for remaining n-α heads

// Step 2: Proportional allocation
c_i = B * (NSV_i / sum(NSV_j)) + s    // cache size for head h_i
```

术语一般如何实现？如何使用？

SSV 计算是 offline 预计算过程：在验证集（随机划分 15% 数据）上运行。CoKV 使用 8×RTX 3090 GPU 并行计算不同 coalition size。250 samples/coalition size 时 MAE<1/256（约 20.93 小时），满足精度要求。推荐进行两次独立采样，MAE<1/n 时取平均作为最终 SSV。SSV 具有任务特异性——不同 task 的 SSV 分布差异显著，但同 task 类型内泛化性好（附录 B.4 交叉验证）。推理时根据用户所选 task 加载对应的 SSV 分数表。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---
