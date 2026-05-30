## Independent Rotation (for LLM Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Independent Rotation（独立旋转）是 ParoQuant 的核心约束设计：在 n 维通道空间中，选取一组互不重叠的通道对 P={(i_1,j_1),...,(i_m,j_m)}，满足 ∀k≠l: {i_k,j_k}∩{i_l,j_l}=∅（每个通道最多出现在一对中），对每组对施加一个 Givens 旋转。定义为 R(P,Θ)=∏_{k=1}^{m} G(i_k,j_k,θ_k)。由于 pairs 互不重叠，所有 Givens 旋转之间完全独立、无数据依赖、可同步执行。每个 independent rotation 含最多 n/2 对（128 通道时为 64 对），对应 n/2 个可学习参数（仅为 n×n 全正交矩阵 n² 参数的 1/(n-1)）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParoQuant 的 pair 选择算法 (Algorithm A1)：
```
# g: group size, K: number of rotations, N: pairs per rotation
P_all = shuffle({(i,j) | 1≤i<j≤g})  # 所有可能的配对, 随机排列
A[i,j] = 1 for i≠j else 0            # 全局可用性矩阵
P_1..P_K = [[] for _ in range(K)]
for r in 1..K:
    A_rot = copy(A)                   # 当前 rotation 的通道可用性
    for (i,j) in P_all:
        if |P_r| = N: break
        if A_rot[i,j] == 0: continue  # 通道已被占用
        P_r.append((i,j))
        A_rot[i,:]=0; A_rot[:,i]=0    # 禁用通道 i
        A_rot[j,:]=0; A_rot[:,j]=0    # 禁用通道 j
        A[i,j]=0; A[j,i]=0            # 跨 rotation 不复用此 pair
```
每个 independent rotation 内所有 pair 并发执行，无需 barrier 同步。跨 rotation 不复用 pair 实现更丰富的通道对组合。推理时 K 个 rotations 在 fused kernel 内融合执行，一次激活加载，K 次旋转均在 shared memory 上完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Independent Rotation 的约束导致表达能力有限（O(n) vs 全旋转 O(n²)），补偿策略：(1) 串联 K=8 个使用不同 pairings 的 rotations，扩展有效参数空间；(2) 与 channel-wise scaling 联合使用——scaling 负责全局幅值均衡，rotation 负责局部 token 级对齐；(3) 跨 rotation 不复用 pair 使组合空间最大化。推理开销：每 token 每 group 计算量为 O(K·n) 次 FMA（K=8, n=128 时为 1024 FMA），相比 FP16 GEMM 的 O(n·D)（D 为 hidden_dim >> n）可忽略。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---
