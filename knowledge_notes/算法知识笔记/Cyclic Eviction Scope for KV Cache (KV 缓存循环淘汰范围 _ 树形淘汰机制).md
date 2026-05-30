## Cyclic Eviction Scope for KV Cache (KV 缓存循环淘汰范围 / 树形淘汰机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cyclic Eviction Scope 是 TreeKV 的核心淘汰机制。在 decoding 每 step，当 KV cache 容量满时，淘汰决策局限在相邻两个 token {idx, idx+1} 构成的"淘汰范围（eviction scope）"内——比较两者的平均 attention weight，淘汰较低分者。idx 每 step 循环递增 (idx mod c + 1)，使得淘汰范围从 cache 头部平滑移到尾部再回到头部。

与 H2O/TOVA 的全局贪心排序（每次在所有 token 中选最差者淘汰）不同，cyclic eviction scope 有三个关键特性：(1) **O(1) 每步开销**——仅比较两个标量值，无排序；(2) **均匀空间覆盖**——每轮完整循环（c 步）中每个位置恰好参与两次比较（一次作为 idx 左端点，一次作为 idx+1 右端点），避免了 H2O 的区域偏差；(3) **树形竞争**——连续多轮形成二叉树竞争，相邻 token 逐级比较，胜者保留到下一轮，最终形成 coarse-to-fine 的信息层次（左侧远距离 token 淘汰率高，保留密度低；右侧近距离 token 保留密度高）。

从算法pipeline角度拆解术语。

**Cyclic Eviction Scope 树形竞争示意**:

```
初始 cache (c=8): [T1, T2, T3, T4, T5, T6, T7, T8]

第 1 轮循环 (idx=1→2→...→8):
  idx=1: scope={1,2}, 比较 T1 vs T2 → 淘汰低分者
  idx=2: scope={3,4}, 比较 T3 vs T4 → 淘汰低分者
  ... 每步 idx 循环递增，淘汰后 cache 重新索引

树形结构示意（连续多轮后）:
          [T1]                    ← 最左侧，每轮都面临淘汰，存活概率最低
         /    \
      [T1]   [T3]                ← 中间距离，已被淘汰多轮
             /    \
          [T3]   [T5,T6]         ← 近端，保留密度最高（"右密"）
```

**伪代码**:
```
idx = 1
for each step when cache full:
    S_avg = S / C                # 每个 token 的平均 attention weight
    if S_avg[idx] > S_avg[idx+1]:
        evict (idx+1)-th KV pair
    else:
        evict idx-th KV pair
    idx = (idx + 1) % c + 1      # 循环递增
```

**Annotations**:
- `S_avg[idx]` 和 `S_avg[idx+1]` 是 cache 中第 idx 和 idx+1 个 token 的平均 attention weight（非原始序列位置）
- idx 的循环范围是 1..c（cache 容量），保证每步淘汰后 cache 大小回到 c
- 新 token 追加到 cache 尾部，其初始 S=0, C=0，在首次参与淘汰时与左侧老 token 比较——若老 token 重要性低则被淘汰（新 token 得以保留），若新 token 尚无足够重要性证据则被淘汰
- Ablation 证实（Figure 5）：即使完全不用 attention weight（每次固定淘汰左侧 token），仅靠循环淘汰范围的树形空间分布，perplexity 已远超 H2O 和 TOVA

术语一般如何实现？如何使用？

HuggingFace Transformers 实现，per-layer 或 per-head 维护 idx (int) 和 S_avg (float array, size c)。每 decode step 开销为 1 次标量比较 + 1 次模运算 = 可忽略。与 H2O top-k 排序 O(c log c) 相比，在 batch serving 场景下优势显著。TreeKV_Select_Left_Token 变体（固定淘汰左侧，零 attention 开销）在 PG19 65k 书上 perplexity 与完整 TreeKV 差距极小，证明树结构本身是性能核心驱动力。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures
