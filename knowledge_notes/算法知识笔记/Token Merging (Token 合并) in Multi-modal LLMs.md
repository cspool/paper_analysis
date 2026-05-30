## Token Merging (Token 合并) in Multi-modal LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Merging 是一种通过合并相似 token 来减少多模态 LLM 输入 token 数量的训练无关技术。在 AIM 论文中，Token Merging 发生在 Visual Encoder 之后、LLM 输入之前：将 N⁰ 个视觉 token embedding 按余弦相似度配对，每对最相似的 token 取均值合并，迭代多轮直至达到目标保留率。与 ToMe（Token Merging for ViT）在 Vision Encoder 每层内做合并不同，AIM 的 Token Merging 在 Encoder 输出后一次性执行，对 Encoder 架构无侵入，即插即用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Token Merging 伪代码（AIM 风格）**：

```
// 输入：visual tokens v ∈ R^{N×D}，目标保留率 r_merge
// 输出：合并后的 tokens v' ∈ R^{(N×r_merge)×D}

def token_merging(v, r_merge):
    current = v
    N = len(current)
    I = ceil(log2(1/r_merge))  // 所需迭代轮数
    
    for iter in 1..I:
        // 将 tokens 分成 A（偶数位置）和 B（奇数位置）
        A = current[0::2]  // 偶数索引
        B = current[1::2]  // 奇数索引
        
        // 计算 A 和 B 之间的余弦相似度矩阵
        sim = cosine_similarity(A, B)  // [N/2, N/2]
        
        // 对 A 中每个 token，找到 B 中最相似的 token
        for i in 0..len(A):
            j_star = argmax(sim[i])      // B 中最匹配的索引
            merged = (A[i] + B[j_star]) / 2.0  // 取平均合并
            current.append(merged)
        
        N = len(current)
    
    return current  // 共 N⁰ × (1/2)^I = N⁰ × r_merge 个 token
```

**视频场景的特殊处理**：合并仅在单帧内（spatial）进行，不跨帧（temporal）合并。消融实验表明跨帧合并在低保留率下显著损害性能（如 r=3.1% 时 temporal merging 47.4 vs spatial 52.3 on VideoMME），因为跨帧合并破坏 token 的时序顺序。

术语一般如何实现？如何使用？

在 AIM 实现中，Token Merging 以函数形式插入到 LLaVA 推理流程的 Visual Encoder 输出与 LLM 输入之间。默认配置：video LLM 保留 25%（迭代 2 轮），image LLM 保留 12.5%（迭代 3 轮）。额外计算开销极小：video 场景 88.25 GFLOPs，仅占 Qwen2-7B 推理的 0.6%。代码开源：https://github.com/LaVi-Lab/AIM。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

---
