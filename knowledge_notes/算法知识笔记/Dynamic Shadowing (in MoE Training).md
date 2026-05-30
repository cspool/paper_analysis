## Dynamic Shadowing (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Shadowing 是 FasterMoE 提出的运行时负载均衡策略，用于解决 MoE 分布式训练中因 skewed expert selection 导致的动态负载不均衡问题。核心思想是：将热门 expert 的模型参数广播复制到所有 worker（"影子化"），使得原本需要远程发送的大量 input tokens 被替换为少量的模型参数传输，热门 expert 的计算在各 worker 本地执行。影子化决策基于性能模型在每 iteration 动态判断——当 token 传输开销大于模型传输开销，或减少的计算延迟大于增加的通信开销时启用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Algorithm 1: SelectShadowExperts (每 iteration 每 worker 执行)
# 输入: B[N] - 每 worker 的 batch size (token 数)
# 输出: E_s - 需影子化的 expert 集合

def SelectShadowExperts(B):
    B_max = max(B)
    c_min = Lat_imbl(B_max)        # 当前不均衡配置的延迟
  
    # Lat_imbl(B) = max_w{3·4B_wαH²/P + 4·B_wH/W_net}   (Eq. 7)
    #   3×GeMM (1 forward + 2 backward) + 4×all-to-all
  
    E_s = []
    for i, B_i in sorted(B, key=lambda x: -x[1]):  # 降序遍历
        B_i = T[i][i]           # 保留本地 tokens
        for j != i:
            B_i += T[i][i]      # 影子化后在其他 worker 本地执行
  
        B_max_prime = max(B)    # 影子化后的最大 batch
        c = Lat_shadow(len(E_s)+1, B_max_prime)
  
        # Lat_shadow(r, B') = max_w{3·4B'_wαH²/P} + 2r·2αH²/W_net   (Eq. 8)
        #   第一项: 均衡后的 computation; 第二项: 广播 r 个 expert 参数的开销
  
        if c < c_min:           # 影子化改善延迟则采纳
            c_min = c
            E_s.append(i)
        else:
            return E_s          # 一旦不改善即停止

# 影子化启用条件 (简化):
# 条件1: B_max > rαH     → token 传输开销 > 模型传输开销
# 条件2: 3(B_max-B'_max)αH/(rαH-B_max) > P/W_net → 减少的计算 > 增加的通信
```

执行流程：(1) Forward: broadcast expert 参数到所有 worker → 各 worker 本地计算影子化 expert 的 GeMM → 非影子化 expert 仍通过 all-to-all 远程计算；(2) Backward: 各 worker 本地计算影子化 expert 的梯度 → reduce 梯度到原 worker → 原 worker 更新参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 实现中，动态影子化基于 FastMoE 扩展，决策逻辑位于 `fastermoe/fmoe/transformer.py:34`。矩阵 T（token-to-expert 分配）在所有 worker 间共享，无需额外通信。实验显示平均 19% 的 experts 被影子化，单 expert 影子化最大加速 1.97×。在 *johnny*（16 GPU）上单独启用影子化加速 1.95×，在 *trevor*（64 GPU）上加速 4.74×——更大规模下负载不均衡问题更严重，影子化收益更大。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
