## Representation Collapse in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Representation Collapse（表示坍塌）是 MoE 训练中的一种退化现象：多数输入 token 被路由到少数几个 expert，导致这些 expert 负载过高而其他 expert 被闲置。这不仅是负载均衡问题，更导致 expert 的表征多样性丧失——闲置 expert 的梯度更新极少，逐渐失去有效的知识表示。Chi et al. (2022) 系统分析了这一现象：随着训练进行，gate 对部分 expert 的路由概率趋近于 0，这些 expert 接收的 token 越来越少，形成正反馈循环（越少被激活 → 梯度越少 → 能力越差 → 越不被选择）。在 homogeneous MoE 中，representation collapse 表现为少数"赢家 expert"占据大部分 token，其余 expert 退化；在 heterogeneous MoE 中，collapse 更严重——大 expert 天然能力更强，router 倾向于"赢家通吃"，小 expert 完全被边缘化（HMoE Section 3.2, Figure 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Representation collapse 的正反馈循环机制：

```
Training step t:
  1. Router: P = softmax(x @ W_r)
     # 此时 expert_a 和 expert_b 的概率已极化
  2. Token assignment: 80% tokens → expert_a, 15% → expert_b, 5% → rest
  3. Expert_a 接收大量 token → 梯度估计准确 → 能力提升
  4. Expert_c 几乎无 token → 梯度噪声大/无梯度 → 能力停滞
  5. Step t+1: Router 观察到 expert_a 能力更强 → 分配更多 token
     → 正反馈循环加速 → collapse 加剧
```

HMoE 通过两种机制缓解 representation collapse：
- **异构容量**：不同大小的 expert 天然具有不同的能力范围。小 expert 处理简单 token 时表现不逊于大 expert（因简单 token 不需要深层表示），因此 router 可以被 P-Penalty 引导将简单 token 分配给经济的小 expert，打破"赢家通吃"逻辑。
- **P-Penalty Loss**：显式惩罚激活大 expert（M_i 包含 h_ffn,i），使得"激活大 expert"比"激活小 expert"有更高的 loss 代价，打破 collapse 的正反馈。

实验验证（Figure 13）：homogeneous MoE 中 expert 的 Wasserstein distance 显示 expert 形成 2 个聚类（a/b/c 高度相似），而 HMoE 中 expert 按大小形成差异化分组（相似大小 expert 聚为一类），表明异构设计有效促进了 expert 分化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Chi et al. (2022) 提出 X-MoE routing 缓解 collapse：对 hidden state 做低秩投影后与 expert embedding 做余弦相似度，避免 softmax 的 winner-take-all 效应。HMoE 从另一个角度——通过异构容量 + P-Penalty 创建"积极差异化"而非"消极缓解"——让 collapse 的正反馈在异构建模空间中自然收敛到各 expert 按 token 复杂度分化的均衡状态。HMoE 实验显示 representation collapse 被有效缓解：训练后期各 expert 的激活频率保持稳定的差异化分布（而非 collapse 到 1-2 个 expert）。

RMoE (Qiu et al., 2025) 从 router 设计的角度分析了 representation collapse 的另一个侧面：(1) 单线性层 router 的局限性——token hidden states 通过 softmax 计算 gating score 时，embedding 容易 collapse 到 expert embedding 附近，导致 softmax 输出近乎 one-hot（gate entropy 极低），Top-k 退化为 Top-1；(2) RMoE 通过在 router 中引入 GRU + 逐层投影 Proj_i 间接缓解了 collapse——Proj_i 将 hidden state 与 expert embedding 分离（类似 XMoE 的低维投影策略），GRU 提供跨层路由信息使 router gate score 分布适度平坦（高熵但非随机）。实验验证：RMoE 的 gate entropy 分布比 SMoE 更均匀但不像 RandomMoE 那样完全扁平，在 exploration vs exploitation 之间取得更好平衡。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Layerwise Recurrent Router for Mixture-of-Experts
