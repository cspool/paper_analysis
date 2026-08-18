## Top-K Gating（门控路由 / Gating Network）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K Gating 是 MoE 层的路由决策机制：gating network（通常 1-2 个前馈层 + softmax + top-k 选择）对每个输入 token 计算所有专家的路由分数，选出分数最高的 k 个专家参与计算，其余专家不激活。在 STEP 论文中，gating 打分公式为 s = W_gate·x + b_gate（Eq.4），选出 top-k 索引 l_1..l_k，路由权重经 softmax 归一化 w_i^r = e^{s_{l_i}} / Σ e^{s_{l_j}}（Eq.5），输出 y_routed = Σ w_i^r·E_{l_i}(x)（Eq.6）。关键洞察：不同层的 top-k 权重分布极不均——部分层的低排名专家（如 top-4 的第 3/4 名）平均路由权重 ≤0.05（Fig.3，Qwen1.5-MoE-A2.7B 与 DeepSeek-V2-Lite-Chat 在 MMLU 上），对输出贡献极小却照常进入加载与计算；这构成 STEP"空间感知分配"的直接动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def topk_gating(x, W_gate, b_gate, k, N):
    # x: [seq, hidden]；N = 该层 routed 专家总数
    s = x @ W_gate + b_gate          # (seq, N) 路由分数（Eq.4）
    topk_idx = argsort(s, desc)[:, :k]            # 每 token 选 top-k 索引
    topk_w   = softmax(gather(s, topk_idx), dim=-1)  # (seq, k) 权重（Eq.5）
    # STEP 的扩展：gating 仍对全部 N 个专家算分（含已当选的临时 shared），
    # 保证专家统计一致、支持下一窗口选举；但只有未当选的 k−c 个走动态选择
    return topk_idx, topk_w
```
Annotations：k=每 token 激活 routed 专家数（Mixtral top-2、Qwen1.5-MoE top-4、DeepSeek-V2-Lite top-6），N=routed 专家总数（8/60/64），s=路由分数向量。固定 top-k 的问题：简单 token 浪费计算（高权重集中在 1 个专家时其余 k−1 个几乎白算），困难 token 可能计算不足；STEP 用层内阈值 θ 与窗口投票把"固定 k"变成"层内动态 k_l + 窗口内 k−c"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HuggingFace Transformers 的 MoE layer forward 中 gate 为 nn.Linear(hidden, N)（可带 bias，如 Mixtral 的 noisy top-k gating），后接 softmax+topk；辅助负载均衡损失（switch loss/aux loss）防专家坍缩。STEP 的离线阶段用校准数据集前向收集每层 top-k score 分布，据此设定归一化权重阈值 θ（默认 Mixtral 0.25、Qwen 0.13、DeepSeek 0.07）确定层内 k_l；在线阶段每 decode step 记录 top-2k 专家投票（不只 top-k，扩大候选视野），窗口末按票数选举临时 shared。STEP 与一般 MoE 的区别：gating 计算不缩水（仍算全部专家分数），缩水的是"参与加载与计算的专家集合"。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
