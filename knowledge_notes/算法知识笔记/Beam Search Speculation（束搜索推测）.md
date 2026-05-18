## Beam Search Speculation（束搜索推测）

术语是什么？通过联网搜索让回答具体和精准。
Beam Search Speculation 是 AdaServe 提出的 speculative decoding 中 draft model 生成 candidate token tree 的方法。与传统 speculative decoding 每次只生成一个候选 token 序列不同，beam search speculation 对每个请求执行 d 步 beam search（每步维持宽度 w 的 beam），形成树状候选 token 结构。每步从当前层 w 个 token 出发各扩展多个 child token，记录由 draft model logits 近似的 path probability（路径上各节点概率的累积积）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: root token r, draft model M_d, depth d, beam width w
Output: candidate token tree T

T = {r}
current_layer = {r}
for step = 1 to d:
    candidates = []
    for each node n in current_layer:
        logits_n = M_d(prefix + path_to(n))
        top_k = argmax_k(logits_n, k)
        for each token t in top_k:
            prob = softmax(logits_n)[t]
            path_prob = prob * n.path_prob
            candidates.add(child(n, t, path_prob))
    current_layer = argmax_w(candidates, key=path_prob)
    T.add_all(current_layer)
```
每步扩展多分支形成 tree 而非 chain，保留 w 个最优路径，记录 path probability 供后续 SLO-customized selection 使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AdaServe 中由 GPU 上 colocate 的 draft model（如 Llama-3.2-1B-Instruct）执行。深度 d 和宽度 w 由 scheduler 根据活跃请求数动态调节。CUDA Graph 预捕获固定形状 decoding steps 减少 kernel launch overhead。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

