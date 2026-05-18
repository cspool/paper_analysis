## Model Family Serving（模型家族服务）

术语是什么？通过联网搜索让回答具体和精准。
Model Family Serving是指在一个serving系统中同时服务同一模型家族的多个不同规模模型（如Llama2-7B/13B/70B、Qwen-7B/14B等）的场景和技术。不同于传统的"single-model serving"（系统为每个模型规模独立部署和维护），Model Family Serving面临的核心挑战是：(1) 不同模型层数、head数、hidden size不同→无法将不同模型的请求放入同一高效batch；(2) KV cache绑定具体模型参数和层结构→不能在小模型和大模型之间直接共享；(3) GPU显存随着模型数增加而线性增长；(4) decode阶段GPU compute利用率低（每个模型各自小batch decode）。MFS通过将model family折叠为单一multi-tier嵌套模型来解决这些问题。该概念也关联到更广泛的"model-less inference"趋势——用户不再显式指定模型，而是指定latency/quality需求，由系统自动选择合适tier。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Model Family Serving的传统方案 vs MFS方案对比：

传统方案（Orca + 独立部署Llama2-7B + Llama2-13B）：
1. 系统维护两个独立模型实例（各自加载完整权重到GPU显存）。
2. 两个独立waiting queue：7B请求队列和13B请求队列。
3. 两个独立KV cache pool（不可跨模型共享）。
4. 两个独立batch执行流（各自selective batching）。
5. Speculative sampling：独立7B draft model生成候选token→传给独立13B target model验证→13B需从头prefill→无法复用7B KV cache。
6. 显存占用 = W(7B) + W(13B) + KV(7B) + KV(13B)。

MFS方案（Multi-tier嵌套模型）：
1. 系统维护单一multi-tier模型checkpoint（含3个tier，共享低层参数）。
2. 单一request pool + tier-level scheduler管理三个tier队列。
3. 单一KV cache pool（公共前缀层KV可跨tier共享）。
4. Group batching合并不同tier请求在公共层一起执行。
5. Tier-aware speculative sampling：tier-1 draft→tier-3 verify，直接复用tier-1 KV cache。
6. 显存占用 = W(shared_layers) + W(tier_specific_heads) + KV(pooled)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MFS的Model Family Serving通过Knowledge Precipitation + Multi-Tier Model + Group Batching + Shareable KV Cache四个组件协同实现。论文在Llama2 model family（7B/13B）和Qwen model family（7B/14B）上验证。此外，Model Family Serving的调度策略（如何为请求选择tier）在MFS中是plug-and-play的——系统不固定分配策略，支持外部policy（如用户显式指定模型、基于SLO的自动选择、基于请求复杂度的自适应选择等）。论文报告的性能指标均在假设tier选择策略已知的前提下测量。实际生产环境中的tier选择策略（如根据prompt复杂度动态判断所需模型规模）仍是开放问题。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

---
