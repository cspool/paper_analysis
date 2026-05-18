## Shareable KV Cache（可共享KV缓存）

术语是什么？通过联网搜索让回答具体和精准。
Shareable KV Cache是MFS论文针对multi-tier模型提出的一种KV缓存管理方案。在传统serving系统中，每个模型的KV cache绑定特定模型参数和层结构，小模型的KV cache无法被大模型复用（因为hidden size、head数、层结构不同）。MFS利用multi-tier模型的嵌套结构特性——低tier的层是高tier层的严格前缀，且所有tier共享相同的hidden size和head配置——使得低tier生成的KV cache对高tier完全兼容。KV cache manager将公共前缀层的KV cache标记为"multi-tier shareable"，当请求从低tier切换到高tier继续执行时，高tier直接复用低tier在公共前缀层产生的KV cache，无需重新prefill或重新计算。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Shareable KV Cache在MFS serving系统中的运转流程（以3-tier模型，R2从tier-2升级到tier-3为例）：

1. R2初始按tier-2执行：经过第0-32层（tier-1 + tier-2公共层），KV cache manager在第0-17层（tier-1层）和第18-31层（tier-2层）产生的KV cache条目上标记`shareable=true, tier_scope=[1,2,3]`。
2. R2在tier-2完成后采样并返回部分结果→但用户后续要求更高质量→front-end将R2重新提交为tier-3请求。
3. R2作为tier-3请求重新进入：KV cache manager查询R2的已有KV cache→发现第0-31层的KV cache标记为shareable且tier_scope包含3→直接加载已有KV cache（无需重新prefill第0-31层）。
4. R2从第32层开始decode（tier-3独占层）→仅需对新层（32-39）生成KV cache。
5. 最终R2的KV cache = 复用第0-31层KV + 新生成第32-39层KV。

对比Orca的独立部署方案（小模型7B和大模型13B独立部署）：R2若从7B模型获取部分结果后切换到13B模型，13B模型需从头prefill所有历史token→重复计算第0-32层的KV cache→产生额外的prefill latency和GPU compute开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MFS的KV cache manager在Orca的KV cache管理基础上增加了以下功能：(1) 每个KV cache条目增加tier_scope字段，记录哪些tier可以复用该条目；(2) 在请求执行时自动标记公共前缀层KV cache为shareable；(3) 在请求切换tier时查询已有KV cache的tier_scope并加载兼容条目；(4) 当shareable KV cache不再被任何tier使用时按标准LRU/引用计数策略释放。论文报告Shareable KV Cache相比Orca独立部署方案最多降低47.8% GPU memory footprint。该技术的核心前提：(a) multi-tier模型保持layer-only split（不切head），确保KV cache维度跨tier一致；(b) tier之间是严格嵌套关系（tier-1层 ⊂ tier-2层 ⊂ tier-3层），而非任意组合。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs
