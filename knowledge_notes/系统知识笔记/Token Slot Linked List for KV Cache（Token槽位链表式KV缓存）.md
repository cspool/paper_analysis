## Token Slot Linked List for KV Cache（Token槽位链表式KV缓存）

术语是什么？通过联网搜索让回答具体和精准。
Token Slot Linked List 是 LightLLM 采用的 KV cache 底层数据结构，也是 PiLLM 实现批级 KV cache 共享池的基础。与 vLLM PagedAttention 的固定大小 block table（通常每 block 16 tokens）不同，Token Slot Linked List 以单个 token 为粒度分配 KV cache slot，每个 slot 通过指针链接到同一请求的下一个 token slot 构成 per-request 链表。请求需新 token 空间时从全局 free list 取一个 slot 接入链表尾部，释放时归还 free list。细粒度设计天然支持动态、不规则分配和池化共享——请求生成少于预测值时未用 slots 留在池中，生成超过预测值时从池中取更多 slots。

从系统架构角度拆解术语：
1. **初始化**：预分配 token slot 数组作为全局池。
2. **Prefill**：为每个 input token 分配一个 slot，链表首指针记录在请求元数据中。
3. **Decode 生成**：每生成新 token，从全局 free list 取 slot → 接入链表尾部 → 存储该 token 的 K/V。
4. **请求完成/Evict**：遍历链表所有 slot → 归还全局 free list。
5. **共享池预算**：PiLLM 的 batch memory pool 维护 batch 内已分配 slot 总数，接近预测上界时限制新分配或触发 eviction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LightLLM 中通过预分配连续内存数组 + free list 管理实现。相比 PagedAttention 固定 block 的优势：无内部碎片（block 内未使用的 token slot 不浪费），细粒度分配更适合 batch 级共享。代价是指针管理开销增加——每个 slot 需存 next pointer。适合与 batch-aware scheduling 配合。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
