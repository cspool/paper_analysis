## Ring Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Ring Attention是一种context parallelism实现，将GPU组织为逻辑环拓扑，通过沿环轮转KV chunks实现跨设备完整attention。每个GPU持有部分Q chunk和部分KV chunk。每个step：本地Q×当前KV的局部attention→发送KV到下一个rank→从上一个rank接收新KV。重复CP-1次后，所有Q已attend到完整KV。通常与FlashAttention结合实现computation-communication overlap——在发送/接收KV时同步计算attention。

Ring Attention的局限性（UltraAttn指出的）：(1) Stripe-like partition高通信量——projection O(N) vs curled-up $O(\sqrt{N})$；(2) Inflexible kernel granularity——每step计算量极小导致低SM utilization；(3) 带宽浪费——所有KV轮转到所有device（zigzag ring约25%冗余，standard ring约50%）；(4) 跨节点scaling差——ring仅使用2 NIC单向带宽（75%浪费）。执行流程（CP=4）：
Step 1: GPU1(Q1×KV1), GPU2(Q2×KV2), GPU3(Q3×KV3), GPU4(Q4×KV4)
Step 2: GPU1(Q1×KV4), GPU2(Q2×KV1), GPU3(Q3×KV2), GPU4(Q4×KV3)
Step 3: GPU1(Q1×KV3), GPU2(Q2×KV4), GPU3(Q3×KV1), GPU4(Q4×KV2)
Step 4: GPU1(Q1×KV2), GPU2(Q2×KV3), GPU3(Q3×KV4), GPU4(Q4×KV1)

术语一般如何实现？如何使用？实现：ring-flash-attention (zhuzilin)、feifeibear/long-context-attention、NVIDIA Megatron-CP。适用场景：long-context LLM training (256K-1M+ tokens)、long-video generation。与FlashAttention+NCCL结合，替换PyTorch的scaled_dot_product_attention调用。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
