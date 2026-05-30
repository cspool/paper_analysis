## Pull-based RDMA Image Token Transfer

术语是什么？

Pull-based RDMA Image Token Transfer 是 ModServe 中 Image Instance 向 Text Instance 传输 image tokens 的机制。与 push-based 方法（image tokens 生成后立即推送，需要 premature 的 target Text Instance 决策）不同，pull-based 方法延迟传输至所有 image tokens 就绪，由 Router 在拥有完整信息（token counts、各 Text Instance queue size、runtime load）后选择最优目标 Text Instance，再由该 Text Instance 通过 RDMA 从 Image Instances pull tokens。

从系统架构角度拆解术语：

Pull-based Transfer 流程（以 4-image request 为例）：
```
Step 1: 4 images → 分配到 2 个 Image Instances（各 2 images）
Step 2: Image Instance 1 & 2 并行 encoding:
  → 完成 → 注册 RDMA memory region
  → 发送 RDMA 地址和 token 元数据给 Image Pool Manager
Step 3: Image Pool Manager 聚合完成信号
  → 通知 Text Pool Manager: "request X ready, RDMA addrs: [addr1, addr2]"
Step 4: Text Pool Manager 选择 load 最小的 Text Instance
  → 下发 RDMA 地址列表给该 Text Instance
Step 5: Text Instance pull via NCCL + GPU Direct RDMA:
  → 从 Image Instance 1 GPU memory 读取 tokens_1
  → 从 Image Instance 2 GPU memory 读取 tokens_2
  → 合并 → Connector forward → LLM prefill
```

Overhead: InfiniBand RDMA P99 transfer latency = 5ms（<0.5% TTFT for CroAttn, <0.3% for DecOnly）。TCP over Ethernet: P50 100ms, P99 180ms——但 ModServe over TCP 仍实现 35% TTFT reduction at high load（vs InfiniBand 的 46%）。

术语一般如何实现？如何使用？

ModServe 使用 PyTorch distributed communication + NCCL backend + GPU Direct RDMA。Co-location 优化：当 Text Instance 未占用 server 全部 GPU 时，同一 server 内的 Image Instance 可避免网络传输——通过 NCCL intra-server communication。论文支持 InfiniBand 和 Ethernet 两种 media。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
