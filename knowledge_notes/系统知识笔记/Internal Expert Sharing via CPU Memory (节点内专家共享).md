## Internal Expert Sharing via CPU Memory (节点内专家共享)

术语是什么？
Internal Expert Sharing 是 PopFetcher 提出的一种 GPU 集群节点内的 expert 参数共享机制。当某个 GPU worker 从 remote 节点预取了 expert 后，该 expert 参数被缓存到节点服务器的 CPU memory 中，同节点内其他 GPU worker 可直接从 CPU memory 读取，绕过重复的跨节点网络传输。利用 CPU memory 作为节点内 expert 共享的中间层缓存。

从系统架构角度拆解术语：
节点内拓扑感知 expert 检索：
```
// 节点内：GPU0--NVLink(1800GB/s)--GPU1--NVLink--GPU2--NVLink--GPU3
// GPU↔CPU：PCIe 5.0 (64GB/s)
// 跨节点：GDR NIC (400Gb/s)

// Worker w 需要 remote expert E_n^i 时：
if E_n^i already in local CPU memory (cached by another local GPU):
    pull from CPU memory via PCIe     // 跳过跨节点 GDR NIC
else:
    pull from remote via GDR NIC      // 跨节点拉取
    cache to local CPU memory          // 缓存供同节点其他 GPU 使用
```

Cache manager 维护 per-server 的 expert 参数缓存，记录哪些 remote expert 已通过本节点任意 GPU 预取到 CPU memory。利用 CPU memory 的大容量（数百 GB-TB 级）弥补 GPU memory 有限（24-80GB）的短板。

术语一般如何实现？如何使用？
实现为 PopFetcher 中的 server-level cache manager（Python 管理 CPU memory 中 torch tensor 的 expert 参数缓存）。各 GPU worker 通过 PCIe 访问。适用于多 GPU 节点场景（如 8×GPU 节点），当多个 GPU 的 token routing 倾向于相同的热门 expert 时，仅需一次跨节点传输即可服务所有本地 GPU。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
