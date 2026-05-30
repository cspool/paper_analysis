## Hardware-Managed HBM with ATU and PDU (硬件管理的HBM缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hardware-Managed HBM 是论文为 wafer-scale GPU D2D controller 提出的扩展机制，通过在 D2D controller 中集成 **Address Translation Unit (ATU)** 和 **Prediction Unit (PDU)**，实现对远程 HBM 中热门 expert 权重的自动本地缓存。在整个过程中，软件/CUDA 程序完全无感知——SM 发出的 memory request 的物理地址不变，硬件层自动判断数据是本地还是远程并路由。这与传统 GPU 将所有 HBM die 视为 uniform memory space（不区分 local vs remote）形成对比。核心思想：利用 MoE expert selection 的 temporal locality（Insight 2），将远程热门 expert 缓存在本地 HBM，避免重复的 D2D remote reads。

从硬件架构角度拆解术语：
ATU 和 PDU 的工作流程：

```
D2D Controller 数据流（含 ATU + PDU）:
=========================================

Case 1: Remote Read (数据未被本地缓存)
──────────────────────────────────────
SM issues read request for address A_remote (expert e, remote die)
  │
  ▼
D2D Controller:
  ├── check PDU.is_local[e] → 0 (not cached)
  ├── 正常 D2D read: XY routing → remote die → read HBM → return
  │
  ▼
Data returns to D2D Controller:
  ├── Send data to SM (regardless of caching decision)
  ├── PDU: check PDU.cp_en[e]
  │   ├── if cp_en[e] == 1:
  │   │   ├── 分配本地 HBM 空间 A_local
  │   │   ├── Write data to LLC (30ns write latency)
  │   │   ├── Write data to local HBM
  │   │   ├── ATU: add entry [A_remote → A_local]
  │   │   └── PDU: set is_local[e] = 1
  │   └── if cp_en[e] == 0:
  │       └── No action (仅返回数据给 SM)

Case 2: Local Read (数据已被本地缓存)
──────────────────────────────────────
SM issues read request for address A_remote (expert e)
  │
  ▼
D2D Controller:
  ├── PDU: check is_local[e] → 1 (cached locally)
  ├── ATU: translate A_remote → A_local
  ├── Redirect to local LLC (not D2D network!)
  │   ├── LLC hit (100ns): return data to SM
  │   └── LLC miss (110ns penalty): fetch from local HBM (300ns)
  └── 结果: 避免了多跳 D2D latency

Hardware Specifications (per die, 5nm):
- ATU: 4.25 KB SRAM, 68-bit entries (64-bit addr + 4-bit control)
  └── 容量: ~512 entries, 支持 512 remote-to-local mappings
- PDU Prediction Table: 128 B register file, 16-bit entries
  └── 每 entry: cp_en (1 bit) + is_local (1 bit) + 保留 (14 bits)
  └── 容量: 64 entries, 每个 die 最多 track 64 experts
- Total area/die: ~0.0068 mm²; power: ~390 mW
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PDU 的 cp_en bits 由 Global CP 在 kernel launch 时通过 Data-Driven Predictor 计算并配置到各 die 的 Local CP → Local CP 写入 PDU registers。
- ATU 的 address mapping 在第一次 remote read + caching 时动态建立（由 D2D controller 硬件自动执行，无需 Global CP 干预）。
- 与 CPU cache 的区别：这不是传统意义的 cache（不基于 LRU/replacement policy），而是 predictor-driven 的 selective duplication——只有被 predictor 认为"将在下一 token 被使用"的 remote expert 才会被复制。
- 效果：Allo+Pred 相比 Allo Only 进一步将 remote DRAM reads 转换为 local reads（Figure 14 DRAM access breakdown），hop count 从 142× 再降低到 >213×，但 throughput 仅额外提升 1.1×（因为 hop count 已经不是性能瓶颈——allocation 已将大多数请求分配到本地 die）。
- 可推广到其他 memory hierarchy：CXL-based systems (local DRAM + remote CXL memory), SSD offloading systems (DRAM + flash), PIM systems (local + remote DRAM dies)。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting
