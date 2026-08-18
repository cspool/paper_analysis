## DTile / NTile 与 Tile Group（内存 plane 与计算 tile 配对 + 专用链路/AXI）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DTile/NTile：计算 die 上按下方内存类型分类的 tile——DRAM 下方的 tile 为 DTile、NVM 下方的为 NTile（MAC datapath 相同，仅内存关联不同）。tile group：把同相对位置的 DTile 与 NTile 配对成组（组数 = min(#DTile, #NTile)，#DTile<#NTile 时每组 1 个 DTile + 若干 NTile），组内用专用高速链路、跨组用 AXI fabric 全片连接。数据供给：NTile 为 DTile 提供 NVM 中的 Weight/KVCache，DTile 为 NTile 提供 DRAM 中的 IA 等。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（FFN GEMM 例子）：tile group 内 NVM plane 存其 NTile 的 Weight 与同组 DTile 的均匀 Weight 切块、DRAM plane 存 IA 行 → NTile 从 NVM plane 读 Weight 块（70% 读带宽利用）、DTile 从 DRAM plane 读/写 IA → 组内专用链路交换跨内存数据 → 各 tile 本地 MAC 计算；当需跨组分布时走 AXI fabric。该布局避免 plane starvation 与 compute-tile-memory-plane 失配，是带宽利用中心数据流的硬件基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 GPGPU-Sim 中以 CUDA 实现 plane-aware tile 映射 + 双缓冲；专用链路/AXI 的带宽在解析模型中归入 ICNT 与 tile 间传输。SHyLA 本体未开源。GQA 场景 decode 按 attention-group 级并行到 tile 内，减少 KVCache 共享的跨 tile 传输。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
