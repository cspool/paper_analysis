## Bank Chaining（拓扑保持冗余：3D-DRAM 的 bank 链式冗余修复）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bank Chaining 是 Raptor 提出的拓扑保持冗余（topology-preserving redundancy）方案：在高度 bank 化的 3D-DRAM die 上，把 N 个功能 bank 与 M 个冗余 bank 逻辑上链成一条，逻辑 die 上用轻量层级多路复用（mux）从 N+M 个物理 bank 中选出任意连续 N-bank 窗口来形成 N/3 个 3-bank channel，从而容忍任意位置最多 M 个故障 bank，同时保持 channel 宽度、映射与布线对称。背景：高 bank 数 die 上几个故障 bank 就丢弃整片不经济，朴素做法是禁用坏 bank 导致窄/不对称 channel 卡死卡级带宽；把冗余 bank 放"beachfront"再绕线则增加金属用量/延迟/功耗。Bank chaining 把冗余 bank 与常规 bank 内联分布，选择后 channel 逻辑连续、布线局部（逻辑 die 上靠近对应 TE/WB 处）、仅需 M 级 mux（N=24、M=2 时只要 2 级）。840 bank 中预留 72 个作为冗余，768 个可用组成 256 channel × 3 bank。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 3D-DRAM die 制造/运行流程中：每 bank 有备用行列，若仍修不掉缺陷则该 bank 标记故障 → bank chaining 在封装前（wafer 级）与封装后（stack 级）修复流程中重排：控制器扫描 N+M 链，跳过故障 bank，从连续健康窗口组成 3-bank channel（例：24+2=26 个 bank 的链中任一 24 个健康 bank → 8 个 channel×3 bank，容忍 2 个任意位置故障）。选择只改逻辑 die 上的 mux 配置，通道宽度、映射、路由全部对称，不破坏给计算侧的逻辑视图。论文实验（图 11）扫描 group size 与冗余 bank 数，显示冗余 bank 数增加显著提升 channel 恢复率（具体良率数字未披露）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：逻辑 die 上的轻量层级 mux + 每 bank 可配置/链式组织，修复在封装前后流程中完成（与行业 DRAM 良率修复一致）；bank 几何（1364 行/bank）同时支持 thermal-aware refresh 与交错 ECC 的低开销共置。使用方式：作为高 bank 数 3D-DRAM 的良率/可靠性手段，保证 100TB/s 全宽带宽不因少数故障 bank 而打折；与 stream blocking 的 3-bank/channel 映射直接耦合（冗余 72 bank 正是 1024 需求与 840 实际的差额）。价值：在 422W/105°C 量产约束下维持"全宽、对称、局部布线"的 channel 拓扑。

涉及论文标题：
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
