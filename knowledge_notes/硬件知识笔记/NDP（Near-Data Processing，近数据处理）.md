## NDP（Near-Data Processing，近数据处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NDP 是把专用计算逻辑放置在数据存储附近的架构范式：与其把海量数据搬到处理器（CPU/GPU），不如把计算下沉到内存侧，从而绕开"内存墙"（memory wall），有效带宽可提升最多两个数量级。NDP 形态包括 DRAM/PIM（计算在 bank 内，如 UPMEM、SK hynix GDDR6-AiM）、近存计算（计算在内存模组逻辑层，如 HBM-PIM）、DIMM-based NDP（计算逻辑放在 DIMM 的 RCD/DB 等缓冲芯片旁）以及 CXL-attached NDP（计算在 CXL 内存扩展侧，如 MoNDE）。在 ANNS/向量检索中，距离计算算术强度极低、纯内存受限，NDP 用高内部带宽直接放大吞吐（vault 笔记：/data3/paper_analysis/knowledge_notes/硬件知识笔记/Near-Data Processing (NDP) for MoE.md 给出 MoE 场景的 NDP 范式；本论文用 DIMM-based NDP）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
NASZIP 的 DIMM-based NDP 运转流程：① host CPU 把距离计算命令与向量位置下发给 NMA（near-memory accelerator，集成在 DB 芯片）；② 每个 sub-channel 内的 NMA 独立取本地向量并计算距离（利用 DRAM 内部带宽）；③ 结果经共享优先队列合并后回传 CPU，CPU 查邻居表决定下一 hop。三步流程中 CPU 侧邻居查找、跨 sub-channel 通信、距离计算分别成为瓶颈（论文 Fig.4a 分解：CPU 查找占 31.7%、跨通道访问显著、距离计算主导）。NDP 的价值在于把 ② 的带宽做大并把 ①③ 的串行开销消掉（DaM 卸载邻居查找 + 双 VPE 并行 + LNC 缓存）。通用例（MoNDE，vault 笔记）：GPU 发激活值而非权重，NDP 核在 LPDDR 旁算 cold-expert 的 FFN，激活传输 << 权重传输，与 GPU 并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：把轻量计算单元（如 NASZIP 的 VPE/LNC/共享优先队列/控制器，面积 0.7091 mm²，远小于 RCD/DB 的 10.22 mm²）封装进 DIMM 缓冲芯片；标准 DRAM die 不改动，保持 host 兼容、复用处理器 DDR 控制器。评估用周期精确 NDP 模拟器（本论文用 UniNDP，修改支持 rank-level parallelism）。使用场景：memory-bound 的检索/推理负载（ANNS、MoE 冷专家、GEMV 解码）；业界 MCRDIMM/MRDIMM（Renesas RG5R188 MRCD、Montage M88MR5RCD01，8800 MT/s）正把类似近存逻辑商品化（Web 证据）。开源：NASZIP 全套模拟代码 https://github.com/Intelligent-Computing-Research-Group/NasZip。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
