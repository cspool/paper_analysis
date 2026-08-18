## TBNP（Tree-Based Neighborhood Prefetching，树状邻域预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA CUDA 驱动（open-gpu-kernel-modules 的 nvidia-uvm 模块）中默认的 CPU-GPU UVM 页面预取机制（ISCA'19 [21] 逆向揭示）：分配的 UVM 内存先分 2MB 大页（VABlock），再切成 64KB 逻辑块构成完全二叉树；每个非叶节点记录已迁移/总子节点数。far-fault 在叶子级迁移整个 64KB 块；当某子树 >50% 叶子已迁移，就预取该子树剩余叶子（图 2 例子：四次 far-fault 后 320KB 节点触发 B_5^3/B_6^3/B_7^3 预取）。预取大小从 64KB 自适应到 1MB（Web evidence: ISCA'19 "Interplay between hardware prefetcher and page eviction policy in CPU-GPU unified virtual memory"）。变体：TBNP-O（on-touch 迁移）、TBNP-F（first-touch 迁移）、TBNP-AT（自适应阈值，硬件计数器在 remote zero-copy 与迁移间调整）、TBNP-EA（Early Adaptor，按 page fault 波动动态调阈值）、Forest（按访问序列自适应块/树深度、1-bit 元数据合并/隔离树）。配套概念：Beyond VABlock 等研究指出 TBN 只在单个 2MB VABlock 内检测局部性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TBNP 逻辑实现在驱动软件（非 RTL），但行为被建模进 GMMU/TBNp 逻辑：GMMU 遍历完全二叉树找预取候选（节点有效性超 50% 容量即预取）。运转流程：GPU 访问远页 → far-fault → 驱动迁移 64KB 叶子块 → 更新各祖先节点有效大小 → 某祖先超 50% 触发预取剩余叶子 → 数据经 PCIe/NVLink 传输。多 GPU 下失效机理（LIBRA 论证）：(1) 工作负载跨 GPU 分区、每 GPU 只访问数据子集，空间局部性弱；(2) 多 GPU 并发访问同页造成争用与 ping-pong、冗余预取；(3) 以 accuracy 换 coverage——TBNP-EA accuracy 32%/coverage 33%、Forest 42%/42%，迁移+远程访问占执行时间 43%–45%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 NVIDIA open-gpu-kernel-modules 的 nvidia-uvm 驱动（https://github.com/NVIDIA/open-gpu-kernel-modules）；UVM 分配按 2MB VA block 建树，驱动批量处理 far-fault（升序 VA、2MB 粒度），64KB/4KB 投机预取粒度（Nsight Systems 可 profiling HtoD/DtoH 迁移事件）。研究界在 MGPUsim 等模拟器中复现并作为 CPU-GPU 页面预取 SOTA baseline。LIBRA 用它作为主要对比对象，指出其在多 GPU 下应被 stride-based 预取取代（accuracy 81.8% vs 42%）。论文未开源 LIBRA 实现（无法确认）。


LÆGIS 补充视角（ISCA'26，CC 下 TBNp 的预取阈值权衡）：LÆGIS 在 CC 场景重新评估 TBNp：预取阈值 Pt（迁移叶比例超过 Pt 即预取剩余叶）控制预取激进程度，Pt=51% 为默认、Pt=1% 为 aggressive。关键发现（Observation 1）：aggressive 预取（Pt=1%）减少 GPU-CPU fault batch 交互次数，但把负担转移到加密——每批服务更多 base 页、加密数据量增大，加密可占 batch 处理 CPU 时间的 70%+（CNN 实测）；GEMM 中 Pt 从 1%→91% 时加密占比从 44% 降到 16%，但 fault batch 数从 149 增到 1310（8.7×），batch 处理开销反超。3DCONV/NW/SSSP 在 aggressive 预取下反而比 Baseline 慢 2-5%（加密量增加抵消少故障收益），CNN 的预取收益从 pIdeal 的 1.97× 掉到 pBaseline 的 1.08×。LÆGIS 的预加密方案（IFN-LÆGIS）消除这一权衡：预加密使 pIFN-LÆGIS（aggressive 预取）达 2.74×、优于默认预取的 2.22×。

涉及论文标题：
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
