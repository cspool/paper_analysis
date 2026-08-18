## NVIDIA TLB Sub-entries（L2/L3 TLB 子项）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA 自 Volta/Turing 起 L2/L3 TLB 每项含 16 个 sub-entry：一个 entry 覆盖对齐的 32KB/1MB/32MB 区域，每个 sub-entry 映射区域内一个 4KB/64KB/2MB 页；L1 TLB 仍是一对一传统项。查找流程：虚拟地址拆为 VPN 与 offset，VPN 低位拆为 TLB index（选组）与 sub-entry index，高位为 VPB（virtual page base）；TLB index 选组 → VPB 比对标签（entry hit）→ sub-entry index 选槽，槽有效即命中；无 VPB 匹配则 miss 触发页走查。走查完成时若页落在已有 entry 覆盖区域内则装入对应 sub-entry，否则 LRU 换出整项（清空 16 个 sub-entry）后新建。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
sub-entry 结构用少量 tag 覆盖大区域，以低成本扩大 TLB 覆盖，且 4KB 小页与 2MB/64KB 大页可共存于同一 entry。CDFD 直接对齐该结构：32MB 粗粒度复制（一个 TLB entry 最多覆盖 32MB 范围），2MB/64KB 细粒度去重对应 sub-entry 页大小；CDB 也仿 sub-entry 组织（32MB 页 = 16×2MB 子项，1MB 页 = 16×64KB 子项）。论文配置：L2 TLB 128 项 8-way（16 sub-entries/项，10 cycle），L3 TLB 1024 项 8-way（16 sub-entries/项，40 cycle），均 LRU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Ampere L2 TLB 每项 16 sub-entry（GPC 共享）、L3 TLB 全 GPU 共享（MIG 不划分 L3）；STAR（MICRO'24）等研究利用 sub-entry 做多实例共享感知优化（Web evidence: https://dl.acm.org/doi/abs/10.1109/MICRO61859.2024.00031 ）。对软件透明，由 MMU 硬件维护。

ConServe 补充视角（ISCA'26）：sub-entry 利用率是 ConServe 相对 vAttention-Conv 的优势来源——vAttention-Conv 按 max-context 预留 slice，把并发会话的活跃 KV 区在 VA 上推得很远，每个会话占一个 32 MB 覆盖的 TLB entry 却只用少数 sub-entry（Llama-3-8B BF16 约 4 KB/token/layer，4K-token 会话每层约 8 个 2 MB 页 = 8/16 sub-entries，50% 利用率），批量越大低利用率 entry 越多、被逐出越早；ConServe 按需增长使 batch 内活跃 KV 页在 VA 上紧凑、sub-entry 利用率与翻译复用更高。ConServe 还量化了翻译局部性：连续布局使相邻页共享上层 PTE 索引（页走查缓存复用高），散页布局每次走查都 miss——表现为 FlashInfer-paged vs native 长 scoreboard stall 84.64% vs 79.37%、eligible warps/cycle 0.718 vs 0.825（A100，Llama-3-8B）。

LIBRA 补充视角（ISCA'26，层次化 GPU TLB 全貌）：论文建模完整三级 TLB 层次——L1 TLB 16 条目 16-way、1-cycle、TPC 共享（每 TPC 覆盖两 SM）、LRU；L2 TLB 128 条目 8-way、16 sub-entries/entry、10-cycle、GPC 共享；L3 TLB 1024 条目 8-way、16 sub-entries/entry、40-cycle、GPU 共享；每 GPU 独立 local page table 与 GMMU，CPU 侧 UVM driver 维护统一页表协调跨 GPU 故障。翻译流程：L1 cache 与 L1 TLB 并行访问（virtually indexed, physically tagged）→ L1 TLB miss 查 MSHR → L2 TLB → L3 TLB → GMMU。LIBRA 把 L3 TLB miss 作为访问模式学习输入（转发 MMP 的 MMAT），并把 TLB-miss 元数据扩展携带 source SM 信息以支持每 SM 独立的多 way 多 stride 学习。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
