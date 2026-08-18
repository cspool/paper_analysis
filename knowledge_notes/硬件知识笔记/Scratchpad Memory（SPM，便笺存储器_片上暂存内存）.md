## Scratchpad Memory（SPM，便笺存储器/片上暂存内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SPM 是片上、由软件显式管理的 SRAM，相比硬件管理的 cache 更节能、访问更快、面积更小 [14,27-30,89]，但要求软件负责代码/数据的放置（无自动缓存逻辑）。EHS 用的低功耗 MCU（如 MSP430 系列）通常无 cache（断电难以保持 cache 一致性），普遍以 SPM 为片上存储、以 NVM 为主存。SPM 是易失的：断电即丢失全部内容。
- 本论文 MANATEE 的关键洞察：把 SPM 当作"主存"、把片外 NVM 当作"二级存储"，从而天然获得安全主存——SPM 在信任边界内且断电即消失，只有持久化到 NVM 时才需要加密。这是把数据保密性从"每次访问都加解密"降为"仅页交换/断电时加解密"的根源。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（MSP430FR5994，默认 SPM 512B）：SPM 结构化为若干 page frame（如 5 个 frame），每个 frame 存放一个 64B NVM 页的解密副本；heap/stack 预载入 SPM 作为专用页，data 页按需载入预映射 frame。load/store 指令经编译器 hint（页号+颜色）直接索引 Buffer Table 判定页是否驻留；命中则按页内偏移在 SPM 内访问（零加密开销），未命中则加密驱逐旧帧、解密载入新页。SPM 大小对性能影响显著：512B 在漏电与页 miss 间最平衡（更大 SPM 增漏电、更小增 miss）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：片上 SRAM + 软件管理（无 MMU/TLB 协助）。MANATEE 用"编译期页着色决定页→frame 映射 + 运行时 page manager 管理交换"，纯软件实现、无硬件修改。对比方案 NVSRAM 依赖整 SPM checkpoint（断电把整个 SPM 加密写 NVM），在 100µF/1mF 小电容下无法运行。论文未给出公开代码，无法确认是否开源。
- LoRA/CGRA 中的 SPM（ISCA'26）：作为 CGRA 的片上数据缓冲——12 个 SPM bank（各 4KB，共 48KB）与 IOB 相连，IOB 经 TileLink + DMA 从 L2（128KB）加载/回写数据；loop kernel 数据先 LOAD 进 SPM 再被 CGRA 各 FU 访问，结果 STORE 回主存（周期分解中 LOAD/STORE 阶段）。后端工具用内存分区算法把数据分配到多 bank SPM 并把 bank 冲突访问调度到不同时间槽，防止内存争用；仿射访问地址由 IOB 控制器按配置生成，非仿射访问（如 A[B[i]]、A[i*i]）地址由其他 FU 运行时计算后经第二输入喂给 IOB。

- SMOOTH 中的 SPM（ISCA'26，移动 NPU LLM 推理）：把 SPM 从"tensor/tile 级连续地址分配"升级为 block 级虚拟化 + 硬件动态管理——固定大小块（默认 1KB）消除外部碎片；direct-mapped block table（p_blk/cont/use_cnt）+ bitmap 空闲表做低开销地址翻译，address_check 对连续区旁路翻译（dual-mode hybrid，连续访问保持传统 SPM 零开销直通）；硬件 use_cnt/end_cmd 驱动 early reclamation，idle 带宽按 N_preload=⌊U×BW/Block_size⌋ 预取。编译器只做静态 lifetime 分析标注 use_cnt，分配/回收交给硬件 Dynamic Memory Controller（DMC）。动机：LLM decode 期 GEMV（低 OI）与 softmax/GELU（高 OI）交替导致突发带宽，粗粒度连续 SPM 既碎片化又无法利用短时带宽窗口；编译器理想版 SPM 在 4K 序列 stall 仍多 32.7%。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
