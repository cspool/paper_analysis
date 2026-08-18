## Block-based SPM 虚拟化与低开销地址翻译（block table + bitmap + address_check 双模式直通）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 SMOOTH（ISCA'26，移动 NPU 片上内存管理）提出的片上 scratchpad（SPM）管理机制：把传统"tensor/tile 级连续地址分配"换成"固定大小 block 级虚拟化分配"。SPM 被切成与硬件处理单元对齐的固定大小块（默认 1 KB，可按模型维度设置），编译器可见的逻辑地址（virtual address）与物理 SRAM 块地址（physical block address）解耦——逻辑相邻的 tensor 可以放在物理上不连续的块里，从而消除 tile 尺寸不匹配和融合算子造成的碎片空洞。类似虚拟内存的分页，但虚拟地址空间不大于物理 SRAM（所有存活数据必须装得下 SRAM），因此可以用极小的 direct-mapped block table 翻译。核心数据结构：① block table——每项存 p_blk（物理块地址）、cont（连续分配的块数）、use_cnt（编译器静态 lifetime 分析标注的剩余使用次数）；② bitmap 空闲表——逐块标记分配状态，用于快速找空闲区与回收。配套 address_check 模块实现双模式混合（dual-mode hybrid）：当访问落在已翻译的连续区（如 p_blk=0x2400、cont=4）内时直接按物理地址访问、旁路 block table（保持传统 SPM 的零开销直通快路径）；只有跨块边界且下一块物理不连续时才重新置 lookup 标志做翻译。分配策略：case ① 有足够大连续空闲区则整段分配（记录 p_blk/cont/use_cnt）；case ② 碎片化导致没有足够大连续区时，用 find_zero 模块找最长空闲区，从起始地址顺序分配，不足再找下一个最长区（如先 0x09–0x0C 再 0x01–0x03），多段拼接。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（移动 NPU 8MB SRAM，decode 期一个 token）：编译器为 QKV 投影/FlashAttention/FFN 融合 kernel 的每个 tile 标注 use_cnt 后，DMC 在 SPM 中为 4MB 的虚拟地址 0x05 分配块——bitmap 显示物理块 0x02–0x08 连续空闲，则分配并记 p_blk=0x02、cont=7、use_cnt=2；若空闲区被碎片化，则 find_zero 依次找到 0x09–0x0C、0x01–0x03 两段拼接分配。执行时 buffer 访问数据 a：若 a 在已缓存的连续范围（p_blk, cont）内，直接按物理地址取（addr_check 直通）；跨块边界且不连续（如从 0x2400 区进入非连续的下一块）才重新查 block table。该机制让"只要有一小块空闲空间就能预取 V_cache 的单个 block"成为可能，是碎片化场景下维持高 SRAM 利用率的关键。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL，在 LLMCompass（基于 ScaleSim 的 cycle-accurate LLM 模拟器）中集成 end-to-end SRAM 管理器建模；block table + bitmap 由 DMC 维护，address_check 逻辑在 buffer 侧监控块边界地址位（如 1KB 块对应第 10 个地址位）。合成用 Yosys + ASAP7 7nm PDK：bt_lookup 时延 615.2ps、addr_check 时延 83.7ps、alloc 1508.2ps、find_zero 364.4ps、free 654.6ps，功耗 pW 级，SRAM 侧控制逻辑面积开销约 0.095%（相对估算 NPU 面积）。开源：https://github.com/skkim-caslab/SMOOTH（AE 脚本 src/verilog/run_all.sh 综合、src/ae/table1/get_area.py 出面积表）。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
