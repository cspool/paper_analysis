## Algebraic-Layout Mapping（代数-布局映射：AL Table / AL TLB / MV Translation）

术语解释
DySHARP 在目的 GPU Hub 内新增的硬件 memory manager，把包中携带的代数索引（algebraic index，跨 GPU 一致）翻译为本地虚拟地址（layout index，各 GPU 独立），支撑动态 multimem 寻址下的地址不对称性。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dispatch 是 AllGather 的动态变体：每个 token 的"代数索引"（结果代数张量中的位置）跨 GPU 相同，但只有子集 GPU 收到该 token，各 GPU 将碎片化的代数张量压缩成稠密 layout tensor，layout index 天然不对称。硬件 memory manager 维护两者映射：AL Table（DRAM，每项 4B = 1b Valid + 31b LIdx，共 4×nToken 字节；多专家同卡时分子表）记录代数块→布局块；Dispatch 时对未见过的 AIdx 按计数器累加式分配下一个布局块并登记映射；Combine 复用同一映射（专家计算不改变 token 顺序）。MV translation 公式：AIdx = (MAddr − MBase)/bsize；VAddr = VBase + LIdx×bsize + MAddr%bsize。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
访问路径：请求包到达 Hub → 查 AL TLB（512 项；tag = ExpertID 拼接 AIdx，CAM 实现，buffer 为 SRAM）→ 命中直接得 LIdx → MV translation 得虚拟地址 → 走既有 Link MMU 做虚拟-物理翻译 → 访存。Miss 时访问 DRAM 中的 AL Table 并把条目填入 AL TLB；Dispatch 首次触达某代数块时分配布局块。因为软件对同一 token 向量内元素连续访问，AL TLB 局部性强（512 项为 hit-rate 与开销的 sweet spot）。MV translation 与 AL 管理解耦：Dispatch 与 Combine 共享同一映射但操作不同虚拟地址空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与动态 multimem 寻址的 CUDA Runtime 接口配套：cuDyMulticastBindAddr 指定 ntoken（multimem 空间大小）与 nactive[expert]（各专家接收 token 数，由 gate 路由在 Dispatch 前给出）。开销：1M token 时 AL Table 仅 4MB/层，相对 GPU DRAM（40s-100s GB）可忽略。属于非侵入式扩展：置于虚拟-物理翻译之前，不改变既有功能、不占用既有数据通路。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
