## Stream Blocking（流式分块：3D-DRAM 权重/KV cache 的 per-channel 128B flit 流式映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stream Blocking 是 d-Matrix Raptor（首个面向生成式推理的 3D-DRAM 加速器，ISCA'26）提出的 3D-DRAM 数据映射机制：把模型权重与 KV cache 按 per-channel 的近单位步长（near-unit-stride）128B flit 流式块（stream-blocked tile）布局，使每次列访问都被利用、最大化 row-buffer 局部性，从而在不复杂化数据通路的前提下消除 overfetch。背景：Raptor 的 3D-DRAM die 每 bank 是 1364×124 阵列（row buffer 跨全部 124 列，每列访问返回 256bit=32B），每 slice 有 16 个 DRAM channel 喂 16 个 weight buffer（WB），每 channel 交付 128B flit；单 bank 单列只有 32B，配 128B flit 需要 4 个 bank 同时访问，但 die 只有 840 bank（预留 72 冗余后 768 可用），每 channel 实际只有 3 bank → 3 个 bank 同取返回 96B，配 128B flit 需要两次访问造成系统性 overfetch。朴素列交错（column staggering）可消除 overfetch 但需要 192B 数据移位网络与 per-address 列模式跟踪，阻碍时序收敛。Stream blocking 的解法：软件栈把权重/KV tile 切成 per-channel 块，每 flit 拆为 96B aligned 部分（3 bank 同列索引各 32B）+ 32B partial 部分，相邻 flit 的 partial 打包进共享 partial region；控制器读时先访问 partial region（缓存 96B 的 32B 碎片），第二次访问取 96B aligned 区域并合并——两步固定模式、仅 96B×2 小缓冲、固定索引拼接，全利用每列访问。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Raptor 硬件架构中，stream blocking 是 slice（4×4 tensor engine + 16 WB + 16 channel）与 gang/chiplet 层级下"channel 化映射"的实现核心：每 slice 的 16 个独立 channel 各服务一个 TE 的 WB，channel 间并行，refresh/scrub 不跨 channel 停顿。具体流程（单层 KV-cache 映射，Llama-3.1-70B GQA 8 KV 头 head dim 128 FP16）：每 token 产生 2×8×128×2B=4KB KV 状态 → 4K 上下文 16MB 层缓存 → 切成 1024 个 16KB stream-blocked tile 均摊到 16 channel（64 tile/channel）→ 16KB tile = 128 个 128B flit，存为 96B aligned（3 bank 同列）+ 32B partial → 读 tile 约跨每 bank 两行，连续 flit 顺序走同开行的连续列（124 列×32B=3968B/行，开行后走完再激活下一行）。decode 时 TE 把 tile 作为顺序列走流式读取；新 KV 追加到 tile 尾部并带 stream-flipping metadata，每步只扩展一行 flit 对；16KB tile 粒度匹配 paged-attention 的 ≥4KB page，分配/逐出在 page 边界进行，不破坏 row-buffer 局部性。700MHz 下实现 2.5ns 平均 flit 延迟、105TB/s/card（12.5× HBM3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件栈（d-Matrix 的编译器/运行时侧）负责把权重与 KV cache 按 slice/16 channel 切成 16KB per-channel stream-blocked tile，并控制 tile 放置（与 stream flipping 的 metadata 布局协同）；硬件侧控制器按固定两步访问模式（partial region 读 + aligned 区域读、96B 缓冲合并成 128B flit）工作，bank 按 gang/slice 层级映射成平衡 channel（每 slice 16、每 chiplet 256）。使用方式：作为"权重/KV 内存布局 + channel 调度"的数据映射范式，decode 时每个 TE 经专属 channel 顺序流式读权重/KV tile；bank-per-channel 分组（2/3/4）影响带宽/延迟/冗余预算——论文实测 3-bank 设计（保留 72 bank 冗余）在 700MHz 达 105TB/s、2.5ns，2-bank 保留更多冗余、4-bank 为假设的多 bank die 配置。局限：需要软件显式控制 tile 布局（近单位步长流式访问假设），不适合随机/非流式访问模式。

涉及论文标题：
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
