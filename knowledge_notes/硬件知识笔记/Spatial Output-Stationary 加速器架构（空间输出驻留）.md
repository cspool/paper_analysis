## Spatial Output-Stationary 加速器架构（空间输出驻留）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spatial output-stationary 是 TFHE 加速器的经典数据流组织（Morphling HPCA'24、Matrix TACO'25）：每个 PE 行在整个 PBS 期间持有部分累计的 GLWE 多项式（输出驻留在 PE 内），而 BSK 以严格单遍流从 DRAM 扫过整个 PE 阵列，每 PE 消费经过它的 BSK chunk 后即弃。由于每个 BSK 系数在一次 bootstrapping 中恰好被消费一次、BSK 又远大于片内容量，架构必须为每次 bootstrapping 重新流式取整份 BSK，无时间局部性可挖。三种并行度被硬映射到阵列几何：coefficient 并行（PE 内 N 系数级 MAC）、row 并行（同一 post-FFT 系数广播到一行 PE 做 (k+1) 个点积）、ciphertext 并行（不同行处理不同 PBS，设计时硬编码 batch 大小）。
- 该组织对 Boolean/低 bit-width TFHE 有效（N 小、k=2~3、key 仅 MB 级），但 multi-bit TFHE 的大参数集使三维并行全部恶化（见"TFHE 三级并行"条目）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（以 Morphling 式 4×4 PE 阵列为例子）：BSK chunk 从 DRAM → NoC 广播到一行 4 个 PE（row 并行，做 k+1 个独立点积）→ 每 PE 的 FFT（8-parallel R2MDC）把 GLWE 多项式变换后与 BSK subchunk 在 PE 内 MAC 并累计 → 下一 chunk 继续，直到整个 BSK 扫完 → 输出 GLWE 结果。瓶颈：(1) 加宽 PE（coefficient 吞吐 ×c）要求 BSK 流带宽也 ×c，Morphling 扩到 2× 系数吞吐就超过两栈 HBM2E 供给，变带宽受限；(2) k=1 时每 FFT 输出仅 2× 复用，宽 PE 行大量空闲；(3) ciphertext 并行需复制每行的 FFT 流水线 + 累加器（N=65536 时 ~59.5MB/条），面积不可承受。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PE 阵列 + 每行独立 R2MDC FFT 流水线与累加器（N=4096 时 in-flight state ~16KB）+ 输出驻留 MAC 单元；BSK 由 DRAM 直通阵列。使用：适合 Boolean/低 bit-width TFHE（MATCHA、Morphling、Matrix）；对 multi-bit 需扩展 FFT 支持 N（Morphling/Matrix 仅到 4096）。FlashTFHE 论文把 Morphling 扩展出 Morphling'（R2MDC FFT 支持 multi-bit 参数集）作为空间 baseline，消融与对比（Figure 20/22）量化其带宽/利用率/面积三重限制，说明为什么必须转向时间域复用。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
