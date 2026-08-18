## Memory bank conflict（内存 bank 冲突）与冲突无关查找

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SRAM/片上存储按 bank 组织，多个请求同时访问同一 bank 时发生 bank conflict：被串行化（硬件强制错开）或阻塞。VQ 的权重码本（WC）查找是典型冲突源：WI 索引不规则（1-to-many），同列索引常落同一 bank（如索引 5 与 3 同 bank），导致 decode 性能瓶颈。现有方案靠复制（GOBO/LUT-DLA 寄存器复制）或广播（FIGLUT 16×32×(8×16bit) 广播带宽）码本到多 PE 缓解，代价是面积/带宽剧增且只能支持 ≤16 条目码本；VQ-LLM 用 hot/cold 频率 profiling 只缓解不消除。EVA 从根上消除：改为从输出码本（OC）查找——OC 由 GEMM 计算产生，其行与 WI 行共享高度 V=K/d，每行映射独立 bank，同列不同索引自动落不同 bank（对应不同输入向量 v0/v1），完全并行无冲突；且带宽从 d 个 FP16/访问降为 1 个。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 冲突示例（WC 查找）：WI 列 [5, 3] 同时访问 WC bank → bank[5%B]==bank[3%B] 冲突 → 串行
# EVA OC 查找：OC 行 i 映射 bank i，WI 索引 (row, col) → bank=row，
# 同列不同行（5、3 行）→ bank 5 vs bank 3 → 无冲突并行
```
Table X 量化：WC 4-bank 配置下全冲突 VQ 1.00×、VQ-LLM（50% 冲突）1.74×、EVA 无冲突 2.06×、EVA EU-32×4 达 64.84×；OC 查找把 Codebook SRAM 存储/带宽需求降 8×（d=8→1：4KB→2KB/16KB/64KB 随 EU 扩展，带宽 64B/cycle→8/64/256B/cycle）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：多 bank SRAM（EVA 的 OC buffer 32 bank×1 FP16/cycle，4 EU 时 256B/cycle）；bank 数与行数对齐（每 OC 行一个 bank）。使用方式：冲突分析是 LUT 加速器设计的核心（Table I 对比 GOBO/FIGLUT/LUT-DLA 的复制/广播开销）；EVA 证明"换查找对象"（查 OC 而非 WC）比"复制/广播码本"更优——不增加 bank、带宽更低、冲突为零。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

HiT 补充视角（ISCA'26，bank conflict 作为 Gustavson 大规模化的瓶颈）：HiT 系统分析了 bank conflict 如何随并行度放大：Gustavson 数据流下每个 PE 行独立请求所需 B 行（由 A 非零决定），并发请求映射到共享 32-bank cache，同周期命中同一 bank 即冲突、强制 stall（图 5：两个 PE 都请求 Bank 1，只执行 1/4 乘法）。模型化 32-bank SRAM（每 bank 每周期 1 R/W）后显示：速度提升在 256 PE 行时饱和、MAC 利用率掉到 ~10%，且冲突概率随请求数增加——结论是"大规模 Gust 稀疏加速器的瓶颈是内存子系统而非计算"，增加 bank 数虽降冲突概率但 crossbar 面积/功耗二次增长不可行。HiT 的对抗方案是把访存变成"结构化流式"：外积数据流按列处理 A、B 行顺序从专属 bank 流式读取（每 Compute Row 静态连接 4 个 Global Memory bank，互不访问其他 bank），从根源消除并发随机访问，使吞吐随计算并行度扩展。
