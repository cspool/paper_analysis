## HMUX 流水线功能单元（Rotation / Decomposition / FFT / VMA / IFFT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CASCADE 每个 HMUX Chiplet（HC）内部的专用计算单元集合，深度流水实现一次 HMUX 的完整数据通路：(1) Rotation Unit——对 ACC 做 negacyclic rotation（X^{a_i} 旋转）与多项式减法；(2) Decomposition Unit——对多项式系数做位分解（bit-slicing + 舍入），把 (k+1) 个多项式分解为 (k+1)×l 个，使外积变为多项式乘累加序列；(3) FFT Unit——把多项式乘法从 O(N²) 降到 O(NlogN)，log2N 级 butterfly，BU 个并行 butterfly 单元（每级处理 2·BU 系数，总约 log2N·N/(2·BU) cycle），FFT 控制器按 [26] 的 conflict-free 地址生成避免访问冲突；(4) VMA（Vector Multiplication-Add）Unit——FFT 域逐系数乘加实现外积（向量乘法单元 + 累加器），IP=256 并行度；(5) IFFT Unit——转回时域。因 Decomposition 使 FFT 侧多项式数多于 IFFT 侧，FFT 分配更多资源维持流水利用率。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一个 HMUX 在 HC 内的系数粒度数据通路）：ACC_{i-1} 系数流进入 → Rotation（X^{a_i}·ACC − ACC 类旋转+减法）→ Decomposition（按基位切片+舍入成 l 层）→ FFT（逐级 butterfly，conflict-free 取数）→ VMA（BSK 频域多项式逐系数乘加、累加部分积）→ IFFT → 输出 ACC_i 系数。所有单元以 PCG 粒度流水重叠；单元间流式传输、double buffer 缓冲。外积（VMA）占计算成本主体；key-switching 等轻量操作不在这些单元内执行，而由 HC0 的 VPU 承担（避免打断 HMUX 流水）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 综合（TSMC 28nm、1.2 GHz），单 HC 面积 92.5 mm²、TDP 29.91 W；功能单元面积/功耗见表 II（VMA 22.2 mm²/11.6 W、BSK SRAM 35.5 mm²/5 W 等）。使用：五个单元固定实例化（非可重构），依靠 IP（VMA 并行度）与 BU（FFT butterfly 数）两个参数调节性能/面积；设计空间探索中 IP=256 提供最佳面积归一化性能（IP<256 计算受限、IP>256 边际收益递减）。这些单元与 AutoFHE 的 CPE 模板（PoV/PoD/BFU/IBFU/PoE）在功能上对应（旋转/分解/FFT/IFFT/外积），但 CASCADE 是固定架构实现而非模板生成。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
