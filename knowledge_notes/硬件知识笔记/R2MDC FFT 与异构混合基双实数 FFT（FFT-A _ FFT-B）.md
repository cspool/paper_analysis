## R2MDC FFT 与异构混合基双实数 FFT（FFT-A / FFT-B）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- R2MDC（Radix-2 Multi-path Delay Commutator）是流式 FFT 架构，TFHE 加速器的经典 FFT 实现（Morphling 用 8-parallel R2MDC）：深度流水、每流水线 in-flight 状态随 N 线性增长（N=4096 时 ~16KB，N=65536 时 ~59.5MB/条），通过复制流水线提升吞吐。双实数 FFT（double-real FFT）用一个 N/2-point 复 FFT 处理两个 N-point 实序列或一个 N-point 实多项式（把实部/虚部打包两个实序列），FlashTFHE 用 2^15-point 复 FFT 处理 2^16 度实多项式。异构混合基 FFT 是 FlashTFHE 的关键设计：2^15 不是完全平方数（不像 CraterLake 的 256×256 可拆成同构单元），故用两种功能单元——FFT-A 处理 256-point 序列（对称设计、√256 lanes、tile 化）与 FFT-B 处理 128-point 序列（非对称设计，拆 4×32-point 再各拆 8-point tiles），经 transpose 单元互连（divide-and-conquer 大输入）；蝴蝶单元用混合基，radix-4 比 radix-2 少 25% 复数乘法；支持 early-exit 跳过未用 stage（TFHE 的 N 可变，不像 CKKS 固定 65536）。
- 面积/吞吐：异构 FFT 集群相对 8-parallel R2MDC 面积 1.38×、吞吐 32×（单 core 数据）；论文对比中一个 FFT 集群 = 2×FFT-A（1.57mm²/2.95W）+ FFT-B（1.88mm²/4.12W）+ transpose（0.79mm²/1.44W）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 BRU 中运转：GLWE 多项式（2^16 度）→ double-real FFT 打包成 2^15-point 复序列 → 经 transpose 单元在两型单元间切分：256-column 与 128-row 的矩形中间数据（每复数 96-bit = 48-bit 实 + 48-bit 虚）产生吞吐不平衡（读 256 列耗时 2× 读 128 行），FlashTFHE 把相邻复数两两分组形成 128 逻辑列、按奇偶索引分发到两个 FFT-B 并行处理 → FFT 输出每周期一 chunk → VecMAC 与 BSK subchunk 乘累加。I-FFT 在两个 BRU 间共享（比例 l_b:1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Chisel HDL + FPGA 验证 + N16 综合；transpose 中间数据用两端口 SRAM bank 高密度存储。使用：支撑 N 至 2^16（10-bit ciphertext、128-bit 安全），远超 Morphling/Matrix（N≤4096）与 Strix（N≤16384）；early-exit 适配 TFHE 各 workload 的可变 N（CNN N=2048/4096、GPT-2 N=32768、DecisionTree/KNN N=65536）。48-bit 定点保证全参数集正确性（对比 Trinity/UFC 用 NTT，论文实测 16nm 下 NTT 块面积 2.73×、功耗 3.79× 于 FFT，是 Trinity 面积劣势来源之一）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
