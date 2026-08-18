## IV Bank（HBM 内显式 IV 管理）与同步计数式 IV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IV（Initialization Vector，初始化向量）是计数器模式加密（如 AES-GCM/CTR）中保证 OTP 唯一性的输入：同一 key 下 nonce（IV+计数器）绝不重复。本论文对比两种 IV 管理范式：(1) **同步计数式 IV**（当前 GPU CC baseline）：IV 由访问顺序隐式推导（IV_t ← increment(IV_{t-1})），CPU/GPU 靠对称访问保持同步（兼防重放），但加密只能同步执行、位于关键路径，无法提前/乱序；(2) **显式 IV Bank**（LÆGIS 提出）：在每个 GPU HBM stack 预留 8 MB（512K 行 × 128-bit）存 per-VABlock（2 MB）的 IV Bank Entry（IBE），IV 可随机访问、加密与同步解耦、支持乱序预加密，且无需完整性树（HBM 可信假设）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
IV Bank 的硬件运转：IBE 结构 = 19-bit ID + 77-bit 随机值 RV + V/D/O/R 控制位 + 9-bit CTR（128-bit 一行）；ID 复用页目录 PDE0 的 19 个未用位存于页表，页表遍历时取 ID → 索引 IV Bank（物理地址映射散列到 8 channel × 16 bank，1 KB row 存 64 个 IV、64 行/bank 单 sub-array）→ 内存控制器（MC）取 IBE → 经共享 crossbar 送 CE → 片上 16 项 IV cache（20-cycle 命中，LRU）缓存 → CE 构造 128-bit 输入 = 96-bit IV（RV||ID）+ 17-bit 块索引 + padding → 流水 AES 引擎生成 OTP 做加/解密。CPU 侧 IV Bank 存于 TDX 保护内存或 HBM。IV 构造规避冲突：不用地址相关 IV（页换出/预取改物理地址）、随机初始化用生日界分析（碰撞概率 q²·2^-95.26），故 ID 保证唯一性 + RV 保证随机性；RV 溢出触发 R 位 key rotation 重加密；4 KB/64 KB 小页共享所在 2 MB VABlock 的 ID/RV、用 9-bit CTR 记录已迁移 base 页数；换出时 RV 递增刷新整块。随机初始化/乱序加密兼容现有预取与逐出策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HBM-resident 元数据 + 固件管理地址映射（论文在 GPGPU-Sim 中建模 IV Bank、IV cache hit/miss 与写回策略）；硬件开销仅 8 MB HBM + 256B SRAM（0.02% L2），无新增密码引擎。使用：GPU 侧解密时 CE 从 HBM IV Bank 取 IBE 重建 IV；CPU 侧预加密线程同样从 CPU IV Bank 取 IV 提前加密；DMA 时加密页附带 128-bit（MAC||ID），GPU 据此取正确 IBE。多 stack/多 GPU 扩展只需固件更新地址映射或复制 IV Bank（8 MB/GPU），19-bit ID 支持至 1 TB HBM。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
