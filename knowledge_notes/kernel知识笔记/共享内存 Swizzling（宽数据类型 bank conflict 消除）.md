## 共享内存 Swizzling（宽数据类型 bank conflict 消除）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 共享内存 Swizzling 是对共享内存地址做位操作重排（典型为 XOR 高位/低位）以消除 bank conflict 的布局技术：NVIDIA 共享内存组织为 32 个 bank（每 bank 4 字节），同一 warp 内多个线程同周期访问同一 bank 的不同地址会触发冲突、访问被串行化（最多 32-way）。MNEMOS 的场景特殊在"宽数据类型"：每个元素是复数 FP64（16 字节），横跨 4 个连续 4 字节 bank——普通每元素单 bank 的 swizzle 模式失效，需设计"任意 8 个连续线程映射到 32 个互异 bank"的模式（图 9 用 4×8 矩阵示例，不同颜色为 8 lane 子组）。这与 Hopper 上 TMA 的 32B/64B/128B swizzle（ThunderKittens/TMA 文档）同族，但为 16B 元素定制。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS FFT kernel 的转置在共享内存完成，朴素行列访问引发严重 bank 冲突。伪代码（16B 复数元素的 XOR swizzle）：
```
# 线程 t 访问逻辑地址 addr(t) = row*W + col（16B 元素，索引为元素单位）
# swizzle：把元素地址的若干位 XOR 到高地址位，使同周期活跃线程分散到 32 bank
swizzled = addr ^ ((addr & mask_lo) << shift)      # 位交换
# 设计目标：任意 8 个连续线程 → 32 个互异 bank
# 校验：每线程 16B 占 4 bank，8 线程 × 4 bank = 32 bank 全覆盖、零重叠
```
- Annotations：`mask_lo`/`shift` 按元素宽度与矩阵形状选取（16B 元素需跨 4 bank 粒度设计）；消除转置阶段的 stall_MIO_throttle（共享内存/L1 争用）——MNEMOS 实测 FFT 的该 stall 延迟降 3.2×（N̄=256）；64 点特化进一步用 WMMA fragment 布局省掉一次显式转置（详见"四步 FFT"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CUDA kernel 内手工地址计算（如 `__shfl_xor` 无关的位运算），或由框架自动布局——Triton/TileLang 的 MakeSwizzleLayout、ThunderKittens 的 32/64/128B swizzle、TMA 的 swizzle 描述符（cuTensorMapEncodeTiled）均自动处理标准宽度元素。MNEMOS 因 16B 复数 FP64 超出框架默认宽度而手工设计。使用要点：(1) 先确认元素宽度与 bank 宽度的关系（16B=4 bank）；(2) 转置/对角访问是 bank 冲突高发点；(3) swizzle 与 padding 可组合（padding 破坏对齐、swizzle 保持对齐，TMA/HGMMA 兼容性上 swizzle 更优）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
