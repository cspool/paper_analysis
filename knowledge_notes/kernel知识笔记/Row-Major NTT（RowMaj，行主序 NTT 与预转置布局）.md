## Row-Major NTT（RowMaj，行主序 NTT 与预转置布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RowMaj 是 HyperDrive 的 Outer-NTT 内存布局优化：4-step NTT 把多项式看成 2D 数组、先按列做内层 NTT，导致 3/4 的 GMEM 访问是列主序（strided、非连续），无法 coalescing，带宽利用率低并引发 pipeline stall（baseline stage1/2 中 GMEM 相关 stall 占 45.9%/22.6%）。RowMaj 引入预转置（pre-transposed）数据格式：密文、密钥、明文在 GPU 生命周期内以转置布局存储，使原本的"列"变为"行"访问，3/4 的 GMEM 访问变为行主序全 coalesced。
- 预转置只改内存布局不改数值；转置成本近零——通过多项式操作时直接按转置布局输出（而非单独转置 kernel），且在编码/解码阶段对单 limb 数据完成（RNS 分解前/重构后），避免对多 limb RNS 表示做昂贵转置。配合 TFOP（Twiddle Factor access OPtimization）：negacyclic 卷积 twiddle 顺序读（INTT 预乘 N^{-1} 省一次模乘）、外层 Hadamard 用预排序的 TF-XY、内层 Hadamard/Residual NTT 用 SMEM 中的 TF256（≤256 元素）、Inner-NTT 的 8×8 TFM 驻 SMEM 跨 warp 复用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Row-Major NTT 的两 kernel 数据流（Alg. 1 + 图 10）：
```
# 编码阶段（单 limb）：按转置布局写入（零成本）
encode(): coeff → 预转置布局（RNS 分解前）
# Kernel 1：并行块行主序 coalesced 读 GMEM → SMEM
a = GMEM-Load(行主序)            # 全 coalesced 128B 事务
SMEM ← a ⊙ ζ_2N[1:N]             # negacyclic 卷积（顺序读）
for each 64 点 Inner-NTT: 寄存器内 MMA 流水（TLMOP/TransOP）
外层 Hadamard EWMult（TF-XY，与 Kernel 2 起点对齐）
行主序写回 GMEM（转置并入写回阶段，无单独 kernel）
# Kernel 2：读回 → 内层 Hadamard（TF256 复用）→ Residual NTT → 全局转置 → 输出
```
- Annotations：优化前基本 NTT kernel 有 7 次 sparse + 4 次 limited-locality GMEM 访问 vs 1 次 coalesced；优化后只有 1 次 limited-locality、其余 9 次全 coalesced。twiddle 存储：每模数 negacyclic 2N 个、外层 Hadamard N 个、TF256/TFM 各 256 个，36 层共 27MB GMEM、每 block 2KB SMEM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 GPU FHE 库的编码/解码路径维护预转置布局，NTT/INTT kernel 按行主序读写，Auto（索引置换）kernel 保持全 coalesced 输入读，元素级 kernel（EWAdd/EWMult）不受影响；关键收益是它消除 NTT 的"多 pad"约束，使跨多项式 kernel（BConv/IP）能与 NTT 融合（见 COOP 条目）。效果：TFOP+RowMaj 使 GMEM stall -59.5%、scheduler stall -27.4%（累计 -39.3%）；H100 上 NTT 吞吐 1669.5 KOPS 验证跨代可扩展。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
