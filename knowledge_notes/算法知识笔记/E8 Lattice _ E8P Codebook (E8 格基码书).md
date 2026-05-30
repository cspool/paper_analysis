## E8 Lattice / E8P Codebook (E8 格基码书)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
E8 Lattice（E8 格）是 8 维空间中的一种特殊格（lattice）结构，由 Viazovska (2017) 证明其实现了 8 维空间中的最高密度球填充（optimal sphere packing），拥有最优的 kissing number（240，即每个球最多接触 240 个等大小球）。E8 格的数学定义：E8 = (Z⁸ ∪ (Z⁸ + ½)) ∩ {x | 1^T x 是偶数}，即所有整数向量和所有半整数向量中，分量和为偶数的向量集合。E8P（"E8 Padded"）是 QuIP# 基于 E8 格构造的 2-bit 8 维向量量化码书。E8P 包含 2¹⁶ = 65536 个码书条目（每个编码 8 个权重），但利用 E8 格的对称性（符号翻转），仅需存储 2⁸ = 256 条源码书（1KiB），解码时通过 7+1 位符号位和 1 位偏移位恢复完整的 16-bit 码字→8 维 FP 向量。E8P 的构造：从等价表示 D̂₈ = {x ∈ Z⁸ + ½ | 1^T x 是偶数} 出发，利用 (D̂₈ − ¼) ∪ (D̂₈ + ¼) = E8 + ¼ 的移位不变性，选择 S ⊂ |D̂₈| 中范数 ≤ √10 的 227 个元素 + 范数 √12 的 29 个"padding"元素，共 256 条源码书。解码时：8 bits 查 S 得绝对值向量，7 bits 控制 7 个符号翻转（第 8 符号由奇偶性推断），1 bit 控制 ±¼ 偏移。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
E8P 在 QuIP# BlockLDLQ 中的使用（以 2-bit 量化一列 8 维权重块为例）：
```
# E8P 解码一个 16-bit 码字 → 8 维浮点向量
# 输入: codeword (16 bits)
# 输入: codebook_abs[256] (256 条源码书，每条 8×4bit 压缩)

function decode_e8p(codeword):
    s_idx = codeword[0:8]           # 8 bits: 源码书索引 (0-255)
    sign_bits = codeword[8:15]      # 7 bits: 符号翻转控制
    shift_bit = codeword[15]        # 1 bit: ±1/4 偏移

    s = codebook_abs[s_idx]         # 查表得 8 个绝对值 (∈ |D̂_8|)
    
    # 确定需要奇数还是偶数个符号翻转
    # (取决于 s 是否在 D̂_8 中)
    parity_needed = parity_required(s)
    
    # 应用 7 个符号位，推断第 8 个
    for i in 0..6:
        if sign_bits[i]: s[i] = -s[i]
    # 第 8 个符号由 parity 推断
    s[7] *= sign_from_parity(sign_bits, parity_needed)
    
    # 应用 ±1/4 偏移
    offset = 0.25 if shift_bit else -0.25
    return [x + offset for x in s]

# 量化时: 找最近 E8P 码字
# w_8d ∈ R^8 → 遍历 256 条源码书，检查符号翻转组合
# → 找 L2 距离最小的码字 → 输出 16-bit 码字
# (实际实现使用 lattice 解码算法而非暴力搜索)
```
E8P 码书形状与 RHT 变换后的高斯分布的 MSE 最优性：图 3 显示 E8 基码书对所有基准码书（D₄ 格、半整数格等）实现最低的元素级 MSE，因为 E8 的 8 维高 packing density + 球状形状匹配高斯分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
E8P 的硬件效率设计：(1) 源码书仅 256×8 条目 × 4bit/条目 = 1KiB，可放入任何现代 GPU 的 L1 cache，即使 32× 复制消除 bank conflict 也仅 32KiB；(2) CUDA kernel 中解码一个 8 维向量需 <5 指令/权重——XOR 符号翻转 + 移位查表 + lop3 融合位操作；(3) 码字存储格式：每个 uint2 打包 4 个 16-bit E8P 码字（= 32 个权重），有效 2 bits/weight；(4) Tensor Core MMA：解码后的 FP16 权重直接送入 `mma.sync.aligned.m16n8k16` PTX 指令做矩阵乘累加，无写入 global memory 的中间步骤；(5) 高比特扩展（RVQ）：4-bit = 2×2-bit E8P，3-bit = 2-bit E8P + 1-bit E8；(6) scaling 因子 ρ 选择：ρ ≈ 0.9（2-bit E8P），通过最小化高斯→E8P 量化 MSE 确定。开源代码：https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quiptools/quiptools_e8p_gemv.cu。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---
