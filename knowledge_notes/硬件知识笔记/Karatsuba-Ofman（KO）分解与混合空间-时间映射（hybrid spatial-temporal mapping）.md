## Karatsuba-Ofman（KO）分解与混合空间-时间映射（hybrid spatial-temporal mapping）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Karatsuba-Ofman（KO）是宽整数乘法的分治算法：把两个大操作数各拆成 n 个等大小 chunk，用少量加法/减法（Pre 步骤）换取更少乘法，再经轻量 Post 合并。KO-n 用 2n−1 个乘法替代 schoolbook 的 n² 个。KO-2（两个 2m-bit 操作数 a=a1|a0、b=b1|b0）：Pre 算 a01=a0+a1、b01=b0+b1，三个乘法 r0=a0×b0、r1=a1×b1、r01=a01×b01（schoolbook 需 4 个），Post 算 r'=r01−r0−r1 并合并 (r1|r0)+(r'<<m)。KO-3 用 6 个乘法替代 9 个。Toom-3 只需 5 个乘法，但含常数除法需专用逻辑、5 个乘法更难映射 lane，故 GenZA 选 KO 保持硬件简单。
- 本论文角色：GenZA PE 用 KO 把多 bitwidth（128/256/384/768-bit）递归分解到 64-bit lane（128/256 用 KO-2；384 用 KO-3→KO-2；768 用 KO-3→KO-2→KO-2），PE 内置 4 组 Pre/Post 阶段（廉价：仅加法器/MUX/位重排），配合 32 条 64-bit 乘法 lane 组成 M=32/lane 个宽乘法器。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 关键创新是 chunk 乘法到 lane 的映射。之前方案：(1) 完全空间分解（ReZK）——把 n 个乘法映射到 n 条 lane，但 KO-2 的 3 个/KO-3 的 6 个乘法在 32 条 lane 的 power-of-two 分配下资源碎片化；(2) 完全时间分解——单 lane 串行执行，768-bit 乘法要 54 cycle 且低并行度 kernel（MSM 桶归约、树式 sumcheck/Merkle 近根层）无法占满 32 lane。GenZA 混合方案：KO-2 分配 2 lanes、KO-3 分配 4 lanes，单次分解 2 cycle 完成（KO-2 利用率 3/(2×2)=75%、KO-3 6/(4×2)=75%，与完全空间相同），但两个独立计算交错（interleave）在分配 lane 上即可达 100% 利用率——所需并行度仅为完全时间的 1/2（KO-3）。
- 流程例子（MNT4-753 的 753-bit 乘法，Figure 2c）：753-bit 拆到 64-bit chunk → 逐级 KO Pre/Post 阶段（KO-3→KO-2→KO-2）重组 bits → chunk 乘法在 lane 上混合时空执行 → 高位累加进位（66-bit 物理乘法器容纳 overflow）→ 最终 753-bit 积。面积代价：PE 去除 scratchpad 后面积介于 384-bit 与 768-bit 全流水模乘器之间（Table VII），768-bit 模乘吞吐 0.2/cycle，吞吐/面积 = 专用 768-bit 设计的 0.53×（多 bitwidth 灵活性的代价）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PE 内 32 lanes（每 lane 64-bit 乘法器 + 2 个 64-bit 模加/减器，物理 66-bit）+ 4 组 KO Pre/Post 阶段 + 可配置 lane 映射（M lanes 组成一个宽乘法器，PE 含 32/M 个）。使用：运行时每 PE 收模式配置（bitwidth、场模、lane 映射）；MNT4-753 全 32 lane 组单 PADD 单元，BN128 每 PE 2 个 PADD 单元（4 宽乘法器）。跨论文复用：任何需要"一套硬件支持 64–768-bit 模算术"的加速器（ZKP/FHE/ECC）都可借鉴该分解+混合映射。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
