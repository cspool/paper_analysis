## Montgomery 模运算（通用域 Montgomery 乘与特殊域简化约简）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Montgomery 模乘（Montgomery 1985）是避免除法的大数模乘算法：把操作数变换到 Montgomery 域（a·R mod p，R=2^k>p），用整数乘法与廉价无除法的约简完成模乘，最后再变换回原域。免除法使其在硬件/软件中大数模乘中占主导。GenZA 用它支持通用大素数域（256–768-bit EC 域，如 BN128/BLS12-381/MNT4-753）；对特殊模（64-bit Goldilocks p=2^64−2^32+1）则利用模值特殊性用 lane 内几次加/减完成约简，不走通用 Montgomery。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 GenZA PE 中（Figure 2a 的 modular multiplier stage）：KO 宽乘法器算整数积 → 配置 adder chain 多周期调度完成 Montgomery 约简（乘 R^{-1} 与模 p 的约简步骤）→ shifter 从乘法器中间积选高低半部分喂给归约。Montgomery 归约与 KO 乘法合起来决定 NTT 蝴蝶的 initiation interval：II≈6.75 cycle（256/384-bit）、10.125 cycle（768-bit）。Goldilocks 域：每 lane 内直接用 p=2^64−2^32+1 的等式用几次加/减完成约简（无需 Montgomery），匹配 Plonky2 的标准 64-bit 实现，也是 lane 内向量运算（Poseidon 等）的底座。
- Annotations：模约简是全流程的一部分——PE lane 的乘法器/模加器做 KO 分解后的子运算，Montgomery 级做最终域约简，二者配合支撑 NTT 蝴蝶 a±b·ω、EC PADD 的 14 次模乘等。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库用 CIOS/FIOS 等 Montgomery 实现；硬件用流水化 Montgomery 乘法器（GenZA 把通用 Montgomery 与特殊模简化混合，避免给特殊模也配昂贵通用单元）。使用：任何 EC/大数模运算加速器（ZKP 的 MSM/NTT、FHE、RSA/ECC）；GenZA 以"可配置 adder chain + shifter 选半"实现通用域，以"lane 内加减"实现 Goldilocks，展示一套 PE 同时服务两种域的配置方式。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
