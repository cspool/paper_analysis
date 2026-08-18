## Subnormal 归一化与位宽扩展（Subnormal Removal via Bit-Width Expansion）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Subnormal（非规格化数）是 IEEE-754 中指数码为 0、尾数无隐藏前导 1 的浮点数：v=(−1)^S·2^(1−B)·(0+M)，用于向零平滑下溢。问题：subnormal 不满足 FPMA 的对数近似 log2(1+M)≈M（因为无前导 1），且低比特格式（FP4/FP3）中 25–50% 的可表示值是 subnormal，导致 FPMA 产生系统性大误差。UNICORE 的解法是位宽扩展（Bit-Width Expansion）：把低比特浮点操作数无损转换到更宽的内部格式（E3M2），使每个可表示值都成为正常数——对 subnormal 尾数左移直到落入 [1,2)，指数同步减小；要求目标格式尾数位 M'≥M 且指数范围 B'≥B+M（足以吸收 M 次归一化移位）。与 AxCore 的近似重映射（把 subnormal 映射到附近正常值、引入额外噪声）不同，这是精确变换。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 归一化流程（图 4，FP4 → E3M2）：任意 FP4 变体（E3M0/E2M1/E1M2）的 subnormal 形如 (0+M)，按尾数左移位数 s 归一化为 1.M'（M' 为 M 左移 s 位），指数 E'=E−s；E3M2 提供 2 个额外尾数位与更大指数偏置，保证归一化永不再下溢、全部有效位保留——FP4 的整个 subnormal 区域（最小到 E1M2 的 0.5）都落在 E3M2 正常域内。伪代码：
    ```
    if E==0 and M!=0:      # subnormal
        s = leading_zeros_normalize(M)   # 尾数左移 s 位至 [1,2)
        E' = 1 - B + s;  M' = M << s      # 在 E3M2 中成为正常数
    else:                  # normal 直接映射
        E' = E; M' = M
    ```
  - 关键设计：所有权重/激活仍以原始低比特格式存储传输，E3M2 扩展是仅存在于计算数据通路内的临时精确重编码（保留低比特存储带宽收益，计算位宽略有扩展但 FPMA 加法特性使其开销极小）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Unified Format Converter 内嵌归一化逻辑（尾数左移 + 指数补偿），与 DynFP 解码共用 LUT 通路；正常数直接映射、subnormal 左移归一化。使用：在 FPMA 计算前保证所有操作数为正常数，使 Mitchell 对数近似假设成立，消除低比特模式下的系统性误差；是 UNICORE 精度保持管线的第一级，之后接 FG/CG 双路径补偿。通用背景：IEEE-754 的 subnormal 处理也见于 FP8/FP4 格式研究（如 E2M1 变体的 subnormal 区间），vault 笔记 knowledge_notes/硬件知识笔记/Supernormal Support (SR_SP for Low-Precision Formats).md 与 E2M1 (FP4 Format).md 有低比特 subnormal 编码的相关讨论。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
