## Bit-Plane Representation in Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-Plane（比特平面）在量化中指将量化权重按比特位置分解为多个二值矩阵层。在 BCQ 中每层 B_i ∈ {-1,+1}^{m×n} 是独立二值基，q 个平面构成 q-bit 表示。每平面存为 1 bit/元素（packed binary），比 FP16 压缩 16×。多精度推理时按需加载前 p 个平面（p=2 比 p=4 少 50% 数据），精准节约带宽。非均匀量化中比特平面存的是 centroid index 的二进制展开，而 BCQ 中直接是可计算二值操作数（{-1,+1}）——这是 BCQ 硬件效率的关键。数学本质：Ŵ = Σ α_i B_i，B_i 是"方向"，α_i 是"幅度"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
比特平面在 AnyBCQ 推理计算：

```
hidden ∈ R^{1×K}, BP_1...BP_p (每个 packed 为 K×N bits)

output = zeros(1, N)
for plane_idx in range(p):
    BP = load_packed_bitplane(plane_idx)  # 1-bit per weight
    partial = LUT_GEMM(BP, hidden)  # B_i ∈ {-1,+1}: 加减操作 + LUT加速
    alpha = load_scale(plane_idx, precision=p)  # α_i^{(p)}
    output += alpha * partial
```

存储格式：每平面 row-major 连续存储，每 byte 打包 8 个权重 bit。符号映射：0→-1, 1→+1。BCQ vs 非均匀量化的比特平面对比：总比特数相同（qN bits），但 BCQ 可直接操作（加减），非均匀需额外 bit-transpose + table lookup。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 上比特平面加载：以 byte/word 为单位，通过 bitwise AND + shift 提取单 bit。按需加载：仅加载所需 p 个平面（kernel 中通过 p 参数控制循环次数和显存读取量）。比特平面 vs centroid index 的运行效率：centroid index 方式需 (1) 加载 p 个平面 (2) bit-transpose 重组为 p-bit index (3) 查表获取 centroid 值 (4) GEMM；BCQ 方式仅需 (1) 加载 p 个平面 (2) 直接加减 + LUT 加速 (3) 缩放累加。BCQ 少了 bit-transpose（占 kernel 延迟 35-58%）和 centroid lookup（占 9-17%）。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
