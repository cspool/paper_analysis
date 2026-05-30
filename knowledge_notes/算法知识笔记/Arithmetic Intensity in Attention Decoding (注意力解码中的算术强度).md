## Arithmetic Intensity in Attention Decoding (注意力解码中的算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

算术强度（Arithmetic Intensity, Williams et al., 2009）定义为每字节内存访问所执行的浮点运算次数（FLOPs/Byte），是 Roofline 性能模型的核心度量——判断 workload 是 memory-bound（算术强度低于硬件 roofline 拐点）还是 compute-bound（算术强度高于拐点）。

在 LLM 解码中，标准 MHA 的算术强度极低（约 1 FLOP/byte）：每从 HBM 加载的 BF16 KV 元素（2 bytes）仅执行 1 次 MAC（2 FLOPs），而 H100 的临界算术强度为 989 TFLOPS / 3350 GB/s ≈ 295 FLOPs/byte——MHA 解码时 GPU 利用率可低至 7%（Recasens et al., 2025）。

该论文推导了通用算术强度公式：
$$\text{AI} \approx \frac{2 \cdot L}{2 + \frac{m_{kv}}{g_q} \cdot L} \approx \frac{2g_q}{m_{kv}} \quad (L \gg h_q)$$

其中 g_q = h_q / h_{kv}（group size），m_{kv} ∈ {1,2}（1=共享 KV state，2=分离 K/V），L 为序列长度。

从算法pipeline角度拆解，给出具体例子。

**提升算术强度的三个独立维度**：

1. **增加 g_q**（更大 group size）：更多 query head 共享一个 KV head → 同一 KB 加载服务更多 FLOPs
   - MQA: g_q = h_q, AI ≈ h_q
   - GQA-4: g_q = 4, AI ≈ 4
2. **减少 m_kv**（KV tying, m_kv: 2→1）：分离 K/V → 共享 state → 内存加载减半 → AI 翻倍
   - GQA: m_kv=2, AI ≈ g_q
   - GTA: m_kv=1, AI ≈ 2g_q
3. **增加 h_q**（更多 query heads）：MLA/GLA 通过低秩参数再分配在保持 latent 维度不变的前提下增加 query head 数
   - MLA: h_q=128, AI ≈ 256（接近 H100 compute roof）

**Roofline 分析（H100 BF16, L_q=1 vs L_q=2）：**

| 方法 | AI (L_q=1) | AI (L_q=2) | 状态 (L_q=1) | 状态 (L_q=2) |
|------|-----------|-----------|-------------|-------------|
| MHA (hq=128) | ~1 | ~1 | Memory-bound | Memory-bound |
| GQA-4 (hq=128) | ~4 | ~4 | Memory-bound | Memory-bound |
| MQA (hq=128) | ~128 | ~128 | Bandwidth roof | Bandwidth roof |
| GLA-2 (hq=128) | ~128 | ~128 | Bandwidth roof | **Compute inflection** |
| MLA (hq=128) | ~256 | ~256 | Near compute roof | Beyond compute roof |

术语一般如何实现？如何使用？

实践中指导 attention 变体选择：目标 hardware roofline 拐点决定了理想算术强度。H100 上 L_q=1 解码时，MLA（AI≈256）接近 compute roof→更高效的 compute 利用；GLA-2（AI≈128）在 bandwidth roof→内存带宽利用更高（93% vs FlashMLA 72%）。L_q=2（推测解码）时，GLA-2 达 compute inflection point→2× faster than FlashMLA。算术强度分析是 GTA 和 GLA 设计的核心指导原则。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---
