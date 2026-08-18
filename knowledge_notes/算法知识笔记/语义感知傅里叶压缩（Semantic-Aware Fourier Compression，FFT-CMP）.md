## 语义感知傅里叶压缩（Semantic-Aware Fourier Compression，FFT-CMP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FFT-CMP 是 MLX 提出的序列维压缩方法：利用 LLM 层沿序列维 N 的语义频率局部性，把 Q/K/V 的 FFT 频谱中能量较低的高频分量截掉，再把保留的低频系数逆变换回一个更短的 token 表示。关键洞察：浅层 transformer 关注局部细粒度 token 细节（能量在高频），深层编码更广上下文（能量偏向低频）——论文对 Llama2-7B 各层 Q/K/V 做 FFT 验证（Fig.5/6：layer 1 高频主导、layer 16 低频主导）。对每层定义语义 chunk 长度 L=N/f_H（f_H 为能量超过相对阈值的最谱峰，Pow2Round 到 2 的幂做硬件友好对齐），把 Q,K,V∈R^(N×D) 重塑为 N/L 个 chunk，每 chunk 沿序列维做 L 点 FFT、保留前 sL 个低频系数、sL 点 iFFT 生成缩短表示。prefill 代价从 O(N²D) 降为 O(s²N²D)，附加 chunked-FFT 开销仅 O(ND log L)。s 是可调压缩率（评估 s=0.5/0.75）。该方法的优点：保留信息丰富的低频分量（对照 FNet 式全局 2D-FFT 的精度损失）、按层自适应、天然兼容 decode 的增量更新（chunk-granular 压缩 KV cache）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FFT-CMP 的 prefill 与 decode pipeline：
```
# 离线/每层：确定 L = Pow2Round(N / f_H)（f_H = 能量超阈值的高频谱峰）
# prefill（Q 示例，K/V 同）：
for c in 0..N//L-1:
    F_c = FFT_L(Q[cL:(c+1)L, :])          # 每 chunk L 点 FFT（每特征维）
    Qs[c] = IFFT_{sL}(F_c[:sL, :])        # 截断到 sL 低频系数 → 缩短 token 表示
# 注意力在缩短序列上执行：Attn = softmax(Qs·Ks^T/√d)·Vs（注意力矩阵 sN×sN）
# decode（append-only chunk 压缩 KV cache）：
#   新 token 累积到 L 才触发一次 FFT 压缩 → append 新压缩块；已完成 chunk 复用缓存压缩块
#   固定 L，不重变换整个 prefix；FFT 开销在 L 个 token 上摊销
```
张量计算例子（Llama2-7B、N=2048、D=4096、L=256、s=0.5）：Q 重塑为 8 个 256×4096 chunk，每 chunk 256 点 FFT、保留 128 个低频系数、128 点 iFFT → 8×128×4096=1024×4096 缩短 Q；注意力矩阵 2048×2048→1024×1024（4× 缩减），KV 流量与缓冲压力同步下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) 算法：QKV 投影后按层 FFT 压缩（BERT 上可逐层应用 k 层：替换全部 12 层达 69% FLOP 削减、仅 1.75% EM/1.3% F1 损失；ViT 上 s=0.5 达 65% FLOP 削减、1.6% 精度降，优于 FNet 2-3% 损失）；(2) 软件：PyTorch 层实现（torch.fft），H100 prefill 长序列 2.72× vs eager / 1.64× vs FlashAttention2（未融合 FA、TensorCore 不支持蝴蝶故受限），decode 减少 KV-cache 流量贡献 1.4-1.9×；(3) 硬件：MLX 上 FFT-CMP 与 BSMM 在 SIMD-striped scratchpad 上对齐（列向 SIMD lane 对齐序列轴 N 做 BSMM、行向流式隐藏轴 D 做 chunk FFT，避免全阵列转置），L 点 segment 形成闭集依赖。局限：s 过小精度下降、短序列收益有限（H100 上短序列无明显加速）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
