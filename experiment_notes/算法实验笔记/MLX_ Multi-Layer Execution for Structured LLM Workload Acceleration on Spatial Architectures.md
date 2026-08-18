## MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：**语义感知傅里叶压缩（Semantic-Aware Fourier Compression, FFT-CMP）+ 分层蝴蝶分解（Hierarchical Butterfly Decomposition, hierarchical BSMM）** 构成混合化 Transformer block（hybridized transformer block）。(1) FFT-CMP：观察到 LLM 层沿序列维呈现语义频率局部性（浅层偏高频细粒度 token 细节，深层偏低频上下文抽象，经 Llama2-7B Q/K/V 频谱验证），对投影后的 Q,K,V ∈ R^(N×D) 按每层 chunk 长度 L（L=N/f_H 取 power-of-two，f_H 为能量超过相对阈值的高频谱峰）分块，每块做 L 点 FFT→截断保留前 sL 个低频系数→sL 点 iFFT 生成缩短 token 表示（丢弃低能量高频分量，s 为可调压缩率）。prefill 代价从 O(N²D) 降到 O(s²N²D)，附加 chunked-FFT 开销仅 O(ND log L)。decode 侧用固定 L 的 append-only chunk-granular KV cache：新 token 累积到 L 才触发一次 FFT 压缩并追加新压缩块，已完成的 chunk 复用缓存压缩块，兼容 KV-cache 解码。(2) hierarchical BSMM：把权重矩阵 W 分成 (D/B)×(D/B) 个 B×B tile，仅在每个 tile 内应用蝴蝶因子，参数/计算复杂度从全局分解 O(D log D) 降到 O((D²/B) log B)；B 是第二个可调精度-效率旋钮（论文评估 B∈{16,32,64}，B=32 最佳）。两者结合在序列维 N 与隐藏维 D 两个正交维度暴露并行性。
  - 实验比较什么：(a) 算法精度-计算折中：FFT-CMP（s=0.5）在 ViT 上达到 65% FLOP 削减仅 1.6% 精度下降，优于 FNet 式 2D-FFT（同 FLOP 削减但 2-3% 精度损失）；BERT 上 k 层渐进应用（s=0.5）替换全部 12 层达 69% FLOP 削减、仅 1.75% EM 与 1.3% F1 损失；Llama2-7B/InternLM2-7B 用 LoRA 微调压缩层，>60% 层应用后 s=0.75 削减 57-64%、s=0.5 削减 67-72% 的 QKV+Attention 计算，总体精度下降 <1.45%（Winogrande-xl/Wikitext-2/103/Ada-LEval）。(b) H100 上对比原模型 eager attention 与 FlashAttention2（保守设置 s=0.5,B=32）：prefill 长序列最多 2.72×（vs eager）/1.64×（vs FA），短序列收益小（因 FFT-CMP 在 PyTorch 层实现、未与 FA 融合，TensorCore 对蝴蝶稀疏支持有限、回退到 CUDA core）；decode 结合块 BSMM 达 1.4-1.9× 端到端加速。
- 硬件平台是什么，配置是什么。
  - 算法验证运行平台：NVIDIA H100 GPU（prefill N=512/8K，cuFFT 优化的 roofline 分析）；NVIDIA AGX Orin（batch 64，N=512/8K，FFT-based transformer block 相对 dense baseline 仅 3.77×/2.56× 加速）；Jetson Xavier（12nm、1.7 TFLOP/s CUDA 峰值、6 TFLOP/s TCU 峰值、15W，16 GB 内存）；RTX-3090（structured-workload sweep）。FFT 用优化 cuFFT；kernel 以 PyTorch 层实现。
- 模型是什么。数据集和bench分别是什么。
  - 模型：ViT（196/1K patch，从头训练验证蝴蝶稀疏理论）、FABNet（128/768）、BERT（8K/1K，EM/F1 指标）、BERT-SQuAD（B0，512/1K）、InternLM2-7B（GQA，1K-4K/4K）、Llama2-7B（128-2K/4K）。数据集与 bench：Winogrande-xl（N=512）、Wikitext-2/103（1K/2K）、Ada-LEval（1K/2K/4K）；LLM 精度评估用 LoRA 微调（QA-LoRA 式）压缩层。对比基线：FNet 式 2D-FFT token mixing、dense Transformer、全局蝴蝶分解（Monarch/butterfly factorization）等。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文（ISCA 2026，ICT CAS，Best Paper Candidate）未提供任何开源代码或仓库（联网搜索未找到）；评估以 PyTorch 层实现（FFT 用 torch.fft，BSMM 可用 torch 矩阵分块实现）。算法 pipeline 伪代码（一次 prefill 的注意力阶段，chunk 长度 L、压缩率 s）：
    ```
    # 1) 逐层确定语义 chunk 长度：f_H = 最高能量超过阈值的高频谱峰；L = Pow2Round(N/f_H)
    # 2) FFT 压缩（对每个 chunk，沿序列维）：
    for each chunk c in range(N//L):                 # N/L 个 chunk
        F_c = FFT_L(Q[c*L:(c+1)*L, :])               # L 点 FFT，每特征维
        F_c_trunc = F_c[:s*L, :]                     # 保留前 sL 个低频系数
        Qs[c] = IFFT_{sL}(F_c_trunc)                 # sL 点 iFFT，得缩短 token 表示
    # K、V 同样处理；压缩后序列长 sN，注意力矩阵降为 sN×sN
    # 3) 注意力在缩短序列上执行：Attn = softmax(Qs·Ks^T/√d)·Vs
    # 4) QKV/FFN 投影用 hierarchical BSMM（B×B tile 内蝴蝶分解）：
    #    W → 分 (D/B)×(D/B) 个 tile；每 tile W_b = ∏_{k=1}^{log2 B} B_B^(k)（B×B 蝴蝶因子）
    #    输出 Y = X @ W 等价于逐 tile 做 B×B 蝴蝶稀疏矩阵乘，复杂度 O(D²/B·log B)
    # 5) decode：新 token 累积到 L 才 FFT 压缩一次并 append 新块到压缩 KV cache
    ```
    张量计算例子（Llama2-7B、N=2048、D=4096、L=256、s=0.5）：Q∈R^(2048×4096) 重塑为 8 个 256×4096 chunk，每 chunk 做 256 点 FFT、保留 128 个低频系数、128 点 iFFT，得 8×128×4096=1024×4096 的缩短 Q，注意力矩阵从 2048×2048 缩到 1024×1024（4× 缩减）；FFN/QKV 投影的 W 按 B=32 分 tile 做蝴蝶分解，复杂度 O(D²/B·log B)=O(4096²/32·5)。
