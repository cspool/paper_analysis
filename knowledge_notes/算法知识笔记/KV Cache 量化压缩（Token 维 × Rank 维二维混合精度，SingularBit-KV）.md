## KV Cache 量化压缩（Token 维 × Rank 维二维混合精度，SingularBit-KV）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 量化压缩指在自回归解码过程中把不断增长的 key/value 缓存以低比特表示存储，减少 KV 的 DRAM 容量与每步注意力访存带宽（KV 流量随序列长度 O(N²) 增长，长上下文/推理场景成为主瓶颈）。先前方法分两类：(1) 纯量化——KIVI [11] key 逐 channel、value 逐 token 的非对称 2-bit（key 有通道级 outlier、value 有 token 级模式）、KVQuant [16] pre-RoPE 量化+非均匀量化、ZipCache [33] 按注意力显著性分配 token 级位宽、GEAR [30] 量化+低秩校正+稀疏离群补偿；(2) 纯低秩——PALU [31] 低秩投影缓存中间态、MatryoshkaKV、ReCalKV [32]。SingularBit-KV 的创新：在 token 维按注意力重要性分 5 级位宽（b~b+4），同时在同一 token 内按 K/V 投影权重的奇异值边界做 rank 维混合精度——二维同时压缩，且缓存中间表示 K'=xU_{W_K}、V'=xU_{W_V}（只乘 U 矩阵），attention 前用 V^T 重构。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 每 decode 步的三阶段 pipeline（论文 Fig.6）：①注意力图更新：当前注意力图 $A_t\in\mathbb{R}^{H\times N_t}$ 逐 head max pooling 后归一化得 $\tilde{a}_t$，追加进 recent-k 窗口 $M_t\in\mathbb{R}^{k\times N_t}$（k=128）并逐出最旧；②重要性分数：$\mathcal{I}_i=\max_{j\in[t-k+1,t]}\tilde{a}_j[i]$（取近 k 步 query 方向的最大，因在线量化不可逆、需保守保留峰值注意力）；③精度分配+压缩：token 精度策略按线性递增容量调度 $l_i\cdot 2^{b+i}=m\cdot i+c$（$\sum l_i=1$ 解出区间边界 $s_0..s_5$，重尾分布下高重要度区间获得更多量化容量），把 $\mathcal{I}_i$ 映射到 5 级位宽；rank 精度策略在该 token 最大位宽内按 SingularBit-W 边界逐级降精度。伪代码：
  ```
  # 每 decode 步 t
  a_t = normalize(max_pool(A_t, dim=head))       # 逐 head 最大池化
  M_t = push_evict(M_t, a_t, k=128)
  I_i  = max(M_t[:, i])                          # 每 token 重要性（近 k 步最大）
  b_tok = map_token_precision(I_i, l_i*2^(b+i)=m*i+c)   # 5 级 token 位宽
  Kp = x @ U_WK;  Vp = x @ U_WV                  # 只乘 U，缓存中间表示（无在线 SVD）
  qKV = quantize_rankwise(Kp, Vp, b_tok, rank_boundaries)  # FP16→INTx 逐 rank 降位
  packed = bitpack_no_padding(qKV, b_tok)        # 紧凑打包 + 按位宽路由物理地址
  # attention 时: K=Kp@V_WK^T, V=Vp@V_WV^T 重构（tensor core 上，+5%@ctx64/+2%@ctx2048 延迟）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：算法在 GPU 上用 PyTorch 可实现（注意 K'/V' 缓存 + 每 token 位宽表 + 反量化重构 kernel；KIVI 有开源实现 https://github.com/Zefan-Cai/KIVI），SingularBit 则用专用硬件（SingularBit Compression Engine：precision allocator + FP16-to-INTx 量化器 + 位打包器 + 路由）在线执行，避免 GPU 的格式转换/数据移动开销。使用注意：在线量化的不可逆性决定了重要性打分必须保守（用 max 而非 mean）；base precision b 是压缩-精度权衡旋钮（论文选精度下降 <1% 的工作点）。论文结果：KV2 下 Llama-3-8B-Instruct CoQA 61.5%（FP16 63.5%，仅 -2.0%）/TruthfulQA 59.5%，GSM8K 0.81~0.85（KVQuant/PALU/ZipCache 等 0–30% 崩坏），压缩率 84–86%；LongBench 上 SingularBit-KV 44.4% vs ReCalKV 29.6%。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
