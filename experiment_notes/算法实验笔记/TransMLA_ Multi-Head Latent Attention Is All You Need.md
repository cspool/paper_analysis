## TransMLA: Multi-Head Latent Attention Is All You Need

- 属于算法pipeline的实现是什么？实验比较什么？
  TransMLA 是一种将 GQA（Group-Query Attention）预训练模型（LLaMA、Qwen、Gemma、Mistral/Mixtral）无缝转换为 MLA（Multi-Head Latent Attention）模型的框架。核心实现包括三个技术：(1) **RoRoPE**——对合并后的多 head key 向量按 RoPE 频率维度分组，在每个 2D 子空间内用正交矩阵 U_l 旋转 Q 和 K，通过 PCA 将位置信息集中到第一个 attention head 中（等价变换，不改变 attention 输出）；(2) **FreqFold**——利用相邻 RoPE 频率的相似性，将多个频率维度的 key 段拼接后做联合 PCA，使 K_rope 能占用更多维度以保留更丰富的位置信息；(3) **BKV-PCA**——先计算 α = E[||K_NoPE||₂]/E[||V||₂]，将 K_NoPE 缩放 1/α 使 norm 与 V 对齐后，对 [K_NoPE; V] 联合做 PCA 低秩分解，避免 key 主导主成分方向导致 value 信息丢失。转换后的模型兼容 DeepSeek 代码库，可直接使用 DeepSeek 的 Absorb 操作进行推理加速。

  实验比较：(1) SmolLM-1.7B 和 LLaMA-2-7B 在三种 KV cache 压缩率（-68.75%、-81.25%、-87.50%，LLaMA 额外 -92.97%）下与 MHA2MLA 方法在 6 个 benchmark（MMLU、ARC、PIQA、HellaSwag、OpenBookQA、Winogrande）上的准确率对比（Table 1）；(2) 不同 fine-tuning token 量（0 / 300M-1B / 500M-6B）下的性能恢复曲线；(3) LLaMA-3-8B 上 RoRoPE + FreqFold 的 key norm 分布可视化和 RoPE 去除比例 vs log-perplexity 消融实验（Figure 3）；(4) KV balancing 前后的 key/value norm 对比和 weight-based vs activation-based PCA 消融（Figure 4）；(5) 三款消费级 GPU（165.2 TFLOPS 24GB、312 TFLOPS 40GB、320 TFLOPS 64GB）上 vLLM 推理吞吐量对比（Table 4），输入/输出等长设置，1K-32K 总 context length。

- 硬件平台是什么，配置是什么。
  **训练**：8-GPU 机器，每 GPU 40GB 显存，312 TFLOPS FP16 算力。**推理 benchmark**：三款消费级 AI 加速器——165.2 TFLOPS / 24GB、312 TFLOPS / 40GB、320 TFLOPS / 64GB。使用 vLLM 推理框架。

- 模型是什么。数据集和bench分别是什么。
  **模型**：SmolLM-1.7B（1T tokens 预训练）、LLaMA-2-7B（2T tokens 预训练）、LLaMA-3-8B（仅用于分析实验）。**训练数据集**：SmolLM pretraining corpus，组成——FineWeb-Edu-Dedup (70%)、Cosmopedia-v2 (15%)、Python-Edu (6%)、Open-Web-Math (8%)、StackOverflow (1%)。**分析/校准数据集**：WikiText-2（用于 RoRoPE PCA 校准和 perplexity 评估）。**Benchmark**：MMLU、ARC (easy + challenge)、PIQA、HellaSwag、OpenBookQA、Winogrande（6 个常识推理任务）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **开源**：https://github.com/MuLabPKU/TransMLA（NeurIPS 2025 Spotlight）。支持 LLaMA-2/3、Qwen2.5、Gemma2、Mistral、Mixtral 模型转换，兼容 DeepSeek 代码库（vLLM、SGlang、FlashMLA）。

  **算法 Pipeline 伪代码（以 LLaMA-2-7B GQA → MLA 转换为例）**：

  ```
  Input: GQA model with h query heads, g KV groups, d per-head dim, D hidden dim
         Calibration dataset (WikiText-2 subset)

  Step 1 — Merge all KV heads:
    W^{DKV} = [W^K; W^V] ∈ R^{2gd × D}   // 合并所有 KV head 投影矩阵
    初始化 W_i^{UK} ∈ R^{d × gd}：对 head i 所属 group j，W_i^{UK}[:, jd:(j+1)d] = I_d
    初始化 W_i^{UV} 同理（identity selector）
    // 此时 K cache = c_t^{KV} = W^{DKV} x_t ∈ R^{2gd}，与原始 GQA 相同

  Step 2 — RoRoPE: 解耦 RoPE 位置信息:
    For each RoPE frequency l ∈ {1, ..., d/2}:
      // 收集所有 g 个 head 中第 l 个 RoPE 子空间
      K_x_real = concat across heads: k_j^{[2l-1::d]} for j=0..g-1  // g-dim vectors
      K_y_imag = concat across heads: k_j^{[2l::d]} for j=0..g-1    // g-dim vectors
      // 构建联合协方差矩阵
      Σ_l = K_x_real^T K_x_real + K_y_imag^T K_y_imag  ∈ R^{g×g}
      // 特征分解得到最优正交旋转矩阵 U_l
      U_l = eigendecomposition(Σ_l).eigenvectors  // 按特征值降序排列
      // 旋转 W^K 和 W_i^{UK}（等价变换，不改变 attention 输出）
      W^K[l-related dims] ← U_l applied to corresponding dimensions
      W_i^{UK}[l-related dims] ← U_l applied to corresponding dimensions
    // 旋转后第一个 head 的 key 集中了主要位置分量 → K_rope
    // 其余 head 的 key 位置信息可忽略 → K_nope，移除其 RoPE

  Step 3 — FreqFold（可选）:
    // 将频率相近的 M 个 RoPE 频率索引合并
    For each merged group of M frequencies:
      Concat the M×2g dimensional segments across heads and frequencies
      Perform joint PCA on the concatenated vectors
      Keep top M principal components for K_rope
    // 使 K_rope 占用 M×d（而非仅 d）维度，保留更多位置信息

  Step 4 — BKV-PCA: 联合低秩压缩 K_nope 和 V:
    α = E[||W_NoPE^{DK} x_t||₂] / E[||W^{DV} x_t||₂]  // norm 平衡因子
    // 缩放后拼接
    c_NoPE,t = [1/α · W_NoPE^{DK} x_t; W^{DV} x_t] ∈ R^{(2g-1)d}
    // 在 calibration set 上 PCA
    R_KV = top-r_kv eigenvectors of Cov(c_NoPE)  ∈ R^{(2g-1)d × r_kv}
    // 低秩分解
    W^{DKV'} = R_KV^T · [W_NoPE^{DK}; W^{DV}]  ∈ R^{r_kv × D}
    W^{UKV'} = [W_NoPE^{UK} 0; 0 W^{UV}] · R_KV  ∈ R^{2hd × r_kv}
    // 推理时仅缓存 c_t^{KV'} = W^{DKV'} x_t ∈ R^{r_kv}（压缩后 KV cache）

  Step 5 — Fine-tuning（可选，恢复性能）:
    batch_size=64/256, lr=1e-4 或 2e-5, warmup 0-3%, constant/cosine scheduler
    seq_len=2048(SmolLM) 或 4096(LLaMA), tokens=300M-6B
  ```

  **推理时 Absorb 操作**（MLA 推理范式，Equation 10）：
  ```
  // 将 W_i^{UK} 吸收到 query projection 中，避免先投影再计算
  q̂_{t,i} = [(W_i^{UK})^T q_{t,i}^C; q_{t,i}^R]  // 变换后的 query
  k̂_t = [c_t^{KV}; k_t^R]                          // 共享 latent key
  // 所有 head 共享一个 KV head（类似 MQA），仅需缓存 c_t^{KV}
  ô_{t,i} = Σ_j softmax(q̂_{t,i}^T k̂_j / √(d+d^R)) · c_j^{KV}
  y_t = W^O [W_1^{UV} ô_{t,1}; ...; W_h^{UV} ô_{t,h}]
  ```
