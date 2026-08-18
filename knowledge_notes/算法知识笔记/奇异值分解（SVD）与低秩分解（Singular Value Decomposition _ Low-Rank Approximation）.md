## 奇异值分解（SVD）与低秩分解（Singular Value Decomposition / Low-Rank Approximation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 奇异值分解（SVD）把任意实矩阵 $\mathbf{W}\in\mathbb{R}^{m\times n}$ 分解为 $\mathbf{W}=\mathbf{U}\mathbf{S}\mathbf{V}^T$：$\mathbf{U}$（$m\times r$，左奇异向量、列空间正交基）、$\mathbf{S}$（对角矩阵，奇异值 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r\ge0$）、$\mathbf{V}^T$（$r\times n$，右奇异向量、行空间正交基）。低秩近似即保留前 $k$ 大奇异值对应的分量、截断尾部，$k$ 值越接近满秩误差越小；Eckart–Young 定理保证截断 SVD 在 Frobenius 范数下是最优低秩逼近。关键性质：LLM 权重矩阵的奇异值近似指数衰减（SingularBit 论文 Fig.4 实测 Llama2-7B 逐层），即少数主导 rank 分量承载大部分信号能量、长尾贡献极小——这使"按 rank 重要性分配资源"成为可行的压缩信号。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SVD 在 LLM 压缩 pipeline 中的两种用法：(1) 低秩截断压缩（SVD-LLM [40] 用截断感知数据白化对齐奇异值与压缩损失、ASVD [39] 分解前做激活感知变换抑制 outlier、SliceGPT 正交变换后删行删列）：$W\approx\hat{U}\cdot S_{[:k]}\cdot\hat{V}^T$，直接减少参数与 FLOPs，但压缩率不如量化且 rank 截断即"0-bit 分配"会整体丢失信息；(2) 作为精度分配信号（SingularBit 用法，见下一个术语）。SingularBit 的分解式（论文 Eq.1）：$\mathbf{W}^T=\sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^T$，累计尾部占比 $C_i=\frac{\sum_{k=i}^{r-1}\sigma_k}{\sum_{k=0}^{r-1}\sigma_k}$ 量化每 rank 的信息密度。伪代码：
  ```
  U, S, VT = svd(W.T)              # 一次离线分解，得 U∈R^{ich×r}, S=diag(σ), VT∈R^{r×och}
  # 后续：按 σ 分精度区域 / 或截断保留前 k 个分量
  W_approx = U[:, :k] @ diag(S[:k]) @ VT[:k, :]     # 低秩截断（SVD-LLM/ASVD 式）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：数值上用 LAPACK/CPU-GPU 的 gesvd/gesdd（或 PyTorch torch.linalg.svd、JAX jnp.linalg.svd）求稠密矩阵 SVD；LLM 压缩实践中对每个线性层权重独立分解（O(L) 次小规模 SVD，可离线并行），calibration 数据仅用于量化阶段的 Hessian 而非分解。使用注意：SVD 在 Frobenius 范数意义下最优但不对齐实际激活分布，故 ASVD/SVD-LLM 引入激活感知/白化预处理；SingularBit 不做截断而是把奇异值作为 rank 级混合精度分配依据（保留全 rank、只降位宽），并用 GPTQ 误差反馈框架量化分解后的 U/V^T 分量。论文数据：LLM 权重奇异值近似指数衰减，SingularBit-W 在 2-bit 平均精度下 LLaMA-7B Wiki 困惑度 7.56（RTN/GPTQ/AWQ 为 1.9e3/44.01/2.6e5），证明"全保留+按重要性降位"优于"截断"或"均匀量化"。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
