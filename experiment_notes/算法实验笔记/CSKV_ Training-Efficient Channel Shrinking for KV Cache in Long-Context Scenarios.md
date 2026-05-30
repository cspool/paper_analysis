## CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：基于低秩分解的 KV Cache 通道收缩（Channel Shrinking）。核心为：(1) 对 Key/Value 权重矩阵 W^K, W^V 进行低秩分解 W^K ≈ A^K B^K，其中 A^K ∈ R^{hin×hcomp}，B^K ∈ R^{hcomp×hout}，hcomp < hout，存储中间低维特征 hcomp 作为压缩 KV Cache；(2) 双分支 KV Cache：近期 m 个 token 保留完整精度（SVD 不降维），历史 token 使用压缩表示；(3) ASVD (Activation-aware SVD) 初始化 + 逐层 MSE 重建损失微调，仅需 90 分钟/A100。
  - 实验比较：在 LongEval、LongBench、LVEval 三个长上下文 benchmark 上，对比 CSKV 与 StreamingLLM（token pruning）、H2O（token pruning）、ASVD（channel shrinking）在 50% 和 80% 压缩率下的性能。消融：初始化方法（Random vs SVD vs ASVD）、窗口大小（2-4096）、KV 压缩率分配、4-bit 量化兼容性。

- 硬件平台是什么，配置是什么。
  - 训练：单张 NVIDIA A100-80G GPU；微调一个 7B 模型耗时 90 分钟。
  - 推理评测硬件：论文未明确说明具体 GPU 型号。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LongChat-7B-v1.5-32k（LLaMA 架构）、Mistral-7B-Instruct-v0.2。
  - 微调数据集：scaled-down version of the Pile（HuggingFace: ola13/small-the_pile），epoch=1，batch_size=1，AdamW optimizer，lr=5e-5。
  - Benchmark：LongEval（200/300/400/500 lines 子集，平均长度 4k-10k）、LongBench-E（qasper, hotpotqa, multifieldqa_en, gov_report, triviaqa）、LVEval（16K 子集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wln20/CSKV

**核心张量计算流程：**

```
# ========= 权重低秩分解 =========
# 对每层的 W_K, W_V 做 ASVD 低秩分解
W_K ∈ R^{hin×hout} → A_K ∈ R^{hin×hcomp}, B_K ∈ R^{hcomp×hout}
W_V ∈ R^{hin×hout} → A_V ∈ R^{hin×hcomp}, B_V ∈ R^{hcomp×hout}
# hcomp << hout（压缩率 = (hout - hcomp) / hout）

# ========= Prefilling 阶段 =========
# 输入 X ∈ R^{n×hin}（prompt 有 n 个 token）
K = X @ W_K                         # 完整 Key，用于 attention 计算
K_C = X @ A_K                       # 压缩 Key → 存入 Compressed Key Cache (n, hcomp)
K_local = K[-m:, :]                 # 保留最后 m 个 token 完整精度

# Value 同理

# ========= Decoding 阶段（以第 (n+1) 个 token 为例）=========
# 当前 token x ∈ R^{1×hin}
k = x @ W_K                         # 完整 Key
k_C = x @ A_K                       # 压缩 Key

# 更新缓存（Compressed Key Cache 有 n+1 token，Full Cache 有 m+1 token）
Compressed_Key_Cache.append(k_C)    # → (n+1, hcomp)
Full_Key_Cache.append(k)            # → (m+1, hout)

# 重建完整 Key 矩阵用于 Attention
K_hat_empty = Compressed_Key_Cache[:(n-m), :]  # 旧 token 的压缩特征
K_hat = K_hat_empty @ B_K                       # 低维 → 高维重建
K_final = concat([K_hat, Full_Key_Cache])       # 拼接得到完整 Key

# 保持窗口大小 m：移除 Full Cache 中最旧 token
```

**逐层重建训练流程：**

```
# SVD-based 初始化（ASVD, α=0.5, Absolute Mean Value scaling）
# 从标定数据采样 256 个样本计算缩放矩阵 S
for layer in range(n_layers):
    A_K[layer], B_K[layer] = ASVD_decompose(W_K[layer], calib_data)
    A_V[layer], B_V[layer] = ASVD_decompose(W_V[layer], calib_data)

# 逐层训练
for layer in range(n_layers):
    for X in train_loader:
        K = X @ W_K[layer].T          # 原始 Key 激活
        K_hat = X @ A_K[layer].T @ B_K[layer].T  # 低秩重建 Key
        loss_K = MSELoss(K, K_hat)
        
        V = X @ W_V[layer].T
        V_hat = X @ A_V[layer].T @ B_V[layer].T
        loss_V = MSELoss(V, V_hat)
        
        loss = loss_K + loss_V
        loss.backward()
        optimizer.step()

# 全局损失: L_all = Σ_{j=1}^{n_l} (L_{K,j} + L_{V,j})
```

**量化集成（KIVI 4-bit）：**
- 80% 通道压缩 + 4-bit 量化 = 95% 总压缩率
- PTQ 直接量化崩溃（Avg.Acc 0.00），必须使用 QAT
- QAT 模式：80% 通道压缩 + 4-bit → 95% 总压缩 → Avg.Acc 0.90（vs baseline 0.99）
