## Layerwise Recurrent Router for Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Layerwise Recurrent Router for Mixture-of-Experts (RMoE)，在 MoE 路由过程中引入轻量级 Gated Recurrent Unit (GRU)，逐层建立跨层路由决策的依赖关系。具体实现：在第 i 层，先用逐层独立的投影层（Proj_i）将 token hidden state x_i ∈ R^h 降维到 GRU 状态维度 x_i' ∈ R^p（p=128），然后与上一层 GRU 输出 h_{i-1} 拼接送入跨层共享的 GRU 单元，得到当前层 GRU 输出 h_i，最后将 h_i 输入该层 router (linear layer + softmax + top-k) 选择 expert 并执行标准 MoE 计算。
  - 实验比较：在 language modeling (Enwiki8 BPC, WikiText-103 PPL)、大规模 pre-training + SFT (0.91B, 15B 模型)、多 benchmark 评估 (ARC-Easy, Hellaswag, PIQA, SciQ, LAMBADA, MMLU, GSM8K, HumanEval) 下，与 SMoE (标准 linear router)、HyperMoE、SMoE-MLP、RandomMoE、CosineMoE、XMoE 等 baseline 比较。还与 XMoE 结合验证正交兼容性。消融实验拆解 layerwise recurrence、Recurrent Gradient、层投影器、GRU vs RNN vs LSTM 等组件贡献。

- 硬件平台是什么，配置是什么。
  - 8-layer 小模型 (hidden=352, 16 experts top-2)：1 张 NVIDIA A100 GPU，约 21 小时。
  - 0.91B 模型 (24-layer, hidden=1280, 16 experts top-4, fine-grained MoE)：8 张 NVIDIA A100 GPU，约 5 天 pre-training + 2 小时 SFT。
  - 15B 模型 (activate 2.7B)：使用 Megablocks 框架，论文未明确说明 GPU 数量，据训练规模推测为多卡 A100 集群。

- 模型是什么。数据集和bench分别是什么。
  - 模型：decoder-only transformer，小模型 8 层 hidden=352，中模型 24 层 hidden=1280 (Llama-style, RoPE + SwiGLU + RMSNorm)，大模型 15B total/2.7B activated (DeepSeek-MoE style, fine-grained + shared experts)。
  - 数据集：Enwiki8 (character-level LM, BPC)，WikiText-103 (word-level LM, PPL)，大规模 pre-training 使用多语言语料 (Wikipedia + 金融 + 法律文本, 40B/120B/400B tokens)，SFT 使用 Alpaca (52K instruction-response pairs)。
  - Benchmark：ARC-Easy (acc), Hellaswag (acc_norm), PIQA (acc_norm), SciQ (acc), LAMBADA (acc), MMLU (acc), GSM8K (acc), HumanEval (pass@k)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/qiuzh20/RMoE
  - 算法 pipeline 伪代码：

```
# 初始化：跨层共享的 GRU 单元 + 每层独立的 Proj_i 和 Router G_i
h_0 = zeros(p)  # 初始 GRU 状态，维度 p=128

for i in range(num_layers):
    # Step 1: 逐层投影降维
    x_i_prime = Proj_i(x_i)  # x_i ∈ R^h → x_i_prime ∈ R^p

    # Step 2: GRU 跨层循环（Eq.5）
    s_i = sigmoid(W_s @ x_i_prime + U_s @ h_{i-1})   # reset gate
    z_i = sigmoid(W_z @ x_i_prime + U_z @ h_{i-1})   # update gate
    h_tilde = tanh(W_h @ x_i_prime + s_i ⊙ (W_h @ h_{i-1}))
    h_i = (1 - z_i) ⊙ h_tilde + z_i ⊙ h_{i-1}

    # Step 3: 基于 GRU 输出的 MoE routing（Eq.6）
    score_i = softmax(h_i @ G_i)         # G_i ∈ R^(p, N), N 个 experts
    topk_idx, topk_val = topk(score_i, k)

    # Step 4: 稀疏 MoE 计算
    y_i = sum_{n in topk_idx} topk_val[n] * Expert_n(x_i)
```

  - 关键设计：(a) 每层使用独立 Proj_i 而非共享投影器，因为不同层的 hidden state norm/分布差异大；(b) 跨层共享 GRU 单元以引入跨层路由信息；(c) GRU 额外提供 Recurrent Gradient 路径，优化 router 训练；(d) 该设计正交于现有 MoE 方法（如 XMoE, DeepSeekMoE），可无缝组合。
