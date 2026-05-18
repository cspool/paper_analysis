## Mamba (State Space Model) for Time Series in LLM Systems（面向LLM系统的Mamba时序模型）

术语是什么？通过联网搜索让回答具体和精准。

Mamba是由Gu和Dao(2024)提出的基于State Space Model (SSM)的序列建模架构，通过选择性状态空间（selective state space）机制突破传统SSM的线性时不变限制。核心公式：h_t = exp(Δt·A)·h_{t-1} + (Δt·A)^{-1}·(exp(Δt·A) - I)·Δt·B·x_t，其中Δt = softplus(W_Δ·x_t + b_Δ)是输入依赖的选择性更新门，A和B是连续时间系统参数。相比RNN，Mamba训练时支持并行扫描（parallel scan），推理时recurrent逐步计算；相比Transformer，Mamba具有线性计算复杂度（而非二次attention），在长序列上参数效率更高。

在TailorLLM的AdapterMgr中：Mamba被用作时间序列特征提取器，从用户历史访问序列中提取时序模式。单层Mamba Block以滑动窗口H=100的访问序列为输入，通过隐藏状态h_t ∈ R^{128}递推式地编码用户行为的长程依赖。训练时使用Parallel模式批量处理序列，推理时使用Recurrent模式逐步更新状态。Mamba在这里替代了传统RNN/LSTM（无法并行训练）和CNN（局部感受野无法捕捉长程依赖）用于cache替换决策的时序编码。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mamba作为AdapterMgr时序特征提取器的计算流程：
```
输入: 用户历史访问序列 x_{t-H+1:t} ∈ R^{H×d} (H=100, d=128)
输出: 时序特征 h_t ∈ R^d

// 训练时 Parallel Mode:
1: Δ_{t-H+1:t} = softplus(W_Δ @ x_{t-H+1:t} + b_Δ)  // 选择性gate
2: Ā_i = exp(Δ_i · A)                                // 离散化状态转移
3: B̄_i = (Δ_i · A)^{-1} · (exp(Δ_i · A) - I) · Δ_i · B  // 离散化输入投影
4: h_{t-H+1:t} = parallel_scan(Ā, B̄, x)               // 并行扫描

// 推理时 Recurrent Mode:
5: h_t = Ā_t · h_{t-1} + B̄_t · x_t                    // 逐步递推
```

在AdapterMgr pipeline中的位置：
```
6: E(X) = W_x · X + positional_encoding    // 用户序列embedding (H=100, d=128)
7: E(L) = W_l · L                           // cache状态embedding (w=5, d=128)
8: h_t = Mamba(E(X))                        // Mamba提取时序特征 [H×d] → [d]
9: F_fused = Concat(W_f · E(L), h_t)       // 双模态融合
10: F_out = LayerNorm(F_fused)
11: π̂ = Softmax(MLP(F_out))                 // 输出替换策略向量
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Mamba开源实现：https://github.com/state-spaces/mamba (CUDA官方kernel + PyTorch接口)。通过`Mamba(d_model, d_state, d_conv)`类实例化，支持`mamba.forward(x)`。在TailorLLM的AdapterMgr中，使用单层Mamba Block（d_model=128）作为轻量级时序编码器，训练时输入shape为[batch, H, d]的可变长序列。选择Mamba而非Transformer/RRNA的原因：(1) 并行训练效率高；(2) 可建模全局时序依赖；(3) 参数效率——单层即可达到或超过Transformer多层性能。论文未明确说明Mamba block的具体超参数（d_state, d_conv）。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

