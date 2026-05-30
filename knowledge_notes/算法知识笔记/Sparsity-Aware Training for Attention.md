## Sparsity-Aware Training for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity-Aware Training for Attention是一种fine-tuning策略，在训练/微调过程中应用稀疏注意力模式（而非dense attention），使模型学会将重要信息集中在被保留的attention block中，从而在推理时应用稀疏attention时更robust。BLASST的实现方式：在fine-tuning的forward pass中应用BLASST的threshold-based block skipping（根据m̃-m < ln(λ)条件跳过block），backward pass中被跳过的block自然不接收梯度（因其forward未计算），无需auxiliary loss或architecture change。模型在训练中学会将关键attention信息集中在高score block中，使得推理时即使aggressive pruning（70-90% sparsity）也能维持准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

训练pipeline与标准fine-tuning的唯一区别在于attention forward pass：

```
# 标准fine-tuning forward:
O = FlashAttention(Q, K, V)           # dense attention

# Sparsity-aware training forward:
O = BLASST(Q, K, V, λ)               # sparse attention with threshold λ
# Backward: 自动通过O的computational graph反向传播
# 被跳过block无compute node → 自然无梯度回传
```

梯度仅流过实际参与计算的attention block。采用ProLong论文中的curriculum training策略逐步增加sparsity。效果（BLASST论文Figure 6）：在RULER benchmark上，sparsity-aware trained模型在50-75% sparsity区间的accuracy退化降低至training-free的1/1.7。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sparsity-aware training的实现需要：(1) 可微的sparse attention kernel（如BLASST，forward中有skip decision但仅对non-skipped block构造compute graph）；(2) 稀疏模式在训练中保持固定（BLASST使用固定threshold λ，而非动态变化的sparsity pattern）；(3) curriculum scheduling——gradually降低threshold以增加sparsity。与pruning-aware training、dropout-based sparsity training等方法属于同一思路的不同实现。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

DSV的两阶段训练是另一种sparsity-aware training范式，与BLASST的关键区别：(1) DSV使用可训练的sparsity predictor（低秩矩阵W_Q^lr, W_K^lr）来预测critical KV pairs，而非使用固定threshold；(2) DSV在Stage 1训练predictor、Stage 2激活稀疏计算，BLASST在fine-tuning中直接应用threshold；(3) DSV的稀疏模式是per-query动态的（每个query有不同critical KV set），BLASST是per-block固定的。DSV的predictor训练loss：L_approx = 0.95·CosLoss(QK_lr, QK_main) + 0.05·NormLoss(QK_lr, QK_main)，predictor的计算图从主模型detached。
