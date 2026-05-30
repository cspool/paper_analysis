## Taylor Feature Map for Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Taylor Feature Map是基于Taylor级数展开近似softmax-exponential的确定性特征映射：exp(x)≈1+x+x²/2!+...+x^k/k!，截断到k阶产生多项式核φ(q)^Tφ(k)=Σ_{m=0}^{k}(q^Tk)^m/m!。区别于Performer的随机傅里叶特征(RBF kernel)，Taylor map是确定性的无随机投影噪声。Based架构(Zhang et al. 2024, Hedgehog & Porcupine)使用2阶Taylor近似：φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2。二阶项产生高维展开d̃≈273(for base dim d=16)，使IO-aware kernel管理warp-register分片的KV-state矩阵关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
exp(α) ≈ 1 + α + α²/2    (2nd-order Taylor at α=0)

// Based feature map (可分离):
φ_1(q) = q / sqrt(sqrt(d))              // 一阶项
φ_2(q) = vec(q ⊗ q) / sqrt(2) / sqrt(d) // 二阶项, d(d+1)/2 dim
φ(q) = concat(1, φ_1(q), φ_2(q))       // 总dim ≈ 1+d+d(d+1)/2

// JRT/PLA使用相同feature map，encoder和decoder各自独立投影矩阵
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Based实现中feature dim d=16, 总展开dim d̃≈273。IO-aware CUDA kernel(ThunderKittens)将KV-state(R^{d×273})分片存储在warp registers中，A0/A1/A2寄存器分别存0/1/2阶项贡献。JRT论文的所有模型(JRT-Prompt, JRT-RNN)均使用Based的Taylor feature map。该映射也可被PLA、causal LA等任何线性注意力变体使用。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
