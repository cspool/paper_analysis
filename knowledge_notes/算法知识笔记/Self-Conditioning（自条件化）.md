## Self-Conditioning（自条件化）

术语是什么？通过联网搜索让回答具体和精准。
Self-Conditioning是由Chen et al. (2023, "Analog Bits")提出的扩散模型条件化技术。核心思想：将模型在上一去噪步的输出（intermediate prediction x̂'或ŝ'）作为当前步的额外条件输入，使模型利用自身已有预测改进当前估计。在ELF中的具体形式：在Flow Matching的denoising branch训练时，50%概率先用当前状态z_t做一次forward pass得到intermediate prediction x̂'，然后将[z_t, x̂']（channel-wise concatenation）通过线性投影层映射回原始维度作为网络输入进行第二次forward pass；另50%概率使用all-zero作为self-conditioning（即无条件）。推理时，self-conditioning使用上一步的prediction x̂_{t+dt}，无需额外forward pass。Self-Conditioning已在多个DLMs中成为标准组件（SED、CDCD、LD4LG、TESS、TEncDM等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ELF中的Self-Conditioning训练与推理流程：
```
// 训练期（Alg.3）
1: z_t = t·x + (1-t)·ε              // 当前noisy state
2: z_no_sc = proj(concat([z_t, 0]))  // 无self-cond: concat zeros
3: x_no_sc = net(z_no_sc, t)         // 第一次forward pass
4: z_sc = proj(concat([z_t, stopgrad(x_no_sc)]))  // 有self-cond: concat前次预测
5: x_sc = net(z_sc, t)               // 第二次forward pass
6: mask ~ Bernoulli(0.5)            // 50%概率使用self-cond
7: x_pred = mask ? x_sc : x_no_sc
8: v_pred = (x_pred - z_t) / (1-t)
9: loss = MSE(v_pred, v_target)

// 推理期（Alg.5）
1: x_pred = 0                         // 初始化为零
2: for each time step t:
3:     z_sc = proj(concat([z_t, x_pred]))  // 用上一步预测
4:     x_pred = net(z_sc, t)           // 单次forward pass（无额外开销）
5:     v = (x_pred - z_t) / (1-t)
6:     z_{t+dt} = z_t + dt·v
```

关键设计选择：(1) Gradient stop：intermediate prediction x̂'的梯度被stop，避免通过self-conditioning形成循环梯度；(2) 投影层：channel维度从2d→d（d为embedding dim），使用线性层；(3) 训练期概率：50%是常用选择，平衡conditional和unconditional mode；(4) 推理零开销：推理时无需额外forward pass（self-cond来自上一步已有预测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Self-Conditioning在ELF中有双重作用：(1) 直接改善去噪质量——模型在已有预测的引导下做出更consistent的估计；(2) 作为CFG的conditioning signal——ELF的training-time CFG使用self-conditioning prediction作为"条件"，无需外部class label或text prompt即可实现CFG。实现上，Self-Conditioning通过channel-wise concatenation + linear projection融入网络输入，与in-context conditioning（prepend control tokens）协同工作。该技术对训练overhead极小（仅增加一次额外的forward pass和stop-gradient操作，且仅50%概率触发），推理时完全零开销。在ELF的ablation中（Appendix），无self-conditioning的CFG会显著降低生成质量。

涉及论文标题：
- ELF: Embedded Language Flows

---

