## Draft-Free Parallel Decoding (PD)

术语是什么？
Draft-Free Parallel Decoding (PD) 是 VAR-Turbo 提出的无需 draft model 的并行视觉 token 解码算法。核心 insight：图像 visual token 的空间相关性远高于语言 token（entropy 分析表明图像 token 更冗余），因此可直接利用生成模型自身的置信度在每轮选择多个高置信 token 同时解码，无需依赖额外的 draft model（如 speculative decoding 中的小模型）。PD 通过 Gumbel sampling 对所有 masked 位置预测 token 和概率，按 schedule r(t) 确定每轮释放的 token 数 K=N×(1-r(t))，再经 TopK 选择最高置信度的 K 个 token 并行替换。

从算法pipeline角度拆解术语：
PD 每轮推理流程：
```
输入：token sequence V_t, mask array M (True=masked), schedule r(t)
1. Transformer forward：对所有当前位置输出 logits
2. Gumbel sampling：对每个 masked 位置采样 predicted token 和 confidence score
3. Mask-out：已 unmasked 位置（M[i]=False）的 confidence 设为 -inf
4. Compute K = N * (1 - r(t))  // 本轮释放的 token 数
5. TopK selection：选 confidence 最高的 K 个位置
6. Token replacement：K 个位置的 token 更新为 predicted token，mask 置 False
7. 其余 token 保持 masked，进入下一轮
```
与 speculative decoding 的关键区别：PD 不依赖 draft model（无额外模型开销），且每轮可并行释放多个 token（最高 64），远高于语言 speculative decoding 的 2-3 token/step。PD 需 PD-aware training 选择 sampling temperature、masking ratio r(t)、guidance scale 等超参数。

术语一般如何实现？如何使用？
PD 将 VAR 的串行步数从 256（256×256）降至 8-32 步（减少 >80%），在 VAR-Turbo-Balance 模式下 256×256 仅需 8 步。实现需配合 TopK 排序硬件（如 Radix Sort Core）加速大 K TopK 选择（N=4096 时 K 可达 1936）。PD 与 TA、DB 叠加使用，从跨迭代和迭代内两个维度联合加速。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

