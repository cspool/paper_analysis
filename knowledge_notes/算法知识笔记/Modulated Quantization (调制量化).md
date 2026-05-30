## Modulated Quantization (调制量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modulated Quantization（调制量化）是 MoDiff 论文提出的核心量化策略。不同于标准 PTQ 直接量化每步的原始激活值 a_t，调制量化利用扩散模型相邻时间步之间的时序相似性，将量化目标从原始激活转换为相邻时间步的差值 a_t − a_{t+1}。其数学基础是利用线性算子 A（如全连接层、卷积层）的线性性：o_t = A(a_t) = A(a_t − a_{t+1}) + o_{t+1}。因此只需量化差值 Δ_t = a_t − a_{t+1}，计算 A(Q(Δ_t)) 后加上缓存的上一时间步输出 o_{t+1} 即可得到当前步输出的近似。由于差值范围通常比原始激活小约 10×（见图1b 橙色 vs 蓝色分布），同等 bit-width 下量化误差大幅降低。当差值范围低于阈值时，Q 可分配 0-bit（即跳过该步计算），此时调制量化退化为缓存方法（如 DeepCache）的超集。调制量化与量化方法无关（agnostic to quantizer），可应用于任意 PTQ 方法（Q-Diffusion、LCQ、LTQ 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在扩散模型去噪 U-Net 中，调制量化的算法流程为（以单层线性算子 A 为例）：

```
# === 初始化 (t=T) ===
a_hat[T] = Q(a_T)           # 第一步对原始激活量化
o_hat[T] = A(a_hat[T])      # 第一步全量化计算

# === 迭代 (t = T-1 到 1) ===
for t in range(T-1, 0, -1):
    delta = a_t - a_{t+1}           # 计算时序差值
    delta_q = Q(delta)              # 量化差值（低 bit）
    o_hat[t] = A(delta_q) + o_hat[t+1]  # 增量计算 + 缓存输出
```

关键性质：
- 差值 a_t − a_{t+1} 的量化误差由 Theorem 4.3 控制：||x − Q(x)||² ≤ (max(x)−min(x))²d/(2^b−1)²。由于差值范围约 10× 小于原始激活，等 bit 下误差约 100× 更小，或可用低 3-4 bit 达到同等误差。
- 该方法为每个线性层独立执行，不改变层间数据流或模型架构。
- 无需重新训练（training-free），属于 PTQ 范畴。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. 仅适用于线性算子（Linear、Conv2d），因其线性性 o_t = A(a_t − a_{t+1}) + o_{t+1} 严格成立。
2. 需移除应用 MoDiff 的层的 bias 项，避免 bias 在跨步迭代中重复累积。
3. 第一步 (t=T) 使用 Warm-up（全精度或高精度激活），为后续差值计算提供高质量基准。
4. 量化器 Q 可选用任意 PTQ 方法（min-max dynamic、Q-Diffusion 校准量化、tensor-wise、channel-wise 等）。
5. 代码开源：https://github.com/WeizhiGao/MoDiff，基于 Q-Diffusion 和 BRECQ（PyTorch）代码库构建。
6. 关键结果：CIFAR-10 W8A3 下 LCQ+MoDiff FID=4.14（vs FP=4.24），计算量从 1636 GBops 降至 154 GBops（10× 节省）。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
