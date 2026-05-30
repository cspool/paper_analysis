## Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 MoDiff（Modulated Diffusion）框架，通过 modulated quantization（调制量化）和 error-compensated modulation（误差补偿调制）两个创新机制加速扩散模型的生成过程。核心思想是对相邻时间步之间的激活差值（temporal difference a_t − a_{t+1}）而非原始激活值进行量化——因为差值范围远小于原始激活（约10×），从而可以用更低 bit-width 的量化达到同等精度。实验将 MoDiff 应用于 Q-Diffusion（Q-Diff）和动态逐通道量化（LCQ）两种 baseline PTQ 方法，比较在 CIFAR-10、LSUN-Churches、LSUN-Bedrooms 上不同激活 bit-width（8/6/4/3/2-bit）下的生成质量（IS、FID、sFID）和理论计算量（GBops）。同时在 Stable Diffusion v1.4（MS-COCO）、DiT-XL/2（ImageNet）、SDXL-Turbo（few-step）上验证泛化性。还包含 DDPM、DPM-Solver、PLMS 等不同 sampler 的兼容性实验。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体硬件平台。效率评估使用 DeepSpeed 工具统计每个去噪步骤的单图理论二元运算次数（BOPs/GBops），不报告实际 wall-clock time 加速。论文明确说明硬件实现是未来工作方向。

- 模型是什么。数据集和bench分别是什么。
  模型：DDIM（CIFAR-10，100步）、Latent Diffusion Model LDM-4（LSUN-Bedrooms，500步）和 LDM-8（LSUN-Churches，200步）、Stable Diffusion v1.4（MS-COCO 2014，50步，DPM solver）、DiT-XL/2（ImageNet 256×256，50步，Transformer架构）、SDXL-Turbo（few-step，2/4/8步，结合 MixDQ）。数据集：CIFAR-10（32×32）、LSUN-Bedrooms（256×256）、LSUN-Church-Outdoor（256×256）、MS-COCO 2014、ImageNet 256×256。评估指标：IS、FID、sFID（基于50k生成图像），Precision/Recall，以及 GBops。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/WeizhiGao/MoDiff

  算法 pipeline 伪代码（以单层线性算子 A^(l) 为例）：

  ```
  # === 初始化 (t=T, 第一步: Warm-up with full precision) ===
  a_hat[T] = Q(a_T)                          # 量化原始激活 (Eq.8)
  o_hat[T] = A(a_hat[T])                     # 量化后计算输出 (Eq.9)

  # === 迭代 (t = T-1 到 1): Error-Compensated Modulation ===
  for t in range(T-1, 0, -1):
      # Step 1: 误差补偿激活重建 (Eq.13)
      a_hat[t] = Q(a_t - a_hat[t+1]) + a_hat[t+1]

      # Step 2: 调制量化计算 (Eq.14)
      o_hat[t] = A(Q(a_t - a_hat[t+1])) + o_hat[t+1]
  ```

  关键张量计算与实现细节：
  - 标准 PTQ：各时间步独立量化原始激活 a_t → Q(a_t) → A(Q(a_t))，误差独立、各步不共享信息
  - MoDiff 调制计算：利用线性算子 A 的线性性，将 o_t = A(a_t) 等价重写为 o_t = A(a_t − a_{t+1}) + o_{t+1}
  - 调制量化：对差值 Δ_t = a_t − a_{t+1} 量化，其范围约 10× 小于原始激活，同等 bit-width 下量化误差显著降低
  - 误差补偿：用 â_{t+1} 替代 a_{t+1}，使上一步量化误差 e_{t+1} = a_{t+1} − â_{t+1} 被自动注入到下一步差值计算中补偿。Theorem 4.4 证明标准调制误差累积呈 2^{T−k} 指数增长，误差补偿调制呈 (2c)^{T−k} (c<1/2) 指数衰减
  - 与缓存方法关系：当激活差值范围低于阈值时，Q 可分配 0-bit（即跳过计算），此时 MoDiff 退化为 DeepCache 等缓存方法的超集
  - 实现要点：(1) 移除所有应用 MoDiff 的层的 bias 项；(2) 第一步 Warm-up 使用全精度激活；(3) 逐层重构校准数据集；(4) 逐层而非逐 block 重构以保证稳定性
  - 额外内存开销：CIFAR-10 单图生成时仅为 35–39 MB（含中间变量 â_t 和 ô_t），batch size 增大后仍可控
