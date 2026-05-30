## Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

- baseline方法是什么？
  Baseline 是**标准 PTQ 方法（Q-Diffusion / LCQ）直接量化扩散模型的原始激活值**。各时间步 t 独立进行：加载激活 a_t → 计算量化参数 s, z（min-max 动态计算或通过校准集 MSE 优化）→ 量化 a_t 到低 bit integer → 反量化后送入线性层 A 计算 → 输出 o_t = A(Q(a_t))。每步独立、无跨时间步信息共享。
  
  Baseline 全栈执行例子（以 DDIM + LCQ, CIFAR-10, 100步去噪为例）：
  - 算法层：对去噪 U-Net 中每个线性层/卷积层，逐时间步独立：a_t → 逐通道 min-max scaling → clamp+round → b-bit int → dequantize → A(a_hat_t) → 传递到下一层。各时间步量化参数 s_t, z_t 独立于其他步，无跨步信息或误差补偿。
  - 系统框架层：基于 PyTorch + Q-Diffusion 代码库或 BRECQ 框架，校准数据逐时间步采样。推理时 fake-quantization 模拟量化推理。
  - 编译框架/kernel调度/硬件架构：论文未明确说明（标准 PyTorch CUDA kernel 推理，无硬件加速实现）。
  
  Baseline 核心缺陷（由论文 preliminary study 揭示）：
  1. **激活范围跨时间步波动大**：不同时间步的激活值范围差异显著（图1b 蓝色 violin plot），导致单一量化参数难以覆盖所有时间步，造成严重 clipping/rounding 误差。
  2. **激活分布含大量 outlier**：每个时间步内激活呈长尾分布，大量 outlier 撑大量化步长 s，使得正常值被粗粒度量级覆盖，量化误差大。
  3. **低比特（<6-bit）下质量崩溃**：8-bit activation 是现有 PTQ 方法的安全下限，降至 6-bit 时 FID/sFID 显著退化（如 Q-Diff 8/4 bit CIFAR-10 sFID 从 4.49 → 100.37），4-bit 以下基本不可用。
  4. **缓存方法（如 DeepCache）存在误差累积**：重用历史计算结果跳过某些时间步，但 reuse 误差随步数累积（图1a），最终步误差可达 40%，需 heuristic 手动调 reuse schedule。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoDiff（Modulated Diffusion）**，通过 modulated quantization + error compensation 两大核心机制解决 baseline 缺陷：
  
  **(1) 调制量化（Modulated Quantization）解决缺陷 1-3**
  不直接量化原始激活 a_t，而是量化相邻时间步的差值 a_t − a_{t+1}。利用线性算子 A 的线性性，将计算等价重写为：
  o_t = A(a_t) = A(a_t − a_{t+1}) + o_{t+1}
  然后对差值进行量化：Q(a_t − a_{t+1}) → A(Q(a_t − a_{t+1})) + o_{t+1}
  
  关键洞察（图1b 橙色 vs 蓝色对比）：
  - 原始激活范围大、波动大、含 outlier → 量化误差大
  - 差值范围约 10× 更小、跨步一致性好、分布集中 → 同等 bit-width 量化误差大幅降低
  - Theorem 4.3 量化误差 bound：||x − Q(x)||² ≤ (max(x)−min(x))²d/(2^b−1)²，差值范围缩小直接降低误差 bound
  - 效果：使 PTQ 激活位宽从 8-bit 推至 3-bit 仍无损（CIFAR-10 LCQ+MoDiff W8A3 FID=4.14 vs FP=4.24）

  **(2) 误差补偿调制（Error-Compensated Modulation）解决缺陷 1, 4**
  标准调制直接用原始 a_{t+1}：o_t = A(Q(a_t − a_{t+1})) + o_{t+1}
  问题：量化误差 (a_{t+1} − Q(a_{t+1})) 在每步累积且被缓存传递。
  
  MoDiff 误差补偿：用 â_{t+1} = Q(a_{t+1} − â_{t+2}) + â_{t+2} 替代 a_{t+1}，使上一步量化误差被显式纳入下一步的差值计算：
  â_t = Q(a_t − â_{t+1}) + â_{t+1}
  ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}
  
  重写后等价于：ô_t = A(Q(a_t − a_{t+1} + e_{t+1})) + o_{t+1} − A(e_{t+1})
  即上步误差 e_{t+1} 在下一步被减去 A(e_{t+1})、同时注入 Q 的输入中，实现自动抵消。
  
  Theorem 4.4 理论保证：标准调制误差 O(2^{T−k}) 指数增长，误差补偿调制误差 O((2c)^{T−k}) (c<1/2) 指数衰减。

  全栈执行例子对比（DDIM + LCQ+MoDiff, CIFAR-10, 100步）：
  - 算法层：不再每步独立量化，而是跨步耦合：
    (1) t=T (warm-up)：â_T = Q(a_T), ô_T = A(â_T)
    (2) t=T−1→1：â_t = Q(a_t − â_{t+1}) + â_{t+1}, ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}
    每层独立执行此流程。解耦后仅需缓存 â_t 和 ô_t（额外内存约 3-4 MB per layer）。
  - 系统框架层：基于 Q-Diffusion + BRECQ 代码库，MoDiff 作为 plugin 无侵入集成。校准数据集重构为基于 MoDiff pipeline 的输入输出对，逐层独立校准以保持稳定。
  - 编译框架/kernel调度/硬件架构：论文未明确说明。硬件实现标为 future work。
  
  关键结果：
  - CIFAR-10 W8A3：LCQ+MoDiff FID=4.14（vs FP=4.24，vs LCQ alone=143.39），IS=9.02（vs FP=9.00），10× 运算节省（154 vs 1636 GBops）
  - LSUN-Churches W8A3：LCQ+MoDiff FID=12.05（vs LCQ=341.62）
  - Stable Diffusion W8A6：LTQ+MoDiff FID=13.21（vs LTQ=71.38）
  - DiT-XL/2 W8A6：PTQ4DiT+MoDiff FID=54.74（vs PTQ4DiT=200.26）
  - 兼容所有 sampler（DDIM/DDPM/DPM/PLMS），兼容 QAT 方法 MixDQ
