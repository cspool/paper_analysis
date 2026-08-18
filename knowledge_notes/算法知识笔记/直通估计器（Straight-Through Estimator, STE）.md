## 直通估计器（Straight-Through Estimator, STE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- STE（Bengio 等 2013，arXiv:1308.3432）是训练含不可导操作（sign/round 量化、二值化）网络的梯度近似技术：前向用真实不可导操作（如二值化 sign），反向把"零梯度"替换为代理梯度（常用恒等/直通，即 ∂L/∂x ≈ ∂L/∂x̂，或饱和 STE 1_{|x|≤1}），让梯度"直通"不可导点到达潜在权重。Yin 等（ICLR 2019，arXiv:1903.05662）从理论上把 STE 梯度形式化为"粗梯度"（coarse gradient），证明恰当选取的 STE 的期望粗梯度与真实梯度正相关、其负方向是下降方向。Moirai 用它解决 BNN 在线训练的核心难题：sign 函数几乎处处零梯度，STE 让梯度绕过整个网络的二值化器，成功更新所有层的潜在权重 W_raw（含 7-bit 与 4-bit 层），使 L1D 预取器能实时在线学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Moirai 的 BCC 反向计算（带 STE 的梯度流）：
  ```
  # 前向：A_bin = sign(A);  W_bin = sign(W_raw)
  # 反向（STE：把 sign 的导数近似为恒等直通）：
  dL/dW_raw = dL/dW_bin * 1      # STE：d(sign(x))/dx ≈ 1
             = ΔW_raw^k          # 式 2：ΔW_raw^k = G_{i+1} * Ac_i^k
  W_raw <- W_raw - lr * ΔW_raw   # 7-bit(首层)/4-bit(其余层) 潜在权重更新
  ```
  STE 让梯度 dL/dW_bin 直通到 W_raw，因此二值化不阻断学习；配合混合精度潜在权重（首层 7-bit 保特征保真度、其余 4-bit 省面积）保证梯度累积精度。
- 关键点：STE 是"近似梯度"，代理导数选择（恒等 vs 饱和 vs sigmoid 导数）影响训练稳定性；Moirai 与 LSQ 等量化方法都用 STE 让梯度穿过 round/clip 或 sign。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 中自定义 autograd Function（forward 用 sign/round，backward 返回恒等梯度）或 `torch.where`/直通写法；LSQ 等 QAT 库内置 STE 自定义算子。硬件实现：Moirai 的"Gradient Computation"块因前向权重已二值化，完全避开复杂乘法，综合为条件符号翻转器阵列 + 浅加法树（跨 K 通道累加）。使用场景：所有二值/量化网络训练（BNN、QAT、LSQ）、量化感知训练部署到 INT8 NPU；Moirai 首次把它放进 L1D 预取器的片上在线训练路径。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
