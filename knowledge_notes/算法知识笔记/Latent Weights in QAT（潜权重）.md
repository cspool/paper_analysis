## Latent Weights in QAT（潜权重）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Latent Weights（潜权重）是 QAT 中维护的全精度（FP32）权重参数，作为 quantizer 的输入而非直接参与前向计算。QAT 的三步循环：(1) 前向：潜权重 w → quantizer（normalize → discretize → de-normalize）→ 量化权重 w_q → 计算输出和 loss；(2) 反向：STE 将 ∂L/∂w_q 的梯度近似直通回传到 w；(3) 优化：用梯度更新潜权重 w（而非 w_q）。潜权重的存在使离散量化权重可通过连续优化间接训练。但这也带来核心问题：量化权重 w_q 的 effective step size 不仅取决于 LR，还取决于潜权重是否越过 quantizer 的 transition point（即潜权重的分布）。训练后期潜权重倾向于在 transition point 附近聚集（Fig. 3c），导致即使极小 LR 也能触发大量量化权重 transition，造成训练不稳定——这正是 TR 调度要解决的核心痛点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
潜权重在 QAT 中的完整流转（2-bit 权重量化为例）：
```
初始化: w = pretrained_FP_weights  (FP32, nn.Parameter)

每次迭代 t:
    # 前向: 潜权重 → 归一化 → 离散化 → 反归一化 → 量化权重
    w_n = clip(γ·w/s, α, β)        # 归一化潜权重到 [-2, 1]（α=-2, β=1, γ=2）
    w_d = round(w_n)                # 离散化为整数 {-2, -1, 0, 1}
    w_q = w_d/γ                     # 固定反归一化（γ=2 → w_q∈{-1.0,-0.5,0,0.5}）
    output = conv2d(x, w_q)         # 用量化权重计算

    # 反向: STE 直通梯度
    ∂L/∂w_q →（STE: ∂round/∂w_n=1）→ ∂L/∂w_n →（chain rule through clip）→ ∂L/∂w

    # 优化: 更新潜权重
    w^{t+1} = w^t - U^t·g^t          # TR 调度 (TALR)
    # 或 w^{t+1} = w^t - μ^t·g^t    # 传统 LR 调度
```
关键性质：(1) 潜权重是连续值——LR 对潜权重的 effective step size 控制有效（Fig. 1b 蓝线 vs Fig. 2b 蓝线，均与 LR 相关）；(2) 潜权重的分布决定量化权重的 transition 行为——训练后期潜权重聚集在 transition point 附近（Fig. 2c），即使极小 LR 也触发频繁 transition；(3) TR 调度的 TALR 通过负反馈机制隐式考虑潜权重分布——当潜权重接近 transition point 时，K^t 自然升高，U^t 自动降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
潜权重在 PyTorch 中的实现：每个含 quantizer 的层持有 `self.weight` (nn.Parameter, FP32)，前向传播中 `w_q = self.quantizer(self.weight)` 后才用于 F.conv2d。Optimizer 的 param_groups 注册的是 `self.weight`（潜权重）而非 w_q。TR 调度在 optimizer.step() 中修改潜权重的有效学习率（即 TALR），使 w 的实际更新量由 TR 反馈控制。注意：部分 QAT 方法允许 weight quantizer 的 scale s 也是可学习参数——此时 scale 梯度也流经 STE，但当 TR 调度启用时 s 必须固定，否则 s 变化独立于 w 也能触发 transition，干扰 TR 控制。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---
