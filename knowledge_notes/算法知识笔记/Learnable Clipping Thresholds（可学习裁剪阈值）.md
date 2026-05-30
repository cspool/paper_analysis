## Learnable Clipping Thresholds（可学习裁剪阈值）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Clipping Thresholds (LCT) 是 FlatQuant 中用于进一步降低量化误差的组件。在仿射变换平坦化分布后，仍有少量极端值可能导致量化范围过大（浪费量化区间）。裁剪阈值 α ∈ (0,1) 经 sigmoid 函数约束后，定义量化范围为 [−α·max(|X|), α·max(|X|)]（或类似比例），在覆盖大部分信号和排除极端离群值之间寻求最优平衡。与传统网格搜索（如 QuaRot 使用固定 α_a=0.9、α_kv=0.95）不同，FlatQuant 将 α_w（权重裁剪）和 α_a（激活裁剪）作为每层可学习参数，与仿射变换矩阵 P 和缩放向量 c 联合通过梯度下降优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlatQuant 中 LCT 的量化流程（激活侧）：

```
// 传统 RTN 量化（无裁剪）
X_q = round(clamp(X / s, -2^{b-1}+1, 2^{b-1}-1))

// FlatQuant 带 LCT 的量化
α_a = sigmoid(α_a_raw)                    // α_a_raw 为可学习参数
X_max = max(|X|)
clip_bound = α_a * X_max                   // 缩小量化范围
s = clip_bound / (2^{b-1} - 1)
X_q = round(clamp(X / s, -2^{b-1}+1, 2^{b-1}-1))
// 极端值被裁剪到 clip_bound
```

**关键设计**：LCT 必须放在仿射变换之后而非之前。消融实验（Table 18）表明 LCT before transformation 仅带来 marginal gain，而 LCT after transformation 提升显著（PPL 从 7.95→6.98）。原因是仿射变换先将离群值在通道间重新分配，然后裁剪可有效移除更大比例的极端值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LCT 通过 PyTorch 的 nn.Parameter 实现，α_raw 初始化为使 sigmoid 后约为 0.99（接近不裁剪）。优化时使用独立的更高学习率（5e-2 vs 变换矩阵的 5e-3），因为裁剪阈值需要快速响应分布变化。训练完成后 α 值被固化到模型中，推理时以固定阈值进行裁剪，无额外开销。FlatQuant 中 LCT 对权重和激活分别设置，且对 KV cache 也使用独立的裁剪阈值。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models

**OmniQuant 的 Learnable Weight Clipping (LWC)** 是该论文的核心贡献之一，与 FlatQuant 的 LCT 不同：

(1) **参数化方式**：LWC 学习**相对截断强度** γ ∈ [0,1], β ∈ [0,1]（分别控制上界和下界），而非绝对阈值 α。量化公式为 h = (γ·max(W) - β·min(W)) / (2^N-1), W_q = clamp(round(W/h)+z, 0, 2^N-1)。当 γ=β=1 时退化为标准 MinMax 量化。这种相对缩放设计使 LWC 在 LET（等效变换）每轮改变权重分布时仍能稳定收敛——而 PACT 直接学习绝对截断阈值和 LSQ 直接学习绝对 scale/zero-point 在权重分布动态变化时发散（OmniQuant Table A14, Figure A5 实验证明）。

(2) **与 LET 的协同**：LWC 通过 block-wise 量化误差最小化框架与 LET 联合训练。LET 将激活量化难度迁移到权重（加重了 weight quantization 负担），LWC 恰好专门降低 weight quantization 难度，形成递进优化关系。消融实验（Table A2）表明 LET+LWC 联合训练 PPL=12.87，优于 LET alone (16.97) 和 SmoothQuant+LWC (15.80)，证明了 differentiable joint optimization 的关键性。

(3) **训练效率**：LWC 仅引入 2 个可学习参数（γ,β），默认 20 epochs SGD（W2A16 需 40 epochs），LLaMA-7B 在单卡 A100-40G 上约 1.1h（weight-only）。

---
