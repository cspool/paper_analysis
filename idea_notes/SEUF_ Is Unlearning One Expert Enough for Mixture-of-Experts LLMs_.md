## SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?

- baseline方法是什么？
  Baseline 是直接在 MoE LLM 上应用现有 unlearning 方法（GA、GDIFF、NPO、RMU），对所有参数（或 experts+router 全量参数）进行梯度更新以最小化 forget loss。全栈执行例子：
  - **模型推理算法层**：给定 MoE LLM（如 DeepSeek-V2-Lite），forget set D_f 包含待遗忘知识（如危险化学知识），retain set D_r 包含需保留的通用知识。现有 unlearning 方法对所有 expert FFN 权重 + router 权重进行梯度更新，目标为 min_θ l_f(θ; D_f) + λ l_r(θ; D_r)。例如 GA (Gradient Ascent) 直接对 forget set 做梯度上升使模型遗忘，RMU (Representation Misdirection) 对特定层 MLP 的 hidden representation 施加 steering vector 扰动。每次迭代后，router 为每个 token 计算 gating score g^{(l)} = Softmax(Router(u_t^{(l)}))，选 Top-K expert 做 FFN 计算 h_t' = u_t + Σ_i g_i * FFN_i(u_t)。
  - **系统框架层**：论文未明确说明（不涉及框架修改，直接对 Hugging Face 加载的模型做梯度更新）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（标准 PyTorch training loop，无特殊 kernel）。
  - **硬件架构层**：NVIDIA A100 GPU（论文 Sec. 5 "∼1 GPU hour on an A100 per soft prompt"）。

  Baseline 的核心缺陷：
  (1) **Expert selection shift ("short-cut")**：在 unlearning 过程中，Router 会逐渐将 token 从原本最相关的 target expert 切换到非目标 expert（Fig. 3a 显示 expert selection overlap ratio 持续下降）。原因：Router 发现切换激活的 expert 比真正抹除 target expert 中的知识更容易降低 forget loss——非目标 expert 原本不包含 target knowledge，对其做 unlearning 的 forget loss 更低，但这实际上是"假遗忘"。
  (2) **过度遗忘导致 utility 崩溃**：由于 expert selection shift，非目标 expert 被频繁激活参与 unlearning，但其原本包含的是与 forget set 无关的知识。强制对这些 expert 做 unlearning（即破坏其正常知识表示）导致模型 utility 严重下降——Table 1 显示 Qwen 在 GA unlearning 后 UT 从 0.5979 降到 0.3393（44% 下降），DeepSeek 从 0.5500 降到 0.3145（43% 下降）。
  (3) **Router 固定也无法解决**：即使固定 router 参数不动，unlearning 仍可间接影响 router 选择——因为第 l 层的 router 决策依赖前一层的 expert output，而前一层的 expert 已被 unlearning 修改，导致 cascading shift。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **SEUF (Selected Experts Unlearning Framework)**，通过"识别→锚定→聚焦遗忘"三步来解决 baseline 的 expert selection shift 和过度遗忘问题。全栈执行例子：
  - **模型推理算法层**：核心三步——(a) **Expert Attribution**：从 forget set D_f 随机采样 ~100K tokens 的子集 D_s，对每个 token t 在第 l 层的 Router 输出 g_{i,t}^{(l)} = Softmax(Router(u_t))[i]，按 s_i^{(l)} = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t}^{(l)} 计算每个 expert 的全局 affinity score。在 DeepSeek-V2-Lite 的 64 experts 中，通常仅 ~6-9 个 expert 被频繁激活（长尾分布，Insight 1）。跨所有 layer 排序选 top-1 expert（M=1 时性能最优，Insight 4）；(b) **Router Anchor Loss**：L_anchor^{(l)} = ||g^{(l)} - a^{(l)}||_2^2，其中 a_i = 1 当且仅当 expert i 为选中的 target expert。这个 MSE loss 强制 router 在 unlearning 全过程中持续输出接近 [0,...,1,...,0] 的 gating 分布，防止 router 切换激活其他 expert；(c) **Focused Unlearning**：仅对 target expert 的 FFN 权重和对应 router 做梯度更新，冻结其他所有参数（仅更新 0.06% 参数）。损失函数：min_θ l_f(θ; D_f) + λ l_r(θ; D_r) + α * L_anchor（α=1 最优）。
  - **系统框架层**：论文未明确说明（不修改框架，可直接在 PyTorch 训练循环中实现）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A100 GPU。

  **设计思路核心映射**：

  - 缺陷(1) "Expert selection shift (Router 作弊切换)" → 方案：**Router Anchor Loss**。通过 MSE loss L_anchor = ||g - a||_2^2 强制 router 输出固定在 target expert 上，消除 router 的"自由度"。这确保了整个 unlearning 过程中，forget set 的 token 始终被路由到 target expert，无法通过切换到其他 expert 来欺骗地降低 forget loss。Fig. 3b 验证了 forget loss 的降低是真正的知识抹除而非路由作弊。

  - 缺陷(2) "非目标 expert 被强迫参与 unlearning 导致 utility 崩溃" → 方案：**Focused Unlearning on top-1 expert only**。仅对 target expert 进行梯度更新，非目标 expert 完全冻结，确保它们存储的通用知识不受任何影响。Table 3 定量验证：GA+SEUF 将 Qwen 的 UT 从 0.3393（baseline GA）恢复到 0.5012（接近 pretrained 0.5979），同时 FE 保持 0.2987（vs baseline 0.2953，几乎不变）。在 RMU+SEUF 上效果更显著：UT 从 0.3560 恢复到 0.5351。

  - 缺陷(3) "Router 固定也无法阻止 cascading shift（因前层 expert 输出变化间接影响后续 router）" → 方案：**三层联合设计**。Expert Attribution 精确定位最相关的单个 expert + Anchor Loss 强制该 expert 持续激活 + 仅对该 expert 做 unlearning。由于所有非目标 expert 被冻结，前层非目标 expert 的输出不变，因此即使 router 未固定（实际上 router 也被冻结仅 target expert 的 router 可训练），非目标层的 cascading shift 也被消除。

  关键定量对比（SEUF vs Baseline）：
  | 对比维度 | Baseline（直接 unlearn） | SEUF |
  |---------|----------------------|------|
  | 更新参数比例 | 100% (或 experts+router 全部) | 0.06% (仅 top-1 expert + 对应 router) |
  | Expert selection 稳定性 | 持续 shift（overlap ratio 下降） | 稳定（anchor loss 强制保持） |
  | GA on Qwen/WMDP UT | 0.3393 | 0.5012 (+47.8% 相对改善) |
  | RMU on Qwen/WMDP UT | 0.3560 | 0.5351 (+50.3% 相对改善) |
  | GDIFF on DeepSeek/WMDP UT | 0.3929 | 0.4895 (+24.6% 相对改善) |
  | GCG jailbreak 后 FE | 未测试 | 保持 0.01（知识不可恢复） |

  论文核心洞察：MoE unlearning 的秘密在于"少即是多"——unlearning 一个 expert 就足够（Unlearning One Expert Is Enough），关键在于选对 expert 并锁定 router。这一发现颠覆了传统 unlearning"更新越多参数遗忘越彻底"的直觉，揭示 MoE 架构中知识高度集中在少数 expert 的特性。
