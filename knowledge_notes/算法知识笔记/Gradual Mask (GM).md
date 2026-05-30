## Gradual Mask (GM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gradual Mask（GM，渐进掩码）是 AffineQuant 的核心优化稳定性机制，用于在有限校准数据下安全地优化 d×d 仿射变换矩阵 A（d 可达 4096+）。核心思想：(1) 在优化初期冻结 A 的所有非对角线元素（仅更新对角线），使模型从严格对角占优（对角矩阵）的安全起点出发；(2) 随着 epoch 推进，按 (e/total_epochs) × hidden_size 的半径逐步释放靠近对角线的元素参与优化；(3) 释放的非对角线元素乘以稳定性因子 α（<1），抑制其幅度和梯度更新率。GM 的数学定义（Eq. 6）：GM_ij = 1（i=j, 对角线），= α（0 < |i-j| ≤ e/t × hidden_size），= 0（otherwise）。GM 通过双重机制保可逆性：(a) 前向：A* = A ∘ GM，缩小非对角线幅度；(b) 反向：A_{e+1} = A_e + η·GM·∂L/∂A*，GM 调节非对角线学习率。实验证实 GM 是不可或缺的——无 GM 时 LLaMA-7B w2a16 训练崩溃（PPL=NaN），OPT-125M w3a16 PPL 从 32.10 恶化至 53.52。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 d=4096, total_epochs=20, α=0.01 为例的 GM 演化过程：
```
Epoch 1 (e=1): radius = 1/20 * 4096 ≈ 204
  可学习参数: 对角线 4096 + 2*(1+...+204)*α = 4096 + 41406*0.01 ≈ 4510
  实际效果: 几乎等价于 OmniQuant（仅对角线优化）

Epoch 5 (e=5): radius = 5/20 * 4096 = 1024
  可学习参数: 4096 + 2*(1+...+1024)*α = 4096 + 1047552*0.01 ≈ 14572

Epoch 10 (e=10): radius = 2048
  可学习参数: 4096 + 2*(1+...+2048)*α = 4096 + 4193280*0.01 ≈ 45627

Epoch 20 (e=20): radius = 4096
  可学习参数: 4096*4096 ≈ 16.8M（全部解冻）

伪代码：
def create_gradual_mask(d, epoch, total_epochs, alpha):
    radius = int(epoch / total_epochs * d)
    GM = torch.zeros(d, d)
    for i in range(d):
        GM[i, i] = 1.0                          # 对角线
        start = max(0, i - radius)
        end = min(d, i + radius + 1)
        for j in range(start, end):
            if i != j:
                GM[i, j] = alpha                # 非对角线近邻
    return GM
```
注意：attention 模块在每个 head 内独立应用 GM（每 head dim=128），而非在整个 qkv_proj 的 4096 维度上统一应用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中实现 GM 的方式：(1) 每 epoch 前生成 GM 张量（基于当前 epoch 和 α）；(2) 前向 `A_masked = A * GM`，然后 `A_inv = linalg.inv(A_masked)`；(3) GM 不在计算图中（无需对 GM 求梯度），梯度通过 A* 流回 A 时自动被 GM 缩放：`A.grad = GM * A_star.grad`。α 的选择策略：小模型（≤6.7B）α=1（自由更新，因矩阵维度小可天然稳定）；大模型高比特（≥3-bit）α=1e-2（适度抑制）；低比特 α∈{1e-2, 1e-3, 1e-4}（强抑制防崩溃）。当 α→0 时，GM 退化为仅更新对角线（等价于 OmniQuant）。该方法的创新类似 Adaround 中的 gradual β 控制（逐步释放舍入参数），但目的不同——Adaround 是防局部最优，GM 是保证矩阵可逆性。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---
