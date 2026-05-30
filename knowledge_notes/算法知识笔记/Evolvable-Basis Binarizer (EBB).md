## Evolvable-Basis Binarizer (EBB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Evolvable-Basis Binarizer (EBB) 是 BinaryDM（ICLR 2025）提出的用于扩散模型权重二值化的可演化基二值化器。核心设计：训练第一阶段使用双基残差二值化 w_EBB^bi = σ_I * sign(w) + σ_II * sign(w - σ_I * sign(w)) 作为过渡状态，候选值从 2 个扩展到 {±σ_I ± σ_II} 共 4 种组合，显著增强信息熵和表征空间；然后通过正则化 L_EBB = τ/N * Σ σ_II^i（τ=9e-2）驱动高阶基 σ_II→0；第二阶段移除高阶项，简化为 w^bi = σ_I * sign(w) 达到真正全二值化。EBB 的 "可演化性" 使高阶基仅作训练的脚手架，推理时无额外开销。仅应用于 DM 首尾各 6 层（约 15% 参数），中间层用 vanilla binarizer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BinaryDM 中 EBB 两阶段训练流程：
```
# Stage 1: Multi-basis EBB + Regularization
σ_I = ||w|| / n
σ_II = ||w - σ_I * sign(w)|| / n
for iter in range(stage1_iters):
    w_ebb = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w))
    o = σ_I*(a ⊗ sign(w)) + σ_II*(a ⊗ sign(w - σ_I*sign(w)))  # ⊗ = XNOR+popcount
    L = L_simple + (τ/N)*Σ σ_II + λ*L_LRM
    # STE backward with ∂σ_I/∂σ_II from Eq.7-8

# Stage 2: Single-basis Binarization
for iter in range(stage2_iters):
    w_bi = σ_I * sign(w)
    o = σ_I * (a ⊗ sign(w))
    L = L_simple + λ*L_LRM
```
EBB 仅在 feature_scale ≥ input_scale/2 的层应用。消融：W1A4, Head+Tail=6, FID=8.53(vs baseline 10.87)；Head+Tail=12 反升至 FID=7.10（过渡参数过多导致不稳定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch QAT 实现：被 EBB 覆盖的卷积/线性层维护 (w, σ_I, σ_II) 三组参数，~100K iter 时切换阶段。正则化系数 τ=9e-2 最优（τ=0 FID=8.01, τ=9e-2 FID=6.99）。开源：https://github.com/Xingyu-Zheng/BinaryDM。训练约 200K iter，LDM batch=4，DDIM batch=64，lr 为全精度的 1/10~1/100。

涉及论文标题：
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

---
