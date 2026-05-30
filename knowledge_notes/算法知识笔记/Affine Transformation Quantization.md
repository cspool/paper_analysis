## Affine Transformation Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Affine Transformation Quantization（仿射变换量化）是 AffineQuant 论文提出的 PTQ 等价变换方法，用一个完整的 d×d 可逆仿射矩阵 A 替代传统方法中受限的对角缩放向量 s。核心公式为 argmin_A ||XW - XA⁻¹Q(AW)||²_F。其物理含义：对权重矩阵 W 的每个 output channel（d 维向量），仿射变换 A 对其进行旋转和缩放的任意组合，使其更好地对齐到量化器的 2ⁿ-1 个固定点上。相比于缩放变换（仅统一拉伸/压缩每个 channel 的各维度）和平移变换（仅整体平移），仿射变换可以改变 channel 内各维度的相对数值关系，实现最大化的等价变换优化空间。论文证明，当 A 退化为对角矩阵时，该方法等价于 OmniQuant/SmoothQuant/AWQ；当 A 为置换矩阵时，等价于 RPTQ 的重排。同时引入 learnable shift δ，与仿射变换正交叠加。为在有限校准数据下稳定优化 d² 自由参数，基于 Levy-Desplanques 定理提出 Gradual Mask 机制确保 A 在优化中始终保持可逆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA2-7B qkv_proj 层（d=4096）为例：
```
# 初始化
A = torch.eye(4096) * smoothquant_scale  # 对角初始化，严格对角占优
δ = torch.zeros(4096)                     # shift 初始化

for epoch in range(20):
    # Gradual Mask (Eq. 6)
    radius = int((epoch+1) / 20 * 4096)
    GM = torch.zeros(4096, 4096)
    GM.fill_diagonal_(1.0)
    for d in range(1, radius+1):
        GM.diagonal(d).fill_(α)   # α=0.01 for low-bit
        GM.diagonal(-d).fill_(α)
    
    # 前向 (Eq. 7)
    A_star = A * GM                     # Hadamard 积
    A_inv = torch.linalg.inv(A_star)    # GM 保证 A* 严格对角占优
    X_t = (X - δ) @ A_inv.T            # 激活: 平移 + 右乘 A⁻¹
    W_t = quantize(A_star @ W, 4bit)   # 权重: 左乘 A + INT4 量化
    bias_t = bias + δ @ W              # bias: 平移补偿
    
    loss = MSE(block(X_t, W_t, bias_t), block_fp(X, W, bias))
    loss.backward()                     # GM 抑制非对角线更新
    optimizer.step()

# 推理时合并
W_final = quantize(A_final @ W, 4bit)          # A 合并入权重
bias_final = bias + δ_final @ W_final          # δ 合并入 bias
ln_weight *= diag(A_ln)                         # LayerNorm 对角的 A 合并
```
关键结果：LLaMA2-7B w4a4 C4 PPL 15.76（OmniQuant 18.02, ↓2.26）；LLaMA-30B w4a4 zero-shot avg 58.61%（OmniQuant 56.63%, ↑1.98%）；均无额外推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/bytedance/AffineQuant。基于 PyTorch + HuggingFace，修改自 OmniQuant。关键实现细节：(1) 对每个 transformer block 内的 qkv_proj、out_proj、fc1、fc2 分别维护 A 和 δ，但 fc1-fc2 之间因 GELU/SiLU 使等价变换失效而排除仿射变换；(2) attention 模块在每个 head 内独立应用 A（每 head dim d_head=128）；(3) 矩阵求逆使用 PyTorch linalg.inv，float-double 混合精度（模型 float + A 矩阵 double）在误差和资源间平衡最佳；(4) 优化超参（lr、epoch、clipping）对齐 OmniQuant。稳定性因子 α 的选择：小模型（≤6.7B）α=1；大模型且 ≥3-bit α=1e-2；低比特 α∈{1e-2, 1e-3, 1e-4}。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---
