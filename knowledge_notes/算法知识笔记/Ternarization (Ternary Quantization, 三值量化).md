## Ternarization (Ternary Quantization, 三值量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ternarization（三值量化）是将神经网络权重的取值空间约束为 {-1, 0, +1} 三个离散值的极端压缩技术，等效位宽约为 log₂(3) ≈ 1.58 bit/权重。三值化的核心优势：(1) 存储压缩——仅需 2-bit 索引（编码 3 种状态，实际可更紧凑编码为 ~1.58 bit），远低于 FP16（16 bit）和常规 2-bit 量化（4 种状态的均匀量化）；(2) 计算效率——矩阵乘法 ŴX = (αT+μ)X 中，T 仅含 {-1,0,1} 使得乘法退化为加减法（T_ij=1 加 X_j，T_ij=-1 减 X_j，T_ij=0 跳过），消除绝大多数浮点乘法；(3) 表达能力——相比二值化（±1），三值化的零值 (0) 能更好地匹配 LLM 权重常见的单峰分布（大量权重接近零），自适应的 (α, μ) 参数进一步捕获幅值和偏移信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TWN（Ternary Weight Networks, Li et al. 2016）的经典对称三值化流程：
```
# 输入: W ∈ R^{n×m} (FP16 weights)
# 输出: α (scaling factor), T ∈ {-1,0,1}^{n×m}

for i in range(n):
    Δ_i = 0.75 * mean(|W[i,:]|)            # 阈值估计(假设均匀/正态分布)
    T[i,j] =  1  if W[i,j] >  Δ_i else \
             -1  if W[i,j] < -Δ_i else 0
    α_i = Σ_j(T[i,j] * W[i,j]) / Σ_j(|T[i,j]|)
# 推理: y ≈ α * (T @ x), T@x 仅需加法/减法/跳过
```
PT²-LLM 将对称三值化扩展为非对称：Ŵ = αT + μ，引入逐行偏移 μ 捕获非零均值权重分布。其 ATQ 将一次性阈值估计替换为 ITF+AGA 两阶段无训练优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
三值化在训练场景（QAT/TTQ/BitNet b1.58）中通过 STE（Straight-Through Estimator）在反向传播中近似梯度，前向使用三值权重。在 PTQ 场景中（如 PT²-LLM），三值化通过闭式解迭代优化（无需梯度反传），属于后训练压缩。BitNet.cpp（Wang et al. 2025）提供三值模型在 CPU 上的高效推理实现，利用三值乘法的计算特性（仅加减）实现加速。TereFiC（Yin et al. 2025）将三值推理部署到 FPGA。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models
