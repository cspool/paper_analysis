## Mixed Precision Quantization（混合精度量化）

术语是什么？通过联网搜索让回答具体和精准。

Mixed Precision Quantization（混合精度量化）是一种对模型tensor中不同部分使用不同bit-width精度进行量化的技术。与uniform precision quantization（所有tensor使用同一精度如全INT4或全INT8）不同，混合精度量化利用LLM activation中的outlier sparsity property——少数值显著大于多数值——对error-sensitive的outlier区域分配更高精度（如INT8或FP16），对大多数normal值使用更低精度（如INT4），从而在保持模型准确率的同时最大化计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

混合精度量化在LLM推理pipeline中的三种粒度（以4-bit W4A4场景为例）：

**(1) Tensor-wise Mixed Precision（粗粒度）**：
不同模型模块使用不同精度。如MxMoE中expert层用INT4、shared层用INT8。计算在每个模块内为uniform precision，易实现但与Tensor Core兼容性最好。

**(2) Channel-wise Mixed Precision（通道级，如MixQ/LLM.int8()）**：
```
Input: activation X [tokens, channels], outlier budget k_o
// 离线或在线识别channel-wise outliers
for c in 0..channels:
    max_vals[c] = max(|X[:, c]|)
O = topk_indices(max_vals, k_o)  // top-k outlier channels

// 混合精度GEMM沿reduction dimension分解
// Normal channels: INT4 weight × INT4 activation
C_normal = INT4_GEMM(W_normal, X_normal)
// Outlier channels: INT8/FP16 weight × INT8/FP16 activation  
C_outlier = INT8_GEMM(W_outlier, X_outlier)
// 结果相加
C = C_normal + C_outlier
```
优势：outlier channels沿reduction dimension，可自然分解matrix multiplication为两个独立dense GEMM，与Tensor Core完全兼容。

**(3) Token-wise Mixed Precision（令牌级，如RoMeo的RTMPQ）**：
```
Input: rotated activation X' [tokens, channels], outlier budget k_o
// Hadamard rotation先将channel outliers迁移到token维度
X' = X · H  // H为Hadamard矩阵

// 在线per-token outlier检测
for t in 0..tokens:
    max_vals[t] = max(|X'[t, :]|)
O_A = topk_indices(max_vals, k_o)  // top-k outlier tokens

// Token-wise mixed precision quantization
X'_Q = zeros_like(X')
for t in 0..tokens:
    if t in O_A:
        X'_Q[t, :] = INT8_quantize(X'[t, :])  // outlier token: INT8
    else:
        X'_Q[t, :] = INT4_quantize(X'[t, :])  // normal token: INT4

// Cross-precision GEMM（非归约维度混合精度）
// 四种精度组合: W4A4, W4A8, W8A4, W8A8
```
特征：token-wise outliers沿non-reduction dimension→sparse computation pattern→需要permutation-free等系统优化才能高效映射到GPU。

混合精度量化对LLM推理的关键trade-off：outlier比例越高→准确率越好但计算效率越低。RoMeo显示5% outlier tokens用INT8即可在Qwen3-8B上实现10.97 PPL（vs QuaRot 11.53），理论加速比公式：S = 1/(P_INT4/4 + (1-P_INT4)/2)，其中P_INT4为纯INT4计算比例。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- **Channel-wise**: MixQ使用静态offline calibration dataset分析per-channel max activation→离线识别outlier channels→serving时直接使用预计算outlier mask。LLM.int8()使用online per-channel outlier detection。
- **Token-wise**: RoMeo的RTMPQ使用online per-token row-max + top-k selection→fused Triton kernel实现。token-wise的outlier detection必须在线完成（outlier来自input sequence的语言特征，非静态模型特性）。
- **Group-wise**: Atom使用finer group-wise granularity（group size=128）→每group独立scaling factor→更高准确率但更多dequantization overhead。
- 对于weight矩阵，mixed precision可在offline完成（weight不随输入变化）。但RoMeo指出Hadamard旋转后weight也需mixed precision（H^T预乘amplify weight non-uniformity）。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

