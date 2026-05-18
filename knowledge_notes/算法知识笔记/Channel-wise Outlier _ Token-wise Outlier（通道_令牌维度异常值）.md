## Channel-wise Outlier / Token-wise Outlier（通道/令牌维度异常值）

术语是什么？通过联网搜索让回答具体和精准。

在LLM activation tensor中，outlier指那些显著大于其他值的极少数activation值。根据outlier在activation tensor（shape: [tokens, channels]）中的集中维度，分为两类：

- **Channel-wise Outlier (CO)**：outlier集中在特定channel（embedding position），即某些channel的所有token普遍具有更大的activation值。源于模型内部结构（特定FFN神经元输出）。模式相对稳定、可离线分析。
- **Token-wise Outlier (TO)**：outlier集中在特定token（输入序列中的特定词/短语），即某些token的所有channel普遍具有更大的activation值。源于输入语言特征（罕见词、特殊标点等）。模式动态、不可预测，需在线检测。

RoMeo首次明确指出dual-dimensional outliers是4-bit量化性能下降的根本原因：channel-wise方法（MixQ）移除top-256 outlier channels后max activation从1272降至110（8-bit足够），但4-bit下残余token-wise outliers仍导致严重量化误差。Hadamard rotation可将peak从1272降至58.5（channel-wise outlier被平滑），再通过token-wise mixed precision将残余TO移除后peak进一步降至18.6。

从算法pipeline角度拆解术语：

```
Activation Tensor X [M tokens, K channels]

Channel-wise Outlier检测（如MixQ）:
  max_per_channel = reduce_max(|X|, axis=0)  // shape: [K]
  outlier_channels = topk(max_per_channel, k_co)

Token-wise Outlier检测（如RoMeo，在Hadamard rotation后）:
  X_rotated = Hadamard_rotate(X)  // [M, K]
  max_per_token = reduce_max(|X_rotated|, axis=1)  // shape: [M]
  outlier_tokens = topk(max_per_token, k_to)

关键差异：
- CO沿channel维度（activation columns）→位于GEMM的reduction dimension (K)
  → 可分解为两个独立dense GEMM (W4A4 + W8A8)
- TO沿token维度（activation rows）→位于GEMM的non-reduction dimension (M)
  → 无法简单分解，产生sparse computation pattern
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Channel-wise outlier检测：使用offline calibration dataset（如WikiText的若干样本）运行模型→收集各layer activation→统计per-channel max值→topk选择outlier channels→serving时固定使用。MixQ进一步提出用小型predictor模型在线预测outlier channels以提高适应性。
- Token-wise outlier检测：必须在serving时在线执行。RoMeo使用fused Triton kernel：per-token row-max reduction（parallel reduction over K dimension）→topk selection→生成outlier mask。动态性来自token-wise outliers由输入prompt的语言特征决定（如特定罕见词触发某些attention pattern），无法离线预判。
- 双维度outlier处理策略：Hadamard rotation将CO平滑并迁移到token维度→两个维度的outlier统一为TO→仅需处理一种outlier类型。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

