## 权重-激活量化（WAQ）与权重-only 量化（WOQ）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WOQ（weight-only quantization）只量化权重、激活保持 FP16（W4A16 等），推理时需把权重反量化回 FP16 再做 GEMM——反量化开销可占 GEMM 时间的 20-90%（OASIS §I 引用 [25][30]），且激活仍占内存与带宽、无法利用低精度计算单元。WAQ（weight-activation quantization，双测量化）同时量化权重与激活（W4A4/W8A8）：可全低精度 GEMM（INT4×INT4）、压缩权重与 KV-cache 内存、消除混合格式计算。WAQ 内部两条路线：INT-WAQ（整数等距量化，可被现有低精度硬件直接执行，但表示能力有限、低比特精度差）；NU-WAQ（非均匀码本量化，精度高但索引格式与现有计算单元不兼容，传统执行需反量化回 FP16 再 GEMM，计算优势被抵消）。OASIS 定位即解决"高效低精度 INT-WAQ vs 高精度低效率 NU-WAQ"的两难（图1）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# WOQ (W4A16)：推理每层
W_deq = dequant(W_idx, W_c, per_channel_scales)  # 反量化（开销 20-90%）
Y = W_deq @ X_fp16                               # FP16 GEMM
# INT-WAQ (W4A4)：直接用低精度单元
Y = INT4_GEMM(W_int4, X_int4)                    # Tensor Core / 加速器
# NU-WAQ 传统执行：查码本反量化 + FP16 GEMM
X_deq = C_A[X_idx]; W_deq = C_W[W_idx]
Y = X_deq @ W_deq                                # 反量化抵消量化收益
# OASIS（NU-WAQ 高效版）：WAQ LUT-GEMM 直接算（见下条）
Y = LUT_GEMM(X_idx, W_idx)
```
关键点：WOQ 精度好但 dequant 主导耗时；INT-WAQ 高效但精度差；NU-WAQ 精度好但计算效率差——OASIS 用预计算 Cartesian Product LUT 打破最后一条。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WOQ 代表：GPTQ/AWQ/SpQR（W4A16）；INT-WAQ 代表：SmoothQuant（W8A8）、QuaRot、Atom（group-128 W4A4）；NU-WAQ 代表：SqueezeLLM、K-Means 码本。部署：TensorRT-LLM、MLC-LLM 支持 W4A4 推理；GPU INT4 Tensor Core 支持有限（INT8 更成熟），实际加速低于理论。OASIS 在算法精度上对比 INT-WAQ baselines（RTN/SmoothQuant/QuaRot/Atom），W4A4 下平均 accuracy drop 仅 1.94%、比 Atom 低 6.34%（论文表 III/IV，A100-80GB 上 Transformers+PyTorch+lm-eval-harness 评测，模型 OPT/LLaMA/LLaMA-2/LLaMA-3/Mistral 共 11 个）。

- SMOOTH 模型配置（ISCA'26）：评估的 8 个模型（TinyLLaMA 1.1B、GPT-Neo/GPT-3 XL 1.3B、Gemma-2 2.0B、GPT-3 2.7B、LLaMA2 7.0B、Bloom 7.1B、GPT-3 13B）全部采用 w4a8/int8 权重-激活量化格式，批量 1，对齐移动端部署；SMOOTH 不做任何模型级改动（无精度损失），其 block 级内存管理可正交叠加在 w4a8/int8 等量化方案之上进一步加速。
涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
