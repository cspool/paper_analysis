## ANN-to-SNN 转换（ANN-to-SNN Conversion / SpikeZIP-TF）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ANN-to-SNN 转换（conversion）是把已训练的 ANN（或 QANN）权重/结构改写成等价的 SNN，使 SNN 无需直接训练即可获得高精度，是 SNN 落地的两条路线之一（另一条是 surrogate-gradient 直接训练）。传统转换（IF 神经元 + 发放率编码）存在转换误差（conversion error），SNN 精度低于 ANN。SpikeZIP-TF（arXiv 2406.03470）提出用 ST-BIF 神经元（与 Q-ReLU 数学等价）实现"无损转换"：转换后 SNN 与 QANN 精度完全一致（ELSA 论文 Tab.VII：QANN 与 SNN 精度逐项相等）。ELSA 的全部 benchmark SNN 均按 SpikeZIP-TF 生成。
- 关键思想：QANN 的每个算子（量化卷积/线性、量化 ReLU、softmax、LayerNorm、残差加）都有对应的 spike 版本（MM-sc/MM-ss、ST-BIF、ssoftmax、slayernorm、残差加），把算子逐一替换即得 SNN，且因 ST-BIF=Q-ReLU，数值等价。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 转换 pipeline（QANN → SNN，ELSA 采用 SpikeZIP-TF）：
```
1) 训练 QANN（4-bit 权重/激活，量化 ReLU 与 S_min/S_max 对齐）
2) 逐层替换算子：
   QConv/QLinear  → MM-sc（spike-continuous 矩阵乘：输入为脉冲、权重为连续值）
   QAttention(QK^T, AV) → MM-ss（spike-spike：把 spike tracer 当连续操作数，
                          由两个 MM-sc 实现，SpikeZIP-TF 的做法）
   Quantized ReLU  → ST-BIF 神经元（含 spike tracer，clip 界 = Q-ReLU 界）
   Softmax        → ssoftmax（整数版，输出仍为脉冲）
   LayerNorm      → slayernorm（整数版）
   im2col / residual add → router 侧广播实现
3) 推理：时间步 t=1..T 内按脉冲形式执行，最终输出 = clip(floor(Σ V_t/V_thr), S_min, S_max)
```
- 例（ELSA Tab.VII）：ResNet50 QANN=75.60% → SNN=75.60%（无损）；ViT-S QANN=79.07% → SNN=79.07%。检测侧 YOLOv2（ResNet34 backbone）同理。
- Annotations：MM-sc 是输入脉冲 × 权重（连续值）矩阵乘；MM-ss 是脉冲 × 脉冲（需把 tracer 视作连续数，故实现为两次 MM-sc）；T 为时间步（ELSA 用 32）；tracer 的 clip 界 S_min/S_max 必须与 QANN 量化器一致才能无损。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF 官方实现 GitHub Intelligent-Computing-Research-Group/SpikeZIP_transformer，SpikingJelly 已集成 ann2snn.SpikeZIPTFQANNRecipe；转换后 SNN 可直接在 SNN 框架/加速器上运行。ELSA 的 artifact（ELSA_Algorithm 目录，PyTorch 2.4.1）包含转换与逐时间步精度评估脚本。硬件侧：转换决定加速器必须支持的算子集——ELSA 的 Tab.I 列出 MM-sc（CNN/Transformer）、MM-ss（Transformer 注意力）、ssoftmax/slayernorm/残差加/im2col；这些算子分别映射到 PE（MM）与 router（ssoftmax/slayernorm/im2col/残差加广播）。转换还决定了"弹性推理可用"：因为 ST-BIF 输出逐时间步向 Q-ReLU 结果收敛，早停不会造成本质精度损失（ELSA 早停平均 21.9% 延迟缩减、<0.2% 精度损失）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
