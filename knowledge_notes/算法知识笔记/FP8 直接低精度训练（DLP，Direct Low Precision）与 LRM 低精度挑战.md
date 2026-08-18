## FP8 直接低精度训练（DLP，Direct Low Precision）与 LRM 低精度挑战

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FP8（8-bit 浮点，E4M3 前向/E5M2 反向）是 GPU 低精度算术的主力格式：B200 的 FP8 稠密 FLOPs 是 A100 TF32 的 29×，远超 TF32 的 7×。低精度训练按"量化时机与训练关系"分四类：QAT（量化感知训练，训练中模拟量化但权重保持全精度，提精度不提训练速度）、PTQ（训练后量化，简单但精度损失大）、PQT（PTQ+微调恢复精度）、DLP（Direct Low Precision，直接低精度——全训练与推理都用原生低精度，吞吐收益最大但技术挑战最大）。LoKA 聚焦 DLP，需要原生低精度贯穿整个训练与推理。
- LRM（大型推荐模型）与 LLM 在低精度上的根本差异：①质量约束极紧（0.02% relative log loss 即显著退化），几乎没有近似空间；②架构异构（宽 ensemble、深层次堆叠、专用交互模块，各自数值敏感度不同）；③算术强度低——由大量小 GEMM 紧跟归一化层组成，量化/反量化开销可吞掉低精度收益。直接应用 TorchAO 对 Wukong 全线性层做 FP8（64 H100，tensorwise）实测 1.3× 变慢 + 2.5% relative log loss 退化。生产现状：top-500 Ads 模型 95% TF32 训练、99% FP16 推理、FP8 训练 0%（推理仅 1% PTQ）——数值稳定、小 GEMM 量化开销、通信密集是三大阻碍。
- 关键推论：这些挑战不能用"更好的 FP8 kernel"解决，需系统-模型协同设计（分布感知 profiling 找安全位点 + 模型组件与硬件协同改模扩大安全位点 + 跨 kernel 库逐算子编排最大化收益）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DLP 训练一个 Wukong 线性层的 pipeline：输入 x∈R^{M×N}、权重 W∈R^{N×K} → 量化（按缩放策略把 x、W 从 BF16/FP32 转到 FP8，scales 由统计量推导）→ FP8 GEMM（FP32 快速累加）→ 反量化输出 → 归一化（LoKA 用 BlockNorm 融合进 epilogue）→ 激活（Hard Swish）→ 损失 → 反向传播输入梯度同样走 FP8。相比 QAT（前向模拟量化、权重/梯度全精度），DLP 的前向与反向全部原生 FP8。
- 误差度量链：对每层用学习分布采样合成输入/权重 → 跑 FP8 kernel vs TF32 参考 → 按 MERE=Σ_mΣ_n|(out−ref)/ref| 量化每层误差 → 判定该层是否可安全低精度。MERE 几何均值在真实 LRM 分布下比标准正态输入高 15%，证明随机基准漏检真实量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：低精度库（NVIDIA Transformer Engine、DeepGEMM、FBGEMM、TorchAO、AMD Quark）提供 FP8 kernel 与缩放 recipe（tensorwise/rowwise/blockwise）；训练框架（PyTorch/TorchRec）经自定义 autograd 包装层接入。LoKA 的落地方式：LoKA Probe 在线学每层分布 → 离线 MERE+吞吐评估 → LoKA Mods 改模型（No Bias/BlockNorm/Hard Swish）→ LoKA Dispatch 逐算子选最快满足精度约束的 kernel。效果：Wukong/Interformer/ELFM 上 FP8 全轨迹 loss 与高精度基线持平，训练最高 1.19×/推理 1.4×，生产 5–20% 训练 / 10–17% 推理加速。
- 别名/关联：与知识库"LLM 数值格式与群量化（bf16 / FP8-E5M2 / ...）"条目互补——该条目讲格式本身，本条目讲 DLP 训练方法论与 LRM 特有挑战。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
