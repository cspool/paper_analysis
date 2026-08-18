## No Bias（无偏置模型设计）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- No Bias 是 LoKA Mods 的组件：从 Wukong 模块中移除所有 bias 项（仅最终预测层保留，因不同预测任务 bias 有益）。动机来自 LoKA Probe 发现"bias 项发散"——训练中显著比例 bias 的 L2 范数不收敛、部分达 ≥0.1，级联到后续模块造成越界，在 clamp+量化时令小值完全湮没，是 FP8 训练不稳定的来源之一。
- 借鉴 LLM 趋势：DeepSeek 从所有 FFN 与归一化层移除 bias；PaLM/Falcon 在 FFN 层移除、归一化层保留。LoKA 把该实践引入 LRM 并给出额外系统收益：FSDP per-parameter padding 下，小于 world size 的 bias 张量会引入显著 padding 通信开销，去 bias 同时降低通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（有无 bias 对比）：原始线性层 out=xW+b；No Bias 版 out=xW。对 (2048,256)@(256,768) 的 GEMM，bias 为 (768,) 向量，在 FSDP 分片/填充下 b 的通信与填充开销不再存在；训练中 b 的梯度更新与发散路径也整体消失。LoKA 消融显示 No Bias 是单独贡献最大的延迟降低来源（消除参数开销、简化计算路径），且减少早期训练不稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模型定义中移除 linear/norm 层的 bias 参数（nn.Linear(bias=False) 等），仅预测头保留；低精度 kernel 侧 epilogue 不再有 bias 加项。使用场景：低精度（FP8）LRM 训练/推理的稳定性改进；与 BlockNorm、Hard Swish 合并（三者单独都不足以稳定，合并后 FP8 全轨迹 loss 与高精度基线持平）。注意：不是所有任务都适用——论文明确预测层保留 bias。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
