## Additive Vector Quantization（AQLM，加法向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQLM（Additive Quantization of Language Models，https://github.com/Vahe1994/AQLM）把单个权重向量表示为 C 个码本条目的和（每个码本贡献一个 centroid），即 additive/残差式多层量化，在 2-bit 级达到 SOTA 精度-压缩权衡。EVA 以它作为默认 VQ 算法：C 个码本、每码本 2^n 条目、向量维 d，有效平均精度 q=C·n/d bits（EVA 支持 C=2/3/4 → 2/3/4-bit；Table III：AQLM 2×8=2bit、3×8=3bit、4×8=4bit）。AQLM 的"多码本求和"正好对应 EVA 硬件里 EU 的对角累加（across C0-C3 输出级并行归约）。对比：AQLM-1×16（n=16、65536 条目、单码本）精度与效率都差于 AQLM-4×8（Table III：norm latency 22.86× vs 1.98×）——大码本引入 spurious 乘法且利用率低，验证 n=8 最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AQLM 重构：w_hat = Σ_{c=1..C} B_c[:, i_c]   # 每个权重向量 = C 个 centroid 之和
# EVA 侧：对每个码本算 O_c = X·B_c（输出码本），最终输出 y = Σ_c Lookup(O_c, I_c)（EU 对角累加）
```
例子（AQLM-2×8，d=8,n=8,C=2,q=2bit）：W∈R^{4096×4096} → 2 组 (B_c∈R^{8×256}, I_c∈[0,256)^{512×4096})；decode 时 x∈R^{1×4096} reshape 为 X∈R^{512×8}，O_c=X·B_c∈R^{512×256}，y=Lookup(O_1,I_1)+Lookup(O_2,I_2)∈R^{1×4096}。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AQLM 官方 PyTorch/CUDA 库，离线学习码本（可配合 PV-Tuning 微调码本提升 2-bit 精度）；EVA 算法评估直接用 AQLM 预训练 checkpoint（Hugging Face dbw6/eva collection）跑 perplexity 与下游任务。使用方式：2-bit 级部署选择；MoE 模型上 AQLM-2×8 只掉 5.3pp（Mixtral-8x7B 下游平均），远优于 GPTQ（32.7pp）；EVA 架构与 AQLM 解耦，可换用 GPTVQ 等其他 VQ 算法（GPT-W2* 配置仍优于 FIGLUT 基线 1.15×/2.31×）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
